"""Stage framework: context, status bookkeeping, logging, summaries.

Every stage module (``vndub/stages/<name>.py``) defines::

    NAME = "link"
    VERSION = "1"                 # bump when the stage's output semantics change
    CONFIG_SECTIONS = ("link",)   # config sections that affect its outputs
    def run(ctx: StageContext) -> dict: ...   # returns a one-screen summary

:func:`run_stage` wraps it: per-stage log file ``work/logs/<stage>.log``,
``work/stage_status.json`` (started / finished / hash / summary / error) and a
rich summary table. Item-level caching is done inside stages with
:meth:`StageContext.cache`.
"""

from __future__ import annotations

import importlib
import logging
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from vndub.config import PipelineConfig, ProjectYaml, load_project_yaml, save_project_yaml
from vndub.gpu import ModelRegistry
from vndub.manifest.cache import FileHasher, StageCache
from vndub.manifest.io import Character, LineRecord, iter_jsonl, load_characters, load_lines, read_json, write_json
from vndub.paths import WorkPaths
from vndub.tools import Tools
from vndub.util import get_logger, hash_obj, now_iso

STAGES = [
    "detect", "extract", "parse", "link", "analyze", "translate", "text_emotion",
    "voicebank", "synth", "score", "master", "pack", "report",
]

console = Console(stderr=False, highlight=False)


class StageError(RuntimeError):
    """A stage cannot proceed (missing prerequisite, bad config). Message is user-facing."""


@dataclass
class StageContext:
    cfg: PipelineConfig
    paths: WorkPaths
    tools: Tools
    project_dir: Path
    models: ModelRegistry
    force: bool = False
    only: set[str] | None = None
    limit: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    log: logging.Logger = field(default_factory=lambda: logging.getLogger("vndub"))
    stage: str = ""
    _hasher: FileHasher | None = None
    _caches: list[StageCache] = field(default_factory=list)

    # -- project -------------------------------------------------------------
    @property
    def project(self) -> ProjectYaml | None:
        p = self.paths.project_yaml
        return load_project_yaml(p) if p.exists() else None

    def require_project(self) -> ProjectYaml:
        proj = self.project
        if proj is None:
            raise StageError(f"{self.paths.project_yaml} not found: run `vndub detect --project {self.project_dir}` first")
        return proj

    @property
    def game_dir(self) -> Path:
        proj = self.project
        if proj is not None and proj.game_dir:
            return Path(proj.game_dir)
        return self.project_dir

    # -- languages -----------------------------------------------------------
    @property
    def src_lang(self) -> str:
        """Language of the script text: ``project.yaml: src_lang`` when it is a code, else detected from the lines
        and written back (see :func:`resolve_src_lang`); ``ja`` before any lines exist."""
        return resolve_src_lang(self)

    @property
    def audio_lang(self) -> str:
        """Language of the voice recordings: ``project.yaml: audio_lang`` (the config's when unset), else the text language."""
        proj = self.project
        code = (proj.audio_lang if proj is not None else "") or self.cfg.project.audio_lang
        return code if code and code != "auto" else self.src_lang

    def engine(self):  # -> engines.base.Engine
        from vndub.engines import get_engine

        return get_engine(self.require_project().engine)

    # -- manifests -----------------------------------------------------------
    def lines(self, recovered: bool = True, edits: bool = True) -> list[LineRecord]:
        """``lines.jsonl`` with the source-text patches applied (nothing is written back).

        ``lines.jsonl`` stays owned by parse. Two later sources can fill or correct
        ``text_src`` for every stage that reads lines (see :func:`apply_text_patches`):
        text recovered from the original audio by ``link`` (mode C, ``text_recovered`` in
        ``link.jsonl``, flag ``text_from_asr``) and the user's ``set_src_text`` overrides
        (flag ``text_edited``, the latest one wins). ``recovered=False`` / ``edits=False``
        return the lines without that layer.
        """
        if not self.paths.lines.exists():
            raise StageError(f"{self.paths.lines} not found: run `vndub parse` first")
        return apply_text_patches(self.paths, load_lines(self.paths.lines), recovered=recovered, edits=edits)

    def characters(self) -> dict[str, Character]:
        return load_characters(self.paths.characters)

    def voiced_lines(self, apply_only: bool = True) -> list[LineRecord]:
        """Voiced lines with ``link.jsonl`` rebindings applied (the binding every stage after link uses).

        link.jsonl holds one record per line it judged; its ``audio_id`` is the audio
        that line now owns (``None`` = the voice was moved away / unmatched-and-dropped).
        A file whose audio matches another line's text is rebound to that line, so the
        dub of the file says what the original file said.
        """
        from vndub.manifest.io import AudioIndexRecord, LinkRecord, read_jsonl

        lines = self.lines()
        if self.paths.link.exists():
            audio = {a.id: a for a in read_jsonl(self.paths.audio_index, AudioIndexRecord)}
            links = {r.line_id: r for r in read_jsonl(self.paths.link, LinkRecord)}
            for l in lines:
                r = links.get(l.line_id)
                if r is None or r.audio_id == l.audio_id:
                    continue
                if r.audio_id is None:
                    l.has_voice = False
                    l.flags = [*l.flags, "voice_moved"]
                    continue
                a = audio.get(r.audio_id)
                if a is None:
                    continue
                l.audio_id, l.voice_file, l.has_voice = a.id, a.relpath, True
                l.src_wav, l.src_wav16 = a.wav or None, a.wav16 or None
                l.flags = [*l.flags, "relinked"]
        out = [l for l in lines if l.has_voice and l.audio_id]
        return self.select(out) if apply_only else out

    def select(self, lines: list[LineRecord], voiced_only: bool = True) -> list[LineRecord]:
        """Apply --only / --limit (and the voiced filter)."""
        out = [l for l in lines if (l.has_voice or not voiced_only)]
        if self.only:
            out = [l for l in out if l.line_id in self.only]
        if self.limit is not None:
            out = out[: self.limit]
        return out

    # -- progress ------------------------------------------------------------
    def progress(self, done: int, total: int, item: str | None = None) -> None:
        """Report per-item progress of the running stage (no-op from the CLI).

        The desktop app's job runner installs ``extra["progress_cb"]`` with the signature
        ``(stage, done, total, item)``; it may block (pause) or raise (cancel), so stages
        call this between items, never while holding a half-written output.
        """
        cb = self.extra.get("progress_cb")
        if cb is not None:
            cb(self.stage, done, total, item)

    def check_cancel(self) -> None:
        """Raise the job runner's cancel (``extra["cancel_check"]``, installed by the desktop app) without reporting
        progress: for code that swallows exceptions (a failed model call that is retried) and must not swallow a Cancel.
        No-op from the CLI."""
        cb = self.extra.get("cancel_check")
        if cb is not None:
            cb()

    # -- caching -------------------------------------------------------------
    @property
    def hasher(self) -> FileHasher:
        if self._hasher is None:
            self._hasher = FileHasher(self.paths.cache / "filehash.json")
        return self._hasher

    def cache(self, stage: str, version: str, *sections: str) -> StageCache:
        c = StageCache(self.paths.cache, stage, version, self.cfg.section_hash(*sections), self.hasher)
        if self.force:
            c.clear()
        self._caches.append(c)
        return c

    def save_caches(self) -> None:
        for c in self._caches:
            c.save()
        self._caches.clear()
        if self._hasher is not None:
            self._hasher.save()

    # -- lines to dub (voiced + optionally unvoiced speaker lines) -------------
    def dub_lines(self, apply_only: bool = True) -> list[LineRecord]:
        """Lines the generation stages (synth, score, master) work on.

        ``synth.voice_unvoiced`` off (CLI default): exactly :meth:`voiced_lines`. On (the app): also
        the speaker lines without original audio (mode B, route P): ``has_voice`` is false after the
        ``link`` rebindings, the speaker is not the narrator, the line is not a ``voice sustain``
        continuation (the head line's voice already covers it), not a menu record, and has text.
        Narration is never dubbed. Such lines come back with ``has_voice=False`` and no audio fields,
        in script order, interleaved with the voiced ones; ``--only`` / ``--limit`` apply to the result.
        """
        if not self.cfg.synth.voice_unvoiced:
            return self.voiced_lines(apply_only=apply_only)
        voiced = {l.line_id: l for l in self.voiced_lines(apply_only=False)}
        out: list[LineRecord] = []
        for l in self.lines():
            v = voiced.get(l.line_id)
            if v is not None:
                out.append(v)
                continue
            if l.speaker_id == "narrator" or not (l.text_src or "").strip():
                continue
            if {"voice_sustain", "menu"} & set(l.flags):
                continue
            l.has_voice, l.audio_id, l.src_wav, l.src_wav16, l.voice_file = False, None, None, None, None
            out.append(l)
        return self.select(out, voiced_only=False) if apply_only else out


# ----------------------------------------------------------------------------
# source-text patches (mode C recovery, user edits)
# ----------------------------------------------------------------------------


def resolve_src_lang(ctx: StageContext, lines: list[LineRecord] | None = None, redetect: bool = False) -> str:
    """The language of the script text.

    A code in ``project.yaml`` (or, without a project file, in the config) wins. ``auto`` is resolved by
    :func:`vndub.text.lang.detect_language` over the lines and written into ``project.yaml`` together with the
    detection record (``src_lang_detected``), so every later stage reads the same code. ``redetect`` repeats
    the detection for a value that came from an earlier detection (parse does that over the new lines); a code
    the user set is never replaced. Without lines (nothing parsed yet) the legacy default ``ja`` is returned
    and nothing is written.
    """
    from vndub.text.lang import AUTO, detect_language, language_name

    proj = ctx.project
    code = (proj.src_lang if proj is not None else "") or ctx.cfg.project.src_lang
    was_detected = proj is not None and bool(getattr(proj, "src_lang_detected", None))
    if code and code != AUTO and not (redetect and was_detected):
        return code
    fallback = code if code and code != AUTO else (ctx.cfg.project.src_lang if ctx.cfg.project.src_lang not in ("", AUTO) else "ja")
    if lines is None:
        if not ctx.paths.lines.exists():
            return fallback
        lines = ctx.lines(edits=False)
    guess = detect_language(l.text_src for l in lines)
    if not guess.lang:
        ctx.log.warning("source language: the script text has no letters, keeping %s", fallback)
        return fallback
    if proj is not None:
        proj.src_lang = guess.lang
        proj.src_lang_detected = guess.as_dict()
        save_project_yaml(ctx.paths.project_yaml, proj)
    ctx.log.info("source language: %s (%s), confidence %.2f over %d lines", guess.lang, language_name(guess.lang), guess.confidence, guess.lines)
    return guess.lang


def apply_text_patches(paths: WorkPaths, lines: list[LineRecord], recovered: bool = True, edits: bool = True) -> list[LineRecord]:
    """Fill / correct ``text_src`` in place and return ``lines``.

    1. ``recovered``: a line whose own ``text_src`` is empty takes ``text_recovered`` from its
       ``link.jsonl`` record (mode C) and gets ``flags += ["text_from_asr"]``. Script text always
       wins over a stale recovery.
    2. ``edits``: the latest ``set_src_text`` override per line in ``overrides.jsonl`` (payload
       ``{"text": ...}``) replaces the text and adds ``flags += ["text_edited"]``. An edit of an
       ASR-recovered line keeps both flags (the UI shows the ASR warning only without the edit flag).
    """
    by_id = {l.line_id: l for l in lines}
    if recovered and paths.link.exists():
        for row in iter_jsonl(paths.link):
            text = row.get("text_recovered")
            l = by_id.get(row.get("line_id"))
            if l is None or not isinstance(text, str) or not text.strip() or (l.text_src or "").strip():
                continue
            l.text_src = text
            if "text_from_asr" not in l.flags:
                l.flags = [*l.flags, "text_from_asr"]
    if edits and paths.overrides.exists():
        latest: dict[str, str] = {}
        for row in iter_jsonl(paths.overrides):
            payload = row.get("payload")
            if row.get("action") == "set_src_text" and isinstance(payload, dict) and isinstance(payload.get("text"), str):
                latest[str(row.get("line_id"))] = payload["text"]
        for lid, text in latest.items():
            l = by_id.get(lid)
            if l is None:
                continue
            l.text_src = text
            if "text_edited" not in l.flags:
                l.flags = [*l.flags, "text_edited"]
    return lines


# ----------------------------------------------------------------------------
# status file
# ----------------------------------------------------------------------------


def read_status(paths: WorkPaths) -> dict[str, Any]:
    return read_json(paths.stage_status, {}) or {}


def _write_status(paths: WorkPaths, stage: str, **fields: Any) -> None:
    st = read_status(paths)
    cur = st.get(stage, {})
    cur.update(fields)
    st[stage] = cur
    write_json(paths.stage_status, st)


def stage_module(name: str):
    if name not in STAGES:
        raise StageError(f"unknown stage {name!r}; stages: {', '.join(STAGES)}")
    return importlib.import_module(f"vndub.stages.{name}")


def run_stage(name: str, ctx: StageContext) -> dict[str, Any]:
    mod = stage_module(name)
    ctx.stage = name
    ctx.log = get_logger(name, ctx.paths.log(name))
    sections = getattr(mod, "CONFIG_SECTIONS", ())
    version = getattr(mod, "VERSION", "1")
    stage_hash = hash_obj([name, version, ctx.cfg.section_hash(*sections)])
    t0 = time.perf_counter()
    _write_status(ctx.paths, name, started=now_iso(), finished=None, error=None, hash=stage_hash, version=version)
    ctx.log.info("stage %s started (force=%s only=%s limit=%s)", name, ctx.force, ctx.only, ctx.limit)
    try:
        summary = mod.run(ctx) or {}
    except Exception as e:
        ctx.save_caches()
        _write_status(ctx.paths, name, error=f"{type(e).__name__}: {e}", elapsed_s=round(time.perf_counter() - t0, 2))
        ctx.log.error("stage %s failed: %s\n%s", name, e, traceback.format_exc())
        raise
    ctx.save_caches()
    elapsed = round(time.perf_counter() - t0, 2)
    _write_status(ctx.paths, name, finished=now_iso(), elapsed_s=elapsed, summary=summary)
    ctx.log.info("stage %s finished in %.1fs: %s", name, elapsed, summary)
    print_summary(name, summary, elapsed)
    return summary


def print_summary(name: str, summary: dict[str, Any], elapsed: float) -> None:
    table = Table(title=f"[bold]{name}[/bold]  ({elapsed:.1f}s)", show_header=False, title_justify="left", box=None, pad_edge=False)
    table.add_column("key", style="cyan", no_wrap=True)
    table.add_column("value")
    for k, v in summary.items():
        if isinstance(v, float):
            v = f"{v:.3f}"
        elif isinstance(v, (list, tuple)):
            v = ", ".join(str(x) for x in v[:12]) + (" ..." if len(v) > 12 else "")
        elif isinstance(v, dict):
            v = ", ".join(f"{kk}={vv}" for kk, vv in list(v.items())[:12])
        table.add_row(str(k), str(v))
    console.print(table)


def counter(values: list[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))

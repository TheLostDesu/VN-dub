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

from vndub.config import PipelineConfig, ProjectYaml, load_project_yaml
from vndub.gpu import ModelRegistry
from vndub.manifest.cache import FileHasher, StageCache
from vndub.manifest.io import Character, LineRecord, load_characters, load_lines, read_json, write_json
from vndub.paths import WorkPaths
from vndub.tools import Tools
from vndub.util import get_logger, hash_obj, now_iso

STAGES = [
    "detect", "extract", "parse", "link", "analyze", "translate",
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

    def engine(self):  # -> engines.base.Engine
        from vndub.engines import get_engine

        return get_engine(self.require_project().engine)

    # -- manifests -----------------------------------------------------------
    def lines(self) -> list[LineRecord]:
        if not self.paths.lines.exists():
            raise StageError(f"{self.paths.lines} not found: run `vndub parse` first")
        return load_lines(self.paths.lines)

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

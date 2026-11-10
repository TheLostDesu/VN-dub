"""Read-side helpers shared by the summary, casting, preview and review views.

Everything here only *reads* manifests under ``work/`` (through a cached read-only
:class:`StageContext`). Heavy pipeline modules are imported lazily.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vndub_app import appdirs, session

NARRATOR = "narrator"
#: voiced lines shown without a name that no voice-file prefix attributes (vndub.engines.base.VOICE_OVER)
VOICE_OVER = "voice_over"

_LOCK = threading.Lock()
_SNAP: dict[str, tuple[tuple[Any, ...], "Snapshot"]] = {}


def _mt(p: Path) -> int:
    try:
        return p.stat().st_mtime_ns
    except OSError:
        return 0


@dataclass
class Snapshot:
    """Effective lines of a project (link rebindings and text recovery applied by the pipeline)."""

    lines: list[Any] = field(default_factory=list)          # LineRecord, effective
    voiced: list[Any] = field(default_factory=list)         # LineRecord with a bound voice file
    by_id: dict[str, Any] = field(default_factory=dict)
    voiced_ids: set[str] = field(default_factory=set)

    def is_voiced(self, line_id: str) -> bool:
        return line_id in self.voiced_ids

    def dubbable(self) -> list[Any]:
        """Lines that get a dub: voiced lines, plus unvoiced speaker lines (mode B).

        Narration is context only; ``voice_sustain`` continuations are synthesized with their head.
        """
        out = []
        for l in self.lines:
            if "voice_sustain" in (l.flags or []):
                continue
            if l.line_id in self.voiced_ids:
                out.append(l)
            elif l.speaker_id != NARRATOR and (l.text_src or "").strip():
                out.append(l)
        return out


def snapshot(h: appdirs.ProjectHandle, ctx: Any | None = None) -> Snapshot:
    """Cached effective lines (keyed by the mtimes of lines/link/overrides/audio index)."""
    ctx = ctx or session.read_ctx(h)
    p = ctx.paths
    if not p.lines.exists():
        return Snapshot()
    key = (_mt(p.lines), _mt(p.link), _mt(p.overrides), _mt(p.audio_index))
    with _LOCK:
        hit = _SNAP.get(h.slug)
        if hit and hit[0] == key:
            return hit[1]
    lines = ctx.lines()
    voiced = ctx.voiced_lines(apply_only=False)
    snap = Snapshot(lines=lines, voiced=voiced, by_id={l.line_id: l for l in lines}, voiced_ids={l.line_id for l in voiced})
    with _LOCK:
        _SNAP[h.slug] = (key, snap)
    return snap


def clear_cache(slug: str | None = None) -> None:
    with _LOCK:
        if slug is None:
            _SNAP.clear()
        else:
            _SNAP.pop(slug, None)


# ----------------------------------------------------------------------------
# small manifest readers
# ----------------------------------------------------------------------------


def stage_status(ctx: Any) -> dict[str, Any]:
    from vndub.manifest.io import read_json

    try:
        return read_json(ctx.paths.stage_status, {}) or {}
    except Exception:  # half-written file while a stage runs: treat as unknown
        return {}


def stage_summary(status: dict[str, Any], name: str) -> dict[str, Any]:
    s = (status.get(name) or {}).get("summary")
    return s if isinstance(s, dict) else {}


def load_characters(ctx: Any) -> dict[str, Any]:
    from vndub.manifest.io import load_characters as _lc

    return _lc(ctx.paths.characters) if ctx.paths.characters.exists() else {}


def translations(ctx: Any) -> dict[str, Any]:
    from vndub.manifest.io import TranslationRecord, read_jsonl

    p = ctx.paths.translation(ctx.cfg.project.tgt_lang)
    try:
        return {r.line_id: r for r in read_jsonl(p, TranslationRecord)} if p.exists() else {}
    except ValueError:
        return {}


def selection(ctx: Any) -> dict[str, Any]:
    from vndub.manifest.io import SelectionRecord, read_jsonl

    p = ctx.paths.selection
    try:
        return {s.line_id: s for s in read_jsonl(p, SelectionRecord)} if p.exists() else {}
    except ValueError:
        return {}


def final_index(ctx: Any) -> dict[str, Any]:
    from vndub.manifest.io import FinalIndexRecord, read_jsonl

    p = ctx.paths.final_index
    try:
        return {r.line_id: r for r in read_jsonl(p, FinalIndexRecord)} if p.exists() else {}
    except ValueError:
        return {}


def overrides_by_line(ctx: Any) -> dict[str, list[Any]]:
    from vndub.manifest.io import load_overrides

    try:
        return load_overrides(ctx.paths.overrides) if ctx.paths.overrides.exists() else {}
    except ValueError:
        return {}


def passport(ctx: Any, line_id: str) -> dict[str, Any] | None:
    from vndub.manifest.io import read_json

    try:
        return read_json(ctx.paths.passport(line_id))
    except Exception:
        return None


def display_name(sid: str, chars: dict[str, Any]) -> str:
    c = chars.get(sid)
    if c is None:
        return sid
    return (c.names.get("tgt") or c.names.get("src") or sid) if hasattr(c, "names") else sid


def status_of(sel: Any | None) -> str:
    """Review status of a selection record: ``ok`` | ``check`` | ``failed``.

    Mapping: selection ``auto`` and ``override`` ->
    ``ok`` (the user decided or the gates passed); ``review`` and ``regen`` -> ``check``;
    a record without a winner (nothing could be synthesized / translation failed) -> ``failed``.
    """
    if sel is None or not sel.winner:
        return "failed"
    if sel.status in ("auto", "override"):
        return "ok"
    return "check"


def audio_duration(ctx: Any, line: Any, audio_index: dict[str, Any], passports: dict[str, Any] | None = None) -> float:
    a = audio_index.get(line.audio_id or "")
    if a is not None and getattr(a, "duration", 0):
        return float(a.duration)
    if passports and line.line_id in passports:
        return float(passports[line.line_id].get("duration_s") or 0.0)
    return 0.0

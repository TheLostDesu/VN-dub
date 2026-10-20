"""parse: scripts -> ``manifest/lines.jsonl`` + ``manifest/characters.yaml``.

The engine plugin parses its scripts and binds voices through the audio index
(``VoiceResolver``). ``--import-vntextpatch DIR`` (or ``parse.import_vntextpatch``)
switches any engine to the generic VNTextPatch import with prefix voices.
Existing ``characters.yaml`` user edits (target names, gender, age, route,
notes, performer/rvc/lora) are kept. Audio files no line refers to are
listed in ``manifest/unbound_audio.jsonl`` (link can search them globally).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vndub.engines.base import ExtractResult, ParseResult
from vndub.engines.generic_prefix import import_vntextpatch
from vndub.manifest.io import AudioIndexRecord, Character, load_characters, read_jsonl, save_characters, write_jsonl
from vndub.stages.common import StageContext, StageError

NAME = "parse"
VERSION = "1"
CONFIG_SECTIONS = ("parse",)

USER_FIELDS = ("gender", "age_hint", "route", "notes", "performer", "rvc", "lora", "refs")


def merge_characters(new: dict[str, Character], old: dict[str, Character]) -> dict[str, Character]:
    out: dict[str, Character] = {}
    for sid, c in new.items():
        prev = old.get(sid)
        if prev is not None:
            if prev.names.get("tgt"):
                c.names["tgt"] = prev.names["tgt"]
            for f in USER_FIELDS:
                v = getattr(prev, f)
                if v not in (None, "", [], {}, "unknown"):
                    setattr(c, f, v)
            if prev.voice_prefix:
                c.voice_prefix = prev.voice_prefix
        out[sid] = c
    for sid, c in old.items():  # keep characters the user added by hand
        out.setdefault(sid, c)
    return out


def run(ctx: StageContext) -> dict[str, Any]:
    proj = ctx.require_project()
    ex_path = ctx.paths.raw / "extract.json"
    if not ex_path.exists():
        raise StageError(f"{ex_path} not found: run `vndub extract` first")
    extracted = ExtractResult.load(ex_path)
    audio = [a for a in read_jsonl(ctx.paths.audio_index, AudioIndexRecord) if not a.error] if ctx.paths.audio_index.exists() else []
    imp = ctx.extra.get("import_vntextpatch") or ctx.cfg.parse.import_vntextpatch
    if imp:
        result: ParseResult = import_vntextpatch(ctx, Path(imp), audio)
        source = f"vntextpatch:{imp}"
    else:
        result = ctx.engine().parse(ctx, extracted, audio)
        source = proj.engine
    lines = result.lines
    seen: set[str] = set()
    for l in lines:
        if l.line_id in seen:
            raise StageError(f"duplicate line_id {l.line_id} from the {source} parser")
        seen.add(l.line_id)
    old = load_characters(ctx.paths.characters) if ctx.paths.characters.exists() else {}
    chars = merge_characters(result.characters, old)
    write_jsonl(ctx.paths.lines, lines)
    save_characters(ctx.paths.characters, chars)
    bound = {l.audio_id for l in lines if l.audio_id}
    unbound = [a for a in audio if a.id not in bound]
    write_jsonl(ctx.paths.manifest_dir / "unbound_audio.jsonl", [{"id": a.id, "relpath": a.relpath, "duration": a.duration} for a in unbound])
    for n in result.notes:
        ctx.log.info(n)
    voiced = [l for l in lines if l.has_voice]
    missing = [l for l in lines if "voice_missing" in l.flags]
    speakers: dict[str, int] = {}
    for l in voiced:
        speakers[l.speaker_id] = speakers.get(l.speaker_id, 0) + 1
    coverage = len(bound) / len(audio) if audio else 0.0
    return {
        "source": source,
        "lines": len(lines),
        "voiced": len(voiced),
        "unvoiced": len(lines) - len(voiced),
        "audio_files": len(audio),
        "bound_audio": len(bound),
        "voice_coverage": round(coverage, 3),
        "unbound_audio": len(unbound),
        "voice_missing_refs": len(missing),
        "sustain_lines": sum(1 for l in lines if "voice_sustain" in l.flags),
        "speakers": speakers,
        "characters": len(chars),
        "notes": result.notes[:4],
    }

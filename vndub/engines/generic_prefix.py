"""Generic fallback: text from a VNTextPatch import, voices by ``<prefix>_<index>`` names.

Mode C (no script text at all): with ``link.recover_text`` on, ``parse`` does not stop when
there is no script: :func:`textless_result` makes one voiced line with EMPTY text per voice file
(speaker = file prefix, flag ``voice_only``) so that ``link`` can recover the text by ASR.

Use when the script cannot be parsed for voice references but text + speaker
names are available::

    VNTextPatch extractlocal <game folder> <json folder>
    vndub parse --project <game> --import-vntextpatch <json folder>

VNTextPatch JSON: one file per script, a list of ``{"name": ..., "message": ...}``
(``name`` absent for narration). Voice files are grouped by prefix
(``engine_opts.generic.prefix_regex``, default ``<letters>_?<digits>``) and
sorted by index; each speaker's lines get that speaker's files in script
order. The speaker -> prefix map comes from ``engine_opts.generic.prefix_map``,
then from an existing ``characters.yaml`` ``voice_prefix``, then from name
similarity, then by matching ranks of line counts and file counts. Every
binding made this way is flagged ``prefix_guess``; ``link`` verifies it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from vndub.engines.base import (
    BaseEngine,
    DetectResult,
    ExtractResult,
    LineBuilder,
    ParseResult,
    bind_line,
    build_characters,
    pick_voice_files,
    read_text_any,
)
from vndub.manifest.io import AudioIndexRecord, load_characters

if TYPE_CHECKING:
    from vndub.stages.common import StageContext


def read_vntextpatch_dir(path: Path) -> list[tuple[str, list[dict[str, str]]]]:
    """[(scene, [{name, message}, ...]), ...] in file order."""
    out = []
    for p in sorted(path.rglob("*.json")):
        try:
            data = json.loads(read_text_any(p))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            data = data.get("lines") or data.get("messages") or []
        rows = []
        for x in data if isinstance(data, list) else []:
            if not isinstance(x, dict):
                continue
            msg = x.get("message", x.get("text", ""))
            if not isinstance(msg, str) or not msg.strip():
                continue
            rows.append({"name": str(x.get("name") or ""), "message": msg.strip()})
        out.append((p.relative_to(path).with_suffix("").as_posix().replace("/", "_"), rows))
    return out


def group_by_prefix(audio: list[AudioIndexRecord], regex: str) -> dict[str, list[AudioIndexRecord]]:
    rx = re.compile(regex)
    groups: dict[str, list[tuple[int, str, AudioIndexRecord]]] = {}
    for a in audio:
        stem = PurePosixPath(a.relpath).stem
        m = rx.match(stem)
        if not m:
            continue
        groups.setdefault(m.group("prefix").lower(), []).append((int(m.group("index")), stem, a))
    return {k: [a for _, _, a in sorted(v, key=lambda t: (t[0], t[1]))] for k, v in groups.items()}


def infer_prefix_map(speaker_counts: dict[str, int], groups: dict[str, list[AudioIndexRecord]], explicit: dict[str, str], known: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    notes: list[str] = []
    mapping = {s: p.lower() for s, p in explicit.items() if p}
    for s, p in known.items():
        if s not in mapping and p and p.lower() in groups:
            mapping[s] = p.lower()
    free = [p for p in groups if p not in mapping.values()]
    for s in speaker_counts:
        if s in mapping:
            continue
        cand = [p for p in free if s.lower().startswith(p) or p.startswith(s.lower()[:3])]
        if len(cand) == 1:
            mapping[s] = cand[0]
            free.remove(cand[0])
    rest_s = sorted((s for s in speaker_counts if s not in mapping), key=lambda s: -speaker_counts[s])
    rest_p = sorted(free, key=lambda p: -len(groups[p]))
    for s, p in zip(rest_s, rest_p):
        a, b = speaker_counts[s], len(groups[p])
        if min(a, b) / max(a, b) >= 0.7:
            mapping[s] = p
            notes.append(f"prefix '{p}' -> speaker '{s}' by count rank ({a} lines / {b} files)")
    return mapping, notes


class GenericPrefixEngine(BaseEngine):
    name = "generic"
    out_name = "generic"

    def detect(self, game_dir: Path) -> DetectResult | None:
        voices = pick_voice_files(game_dir, ("voice", "vo", "koe"))
        if not voices:
            return None
        return DetectResult("generic", 0.05, notes=["unknown engine: use --import-vntextpatch for text; voices are assigned by prefix"])

    def extract(self, ctx: "StageContext") -> ExtractResult:
        voices = pick_voice_files(ctx.game_dir, ctx.cfg.extract.voice_dirs)
        return ExtractResult(script_files=[], voice_root=ctx.game_dir, voice_files=voices, notes=["loose voice files; text from VNTextPatch import"])

    def parse(self, ctx: "StageContext", extracted: ExtractResult, audio: list[AudioIndexRecord]) -> ParseResult:
        src = ctx.extra.get("import_vntextpatch") or ctx.cfg.parse.import_vntextpatch
        if not src and ctx.cfg.link.recover_text and audio:
            return textless_result(ctx, audio, "no script text available")
        if not src:
            from vndub.stages.common import StageError

            raise StageError(
                f"engine '{ctx.require_project().engine}' has no voice-aware script parser: export text with "
                "`VNTextPatch extractlocal <game> <json dir>` and run `vndub parse --import-vntextpatch <json dir>`"
            )
        return import_vntextpatch(ctx, Path(src), audio)


def _natural_key(s: str) -> list[object]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def textless_result(ctx: "StageContext", audio: list[AudioIndexRecord], why: str = "") -> ParseResult:
    """Mode C: one voiced line with empty text per voice file, in file-name order.

    The speaker is the file's prefix (``engine_opts.generic.prefix_regex``; files that do not match
    belong to the narrator), the scene is the parent folder (``voice`` for loose files). Lines carry
    ``flags: ["voice_only"]``; ``link`` (``recover_text``) fills the text from the audio.
    """
    rx = re.compile(ctx.cfg.parse.engine_opts.generic.prefix_regex)
    lb = LineBuilder(ctx.cfg.parse.narrator_names)
    for a in sorted(audio, key=lambda r: _natural_key(r.relpath)):
        pp = PurePosixPath(a.relpath)
        m = rx.match(pp.stem)
        prefix = m.group("prefix").lower() if m else ""
        scene = pp.parent.name or "voice"
        line = lb.add(scene, "", speaker_raw=prefix, flags=["voice_only"], source_file=a.relpath)
        bind_line(line, a, a.relpath)
    chars = build_characters(lb.lines)
    note = f"voice-only lines (mode C): {len(lb.lines)} voice files without script text" + (f" ({why})" if why else "")
    return ParseResult(lb.lines, chars, [note])


def import_vntextpatch(ctx: "StageContext", src: Path, audio: list[AudioIndexRecord]) -> ParseResult:
    opts = ctx.cfg.parse.engine_opts.generic
    if not src.exists():
        from vndub.stages.common import StageError

        raise StageError(f"VNTextPatch dir not found: {src}")
    scenes = read_vntextpatch_dir(src)
    lb = LineBuilder(ctx.cfg.parse.narrator_names)
    for scene, rows in scenes:
        for i, r in enumerate(rows):
            lb.add(scene, r["message"], speaker_raw=r["name"], source_file=f"{scene}.json", source_line=i)
    counts: dict[str, int] = {}
    for l in lb.lines:
        if l.speaker_id != "narrator":
            counts[l.speaker_id] = counts.get(l.speaker_id, 0) + 1
    groups = group_by_prefix(audio, opts.prefix_regex)
    known = {k: c.voice_prefix for k, c in load_characters(ctx.paths.characters).items()} if ctx.paths.characters.exists() else {}
    mapping, notes = infer_prefix_map(counts, groups, opts.prefix_map, known)
    cursor = {p: 0 for p in groups}
    for l in lb.lines:
        p = mapping.get(l.speaker_id)
        if not p:
            continue
        files = groups[p]
        k = cursor[p]
        if k < len(files):
            bind_line(l, files[k], files[k].relpath)
            l.flags.append("prefix_guess")
            cursor[p] = k + 1
    for p, k in cursor.items():
        if k < len(groups[p]):
            notes.append(f"prefix '{p}': {len(groups[p]) - k} files left unassigned")
    chars = build_characters(lb.lines)
    for s, p in mapping.items():
        if s in chars:
            chars[s].voice_prefix = p
    notes.insert(0, f"VNTextPatch import: {sum(len(r) for _, r in scenes)} messages from {len(scenes)} files; prefixes {sorted(groups)}")
    return ParseResult(lb.lines, chars, notes)

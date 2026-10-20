"""NScripter / ONScripter: ``nscript.dat`` (XOR 0x84) or ``0.txt``..``99.txt``; arc*.nsa / *.sar / *.ns2.

Voice binding: ``dwave CH,"path"`` (also ``dwaveload``/``dwaveplay``, ``wave``
is BGM-ish and ignored) binds to the next text line. Commands may share a
line separated by ``:``. Speakers are not a language feature: they come from
``engine_opts.nscripter.speaker_regex`` (default ``^【(?P<name>[^】]+)】``) or the
``名前「……」`` convention. Labels (``*name``) give the scene.

Text lines: anything that does not start with a command word, label, comment
or ``~``; inline ``@`` (click wait), ``\\`` (page end), ``/`` (continue),
``!wNNN``/``!dNNN``/``!sNNN`` waits, ``#RRGGBB`` colours and ``$var``/``%var``
are stripped. Loose files beside the exe override archive members, so the
patch is just loose files.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vndub.engines import nsa
from vndub.engines.base import (
    BaseEngine,
    DetectResult,
    ExtractResult,
    LineBuilder,
    ParseResult,
    VoiceResolver,
    bind_line,
    build_characters,
    copy_loose,
    find_audio,
    normalize_speaker,
    write_install,
)
from vndub.manifest.io import AudioIndexRecord, FinalIndexRecord

if TYPE_CHECKING:
    from vndub.stages.common import StageContext

DWAVE_RE = re.compile(r'^(?:dwave|dwaveload|dwaveplay|dwaveloop)\s+(?P<ch>[^,]+)\s*,\s*"(?P<file>[^"]+)"', re.I)
CMD_RE = re.compile(r"^[a-z_][a-z0-9_]*\b", re.I)
STRIP_RE = re.compile(r"!(?:w|d|s)\d+|!s[d]?|#[0-9A-Fa-f]{6}|[@\\/]|[$%][A-Za-z_]\w*")
NAME_QUOTE_RE = re.compile(r"^(?P<name>[^「『\s、。]{1,8})[「『](?P<text>.*)[」』]$")


def split_commands(line: str) -> list[str]:
    """Split ``cmd a:cmd b`` on ':' outside quotes."""
    parts, cur, q = [], [], False
    for c in line:
        if c == '"':
            q = not q
        if c == ":" and not q:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
    parts.append("".join(cur).strip())
    return [p for p in parts if p]


def strip_comment(line: str) -> str:
    out, q = [], False
    for c in line:
        if c == '"':
            q = not q
        if c == ";" and not q:
            break
        out.append(c)
    return "".join(out).rstrip()


def is_text_line(s: str) -> bool:
    if not s:
        return False
    c = s[0]
    if c in "*;~":
        return False
    if c == "`" or c == "^":  # English text markers (ONScripter-EN / ponscripter)
        return True
    if ord(c) > 0x7F:
        return True
    return False


def parse_nscript(text: str, speaker_regex: str) -> list[dict[str, Any]]:
    """-> rows {label, text, speaker, voice, line, raw}"""
    sp_re = re.compile(speaker_regex)
    rows: list[dict[str, Any]] = []
    label = "start"
    pending: str | None = None
    for no, raw in enumerate(text.splitlines(), 1):
        s = strip_comment(raw).strip()
        if not s:
            continue
        if s.startswith("*"):
            label = s[1:].split()[0]
            continue
        if is_text_line(s):
            t = STRIP_RE.sub("", s.lstrip("`^")).strip()
            speaker = ""
            m = sp_re.match(t)
            if m:
                speaker = m.group("name")
                t = t[m.end():]
            else:
                m2 = NAME_QUOTE_RE.match(t)
                if m2:
                    speaker = m2.group("name")
                    t = "「" + m2.group("text") + "」"
            clean = STRIP_RE.sub("", t).strip()
            if clean:
                rows.append({"label": label, "text": clean, "speaker": normalize_speaker(speaker), "voice": pending, "line": no, "raw": raw.strip()})
            pending = None
            continue
        for cmd in split_commands(s):
            m = DWAVE_RE.match(cmd)
            if m:
                pending = m.group("file").replace("\\", "/")
    return rows


def script_text(game: Path, encoding: str = "cp932") -> tuple[str, list[Path]]:
    for name in ("nscript.dat", "pscript.dat"):
        p = game / name
        if p.exists():
            return nsa.decode_nscript(p.read_bytes(), encoding), [p]
    txts = sorted((p for p in game.glob("*.txt") if re.fullmatch(r"\d{1,2}", p.stem)), key=lambda p: int(p.stem))
    if txts:
        parts = []
        for p in txts:
            data = p.read_bytes()
            try:
                parts.append(data.decode("utf-8"))
            except UnicodeDecodeError:
                parts.append(data.decode(encoding, errors="replace"))
        return "\n".join(parts), txts
    return "", []


class NScripterEngine(BaseEngine):
    name = "nscripter"
    out_name = "nscripter"

    def detect(self, game_dir: Path) -> DetectResult | None:
        has_script = (game_dir / "nscript.dat").exists() or (game_dir / "pscript.dat").exists() or (game_dir / "0.txt").exists() or (game_dir / "00.txt").exists()
        archives = nsa.archive_files(game_dir)
        if not (has_script or archives):
            return None
        conf = 0.9 if has_script and archives else 0.8 if has_script else 0.55
        sample = next(iter(find_audio(game_dir / "voice")), None) if (game_dir / "voice").is_dir() else None
        notes = []
        if (game_dir / "nscript.dat").exists():
            notes.append("nscript.dat (XOR 0x84) script")
        return DetectResult("nscripter", conf, version="onscripter", archives=[a.name for a in archives], notes=notes, voice_sample=sample)

    def extract(self, ctx: "StageContext") -> ExtractResult:
        game = ctx.game_dir
        opts = ctx.cfg.parse.engine_opts.nscripter
        raw = ctx.paths.raw / "nscripter"
        raw.mkdir(parents=True, exist_ok=True)
        text, sources = script_text(game, opts.encoding)
        warnings: list[str] = []
        scripts: list[Path] = []
        if text:
            out = raw / "script.txt"
            out.write_text(text, encoding="utf-8")
            scripts.append(out)
        else:
            warnings.append("no nscript.dat / pscript.dat / 0.txt found")
        arc_root = raw / "nsa"
        for a in nsa.archive_files(game):
            dest = arc_root  # shared namespace
            try:
                with nsa.NSAReader(a, opts.encoding) as r:
                    names = [e.name for e in r.entries if Path(e.name).suffix.lower() in (".wav", ".ogg", ".mp3", ".opus", ".flac")]
                    # first archive containing a name wins (ONScripter search order)
                    todo = [n for n in names if not (dest / n).exists()]
                    r.extract_all(dest, todo)
            except (nsa.NSAError, OSError, ValueError) as e:
                warnings.append(f"{a.name}: cannot read ({e})")
        voices: list[Path] = []
        seen: set[str] = set()
        for root in (game, arc_root):
            for v in find_audio(root):
                if root == game and raw in v.parents:
                    continue
                key = v.relative_to(root).as_posix().lower()
                if key in seen:
                    continue
                seen.add(key)
                voices.append(v)
        voices = [v for v in voices if ctx.paths.work not in v.parents or arc_root in v.parents]
        return ExtractResult(script_files=scripts, voice_root=game, voice_files=voices, voice_roots=[arc_root], warnings=warnings, notes=[f"script from {', '.join(p.name for p in sources)}" if sources else "no script"])

    def parse(self, ctx: "StageContext", extracted: ExtractResult, audio: list[AudioIndexRecord]) -> ParseResult:
        opts = ctx.cfg.parse.engine_opts.nscripter
        resolver = VoiceResolver(audio)
        lb = LineBuilder(ctx.cfg.parse.narrator_names)
        for p in extracted.script_files:
            for row in parse_nscript(p.read_text(encoding="utf-8"), opts.speaker_regex):
                rec = lb.add(row["label"], row["text"], speaker_raw=row["speaker"], raw=row["raw"], source_file=p.name, source_line=row["line"])
                if row["voice"]:
                    bind_line(rec, resolver.resolve(row["voice"]), row["voice"])
        return ParseResult(lb.lines, build_characters(lb.lines))

    def pack(self, ctx: "StageContext", finals: list[FinalIndexRecord]) -> dict[str, Any]:
        out = ctx.paths.out_engine("nscripter")
        n = copy_loose(ctx, finals, out)
        write_install(out, "nscripter", [
            f"Copy the contents of this folder next to the game exe ({ctx.game_dir}), keeping the relative paths.",
            "ONScripter/NScripter read loose files before arc*.nsa archives, so the new voices override the originals.",
        ])
        return {"mode": "loose", "files": n, "out": str(out)}

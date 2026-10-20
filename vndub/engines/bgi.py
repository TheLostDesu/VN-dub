"""BGI / Ethornell: ``*.arc`` (BURIKO ARC20) + ``._bp`` scripts.

Extraction: ``bgi-extractor -t -f json -o <out> <archive>`` (tools.yaml:
``bgi_extractor``), which emits speaker + line + voice id per message. The JSON
shape is not documented, so the parser is defensive: it accepts a list of
records or a dict holding one (``lines``/``messages``/``entries``/``texts``),
and reads speaker from ``speaker|name|character|chara``, text from
``text|line|message|msg``, voice from ``voice|voice_id|vo|voice_file``; voice ids
embedded in text as ``(yuki_000003)`` are recognised too.
Repacking BGI archives is weak (BGIKit/EthornellTools archived): pack writes
loose files plus instructions.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from vndub.engines.base import (
    BaseEngine,
    DetectResult,
    ExtractResult,
    LineBuilder,
    ParseResult,
    VoiceResolver,
    bind_line,
    build_characters,
    find_audio,
    read_text_any,
)
from vndub.manifest.io import AudioIndexRecord

if TYPE_CHECKING:
    from vndub.stages.common import StageContext

SPK_KEYS = ("speaker", "name", "character", "chara", "speaker_name")
TXT_KEYS = ("text", "line", "message", "msg", "body")
VO_KEYS = ("voice", "voice_id", "vo", "voice_file", "voiceid")
INLINE_VOICE = re.compile(r"^\(([A-Za-z]+_?\d+)\)\s*")


def iter_records(data: Any) -> Iterable[dict[str, Any]]:
    if isinstance(data, list):
        for x in data:
            if isinstance(x, dict):
                if any(k in x for k in TXT_KEYS):
                    yield x
                else:
                    yield from iter_records(list(x.values()))
            elif isinstance(x, list):
                yield from iter_records(x)
    elif isinstance(data, dict):
        for k in ("lines", "messages", "entries", "texts", "scenario", "data"):
            if k in data:
                yield from iter_records(data[k])
                return
        if any(k in data for k in TXT_KEYS):
            yield data


def _first(d: dict[str, Any], keys: tuple[str, ...]) -> str:
    for k in keys:
        v = d.get(k)
        if isinstance(v, (str, int)) and str(v).strip():
            return str(v).strip()
    return ""


def parse_bgi_json(data: Any) -> list[dict[str, str]]:
    out = []
    for r in iter_records(data):
        text = _first(r, TXT_KEYS)
        voice = _first(r, VO_KEYS)
        m = INLINE_VOICE.match(text)
        if m and not voice:
            voice, text = m.group(1), text[m.end():]
        if not text:
            continue
        out.append({"speaker": _first(r, SPK_KEYS), "text": text, "voice": voice})
    return out


class BgiEngine(BaseEngine):
    name = "bgi"
    out_name = "bgi"

    def detect(self, game_dir: Path) -> DetectResult | None:
        arcs = sorted(game_dir.glob("*.arc"))
        bp = list(game_dir.rglob("*._bp"))[:3]
        exe = (game_dir / "BGI.exe").exists() or any(game_dir.glob("BGI*.exe"))
        jsons = sorted((game_dir / "bgi_json").glob("*.json")) if (game_dir / "bgi_json").is_dir() else []
        if not ((arcs and (bp or exe)) or jsons):
            return None
        mjo = list(game_dir.glob("*.mjo"))
        if mjo:
            return None  # Majiro uses .arc too
        conf = 0.85 if exe else 0.7 if arcs else 0.6
        return DetectResult("bgi", conf, archives=[a.name for a in arcs], notes=["extract runs bgi-extractor -t -f json"])

    def extract(self, ctx: "StageContext") -> ExtractResult:
        game = ctx.game_dir
        raw = ctx.paths.raw / "bgi"
        warnings: list[str] = []
        jsons = sorted(game.rglob("*.json"))
        if not jsons:
            if ctx.tools.resolve("bgi_extractor") is None:
                warnings.append("bgi-extractor not configured (tools.yaml: bgi_extractor); cannot extract scripts")
            else:
                for a in sorted(game.glob("*.arc")):
                    out = raw / a.stem
                    out.mkdir(parents=True, exist_ok=True)
                    r = ctx.tools.run("bgi_extractor", ["-t", "-f", "json", "-o", str(out), str(a)], check=False)
                    if r.returncode != 0:
                        warnings.append(f"bgi-extractor failed on {a.name}: {r.stderr.strip()[-200:]}")
            jsons = sorted(raw.rglob("*.json")) if raw.exists() else []
        voice_root = raw if raw.exists() and find_audio(raw) else game
        return ExtractResult(script_files=jsons, voice_root=voice_root, voice_files=find_audio(voice_root), warnings=warnings)

    def parse(self, ctx: "StageContext", extracted: ExtractResult, audio: list[AudioIndexRecord]) -> ParseResult:
        resolver = VoiceResolver(audio)
        lb = LineBuilder(ctx.cfg.parse.narrator_names)
        notes = []
        for p in extracted.script_files:
            try:
                data = json.loads(read_text_any(p))
            except json.JSONDecodeError as e:
                notes.append(f"{p.name}: invalid JSON ({e})")
                continue
            for i, row in enumerate(parse_bgi_json(data)):
                rec = lb.add(p.stem, row["text"], speaker_raw=row["speaker"], source_file=p.name, source_line=i)
                if row["voice"]:
                    bind_line(rec, resolver.resolve(row["voice"]), row["voice"])
        return ParseResult(lb.lines, build_characters(lb.lines), notes)

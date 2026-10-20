"""TyranoScript / TyranoBuilder: plain ``data/scenario/*.ks``, voices in ``data/sound`` / ``data/voice``.

Voice convention: ``[voconfig sebuf=2 name="akane" vostorage="akane_{number}.ogg" number=1]``
then ``[vostart]``; every message spoken by ``#akane`` auto-plays
``akane_1.ogg``, ``akane_2.ogg``, ... (the counter is per character and
advances per message). ``[vostop]`` stops auto voicing. Explicit
``[playse storage=...]`` right before a message also binds as a voice when the
file lives in a voice folder. ``[chara_new name="akane" jname="あかね"]`` gives
display names. ``#name:face`` sets speaker + expression.

Packing: loose files under ``data/`` (browser/NW.js builds read loose files;
packaged ``app.asar``/``.exe`` builds must be unpacked first, see INSTALL.md).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    read_text_any,
    speaker_id_for,
    write_install,
)
from vndub.engines.kirikiri import TAG_RE, parse_tag
from vndub.manifest.io import AudioIndexRecord, FinalIndexRecord

if TYPE_CHECKING:
    from vndub.stages.common import StageContext

END_TAGS = {"p", "l", "s", "cm", "er"}


def collect_jnames(texts: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for t in texts:
        for m in TAG_RE.finditer(t):
            name, attrs = parse_tag(m.group("body"))
            if name == "chara_new" and attrs.get("name"):
                out[attrs["name"]] = attrs.get("jname", attrs["name"])
    return out


def parse_tyrano(text: str, scene: str, filename: str, state: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """``state`` carries voconfig counters / vostart / jnames across files (play order)."""
    state = state if state is not None else {}
    rows: list[dict[str, Any]] = []
    voconfig: dict[str, dict[str, Any]] = state.setdefault("voconfig", {})
    vo_on = state.get("vo_on", False)
    jnames: dict[str, str] = state.setdefault("jnames", {})
    speaker = ""
    explicit_voice: str | None = None
    buf: list[str] = []
    start = 0
    in_script = False

    def emit(no: int) -> None:
        nonlocal buf, explicit_voice, speaker
        t = "".join(buf).strip()
        if t:
            voice = explicit_voice
            flags: list[str] = []
            if voice is None and vo_on and speaker in voconfig:
                vc = voconfig[speaker]
                voice = vc["pattern"].replace("{number}", str(vc["number"]))
                vc["number"] += 1
                flags.append("voconfig")
            rows.append({"speaker": jnames.get(speaker, speaker), "speaker_key": speaker, "text": t, "voice": voice, "line": start or no, "flags": flags})
            explicit_voice = None
        buf = []

    for no, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if in_script:
            if s.startswith("[endscript"):
                in_script = False
            continue
        if not s or s.startswith(";"):
            continue
        if s.startswith("*"):
            emit(no)
            continue
        if s.startswith("#"):
            emit(no)
            speaker = s[1:].split(":", 1)[0].strip()
            continue
        if s.startswith("@"):
            s = "[" + s[1:] + "]"
        pos = 0
        for m in TAG_RE.finditer(s):
            chunk = s[pos : m.start()]
            if chunk:
                if not buf:
                    start = no
                buf.append(chunk)
            pos = m.end()
            name, attrs = parse_tag(m.group("body"))
            if name == "iscript":
                in_script = True
                break
            if name == "voconfig" and attrs.get("name"):
                key = attrs["name"]
                cur = voconfig.get(key, {"pattern": "", "number": 1})
                if attrs.get("vostorage"):
                    cur["pattern"] = attrs["vostorage"]
                if attrs.get("number"):
                    try:
                        cur["number"] = int(attrs["number"])
                    except ValueError:
                        pass
                voconfig[key] = cur
            elif name == "vostart":
                vo_on = True
            elif name == "vostop":
                vo_on = False
            elif name == "chara_new" and attrs.get("name"):
                jnames[attrs["name"]] = attrs.get("jname", attrs["name"])
            elif name in ("playse", "voice") and attrs.get("storage"):
                explicit_voice = attrs["storage"] if name == "voice" or "voice" in attrs.get("storage", "").lower() or attrs.get("buf") in ("2", "voice") else explicit_voice
            elif name in END_TAGS:
                emit(no)
        tail = s[pos:]
        if tail and not in_script:
            if not buf:
                start = no
            buf.append(tail)
    emit(len(text.splitlines()) + 1)
    state["vo_on"] = vo_on
    return rows


class TyranoEngine(BaseEngine):
    name = "tyrano"
    out_name = "tyrano"

    def detect(self, game_dir: Path) -> DetectResult | None:
        for base in (game_dir, game_dir / "resources" / "app"):
            sc = base / "data" / "scenario"
            if sc.is_dir() and any(sc.rglob("*.ks")):
                conf = 0.9 if (base / "tyrano").is_dir() else 0.8
                return DetectResult("tyrano", conf, notes=[f"scenario dir {sc.relative_to(game_dir)}"])
        if any(game_dir.glob("*.asar")) or (game_dir / "resources" / "app.asar").exists():
            return DetectResult("tyrano", 0.3, notes=["packed app.asar: unpack it (npx asar extract) to reach data/scenario"])
        return None

    def _base(self, game: Path, scenario_dir: str) -> Path:
        for base in (game, game / "resources" / "app"):
            if (base / scenario_dir).is_dir():
                return base
        return game

    def extract(self, ctx: "StageContext") -> ExtractResult:
        sd = ctx.cfg.parse.engine_opts.tyrano.scenario_dir
        base = self._base(ctx.game_dir, sd)
        scripts = sorted((base / sd).rglob("*.ks"))
        voices = []
        for d in ("data/sound", "data/voice", "data/bgm"):
            if (base / d).is_dir() and not d.endswith("bgm"):
                voices += find_audio(base / d)
        return ExtractResult(script_files=scripts, voice_root=base / "data", voice_files=sorted(set(voices)), notes=[f"{len(scripts)} scenario files"])

    def parse(self, ctx: "StageContext", extracted: ExtractResult, audio: list[AudioIndexRecord]) -> ParseResult:
        resolver = VoiceResolver(audio)
        lb = LineBuilder(ctx.cfg.parse.narrator_names)
        texts = {p: read_text_any(p) for p in extracted.script_files}
        # first.ks runs first in TyranoScript; the rest in name order
        order = sorted(texts, key=lambda p: (p.name != "first.ks", p.as_posix()))
        state: dict[str, Any] = {"jnames": collect_jnames(list(texts.values()))}
        for p in order:
            for row in parse_tyrano(texts[p], p.stem, p.name, state):
                key = row["speaker_key"]
                sid = speaker_id_for(key, ctx.cfg.parse.narrator_names) if key else "narrator"
                rec = lb.add(p.stem, re.sub(r"\[[^\]]*\]", "", row["text"]).strip(), speaker_raw=row["speaker"], speaker_id=sid, source_file=p.name, source_line=row["line"], flags=list(row["flags"]))
                if row["voice"]:
                    a = resolver.resolve(row["voice"], prefer_dir="sound") or resolver.resolve(row["voice"], prefer_dir="voice")
                    bind_line(rec, a, row["voice"])
        return ParseResult(lb.lines, build_characters(lb.lines))

    def pack(self, ctx: "StageContext", finals: list[FinalIndexRecord]) -> dict[str, Any]:
        out = ctx.paths.out_engine("tyrano") / "data"
        n = copy_loose(ctx, finals, out)
        write_install(ctx.paths.out_engine("tyrano"), "tyrano", [
            "Copy the `data` folder over the game's `data` folder (keeping paths).",
            "Packaged builds (app.asar / single exe) must be unpacked first (e.g. `npx asar extract resources/app.asar resources/app`) and run unpacked.",
        ])
        return {"mode": "loose", "files": n, "out": str(out)}

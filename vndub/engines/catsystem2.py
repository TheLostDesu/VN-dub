"""CatSystem2: KIF ``*.int`` archives, ``.cst`` scenes; voices in ``pcm_<x>.int``.

Extraction: TriggersTools.CatSystem2 CLI (tools.yaml: ``catsystem2_cli``) or
asmodean ``exkifint`` (v2 works, v3 reportedly not) for ``.int``; CST scenes are
decompiled to text by TriggersTools. KIF v3 archives are flagged as encrypted.

Parsing: decompiled CST lines carry a type code: 0x21 = speaker name,
0x20 = message, 0x30 = command (voice command ``pcm <name>``, UNVERIFIED).
Accepted text forms: ``0x21 名前`` / ``[21] 名前`` / ``21: 名前`` per line,
or a JSON list of ``{"type": 33, "content": "..."}``.
Repacking ``pcm_X.int`` works only for unencrypted archives (CatSceneEditor).
"""

from __future__ import annotations

import json
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
    find_audio,
    read_text_any,
)
from vndub.manifest.io import AudioIndexRecord

if TYPE_CHECKING:
    from vndub.stages.common import StageContext

LINE_RE = re.compile(r"^\s*(?:0x|\[)?(?P<type>2[01]|30)\]?\s*[:\t ]\s*(?P<content>.*)$")
VOICE_CMD = re.compile(r"\b(?:pcm|voice|vo)\s+(?P<name>[\w\-]+)", re.I)


def parse_cst_text(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    stripped = text.lstrip()
    items: list[tuple[int, str]] = []
    if stripped.startswith("["):
        try:
            data = json.loads(stripped)
            for x in data:
                if isinstance(x, dict):
                    t = x.get("type")
                    t = int(t, 16) if isinstance(t, str) else int(t)
                    items.append((t, str(x.get("content", ""))))
        except (json.JSONDecodeError, ValueError, TypeError):
            items = []
    if not items:
        for line in text.splitlines():
            m = LINE_RE.match(line)
            if m:
                items.append((int(m.group("type"), 16), m.group("content")))
    speaker, voice = "", None
    for t, content in items:
        if t == 0x21:
            speaker = content.strip()
        elif t == 0x30:
            m = VOICE_CMD.search(content)
            if m:
                voice = m.group("name")
        elif t == 0x20:
            msg = re.sub(r"\\[a-zA-Z]|\[[^\]]*\]", "", content).strip()
            if msg:
                rows.append({"speaker": speaker, "text": msg, "voice": voice or ""})
            speaker, voice = "", None
    return rows


class CatSystem2Engine(BaseEngine):
    name = "catsystem2"
    out_name = "catsystem2"

    def detect(self, game_dir: Path) -> DetectResult | None:
        ints = sorted(game_dir.glob("*.int"))
        csts = list(game_dir.rglob("*.cst"))[:3]
        if not ints and not csts:
            return None
        enc = False
        notes = []
        for p in ints[:5]:
            with open(p, "rb") as f:
                head = f.read(16)
            if head[:4] == b"KIF\x00":
                # KIF v3 has a "__key__.dat" entry used for Blowfish keys -> treat as encrypted
                with open(p, "rb") as f:
                    blob = f.read(4096)
                if b"__key__.dat" in blob:
                    enc = True
                    notes.append(f"{p.name}: KIF with __key__.dat (encrypted)")
        conf = 0.85 if ints and (csts or any(p.name.startswith("scene") for p in ints)) else 0.6
        return DetectResult("catsystem2", conf, encrypted=enc, archives=[p.name for p in ints], notes=notes)

    def extract(self, ctx: "StageContext") -> ExtractResult:
        game = ctx.game_dir
        raw = ctx.paths.raw / "catsystem2"
        warnings: list[str] = []
        texts = list(game.rglob("*.cst.txt")) + list(game.rglob("*.cst.json"))
        if raw.exists():
            texts += list(raw.rglob("*.txt"))
        texts = sorted(set(texts))
        if not texts:
            tool = "catsystem2_cli" if ctx.tools.resolve("catsystem2_cli") else ("exkifint" if ctx.tools.resolve("exkifint") else None)
            if tool is None:
                warnings.append("neither TriggersTools.CatSystem2 (tools.yaml: catsystem2_cli) nor exkifint configured; cannot extract .int")
            else:
                raw.mkdir(parents=True, exist_ok=True)
                for a in sorted(game.glob("*.int")):
                    r = ctx.tools.run(tool, [str(a), str(raw / a.stem)], check=False)
                    if r.returncode != 0:
                        warnings.append(f"{tool} failed on {a.name}: {r.stderr.strip()[-200:]}")
                texts = sorted(raw.rglob("*.txt"))
        root = raw if raw.exists() else game
        return ExtractResult(script_files=texts, voice_root=root, voice_files=find_audio(root), warnings=warnings)

    def parse(self, ctx: "StageContext", extracted: ExtractResult, audio: list[AudioIndexRecord]) -> ParseResult:
        resolver = VoiceResolver(audio)
        lb = LineBuilder(ctx.cfg.parse.narrator_names)
        for p in extracted.script_files:
            for i, row in enumerate(parse_cst_text(read_text_any(p))):
                rec = lb.add(p.name.split(".")[0], row["text"], speaker_raw=row["speaker"], source_file=p.name, source_line=i)
                if row["voice"]:
                    bind_line(rec, resolver.resolve(row["voice"]), row["voice"])
        return ParseResult(lb.lines, build_characters(lb.lines))

    def pack(self, ctx: "StageContext", finals: list) -> dict[str, Any]:
        res = super().pack(ctx, finals)
        from vndub.engines.base import write_install

        write_install(ctx.paths.out_engine(self.out_name), "catsystem2", [
            "Voices live in pcm_<letter>.int KIF archives. Rebuild the matching pcm_*.int with CatSceneEditor (unencrypted archives only) "
            "replacing the files in this folder, keeping names.",
            "Encrypted (KIF v3) archives cannot be rebuilt by available tools (UNVERIFIED).",
        ])
        return res

"""Majiro (``*.arc`` incl. ``voice.arc``, ``*.mjo`` scripts): thin wrapper.

Extraction: AtomCrafty MajiroTools (tools.yaml: ``majiro_tools``):
``.mjo`` -> ``.mjil`` (disassembly, contains the voice calls) + ``.mjres`` CSV
(all dialogue text); ``.arc`` unpacking incl. ``voice.arc``.
TODO: an ``.mjil`` voice-call parser (not verified on a real game). Until then
use ``vndub parse --import-vntextpatch <dir>`` (VNTextPatch supports ``.mjo``)
with prefix-based voice assignment and ``link`` verification.

Commands::

    MajiroTools unpack <game>/voice.arc <work>/raw/majiro/voice
    MajiroTools disassemble <game>/scenario.arc <work>/raw/majiro/mjil

Packing: Majiro searches archives in priority order, so a patch ``.arc``
containing the replaced voices (built with MajiroTools) overrides voice.arc.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from vndub.engines.base import DetectResult, ExtractResult, find_audio
from vndub.engines.generic_prefix import GenericPrefixEngine

if TYPE_CHECKING:
    from vndub.stages.common import StageContext


class MajiroEngine(GenericPrefixEngine):
    name = "majiro"
    out_name = "majiro"

    def detect(self, game_dir: Path) -> DetectResult | None:
        mjo = list(game_dir.rglob("*.mjo"))[:3]
        arcs = sorted(game_dir.glob("*.arc"))
        if not mjo and not (game_dir / "voice.arc").exists():
            return None
        if not mjo and not arcs:
            return None
        return DetectResult("majiro", 0.8 if mjo else 0.5, archives=[a.name for a in arcs], notes=["text via VNTextPatch import; voice by prefix"])

    def extract(self, ctx: "StageContext") -> ExtractResult:
        raw = ctx.paths.raw / "majiro"
        warnings: list[str] = []
        if ctx.tools.resolve("majiro_tools") is None:
            warnings.append("MajiroTools not configured (tools.yaml: majiro_tools); only loose audio is used")
        else:
            for a in sorted(ctx.game_dir.glob("*.arc")):
                r = ctx.tools.run("majiro_tools", ["unpack", str(a), str(raw / a.stem)], check=False)
                if r.returncode != 0:
                    warnings.append(f"MajiroTools unpack failed on {a.name}: {r.stderr.strip()[-200:]}")
        root = raw if raw.exists() else ctx.game_dir
        return ExtractResult(script_files=[], voice_root=root, voice_files=find_audio(root), warnings=warnings, notes=["parse needs --import-vntextpatch"])

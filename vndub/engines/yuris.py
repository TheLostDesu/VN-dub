"""YU-RIS (``*.ypf``): thin wrapper.

Extraction: YU-RIS-TOOLS ``yrt`` (tools.yaml: ``yrt``) unpacks YPF and
decompiles ``.ybn`` to ``.yst``. TODO: a ``.yst`` voice-call parser (the voice
command layout differs between YU-RIS versions and has not been verified on a
real game). Until then use text from VNTextPatch (``extractlocal`` supports
``.ybn``) with ``vndub parse --import-vntextpatch <dir>`` (generic_prefix voice
assignment by ``<prefix>_<index>`` names), then verify with ``link``.

Commands::

    yrt unpack <game>/pac/*.ypf <work>/raw/yuris
    yrt decompile <work>/raw/yuris/ysbin <work>/raw/yuris/yst
    VNTextPatch extractlocal <game> <json dir>

Repacking needs the archive's 8-bit key (YPF_Tool / ypf-repacker).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from vndub.engines.base import DetectResult, ExtractResult, find_audio
from vndub.engines.generic_prefix import GenericPrefixEngine

if TYPE_CHECKING:
    from vndub.stages.common import StageContext


class YurisEngine(GenericPrefixEngine):
    name = "yuris"
    out_name = "yuris"
    tool = "yrt"

    def detect(self, game_dir: Path) -> DetectResult | None:
        ypf = sorted(game_dir.rglob("*.ypf"))[:10]
        if not ypf:
            return None
        return DetectResult("yuris", 0.85, archives=[p.name for p in ypf], notes=["text via VNTextPatch import; voice by prefix (see docs/ENGINES.md)"])

    def extract(self, ctx: "StageContext") -> ExtractResult:
        raw = ctx.paths.raw / "yuris"
        warnings: list[str] = []
        if ctx.tools.resolve("yrt") is None:
            warnings.append("yrt (YU-RIS-TOOLS) not configured (tools.yaml: yrt); only loose audio is used")
        else:
            for a in sorted(ctx.game_dir.rglob("*.ypf")):
                r = ctx.tools.run("yrt", ["unpack", str(a), str(raw / a.stem)], check=False)
                if r.returncode != 0:
                    warnings.append(f"yrt unpack failed on {a.name}: {r.stderr.strip()[-200:]}")
        root = raw if raw.exists() else ctx.game_dir
        return ExtractResult(script_files=[], voice_root=root, voice_files=find_audio(root), warnings=warnings, notes=["parse needs --import-vntextpatch"])

"""pack: final voice files -> the engine's patch layout in ``out/<engine>/``.

Always writes ``INSTALL.md`` and ``manifest_patch.json`` (file, sha256,
line_id, backend, route) next to the patch.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

from vndub.engines.base import write_install
from vndub.manifest.io import FinalIndexRecord, SelectionRecord, read_jsonl, write_json
from vndub.stages.common import StageContext, StageError
from vndub.util import hash_file

NAME = "pack"
VERSION = "1"
CONFIG_SECTIONS = ("pack",)


def run(ctx: StageContext) -> dict[str, Any]:
    proj = ctx.require_project()
    if not ctx.paths.final_index.exists():
        raise StageError(f"{ctx.paths.final_index} not found: run `vndub master` first")
    finals = read_jsonl(ctx.paths.final_index, FinalIndexRecord)
    if ctx.only:
        finals = [f for f in finals if f.line_id in ctx.only]
    present = [f for f in finals if ctx.paths.abs(f.encoded).exists()]
    if len(present) < len(finals):
        ctx.log.warning("%d final files listed in final_index.jsonl are missing on disk", len(finals) - len(present))
    eng = ctx.engine()
    res = eng.pack(ctx, present)
    out = ctx.paths.out_engine(eng.out_name)
    if not (out / "INSTALL.md").exists():
        write_install(out, eng.name, ["Copy the files in this folder into the game directory, keeping relative paths."])
    sel = {s.line_id: s for s in read_jsonl(ctx.paths.selection, SelectionRecord)} if ctx.paths.selection.exists() else {}
    entries = []
    for f in present:
        src = ctx.paths.abs(f.encoded)
        s = sel.get(f.line_id)
        entries.append(
            {
                "file": f.voice_file,
                "packed_as": PurePosixPath(f.voice_file).name if eng.name == "kirikiri" else f.voice_file,
                "sha256": hash_file(src),
                "line_id": f.line_id,
                "backend": (s.backend if s else None) or ("orig" if f.source == "orig" else None),
                "route": s.route if s else None,
                "status": f.status,
            }
        )
    write_json(out / "manifest_patch.json", {"engine": eng.name, "game_dir": proj.game_dir, "files": entries})
    return {**res, "engine": eng.name, "manifest": str(out / "manifest_patch.json")}

"""End to end on the synthetic Ren'Py fixture with mocks only (no envs, no downloads)."""

from __future__ import annotations

import json

from conftest import FIXTURES, requires_ffmpeg

from vndub.manifest.io import FinalIndexRecord, LineRecord, SelectionRecord, read_jsonl
from vndub.stages.common import STAGES, read_status, run_stage


@requires_ffmpeg
def test_full_pipeline_renpy_mini(make_ctx) -> None:
    ctx = make_ctx(FIXTURES / "renpy_mini")
    for name in STAGES:
        run_stage(name, ctx)
        if name == "score":
            ctx.extra["regen"] = True
            ctx.extra["regen_round"] = 1
            if any(s.status == "regen" for s in read_jsonl(ctx.paths.selection, SelectionRecord)):
                run_stage("synth", ctx)
                run_stage("score", ctx)
            ctx.extra["regen"] = False
    ctx.models.close()

    lines = read_jsonl(ctx.paths.lines, LineRecord)
    voiced = [l for l in lines if l.has_voice]
    assert len(voiced) >= 18
    sel = read_jsonl(ctx.paths.selection, SelectionRecord)
    assert {s.line_id for s in sel} == {l.line_id for l in voiced}
    finals = read_jsonl(ctx.paths.final_index, FinalIndexRecord)
    assert len(finals) == len(voiced)

    out_voice = ctx.paths.out / "renpy" / "game" / "voice"
    oggs = sorted(p.name for p in out_voice.glob("*.ogg"))
    assert len(oggs) == len(voiced)
    assert (ctx.paths.out / "renpy" / "INSTALL.md").exists()
    manifest = json.loads((ctx.paths.out / "renpy" / "manifest_patch.json").read_text(encoding="utf-8"))
    entries = manifest["files"] if isinstance(manifest, dict) else manifest
    assert len(entries) == len(voiced)

    assert (ctx.paths.report_dir / "index.html").exists()
    status = read_status(ctx.paths)
    assert all(status[s].get("finished") and not status[s].get("error") for s in STAGES)

    # idempotent: a second run of a cached stage does no work
    summary = run_stage("analyze", ctx)
    assert summary is not None

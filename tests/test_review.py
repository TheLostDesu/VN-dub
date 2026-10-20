"""review: overrides writer, row filtering, batch accept, Gradio app construction."""

from __future__ import annotations

import pytest

from tests.genproj import build
from vndub.manifest.io import OverrideRecord, read_jsonl
from vndub.review import app as review
from vndub.stages import score, synth


@pytest.fixture
def reviewed(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path / "game")
    build(ctx, n_per=2, interjection=False)
    synth.run(ctx)
    score.run(ctx)
    return ctx


def test_write_override_appends(reviewed):
    ctx = reviewed
    lid = ctx.voiced_lines()[0].line_id
    review.write_override(ctx.paths, lid, "regen", {"style_prompt": "angrier"})
    review.write_override(ctx.paths, lid, "keep_original")
    rows = read_jsonl(ctx.paths.overrides, OverrideRecord)
    assert [r.action for r in rows] == ["regen", "keep_original"] and rows[0].payload["style_prompt"] == "angrier"
    assert all(r.ts for r in rows)
    with pytest.raises(ValueError):
        review.write_override(ctx.paths, lid, "delete_everything")


def test_rows_filter_and_batch_accept(reviewed):
    ctx = reviewed
    rows = review.load_rows(ctx)
    assert len(rows) == len(ctx.voiced_lines())
    assert {r["speaker"] for r in review.filter_rows(rows, speaker="haruto")} == {"haruto"}
    assert review.filter_rows(rows, lo=1.01, hi=1.0) == []
    n = review.batch_accept(ctx, 0.0)
    assert n == len(rows)
    score.run(ctx)
    from vndub.stages.genutil import load_selection

    assert all(s.status == "override" for s in load_selection(ctx).values())
    cands = review.candidates(ctx, rows[0]["line_id"])
    assert cands and cands[0]["score"] is not None
    assert "--only" in review.rerun_command(ctx, [rows[0]["line_id"]])


def test_build_app(reviewed):
    import gradio as gr

    blocks = review.build_app(reviewed)
    assert isinstance(blocks, gr.Blocks)

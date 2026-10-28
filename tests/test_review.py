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


# --- release 1.0 review fixes -------------------------------------------------------------------------------------------------


def test_rows_use_the_lines_the_stages_dub(make_ctx, tmp_path):
    from tests.p3_helpers import add_unvoiced
    from vndub.manifest.io import LinkRecord, write_jsonl

    ctx = make_ctx(tmp_path / "game", sets={"synth.voice_unvoiced": True})
    build(ctx, n_per=3, interjection=False)
    ids = add_unvoiced(ctx)
    rows = {r["line_id"]: r for r in review.load_rows(ctx)}
    # route-P lines (no original audio) were invisible: the raw has_voice filter dropped them; narration still is
    assert set(ids["hero"]) <= set(rows) and ids["narrator"] not in rows and ids["menu"] not in rows
    assert [r["line_id"] for r in review.load_rows(ctx)] == [l.line_id for l in ctx.dub_lines(apply_only=False)]
    # a voice the link stage moved away leaves the voiced rows, and the line is an unvoiced (route P) row if it is dubbed that way
    voiced = [l for l in ctx.voiced_lines(apply_only=False)]
    moved = voiced[0]
    write_jsonl(ctx.paths.link, [LinkRecord(line_id=moved.line_id, audio_id=None, status="unmatched")])
    assert ctx.voiced_lines(apply_only=False)[0].line_id != moved.line_id
    rows = {r["line_id"]: r for r in review.load_rows(ctx)}
    assert moved.line_id in rows  # now an unvoiced speaker line
    assert not next(l for l in ctx.dub_lines(apply_only=False) if l.line_id == moved.line_id).has_voice
    ctx.cfg.synth.voice_unvoiced = False
    assert moved.line_id not in {r["line_id"] for r in review.load_rows(ctx)}  # the CLI default dubs voiced lines only


def test_rerun_command_is_shell_neutral_and_quotes_paths(reviewed):
    ctx = reviewed
    cmd = review.rerun_command(ctx, ["s01_000001", "s01_000002"])
    assert "&&" not in cmd
    synth_cmd, score_cmd = cmd.split("\n")
    assert synth_cmd.startswith("uv run vndub synth ") and score_cmd.startswith("uv run vndub score ")
    assert synth_cmd.endswith("--only s01_000001,s01_000002")
    ctx.project_dir = tmp = ctx.project_dir.parent / "my game"  # a folder with a space
    quoted = review.rerun_command(ctx, ["a"])
    assert f'--project "{tmp}"' in quoted and quoted.count('--project "') == 2
    assert review.score_command(ctx, ["a", "b"]) == f'uv run vndub score --project "{tmp}" --only a,b'

"""report: renders with partial data (nothing run yet) and with full generation-side data."""

from __future__ import annotations

from tests.conftest import requires_ffmpeg
from tests.genproj import build
from vndub.config import ProjectYaml, save_project_yaml
from vndub.stages import report, score, synth


def test_report_partial(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path / "game")
    (tmp_path / "game").mkdir()
    save_project_yaml(ctx.paths.project_yaml, ProjectYaml(game_dir=str(tmp_path / "game"), engine="renpy"))
    s = report.run(ctx)
    html = (ctx.paths.report_dir / "index.html").read_text(encoding="utf-8")
    md = (ctx.paths.report_dir / "report.md").read_text(encoding="utf-8")
    assert "not available" in html and "<html" in html and "renpy" in html
    assert "# vndub report" in md
    assert s["finals"] == 0


def test_report_full(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path / "game")
    build(ctx, n_per=2)
    synth.run(ctx)
    score.run(ctx)
    s = report.run(ctx)
    html = (ctx.paths.report_dir / "index.html").read_text(encoding="utf-8")
    assert "mock_tts/A" in html and "sakura" in html and "<audio" in html
    assert "Routes A / B / C" in html
    md = (ctx.paths.report_dir / "report.md").read_text(encoding="utf-8")
    assert "mock_tts/A" in md
    assert s["lines"] > 0
    # self-contained: no external assets
    assert "http://" not in html.replace("http://www.w3.org", "") and "https://" not in html


@requires_ffmpeg
def test_report_says_what_the_patch_holds_and_leaves_out(make_ctx, tmp_path):
    from vndub.manifest.io import SelectionRecord, read_jsonl, write_jsonl
    from vndub.stages import master

    for mode in ("provisional", "skip"):
        ctx = make_ctx(tmp_path / mode / "game", sets={"master.include_review_lines": mode})
        data = build(ctx, n_per=2)
        synth.run(ctx)
        score.run(ctx)
        sel = read_jsonl(ctx.paths.selection, SelectionRecord)
        sel[0].status, sel[0].reason = "review", "no candidate passes the gates after 2 regen round(s) (best fails: cer 0.22)"
        write_jsonl(ctx.paths.selection, sel)
        master.run(ctx)
        s = report.run(ctx)
        html = (ctx.paths.report_dir / "index.html").read_text(encoding="utf-8")
        md = (ctx.paths.report_dir / "report.md").read_text(encoding="utf-8")
        assert "Patch coverage" in html and "## Patch coverage" in md
        row = next(l for l in md.splitlines() if l.startswith(f"| {sel[0].line_id} ") and "| score |" in l)
        if mode == "provisional":
            assert (s["provisional"], s["left_out"]) == (1, 0) and "| provisional |" in row
            assert "in the patch, provisional" in html and "Left out" not in html
        else:
            assert (s["provisional"], s["left_out"]) == (0, 1) and "| left out |" in row
            assert "left out 1 (review 1)" in md and "master.include_review_lines" in md and "no candidate passes the gates" in html
        assert len(sel) == len(data["lines"])

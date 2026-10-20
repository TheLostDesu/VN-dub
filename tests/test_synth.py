"""synth: route planning, candidate counts, interjection originals, sidecars, VC, regen, cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.genproj import build
from vndub.manifest.io import SelectionRecord, load_characters, read_json, save_characters, write_jsonl
from vndub.stages import synth, voicebank
from vndub.stages.common import run_stage


@pytest.fixture
def proj(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path / "game")
    build(ctx)
    return ctx


def _cands(ctx, line_id):
    d = ctx.paths.synth_dir(line_id)
    return sorted(p for p in d.glob("*.wav") if not p.name.endswith(".prevc.wav"))


def test_route_a_counts_and_sidecars(proj):
    ctx = proj
    s = synth.run(ctx)
    assert s["failed"] == 0
    assert "skipped_routes" in s and any(k.endswith(":B") for k in s["skipped_routes"])  # no performer yet
    lines = ctx.voiced_lines()
    normal = [l for l in lines if l.text_src != "えへへ。"]
    interj = [l for l in lines if l.text_src == "えへへ。"][0]
    for l in normal:
        c = _cands(ctx, l.line_id)
        n = ctx.cfg.synth.n_candidates
        # short lines (<= 2 syllables / < 1 s) and high arousal get n_candidates_short
        assert len(c) in (n, ctx.cfg.synth.n_candidates_short)
        side = read_json(c[0].with_suffix(".json"))
        assert side["backend"] == "mock_tts" and side["route"] == "A"
        assert side["text_used"] and side["style_prompt"] and side["ref_paths"]
        assert side["style"]["emotion"]
    ci = _cands(ctx, interj.line_id)
    assert any(p.name.startswith("orig_orig_") for p in ci)
    assert len([p for p in ci if p.name.startswith("mock_tts")]) == ctx.cfg.synth.n_candidates_short
    side = read_json(next(p for p in ci if p.name.startswith("orig_")).with_suffix(".json"))
    assert side["route"] == "orig" and side["text_variant"] == "orig"


def test_cache_second_run(proj):
    s1 = synth.run(proj)
    s2 = synth.run(proj)
    assert s2["cached"] == s1["candidates"] and s2.get("generated", 0) == 0


def test_route_b_with_voicebank_and_vc(proj):
    ctx = proj
    vb = voicebank.run(ctx)
    assert vb["performers"] == 2 and vb["rvc"] == 2
    s = synth.run(ctx)
    assert s["failed"] == 0
    line = ctx.voiced_lines()[0]
    b = [p for p in _cands(ctx, line.line_id) if "_B_" in p.name]
    assert b, "route B candidates expected"
    side = read_json(b[0].with_suffix(".json"))
    assert side["pre_vc_wav"] and Path(ctx.paths.abs(side["pre_vc_wav"])).exists()
    assert side["params"]["vc"]["backend"] == "mock_vc"
    assert "f0_transpose" in side["params"]["vc"]


def test_character_route_override(proj):
    ctx = proj
    voicebank.run(ctx)
    chars = load_characters(ctx.paths.characters)
    chars["haruto"].route = "B"
    save_characters(ctx.paths.characters, chars)
    synth.run(ctx)
    for l in ctx.voiced_lines():
        names = [p.name for p in _cands(ctx, l.line_id) if not p.name.startswith("orig")]
        if l.speaker_id == "haruto":
            assert names and all("_B_" in n for n in names)
        else:
            assert any("_A_" in n for n in names)


def test_regen_new_seeds_keep_old(proj):
    ctx = proj
    synth.run(ctx)
    line = ctx.voiced_lines()[0]
    before = {p.name: read_json(p.with_suffix(".json"))["seed"] for p in _cands(ctx, line.line_id)}
    write_jsonl(ctx.paths.selection, [SelectionRecord(line_id=line.line_id, status="regen", round=0)])
    ctx.extra["regen"] = True
    s = synth.run(ctx)
    assert s["lines"] == 1 and s["round"] == 1
    after = {p.name: read_json(p.with_suffix(".json")) for p in _cands(ctx, line.line_id)}
    new = {n: d for n, d in after.items() if n not in before}
    assert new and all(d["round"] == 1 and d["k"] >= 100 for d in new.values())
    assert not ({d["seed"] for d in new.values()} & set(before.values()))
    assert set(before) <= set(after)  # earlier candidates stay available


def test_set_text_override_bumps_round(proj):
    from vndub.review.app import write_override

    ctx = proj
    synth.run(ctx)
    line = ctx.voiced_lines()[0]
    write_override(ctx.paths, line.line_id, "set_text", {"text": "Совсем другой текст."})
    synth.run(ctx)
    sides = [read_json(p.with_suffix(".json")) for p in _cands(ctx, line.line_id)]
    over = [s for s in sides if s["text_variant"] == "override"]
    assert over and all(s["text_used"] == "Совсем другой текст." and s["round"] == 1 for s in over)


def test_missing_translation_skipped(proj):
    ctx = proj
    trs = (ctx.paths.translation("ru")).read_text(encoding="utf-8").splitlines()
    (ctx.paths.translation("ru")).write_text("\n".join(trs[1:]) + "\n", encoding="utf-8")
    s = synth.run(ctx)
    assert s["no_translation"] == 1


def test_stage_runner_writes_status(proj):
    run_stage("synth", proj)
    st = read_json(proj.paths.stage_status)
    assert st["synth"]["finished"] and st["synth"]["summary"]["candidates"] > 0

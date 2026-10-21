"""Casting screen API (voicebank.list_designs / propose_designs / add_reference_design / import_design /
choose_design / auto_cast / sample_lines / train_rvc) on a temp project with mock backends."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.genproj import build, speechlike
from tests.p3_helpers import add_unvoiced
from vndub.manifest.io import load_characters, read_json, read_yaml, write_json
from vndub.stages import designs as dz
from vndub.stages import voicebank
from vndub.stages.common import run_stage


@pytest.fixture
def proj(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path / "game", sets={"synth.voice_unvoiced": True})
    build(ctx, n_per=3)
    return ctx


def _ids(ds):
    return [d["design_id"] for d in ds]


# --- stars and ranking ---------------------------------------------------------


def test_star_rating_threshold_is_the_three_star_point():
    thr, w = 0.40, 0.20
    assert dz.star_rating(None, thr, w) is None
    assert dz.star_rating(0.40, thr, w) == 3
    assert dz.star_rating(0.60, thr, w) == 5 and dz.star_rating(0.95, thr, w) == 5
    assert dz.star_rating(0.20, thr, w) == 1 and dz.star_rating(-0.3, thr, w) == 1
    assert [dz.star_rating(v, thr, w) for v in (0.30, 0.35, 0.45, 0.50)] == [2, 3, 4, 4]  # half-up at x.5
    # the same formula as the contract: 1 + 4 * (sim - lo) / (hi - lo) with lo = thr - w, hi = thr + w
    for v in (0.21, 0.33, 0.47, 0.58):
        assert dz.star_rating(v, thr, w) == min(5, max(1, int(1 + 4 * (v - 0.2) / 0.4 + 0.5 + 1e-9)))


def test_star_params_use_threshold_and_calibration_width(proj):
    ctx = proj
    p = dz.star_params(ctx)
    assert p["threshold"] == pytest.approx(-1.0) and p["half_width"] == 0.2  # test profile: configured gate, no calibration
    write_json(ctx.paths.spk_calibration, {"threshold": 0.42, "embedder": "x", "eer": 0.1, "same": {"mean": 0.8}, "diff": {"mean": 0.2}})
    ctx.cfg.score.gates.spk_sim_min = "calibrated"
    p = dz.star_params(ctx)
    assert p["threshold"] == 0.42 and p["half_width"] == pytest.approx(0.30)


def test_rank_score_orders_window_then_similarity_then_f0():
    r = dz.rank_score
    assert r(0.5, 1.0, 3.0) > r(0.4, 0.0, 3.0)  # better similarity wins inside the window
    assert r(0.4, 4.0, 3.0) < r(0.1, 0.0, 3.0)  # outside the window loses to any design inside it
    assert r(0.5, 0.5, 3.0) > r(0.5, 2.0, 3.0)  # F0 only breaks ties
    assert r(None, None, 3.0) < r(0.01, None, 3.0)  # unmeasured ranks below measured


# --- sample lines --------------------------------------------------------------


def test_sample_lines_are_translated_lines_then_test_sentences(proj):
    ctx = proj
    got = voicebank.sample_lines(ctx, "sakura", 2)
    assert len(got) == 2 and all(isinstance(t, str) and t for t in got)
    ru = {t.text_tgt for t in dz.gu.load_translations(ctx).values()}
    assert got[0] in ru  # a real translated line of the character
    many = voicebank.sample_lines(ctx, "sakura", 6)
    assert len(many) == 6 and many[-1] in dz.TEST_SENTENCES["ru"]  # filled up with the configured sentences
    ctx_none = voicebank.sample_lines(ctx, "nobody", 2)
    assert ctx_none == dz.TEST_SENTENCES["ru"][:2]


# --- propose / list ------------------------------------------------------------


def test_propose_adds_n_and_list_ranks_them(proj):
    ctx = proj
    assert voicebank.list_designs(ctx, "sakura") == []
    new = voicebank.propose_designs(ctx, "sakura", n=3)
    kinds = [d["kind"] for d in new]
    assert kinds.count("voice_design") == 3 and kinds.count("clone") == 1  # the clone (route A) is offered with the first batch
    for d in new:
        assert d["spk_sim"] is not None and d["stars"] in (1, 2, 3, 4, 5) and d["f0_delta_st"] is not None
        assert 1 <= len(d["samples"]) <= 2 and all(ctx.paths.abs(s["wav"]).exists() and s["text"] for s in d["samples"])
        assert d["license"] and d["created"] and d["backend"] == "mock_tts"
    listed = voicebank.list_designs(ctx, "sakura")
    assert _ids(listed) == _ids(sorted(listed, key=lambda d: -d["rank_score"]))  # best first
    assert sorted(_ids(listed)) == sorted(_ids(new))
    stars = [d["stars"] for d in listed]
    sims = [d["spk_sim"] for d in listed]
    by_sim = sorted(zip(sims, stars), reverse=True)
    assert [s for _, s in by_sim] == sorted(stars, reverse=True)  # more similar never gets fewer stars
    # the store on disk
    data = read_json(dz.designs_path(ctx, "sakura"))
    assert data["version"] == 1 and len(data["designs"]) == 4
    # more: adds n, ids and seeds continue, no second clone
    more = voicebank.propose_designs(ctx, "sakura", n=2)
    assert [d["kind"] for d in more] == ["voice_design"] * 2
    assert len(voicebank.list_designs(ctx, "sakura")) == 6
    seeds = [d["seed"] for d in voicebank.list_designs(ctx, "sakura") if d["kind"] == "voice_design"]
    assert len(set(seeds)) == 5
    # other characters are untouched
    assert voicebank.list_designs(ctx, "haruto") == []


def test_propose_reports_progress_and_uses_sample_line_ids_and_description(proj):
    ctx = proj
    seen = []
    ctx.extra["progress_cb"] = lambda stage, done, total, item: seen.append((stage, done, total, item))
    lid = ctx.voiced_lines()[0].line_id
    tr = dz.gu.load_translations(ctx)[lid].text_tgt
    new = voicebank.propose_designs(ctx, "sakura", n=2, description="строгая учительница, низкий голос", sample_line_ids=[lid])
    assert seen and seen[0][0] == "voicebank" and seen[-1][1] == seen[-1][2]
    assert [s[1] for s in seen] == sorted(s[1] for s in seen)  # monotonic
    vd = [d for d in new if d["kind"] == "voice_design"]
    assert vd[0]["samples"][0]["line_id"] == lid and vd[0]["samples"][0]["text"] == tr
    assert "строгая учительница" in vd[0]["prompt"] and vd[0]["description"] == "строгая учительница, низкий голос"
    assert load_characters(ctx.paths.characters)["sakura"].description == "строгая учительница, низкий голос"  # saved on the character


# --- user sample ---------------------------------------------------------------


def test_add_reference_validates_duration_and_format(proj, tmp_path):
    ctx = proj
    for dur, code in ((3.0, "sample_too_short"), (4.9, "sample_too_short"), (21.0, "sample_too_long")):
        w = tmp_path / f"s{dur}.wav"
        speechlike(w, dur, 200.0)
        with pytest.raises(ValueError) as e:
            voicebank.add_reference_design(ctx, "sakura", str(w))
        assert str(e.value) == code
    txt = tmp_path / "notes.txt"
    txt.write_text("not audio")
    with pytest.raises(ValueError) as e:
        voicebank.add_reference_design(ctx, "sakura", str(txt))
    assert str(e.value) == "sample_bad_format"
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"RIFF....garbage")
    with pytest.raises(ValueError) as e:
        voicebank.add_reference_design(ctx, "sakura", str(broken))
    assert str(e.value) == "sample_bad_format"
    with pytest.raises(ValueError) as e:
        voicebank.add_reference_design(ctx, "sakura", str(tmp_path / "missing.mp3"))
    assert str(e.value) == "sample_bad_format"
    droot = ctx.paths.vb("sakura") / "designs"
    assert voicebank.list_designs(ctx, "sakura") == []
    assert not droot.exists() or not list(droot.iterdir())  # a rejected sample leaves nothing behind
    ok = tmp_path / "ok.wav"
    speechlike(ok, 6.0, 230.0)
    d = voicebank.add_reference_design(ctx, "sakura", str(ok))
    assert d["design_id"] == "ref01" and d["kind"] == "reference" and d["origin"] == "user" and d["spk_sim"] is not None and d["stars"] in range(1, 6)
    assert ctx.paths.abs(d["ref_wav"]).exists() and 5.0 <= d["duration_s"] <= 7.0
    assert voicebank.list_designs(ctx, "sakura")[0]["design_id"] == "ref01"
    for lim, dur in (("min", 5.0), ("max", 20.0)):  # the limits themselves are valid
        w = tmp_path / f"{lim}.wav"
        speechlike(w, dur + (0.05 if lim == "min" else -0.05), 220.0)
        assert voicebank.add_reference_design(ctx, "sakura", str(w))["kind"] == "reference"


def test_add_reference_mp3(proj, tmp_path):
    from tests.conftest import has_ffmpeg

    if not has_ffmpeg():
        pytest.skip("ffmpeg needed to build and decode an mp3")
    import subprocess

    wav = tmp_path / "src.wav"
    speechlike(wav, 7.0, 200.0)
    mp3 = tmp_path / "src.mp3"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav), str(mp3)], check=True)
    d = voicebank.add_reference_design(proj, "haruto", str(mp3))
    assert d["kind"] == "reference" and 6.5 <= d["duration_s"] <= 7.5


# --- library -------------------------------------------------------------------


def test_import_design_from_library(proj, tmp_path):
    ctx = proj
    wav = tmp_path / "lib.wav"
    speechlike(wav, 3.0, 190.0)
    d = voicebank.import_design(
        ctx, "haruto", {"kind": "voice_design", "backend": "mock_tts", "prompt": "A calm male voice.", "description": "library voice", "name": "Calm", "samples": [{"text": "Привет."}]}, [str(wav)],
    )
    assert d["design_id"] == "imp01" and d["kind"] == "voice_design" and d["origin"] == "library" and d["imported"]
    assert d["prompt"] == "A calm male voice." and d["samples"][0]["text"] == "Привет." and ctx.paths.abs(d["samples"][0]["wav"]).exists()
    assert d["spk_sim"] is not None and d["license"] == "MIT"
    assert _ids(voicebank.list_designs(ctx, "haruto")) == ["imp01"]
    with pytest.raises(ValueError, match="design_empty"):
        voicebank.import_design(ctx, "haruto", {}, [])
    with pytest.raises(ValueError, match="sample_bad_format"):
        voicebank.import_design(ctx, "haruto", {}, [str(tmp_path / "nope.wav")])
    assert _ids(voicebank.list_designs(ctx, "haruto")) == ["imp01"]
    # choosing it makes its sample the performer's timbre reference
    voicebank.choose_design(ctx, "haruto", "imp01", "B")
    perf = read_yaml(ctx.paths.vb_performer("haruto"))
    assert perf["design_id"] == "imp01" and len(perf["refs"]) == 1 and perf["refs"][0]["text"] == "Привет."


# --- choose / auto cast ----------------------------------------------------------


def test_choose_writes_characters_yaml_and_performer(proj):
    ctx = proj
    new = voicebank.propose_designs(ctx, "sakura", n=2)
    vd = [d for d in new if d["kind"] == "voice_design"]
    clone = next(d for d in new if d["kind"] == "clone")
    chosen = vd[1]
    out = voicebank.choose_design(ctx, "sakura", chosen["design_id"], "B")
    assert out["route"] == "B" and out["design_id"] == chosen["design_id"] and out["kind"] == "voice_design"
    c = load_characters(ctx.paths.characters)["sakura"]
    assert c.route == "B" and c.performer["design_id"] == chosen["design_id"] and c.performer["chosen_by"] == "user"
    assert c.performer["backend"] == "mock_tts" and c.performer["f0_median_hz"] and c.performer["path"] == "voicebank/sakura/performer.yaml"
    perf = read_yaml(ctx.paths.vb_performer("sakura"))
    assert perf["design_id"] == chosen["design_id"] and perf["source"] == "user" and perf["refs"]
    assert all(ctx.paths.abs(r["path"]).exists() for r in perf["refs"])
    assert [r["text"] for r in perf["refs"][:2]] == [s["text"] for s in chosen["samples"][:2]]
    # changing the choice replaces the performer's clips (no leftovers of the previous design)
    voicebank.choose_design(ctx, "sakura", vd[0]["design_id"], "P")
    perf2 = read_yaml(ctx.paths.vb_performer("sakura"))
    assert perf2["design_id"] == vd[0]["design_id"]
    assert {p.name for p in (ctx.paths.vb("sakura") / "performer").glob("*.wav")} == {Path(r["path"]).name for r in perf2["refs"]}
    assert load_characters(ctx.paths.characters)["sakura"].route == "P"
    # the clone is route A only and leaves the performer alone
    with pytest.raises(ValueError, match="route_mismatch"):
        voicebank.choose_design(ctx, "sakura", "clone", "B")
    voicebank.choose_design(ctx, "sakura", "clone", "A")
    c = load_characters(ctx.paths.characters)["sakura"]
    assert c.route == "A" and c.performer["kind"] == "clone" and c.performer["design_id"] == "clone"
    assert read_yaml(ctx.paths.vb_performer("sakura"))["design_id"] == vd[0]["design_id"]
    for bad in (("nope", "B"), (vd[0]["design_id"], "X")):
        with pytest.raises(ValueError):
            voicebank.choose_design(ctx, "sakura", *bad)
    with pytest.raises(ValueError, match="unknown_character"):
        voicebank.choose_design(ctx, "ghost", "vd01", "B")
    assert clone["design_id"] == "clone"


def test_choose_keeps_tuning_and_updates_rvc_pitch(proj):
    ctx = proj
    run_stage("voicebank", ctx)
    voicebank.set_tuning(ctx, "sakura", {"pitch_st": 2, "character_note": "x"})
    before = read_json(ctx.paths.vb_rvc("sakura") / "meta.json")["f0_transpose"]
    new = voicebank.propose_designs(ctx, "sakura", n=3)
    far = max((d for d in new if d["kind"] == "voice_design"), key=lambda d: abs((d["f0_median_hz"] or 0) - (ctx.characters()["sakura"].performer["f0_median_hz"] or 0)))
    voicebank.choose_design(ctx, "sakura", far["design_id"], "B")
    c = load_characters(ctx.paths.characters)["sakura"]
    assert c.performer["tuning"]["pitch_st"] == 2.0 and c.performer["tuning"]["character_note"] == "x"
    meta = read_json(ctx.paths.vb_rvc("sakura") / "meta.json")
    import math

    assert meta["f0_transpose"] == round(12 * math.log2(c.stats.f0_median_hz / far["f0_median_hz"])) and isinstance(before, int)


def test_auto_cast_takes_the_top_ranked_for_everyone(proj):
    ctx = proj
    add_unvoiced(ctx)
    voicebank.propose_designs(ctx, "sakura", n=3)
    voicebank.propose_designs(ctx, "haruto", n=2)
    voicebank.propose_designs(ctx, "hero", n=2)
    out = voicebank.auto_cast(ctx)
    assert set(out) == {"sakura", "haruto", "hero"}
    chars = load_characters(ctx.paths.characters)
    for sid, did in out.items():
        top = [d for d in voicebank.list_designs(ctx, sid) if d["kind"] != "clone"][0]  # never the clone while there are designs
        assert did == top["design_id"] and chars[sid].performer["design_id"] == did
        assert ctx.paths.vb_performer(sid).exists()
    assert chars["sakura"].route == "B" and chars["haruto"].route == "B" and chars["hero"].route == "P"
    # a character with only a clone design gets it, on route A
    for d in list(voicebank.list_designs(ctx, "haruto")):
        if d["kind"] != "clone":
            dz.remove_design_files(ctx, "haruto", d["design_id"])
    dz.save_designs(ctx, "haruto", [d for d in dz.load_designs(ctx, "haruto") if d["kind"] == "clone"])
    assert voicebank.auto_cast(ctx)["haruto"] == "clone"
    assert load_characters(ctx.paths.characters)["haruto"].route == "A"


# --- characters without original audio (mode B) -------------------------------------


def test_designs_for_a_character_without_audio_come_from_the_sheet(proj):
    ctx = proj
    add_unvoiced(ctx)
    voicebank.set_tuning(ctx, "hero", {"character_note": "ехидный", "timbre_preset": "older"})
    new = voicebank.propose_designs(ctx, "hero", n=3, description="усталый школьник, низкий голос")
    assert [d["kind"] for d in new] == ["voice_design"] * 3  # no clone without original audio
    for d in new:
        assert d["spk_sim"] is None and d["stars"] is None and d["f0_delta_st"] is None
        for frag in ("young male voice", "усталый школьник", "ехидный", "an older, deeper voice", "native Russian"):
            assert frag in d["prompt"]
    assert len({d["prompt"] for d in new}) == 3  # variation hints per design
    lines = [s["line_id"] for s in new[0]["samples"]]
    assert lines and all(l.startswith("s01_0009") for l in lines)  # the character's own translated lines
    assert voicebank.list_designs(ctx, "hero")[0]["design_id"] == new[0]["design_id"]  # no measurement: creation order
    voicebank.choose_design(ctx, "hero", new[1]["design_id"], "P")
    assert load_characters(ctx.paths.characters)["hero"].route == "P"


def test_stage_keeps_the_chosen_design_and_does_not_regenerate(proj):
    ctx = proj
    add_unvoiced(ctx)
    s = run_stage("voicebank", ctx)
    assert s["mode_b"] == ["hero"] and s["performers"] == 3 and s["designs"] == 6
    chars = load_characters(ctx.paths.characters)
    assert chars["hero"].route == "P" and chars["hero"].performer["chosen_by"] == "stage"
    assert chars["sakura"].route is None  # the CLI default (both routes compete) is not changed
    ds = voicebank.list_designs(ctx, "sakura")
    assert len(ds) == 2 and ds[0]["design_id"] == chars["sakura"].performer["design_id"]  # the stage picks the top-ranked
    assert [d["stars"] is not None for d in ds] == [True, True] and all(d["origin"] == "stage" for d in ds)
    # the user picks the other one; a forced rerun keeps it
    other = ds[1]["design_id"]
    voicebank.choose_design(ctx, "sakura", other, "B")
    ctx.force = True
    s2 = run_stage("voicebank", ctx)
    ctx.force = False
    assert load_characters(ctx.paths.characters)["sakura"].performer["design_id"] == other
    assert read_yaml(ctx.paths.vb_performer("sakura"))["design_id"] == other
    assert s2["performers"] == 3
    # not run again: no model loads
    events = len(ctx.models.events)
    run_stage("voicebank", ctx)
    assert not [e for e in ctx.models.events[events:] if e["event"] == "load" and e["model"] in ("mock_tts", "mock_vc")]
    # route B skips nothing for B characters; the user's choice of route A / P skips RVC training
    voicebank.choose_design(ctx, "haruto", voicebank.list_designs(ctx, "haruto")[0]["design_id"], "P")
    import shutil

    shutil.rmtree(ctx.paths.vb_rvc("haruto"))
    s3 = run_stage("voicebank", ctx)
    assert "route P" in s3["rvc_skipped"]["haruto"] and not ctx.paths.vb_rvc("haruto").exists()


def test_stage_offers_the_clone_design_when_asked(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path / "game", sets={"voicebank.performer.offer_clone": True})
    build(ctx, n_per=3)
    run_stage("voicebank", ctx)
    ds = voicebank.list_designs(ctx, "sakura")
    assert sorted(d["kind"] for d in ds) == ["clone", "voice_design", "voice_design"]
    chars = load_characters(ctx.paths.characters)
    assert chars["sakura"].performer["design_id"] != "clone"  # the performer is never the clone


# --- RVC ---------------------------------------------------------------------------


def test_train_rvc_for_one_character(proj):
    ctx = proj
    seen = []
    ctx.extra["progress_cb"] = lambda stage, done, total, item: seen.append((stage, done, total, item))
    voicebank.propose_designs(ctx, "sakura", n=2)
    voicebank.choose_design(ctx, "sakura", voicebank.list_designs(ctx, "sakura")[0]["design_id"], "B")
    seen.clear()
    res = voicebank.train_rvc(ctx, "sakura")
    assert res["state"] == "done" and res["reason_code"] == "none" and not res["cached"]
    assert res["meta"]["model"] == "voicebank/sakura/rvc/model.pth" and ctx.paths.abs(res["meta"]["model"]).exists()
    assert [s[1] for s in seen][0] == 0 and seen[-1][1] == seen[-1][2] == 3 and {s[0] for s in seen} == {"voicebank"} and {s[3] for s in seen} == {"sakura"}
    assert load_characters(ctx.paths.characters)["sakura"].rvc["model"] == res["meta"]["model"]
    assert voicebank.train_rvc(ctx, "sakura")["cached"] is True  # same clips, same config: no second training
    assert voicebank.train_rvc(ctx, "haruto")["state"] == "done"  # another character, independent
    add_unvoiced(ctx)
    assert voicebank.train_rvc(ctx, "hero")["reason_code"] == "no_voiced_lines"
    ctx.cfg.voicebank.rvc.min_minutes = 99.0
    ctx.force = True
    r = voicebank.train_rvc(ctx, "sakura")
    assert r["state"] == "skipped" and r["reason_code"] == "not_enough_audio" and "min clean audio" in r["reason"]
    ctx.cfg.voicebank.rvc.enabled = False
    assert voicebank.train_rvc(ctx, "sakura")["reason_code"] == "disabled"


def test_set_tuning_normalizes_and_keeps_the_performer(proj):
    ctx = proj
    voicebank.propose_designs(ctx, "sakura", n=1)
    voicebank.choose_design(ctx, "sakura", "vd01", "B")
    t = voicebank.set_tuning(ctx, "sakura", {"pitch_st": 50, "speed": 1.1, "timbre_preset": "bogus"})
    assert t["pitch_st"] == 6.0 and t["speed"] == 1.1 and t["timbre_preset"] == ""
    c = load_characters(ctx.paths.characters)["sakura"]
    assert c.performer["tuning"] == t and c.performer["design_id"] == "vd01"
    with pytest.raises(ValueError, match="unknown_character"):
        voicebank.set_tuning(ctx, "ghost", {})


def test_refs_follow_an_edited_source_text(proj):
    """refs.json carries the clip's text (the timbre reference text): a set_src_text edit must refresh it."""
    from vndub.review.app import write_override

    ctx = proj
    run_stage("voicebank", ctx)
    refs = read_json(ctx.paths.vb_refs_json("sakura"))
    assert refs
    lid = refs[0]["line_id"]
    assert refs[0]["text"] != "直した本文。"
    write_override(ctx.paths, lid, "set_src_text", {"text": "直した本文。"})
    run_stage("voicebank", ctx)
    refs2 = {r["line_id"]: r for r in read_json(ctx.paths.vb_refs_json("sakura"))}
    assert refs2[lid]["text"] == "直した本文。"

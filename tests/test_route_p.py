"""Route P (performer voice only) for lines without original audio, voice tuning in synth / score / master,
tuning cache invalidation, progress reporting, and where pack puts the unvoiced files."""

from __future__ import annotations

import json
import shutil
import statistics
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import requires_ffmpeg
from tests.genproj import build, speechlike
from tests.p3_helpers import add_unvoiced
from tests.test_pack import _manifest, _setup
from vndub.audio import features, loudness
from vndub.backends import registry
from vndub.config import ProjectYaml, save_project_yaml
from vndub.engines import xp3
from vndub.manifest.io import FinalIndexRecord, SelectionRecord, load_characters, read_json, read_jsonl, save_characters, write_json, write_jsonl
from vndub.review.app import write_override
from vndub.stages import master, pack, score, synth, voicebank
from vndub.stages.common import run_stage
from vndub.stages.score import apply_scores, composite, evaluate_gates, terms


@pytest.fixture
def pproj(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path / "game", sets={"synth.voice_unvoiced": True})
    data = build(ctx, n_per=3)
    ids = add_unvoiced(ctx)
    return ctx, data, ids


def _cands(ctx, line_id):
    d = ctx.paths.synth_dir(line_id)
    return sorted(p for p in d.glob("*.wav") if not p.name.endswith(".prevc.wav")) if d.exists() else []


def _side(wav: Path) -> dict:
    return read_json(wav.with_suffix(".json"))


# --- which lines are dubbed ------------------------------------------------------


def test_dub_lines_adds_speaker_lines_without_voice_but_never_narration(pproj):
    ctx, data, ids = pproj
    voiced = [l.line_id for l in ctx.voiced_lines()]
    got = ctx.dub_lines()
    assert [l.line_id for l in got if l.has_voice] == voiced
    assert [l.line_id for l in got if not l.has_voice] == ids["hero"]  # not narration, not the voice-sustain continuation, not the menu record
    assert all(l.audio_id is None and l.src_wav is None for l in got if not l.has_voice)
    orders = [l.order for l in got]
    assert orders == sorted(orders)
    ctx.only = {ids["hero"][1], voiced[0]}
    assert {l.line_id for l in ctx.dub_lines()} == {ids["hero"][1], voiced[0]}
    ctx.only = None
    ctx.limit = 2
    assert len(ctx.dub_lines()) == 2
    ctx.limit = None
    assert len(ctx.dub_lines(apply_only=False)) == len(voiced) + 3
    ctx.cfg.synth.voice_unvoiced = False  # the CLI default: exactly the voiced lines
    assert [l.line_id for l in ctx.dub_lines()] == voiced


def test_a_voice_moved_away_by_link_makes_the_line_an_unvoiced_one(pproj):
    ctx, data, ids = pproj
    moved = data["lines"][0]
    write_jsonl(ctx.paths.link, [{"line_id": moved.line_id, "audio_id": None, "status": "unmatched"}])
    assert moved.line_id not in {l.line_id for l in ctx.voiced_lines()}
    d = {l.line_id: l for l in ctx.dub_lines()}
    assert moved.line_id in d and d[moved.line_id].has_voice is False and d[moved.line_id].voice_file is None


def test_unvoiced_lines_are_not_dubbed_unless_asked(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path / "game")  # synth.voice_unvoiced: false
    build(ctx, n_per=2)
    ids = add_unvoiced(ctx)
    run_stage("voicebank", ctx)
    s = run_stage("synth", ctx)
    assert s["lines"] == 5 - 1 + 0 or s["lines"] == len(ctx.voiced_lines())
    assert not any(ctx.paths.synth_dir(l).exists() for l in ids["hero"])
    assert "mode_b" not in read_json(ctx.paths.stage_status)["voicebank"]["summary"]
    assert not ctx.paths.vb("hero").exists()


# --- route P: synth -> score -> master -> pack ---------------------------------------


@requires_ffmpeg
def test_route_p_end_to_end(pproj):
    ctx, data, ids = pproj
    run_stage("voicebank", ctx)
    chars = load_characters(ctx.paths.characters)
    assert chars["hero"].route == "P" and chars["hero"].rvc == {} and chars["sakura"].route is None

    # synth: performer TTS only, straight to the candidate path, style from the text passport (or neutral)
    s = run_stage("synth", ctx)
    assert s["failed"] == 0 and "skipped_routes" not in s
    for lid in ids["hero"]:
        cs = _cands(ctx, lid)
        assert cs and all("_P_" in p.name for p in cs) and not list(ctx.paths.synth_dir(lid).glob("*.prevc.wav"))
        side = _side(cs[0])
        assert side["route"] == "P" and side["backend"] == "mock_tts" and side["params"]["vc"] is None
        assert len(set(side["ref_paths"])) == 1 and "performer" in side["ref_paths"][0]
    angry = _side(_cands(ctx, ids["hero"][0])[0])
    assert angry["style"]["emotion"] == "angry" and angry["style"]["tempo"] == "fast" and "angry" in angry["params"]["instruct"]
    neutral = _side(_cands(ctx, ids["hero"][1])[0])
    assert neutral["style"]["emotion"] == "neutral" and neutral["style"]["intensity"] < 0.5
    assert neutral["params"]["target_duration"] is None  # there is no original duration to aim at
    for lid in (ids["narrator"], ids["sustain"], ids["menu"]):
        assert not ctx.paths.synth_dir(lid).exists()
    sak = next(l for l in ctx.voiced_lines() if l.speaker_id == "sakura")
    assert {p.name.split("_")[2] for p in _cands(ctx, sak.line_id) if not p.name.startswith("orig")} == {"A", "B"}  # voiced characters: unchanged

    # score: no expression metrics, the composite is renormalized over what exists, no gate depends on a reference
    run_stage("score", ctx)
    W = ctx.cfg.score.weights
    for lid in ids["hero"]:
        recs = read_jsonl(ctx.paths.scores(lid))
        assert recs
        for r in recs:
            m = r["metrics"]
            assert r["route"] == "P" and not ({"emo_sim", "adv_delta", "f0_corr", "energy_corr", "pause_match", "prosody", "dur_ratio"} & {k for k, v in m.items() if v is not None})
            assert m["utmos"] is not None and m["cer"] is not None and m["spk_sim"] is not None  # spk_sim vs the performer's sample
            assert any("performer" in n for n in r["notes"])
            t = {k: v for k, v in terms(m).items() if v is not None}
            expect = sum(W[k] * v for k, v in t.items()) / sum(W[k] for k in t)
            assert r["composite"] == pytest.approx(expect, abs=1e-3) and set(t) <= {"utmos", "spk_sim", "cer", "pq"}
            assert {"emo_sim", "prosody", "dur"} <= set(r["missing"])
            assert set(r["gates"]) <= {"utmos", "cer", "spk_sim"} and r["passed"]
    sel = {s_.line_id: s_ for s_ in read_jsonl(ctx.paths.selection, SelectionRecord)}
    assert {sel[l].status for l in ids["hero"]} == {"auto"} and all(sel[l].route == "P" for l in ids["hero"])

    # master: the line gets a file named by its dialogue id; no stretch; loudness of the voiced cast
    m = run_stage("master", ctx)
    assert m["encoded"] == len(ctx.dub_lines()) and "errors" not in m
    finals = {r.line_id: r for r in read_jsonl(ctx.paths.final_index, FinalIndexRecord)}
    for i, lid in enumerate(ids["hero"]):
        r = finals[lid]
        assert (r.model_extra or {}).get("unvoiced") is True and r.voice_file == f"unvoiced/start_{i:08x}.wav" and r.stretch == 1.0
        assert ctx.paths.abs(r.encoded).exists()
        game_median = statistics.median(c.stats.lufs_median for c in load_characters(ctx.paths.characters).values() if c.stats.lufs_median is not None)
        y, sr = features.load(ctx.paths.abs(r.final_wav))
        assert loudness.lufs(y, sr) == pytest.approx(game_median, abs=1.0)
    assert all(not (finals[l.line_id].model_extra or {}).get("unvoiced") for l in ctx.voiced_lines())

    # pack (Ren'Py): files by dialogue id + a generated rpy enabling config.auto_voice; the voiced layout is untouched
    p = run_stage("pack", ctx)
    out = ctx.paths.out_engine("renpy")
    assert p["unvoiced"] == 3 and p["unvoiced_install"] == "renpy auto_voice"
    for i in range(3):
        assert (out / "game" / "voice_dub" / f"start_{i:08x}.wav").exists()
    rpy = (out / "game" / "vndub_auto_voice.rpy").read_text(encoding="utf-8")
    assert 'config.auto_voice = "voice_dub/{id}.wav"' in rpy and rpy.startswith("# Generated by vndub")
    assert (out / "game" / "voice" / "sak_0001.wav").exists() and not list((out / "game" / "voice").glob("start_*"))
    man = _manifest(out)
    un = [f for f in man["files"] if f.get("unvoiced")]
    assert len(un) == 3 and all(f["route"] == "P" and f["renpy_auto_voice"] and f["packed_as"].startswith("game/voice_dub/") for f in un)
    assert len(man["files"]) == len(ctx.dub_lines())
    text = (out / "INSTALL.md").read_text(encoding="utf-8")
    assert "Lines without original voice" in text and "vndub_auto_voice.rpy" in text


@requires_ffmpeg
def test_pack_renpy_uses_the_games_own_auto_voice_pattern(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path / "game", sets={"synth.voice_unvoiced": True})
    build(ctx, n_per=2, codec="vorbis")
    ids = add_unvoiced(ctx, n=2)
    script = Path(ctx.project_dir) / "game" / "script.rpy"
    script.write_text('define config.auto_voice = "audio/vo/{id}.ogg"\nlabel start:\n    "x"\n', encoding="utf-8")
    for st in ("voicebank", "synth", "score", "master", "pack"):
        run_stage(st, ctx)
    out = ctx.paths.out_engine("renpy")
    assert (out / "game" / "audio" / "vo" / "start_00000000.ogg").exists()
    assert not (out / "game" / "vndub_auto_voice.rpy").exists()  # the game's pattern already finds them
    assert "audio/vo/{id}.ogg" in (out / "INSTALL.md").read_text(encoding="utf-8")
    assert all(f["packed_as"].startswith("game/audio/vo/") for f in _manifest(out)["files"] if f.get("unvoiced"))
    assert len(ids["hero"]) == 2


def test_pack_other_engines_put_unvoiced_files_apart_and_say_it_is_manual(make_ctx, tmp_path):
    for engine, voice in (("nscripter", "voice/mio001.wav"), ("kirikiri", "voice/nen_0001.ogg")):
        ctx, game = _setup(make_ctx, tmp_path / engine, engine, [voice])
        enc = ctx.paths.final_encoded / "unvoiced" / "s01_000900.wav"
        enc.parent.mkdir(parents=True, exist_ok=True)
        enc.write_bytes(b"RIFF" + b"\x01" * 60)
        finals = read_jsonl(ctx.paths.final_index, FinalIndexRecord)
        finals.append(FinalIndexRecord(line_id="s01_000900", voice_file="unvoiced/s01_000900.wav", final_wav="final/s01_000900.wav", encoded=ctx.paths.rel(enc), source="synth/s01_000900/mock_tts_P_0.wav", unvoiced=True))
        write_jsonl(ctx.paths.final_index, finals)
        s = run_stage("pack", ctx)
        out = ctx.paths.out_engine(engine)
        assert (out / "unvoiced" / "s01_000900.wav").read_bytes().startswith(b"RIFF")
        assert s["unvoiced"] == 1 and s["unvoiced_install"] == "manual"
        text = (out / "INSTALL.md").read_text(encoding="utf-8")
        assert "NOT install them automatically" in text and "manual" in text.lower()
        un = [f for f in _manifest(out)["files"] if f.get("unvoiced")]
        assert un == [{**un[0], "file": "unvoiced/s01_000900.wav", "unvoiced": True, "manual": True}]
        if engine == "kirikiri":  # nothing of it inside the patch archive
            with xp3.XP3Reader(out / "patch.xp3") as r:
                assert [e.name for e in r.entries] == ["nen_0001.ogg"]


def test_pack_with_only_unvoiced_lines(make_ctx, tmp_path):
    ctx, _ = _setup(make_ctx, tmp_path, "kirikiri", [])
    enc = ctx.paths.final_encoded / "unvoiced" / "a.ogg"
    enc.parent.mkdir(parents=True, exist_ok=True)
    enc.write_bytes(b"OggS" + b"\x00" * 30)
    write_jsonl(ctx.paths.final_index, [FinalIndexRecord(line_id="a", voice_file="unvoiced/a.ogg", final_wav="final/a.wav", encoded=ctx.paths.rel(enc), source="x", unvoiced=True)])
    s = run_stage("pack", ctx)
    out = ctx.paths.out_engine("kirikiri")
    assert s["files"] == 0 and (out / "unvoiced" / "a.ogg").exists() and "manual" in (out / "INSTALL.md").read_text(encoding="utf-8").lower()
    assert not (out / "patch.xp3").exists()


# --- score: reference metrics missing is not a failure -----------------------------------


def test_score_renormalizes_and_no_gate_needs_a_missing_reference_metric():
    from vndub.config import load_config

    cfg = load_config("configs/pipeline.yaml")
    W = cfg.score.weights
    m = {"utmos": 4.0, "cer": 0.02, "spk_sim": 0.6}  # what a line without original audio has
    comp, missing = composite(m, W)
    t = terms(m)
    assert comp == pytest.approx((W["utmos"] * t["utmos"] + W["spk_sim"] * t["spk_sim"] + W["cer"] * t["cer"]) / (W["utmos"] + W["spk_sim"] + W["cer"]), abs=1e-4)
    assert set(missing) == {"pq", "emo_sim", "dur", "prosody"}
    g = evaluate_gates(m, cfg.score.gates, 0.3, route="P")
    assert g == {"utmos": True, "cer": True, "spk_sim": True}  # no emo / duration / prosody gate exists to fail
    from vndub.manifest.io import CandidateScore

    c = apply_scores(CandidateScore(line_id="l", wav="a.wav", backend="x", route="P", metrics=m), cfg, 0.3, False)
    assert c.passed and c.composite == comp
    # a missing duration ratio never fails the duration gate
    assert "dur" not in evaluate_gates({"utmos": 4.0, "cer": 0.0}, cfg.score.gates, 0.3, route="P")


# --- tuning in synth ------------------------------------------------------------------------


TUNED = {"pitch_st": 3.0, "speed": 1.2, "expressiveness": 1.0, "energy": 1.0, "character_note": "ехидный", "timbre_preset": "younger"}


def test_tuning_reaches_the_backend_and_the_sidecar(pproj):
    ctx, data, ids = pproj
    run_stage("voicebank", ctx)
    synth.run(ctx)
    hl = next(l for l in ctx.voiced_lines() if l.speaker_id == "haruto")
    base = {p.name: _side(p) for p in _cands(ctx, hl.line_id) if not p.name.startswith("orig")}
    assert all("tuning" not in s["params"] for s in base.values())
    voicebank.set_tuning(ctx, "haruto", TUNED)
    synth.run(ctx)
    new = {p.name: _side(p) for p in _cands(ctx, hl.line_id) if not p.name.startswith("orig")}
    assert set(new) == set(base)  # regenerated in place, no extra rounds
    a = next(s for n, s in new.items() if "_A_" in n)
    b = next(s for n, s in new.items() if "_B_" in n)
    assert a["params"]["tuning"]["values"]["character_note"] == "ехидный"
    for s in (a, b):
        assert "ехидный" in s["params"]["instruct"] and "a younger, brighter voice" in s["params"]["instruct"] and "loudly" in s["params"]["instruct"]
        assert s["params"]["knobs"]["exaggeration"] > next(iter(base.values()))["params"]["knobs"]["exaggeration"]
    # mock_tts can aim at a duration, so the speed is native: target_duration / speed, nothing left for master
    src = data["audio"][[l.line_id for l in data["lines"]].index(hl.line_id)].duration
    assert a["params"]["target_duration"] == pytest.approx(src / 1.2, abs=0.01)
    plan_a, plan_b = a["params"]["tuning"]["plan"], b["params"]["tuning"]["plan"]
    assert plan_a["speed_native"] and plan_a["post_speed"] == 1.0 and plan_a["post_pitch_st"] == pytest.approx(3.0)  # route A: no VC -> master pitch pass
    assert plan_b["vc_f0_transpose"] == 4 and plan_b["post_pitch_st"] == pytest.approx(0.5)  # route B: +3 and the age chip's +1.5 -> RVC 4 (round 4.5), remainder -> master
    base_b = next(s for n, s in base.items() if "_B_" in n)
    assert b["params"]["vc"]["f0_transpose"] == base_b["params"]["vc"]["f0_transpose"] + 4


def test_changing_the_tuning_regenerates_only_that_character_and_a_line_override_one_line(pproj):
    ctx, data, ids = pproj
    run_stage("voicebank", ctx)
    s0 = synth.run(ctx)
    total = s0["candidates"]
    assert s0["generated"] == total and s0["cached"] == 0
    assert synth.run(ctx)["generated"] == 0  # nothing changed
    per_char = {}
    for l in ctx.dub_lines():
        n = len([p for p in _cands(ctx, l.line_id) if not p.name.startswith("orig")])
        per_char[l.speaker_id] = per_char.get(l.speaker_id, 0) + n

    voicebank.set_tuning(ctx, "haruto", {"pitch_st": 2.0})
    s1 = synth.run(ctx)
    assert s1["generated"] == per_char["haruto"] and s1["cached"] == total - per_char["haruto"]
    assert synth.run(ctx)["generated"] == 0

    voicebank.set_tuning(ctx, "hero", {"energy": 1.0})  # an unvoiced character too
    assert synth.run(ctx)["generated"] == per_char["hero"]

    # a per-line override replaces the character's tuning for that line only
    sak = [l for l in ctx.voiced_lines() if l.speaker_id == "sakura"]
    n_line = len([p for p in _cands(ctx, sak[1].line_id) if not p.name.startswith("orig")])
    write_override(ctx.paths, sak[1].line_id, "tuning", {"speed": 1.1, "character_note": "тихо"})
    s2 = synth.run(ctx)
    assert s2["generated"] == n_line
    side = _side(next(p for p in _cands(ctx, sak[1].line_id) if "_A_" in p.name))
    assert side["params"]["tuning"]["values"]["character_note"] == "тихо" and side["round"] == 0  # not a regen round
    assert synth.run(ctx)["generated"] == 0
    # the override REPLACES the character's tuning (haruto has pitch 2; the line says speed only)
    hl = next(l for l in ctx.voiced_lines() if l.speaker_id == "haruto")
    write_override(ctx.paths, hl.line_id, "tuning", {"speed": 1.15})
    synth.run(ctx)
    v = _side(next(p for p in _cands(ctx, hl.line_id) if "_A_" in p.name))["params"]["tuning"]["values"]
    assert v["pitch_st"] == 0.0 and v["speed"] == 1.15
    # a later override wins; writing the defaults gives the line its untuned candidates back
    write_override(ctx.paths, hl.line_id, "tuning", {})
    synth.run(ctx)
    assert "tuning" not in _side(next(p for p in _cands(ctx, hl.line_id) if "_A_" in p.name))["params"]


def test_candidates_of_an_older_tuning_do_not_linger(pproj):
    ctx, data, ids = pproj
    run_stage("voicebank", ctx)
    synth.run(ctx)
    lid = ids["hero"][1]
    old = _cands(ctx, lid)
    extra = ctx.paths.synth_dir(lid) / "mock_tts_P_100.wav"  # a candidate of an earlier regen round
    shutil.copy2(old[0], extra)
    shutil.copy2(old[0].with_suffix(".json"), extra.with_suffix(".json"))
    synth.run(ctx)
    assert extra.exists()  # same (default) tuning: kept
    voicebank.set_tuning(ctx, "hero", {"expressiveness": 0.0})
    synth.run(ctx)
    assert not extra.exists() and not extra.with_suffix(".json").exists()
    assert {p.name for p in _cands(ctx, lid)} == {p.name for p in old}


# --- tuning in master ------------------------------------------------------------------------


@requires_ffmpeg
def test_master_applies_pitch_speed_and_loudness_offset(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path / "game", sets={"master.stretch.enabled": False})
    data = build(ctx, n_per=2, interjection=False)
    l0, l1 = data["lines"][0], data["lines"][1]
    sel = []
    for line, plan in ((l0, None), (l1, {"post_pitch_st": 4.0, "post_speed": 1.25, "speed": 1.25, "speed_native": False, "lufs_offset_db": 3.0})):
        cand = ctx.paths.synth_dir(line.line_id) / "mock_tts_A_0.wav"
        speechlike(cand, 2.0, 200.0, sr=24000, seed=5, amp=0.2)
        side = {"line_id": line.line_id, "backend": "mock_tts", "route": "A", "k": 0, "params": {"tuning": {"values": {"speed": 1.25}, "plan": plan}} if plan else {}}
        write_json(cand.with_suffix(".json"), side)
        sel.append(SelectionRecord(line_id=line.line_id, winner=ctx.paths.rel(cand), backend="mock_tts", route="A", status="auto"))
    write_jsonl(ctx.paths.selection, sel)
    chars = load_characters(ctx.paths.characters)
    assert l0.speaker_id == l1.speaker_id
    master.run(ctx)
    fin = {r.line_id: r for r in read_jsonl(ctx.paths.final_index, FinalIndexRecord)}
    y0, sr0 = features.load(ctx.paths.abs(fin[l0.line_id].final_wav))
    y1, sr1 = features.load(ctx.paths.abs(fin[l1.line_id].final_wav))
    f0 = lambda y, sr: float(np.median((lambda f: f[f > 0])(features.f0_pyin(y, sr))))  # noqa: E731
    assert (len(y1) / sr1) == pytest.approx((len(y0) / sr0) / 1.25, rel=0.05)  # speed: duration / 1.25
    assert 12 * np.log2(f0(y1, sr1) / f0(y0, sr0)) == pytest.approx(4.0, abs=0.6)  # pitch: +4 semitones
    assert loudness.lufs(y1, sr1) - loudness.lufs(y0, sr0) == pytest.approx(3.0, abs=0.8)  # energy -> +3 dB
    assert read_json(ctx.paths.final_wav(l1.line_id).with_suffix(".json"))["tuning"] == {"pitch_st": 4.0, "speed": 1.25}
    assert "tuning" not in read_json(ctx.paths.final_wav(l0.line_id).with_suffix(".json"))
    assert fin[l0.line_id].stretch == 1.0 and chars  # no source-duration stretch involved
    # cached rerun; a changed plan re-masters only that line
    s2 = master.run(ctx)
    assert s2["cached"] == 2
    side = read_json((ctx.paths.synth_dir(l1.line_id) / "mock_tts_A_0.json"))
    side["params"]["tuning"]["plan"]["post_pitch_st"] = 2.0
    write_json(ctx.paths.synth_dir(l1.line_id) / "mock_tts_A_0.json", side)
    assert master.run(ctx)["cached"] == 1


@requires_ffmpeg
def test_native_speed_moves_the_expected_duration_in_master_and_score(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path / "game")
    data = build(ctx, n_per=2, interjection=False)
    line = data["lines"][0]
    src = data["audio"][0].duration
    cand = ctx.paths.synth_dir(line.line_id) / "mock_tts_A_0.wav"
    speechlike(cand, src / 1.2, 200.0, sr=24000, seed=2, amp=0.2)  # the backend already spoke 1.2x faster
    plan = {"post_pitch_st": 0.0, "post_speed": 1.0, "speed": 1.2, "speed_native": True, "lufs_offset_db": 0.0}
    write_json(cand.with_suffix(".json"), {"line_id": line.line_id, "backend": "mock_tts", "route": "A", "k": 0, "params": {"tuning": {"values": {"speed": 1.2}, "plan": plan}}, "text_plain": "x"})
    side = score.SynthSidecar.model_validate(read_json(cand.with_suffix(".json")))
    assert score.native_speed(side) == 1.2 and score.native_speed(None) == 1.0
    write_jsonl(ctx.paths.selection, [SelectionRecord(line_id=line.line_id, winner=ctx.paths.rel(cand), backend="mock_tts", route="A", status="auto")])
    master.run(ctx)
    r = read_jsonl(ctx.paths.final_index, FinalIndexRecord)[0]
    assert r.stretch == 1.0  # compared with src / 1.2: inside the window, no stretch back to the original length


# --- progress ----------------------------------------------------------------------------------


@requires_ffmpeg
def test_generation_stages_report_progress(pproj):
    ctx, data, ids = pproj
    seen: list[tuple] = []
    ctx.extra["progress_cb"] = lambda stage, done, total, item: seen.append((stage, done, total, item))
    for st in ("voicebank", "synth", "score", "master", "pack"):
        run_stage(st, ctx)
    for st in ("voicebank", "synth", "score", "master", "pack"):
        rows = [r for r in seen if r[0] == st]
        assert rows, st
        assert all(0 <= r[1] <= r[2] for r in rows) and [r[1] for r in rows] == sorted(r[1] for r in rows), st  # monotonic, never past the total
        assert rows[-1][1] == rows[-1][2], st  # ends complete
        assert any(r[3] for r in rows), st  # names the item (line id / character)
    assert {r[3] for r in seen if r[0] == "synth" and r[3]} >= {ids["hero"][0], ids["hero"][1]}


def test_cached_reruns_still_report_completion(pproj):
    ctx, data, ids = pproj
    run_stage("voicebank", ctx)
    synth.run(ctx)
    seen: list[tuple] = []
    ctx.extra["progress_cb"] = lambda stage, done, total, item: seen.append((stage, done, total, item))
    ctx.stage = "synth"
    s = synth.run(ctx)  # every candidate cached: nothing to generate, the app still gets n / n
    assert s["generated"] == 0 and seen and seen[-1][1] == seen[-1][2] == s["candidates"]

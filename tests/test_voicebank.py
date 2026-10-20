"""voicebank: ref selection, performer choice (mock design), RVC meta + f0_transpose, speaker calibration."""

from __future__ import annotations

import math

import pytest

from tests.genproj import build
from vndub.manifest.io import Character, CharacterStats, LineRecord, RefClip, load_characters, read_json, read_yaml
from vndub.stages import voicebank
from vndub.stages.voicebank import performer_description, pitch_word, select_refs


def _line(i: int, text: str = "テストです。") -> LineRecord:
    return LineRecord(line_id=f"l{i}", scene="s", order=i, speaker_id="a", text_src=text, has_voice=True, src_wav=f"audio/src/{i}.wav")


def _pp(snr: float, dur: float, emo: str, bgm: bool = False) -> dict:
    return {"duration_s": dur, "snr_db": snr, "bgm": bgm, "style": {"emotion": emo}, "emotion": {"adv": {"arousal": 0.5, "valence": 0.5}}}


def test_select_refs_filters_and_covers_emotions():
    lines = [_line(i) for i in range(10)]
    pps = {
        "l0": _pp(35, 4, "happy"), "l1": _pp(34, 5, "happy"), "l2": _pp(33, 6, "happy"),
        "l3": _pp(30, 4, "sad"), "l4": _pp(29, 4, "sad"),
        "l5": _pp(40, 4, "neutral"),
        "l6": _pp(10, 4, "angry"),            # too noisy
        "l7": _pp(40, 1.0, "angry"),          # too short
        "l8": _pp(40, 4, "angry", bgm=True),  # BGM
        "l9": _pp(28, 9, "surprised"),
    }
    chosen, relaxed = select_refs(lines, pps, min_snr_db=25, min_sec=3, max_sec=10, per_emotion=2, max_total=10, min_refs=5)
    ids = {c["line"].line_id for c in chosen}
    assert not relaxed
    assert ids.isdisjoint({"l6", "l7", "l8"})
    emos = [c["emotion"] for c in chosen]
    assert emos.count("happy") == 2 and set(emos) == {"happy", "sad", "neutral", "surprised"}
    assert all(c["usage"] in ("timbre", "emotion", "both") for c in chosen)
    # strict filter leaves too few -> relaxed
    chosen, relaxed = select_refs(lines[:3], {k: pps[k] for k in ("l0", "l1", "l2")}, min_snr_db=50, min_sec=3, max_sec=10, per_emotion=2, max_total=10, min_refs=2)
    assert relaxed and len(chosen) >= 2


def test_pitch_and_description():
    assert pitch_word(300, "f") == "high pitch" and pitch_word(180, "f") == "low pitch" and pitch_word(130, "m") == "medium pitch"
    c = Character(gender="f", age_hint="teen", stats=CharacterStats(f0_median_hz=290, emotion_hist={"happy": 0.7, "neutral": 0.3}))
    d = performer_description(c, "ru")
    assert "young female" in d and "high pitch" in d and "native Russian" in d and "cheerful" in d


@pytest.fixture
def vb(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path / "game")
    build(ctx, n_per=3)
    summary = voicebank.run(ctx)
    return ctx, summary


def test_voicebank_outputs(vb):
    ctx, s = vb
    assert s["characters"] == 2 and s["performers"] == 2 and s["rvc"] == 2
    chars = load_characters(ctx.paths.characters)
    for sid in ("sakura", "haruto"):
        refs = [RefClip.model_validate(r) for r in read_json(ctx.paths.vb_refs_json(sid))]
        assert 1 <= len(refs) <= ctx.cfg.voicebank.refs.max_total
        assert all(ctx.paths.abs(r.path).exists() for r in refs)
        assert chars[sid].refs == [r.path for r in refs]
        perf = read_yaml(ctx.paths.vb_performer(sid))
        assert perf["refs"] and all(ctx.paths.abs(r["path"]).exists() for r in perf["refs"])
        assert perf["backend"] == "mock_tts"  # sapi_tts -> mock in tests
        assert perf["f0_median_hz"] and perf["f0_delta_st"] is not None
        meta = read_json(ctx.paths.vb_rvc(sid) / "meta.json")
        expect = round(12 * math.log2(chars[sid].stats.f0_median_hz / perf["f0_median_hz"]))
        assert meta["f0_transpose"] == expect
        assert chars[sid].rvc["model"] == meta["model"]
    cal = read_json(ctx.paths.spk_calibration)
    assert cal["n_same"] > 0 and cal["n_diff"] > 0 and "threshold" in cal and 0 <= cal["eer"] <= 1


def test_performer_emotion_variants(vb):
    ctx, _ = vb
    perf = read_yaml(ctx.paths.vb_performer("sakura"))
    emos = {r["emotion"] for r in perf["refs"]}
    assert "neutral" in emos  # the chosen design's test sentences


def test_voicebank_cached_rerun(vb):
    ctx, _ = vb
    events = len(ctx.models.events)
    s2 = voicebank.run(ctx)
    assert s2["performers"] == 2 and s2["rvc"] == 2
    loads = [e for e in ctx.models.events[events:] if e["event"] == "load" and e["model"] in ("mock_tts", "mock_vc")]
    assert not loads  # designs and RVC were cached

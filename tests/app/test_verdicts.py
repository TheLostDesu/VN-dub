"""Metric -> verdict mapping and the "why" explanation (``vndub_app.verdicts``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vndub_app import verdicts as V

REPO = Path(__file__).resolve().parents[2]

TH = V.Thresholds(utmos_min=3.0, cer_max=0.10, cer_max_interjection=0.5, dur_lo=0.8, dur_hi=1.25, spk_min=0.25)


def key(v: dict | None) -> tuple[str, str] | None:
    return None if v is None else (v["key"], v["level"])


def test_voice_levels_follow_the_speaker_gate() -> None:
    assert TH.spk_good == pytest.approx(0.25 + 0.1875)
    assert key(V.voice_verdict(0.80, TH)) == ("similar", "good")
    assert key(V.voice_verdict(0.30, TH)) == ("close", "ok")
    assert key(V.voice_verdict(0.25, TH)) == ("close", "ok")  # exactly the gate passes
    assert key(V.voice_verdict(0.2499, TH)) == ("different", "bad")
    assert V.voice_verdict(None, TH) is None
    v = V.voice_verdict(0.5, TH)
    assert v["value"] == 0.5 and v["tooltip_params"]["threshold"] == 0.25 and v["metric"] == "voice" and v["tooltip_params"]["reference"] == "original"
    assert V.voice_verdict(0.5, TH, "performer")["tooltip_params"]["reference"] == "performer"


def test_emotion_uses_the_composite_term() -> None:
    # 0.75 * emo + 0.25 * (1 - adv): the same term score.py feeds the composite
    assert V.emotion_term(0.9, 0.2) == pytest.approx(0.75 * 0.9 + 0.25 * 0.8)
    assert V.emotion_term(0.9, None) == pytest.approx(0.9)
    assert V.emotion_term(None, 0.2) == pytest.approx(0.8)
    assert V.emotion_term(None, None) is None
    assert key(V.emotion_verdict(0.9, 0.1)) == ("close", "good")
    assert key(V.emotion_verdict(0.7, 0.4)) == ("partly", "ok")  # 0.625
    assert key(V.emotion_verdict(0.3, 0.6)) == ("far", "bad")
    assert V.emotion_verdict(None, None) is None
    # numbers survive for the tooltip
    p = V.emotion_verdict(0.9, 0.1)["tooltip_params"]
    assert p["emo_sim"] == 0.9 and p["adv_delta"] == 0.1


def test_clarity_levels_and_interjection_gate() -> None:
    assert key(V.clarity_verdict(0.0, TH)) == ("good", "good")
    assert key(V.clarity_verdict(0.05, TH)) == ("good", "good")
    assert key(V.clarity_verdict(0.08, TH)) == ("ok", "ok")
    assert key(V.clarity_verdict(0.11, TH)) == ("poor", "bad")
    # interjections have the soft gate (0.5)
    assert key(V.clarity_verdict(0.3, TH, interjection=True)) == ("ok", "ok")
    assert key(V.clarity_verdict(0.6, TH, interjection=True)) == ("poor", "bad")
    assert V.clarity_verdict(None, TH) is None


def test_length_levels() -> None:
    assert key(V.length_verdict(1.0, TH)) == ("ok", "good")
    assert key(V.length_verdict(1.2, TH)) == ("ok", "ok")  # inside the gate, outside the 0.9-1.1 window
    assert key(V.length_verdict(1.3, TH)) == ("longish", "ok")
    assert key(V.length_verdict(1.6, TH)) == ("too_long", "bad")
    assert key(V.length_verdict(0.75, TH)) == ("shortish", "ok")
    assert key(V.length_verdict(0.5, TH)) == ("too_short", "bad")
    assert V.length_verdict(0, TH) is None and V.length_verdict(None, TH) is None
    assert V.length_verdict(1.3, TH)["tooltip_params"] == {"ratio": 1.3, "min": 0.8, "max": 1.25}


def test_quality_levels() -> None:
    assert key(V.quality_verdict(4.2, TH)) == ("good", "good")
    assert key(V.quality_verdict(3.2, TH)) == ("ok", "ok")
    assert key(V.quality_verdict(2.9, TH)) == ("poor", "bad")
    assert V.quality_verdict(None, TH) is None


def test_verdicts_for_order_and_missing_metrics() -> None:
    m = {"spk_sim": 0.6, "emo_sim": 0.8, "adv_delta": 0.2, "cer": 0.01, "dur_ratio": 1.0, "utmos": 4.0}
    vs = V.verdicts_for(m, TH)
    assert [v["metric"] for v in vs] == ["voice", "emotion", "clarity", "length", "quality"]
    assert all(v["level"] == "good" for v in vs)
    assert V.worst_level(vs) == "good"
    # mode B (no original audio): no voice / emotion reference
    vs_b = V.verdicts_for({"cer": 0.2, "dur_ratio": 1.0, "utmos": 3.5}, TH)
    assert [v["metric"] for v in vs_b] == ["clarity", "length", "quality"]
    assert V.worst_level(vs_b) == "bad"
    # the original clip as a candidate has no CER / length verdicts
    assert [v["metric"] for v in V.verdicts_for({**m, "cer": None}, TH, route="orig")] == ["voice", "emotion", "quality"]
    assert V.verdicts_for({}, TH) == []


def test_thresholds_from_config_and_calibration(repo: Path) -> None:
    from vndub.config import load_config

    cfg = load_config(repo / "configs" / "pipeline.yaml")
    th = V.Thresholds.from_config(cfg)
    assert (th.utmos_min, th.cer_max, th.dur_lo, th.dur_hi) == (cfg.score.gates.utmos_min, cfg.score.gates.cer_max, *cfg.score.gates.dur_ratio)
    assert th.spk_min == cfg.score.gates.spk_sim_default  # "calibrated" without a calibration file
    assert V.Thresholds.from_config(cfg, {"threshold": 0.42}).spk_min == 0.42
    assert th.cer_max_interjection == cfg.score.interjection_cer_max
    assert th.target_window == tuple(cfg.master.stretch.target_window)
    fixed = load_config(repo / "configs" / "pipeline.yaml", {"score.gates.spk_sim_min": 0.5, "score.gates.cer_max": 0.2})
    th2 = V.Thresholds.from_config(fixed, {"threshold": 0.42})
    assert th2.spk_min == 0.5 and th2.cer_max == 0.2


@pytest.mark.parametrize("utmos", [2.0, 2.99, 3.0, 3.01, 4.5])
@pytest.mark.parametrize("cer", [0.0, 0.1, 0.1001, 0.4])
@pytest.mark.parametrize("dur", [0.5, 0.79, 0.8, 1.0, 1.25, 1.26, 1.9])
@pytest.mark.parametrize("spk", [0.1, 0.249, 0.25, 0.6])
def test_verdicts_agree_with_the_score_gates(repo: Path, utmos: float, cer: float, dur: float, spk: float) -> None:
    """A failed gate is never "good"; a "bad" verdict is always a failed gate (same thresholds as score.py)."""
    pytest.importorskip("vndub.stages.score")
    from vndub.config import load_config
    from vndub.stages.score import evaluate_gates

    cfg = load_config(repo / "configs" / "pipeline.yaml")
    th = V.Thresholds.from_config(cfg)
    m = {"utmos": utmos, "cer": cer, "dur_ratio": dur, "spk_sim": spk}
    gates = evaluate_gates(m, cfg.score.gates, th.spk_min)
    by_metric = {v["metric"]: v for v in V.verdicts_for(m, th)}
    pairs = {"quality": "utmos", "clarity": "cer", "length": "dur", "voice": "spk_sim"}
    for metric, gate in pairs.items():
        lvl = by_metric[metric]["level"]
        if not gates[gate]:
            assert lvl != "good", (metric, m)
        if lvl == "bad":
            assert not gates[gate], (metric, m)


def test_why_items() -> None:
    gates = {"cer": False, "utmos": False, "spk_sim": False, "dur": False}
    m = {"cer": 0.23, "utmos": 2.5, "spk_sim": 0.1, "dur_ratio": 1.5}
    items = V.why_items(metrics=m, gates=gates, th=TH, selection_status="review")
    assert [i["key"] for i in items] == ["cer", "utmos", "spk_sim", "dur_ratio"]
    assert items[0] == {"gate": "cer", "key": "cer", "params": {"value": 0.23, "max": 0.1}}
    assert items[3]["params"] == {"value": 1.5, "min": 0.8, "max": 1.25}
    # lines without original audio: the speaker gate is about the performer, with its own sentence
    assert V.why_items(metrics={"spk_sim": 0.1}, gates={"spk_sim": False}, th=TH, voice_reference="performer")[0]["key"] == "spk_sim_performer"
    # interjections: the soft CER limit is reported
    assert V.why_items(metrics={"cer": 0.7}, gates={"cer": False}, th=TH, interjection=True)[0]["params"]["max"] == 0.5
    # passing gates: nothing to explain
    assert V.why_items(metrics={"cer": 0.0}, gates={"cer": True, "dur": True}, th=TH) == []
    # other reasons
    assert [i["key"] for i in V.why_items(metrics=None, gates=None, th=TH, has_candidates=False, selection_reason="no candidates")] == ["no_candidates"]
    keys = [i["key"] for i in V.why_items(metrics={}, gates={}, th=TH, selection_status="override", line_flags=["text_from_asr"], translation_failed=True)]
    assert keys == ["text_from_asr", "translation_failed", "override"]

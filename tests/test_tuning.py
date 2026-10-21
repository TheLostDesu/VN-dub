"""Voice tuning: apply_tuning / tuning_knobs / tuning_plan for every backend capability."""

from __future__ import annotations

import pytest

from vndub.backends import registry
from vndub.manifest.io import Character
from vndub.stages.voicebank import design_prompt
from vndub.text.style_prompt import (
    TUNING_DEFAULTS,
    StyleSpec,
    apply_tuning,
    design_words,
    is_default_tuning,
    normalize_tuning,
    render,
    to_cosy_instruct,
    to_fish_tags,
    to_higgs_tags,
    to_instruction,
    to_voxcpm_prefix,
    tuning_key,
    tuning_knobs,
    tuning_plan,
)


def caps(name: str) -> dict:
    return dict(registry.get(name).caps)


HAPPY = StyleSpec(emotion="happy", intensity=0.45, tempo="normal", loudness="normal", events=["laughter"], arc="rising_end")
CALM = StyleSpec(emotion="neutral", intensity=0.3)


# --- the dict ------------------------------------------------------------------


def test_defaults_clamp_and_unknown_values():
    assert normalize_tuning(None) == TUNING_DEFAULTS == {"pitch_st": 0.0, "speed": 1.0, "expressiveness": 0.5, "energy": 0.0, "timbre_preset": "", "character_note": ""}
    t = normalize_tuning({"pitch_st": 99, "speed": 0.1, "expressiveness": 3, "energy": -5, "timbre_preset": "SHARPER", "character_note": "  ехидная  ", "bogus": 1})
    assert (t["pitch_st"], t["speed"], t["expressiveness"], t["energy"]) == (6.0, 0.8, 1.0, -1.0)
    assert t["timbre_preset"] == "sharper" and t["character_note"] == "ехидная" and "bogus" not in t
    t = normalize_tuning({"pitch_st": "x", "timbre_preset": "robot", "speed": float("nan")})
    assert t == TUNING_DEFAULTS
    assert is_default_tuning({}) and not is_default_tuning({"speed": 1.1})
    assert tuning_key({"expressiveness": 0.5}) is None and tuning_key({"speed": 1.1})["speed"] == 1.1


def test_default_tuning_changes_nothing_and_input_is_not_mutated():
    for t in (None, {}, TUNING_DEFAULTS):
        assert apply_tuning(HAPPY, t) == HAPPY
        for name in ("qwen3_tts", "cosyvoice3", "zonos2", "chatterbox_ml", "mock_tts"):
            assert tuning_knobs(t, caps(name)) == {}
    before = HAPPY.to_dict()
    out = apply_tuning(HAPPY, {"expressiveness": 1.0, "energy": 1.0, "character_note": "x"})
    assert HAPPY.to_dict() == before and out is not HAPPY and out.events is not HAPPY.events


# --- expressiveness / energy (backend independent) ------------------------------


def test_expressiveness_scales_intensity_and_calm_collapses():
    s = StyleSpec(emotion="angry", intensity=0.4, events=["breath"], arc="falling_end", tempo="fast", loudness="loud")
    assert apply_tuning(s, {"expressiveness": 0.5}).intensity == 0.4  # the passport as is
    assert apply_tuning(s, {"expressiveness": 1.0}).intensity == pytest.approx(0.8)
    assert apply_tuning(s, {"expressiveness": 0.25}).intensity == pytest.approx(0.2)
    assert apply_tuning(StyleSpec(emotion="angry", intensity=0.7), {"expressiveness": 1.0}).intensity == 1.0
    calm = apply_tuning(s, {"expressiveness": 0.0})
    assert (calm.emotion, calm.intensity, calm.events, calm.arc, calm.tempo, calm.loudness) == ("neutral", 0.0, [], "flat", "normal", "normal")
    assert calm.emotion_probs == {"neutral": 1.0}
    # amplifying a neutral line stays moderate
    assert apply_tuning(StyleSpec(emotion="neutral", intensity=0.4), {"expressiveness": 1.0}).intensity == pytest.approx(0.45)


def test_energy_shifts_loudness_class():
    s = StyleSpec()
    assert apply_tuning(s, {"energy": 0.1}).loudness == "normal"
    assert apply_tuning(s, {"energy": 0.5}).loudness == "loud" and apply_tuning(s, {"energy": -0.5}).loudness == "soft"
    assert apply_tuning(StyleSpec(loudness="soft"), {"energy": 1.0}).loudness == "loud"  # two steps
    assert apply_tuning(StyleSpec(loudness="loud"), {"energy": 0.5}).loudness == "loud"  # clamped


# --- per backend capability -----------------------------------------------------


def test_instruct_tts_gets_the_words_appended():
    t = {"expressiveness": 1.0, "energy": 0.8, "timbre_preset": "younger", "character_note": "ехидная"}
    for name in ("qwen3_tts", "mock_tts"):  # instruct: qwen / natural
        out = render(apply_tuning(HAPPY, t), caps(name))
        text = out["instruct"]
        assert "ехидная" in text and "a younger, brighter voice" in text and "loudly" in text and "very intense" in text
        assert out["style_prompt"] == text
    # untuned stays as it was
    assert "ехидная" not in render(HAPPY, caps("qwen3_tts"))["instruct"]


def test_cosyvoice_instruct_and_native_speed():
    t = {"character_note": "tired", "timbre_preset": "older", "speed": 1.15}
    spec = apply_tuning(HAPPY, t)
    out = render(spec, caps("cosyvoice3"))
    assert out["instruct"].endswith("<|endofprompt|>") and "tired" in out["instruct"] and "an older, deeper voice" in out["instruct"]
    assert tuning_knobs(t, caps("cosyvoice3")) == {"speed": 1.15}
    plan = tuning_plan(t, caps("cosyvoice3"), "A")
    assert plan["speed_native"] and plan["post_speed"] == 1.0 and plan["speed"] == 1.15


def test_voxcpm_prefix_carries_note_and_intensity():
    spec = apply_tuning(HAPPY, {"expressiveness": 1.0, "character_note": "ехидная"})
    prefix = render(spec, caps("voxcpm2"))["text_prefix"]
    assert prefix.startswith("(") and "ехидная" in prefix and "intense" in prefix
    assert "subdued" in to_voxcpm_prefix(apply_tuning(StyleSpec(emotion="sad", intensity=0.4), {"expressiveness": 0.2}))
    assert "ехидная" in to_voxcpm_prefix(apply_tuning(CALM, {"character_note": "ехидная"}))


def test_fish_and_higgs_tags_follow_energy_and_calm():
    loud = apply_tuning(CALM, {"energy": 1.0})
    assert "[loud]" in render(loud, caps("fish_s2"))["text_prefix"]
    soft = apply_tuning(CALM, {"energy": -1.0})
    assert "[soft]" in to_fish_tags(soft)
    amplified = apply_tuning(HAPPY, {"expressiveness": 1.0})
    assert to_fish_tags(amplified).startswith("[very happy]")
    whisper = StyleSpec(emotion="sad", intensity=0.5, events=["whisper"])
    assert "<|style:whisper|>" in to_higgs_tags(whisper)
    assert "<|style:whisper|>" not in render(apply_tuning(whisper, {"expressiveness": 0.0}), caps("higgs_tts3"))["text_prefix"]
    # words cannot reach tag-only backends: the note is not in their prefix
    assert "ехидная" not in render(apply_tuning(HAPPY, {"character_note": "ехидная"}), caps("fish_s2"))["text_prefix"]


def test_chatterbox_exaggeration_scaled_by_expressiveness():
    opts = {"exaggeration_range": [0.3, 1.2]}
    base = render(HAPPY, caps("chatterbox_ml"), opts)["knobs"]["exaggeration"]
    more = render(apply_tuning(HAPPY, {"expressiveness": 1.0}), caps("chatterbox_ml"), opts)["knobs"]["exaggeration"]
    less = render(apply_tuning(HAPPY, {"expressiveness": 0.25}), caps("chatterbox_ml"), opts)["knobs"]["exaggeration"]
    flat = render(apply_tuning(HAPPY, {"expressiveness": 0.0}), caps("chatterbox_ml"), opts)["knobs"]["exaggeration"]
    assert flat == 0.3 < less < base < more
    assert render(apply_tuning(HAPPY, {"expressiveness": 0.5}), caps("chatterbox_ml"), opts)["knobs"]["exaggeration"] == base
    assert tuning_knobs({"expressiveness": 1.0, "speed": 1.2}, caps("chatterbox_ml")) == {}  # no native speed: master pass


def test_zonos_speaking_rate_and_strength():
    spec = StyleSpec(emotion="happy", intensity=0.4, tempo="fast", emotion_probs={"happy": 1.0})
    base = render(spec, caps("zonos2"))["knobs"]
    assert base["speaking_rate"] == 17.0
    t = {"speed": 1.2, "expressiveness": 1.0}
    assert tuning_knobs(t, caps("zonos2"), base) == {"speaking_rate": 20.4}
    assert render(apply_tuning(spec, t), caps("zonos2"))["knobs"]["emotion_strength"] > base["emotion_strength"]


def test_voice_design_prompt_gets_note_and_timbre_not_the_rest():
    assert design_words({"character_note": "ехидная", "timbre_preset": "softer"}) == "a softer, gentler voice; ехидная"
    assert design_words({"pitch_st": 3, "speed": 1.1, "energy": 0.5, "expressiveness": 1.0}) == ""
    c = Character(names={"src": "a"}, gender="f", age_hint="teen", description="строгая учительница, низкий голос")
    c.performer = {"tuning": {"character_note": "ехидная", "timbre_preset": "older"}}
    p = design_prompt(c, "ru")
    assert "A young female voice" in p and "строгая учительница" in p and "ехидная" in p and "an older, deeper voice" in p
    assert "Character:" not in design_prompt(Character(names={"src": "a"}), "ru")
    assert "строгая" not in design_prompt(c, "ru", description="")  # explicit empty description replaces the sheet's
    assert "Voice: slightly higher" in design_prompt(c, "ru", index=1) and "Voice:" not in p  # variation hints from the 2nd design on


def test_rvc_f0_transpose_added_on_route_b():
    plan = tuning_plan({"pitch_st": 3}, caps("qwen3_tts"), "B")
    assert plan["vc_f0_transpose"] == 3 and plan["post_pitch_st"] == 0.0
    plan = tuning_plan({"pitch_st": -2.4}, caps("qwen3_tts"), "B")
    assert plan["vc_f0_transpose"] == -2 and plan["post_pitch_st"] == pytest.approx(-0.4)
    plan = tuning_plan({"pitch_st": 2.1}, caps("qwen3_tts"), "B")
    assert plan["vc_f0_transpose"] == 2 and plan["post_pitch_st"] == 0.0  # a remainder under 0.25 st is dropped
    # the age chips stand in for words where the VC replaces the timbre
    assert tuning_plan({"timbre_preset": "younger"}, caps("qwen3_tts"), "B")["vc_f0_transpose"] == 2  # round(1.5)
    assert tuning_plan({"timbre_preset": "older", "pitch_st": 1}, caps("qwen3_tts"), "B")["vc_f0_transpose"] == 0


def test_rubberband_fallback_for_routes_and_backends_without_pitch_knob():
    for route in ("A", "P", "C"):
        plan = tuning_plan({"pitch_st": 4}, caps("qwen3_tts"), route)
        assert plan["post_pitch_st"] == 4.0 and plan["vc_f0_transpose"] == 0
    # younger / older: words where the backend reads words, a pitch stand-in where it does not
    assert tuning_plan({"timbre_preset": "younger"}, caps("qwen3_tts"), "A")["post_pitch_st"] == 0.0
    assert tuning_plan({"timbre_preset": "younger"}, caps("chatterbox_ml"), "A")["post_pitch_st"] == 1.5
    assert tuning_plan({"timbre_preset": "older"}, caps("fish_s2"), "P")["post_pitch_st"] == -1.5
    assert tuning_plan({"timbre_preset": "softer"}, caps("fish_s2"), "P")["post_pitch_st"] == 0.0
    assert tuning_plan(None, caps("qwen3_tts"), "A") == {
        "vc_f0_transpose": 0, "post_pitch_st": 0.0, "speed": 1.0, "speed_native": False, "post_speed": 1.0,
        "target_duration_scale": 1.0, "lufs_offset_db": 0.0, "timbre_pitch_st": 0.0,
    }


def test_speed_native_or_master_pass():
    t = {"speed": 1.2}
    p = tuning_plan(t, caps("qwen3_tts"), "A")  # no native speed: master applies it
    assert not p["speed_native"] and p["post_speed"] == 1.2 and p["target_duration_scale"] == 1.0
    p = tuning_plan(t, caps("f5_ru"), "A")  # duration-controllable: target_duration / speed, nothing left for master
    assert p["speed_native"] and p["post_speed"] == 1.0 and p["target_duration_scale"] == pytest.approx(1 / 1.2, abs=1e-4)
    assert tuning_plan(t, caps("zonos2"), "A")["speed_native"]
    assert tuning_plan({"energy": 1.0}, caps("qwen3_tts"), "A")["lufs_offset_db"] == 3.0
    assert tuning_plan({"energy": -0.5}, caps("qwen3_tts"), "A")["lufs_offset_db"] == -1.5

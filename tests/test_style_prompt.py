from __future__ import annotations

import pytest

from vndub.text.style_prompt import (
    StyleSpec,
    build_style,
    emotion_vote,
    render,
    to_cosy_instruct,
    to_emotion_ref,
    to_fish_tags,
    to_higgs_tags,
    to_instruction,
    to_knobs,
    to_voxcpm_prefix,
)

HAPPY = StyleSpec(emotion="happy", intensity=0.63, tempo="fast", loudness="loud", events=["laughter"], voice_quality=["bright"], arc="rising_end", notes="teasing", valence=0.78)


# --- emotion vote -------------------------------------------------------------


def test_vote_weights_e2v_and_anime():
    label, probs = emotion_vote({"happy": 0.5, "neutral": 0.5}, {"Sad": 1.0})
    assert probs["happy"] == pytest.approx(0.3)
    assert probs["neutral"] == pytest.approx(0.3)
    assert probs["sad"] == pytest.approx(0.4)
    assert label == "sad"


def test_vote_keeps_embarrassed_as_extra_class():
    label, probs = emotion_vote({"neutral": 0.4, "happy": 0.6}, {"Embarrassed": 1.0})
    assert "embarrassed" in probs and probs["embarrassed"] == pytest.approx(0.4)
    assert label == "embarrassed"


def test_vote_skips_other_unknown_when_runner_up_is_meaningful():
    label, _ = emotion_vote({"other": 0.5, "angry": 0.3, "neutral": 0.2}, None)
    assert label == "angry"


def test_vote_anime_only_and_empty():
    assert emotion_vote(None, {"Angry": 0.9, "Neutral": 0.1})[0] == "angry"
    assert emotion_vote(None, None) == ("neutral", {"neutral": 1.0})


# --- build_style --------------------------------------------------------------


def _passport(**pros):
    return {
        "emotion": {"e2v": {"probs": {"happy": 0.8, "neutral": 0.2}}, "adv": {"arousal": 0.7, "valence": 0.8, "dominance": 0.5}},
        "events": ["laughter", "bgm"],
        "prosody": pros,
    }


def test_build_style_tempo_loudness_arc():
    stats = {"rate_median": 8.0, "energy_db_median": -22.0, "f0_median_hz": 250.0}
    s = build_style(_passport(rate_mora_per_s=9.5, energy_db_mean=-18.0, arc="rising_end", f0_median_hz=300.0), stats, notes="teasing")
    assert (s.emotion, s.tempo, s.loudness, s.arc) == ("happy", "fast", "loud", "rising_end")
    assert s.intensity == pytest.approx(0.7)
    assert s.events == ["laughter"]  # bgm is not a delivery event
    assert "bright" in s.voice_quality
    assert s.notes == "teasing"
    s2 = build_style(_passport(rate_mora_per_s=6.0, energy_db_mean=-27.0), stats)
    assert (s2.tempo, s2.loudness) == ("slow", "soft")
    s3 = build_style(_passport(rate_mora_per_s=8.5, energy_db_mean=-21.0), stats)
    assert (s3.tempo, s3.loudness, s3.arc) == ("normal", "normal", "flat")


def test_build_style_whisper_rule_and_tense():
    stats = {"energy_db_median": -20.0}
    s = build_style(_passport(energy_db_mean=-30.0, spectral_tilt_db_khz=-0.5, voiced_ratio=0.2), stats)
    assert "whisper" in s.events and "whispery" in s.voice_quality
    p = {"emotion": {"e2v": {"probs": {"angry": 1.0}}, "adv": {"arousal": 0.9, "valence": 0.1}}, "prosody": {}}
    assert "tense" in build_style(p).voice_quality


def test_build_style_without_adv_uses_proxy_and_roundtrips():
    p = {"emotion": {"e2v": {"probs": {"sad": 1.0}}}, "prosody": {}}
    s = build_style(p)
    assert s.emotion == "sad" and 0.0 <= s.intensity <= 0.5
    assert StyleSpec.from_dict(s.to_dict()) == s


# --- renderers ----------------------------------------------------------------


def test_to_instruction_contains_all_channels():
    text = to_instruction(HAPPY)
    for frag in ("happy", "intense", "fast pace", "loudly", "laugh", "bright", "rising intonation", "Delivery: teasing."):
        assert frag in text
    calm = to_instruction(StyleSpec())
    assert "calm, neutral" in calm and "pace" not in calm


def test_to_instruction_caption_toggle():
    s = StyleSpec(emotion="sad", caption="Quiet, trembling voice.")
    assert "Reference performance: Quiet" in to_instruction(s)
    assert "Reference performance" not in to_instruction(s, include_caption=False)


def test_cosy_instruct_format():
    out = to_cosy_instruct(HAPPY)
    assert out.startswith("You are a helpful assistant. ") and out.endswith("<|endofprompt|>")


def test_voxcpm_prefix():
    assert to_voxcpm_prefix(HAPPY) == "(happy, faster, loud, laughing, rising at the end, teasing)"
    assert to_voxcpm_prefix(StyleSpec(emotion="sad", intensity=0.1, tempo="slow")) == "(sad, subdued, slower)"


def test_fish_tags():
    assert to_fish_tags(HAPPY) == "[happy] [laughing] [loud] [fast] [pitch up]"
    assert to_fish_tags(StyleSpec(emotion="angry", intensity=0.9, loudness="loud")) == "[very angry] [shouting]"
    assert to_fish_tags(StyleSpec()) == ""


def test_higgs_tags():
    assert to_higgs_tags(HAPPY) == "<|emotion:happiness|><|prosody:speed_fast|><|sfx:laughter|>"
    w = StyleSpec(emotion="sad", events=["whisper", "cry"])
    assert to_higgs_tags(w) == "<|emotion:sadness|><|style:whisper|><|sfx:crying|>"


def test_knobs_chatterbox_range_and_cross_lingual():
    k = to_knobs(StyleSpec(intensity=0.0), "chatterbox", {"exaggeration_range": [0.3, 1.2], "cfg_weight": 0.5, "cfg_weight_cross_lingual": 0.0})
    assert k == {"exaggeration": 0.3, "cfg_weight": 0.5}
    k = to_knobs(StyleSpec(intensity=1.0), "chatterbox", {"exaggeration_range": [0.3, 1.2]}, cross_lingual=True)
    assert k["exaggeration"] == pytest.approx(1.2) and k["cfg_weight"] == 0.0


def test_knobs_zonos_from_adv():
    s = StyleSpec(emotion="happy", intensity=0.6, valence=0.8, tempo="fast", emotion_probs={"happy": 0.6, "neutral": 0.2, "embarrassed": 0.2})
    k = to_knobs(s, "zonos")
    assert len(k["emotion"]) == 8 and sum(k["emotion"]) == pytest.approx(1.0, abs=1e-3)
    assert k["emotion"][0] > k["emotion"][7]  # happiness > neutral
    assert k["valence"] == 0.8 and k["arousal"] == 0.6 and k["speaking_rate"] == 17.0


def test_knobs_generic():
    assert to_knobs(HAPPY, "generic") == {"tempo": "fast", "loudness": "loud", "intensity": 0.63, "arc": "rising_end", "emotion": "happy"}


@pytest.mark.parametrize(
    "caps,expect_instruct,expect_prefix,knob_key",
    [
        ({"instruct": "qwen"}, "Speak in", "", "tempo"),
        ({"instruct": "natural"}, "Speak in", "", "tempo"),
        ({"instruct": "cosy", "tags": "cosy_inline"}, "<|endofprompt|>", "", "tempo"),
        ({"instruct": "voxcpm"}, None, "(happy", "tempo"),
        ({"tags": "fish"}, None, "[happy]", "tempo"),
        ({"tags": "higgs", "knobs": "zonos"}, None, "<|emotion:happiness|>", "emotion"),
        ({"knobs": "chatterbox"}, None, "", "exaggeration"),
    ],
)
def test_render_per_caps(caps, expect_instruct, expect_prefix, knob_key):
    out = render(HAPPY, caps, {"exaggeration_range": [0.3, 1.2]})
    if expect_instruct is None:
        assert out["instruct"] is None
    else:
        assert expect_instruct in out["instruct"]
    assert out["text_prefix"].startswith(expect_prefix)
    assert knob_key in out["knobs"]
    assert out["style_prompt"]


def test_render_cosy_inline_breath():
    out = render(StyleSpec(events=["breath"]), {"instruct": "cosy", "tags": "cosy_inline"})
    assert out["text_prefix"] == "[breath]"


# --- emotion reference --------------------------------------------------------

REFS = [
    {"path": "a.wav", "emotion": "neutral", "arousal": 0.3, "valence": 0.5, "snr": 30},
    {"path": "b.wav", "emotion": "happy", "arousal": 0.4, "valence": 0.8, "snr": 25},
    {"path": "c.wav", "emotion": "happy", "arousal": 0.8, "valence": 0.8, "snr": 20},
    {"path": "d.wav", "emotion": "sad", "arousal": 0.2, "valence": 0.2, "snr": 35, "usage": "timbre"},
]


def test_emotion_ref_same_class_closest_arousal():
    assert to_emotion_ref(StyleSpec(emotion="happy", intensity=0.75), REFS) == "c.wav"
    assert to_emotion_ref(StyleSpec(emotion="happy", intensity=0.35), REFS) == "b.wav"


def test_emotion_ref_nearest_av_fallback_and_usage_filter():
    # angry (no clip): high arousal, low valence -> nearest is c (0.8 arousal) vs d (0.2,0.2)?
    s = StyleSpec(emotion="fearful", intensity=0.25, valence=0.2)
    assert to_emotion_ref(s, REFS) == "d.wav"
    assert to_emotion_ref(s, REFS, usage="emotion") != "d.wav"
    assert to_emotion_ref(StyleSpec(), []) is None

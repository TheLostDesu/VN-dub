from __future__ import annotations

import pytest

from vndub.text.style_prompt import (
    StyleSpec,
    build_style,
    emotion_vote,
    laugh_phrase,
    laughter_decision,
    laughter_tokens,
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
    # Higgs Audio v3 model card (huggingface.co/bosonai/higgs-audio-v3-tts-4b): emotion values are elation, amusement, anger, sadness,
    # shame ... (no "happiness" / "neutral" / "embarrassment"), style values singing | shouting | whispering
    assert to_higgs_tags(HAPPY) == "<|emotion:amusement|><|prosody:speed_fast|><|sfx:laughter|>"  # a happy laugh is amusement
    assert to_higgs_tags(StyleSpec(emotion="happy", intensity=0.5)) == "<|emotion:elation|>"
    assert to_higgs_tags(StyleSpec()) == ""  # neutral: no tag
    w = StyleSpec(emotion="sad", events=["whisper", "cry"])
    assert to_higgs_tags(w) == "<|emotion:sadness|><|style:whispering|><|sfx:crying|>"
    assert to_higgs_tags(StyleSpec(emotion="angry", intensity=0.9, loudness="loud")) == "<|emotion:anger|><|style:shouting|>"
    assert to_higgs_tags(StyleSpec(emotion="embarrassed")) == "<|emotion:shame|>"


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
        ({"tags": "higgs", "knobs": "zonos"}, None, "<|emotion:amusement|>", "emotion"),
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


# --- laughter -------------------------------------------------------------------


def _laugh(kind, units=2, intensity=0.5, only=False, **kw):
    return StyleSpec(emotion="happy", intensity=intensity, events=["laughter"], laughter={"kind": kind, "units": units, "stage": False, "only": only}, **kw)


def test_laugh_phrase_follows_kind_and_strength():
    assert laugh_phrase(StyleSpec(events=["laughter"])) == "with a light laugh"  # unknown kind: the generic phrase of old
    assert laugh_phrase(_laugh("hearty", intensity=0.5)) == "laughing heartily"
    assert laugh_phrase(_laugh("hearty", intensity=0.3)) == "with a warm, easy laugh"  # soft
    assert laugh_phrase(_laugh("hearty", units=4, intensity=0.7)) == "bursting into hearty laughter"  # long and intense
    assert laugh_phrase(_laugh("giggle", intensity=0.3)) == "giggling softly"
    assert laugh_phrase(_laugh("nervous", intensity=0.5)) == "laughing nervously"
    assert laugh_phrase(_laugh("evil", intensity=0.85)) == "cackling maniacally"
    assert laugh_phrase(_laugh("snort")) == "snorting with laughter"
    assert laugh_phrase(_laugh("chuckle")) == "chuckling"
    assert laugh_phrase(_laugh("unheard-of")) == "with a light laugh"


def test_instruction_uses_the_kind_phrase_and_drops_its_note():
    s = _laugh("giggle", intensity=0.5, notes="giggling, hesitant")
    text = to_instruction(s)
    assert "giggling" in text and "light laugh" not in text
    assert text.count("giggling") == 1 and "Delivery: hesitant." in text  # the kind note is not said twice
    assert "Delivery" not in to_instruction(_laugh("hearty", notes="a hearty laugh"))
    assert "Delivery: playful." in to_instruction(_laugh("hearty", notes="a hearty laugh, playful"))
    assert "light laugh" in to_instruction(HAPPY)  # no kind: as before


def test_laughter_only_line_is_an_instruction_about_the_laugh():
    text = to_instruction(_laugh("nervous", units=3, intensity=0.5, only=True, loudness="soft"))
    assert text.startswith("Only laugh, no words: make a short, nervous laugh")
    assert "Speak in" not in text and "softly" in text
    assert to_instruction(_laugh("hearty", only=True)).startswith("Only laugh, no words: make a hearty, natural laugh")
    # "only" without the laughter event (a calm tuning dropped it) is an ordinary line
    calm = StyleSpec(emotion="neutral", laughter={"kind": "hearty", "only": True})
    assert to_instruction(calm).startswith("Speak in")
    cosy = to_cosy_instruct(_laugh("giggle", only=True))
    assert "Only laugh" in cosy and cosy.endswith("<|endofprompt|>")


def test_voxcpm_and_fish_tags_follow_the_kind():
    assert to_voxcpm_prefix(_laugh("giggle", intensity=0.5)) == "(happy, giggling)"
    assert to_voxcpm_prefix(_laugh("evil", intensity=0.5)) == "(happy, laughing sinisterly)"
    assert to_fish_tags(_laugh("nervous", intensity=0.5)) == "[happy] [laughing nervously]"
    assert to_fish_tags(_laugh("chuckle", intensity=0.5)) == "[happy] [chuckling]"
    assert to_fish_tags(_laugh("hearty", intensity=0.5), inline_laughter=True) == "[happy]"  # the text carries [laughing] itself
    assert to_higgs_tags(_laugh("hearty"), inline_laughter=True) == "<|emotion:amusement|>"  # ... and "<|sfx:laughter|>Haha"


def test_fish_other_events_use_documented_tags():
    # docs.fish.audio / the S2 Pro card list [inhale] [exhale] [sigh] [whisper] [clearing throat]; there is no [breath] / [cough]
    assert to_fish_tags(StyleSpec(events=["breath", "cough"])) == "[inhale] [clearing throat]"


def test_laughter_decision_per_tag_syntax():
    assert laughter_decision({"instruct": "qwen", "tags": "none"}) == "respell"  # Qwen3-TTS: no laugh token, spelling is the lever
    assert laughter_decision({"knobs": "chatterbox"}) == "respell"
    assert laughter_decision({"instruct": "cosy", "tags": "cosy_inline"}) == "inline:[laughter]"
    assert laughter_decision({"tags": "fish"}) == "tag:[laughing]"
    assert laughter_decision({"tags": "higgs"}) == "tag:<|sfx:laughter|>"
    assert laughter_tokens({"tags": "fish"})["chuckle"] == "[chuckling]" and laughter_tokens({"tags": "none"}) == {}
    out = render(HAPPY, {"instruct": "qwen"})
    assert out["laughter"] == "respell"
    assert render(HAPPY, {"tags": "higgs"})["laughter"] == "tag:<|sfx:laughter|>"
    # the Fish / Higgs prefix leaves its own laugh tag out when the text carries it
    assert "[laughing]" in render(HAPPY, {"tags": "fish"})["text_prefix"]
    assert "[laughing]" not in render(HAPPY, {"tags": "fish"}, laughter_inline=True)["text_prefix"]
    assert "sfx:laughter" not in render(HAPPY, {"tags": "higgs"}, laughter_inline=True)["text_prefix"]


def test_stylespec_laughter_is_backward_compatible():
    assert "laughter" not in StyleSpec().to_dict() and "laughter" not in HAPPY.to_dict()  # byte-identical to stored sidecars / cache keys
    old = {k: v for k, v in HAPPY.to_dict().items()}
    assert StyleSpec.from_dict(old).laughter is None
    s = _laugh("giggle")
    assert StyleSpec.from_dict(s.to_dict()) == s and s.to_dict()["laughter"]["kind"] == "giggle"


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


def test_emotion_ref_survives_null_arousal_valence_snr():
    # refs.json carries arousal/valence/snr = null when the ADV model is off or the gate is relaxed
    nulls = [
        {"path": "n1.wav", "emotion": "happy", "arousal": None, "valence": None, "snr": None},
        {"path": "n2.wav", "emotion": "happy", "arousal": None, "valence": None, "snr": 12.0},
        {"path": "n3.wav", "emotion": "sad", "arousal": None, "valence": None, "snr": None},
    ]
    # same emotion class: a null arousal counts as the target intensity, so snr breaks the tie
    assert to_emotion_ref(StyleSpec(emotion="happy", intensity=0.5), nulls) == "n2.wav"
    # nearest-by-(valence, arousal) fallback uses the emotion table for a null value
    assert to_emotion_ref(StyleSpec(emotion="fearful", intensity=0.2, valence=0.2), nulls) in {"n1.wav", "n2.wav", "n3.wav"}


def test_vote_zero_weight_only_model_is_neutral_not_a_crash():
    assert emotion_vote({"happy": 1.0}, None, w_e2v=0.0, w_anime=0.4) == ("neutral", {"neutral": 1.0})
    assert emotion_vote(None, {"Sad": 1.0}, w_e2v=0.6, w_anime=0.0) == ("neutral", {"neutral": 1.0})
    assert emotion_vote({"happy": 1.0}, {"Sad": 1.0}, w_e2v=0.0, w_anime=0.0)[0] == "neutral"

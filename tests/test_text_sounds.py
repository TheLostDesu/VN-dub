"""vndub.text.sounds (non-speech sounds: snoring, kisses, yawns ...) and its delivery layer."""

from __future__ import annotations

import pytest

from vndub.text import sounds
from vndub.text.delivery import Delivery, Layer, llm_layer, merge_layers, rule_pass, to_style, valid_sound, validate_delivery


@pytest.mark.parametrize(
    "text,kind",
    [
        ("ZZZ...", "snore"),
        ("Muah!", "kiss"),
        ("Achoo!", "sneeze"),
        ("Ahem.", "cough"),
        ("Hmph!", "hmph"),
        ("Mmm...", "hum"),
        ("М-м-м...", "hum"),
        ("Му-ах!", "kiss"),
        ("З-З-З...", "snore"),
        ("Хр-р-р...", "snore"),
        ("Апчхи!", "sneeze"),
        ("ぐーぐー", "snore"),
        ("ちゅっ", "kiss"),
    ],
)
def test_find_sounds_kinds(text, kind):
    assert sounds.sound_kind(text) == kind
    assert sounds.is_sound_only(text)
    s = sounds.summarize(text)
    assert s["kind"] == kind and s["only"] is True and s["stage"] is False


def test_plain_speech_has_no_sound():
    assert sounds.find_sounds("Hello.") == []
    assert sounds.summarize("Hello.") is None
    assert not sounds.has_sound("Привет, как дела?")
    assert not sounds.is_sound_only("Hello.")


def test_laughs_are_never_sounds():
    assert sounds.find_sounds("Hehe") == []


def test_sound_inside_speech_is_not_only():
    s = sounds.summarize("Brrr, it's cold.", "en")
    assert s == {"kind": "shiver", "n": 1, "only": False, "stage": False}
    assert not sounds.is_sound_only("Brrr, it's cold.")
    assert sounds.strip_sounds("Brrr, it's cold.", "en").strip(", ") .startswith("it's cold")


def test_stage_directions():
    s = sounds.summarize("*yawns* Good morning.", "en")
    assert s["kind"] == "yawn" and s["stage"] is True and s["only"] is False
    assert sounds.strip_sounds("*yawns* Good morning.", "en").strip() == "Good morning."
    only = sounds.summarize("(храпит)", "ru")
    assert only["kind"] == "snore" and only["stage"] is True and only["only"] is True
    ja = sounds.summarize("（あくび）", "ja")
    assert ja["kind"] == "yawn" and ja["stage"] is True and ja["only"] is True


def test_strip_sounds_leaves_the_rest():
    assert sounds.strip_sounds("ZZZ... fine.", "en").strip(". ") == "fine"
    assert sounds.strip_sounds("Hello.", "en") == "Hello."


def test_load_table_of_the_repo_file():
    t = sounds.load_table()
    assert set(t) == {"langs", "stage", "backends"}
    ru = sounds.lang_block(t, "ru-RU")
    assert ru["snore"] == "Хр-р-р... хр-р-р..." and set(ru) >= set(sounds.KINDS)
    assert sounds.lang_block(t, "de") == sounds.lang_block(t, "en")  # default -> en


def test_render_for_backend():
    t = sounds.load_table()
    text, rec = sounds.render_for_backend("ZZZ...", "ru", "chatterbox_ml", t)
    assert text == "Хр-р-р... хр-р-р..." and rec["kind"] == "snore" and rec["only"] is True
    text, rec = sounds.render_for_backend("ZZZ...", "ru", "fish_s2", t)
    assert text == "[snoring]Хр-р-р... хр-р-р..." and rec["kind"] == "snore"


# ---- delivery ---------------------------------------------------------------------------------------------------------


def test_rule_pass_snore_line():
    lyr = rule_pass("ZZZ...", "", None, "en")
    assert "snore" in lyr.events and lyr.sound == {"kind": "snore", "only": True, "stage": False}
    assert lyr.loudness == "soft" and lyr.intensity == pytest.approx(0.1) and any("snoring, no words" in n for n in lyr.notes)
    d = merge_layers([lyr])
    assert d.sound == {"kind": "snore", "only": True, "stage": False} and d.emotion == "neutral" and d.loudness == "soft"
    assert d.notes.startswith("snoring, no words") or "snoring, no words" in d.notes
    style = to_style(d)
    assert "snore" in style.events


def test_rule_pass_reads_the_translation_too():
    lyr = rule_pass("ずーっ", "Хр-р-р... хр-р-р...", None, "ja")
    assert lyr.sound and lyr.sound["kind"] == "snore"


def test_rule_pass_kinds_emotion_and_tempo():
    gasp = merge_layers([rule_pass("*gasps*", "", None, "en")])
    assert gasp.emotion == "surprised" and gasp.sound["kind"] == "gasp"
    hmph = merge_layers([rule_pass("Hmph!", "", None, "en")])
    assert hmph.sound["kind"] == "hmph" and hmph.emotion in ("angry", "neutral")
    yawn = merge_layers([rule_pass("Yaaawn...", "", None, "en")])
    assert yawn.sound["kind"] == "yawn" and yawn.tempo == "slow" and yawn.intensity <= 0.25
    kiss = merge_layers([rule_pass("Muah!", "", None, "en")])
    assert kiss.sound["kind"] == "kiss" and kiss.sound["only"] is True


def test_caps_sound_is_not_shouting():
    lyr = rule_pass("ZZZZ...", "", None, "en")
    assert lyr.loudness == "soft"  # not "loud" from the capitals rule


def test_a_sound_in_speech_is_not_only():
    d = merge_layers([rule_pass("Brrr, it's cold.", "", None, "en")])
    assert d.sound and d.sound["kind"] == "shiver" and d.sound["only"] is False


def test_laugh_and_sound_in_one_line():
    d = merge_layers([rule_pass("Haha! ZZZ...", "", None, "en")])
    assert "laugh" in d.events and "snore" in d.events
    assert d.laughter and d.sound["kind"] == "snore" and d.sound["only"] is False


def test_llm_layer_takes_the_record_sound():
    lyr = llm_layer(None, {"kind": "snore", "only": True})
    assert lyr is not None and lyr.sound["kind"] == "snore" and "snore" in lyr.events
    assert llm_layer(None, {"kind": "bogus"}) is None
    delivery = {"emotion": "neutral", "intensity": 0.3, "tempo": "normal", "loudness": "normal", "events": [], "confidence": 0.8}
    both = llm_layer(delivery, {"kind": "kiss", "only": False})
    assert both.sound["kind"] == "kiss" and "kiss" in both.events and both.confidence == 0.8
    d = merge_layers([None, None, both])
    assert d.sound["kind"] == "kiss"


def test_validate_delivery_accepts_sound_events():
    v = validate_delivery({"emotion": "neutral", "intensity": 0.2, "tempo": "normal", "loudness": "soft", "events": ["snore"], "confidence": 0.7})
    assert v and v["events"] == ["snore"]
    assert valid_sound({"kind": " Snore ", "only": 1}) == {"kind": "snore", "only": True, "stage": False}
    assert valid_sound({"kind": "zzz"}) is None and valid_sound("snore") is None


def test_delivery_roundtrip_and_old_passports():
    d = merge_layers([rule_pass("ZZZ...", "", None, "en")])
    again = Delivery.from_dict(d.to_dict())
    assert again.sound == d.sound
    plain = merge_layers([rule_pass("Hello there.", "", None, "en")])
    assert "sound" not in plain.to_dict()  # absent without a sound: passports of earlier versions read the same
    assert Delivery.from_dict({"emotion": "happy"}).sound == {}
    assert isinstance(Layer("x").sound, type(None))

"""Round 18: Latin / interpolations in the TTS text, number forms, hesitation is not a hum, laughter misses."""

from __future__ import annotations

import pytest

from vndub.text import laughter, ru, sounds


@pytest.mark.parametrize(
    "src,expected",
    [
        ("Привет, Kevin! Как дела, Mr. Smith?", "Привет, Кевин! Как дела, Мистер Смиз?"),
        ("Я играю в Minecraft на PS5.", "Я играю в Майнкрафт на пи-эс пять."),
        ("Это мой iPhone.", "Это мой айфон."),
        ("x2 урона!", "икс два урона!"),
        ("[player_name], иди сюда!", "иди сюда!"),
        ("Привет, [name]!", "Привет!"),
        ("Привет, [name], как дела?", "Привет, как дела?"),
        ("OK, увидимся.", "окей, увидимся."),
        ("Подключи USB.", "Подключи ю-эс-би."),
        ("В 1999 году мне было 17 лет.", "В тысяча девятьсот девяносто девятом году мне было семнадцать лет."),
        ("Это было в 2026 г. в Москве.", "Это было в две тысячи двадцать шестом году в Москве."),
        ("С 3 друзьями мы пошли в кино.", "С тремя друзьями мы пошли в кино."),
        ("Без 5 минут шесть.", "Без пяти минут шесть."),
        ("К 2 часам.", "К двум часам."),
        ("О 3 вещах.", "О трёх вещах."),
        ("От 1 до 5.", "От одного до пяти."),
        ("Температура -5 градусов.", "Температура минус пять градусов."),
        ("Он занял 1-е место, а я 2-ю.", "Он занял первое место, а я вторую."),
        ("Это стоит $5 или 300 ¥.", "Это стоит пять долларов или триста иен."),
        ("Позвони мне: 555-1234.", "Позвони мне: пять пять пять, один два три четыре."),
        ("Нужно 3-5 дней", "Нужно от трёх до пяти дней"),
        ("Половина — это 1/2.", "Половина — это один из двух."),
        ("Счёт 2:1 в нашу пользу!", "Счёт два — один в нашу пользу!"),
        ("Встречаемся в 7:30.", "Встречаемся в семь тридцать."),
        ("Сенпай~ ♪ Ты такой милый ♥", "Сенпай Ты такой милый"),
    ],
)
def test_normalize_for_tts_r18(src, expected):
    assert ru.normalize_for_tts(src) == expected


def test_translit_latin():
    assert ru.translit_latin("Kevin") == "Кевин"
    assert ru.translit_latin("Привет, Mr. Smith 5!") == "Привет, Мистер Смиз 5!"
    assert ru.translit_latin("ночь") == "ночь"
    assert ru.translit_latin("PS") == "пи-эс"
    for w in ("Hello", "Tokyo", "Alice", "Wi-Fi", "don't"):
        out = ru.translit_latin(w)
        assert out == ru.translit_latin(w) and not any("a" <= c.lower() <= "z" for c in out)


def test_stage_directions_are_not_transliterated_or_dropped():
    assert ru.normalize_for_tts("Kevin, *laughs* [giggles]") == "Кевин, *laughs* [giggles]"


@pytest.mark.parametrize("text", ["Hmm.", "Hmm...", "Mm-hmm.", "Mhm.", "Uh-huh.", "Хм.", "Хм...", "Угу.", "Ага!", "Мм."])
def test_interjections_are_not_sounds(text):
    assert sounds.find_sounds(text) == []
    assert sounds.summarize(text, "auto") is None


def test_hum_stays_a_sound():
    assert sounds.sound_kind("Mmmm...") == "hum"
    assert sounds.sound_kind("М-м-м.") == "hum"
    assert sounds.summarize("*hums*", "en")["kind"] == "hum"


def test_hesitation_lines_have_no_sound_record():
    from vndub.stages import translate

    assert translate.sound_record("Mm-hmm.", {"text_tgt": "Угу."}, "en", "ru") is None
    assert translate.sound_record("Hmm...", {"text_tgt": "Хм..."}, "en", "ru") is None


@pytest.mark.parametrize(
    "text,kind",
    [("Хах.", "hearty"), ("ахах", "hearty"), ("Бу-га-га", "hearty")],
)
def test_laughter_misses(text, kind):
    assert laughter.laughter_kind(text) == kind
    assert laughter.is_laughter_only(text)


@pytest.mark.parametrize(
    "text",
    ["ах", "эх", "ага", "Аха!", "муха", "Бухарест", "Эхо", "Ехать", "Ку-ку", "Фу!", "Haruhi", "Hana", "Ha-chan", "Ahab", "Momo", "Nana", "Ха, конечно!", "Ha, right.", "Ха!", "Ha!", "Hah!"],
)
def test_laughter_false_positives(text):
    assert not laughter.has_laughter(text)

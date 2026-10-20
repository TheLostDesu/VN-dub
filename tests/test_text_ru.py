from __future__ import annotations

import pytest

from vndub.backends.score.cer import normalize_for_cer
from vndub.text import ru


@pytest.mark.parametrize(
    "src,expected",
    [
        ("Я получила 100 баллов на тесте!", "Я получила сто баллов на тесте!"),
        ("Встретимся в 10:30.", "Встретимся в десять тридцать."),
        ("В 7:00 подъём", "В семь часов подъём"),
        ("Скидка 50%!", "Скидка пятьдесят процентов!"),
        ("Только 1% и 3%", "Только один процент и три процента"),
        ("Нужно 3-5 дней", "Нужно от трёх до пяти дней"),
        ("Это 2-й раз и 3-я попытка", "Это второй раз и третья попытка"),
        ("до 5-го числа", "до пятого числа"),
        ("Длина 3,5 метра", "Длина три целых пять десятых метра"),
        ("Комната №7", "Комната номер семь"),
    ],
)
def test_normalize_numbers(src, expected):
    assert ru.normalize_numbers(src) == expected


def test_expand_abbreviations():
    assert ru.expand_abbreviations("т.е. и т.д. и т.п.") == "то есть и так далее и тому подобное"
    assert ru.expand_abbreviations("Это стоит 5 руб.") == "Это стоит 5 рублей"
    assert ru.expand_abbreviations("Купи DVD, OK?") == "Купи ди-ви-ди, окей?"
    assert ru.expand_abbreviations("В 2026 г.") == "В две тысячи двадцать шестой год"


def test_stress_conversions():
    assert ru.strip_stress("в+ыход вы́ход") == "выход выход"
    assert ru.acute_to_plus("вы́ход") == "в+ыход"
    assert ru.plus_to_acute("в+ыход") == "вы́ход"
    # '+' must precede a vowel; stray marks (before consonants / at the end) are dropped
    assert ru.to_plus_format("з+амок вы́ход х+рам +") == "з+амок в+ыход храм "
    assert ru.deyo("ёлка Ёж") == "елка Еж"


def test_count_syllables_and_words():
    assert ru.count_syllables("Привет, как дела?") == 5
    assert ru.count_syllables("в+ыход") == 2
    assert ru.count_words("Ну и ладно... правда?") == 4


def fake_stresser(text: str) -> str:
    return text.replace("дела", "дел+а").replace("елка", "+ёлка").replace("два", "дв+а")


def test_prepare_tts_text_formats():
    tts, plain = ru.prepare_tts_text("Как дела? Вот 2 елка.", fake_stresser, fmt="plus")
    assert tts == "Как дел+а? Вот дв+а +ёлка."
    assert plain == "Как дела? Вот два ёлка."
    tts2, plain2 = ru.prepare_tts_text("Как дела? Вот 2 елка.", fake_stresser, fmt="plain")
    assert tts2 == plain2 == "Как дела? Вот два ёлка."
    tts3, _ = ru.prepare_tts_text("Вот 2 елка.", fake_stresser, fmt="plain", yo=False)
    assert tts3 == "Вот два елка."


def test_prepare_tts_text_rejects_stresser_that_changes_words():
    tts, plain = ru.prepare_tts_text("Привет", lambda t: "Пока", fmt="plus")
    assert tts == plain == "Привет"
    tts, _ = ru.prepare_tts_text("Привет", lambda t: (_ for _ in ()).throw(RuntimeError("boom")), fmt="plus")
    assert tts == "Привет"


def test_prepare_without_stresser_and_cer_reference_matches_cer_normalization():
    tts, plain = ru.prepare_tts_text("Я получила 100 баллов…", None)
    assert tts == "Я получила сто баллов..."
    assert normalize_for_cer(plain) == "я получила сто баллов"

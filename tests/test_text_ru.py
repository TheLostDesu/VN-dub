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


@pytest.mark.parametrize(
    ("plus", "acute"),
    [
        ("На двер+и вис+ит зам+ок, а вдал+и ст+арый з+амок.", "На двери́ виси́т замо́к, а вдали́ ста́рый за́мок."),
        ("+Это о́чень", "Э́то о́чень"),  # at the start of a word; a mark that already is an acute stays
        ("Чт+о т+ы д+елаешь зд+есь?", "Что ты де́лаешь здесь?"),  # a word with one vowel has nothing to choose
        ("К+ое-что, н+е было", "Ко́е-что, не было"),  # each part of a hyphenated word counts on its own
        ("т+ёмный вс+ё, Ё+ж, +Ёлка", "тёмный всё, Ёж, Ёлка"),  # ё is always stressed
        ("2+2 и C+ и а+б", "2+2 и C+ и аб"),  # no stress mark: a plus that is not in a word; a stray one in a word goes
        ("без меток", "без меток"),
        ("", ""),
    ],
)
def test_to_acute_format(plus: str, acute: str) -> None:
    assert ru.to_acute_format(plus) == acute
    assert ru.strip_stress(ru.to_acute_format(plus)) == ru.strip_stress(acute)


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


def test_cm_after_a_number_is_centimetres_not_see():
    assert ru.normalize_for_tts("Мой рост 160 см.") == "Мой рост сто шестьдесят сантиметров."
    assert ru.normalize_for_tts("Рост 160см, вес 50") == "Рост сто шестьдесят сантиметров, вес пятьдесят"
    assert ru.expand_abbreviations("Рост 160 см. Дальше") == "Рост 160 сантиметров. Дальше"
    # "см." that does not follow a number still means "see"
    assert ru.expand_abbreviations("см. рисунок") == "смотри рисунок"
    assert ru.expand_abbreviations("стр. 5, см. рис. 2") == "стр. 5, смотри рис. 2"


def test_strip_stress_keeps_a_plus_that_is_not_a_stress_mark():
    assert ru.strip_stress("2+2") == "2+2"
    assert ru.strip_stress("С++ и 3 + 4") == "С++ и 3 + 4"
    assert ru.strip_stress("в+ыход, з+амок, +ёлка, вы́ход") == "выход, замок, ёлка, выход"
    tts, plain = ru.prepare_tts_text("Это 2+2", None, fmt="plain")
    assert tts == plain == "Это два+два"  # used to collapse into "двадва"


def test_normalize_numbers_leaves_unspellable_huge_numbers_alone():
    huge = "9" * 40
    # num2words raises KeyError from 34 digits up
    assert ru.normalize_numbers(huge) == huge
    assert ru.normalize_numbers(f"Я {huge} раз и 5 раз") == f"Я {huge} раз и пять раз"
    assert ru.normalize_numbers(f"№{huge} и {huge}%") == f"№{huge} и {huge}%"
    assert ru.normalize_numbers(f"{huge}-й") == f"{huge}-й"
    assert ru.normalize_numbers(f"3-{huge}").endswith(huge)
    assert ru.normalize_for_tts(f"В {huge} г.") == f"В {huge} г."
    # 33 digits is still spelled
    assert "9" not in ru.normalize_numbers("9" * 33)

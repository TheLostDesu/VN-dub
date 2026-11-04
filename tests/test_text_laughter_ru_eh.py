"""Russian laughs the first detector missed: "Эхехехе", "Ихихи", "Уфуфу", "Кукуку", "Ня-хаха" (round 17)."""

from __future__ import annotations

import pytest

from vndub.text import laughter as lf


@pytest.mark.parametrize(
    "text,kind,units",
    [
        ("Эхехехе.", "nervous", 4),
        ("Эхехехе...", "nervous", 4),
        ("Эхехе", "nervous", 3),
        ("Эхе-хе", "nervous", 3),
        ("эхе-хе-хе", "nervous", 4),
        ("Эхэхэ", "nervous", 3),
        ("эхэ", "nervous", 2),
        ("Ихихи", "giggle", 3),
        ("Ухухуху", "hearty", 4),
        ("Уфуфу", "giggle", 2),
        ("Фуфуфу", "giggle", 3),
        ("Фу-фу-фу", "giggle", 3),
        ("Кукуку", "evil", 3),
        ("Ку-ку-ку", "evil", 3),
        ("Ня-хаха", "hearty", 2),
        ("Ахаха", "hearty", 3),
        ("Муахаха!", "evil", 2),
    ],
)
def test_new_ru_laughs(text: str, kind: str, units: int) -> None:
    ls = lf.find_laughs(text, "ru")
    assert len(ls) == 1, ls
    assert (ls[0].kind, ls[0].units) == (kind, units)
    assert ls[0].start == 0 and ls[0].text == text.rstrip(".!")  # the whole word
    assert lf.is_laughter_only(text)
    assert lf.strip_laughter(text) == ""


@pytest.mark.parametrize("text", ["Фу!", "Фу, какая гадость.", "Фу-фу.", "Ку-ку.", "Эхо.", "Эх, жаль.", "Ухо.", "Кукушка.", "Эхо-хо.", "Ухудшение."])
def test_new_ru_no_false_positives(text: str) -> None:
    assert lf.find_laughs(text, "ru") == []


def test_strip_and_only_with_words() -> None:
    assert lf.strip_laughter("Эхехехе... ну ладно.") == "ну ладно."
    assert not lf.is_laughter_only("Эхехе, ну ладно.")


def test_ru_nervous_spelling_from_the_table() -> None:
    table = lf.load_table(lf.default_table_path())
    render = lf.table_renderer(lf.spelling_table(table, "ru"))
    assert lf.spell_laughter("Эхехехе", render) == "Эхехехе..."
    assert lf.spell_laughter("Эхе-хе", render) == "Эхехехе..."
    assert lf.spell_laughter("Хе-хе... ну да.", render).startswith("Эхехе...")


def test_unstress_laughs_removes_acute_inside_laughs_only() -> None:
    assert lf.unstress_laughs("Эхехе́хе.") == "Эхехехе."
    assert lf.unstress_laughs("Эхе́.") == "Эхе."
    assert lf.unstress_laughs("Эхе́хе, приве́т.") == "Эхехе, приве́т."
    assert lf.unstress_laughs("Х+а-х+а-х+а, привет.") == "Ха-ха-ха, привет."
    assert lf.unstress_laughs("Привет, м+ука.") == "Привет, м+ука."

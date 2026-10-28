"""Russian text preparation for TTS and for the CER reference.

Pipeline for one translated line (``prepare_tts_text``)::

    text -> normalize_for_tts (numbers -> words, abbreviations expanded,
            Latin abbreviations spelled, whitespace) -> stresser (ruaccent:
            '+' before the stressed vowel, yo-fication) -> output format

Output formats: ``plus`` keeps ``+`` marks (F5-TTS_RUSSIAN); ``plain`` drops
marks but keeps the ё restored by ruaccent (every other backend: LLM-based TTS
handle bare text better than combining accents). ``text_plain`` (the CER
reference) is the normalized text without stress marks.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Callable

from num2words import num2words

STRESS_CHARS = "+́"
VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
_LAT_VOWELS = "aeiouyAEIOUY"


def _n(value: int | float, **kw) -> str:
    return num2words(value, lang="ru", **kw)


def _keep_unreadable(fn: Callable[[re.Match[str]], str]) -> Callable[[re.Match[str]], str]:
    """Wrap a ``re.sub`` callback so a token num2words cannot spell (34+ digit runs raise KeyError) stays as written."""

    def wrapper(m: re.Match[str]) -> str:
        try:
            return fn(m)
        except (KeyError, IndexError, OverflowError, ValueError):
            return m.group(0)

    return wrapper


# ----------------------------------------------------------------------------
# numbers
# ----------------------------------------------------------------------------

_TIME = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_PERCENT = re.compile(r"(\d+(?:[.,]\d+)?)\s?%")
_RANGE = re.compile(r"\b(\d+)\s?[-–—]\s?(\d+)\b")
# 1-й, 2-я, 3-е, 5-го, 7-му, 10-м, 4-х (genitive plural cardinal), 2-ой
_ORD = re.compile(r"\b(\d+)-(й|ый|ой|ий|я|ая|е|ое|ье|го|ого|его|му|ому|ему|м|ым|им|х|ых|их)\b")
_DECIMAL = re.compile(r"\b(\d+)[.,](\d+)\b")
_INT = re.compile(r"\b\d+\b")
_NO_PERIOD = re.compile(r"№\s?(\d+)")


def _plural_form(n: int, one: str, few: str, many: str) -> str:
    n = abs(n)
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def _ordinal(num: int, suffix: str) -> str:
    s = suffix
    if s in ("й", "ый", "ой", "ий"):
        return _n(num, to="ordinal")
    if s in ("я", "ая"):
        return _n(num, to="ordinal", gender="f")
    if s in ("е", "ое", "ье"):
        return _n(num, to="ordinal", gender="n")
    if s in ("го", "ого", "его"):
        return _n(num, to="ordinal", case="g")
    if s in ("му", "ому", "ему"):
        return _n(num, to="ordinal", case="d")
    if s in ("м", "ым", "им"):
        return _n(num, to="ordinal", case="i")
    if s in ("х", "ых", "их"):
        return _n(num, case="g")  # "4-х" = четырёх
    return _n(num)


def normalize_numbers(text: str) -> str:
    t = text or ""
    t = _NO_PERIOD.sub(_keep_unreadable(lambda m: "номер " + _n(int(m.group(1)))), t)

    @_keep_unreadable
    def time_sub(m: re.Match[str]) -> str:
        h, mi = int(m.group(1)), int(m.group(2))
        if mi == 0:
            return f"{_n(h)} {_plural_form(h, 'час', 'часа', 'часов')}"
        return f"{_n(h)} {_n(mi) if mi >= 10 else 'ноль ' + _n(mi)}"

    t = _TIME.sub(time_sub, t)

    @_keep_unreadable
    def pct_sub(m: re.Match[str]) -> str:
        raw = m.group(1).replace(",", ".")
        if "." in raw:
            return f"{_n(float(raw))} процента"
        v = int(raw)
        return f"{_n(v)} {_plural_form(v, 'процент', 'процента', 'процентов')}"

    t = _PERCENT.sub(pct_sub, t)
    t = _ORD.sub(_keep_unreadable(lambda m: _ordinal(int(m.group(1)), m.group(2))), t)
    t = _RANGE.sub(_keep_unreadable(lambda m: f"от {_n(int(m.group(1)), case='g')} до {_n(int(m.group(2)), case='g')}"), t)
    t = _DECIMAL.sub(_keep_unreadable(lambda m: _n(float(f"{m.group(1)}.{m.group(2)}"))), t)
    t = _INT.sub(_keep_unreadable(lambda m: _n(int(m.group(0)))), t)
    return t


# ----------------------------------------------------------------------------
# abbreviations
# ----------------------------------------------------------------------------

_ABBR = [
    (re.compile(r"\bт\.\s?е\.", re.I), "то есть"),
    (re.compile(r"\bи\s+т\.\s?д\.", re.I), "и так далее"),
    (re.compile(r"\bи\s+т\.\s?п\.", re.I), "и тому подобное"),
    (re.compile(r"\bт\.\s?к\.", re.I), "так как"),
    (re.compile(r"\bи\s+др\.", re.I), "и другие"),
    # "160 см." is centimetres; only a "см." that does not follow a number means "see" (the period stays: it may end the sentence)
    (re.compile(r"(?<=\d)\s?см(?![\wа-яёА-ЯЁ])", re.I), " сантиметров"),
    (re.compile(r"\bсм\.", re.I), "смотри"),
    (re.compile(r"\bул\.", re.I), "улица"),
    (re.compile(r"\bруб\.", re.I), "рублей"),
    (re.compile(r"\bкоп\.", re.I), "копеек"),
    (re.compile(r"\bтыс\.", re.I), "тысяч"),
    (re.compile(r"\bмлн\.?", re.I), "миллионов"),
    (re.compile(r"\bмлрд\.?", re.I), "миллиардов"),
    (re.compile(r"\bкм\b\.?"), "километров"),
    (re.compile(r"\b(\d+)\s?г\.", re.I), _keep_unreadable(lambda m: _n(int(m.group(1)), to="ordinal") + " год")),
    (re.compile(r"\bг\.\s(?=[А-ЯЁ])"), "город "),
]

LATIN_LETTERS = {
    "A": "эй", "B": "би", "C": "си", "D": "ди", "E": "и", "F": "эф", "G": "джи", "H": "эйч", "I": "ай",
    "J": "джей", "K": "кей", "L": "эл", "M": "эм", "N": "эн", "O": "оу", "P": "пи", "Q": "кью", "R": "ар",
    "S": "эс", "T": "ти", "U": "ю", "V": "ви", "W": "дабл-ю", "X": "экс", "Y": "уай", "Z": "зед",
}
_LATIN_WORDS = {"ok": "окей", "okay": "окей", "wow": "вау", "bye": "бай", "hi": "хай", "yes": "йес", "no": "ноу", "vs": "против"}
_LATIN_ABBR = re.compile(r"\b[A-Z]{2,6}\b")
_LATIN_WORD = re.compile(r"\b[A-Za-z]+\b")


def expand_abbreviations(text: str) -> str:
    t = text or ""
    for rx, rep in _ABBR:
        t = rx.sub(rep, t)
    t = _LATIN_ABBR.sub(lambda m: "-".join(LATIN_LETTERS[c] for c in m.group(0)) if m.group(0).lower() not in _LATIN_WORDS else _LATIN_WORDS[m.group(0).lower()], t)
    t = _LATIN_WORD.sub(lambda m: _LATIN_WORDS.get(m.group(0).lower(), m.group(0)), t)
    return t


# ----------------------------------------------------------------------------
# stress
# ----------------------------------------------------------------------------


def strip_stress(text: str) -> str:
    t = unicodedata.normalize("NFD", text or "")
    # a "+" is a stress mark only directly before a vowel; "2+2" keeps its plus
    t = re.sub(r"\+(?=[" + VOWELS + _LAT_VOWELS + r"])", "", t.replace("́", ""))
    return unicodedata.normalize("NFC", t)


def acute_to_plus(text: str) -> str:
    """'вы́ход' (vowel + U+0301) -> 'в+ыход'."""
    t = unicodedata.normalize("NFD", text or "")
    out: list[str] = []
    for ch in t:
        if ch == "́" and out:
            prev = out.pop()
            out.append("+" + prev)
        else:
            out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def plus_to_acute(text: str) -> str:
    """'в+ыход' -> 'вы́ход'."""
    return re.sub(r"\+([" + VOWELS + _LAT_VOWELS + r"])", lambda m: m.group(1) + "́", text or "")


def to_plus_format(text: str) -> str:
    """Normalize any stress notation (ruaccent '+', combining acute) to '+' before the vowel."""
    t = acute_to_plus(text or "")
    # drop '+' that is not followed by a vowel (stray marks)
    return re.sub(r"\+(?![" + VOWELS + _LAT_VOWELS + r"])", "", t)


def deyo(text: str) -> str:
    return (text or "").replace("ё", "е").replace("Ё", "Е")


# ----------------------------------------------------------------------------
# misc
# ----------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def count_syllables(text: str) -> int:
    t = strip_stress(text or "")
    return sum(1 for ch in t if ch in VOWELS or ch in _LAT_VOWELS)


def count_words(text: str) -> int:
    return len([w for w in re.split(r"\s+", strip_stress(text or "").strip()) if re.search(r"\w", w)])


def normalize_for_tts(text: str) -> str:
    t = unicodedata.normalize("NFC", text or "")
    t = t.replace("…", "...").replace("　", " ")
    t = expand_abbreviations(t)  # before numbers: "2026 г." needs the digit context
    t = normalize_numbers(t)
    return _WS.sub(" ", t).strip()


def prepare_tts_text(
    text: str,
    stresser: Callable[[str], str] | None = None,
    fmt: str = "plain",
    yo: bool = True,
) -> tuple[str, str]:
    """(text_tts, text_plain). ``stresser`` maps normalized text -> '+'-stressed text."""
    norm = normalize_for_tts(text)
    stressed = norm
    if stresser is not None and norm:
        try:
            stressed = to_plus_format(stresser(norm) or norm)
        except Exception:
            stressed = norm
        # a stresser must not change the words; fall back if it did
        if deyo(strip_stress(stressed)).lower() != deyo(norm).lower():
            stressed = norm
    if not yo:
        stressed = deyo(stressed)
    text_tts = to_plus_format(stressed) if fmt == "plus" else strip_stress(stressed)
    text_plain = strip_stress(stressed)
    return text_tts, text_plain

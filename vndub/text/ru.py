"""Russian text preparation for TTS and for the CER reference.

Pipeline for one translated line (``prepare_tts_text``)::

    text -> normalize_for_tts (numbers -> words, abbreviations expanded,
            Latin abbreviations spelled, whitespace) -> stresser (ruaccent:
            '+' before the stressed vowel, yo-fication) -> output format

Output formats: ``plus`` keeps ``+`` marks (F5-TTS_RUSSIAN); ``plain`` drops
marks but keeps the ё restored by ruaccent (every other backend but Chatterbox:
LLM-based TTS handle bare text better than combining accents). ``acute``, for
Chatterbox, is made from the ``plus`` text at synth (:func:`to_acute_format`).
``text_plain`` (the CER reference) is the normalized text without stress marks.
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
_ORD = re.compile(r"\b(\d+)-(й|ый|ой|ий|я|ая|е|ое|ье|го|ого|его|му|ому|ему|м|ым|им|х|ых|их|ю|ую|ей|ые)\b")
_MINUS = re.compile(r"(?<![\w\-–—.,])[-−](?=\d)")
_YEAR = re.compile(
    r"(?:\b(?P<prep>в|во|к|ко|с|со|до|от|после|около|по|при|о|об|на|для|из|без)\s+)?\b(?P<y>\d{3,4})\s?(?P<w>году|года|годом|годе|год|г\.)(?![а-яё])", re.I
)
_YEAR_PREP_CASE = {"в": "p", "во": "p", "при": "p", "о": "p", "об": "p", "на": "p", "к": "d", "ко": "d"}
_GEN_PREPS = frozenset({"с", "со", "до", "от", "после", "около", "для", "из", "без", "по"})
_YEAR_WORD_CASE = {"году": "p", "годе": "p", "года": "g", "годом": "i", "год": "n"}
_CASE_WORD = {"p": "году", "g": "года", "d": "году", "i": "годом", "n": "год"}
# a governing preposition before a small cardinal: "с 3 друзьями", "без 5 минут", "к 2 часам", "о 3 вещах"
_PREP_NUM = re.compile(
    r"\b(?P<prep>[Сс]о?|[Бб]ез|[Дд]о|[Оо]т|[Дд]ля|[Ии]з|[Уу]|[Пп]осле|[Оо]коло|[Кк]о?|[Нн]ад|[Пп]од|[Мм]ежду|[Оо]б?|[Пп]ри)\s+(?P<n>\d{1,2})(?!\d)(?![.,]\d)(?!\s?[%:/])(?![\-–—]\s?\d)(?=(?:\s+(?P<rest>[а-яёА-ЯЁ]+))?)"
)
_PREP_CASE = {"без": "g", "до": "g", "от": "g", "для": "g", "из": "g", "у": "g", "после": "g", "около": "g", "к": "d", "ко": "d", "над": "i", "под": "i", "между": "i", "о": "p", "об": "p", "при": "p"}
_INST_END = re.compile(r"(?:ами|ями|ом|ем|ёй|ой|ею|ью)$")
_GEN_END = re.compile(r"(?:ов|ев|ёв|ей|ых|их|[ьй]|[бвгджзклмнпрстфхцчшщ])$")
_CURRENCY = {
    "$": ("доллар", "доллара", "долларов", "m"), "€": ("евро", "евро", "евро", "m"), "¥": ("иена", "иены", "иен", "f"), "₽": ("рубль", "рубля", "рублей", "m"),
}
_CUR = re.compile(r"(?:(?P<pre>[$€¥₽])\s?(?P<a>\d+(?:[.,]\d+)?)|(?P<b>\d+(?:[.,]\d+)?)\s?(?P<post>[$€¥₽]))")
_PHONE = re.compile(r"(?<![\d\-])\d{2,4}(?:-\d{2,4}){1,}(?![\d\-])")
_SLASH = re.compile(r"(?<![\d/])(\d{1,2})/(\d{1,2})(?![\d/])")
_SCORE = re.compile(r"(?<![\d:])(\d{1,3}):(\d{1,3})(?![\d:])")
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
    if s in ("х", "их"):
        return _n(num, case="g")  # "4-х" = четырёх
    m = _n(num, to="ordinal")
    if s in ("ю", "ую"):
        return re.sub(r"яя$", "юю", re.sub(r"ая$", "ую", _n(num, to="ordinal", gender="f")))  # accusative feminine
    if s == "ей":
        return re.sub(r"(ый|ой)$", "ой", re.sub(r"ий$", "ьей", m))
    if s == "ые":
        return re.sub(r"(ый|ой)$", "ые", re.sub(r"ий$", "ьи", m))
    if s == "ых":
        return re.sub(r"(ый|ой)$", "ых", re.sub(r"ий$", "ьих", m))
    return _n(num)


def _year_sub(m: re.Match[str]) -> str:
    prep, y, w = m.group("prep"), int(m.group("y")), m.group("w").lower()
    p = (prep or "").lower()
    if w == "г.":
        case = _YEAR_PREP_CASE.get(p) or ("g" if p in _GEN_PREPS else "n")
    else:
        case = _YEAR_WORD_CASE[w]
        if w == "году" and p in ("к", "ко"):
            case = "d"
    word = "году" if case in ("p", "d") else {"g": "года", "i": "годом", "n": "год"}[case]
    num = _n(y, to="ordinal", case=case) if case != "n" else _n(y, to="ordinal")
    return (f"{prep} " if prep else "") + f"{num} {word}"


def _prep_num_sub(m: re.Match[str]) -> str:
    prep, n, rest = m.group("prep").lower(), int(m.group("n")), (m.group("rest") or "").lower()
    case = _PREP_CASE.get(prep)
    if prep in ("с", "со"):
        if rest in ("по", "до"):
            case = "g"
        elif _INST_END.search(rest):
            case = "i"
        elif rest.endswith(("ов", "ев", "ёв", "ей", "ых", "их")) or rest in ("раз", "лет", "минут", "часов"):
            case = "g"
    one = n % 10 == 1 and n % 100 != 11
    if case is None or n == 0 or (one and not (rest in ("до", "по", "") and case == "g")):
        return m.group(0)  # unsure ("один" needs a gender, known only before "до" / "по" / nothing): keep the nominative, the plain INT pass spells it
    if case == "i" and prep != "с" and prep != "со" and not _INST_END.search(rest):
        return m.group(0)
    return f"{m.group('prep')} {_n(n, case=case)}"


def _currency_sub(m: re.Match[str]) -> str:
    sym = m.group("pre") or m.group("post")
    raw = (m.group("a") or m.group("b")).replace(",", ".")
    one, few, many, g = _CURRENCY[sym]
    if "." in raw:
        return f"{_n(float(raw))} {few}"
    v = int(raw)
    return f"{_n(v, gender='f') if g == 'f' and v % 10 in (1, 2) and v % 100 not in (11, 12) else _n(v)} {_plural_form(v, one, few, many)}"


def _digit_groups(m: re.Match[str]) -> str:
    groups = m.group(0).split("-")
    date_like = len(groups) >= 3 and any(len(g) == 4 for g in groups)
    if len(groups) == 2 and not (len(groups[0]) == 3 and len(groups[1]) == 4):
        return m.group(0)  # a plain "10-20" / "1990-2000": a range
    out = []
    for g in groups:
        if len(g) <= 2 or (date_like and len(g) == 4):
            out.append(_n(int(g)))
        else:
            out.append(" ".join(_n(int(d)) for d in g))
    return ", ".join(out)


def normalize_numbers(text: str) -> str:
    t = text or ""
    t = _YEAR.sub(_keep_unreadable(_year_sub), t)
    t = _PHONE.sub(_keep_unreadable(_digit_groups), t)
    t = _MINUS.sub("минус ", t)
    t = _CUR.sub(_keep_unreadable(_currency_sub), t)
    t = _SLASH.sub(_keep_unreadable(lambda m: f"{_n(int(m.group(1)))} из {_n(int(m.group(2)), case='g')}"), t)
    t = _NO_PERIOD.sub(_keep_unreadable(lambda m: "номер " + _n(int(m.group(1)))), t)

    @_keep_unreadable
    def time_sub(m: re.Match[str]) -> str:
        h, mi = int(m.group(1)), int(m.group(2))
        if mi == 0:
            return f"{_n(h)} {_plural_form(h, 'час', 'часа', 'часов')}"
        return f"{_n(h)} {_n(mi) if mi >= 10 else 'ноль ' + _n(mi)}"

    t = _TIME.sub(time_sub, t)
    t = _SCORE.sub(_keep_unreadable(lambda m: f"{_n(int(m.group(1)))} — {_n(int(m.group(2)))}"), t)  # "2:1" is a score, not a clock

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
    t = _PREP_NUM.sub(_keep_unreadable(_prep_num_sub), t)
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
# Latin words -> Cyrillic (the voice models read Latin in English or spell it; CER then fails the take)
# ----------------------------------------------------------------------------

_TL_TITLE = re.compile(r"\b(Mr|Mrs|Ms|Dr)\b\.?", re.I)
_TL_TITLES = {"mr": "мистер", "mrs": "миссис", "ms": "миз", "dr": "доктор"}
_TL_EXC = {
    "ok": "окей", "okay": "окей", "wow": "вау", "bye": "бай", "hi": "хай", "yes": "йес", "no": "ноу", "vs": "против", "i": "ай",
    "you": "ю", "love": "лав", "hello": "хеллоу", "hey": "хей", "oh": "оу", "sorry": "сорри", "thanks": "сэнкс", "please": "плиз",
    "senpai": "сенпай", "sensei": "сэнсэй", "kun": "кун", "chan": "тян", "san": "сан", "sama": "сама", "iphone": "айфон",
    "minecraft": "майнкрафт", "youtube": "ютуб", "internet": "интернет", "online": "онлайн", "game": "гейм", "boss": "босс",
    "level": "левел", "love's": "лав", "the": "зе", "and": "энд", "of": "оф", "to": "ту", "my": "май", "me": "ми", "is": "из",
}
_TL_DIGRAPHS = {
    "tion": "шн", "tch": "ч", "igh": "ай", "sch": "ск", "sh": "ш", "ch": "ч", "th": "з", "ph": "ф", "ck": "к", "oo": "у", "ee": "и",
    "ea": "и", "ou": "ау", "ow": "ау", "ai": "эй", "ay": "эй", "oy": "ой", "oi": "ой", "qu": "кв", "kh": "х", "wh": "в", "ie": "и",
    "oa": "оу", "au": "о", "ew": "ью", "ng": "нг",
}
_TL_SINGLE = dict(zip("abcdefghijklmnopqrstuvwxyz", ["а", "б", "к", "д", "е", "ф", "г", "х", "и", "дж", "к", "л", "м", "н", "о", "п", "к", "р", "с", "т", "у", "в", "в", "кс", "и", "з"]))
_TL_WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")
_LAT_VOWEL_SET = set("aeiouy")


def _translit_word(lw: str) -> str:
    """Rule-based English-oriented approximation of one lowercase Latin word: pronounceable and deterministic, not exact."""
    w = lw
    if len(w) > 3 and w.endswith("e") and w[-2] not in _LAT_VOWEL_SET and any(c in _LAT_VOWEL_SET for c in w[:-2]):
        w = w[:-1]  # silent final e ("mike" -> "мик")
    out: list[str] = []
    i = 0
    while i < len(w):
        for n in (4, 3, 2):
            d = _TL_DIGRAPHS.get(w[i : i + n]) if len(w[i : i + n]) == n else None
            if d:
                out.append(d)
                i += n
                break
        else:
            c = w[i]
            nxt = w[i + 1] if i + 1 < len(w) else ""
            if c == "c" and nxt in ("e", "i", "y"):
                out.append("с")
            elif c == "e" and i == 0:
                out.append("э")
            elif c == "y" and nxt and nxt in _LAT_VOWEL_SET and (i == 0 or w[i - 1] in _LAT_VOWEL_SET):
                out.append("й")
            else:
                out.append(_TL_SINGLE.get(c, c))
            i += 1
    return "".join(out)


def _spell_caps(w: str) -> str:
    return "-".join(LATIN_LETTERS.get(c, c) for c in w.upper())


def _translit_token(w: str) -> str:
    lw = w.lower()
    if lw in _TL_EXC:
        res = _TL_EXC[lw]
    elif w.isupper() and 2 <= len(w) <= 6:
        return _spell_caps(w)
    elif len(w) == 1 and w.isupper():
        return LATIN_LETTERS.get(w, w).lower().capitalize()
    else:
        res = _translit_word(lw)
    return res[:1].upper() + res[1:] if w[:1].isupper() else res


def translit_latin(text: str) -> str:
    """Every Latin-letter word as a Cyrillic approximation ("Kevin" -> "Кевин", "Mr. Smith" -> "Мистер Смиз", "USB" -> "ю-эс-би").

    Cyrillic words, numbers and punctuation stay as they are. Titles (Mr, Mrs, Dr) take their period with them.
    """
    t = _TL_TITLE.sub(lambda m: (_TL_TITLES[m.group(1).lower()].capitalize() if m.group(1)[:1].isupper() else _TL_TITLES[m.group(1).lower()]), text or "")
    return _TL_WORD.sub(lambda m: _translit_token(m.group(0).replace("'", "").replace("’", "")), t)


# Ren'Py interpolations ("[player_name]", "[mc.name!t]"): the name is unknown when the audio is made, so it is not spoken
_INTERP = re.compile(r"\[[A-Za-z_][\w.!\[\]]*\]")
_ALNUM = re.compile(r"(?<![A-Za-z\d])([A-Za-z]{1,6})(\d{1,4})(?![A-Za-z\d])")
_DECOR = re.compile(r"[♪♫♬♥♡❤☆★♩]+")


def _strip_interpolations(text: str) -> str:
    from vndub.text import laughter as lg, sounds  # lazy: those modules do not import this one, but keep the import graph flat

    def drop(m: re.Match[str]) -> str:
        # "[laughs]" / "[yawns]" are stage directions the laughter / sounds modules own: leave them
        return m.group(0) if lg.find_laughs(m.group(0)) or sounds.find_sounds(m.group(0)) else "\x00"

    t = _INTERP.sub(drop, text)
    if "\x00" not in t:
        return t
    t = re.sub(r"^\s*\x00\s*,?\s*", "", t)  # "[name], иди" -> "иди"
    t = re.sub(r",?[ \t]*\x00", "", t)  # "Привет, [name]!" -> "Привет!"
    t = re.sub(r"([.!?…]\s*),\s*", r"\1", t)
    return t


def _translit_outside_sounds(t: str) -> str:
    """:func:`translit_latin` except on written laughs / sounds ("Hehe", "*yawns*", "Zzz"): synth respells those by its own tables."""
    if not re.search(r"[A-Za-z]", t):
        return t
    from vndub.text import laughter as lg, sounds

    spans = sorted((x.start, x.end) for x in [*lg.find_laughs(t), *sounds.find_sounds(t)])
    out: list[str] = []
    pos = 0
    for a, b in spans:
        if a < pos:
            continue
        out += [translit_latin(t[pos:a]), t[a:b]]
        pos = b
    out.append(translit_latin(t[pos:]))
    return "".join(out)


def _spell_alnum(m: re.Match[str]) -> str:
    letters, digits = m.group(1), m.group(2)
    if letters == "x":
        return f"икс {digits}"
    return f"{_translit_token(letters) if not letters.isupper() or len(letters) == 1 else _spell_caps(letters)} {digits}"


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


_RU_WORD = re.compile(r"(?:\+?[а-яёА-ЯЁ])+")


def to_acute_format(text: str) -> str:
    """'в+ыход' -> 'вы́ход', written as ``russian_text_stresser`` writes it, the stresser chatterbox-tts calls for Russian.

    That one takes its marks from Wiktionary's word forms, which have none on a word with one vowel or on ё: there the stress
    has nothing to choose. ruaccent marks both ('чт+о', sometimes), so those marks are dropped here.
    """

    def word(m: re.Match[str]) -> str:
        w = m.group(0).replace("+ё", "ё").replace("+Ё", "Ё")
        return w if sum(ch in VOWELS for ch in w) > 1 else w.replace("+", "")

    return plus_to_acute(_RU_WORD.sub(word, text or ""))


def graft_stress(original: str, stressed: str) -> str | None:
    """The marks of ``stressed`` put onto the words of ``original``, or None when the two do not have the same words.

    ruaccent rewrites more than the stress: it deletes straight quotes, apostrophes, ``*``, ``/``, ``%`` and any other
    character outside its small alphabet and closes the gaps around dashes and punctuation. A line with a ``"`` came
    back different, so the "a stresser must not change the words" guard threw the whole line's marks away. Here the
    original text stays as it is and only its Cyrillic words are replaced by the marked ones, if they are the same
    words in the same order (ё/е aside).
    """
    new = _RU_WORD.findall(stressed or "")
    old = _RU_WORD.findall(original or "")
    if len(new) != len(old) or any(deyo(strip_stress(a)).lower() != deyo(b).lower() for a, b in zip(new, old)):
        return None
    it = iter(new)
    return _RU_WORD.sub(lambda m: next(it), original)


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
    t = _strip_interpolations(t)
    t = _DECOR.sub(" ", t)
    t = re.sub(r"(?<=\w)~(?=\w)", "-", t).replace("~", "")  # a trailing "~" lengthens nothing for the TTS
    t = _ALNUM.sub(_spell_alnum, t)  # "PS5" -> "пи-эс 5", "x2" -> "икс 2"
    t = _YEAR.sub(_keep_unreadable(_year_sub), t)  # before the abbreviations: "в 2026 г." needs its preposition for the case
    t = expand_abbreviations(t)  # before numbers: "2026 г." needs the digit context
    t = normalize_numbers(t)
    t = _translit_outside_sounds(t)  # leftovers ("Kevin", "Smith"): never leave Latin for the TTS
    return _WS.sub(" ", t).strip()


def prepare_tts_text(
    text: str,
    stresser: Callable[[str], str] | None = None,
    fmt: str = "plain",
    yo: bool | str = True,
) -> tuple[str, str]:
    """(text_tts, text_plain). ``stresser`` maps normalized text -> '+'-stressed text.

    ``yo``: False takes every ё back to е; True and "safe" keep the ё the text has (what the stresser added is
    decided inside the stresser, see ``translate.stress.yo``).
    """
    norm = normalize_for_tts(text)
    stressed = norm
    if stresser is not None and norm:
        try:
            stressed = to_plus_format(stresser(norm) or norm)
        except Exception:
            stressed = norm
        # a stresser must not change the words; keep its marks on the original text if it changed more than the marks
        if deyo(strip_stress(stressed)).lower() != deyo(norm).lower():
            stressed = graft_stress(norm, stressed) or norm
    if not yo:
        stressed = deyo(stressed)
    text_tts = to_plus_format(stressed) if fmt == "plus" else strip_stress(stressed)
    text_plain = strip_stress(stressed)
    return text_tts, text_plain

"""Source-language detection and language-aware text measures.

Visual novels are not always Japanese: the pipeline must know the language of the script text
(``project.src_lang``) to build the translation prompt, to pick the interjection lexicon and the
narration cues, to measure how long a line should take to say and to choose an ASR / aligner that
understands the voices.

:func:`detect_language` needs no model. It counts the letters of the script lines by Unicode
script -- kana, han (kanji / hanzi), hangul, Cyrillic, Latin -- and decides from the shares:
kana among the CJK letters means Japanese, han without kana Chinese, hangul Korean, Cyrillic
Russian (Ukrainian when its own letters і ї є ґ appear), Latin one of en / de / fr / es / it / pt by
a small stop-word vote (English when nothing wins). The confidence is the share of the winning
script among all letters, scaled down for very short samples.

:func:`speech_units` is the length of a line in *mora equivalents*: exact morae for Japanese
(:func:`vndub.text.ja.count_morae`), otherwise an estimated syllable count scaled by
``SYLLABLE_TO_MORA`` so that the link stage's plausible speaking-rate window (``link.rate_min`` /
``rate_max``, morae per second) applies to every language.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable

from vndub.text import ja

AUTO = "auto"

LANG_NAMES = {
    "ja": "Japanese", "en": "English", "ru": "Russian", "zh": "Chinese", "ko": "Korean", "uk": "Ukrainian",
    "de": "German", "fr": "French", "es": "Spanish", "it": "Italian", "pt": "Portuguese",
}

#: languages whose script has no word spaces (matching / rate measures treat characters as units)
CJK_LANGS = frozenset({"ja", "zh", "ko"})

#: a syllable of a spaced language lasts about one and a half Japanese morae at a comparable pace
SYLLABLE_TO_MORA = 1.5

# stop words (lower case, whole words) that separate the Latin-script languages a VN may be in
_STOPWORDS = {
    "en": {"the", "and", "you", "that", "this", "with", "what", "have", "not", "for", "are", "but", "was", "your", "it's", "don't", "just", "like", "know", "i'm", "we", "he", "she", "they", "there"},
    "de": {"und", "der", "die", "das", "nicht", "ich", "ist", "sie", "ein", "eine", "mit", "auf", "dass", "wir", "ihr", "auch", "noch", "wie", "aber", "was", "du", "es", "den", "dem", "sich"},
    "fr": {"les", "des", "une", "est", "pas", "que", "qui", "vous", "nous", "avec", "dans", "pour", "mais", "c'est", "je", "tu", "ça", "bien", "sur", "même", "le", "la", "et", "ne", "un"},
    "es": {"que", "los", "las", "una", "por", "para", "con", "pero", "como", "está", "qué", "esto", "muy", "tiene", "hay", "porque", "usted", "ella", "nada", "sí", "el", "yo", "es", "no", "de"},
    "it": {"che", "non", "una", "per", "con", "come", "sono", "questo", "anche", "cosa", "ma", "della", "del", "ho", "sei", "perché", "molto", "bene", "ci", "tu", "il", "di", "è", "un", "la"},
    "pt": {"que", "não", "uma", "para", "com", "mas", "como", "você", "isso", "está", "muito", "ele", "ela", "bem", "por", "mais", "meu", "sua", "já", "então", "de", "eu", "um", "os", "se"},
}
_UK_LETTERS = set("іїєґІЇЄҐ")
_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)
_VOWELS = set("aeiouyаеёиоуыэюяіїєөүұ")


def language_name(code: str) -> str:
    return LANG_NAMES.get((code or "").lower().split("-")[0], code)


def _script(ch: str) -> str | None:
    o = ord(ch)
    if 0x3041 <= o <= 0x30FF or 0x31F0 <= o <= 0x31FF or 0xFF66 <= o <= 0xFF9F:
        return "kana"
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF or 0x20000 <= o <= 0x2FA1F or ch in "々〆ヶ":
        return "han"
    if 0xAC00 <= o <= 0xD7AF or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F:
        return "hangul"
    if 0x0400 <= o <= 0x052F:
        return "cyrillic"
    if o < 0x0250 or 0x1E00 <= o <= 0x1EFF:
        return "latin" if ch.isalpha() else None
    return "other" if ch.isalpha() else None


@dataclass
class LangGuess:
    lang: str
    confidence: float
    letters: int = 0
    scripts: dict[str, int] = field(default_factory=dict)
    lines: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"lang": self.lang, "confidence": round(self.confidence, 3), "letters": self.letters, "lines": self.lines, "scripts": dict(self.scripts)}


def _clean(text: str) -> str:
    return unicodedata.normalize("NFKC", ja.strip_markup(ja.strip_ruby(text or "")))


def detect_language(texts: Iterable[str], max_lines: int = 4000, min_letters: int = 40) -> LangGuess:
    """Guess the language of the script lines (see the module docstring).

    ``max_lines`` lines are sampled evenly over the input; fewer than ``min_letters`` letters scale the
    confidence down. An empty input gives ``LangGuess("", 0.0)``.
    """
    rows = [t for t in texts if t and t.strip()]
    if len(rows) > max_lines:
        step = len(rows) / max_lines
        rows = [rows[int(i * step)] for i in range(max_lines)]
    counts: dict[str, int] = {"kana": 0, "han": 0, "hangul": 0, "cyrillic": 0, "latin": 0, "other": 0}
    words: list[str] = []
    uk = 0
    for raw in rows:
        t = _clean(raw)
        had_latin = counts["latin"]
        for ch in t:
            s = _script(ch)
            if s is None:
                continue
            counts[s] += 1
            if s == "cyrillic" and ch in _UK_LETTERS:
                uk += 1
        if counts["latin"] > had_latin and len(words) < 20000:
            words.extend(w.lower() for w in _WORD.findall(t) if w.isascii())
    letters = sum(counts.values())
    if letters == 0:
        return LangGuess("", 0.0, 0, counts, len(rows))
    cjk = counts["kana"] + counts["han"]
    share = {k: v / letters for k, v in counts.items()}
    if cjk and counts["kana"] / cjk >= 0.1 and cjk + counts["hangul"] >= counts["latin"] + counts["cyrillic"]:
        lang, conf = "ja", share["kana"] + share["han"]
    elif counts["hangul"] and share["hangul"] >= max(share["latin"], share["cyrillic"], share["han"]):
        lang, conf = "ko", share["hangul"]
    elif cjk and cjk >= counts["latin"] + counts["cyrillic"]:
        lang, conf = "zh", share["han"]
    elif counts["cyrillic"] >= counts["latin"]:
        lang, conf = ("uk" if uk / max(1, counts["cyrillic"]) >= 0.01 else "ru"), share["cyrillic"]
    else:
        lang, conf = _latin_language(words), share["latin"]
        if not words:
            conf *= 0.8
    if letters < min_letters:
        conf *= letters / min_letters
    return LangGuess(lang, round(min(1.0, conf), 3), letters, counts, len(rows))


def _latin_language(words: list[str]) -> str:
    if not words:
        return "en"
    hits = {lang: sum(1 for w in words if w in sw) for lang, sw in _STOPWORDS.items()}
    best = max(hits, key=lambda k: hits[k])
    ranked = sorted(hits.values(), reverse=True)
    if hits[best] == 0 or (best != "en" and ranked[0] < 1.3 * ranked[1]):
        return "en"
    return best


def count_syllables(text: str, lang: str = "en") -> int:
    """Estimated syllables of a line in a spaced language: vowel groups per word (at least one per word),
    one per han / hangul character, two per digit. Japanese is measured in morae by :mod:`vndub.text.ja`."""
    if lang == "ja":
        return ja.count_morae(text)
    t = _clean(text)
    n = 0.0
    word: list[str] = []

    def flush() -> None:
        nonlocal n
        if word:
            w = "".join(word)
            base = "".join(ch for ch in unicodedata.normalize("NFD", w) if not unicodedata.combining(ch)).lower()
            groups = 0
            prev = False
            for ch in base:
                v = ch in _VOWELS
                if v and not prev:
                    groups += 1
                prev = v
            n += max(1, groups)
            word.clear()

    for ch in t:
        s = _script(ch)
        if s in ("latin", "cyrillic", "other"):
            word.append(ch)
            continue
        flush()
        if s == "kana":
            n += 0 if ch in ja._SMALL_KANA else 1
        elif s in ("han", "hangul"):
            n += 1
        elif ch.isdigit():
            n += 2
    flush()
    return int(round(n))


def speech_units(text: str, lang: str = "ja") -> int:
    """Length of a line in mora equivalents (the unit of ``link.rate_min`` / ``rate_max``)."""
    if not lang or lang in ("ja", AUTO):
        return ja.count_morae(text)
    return int(round(count_syllables(text, lang) * SYLLABLE_TO_MORA))

"""Japanese text helpers: normalization for fuzzy matching, mora counting, interjections.

``normalize_for_match`` is applied identically to ASR output and script text
before RapidFuzz comparison: NFKC, ruby/markup removal, katakana -> hiragana
(ASR and scripts disagree on script choice more often than on sounds),
lowercase, and removal of punctuation, symbols, whitespace, ellipses and quotes.

``count_morae`` counts kana morae exactly (small ゃゅょぁぃぅぇぉゎ merge with
the previous kana; っ, ー and ん count) and *estimates* the rest: a kanji
counts 2 morae (the average on-/kun-reading length in dialogue), a Latin word
counts its vowel groups, a digit counts 2.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Mapping

import yaml

# |漢字《かんじ》 (Aozora) and 漢字《かんじ》
_RUBY_AOZORA = re.compile(r"[|｜]([^《|｜]+)《[^》]*》")
_RUBY_BARE = re.compile(r"《[^》]*》")
# Ren'Py {rt}reading{/rt}, KAG [ruby text="..."], generic [...] tags, Ren'Py {...} tags
_RUBY_RENPY = re.compile(r"\{rt\}.*?\{/rt\}")
_TAG_BRACKET = re.compile(r"\[[^\]\n]*\]")
_TAG_BRACE = re.compile(r"\{[^}\n]*\}")
_WS = re.compile(r"\s+")

_SMALL_KANA = set("ゃゅょぁぃぅぇぉゎャュョァィゥェォヮ")


def strip_ruby(text: str) -> str:
    """Remove ruby readings, keeping the base text."""
    t = _RUBY_AOZORA.sub(r"\1", text or "")
    t = _RUBY_BARE.sub("", t)
    t = _RUBY_RENPY.sub("", t)
    return t


def strip_markup(text: str) -> str:
    """Remove KAG ``[tags]`` and Ren'Py ``{tags}`` (after ruby readings were dropped)."""
    t = _TAG_BRACKET.sub("", text or "")
    return _TAG_BRACE.sub("", t)


def kata_to_hira(text: str) -> str:
    out = []
    for ch in text:
        o = ord(ch)
        # ァ (U+30A1) .. ヶ (U+30F6) -> ぁ .. ゖ
        if 0x30A1 <= o <= 0x30F6:
            out.append(chr(o - 0x60))
        else:
            out.append(ch)
    return "".join(out)


def _is_drop(ch: str) -> bool:
    if ch.isspace():
        return True
    cat = unicodedata.category(ch)
    # P* punctuation, S* symbols (♪, ～ after NFKC is ~), Z* separators, C* control/format
    return cat[0] in ("P", "S", "Z", "C")


def normalize_for_match(text: str) -> str:
    t = strip_markup(strip_ruby(text or ""))
    t = unicodedata.normalize("NFKC", t)
    t = kata_to_hira(t).lower()
    # long-vowel mark and its NFKC variants are kept (they are sounds); wave dash dropped
    return "".join(ch for ch in t if not _is_drop(ch) or ch == "ー")


def _is_kana(ch: str) -> bool:
    o = ord(ch)
    return 0x3041 <= o <= 0x309F or 0x30A0 <= o <= 0x30FF or ch == "ー"


def _is_kanji(ch: str) -> bool:
    o = ord(ch)
    return 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF or ch in "々〆ヶ"


_LATIN_VOWELS = re.compile(r"[aeiouy]+", re.I)


def count_morae(text: str) -> int:
    t = unicodedata.normalize("NFKC", strip_markup(strip_ruby(text or "")))
    n = 0.0
    latin: list[str] = []

    def flush_latin() -> None:
        nonlocal n
        if latin:
            word = "".join(latin)
            n += max(1, len(_LATIN_VOWELS.findall(word)))
            latin.clear()

    for ch in t:
        if ch.isascii() and ch.isalpha():
            latin.append(ch)
            continue
        flush_latin()
        if _is_kana(ch):
            if ch not in _SMALL_KANA:
                n += 1
        elif _is_kanji(ch):
            n += 2
        elif ch.isdigit():
            n += 2
    flush_latin()
    return int(round(n))


def is_kana_only(text: str) -> bool:
    t = normalize_for_match(text)
    return bool(t) and all(_is_kana(ch) or ch in "ー" for ch in t)


# ----------------------------------------------------------------------------
# interjections
# ----------------------------------------------------------------------------


def load_interjections(path: str | Path | None) -> dict[str, str]:
    """``configs/interjections.yaml`` -> {japanese: russian}. Missing file -> {}."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {str(k): str(v) for k, v in data.items() if k is not None and v is not None}


def _exact_key(text: str) -> str:
    """NFKC + strip whitespace, ellipses, quotes and sentence punctuation except '?'."""
    t = unicodedata.normalize("NFKC", strip_markup(strip_ruby(text or ""))).strip()
    t = t.replace("?", "？")
    keep = []
    for ch in t:
        if ch == "？":
            keep.append(ch)
        elif ch.isspace() or unicodedata.category(ch)[0] in ("P", "S", "Z"):
            continue
        else:
            keep.append(ch)
    return "".join(keep)


def _ends_with_question(text: str) -> bool:
    t = unicodedata.normalize("NFKC", text or "").rstrip(" …。.!！~～ー　")
    return t.endswith("?") or t.endswith("？")


def match_interjection(text_src: str, lexicon: Mapping[str, str], max_morae: int = 2) -> str | None:
    """Target-language equivalent of an interjection line, or None.

    Lookup order: exact key (with '?'), key without punctuation, then a
    repeated interjection (``うんうん`` -> ``угу-угу`` style when only ``うん``
    is known). A question mark in the source is carried to the result.
    """
    if not text_src or not lexicon:
        return None
    exact = {_exact_key(k): v for k, v in lexicon.items()}
    plain = {normalize_for_match(k): v for k, v in lexicon.items()}
    ek = _exact_key(text_src)
    pk = normalize_for_match(text_src)
    if not pk:
        return None
    val = exact.get(ek)
    if val is None:
        val = plain.get(pk)
    if val is None:
        # repetition of a known interjection: "うんうんうん"
        for k, v in plain.items():
            if k and len(pk) % len(k) == 0 and pk == k * (len(pk) // len(k)) and len(pk) // len(k) <= 3:
                val = "-".join([v.rstrip("?!")] * (len(pk) // len(k)))
                break
    if val is None:
        return None
    if _ends_with_question(text_src) and not val.endswith("?"):
        val = val.rstrip("!.") + "?"
    return val


def is_interjection(text: str, lexicon: Mapping[str, str] | None = None, max_morae: int = 2) -> bool:
    """True for lexicon interjections and for very short kana-only lines (<= max_morae)."""
    if match_interjection(text, lexicon or {}, max_morae) is not None:
        return True
    return is_kana_only(text) and count_morae(text) <= max_morae

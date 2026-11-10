"""Character error rate of a candidate against the stress-free target text.

Both sides are normalized identically: stress marks (``+`` and U+0301)
removed, ё -> е, lowercase, punctuation dropped, whitespace collapsed.
"""

from __future__ import annotations

import re
import unicodedata

_STRESS = re.compile("[+́]")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize_for_cer(text: str) -> str:
    t = unicodedata.normalize("NFC", text or "")
    t = _STRESS.sub("", t)
    t = t.replace("ё", "е").replace("Ё", "Е").lower()
    t = _PUNCT.sub(" ", t)
    t = t.replace("_", " ")
    return _WS.sub(" ", t).strip()


_LEXICON: dict[str, tuple[str, int]] | None = None


def _number_lexicon() -> dict[str, tuple[str, int]]:
    """Every Russian number word form (cardinal and ordinal, all cases and genders, as :func:`normalize_for_cer` writes
    it) -> (kind, value): kind ``add`` for 0..900, ``mul`` for thousand / million / billion. Built from num2words once."""
    global _LEXICON
    if _LEXICON is not None:
        return _LEXICON
    from num2words import num2words

    lex: dict[str, tuple[str, int]] = {}
    adds = [*range(20), *range(20, 100, 10), *range(100, 1000, 100)]
    muls = [1000, 10**6, 10**9]

    def forms(n: int) -> set[str]:
        out: set[str] = set()
        for kw in [{}, *({"case": c} for c in ("g", "d", "a", "i", "p")), *({"gender": g} for g in ("f", "n"))]:
            for to in ("cardinal", "ordinal"):
                for extra in ({}, {"plural": True}) if to == "ordinal" else ({},):
                    try:
                        out.add(normalize_for_cer(num2words(n, lang="ru", to=to, **kw, **extra)))
                    except Exception:  # an older num2words without that case / gender: the nominative still counts
                        continue
                    for case in ("g", "d", "i", "p") if kw.get("gender") else ():
                        try:
                            out.add(normalize_for_cer(num2words(n, lang="ru", to=to, case=case, **kw, **extra)))
                        except Exception:
                            continue
        return out

    for n in adds:
        for f in forms(n):
            if f and " " not in f:
                lex.setdefault(f, ("add", n))
    for m in muls:
        for k in (1, 2, 5):
            for f in forms(m * k):
                w = f.split()[-1] if f else ""
                if w and w not in lex:
                    lex[w] = ("mul", m)
    for w in ("нуль", "нуля", "нулем"):
        lex.setdefault(w, ("add", 0))
    for w in ("одну", "одной", "одною"):
        lex.setdefault(w, ("add", 1))
    _LEXICON = lex
    return lex


def numbers_to_digits(text: str) -> str:
    """Runs of Russian number words in a normalized text -> one digit token each ("с тремя друзьями" -> "с 3 друзьями",
    "в девяносто девятом году" -> "в 99 году"): the reference spells a number in the case the sentence needs, the ASR
    writes digits that expand to the nominative, so both sides are compared by value, not by ending."""
    lex = _number_lexicon()
    out: list[str] = []
    total = cur = 0
    limit: int | None = None  # the largest additive part the running number can still take; None = no number running
    mul_limit = 10**12

    def flush() -> None:
        nonlocal total, cur, limit, mul_limit
        if limit is not None:
            out.append(str(total + cur))
        total = cur = 0
        limit, mul_limit = None, 10**12

    for w in text.split():
        hit = lex.get(w)
        if hit is None:
            flush()
            out.append(w)
            continue
        kind, v = hit
        if kind == "mul":
            if limit is not None and v >= mul_limit:
                flush()
            total += max(cur, 1) * v
            cur, limit, mul_limit = 0, v, v
            continue
        if limit is not None and v >= limit:
            flush()
        if limit is None:
            limit = 10**12
        cur += v
        limit = 100 if v >= 100 else (10 if v >= 20 and v % 10 == 0 else 0)
        if v == 0:
            limit = 0
    flush()
    return " ".join(out)


def cer(reference: str, hypothesis: str) -> float:
    import jiwer

    ref = normalize_for_cer(reference)
    hyp = normalize_for_cer(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    if not hyp:
        return 1.0
    return float(min(1.0, jiwer.cer(ref, hyp)))

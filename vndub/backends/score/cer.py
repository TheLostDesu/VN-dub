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


def cer(reference: str, hypothesis: str) -> float:
    import jiwer

    ref = normalize_for_cer(reference)
    hyp = normalize_for_cer(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    if not hyp:
        return 1.0
    return float(min(1.0, jiwer.cer(ref, hyp)))

"""Metric -> one-word verdict mapping for the review screen (spec Step 6).

The review UI never shows raw metrics: every candidate gets up to five *verdicts*

======== ===================================== ============================ ==========================
metric   source metric(s)                       keys (i18n verdict.<m>.<k>)  levels
======== ===================================== ============================ ==========================
voice    ``spk_sim`` vs the spk gate           similar / close / different  good / ok / bad
emotion  composite emotion term (emo_sim+adv)  close / partly / far         good / ok / bad
clarity  ``cer`` vs the cer gate               good / ok / poor             good / ok / bad
length   ``dur_ratio`` vs the dur gate         ok / longish / too_long /    good|ok / ok / bad
                                               shortish / too_short
quality  ``utmos`` vs the utmos gate           good / ok / poor             good / ok / bad
======== ===================================== ============================ ==========================

The thresholds are the project's score gates (``score.gates.*``, calibrated speaker
threshold, ``interjection_cer_max``) so a "bad" verdict and a failed gate always agree;
the only extra constants are the "comfortably good" bands, defined relative to the gate
(see :class:`Thresholds`). The numbers stay in ``value`` / ``tooltip_params`` for the tooltip.

:func:`why_items` explains a line that needs attention in terms of the failing gates
(i18n ``why.<key>``): ``cer``, ``utmos``, ``spk_sim`` (``spk_sim_performer`` for lines without original
audio), ``dur_ratio``, ``text_from_asr``, ``no_candidates``, ``override``, ``translation_failed`` and
``pending_change`` (added by ``reviewdata``).

Pure functions: no pipeline imports, so this module is cheap to import and unit-tested.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

LEVEL_GOOD, LEVEL_OK, LEVEL_BAD = "good", "ok", "bad"

# emotion verdict has no gate in score.py; these bands are on the same 0..1 term the composite uses
EMOTION_CLOSE = 0.75
EMOTION_PARTLY = 0.55
# length: how far beyond the gate a ratio may be before it is "too_*" instead of "-ish"
LENGTH_BAND = 1.20
# "comfortably good" bands relative to the gates
UTMOS_GOOD_MARGIN = 0.5
CER_GOOD_FRACTION = 0.5

METRICS = ("voice", "emotion", "clarity", "length", "quality")


@dataclass(frozen=True)
class Thresholds:
    """Gate values of the project config (the same ones ``score`` applied)."""

    utmos_min: float = 3.0
    cer_max: float = 0.10
    cer_max_interjection: float = 0.5
    dur_lo: float = 0.8
    dur_hi: float = 1.25
    spk_min: float = 0.25
    target_window: tuple[float, float] = (0.9, 1.1)

    @property
    def spk_good(self) -> float:
        """Speaker similarity that is clearly above the gate."""
        return self.spk_min + max(0.05, 0.25 * (1.0 - self.spk_min))

    @classmethod
    def from_config(cls, cfg: Any, calibration: dict[str, Any] | None = None) -> "Thresholds":
        """Read the gates from a ``PipelineConfig`` (mirrors ``score.spk_threshold`` for the speaker gate)."""
        g = cfg.score.gates
        if isinstance(g.spk_sim_min, (int, float)):
            spk = float(g.spk_sim_min)
        elif calibration and calibration.get("threshold") is not None:
            spk = float(calibration["threshold"])
        else:
            spk = float(g.spk_sim_default)
        win = tuple(cfg.master.stretch.target_window)
        return cls(
            utmos_min=float(g.utmos_min),
            cer_max=float(g.cer_max),
            cer_max_interjection=float(cfg.score.interjection_cer_max),
            dur_lo=float(g.dur_ratio[0]),
            dur_hi=float(g.dur_ratio[1]),
            spk_min=spk,
            target_window=(float(win[0]), float(win[1])),
        )


def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _r(x: float | None, n: int = 3) -> float | None:
    return None if x is None else round(float(x), n)


# ----------------------------------------------------------------------------
# single verdicts
# ----------------------------------------------------------------------------


def voice_verdict(spk_sim: Any, th: Thresholds, reference: str = "original") -> dict[str, Any] | None:
    v = _num(spk_sim)
    if v is None:
        return None
    if v >= th.spk_good:
        key, level = "similar", LEVEL_GOOD
    elif v >= th.spk_min:
        key, level = "close", LEVEL_OK
    else:
        key, level = "different", LEVEL_BAD
    return {
        "metric": "voice", "key": key, "level": level, "value": _r(v),
        "tooltip_params": {"spk_sim": _r(v), "threshold": _r(th.spk_min), "good_from": _r(th.spk_good), "reference": reference},
    }


def emotion_term(emo_sim: Any, adv_delta: Any) -> float | None:
    """The emotion term of the composite (``score.terms``): 0.75 * emo_sim + 0.25 * (1 - adv_delta)."""
    emo, adv = _num(emo_sim), _num(adv_delta)
    clip = lambda x: min(1.0, max(0.0, x))  # noqa: E731
    if emo is not None and adv is not None:
        return 0.75 * clip(emo) + 0.25 * (1.0 - clip(adv))
    if emo is not None:
        return clip(emo)
    if adv is not None:
        return 1.0 - clip(adv)
    return None


def emotion_verdict(emo_sim: Any, adv_delta: Any) -> dict[str, Any] | None:
    t = emotion_term(emo_sim, adv_delta)
    if t is None:
        return None
    if t >= EMOTION_CLOSE:
        key, level = "close", LEVEL_GOOD
    elif t >= EMOTION_PARTLY:
        key, level = "partly", LEVEL_OK
    else:
        key, level = "far", LEVEL_BAD
    return {
        "metric": "emotion", "key": key, "level": level, "value": _r(t),
        "tooltip_params": {"emo_sim": _r(_num(emo_sim)), "adv_delta": _r(_num(adv_delta)), "term": _r(t), "close_from": EMOTION_CLOSE, "partly_from": EMOTION_PARTLY},
    }


def clarity_verdict(cer: Any, th: Thresholds, interjection: bool = False) -> dict[str, Any] | None:
    v = _num(cer)
    if v is None:
        return None
    cmax = th.cer_max_interjection if interjection else th.cer_max
    if v <= cmax * CER_GOOD_FRACTION:
        key, level = "good", LEVEL_GOOD
    elif v <= cmax:
        key, level = "ok", LEVEL_OK
    else:
        key, level = "poor", LEVEL_BAD
    return {"metric": "clarity", "key": key, "level": level, "value": _r(v), "tooltip_params": {"cer": _r(v), "max": _r(cmax)}}


def length_verdict(dur_ratio: Any, th: Thresholds) -> dict[str, Any] | None:
    r = _num(dur_ratio)
    if r is None or r <= 0:
        return None
    lo, hi = th.dur_lo, th.dur_hi
    w_lo, w_hi = th.target_window
    if r > hi:
        key, level = ("longish", LEVEL_OK) if r <= hi * LENGTH_BAND else ("too_long", LEVEL_BAD)
    elif r < lo:
        key, level = ("shortish", LEVEL_OK) if r >= lo / LENGTH_BAND else ("too_short", LEVEL_BAD)
    else:
        key, level = "ok", (LEVEL_GOOD if w_lo <= r <= w_hi else LEVEL_OK)
    return {"metric": "length", "key": key, "level": level, "value": _r(r), "tooltip_params": {"ratio": _r(r), "min": _r(lo), "max": _r(hi)}}


def quality_verdict(utmos: Any, th: Thresholds) -> dict[str, Any] | None:
    v = _num(utmos)
    if v is None:
        return None
    if v >= th.utmos_min + UTMOS_GOOD_MARGIN:
        key, level = "good", LEVEL_GOOD
    elif v >= th.utmos_min:
        key, level = "ok", LEVEL_OK
    else:
        key, level = "poor", LEVEL_BAD
    return {"metric": "quality", "key": key, "level": level, "value": _r(v), "tooltip_params": {"utmos": _r(v), "min": _r(th.utmos_min)}}


def verdicts_for(metrics: dict[str, Any], th: Thresholds, *, route: str = "", interjection: bool = False, reference: str = "original") -> list[dict[str, Any]]:
    """All available verdicts of one candidate, in the order voice, emotion, clarity, length, quality.

    A metric that was not measured (mode-B lines have no emotion/voice reference, ``orig``
    candidates have no CER) simply produces no verdict. ``reference`` says what the voice verdict
    compares with: ``original`` (the character's own voice) or ``performer`` (lines without original
    audio: consistency with the chosen performer's sample; the tooltip must not say "like the original").
    """
    out = [
        voice_verdict(metrics.get("spk_sim"), th, reference),
        emotion_verdict(metrics.get("emo_sim"), metrics.get("adv_delta")),
        None if route == "orig" else clarity_verdict(metrics.get("cer"), th, interjection),
        None if route == "orig" else length_verdict(metrics.get("dur_ratio"), th),
        quality_verdict(metrics.get("utmos"), th),
    ]
    return [v for v in out if v is not None]


def worst_level(verdicts: Iterable[dict[str, Any]]) -> str:
    order = {LEVEL_GOOD: 0, LEVEL_OK: 1, LEVEL_BAD: 2}
    return max((v["level"] for v in verdicts), key=lambda l: order.get(l, 0), default=LEVEL_GOOD)


# ----------------------------------------------------------------------------
# "why is this line here"
# ----------------------------------------------------------------------------


def why_items(
    *,
    metrics: dict[str, Any] | None,
    gates: dict[str, bool] | None,
    th: Thresholds,
    interjection: bool = False,
    has_candidates: bool = True,
    selection_status: str | None = None,
    selection_reason: str = "",
    line_flags: Iterable[str] = (),
    translation_failed: bool = False,
    voice_reference: str = "original",
) -> list[dict[str, Any]]:
    """Failing gates (and other reasons) of the line's current best candidate as ``[{gate, key, params}]``.

    ``gates`` is the persisted gate map of that candidate (``scores/<id>.jsonl``); the
    thresholds in ``params`` come from the project config so the sentence can say
    "CER 0.23 is above the allowed 0.10".
    """
    items: list[dict[str, Any]] = []
    m = metrics or {}
    if "text_from_asr" in set(line_flags):
        items.append({"gate": "source", "key": "text_from_asr", "params": {}})
    if translation_failed:
        items.append({"gate": "translation", "key": "translation_failed", "params": {}})
    if selection_status == "override":
        items.append({"gate": "review", "key": "override", "params": {"detail": selection_reason}})
    if not has_candidates:
        items.append({"gate": "synth", "key": "no_candidates", "params": {"detail": selection_reason}})
        return items
    g = gates or {}
    if g.get("cer") is False and _num(m.get("cer")) is not None:
        items.append({"gate": "cer", "key": "cer", "params": {"value": _r(_num(m["cer"])), "max": _r(th.cer_max_interjection if interjection else th.cer_max)}})
    if g.get("utmos") is False and _num(m.get("utmos")) is not None:
        items.append({"gate": "utmos", "key": "utmos", "params": {"value": _r(_num(m["utmos"])), "min": _r(th.utmos_min)}})
    if g.get("spk_sim") is False and _num(m.get("spk_sim")) is not None:
        key = "spk_sim" if voice_reference == "original" else "spk_sim_performer"
        items.append({"gate": "spk_sim", "key": key, "params": {"value": _r(_num(m["spk_sim"])), "min": _r(th.spk_min)}})
    if g.get("dur") is False and _num(m.get("dur_ratio")) is not None:
        items.append({"gate": "dur", "key": "dur_ratio", "params": {"value": _r(_num(m["dur_ratio"])), "min": _r(th.dur_lo), "max": _r(th.dur_hi)}})
    return items

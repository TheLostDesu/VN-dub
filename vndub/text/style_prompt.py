"""Expression passport -> :class:`StyleSpec` -> per-backend style controls.

Rules (from the design):

* emotion   = argmax of 0.6 * emotion2vec + 0.4 * anime-SER (anime labels mapped
  to the e2v set; ``embarrassed`` kept as an extra class);
* intensity = arousal (A/D/V); falls back to a probability-derived proxy;
* tempo     = mora rate relative to the character median (+-15% -> fast/slow);
* loudness  = mean energy relative to the character median (+-3 dB);
* arc       = slope of the last voiced third of the F0 contour;
* events    = SenseVoice tags + heuristics (whisper = low energy + flat tilt).

Renderers turn a StyleSpec into what each backend understands: a natural
language instruction (Qwen3-TTS CustomVoice ``instruct``, CosyVoice3 instruct
before ``<|endofprompt|>``, VoxCPM2 parenthesized style), inline tags (Fish S2,
Higgs), or scalar knobs (Chatterbox ``exaggeration``, Zonos2 sliders), plus the
emotion reference clip choice.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

E2V_SET = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown"]
EXTRA = ["embarrassed"]
ALL_EMOTIONS = E2V_SET + EXTRA

ANIME_TO_E2V = {
    "angry": "angry", "disgusted": "disgusted", "embarrassed": "embarrassed", "fearful": "fearful",
    "happy": "happy", "sad": "sad", "surprised": "surprised", "neutral": "neutral",
    "sexual1": "other", "sexual2": "other",
}

# rough valence/arousal anchors used for nearest-emotion fallbacks
EMOTION_VA = {
    "angry": (0.2, 0.85), "disgusted": (0.25, 0.55), "fearful": (0.2, 0.75), "happy": (0.85, 0.7),
    "neutral": (0.5, 0.35), "other": (0.5, 0.5), "sad": (0.2, 0.25), "surprised": (0.65, 0.8),
    "unknown": (0.5, 0.4), "embarrassed": (0.45, 0.55),
}

_ADJ = {
    "angry": "angry", "disgusted": "disgusted", "fearful": "frightened", "happy": "happy",
    "neutral": "calm, neutral", "other": "expressive", "sad": "sad", "surprised": "surprised",
    "unknown": "natural", "embarrassed": "embarrassed, flustered",
}

TEMPO_THRESHOLD = 0.15
LOUDNESS_DB = 3.0


@dataclass
class StyleSpec:
    emotion: str = "neutral"
    intensity: float = 0.5
    tempo: str = "normal"  # slow | normal | fast
    loudness: str = "normal"  # soft | normal | loud
    events: list[str] = field(default_factory=list)
    voice_quality: list[str] = field(default_factory=list)
    arc: str = "flat"  # rising_end | falling_end | flat
    notes: str = ""
    valence: float | None = None
    dominance: float | None = None
    confidence: float = 0.0
    caption: str = ""
    emotion_probs: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "StyleSpec":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)

    @property
    def is_high_arousal(self) -> bool:
        return self.intensity >= 0.75


# ----------------------------------------------------------------------------
# building
# ----------------------------------------------------------------------------


def _norm(probs: Mapping[str, float]) -> dict[str, float]:
    tot = sum(max(0.0, float(v)) for v in probs.values())
    if tot <= 0:
        return {}
    return {k: max(0.0, float(v)) / tot for k, v in probs.items()}


def emotion_vote(
    e2v_probs: Mapping[str, float] | None,
    anime_probs: Mapping[str, float] | None,
    w_e2v: float = 0.6,
    w_anime: float = 0.4,
) -> tuple[str, dict[str, float]]:
    """Weighted vote over the e2v label set (+ embarrassed). Returns (label, combined probs)."""
    combined = {k: 0.0 for k in ALL_EMOTIONS}
    e = _norm(e2v_probs or {})
    a_raw = _norm(anime_probs or {})
    a: dict[str, float] = {}
    for k, v in a_raw.items():
        m = ANIME_TO_E2V.get(k.lower())
        if m:
            a[m] = a.get(m, 0.0) + v
    if not e and not a:
        return "neutral", {"neutral": 1.0}
    we = w_e2v if e else 0.0
    wa = w_anime if a else 0.0
    s = we + wa
    for k, v in e.items():
        if k in combined:
            combined[k] += we / s * v
    for k, v in a.items():
        combined[k] = combined.get(k, 0.0) + wa / s * v
    # "other"/"unknown" are not useful directions for synthesis
    ranked = sorted(combined.items(), key=lambda kv: -kv[1])
    label = ranked[0][0]
    if label in ("other", "unknown") and len(ranked) > 1 and ranked[1][1] > 0.15:
        label = ranked[1][0]
    return label, {k: round(v, 4) for k, v in combined.items() if v > 0}


def _get(d: Mapping[str, Any] | None, *path: str, default: Any = None) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, Mapping) or p not in cur:
            return default
        cur = cur[p]
    return cur if cur is not None else default


def build_style(
    passport: Mapping[str, Any],
    char_stats: Mapping[str, Any] | None = None,
    notes: str = "",
    w_e2v: float = 0.6,
    w_anime: float = 0.4,
    use_caption: bool = True,
) -> StyleSpec:
    stats = dict(char_stats or {})
    label, probs = emotion_vote(_get(passport, "emotion", "e2v", "probs"), _get(passport, "emotion", "anime", "probs"), w_e2v, w_anime)
    adv = _get(passport, "emotion", "adv", default={}) or {}
    arousal = adv.get("arousal")
    if arousal is None:
        arousal = EMOTION_VA.get(label, (0.5, 0.5))[1] * (0.5 + 0.5 * probs.get(label, 0.5))
    pros = passport.get("prosody") or {}

    tempo = "normal"
    rate, rate_med = pros.get("rate_mora_per_s"), stats.get("rate_median")
    if rate and rate_med:
        r = float(rate) / float(rate_med)
        tempo = "fast" if r > 1 + TEMPO_THRESHOLD else "slow" if r < 1 - TEMPO_THRESHOLD else "normal"

    loudness = "normal"
    e, e_med = pros.get("energy_db_mean"), stats.get("energy_db_median")
    if e is not None and e_med is not None:
        d = float(e) - float(e_med)
        loudness = "loud" if d > LOUDNESS_DB else "soft" if d < -LOUDNESS_DB else "normal"

    events = [ev for ev in (passport.get("events") or []) if ev != "bgm"]
    vq: list[str] = []
    tilt = pros.get("spectral_tilt_db_khz")
    voiced = pros.get("voiced_ratio")
    if loudness == "soft" and tilt is not None and tilt > -2.0 and (voiced is None or voiced < 0.5):
        if "whisper" not in events:
            events.append("whisper")
        vq.append("whispery")
    elif tilt is not None and tilt > -1.0:
        vq.append("breathy")
    f0_med, char_f0 = pros.get("f0_median_hz"), stats.get("f0_median_hz")
    if f0_med and char_f0:
        st = 12 * math.log2(float(f0_med) / float(char_f0))
        if st > 2.0:
            vq.append("bright")
        elif st < -2.0:
            vq.append("low")
    valence = adv.get("valence")
    if float(arousal) > 0.7 and valence is not None and float(valence) < 0.35:
        vq.append("tense")

    arc = pros.get("arc") or "flat"
    caption = passport.get("caption") or "" if use_caption else ""
    return StyleSpec(
        emotion=label,
        intensity=round(float(min(1.0, max(0.0, arousal))), 3),
        tempo=tempo,
        loudness=loudness,
        events=sorted(set(events), key=events.index),
        voice_quality=vq,
        arc=arc,
        notes=notes or "",
        valence=None if valence is None else round(float(valence), 3),
        dominance=None if adv.get("dominance") is None else round(float(adv["dominance"]), 3),
        confidence=round(probs.get(label, 0.0), 3),
        caption=caption or "",
        emotion_probs=probs,
    )


# ----------------------------------------------------------------------------
# renderers
# ----------------------------------------------------------------------------


def _intensity_word(x: float) -> str:
    if x >= 0.8:
        return "very intense"
    if x >= 0.6:
        return "intense"
    if x >= 0.4:
        return "moderately expressive"
    if x >= 0.2:
        return "restrained"
    return "very subdued"


_EVENT_PHRASE = {
    "laughter": "with a light laugh",
    "cry": "on the verge of tears, voice breaking",
    "breath": "with an audible breath",
    "whisper": "whispering",
    "sigh": "with a sigh",
    "cough": "with a small cough",
}


def to_instruction(spec: StyleSpec, lang: str = "en", include_caption: bool = True) -> str:
    """Natural-language instruction (English; most instruct TTS follow English best)."""
    if lang != "en":
        # Only English templates are maintained; instruct models accept English.
        lang = "en"
    parts = [f"Speak in a {_ADJ.get(spec.emotion, spec.emotion)} tone, {_intensity_word(spec.intensity)}"]
    if spec.tempo != "normal":
        parts.append("at a fast pace" if spec.tempo == "fast" else "at a slow pace")
    if spec.loudness != "normal":
        parts.append("loudly" if spec.loudness == "loud" else "softly")
    for ev in spec.events:
        if ev in _EVENT_PHRASE:
            parts.append(_EVENT_PHRASE[ev])
    vq = [q for q in spec.voice_quality if q not in ("whispery",)]
    if vq:
        parts.append("voice " + " and ".join(vq))
    if spec.arc == "rising_end":
        parts.append("with rising intonation at the end")
    elif spec.arc == "falling_end":
        parts.append("with falling intonation at the end")
    text = ", ".join(parts) + "."
    if spec.notes:
        text += f" Delivery: {spec.notes.strip().rstrip('.')}."
    if include_caption and spec.caption:
        text += f" Reference performance: {spec.caption.strip()}"
    return text


def to_cosy_instruct(spec: StyleSpec) -> str:
    """CosyVoice3 instruct prompt: instruction text before ``<|endofprompt|>``."""
    return f"You are a helpful assistant. {to_instruction(spec, include_caption=False)}<|endofprompt|>"


def to_voxcpm_prefix(spec: StyleSpec) -> str:
    """VoxCPM2 style text in parentheses, prepended to the target text."""
    words = [_ADJ.get(spec.emotion, spec.emotion).split(",")[0]]
    if spec.intensity >= 0.7:
        words.append("intense")
    elif spec.intensity <= 0.25:
        words.append("subdued")
    if spec.tempo != "normal":
        words.append("faster" if spec.tempo == "fast" else "slower")
    if spec.loudness != "normal":
        words.append("loud" if spec.loudness == "loud" else "soft")
    words += [{"laughter": "laughing", "cry": "tearful", "whisper": "whispering", "breath": "breathy", "sigh": "sighing"}.get(e, "") for e in spec.events]
    if spec.arc == "rising_end":
        words.append("rising at the end")
    if spec.notes:
        words.append(spec.notes.strip().rstrip("."))
    return "(" + ", ".join(w for w in words if w) + ")"


_FISH_EMO = {
    "angry": "angry", "disgusted": "disgusted", "fearful": "scared", "happy": "happy", "neutral": "",
    "sad": "sad", "surprised": "surprised", "embarrassed": "embarrassed", "other": "", "unknown": "",
}
_FISH_EVENT = {"laughter": "laughing", "cry": "crying", "breath": "breath", "whisper": "whisper", "sigh": "sigh", "cough": "cough"}


def to_fish_tags(spec: StyleSpec) -> str:
    """Fish S2 inline tags placed before the line, e.g. ``[happy] [laughing]``."""
    tags = []
    emo = _FISH_EMO.get(spec.emotion, spec.emotion)
    if emo:
        tags.append(("very " if spec.intensity >= 0.8 else "") + emo)
    tags += [_FISH_EVENT[e] for e in spec.events if e in _FISH_EVENT]
    if spec.loudness == "loud":
        tags.append("shouting" if spec.intensity >= 0.75 else "loud")
    elif spec.loudness == "soft" and "whisper" not in spec.events:
        tags.append("soft")
    if spec.tempo == "fast":
        tags.append("fast")
    elif spec.tempo == "slow":
        tags.append("slow")
    if spec.arc == "rising_end":
        tags.append("pitch up")
    return " ".join(f"[{t}]" for t in tags)


_HIGGS_EMO = {
    "angry": "anger", "disgusted": "disgust", "fearful": "fear", "happy": "happiness", "neutral": "neutral",
    "sad": "sadness", "surprised": "surprise", "embarrassed": "embarrassment", "other": "neutral", "unknown": "neutral",
}


def to_higgs_tags(spec: StyleSpec) -> str:
    tags = [f"<|emotion:{_HIGGS_EMO.get(spec.emotion, 'neutral')}|>"]
    if "whisper" in spec.events:
        tags.append("<|style:whisper|>")
    if spec.tempo != "normal":
        tags.append(f"<|prosody:speed_{'fast' if spec.tempo == 'fast' else 'slow'}|>")
    sfx = {"cry": "crying", "breath": "breathing"}
    for e in spec.events:
        if e in ("laughter", "cry", "sigh", "cough", "breath"):
            tags.append(f"<|sfx:{sfx.get(e, e)}|>")
    return "".join(tags)


# Zonos emotion slider order: happiness, sadness, disgust, fear, surprise, anger, other, neutral
_ZONOS_ORDER = ["happy", "sad", "disgusted", "fearful", "surprised", "angry", "other", "neutral"]


def to_knobs(spec: StyleSpec, backend: str, opts: Mapping[str, Any] | None = None, cross_lingual: bool = False) -> dict[str, Any]:
    """Scalar controls for knob-driven backends (``caps.knobs``)."""
    opts = dict(opts or {})
    if backend == "chatterbox":
        lo, hi = opts.get("exaggeration_range", (0.3, 1.2))
        exag = lo + (hi - lo) * float(spec.intensity)
        cfg = opts.get("cfg_weight_cross_lingual", 0.0) if cross_lingual else opts.get("cfg_weight", 0.5)
        if spec.tempo == "fast" and not cross_lingual:
            cfg = max(0.2, float(cfg) - 0.15)  # lower cfg -> faster, more animated delivery
        return {"exaggeration": round(float(exag), 3), "cfg_weight": float(cfg)}
    if backend == "zonos":
        probs = dict(spec.emotion_probs or {spec.emotion: 1.0})
        vec = [float(probs.get(k, 0.0)) for k in _ZONOS_ORDER]
        vec[_ZONOS_ORDER.index("happy")] += float(probs.get("embarrassed", 0.0)) * 0.5
        s = sum(vec) or 1.0
        vec = [round(v / s, 4) for v in vec]
        return {
            "emotion": vec,
            "valence": spec.valence if spec.valence is not None else EMOTION_VA.get(spec.emotion, (0.5, 0.5))[0],
            "arousal": spec.intensity,
            "emotion_strength": round(0.5 + 0.8 * float(spec.intensity), 3),
            "speaking_rate": {"fast": 17.0, "slow": 11.0}.get(spec.tempo, 14.0),
        }
    # generic knobs consumed by mock/sapi backends
    return {"tempo": spec.tempo, "loudness": spec.loudness, "intensity": spec.intensity, "arc": spec.arc, "emotion": spec.emotion}


def render(spec: StyleSpec, caps: Mapping[str, Any], opts: Mapping[str, Any] | None = None, cross_lingual: bool = False) -> dict[str, Any]:
    """Everything a backend needs from the style: ``instruct``, ``text_prefix``, ``knobs``, ``style_prompt``.

    ``style_prompt`` is the human-readable string stored in the synth sidecar.
    """
    instruct_kind = caps.get("instruct", "none")
    tags_kind = caps.get("tags", "none")
    knobs_kind = caps.get("knobs", "none")
    out: dict[str, Any] = {"instruct": None, "text_prefix": "", "knobs": {}, "style_prompt": ""}
    if instruct_kind in ("natural", "qwen"):
        out["instruct"] = to_instruction(spec)
    elif instruct_kind == "cosy":
        out["instruct"] = to_cosy_instruct(spec)
    elif instruct_kind == "voxcpm":
        out["text_prefix"] = to_voxcpm_prefix(spec)
    if tags_kind == "fish":
        out["text_prefix"] = (out["text_prefix"] + " " + to_fish_tags(spec)).strip()
    elif tags_kind == "higgs":
        out["text_prefix"] = (out["text_prefix"] + to_higgs_tags(spec)).strip()
    elif tags_kind == "cosy_inline" and "breath" in spec.events:
        out["text_prefix"] = (out["text_prefix"] + " [breath]").strip()
    if knobs_kind in ("chatterbox", "zonos"):
        out["knobs"] = to_knobs(spec, knobs_kind, opts, cross_lingual)
    else:
        out["knobs"] = to_knobs(spec, "generic")
    out["style_prompt"] = out["instruct"] or out["text_prefix"] or to_instruction(spec, include_caption=False)
    return out


def to_emotion_ref(spec: StyleSpec, refs: Iterable[Mapping[str, Any]], usage: str | None = None) -> str | None:
    """Pick the ref clip with the same emotion class; else nearest by (valence, arousal).

    ``refs`` entries: ``{path, emotion, arousal?, valence?, usage?, snr?}``.
    """
    pool = [r for r in refs if usage is None or r.get("usage") in (None, usage, "both")]
    if not pool:
        return None
    same = [r for r in pool if r.get("emotion") == spec.emotion]
    if same:
        return max(same, key=lambda r: (-abs(float(r.get("arousal", spec.intensity)) - spec.intensity), float(r.get("snr", 0))))["path"]
    tv = spec.valence if spec.valence is not None else EMOTION_VA.get(spec.emotion, (0.5, 0.5))[0]
    ta = spec.intensity

    def dist(r: Mapping[str, Any]) -> float:
        v = r.get("valence", EMOTION_VA.get(r.get("emotion", "neutral"), (0.5, 0.5))[0])
        a = r.get("arousal", EMOTION_VA.get(r.get("emotion", "neutral"), (0.5, 0.5))[1])
        return math.hypot(float(v) - tv, float(a) - ta)

    return min(pool, key=dist)["path"]

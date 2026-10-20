"""Voice activity: energy VAD (core, CPU) and pause / silence extraction.

The default analyze profile uses silero-vad from the ``asr`` env (faster-whisper
bundles the ONNX model); this module is the dependency-free fallback and the
shared post-processing (segments -> pauses, leading/trailing silence).
"""

from __future__ import annotations

import numpy as np

from vndub.audio.features import HOP_S, rms_db


def energy_vad(
    y: np.ndarray,
    sr: int,
    rel_threshold_db: float = -35.0,
    abs_floor_db: float = -60.0,
    min_speech_s: float = 0.08,
    min_silence_s: float = 0.12,
    hop_s: float = HOP_S,
) -> list[tuple[float, float]]:
    """Speech segments (seconds) where energy exceeds max - |rel| dB (and an absolute floor)."""
    e = rms_db(y, sr, hop_s=hop_s)
    if e.size == 0:
        return []
    thr = max(float(e.max()) + rel_threshold_db, abs_floor_db)
    active = e > thr
    segs: list[list[float]] = []
    start = None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        elif not a and start is not None:
            segs.append([start * hop_s, i * hop_s])
            start = None
    if start is not None:
        segs.append([start * hop_s, len(active) * hop_s])
    # merge short gaps, drop short blips
    merged: list[list[float]] = []
    for s in segs:
        if merged and s[0] - merged[-1][1] < min_silence_s:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    total = len(y) / sr
    return [(round(a, 3), round(min(b, total), 3)) for a, b in merged if b - a >= min_speech_s]


def silence_profile(segments: list[tuple[float, float]], duration: float, min_pause_s: float = 0.15) -> dict:
    """Leading/trailing silence and inner pauses from speech segments."""
    if not segments:
        return {"leading_silence_s": round(duration, 3), "trailing_silence_s": 0.0, "pauses": [], "speech_s": 0.0}
    lead = segments[0][0]
    trail = max(0.0, duration - segments[-1][1])
    pauses = []
    for (a0, a1), (b0, _) in zip(segments, segments[1:]):
        gap = b0 - a1
        if gap >= min_pause_s:
            pauses.append({"t": round(a1, 3), "d": round(gap, 3)})
    speech = sum(b - a for a, b in segments)
    return {
        "leading_silence_s": round(lead, 3),
        "trailing_silence_s": round(trail, 3),
        "pauses": pauses,
        "speech_s": round(speech, 3),
    }

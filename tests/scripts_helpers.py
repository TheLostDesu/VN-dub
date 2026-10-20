"""Small audio generators for tests."""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf


def tone_bytes(f0: float, seconds: float = 0.3, sr: int = 16000) -> bytes:
    t = np.arange(int(seconds * sr)) / sr
    y = (0.2 * np.sin(2 * np.pi * f0 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()

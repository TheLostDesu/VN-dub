"""Integrated loudness (ITU-R BS.1770 via pyloudnorm) and a true-peak limiter."""

from __future__ import annotations

import numpy as np


def lufs(y: np.ndarray, sr: int) -> float | None:
    import pyloudnorm as pyln

    if y.ndim == 2:
        data = y.T if y.shape[0] < y.shape[1] else y
    else:
        data = y
    min_len = int(0.4 * sr) + 1  # BS.1770 gating block
    if 0 < data.shape[0] < min_len:
        # tile, do not zero-pad: padding adds silence that the gate averages in, so a 0.2 s tone read 3 dB and a
        # 0.1 s one 6 dB too low (and master then boosted short interjections by that much)
        reps = -(-min_len // data.shape[0])
        data = np.tile(data, (reps, 1) if data.ndim == 2 else reps)[:min_len]
    elif data.shape[0] == 0:
        return None
    meter = pyln.Meter(sr)
    val = float(meter.integrated_loudness(data))
    return None if not np.isfinite(val) else val


def true_peak_db(y: np.ndarray, sr: int, oversample: int = 4) -> float:
    from scipy.signal import resample_poly

    x = y if y.ndim == 1 else y.reshape(-1)
    if x.size == 0:
        return -120.0
    up = resample_poly(x, oversample, 1)
    peak = float(np.max(np.abs(up))) if up.size else 0.0
    return 20 * np.log10(max(peak, 1e-9))


def normalize(y: np.ndarray, sr: int, target_lufs: float, ceiling_dbtp: float = -1.0) -> tuple[np.ndarray, dict]:
    """Gain to ``target_lufs`` then limit true peak to ``ceiling_dbtp``."""
    before = lufs(y, sr)
    out = y.astype(np.float64)
    gain_db = 0.0
    if before is not None:
        gain_db = target_lufs - before
        out = out * (10 ** (gain_db / 20))
    out, limited_db = limit_true_peak(out, sr, ceiling_dbtp)
    after = lufs(out, sr)
    return out.astype(np.float32), {"lufs_before": before, "lufs_after": after, "gain_db": round(gain_db, 2), "limiter_db": round(limited_db, 2)}


def limit_true_peak(y: np.ndarray, sr: int, ceiling_dbtp: float = -1.0, release_s: float = 0.05) -> tuple[np.ndarray, float]:
    """Look-ahead-free gain-riding limiter on the oversampled envelope.

    Speech lines are short, so a smooth per-sample gain curve computed from the
    true-peak envelope is enough; returns (signal, max gain reduction dB).
    """
    tp = true_peak_db(y, sr)
    if tp <= ceiling_dbtp:
        return y, 0.0
    from scipy.ndimage import maximum_filter1d
    from scipy.signal import resample_poly

    ceiling = 10 ** (ceiling_dbtp / 20)
    x = y.reshape(-1)
    env = np.abs(resample_poly(x, 4, 1))[::4][: x.size]
    if env.size < x.size:
        env = np.pad(env, (0, x.size - env.size), mode="edge")
    win = min(max(1, int(release_s * sr)), x.size)  # np.convolve(..., "same") returns max(len) samples: a clip shorter than the window would break the shapes
    env = maximum_filter1d(np.maximum(env, np.abs(x)), size=win)
    gain = np.minimum(1.0, ceiling / np.maximum(env, 1e-9))
    # smooth the gain curve (moving average over the release window)
    kernel = np.ones(win) / win
    gain = np.minimum(gain, np.convolve(gain, kernel, mode="same"))
    out = x * gain
    # final safety: hard scale if oversampled peak still exceeds the ceiling
    tp2 = true_peak_db(out, sr)
    if tp2 > ceiling_dbtp:
        out = out * (10 ** ((ceiling_dbtp - tp2) / 20))
    reduction = float(20 * np.log10(max(np.min(gain), 1e-9)))
    return out.reshape(y.shape), reduction

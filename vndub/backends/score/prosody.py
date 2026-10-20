"""Prosody similarity between the original line and a candidate (CPU, core).

* F0: log-F0 on voiced frames only, each contour resampled to 100 points,
  DTW-aligned (fastdtw), then Pearson correlation.
* Energy: dB contour, same procedure.
* Pauses: Jaccard over pause positions in normalized time (20 bins).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

N_POINTS = 100


def _resample(x: np.ndarray, n: int = N_POINTS) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return x
    if x.size == 1:
        return np.repeat(x, n)
    src = np.linspace(0.0, 1.0, x.size)
    dst = np.linspace(0.0, 1.0, n)
    return np.interp(dst, src, x)


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 3 or b.size < 3:
        return None
    sa, sb = a.std(), b.std()
    if sa < 1e-9 or sb < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def dtw_corr(a: Sequence[float], b: Sequence[float]) -> float | None:
    from fastdtw import fastdtw

    ra, rb = _resample(np.asarray(a)), _resample(np.asarray(b))
    if ra.size == 0 or rb.size == 0:
        return None
    # z-normalize before DTW so the path follows shape, not offset
    za = (ra - ra.mean()) / (ra.std() or 1.0)
    zb = (rb - rb.mean()) / (rb.std() or 1.0)
    _, path = fastdtw(za.reshape(-1, 1), zb.reshape(-1, 1), radius=5, dist=lambda x, y: abs(float(x[0]) - float(y[0])))
    ia = np.array([p[0] for p in path])
    ib = np.array([p[1] for p in path])
    return _pearson(ra[ia], rb[ib])


def f0_corr(f0_src: Sequence[float], f0_cand: Sequence[float]) -> float | None:
    """F0 contours in Hz with 0 for unvoiced frames."""
    a = np.asarray(f0_src, dtype=np.float64)
    b = np.asarray(f0_cand, dtype=np.float64)
    a = np.log(a[a > 0]) if a.size else a
    b = np.log(b[b > 0]) if b.size else b
    if a.size < 5 or b.size < 5:
        return None
    return dtw_corr(a, b)


def energy_corr(e_src: Sequence[float], e_cand: Sequence[float]) -> float | None:
    a = np.asarray(e_src, dtype=np.float64)
    b = np.asarray(e_cand, dtype=np.float64)
    if a.size < 5 or b.size < 5:
        return None
    return dtw_corr(a, b)


def pause_match(pauses_src: Sequence[dict], dur_src: float, pauses_cand: Sequence[dict], dur_cand: float, bins: int = 20) -> float:
    """Jaccard over pause-center bins in normalized time; 1.0 when neither has pauses."""

    def binset(pauses: Sequence[dict], dur: float) -> set[int]:
        out = set()
        if dur <= 0:
            return out
        for p in pauses:
            c = (float(p["t"]) + float(p.get("d", 0.0)) / 2) / dur
            out.add(min(bins - 1, max(0, int(c * bins))))
        return out

    a, b = binset(pauses_src, dur_src), binset(pauses_cand, dur_cand)
    if not a and not b:
        return 1.0
    # tolerate off-by-one bin
    hits = sum(1 for x in a if x in b or x - 1 in b or x + 1 in b)
    union = len(a) + len(b) - hits
    return float(hits / union) if union else 1.0


def prosody_score(f0c: float | None, ec: float | None, pm: float | None) -> float | None:
    """Combined prosody term in [0, 1] for the composite (mean of available parts)."""
    parts = []
    if f0c is not None:
        parts.append((f0c + 1) / 2)
    if ec is not None:
        parts.append((ec + 1) / 2)
    if pm is not None:
        parts.append(pm)
    return float(np.mean(parts)) if parts else None

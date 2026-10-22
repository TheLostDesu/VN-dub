"""Speaker embedders and similarity math.

Embeddings are computed in the ``score`` env (speechbrain ECAPA, WavLM-SV) and
written as ``.npy``; similarity, EER calibration and clustering run in core.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from vndub.backends.base import BackendSpec, WeightSpec

SPECS = [
    BackendSpec(
        name="ecapa",
        role="embedder",
        env="main",
        adapter="ecapa",
        license="Apache-2.0",
        commercial_ok=True,
        vram_estimate_gb=0.4,
        cpu_ok=True,
        weights=(WeightSpec("hf", repo="speechbrain/spkrec-ecapa-voxceleb", size_gb=0.09),),
        profiles=("default", "test"),
        mock="mock_embedder",
        description="speechbrain ECAPA-TDNN x-vectors (192-d): clustering, speaker-sim",
        homepage="https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb",
        heavy=False,
    ),
    BackendSpec(
        name="wavlm_sv",
        role="embedder",
        env="main",
        adapter="wavlm_sv",
        license="MIT",
        commercial_ok=True,
        vram_estimate_gb=0.8,
        cpu_ok=True,
        weights=(WeightSpec("hf", repo="microsoft/wavlm-base-plus-sv", size_gb=0.4),),
        profiles=("optional",),
        mock="mock_embedder",
        description="WavLM-base-plus-SV x-vector: second speaker-sim opinion",
        homepage="https://huggingface.co/microsoft/wavlm-base-plus-sv",
    ),
]


def load_emb(path: str | Path) -> np.ndarray:
    v = np.load(path).astype(np.float32).reshape(-1)
    return v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def max_cosine(emb: np.ndarray, refs: Iterable[np.ndarray]) -> float | None:
    sims = [cosine(emb, r) for r in refs]
    return max(sims) if sims else None


def centroid(embs: Sequence[np.ndarray]) -> np.ndarray:
    m = np.stack([e / (np.linalg.norm(e) or 1.0) for e in embs]).mean(axis=0)
    return m / (np.linalg.norm(m) or 1.0)


def calibrate_eer(same: Sequence[float], diff: Sequence[float]) -> tuple[float, float]:
    """Threshold at the equal-error-rate point and the EER itself.

    ``same``: cosines of same-character pairs; ``diff``: different-character pairs.
    """
    same_a = np.asarray(same, dtype=np.float64)
    diff_a = np.asarray(diff, dtype=np.float64)
    if same_a.size == 0 or diff_a.size == 0:
        raise ValueError("need both same and different pairs to calibrate")
    thresholds = np.unique(np.concatenate([same_a, diff_a]))
    best_t, best_gap, best_eer = float(thresholds[0]), np.inf, 1.0
    for t in thresholds:
        frr = float(np.mean(same_a < t))  # same pairs rejected
        far = float(np.mean(diff_a >= t))  # different pairs accepted
        gap = abs(frr - far)
        if gap < best_gap:
            best_t, best_gap, best_eer = float(t), gap, (frr + far) / 2
    return best_t, best_eer


def agglomerative(embs: np.ndarray, distance_threshold: float) -> np.ndarray:
    """Average-linkage agglomerative clustering on cosine distance -> labels."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    n = len(embs)
    if n == 0:
        return np.zeros(0, dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)
    d = pdist(embs, metric="cosine")
    d = np.nan_to_num(d, nan=1.0)
    z = linkage(d, method="average")
    return fcluster(z, t=distance_threshold, criterion="distance").astype(int) - 1

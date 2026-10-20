"""emotion2vec+ (FunASR): 9-class emotion probabilities + 768-d utterance embedding.

The same model must be used in ``analyze`` and ``score`` (emotion similarity is
the cosine between embeddings of the original and the candidate).
"""

from vndub.backends.base import BackendSpec, WeightSpec

E2V_LABELS = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown"]

_COMMON = dict(
    role="ser",
    env="funasr",
    adapter="emotion2vec",
    license="MIT (code); emotion2vec model license (commercial terms UNVERIFIED)",
    commercial_ok=False,
    cpu_ok=True,
    mock="mock_ser",
    homepage="https://github.com/ddlBoJack/emotion2vec",
)

SPECS = [
    BackendSpec(
        name="emotion2vec_plus_large",
        vram_estimate_gb=1.8,
        weights=(WeightSpec("hf", repo="emotion2vec/emotion2vec_plus_large", size_gb=1.2),),
        fallback="emotion2vec_plus_base",
        description="emotion2vec+ large (~300M, 42k h): categorical emotion + embedding",
        **_COMMON,
    ),
    BackendSpec(
        name="emotion2vec_plus_base",
        vram_estimate_gb=0.8,
        weights=(WeightSpec("hf", repo="emotion2vec/emotion2vec_plus_base", size_gb=0.4),),
        description="emotion2vec+ base (~90M): fallback",
        **_COMMON,
    ),
    BackendSpec(
        name="emotion2vec_plus_seed",
        vram_estimate_gb=0.8,
        weights=(WeightSpec("hf", repo="emotion2vec/emotion2vec_plus_seed", size_gb=0.4),),
        profiles=("test",),
        description="emotion2vec+ seed: small CPU-capable model for the test profile",
        **_COMMON,
    ),
]

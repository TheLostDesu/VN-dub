"""audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim: arousal/dominance/valence.

CC-BY-NC-SA-4.0 -> research only. Batch size 1 (known batched-inference
inconsistency).
"""

from vndub.backends.base import BackendSpec, WeightSpec

SPECS = [
    BackendSpec(
        name="audeering_adv",
        role="adv",
        env="main",
        adapter="audeering_adv",
        license="CC-BY-NC-SA-4.0",
        commercial_ok=False,
        vram_estimate_gb=1.0,
        cpu_ok=True,
        weights=(WeightSpec("hf", repo="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim", size_gb=0.7),),
        options={"batch_size": 1},
        mock="mock_adv",
        description="Dimensional SER (A/D/V in ~0..1)",
        homepage="https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
    ),
]

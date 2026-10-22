"""Qwen3-ForcedAligner-0.6B (Apache-2.0): char timestamps for ja and ru."""

from vndub.backends.base import BackendSpec, WeightSpec

SPECS = [
    BackendSpec(
        name="qwen3_aligner",
        role="align",
        env="main",
        adapter="qwen3_aligner",
        license="Apache-2.0",
        commercial_ok=True,
        vram_estimate_gb=1.5,
        cpu_ok=True,
        weights=(WeightSpec("hf", repo="Qwen/Qwen3-ForcedAligner-0.6B", size_gb=1.3),),
        fallback="whisperx_ja",
        mock="mock_aligner",
        description="NAR forced aligner, 11 languages incl. Japanese and Russian (<=5 min audio)",
        homepage="https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B",
    ),
]

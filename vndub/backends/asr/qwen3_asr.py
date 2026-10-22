"""Qwen3-ASR 1.7B (Apache-2.0): optional third-opinion ASR in the ``hf`` env."""

from vndub.backends.base import BackendSpec, WeightSpec

SPECS = [
    BackendSpec(
        name="qwen3_asr",
        role="asr",
        env="main",
        adapter="qwen3_asr",
        license="Apache-2.0",
        commercial_ok=True,
        vram_estimate_gb=4.0,
        cpu_ok=True,
        weights=(WeightSpec("hf", repo="Qwen/Qwen3-ASR-1.7B", size_gb=4.0),),
        profiles=("optional",),
        mock="mock_asr",
        description="52-language ASR; benchmark against anime-whisper on your own lines",
        homepage="https://github.com/QwenLM/Qwen3-ASR",
    ),
]

"""Qwen2-Audio-7B-Instruct in 4-bit NF4 (bitsandbytes): optional style captions.

Fit in 8 GB is UNVERIFIED; on OOM the analyze stage disables captions and logs.
"""

from vndub.backends.base import BackendSpec, WeightSpec

CAPTION_PROMPT = (
    "Describe how this line is spoken: emotion, intensity, tempo, voice quality "
    "(e.g. breathy, tense, whispery, bright), and notable events (laughter, sigh, "
    "crying, breath). Answer in English in at most 40 words."
)

SPECS = [
    BackendSpec(
        name="qwen2_audio",
        role="caption",
        env="main",
        adapter="qwen2_audio",
        license="Apache-2.0",
        commercial_ok=True,
        vram_estimate_gb=5.8,
        weights=(WeightSpec("hf", repo="Qwen/Qwen2-Audio-7B-Instruct", size_gb=16.0),),
        options={"quantization": "nf4", "max_new_tokens": 80},
        profiles=("optional",),
        description="Natural-language caption of the delivery (offline preprocessing only)",
        homepage="https://github.com/QwenLM/Qwen2-Audio",
    ),
]

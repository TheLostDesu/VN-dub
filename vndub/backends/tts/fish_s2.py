"""Fish Audio S2 Pro via audio.cpp GGUF Q4_K (research license, optional).

Runs the audio.cpp CLI (path in ``tools.yaml: audiocpp``) from the ``gguf``
env adapter; emotion via free-form inline tags (``[angry] [laughing] [whisper]``).
"""

from vndub.backends.base import BackendSpec, WeightSpec

SPECS = [
    BackendSpec(
        name="fish_s2",
        role="tts",
        env="gguf",
        adapter="fish_s2",
        license="Fish Audio Research License (non-commercial)",
        commercial_ok=False,
        vram_estimate_gb=7.5,
        weights=(WeightSpec("hf_file", repo="fishaudio/s2-pro-gguf", allow=("*Q4_K*.gguf", "*codec*"), subdir="s2-pro-gguf", size_gb=3.5),),
        options={"tool": "audiocpp", "quant": "Q4_K"},
        caps={"instruct": "none", "tags": "fish", "knobs": "none", "duration": False, "stress_format": "plain", "needs_ref_text": True, "sr": 44100, "langs": ["ja", "ru", "en", "zh"]},
        profiles=("optional",),
        mock="mock_tts",
        description="Most expressive tag-directed model that fits 8 GB (GGUF); research only",
        homepage="https://huggingface.co/fishaudio/s2-pro",
    ),
]

"""MOSS-TTS v1.5 through its llama.cpp path (Apache-2.0, optional), ``gguf`` env.

The only Russian model with explicit duration control (target token count).
"""

from vndub.backends.base import BackendSpec, WeightSpec

SPECS = [
    BackendSpec(
        name="moss_tts",
        role="tts",
        env="gguf",
        adapter="moss_tts",
        license="Apache-2.0",
        commercial_ok=True,
        vram_estimate_gb=7.0,
        weights=(WeightSpec("hf_file", repo="OpenMOSS-Team/MOSS-TTS-v1.5-GGUF", allow=("*Q4_K_M*.gguf", "*codec*", "*.json"), subdir="MOSS-TTS-v1.5-GGUF", size_gb=5.0),),
        options={"tool": "moss_tts_cli", "tokens_per_second": 12.5},
        caps={"instruct": "none", "tags": "none", "knobs": "none", "duration": True, "stress_format": "plain", "needs_ref_text": True, "sr": 24000, "langs": ["ru", "ja", "en", "zh"]},
        profiles=("optional",),
        mock="mock_tts",
        description="MOSS-TTS: duration control via token count",
        homepage="https://github.com/OpenMOSS/MOSS-TTS",
    ),
]

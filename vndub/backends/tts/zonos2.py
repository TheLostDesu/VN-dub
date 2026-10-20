"""Zonos2 through zonos2.cpp Q4_K (optional), ``gguf`` env.

Emotion sliders + valence/arousal mapped directly from the A/D/V passport.
License: README says MIT, blog says Apache-2.0 (both permissive).
"""

from vndub.backends.base import BackendSpec, WeightSpec

SPECS = [
    BackendSpec(
        name="zonos2",
        role="tts",
        env="gguf",
        adapter="zonos2",
        license="MIT/Apache-2.0 (conflicting statements, both permissive)",
        commercial_ok=True,
        vram_estimate_gb=6.5,
        weights=(WeightSpec("hf_file", repo="Zyphra/ZONOS2-GGUF", allow=("*Q4_K*.gguf", "*vocoder*", "*.json"), subdir="ZONOS2-GGUF", size_gb=5.5),),
        options={"tool": "zonos2_cli", "accurate_mode": True},
        caps={"instruct": "none", "tags": "none", "knobs": "zonos", "duration": False, "stress_format": "plain", "needs_ref_text": False, "sr": 44100, "langs": ["ru", "ja", "en"]},
        profiles=("optional",),
        mock="mock_tts",
        description="Zonos2: emotion / valence / arousal sliders",
        homepage="https://github.com/Zyphra/zonos2.cpp",
    ),
]

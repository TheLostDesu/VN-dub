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
        # repo files: zonos2-{f16,q8_0,q6_k,q5_k,q4_k}.gguf + dac.gguf (vocoder) + spk-encoder.gguf (cloning); no json
        weights=(WeightSpec("hf_file", repo="Zyphra/ZONOS2-GGUF", allow=("zonos2-q4_k.gguf", "dac.gguf", "spk-encoder.gguf"), subdir="ZONOS2-GGUF", size_gb=5.2),),
        options={"tool": "zonos2_cli", "accurate_mode": True, "model_glob": "zonos2-q4_k.gguf"},
        caps={"instruct": "none", "tags": "none", "knobs": "zonos", "duration": False, "stress_format": "plain", "needs_ref_text": False, "sr": 44100, "langs": ["ru", "ja", "en"]},
        profiles=("optional",),
        mock="mock_tts",
        description="Zonos2: emotion / valence / arousal sliders",
        homepage="https://github.com/Zyphra/zonos2.cpp",
    ),
]

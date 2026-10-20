"""F5-TTS + Misha24-10/F5-TTS_RUSSIAN (CC-BY-NC-4.0, optional), own ``f5`` env.

Explicit duration (``fix_duration`` = reference + target seconds) and explicit
stress marks (``+`` before the stressed vowel, produced by ruaccent).
"""

from vndub.backends.base import BackendSpec, WeightSpec

SPECS = [
    BackendSpec(
        name="f5_ru",
        role="tts",
        env="f5",
        adapter="f5_ru",
        license="CC-BY-NC-4.0 (Russian checkpoint)",
        commercial_ok=False,
        vram_estimate_gb=3.0,
        cpu_ok=True,
        weights=(WeightSpec("hf", repo="Misha24-10/F5-TTS_RUSSIAN", size_gb=5.5),),
        options={"ckpt_glob": "*v4*winter*/*.safetensors", "vocab_glob": "*vocab*.txt", "nfe_step": 32, "cfg_strength": 2.0},
        caps={"instruct": "none", "tags": "none", "knobs": "none", "duration": True, "stress_format": "plus", "needs_ref_text": True, "sr": 24000, "langs": ["ru", "en"]},
        profiles=("optional",),
        mock="mock_tts",
        description="Russian F5-TTS with fix_duration and '+' stress marks",
        homepage="https://huggingface.co/Misha24-10/F5-TTS_RUSSIAN",
    ),
]

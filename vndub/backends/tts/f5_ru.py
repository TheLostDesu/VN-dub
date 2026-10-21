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
        # the repo holds many 5 GB training checkpoints: pull only the v4_winter inference weights + vocab
        weights=(WeightSpec("hf", repo="Misha24-10/F5-TTS_RUSSIAN", allow=("F5TTS_v1_Base_v4_winter/model_212000.safetensors", "F5TTS_v1_Base/vocab.txt"), size_gb=1.35),),
        options={"arch": "F5TTS_v1_Base", "ckpt_glob": "F5TTS_v1_Base_v4_winter/*.safetensors", "vocab_glob": "F5TTS_v1_Base/vocab.txt", "nfe_step": 32, "cfg_strength": 2.0},
        caps={"instruct": "none", "tags": "none", "knobs": "none", "duration": True, "stress_format": "plus", "needs_ref_text": True, "sr": 24000, "langs": ["ru", "en"]},
        profiles=("optional",),
        mock="mock_tts",
        description="Russian F5-TTS with fix_duration and '+' stress marks",
        homepage="https://huggingface.co/Misha24-10/F5-TTS_RUSSIAN",
    ),
]

"""Seed-VC (GPL-3.0; repo archived 2025-11), own ``seedvc`` env.

Zero-shot timbre conversion with ``convert_style=False`` (keeps the source
intonation). Used when a character has too little audio for RVC, and as an
alternative candidate on emotional peaks.
"""

from vndub.backends.base import BackendSpec, WeightSpec

SPECS = [
    BackendSpec(
        name="seedvc",
        role="vc",
        env="seedvc",
        adapter="seedvc",
        license="GPL-3.0",
        commercial_ok=True,
        vram_estimate_gb=3.0,
        cpu_ok=True,
        weights=(WeightSpec("hf", repo="Plachta/Seed-VC", size_gb=1.6),),
        options={"convert_style": False, "diffusion_steps": 30, "inference_cfg_rate": 0.7, "length_adjust": 1.0},
        caps={"trainable": False, "zero_shot": True, "keeps_prosody": True},
        mock="mock_vc",
        description="Zero-shot VC fallback",
        homepage="https://github.com/Plachtaa/seed-vc",
    ),
]

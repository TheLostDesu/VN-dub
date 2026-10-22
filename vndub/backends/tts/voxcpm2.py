"""VoxCPM2 2B (OpenBMB, Apache-2.0): controllable cloning + style text + voice design.

Official footprint "~8 GB": disabled by default until ``--selftest`` passes on
the machine. Style instruction is given as a parenthesized prefix, e.g.
``(cheerful, slightly faster)``. LoRA scripts make it the route-C default.
"""

from vndub.backends.base import BackendSpec, WeightSpec

SPECS = [
    BackendSpec(
        name="voxcpm2",
        role="tts",
        env="main",
        adapter="voxcpm2",
        license="Apache-2.0",
        commercial_ok=True,
        vram_estimate_gb=7.8,
        cpu_ok=False,
        weights=(WeightSpec("hf", repo="openbmb/VoxCPM2", size_gb=4.5),),
        options={"cfg_value": 2.0, "inference_timesteps": 10, "normalize": True},
        caps={
            "instruct": "voxcpm",
            "tags": "none",
            "knobs": "none",
            "duration": False,
            "stress_format": "plain",
            "needs_ref_text": False,
            "cross_lingual": True,
            "design": True,
            "lora": True,
            "sr": 48000,
            "langs": ["ru", "ja", "en", "zh"],
        },
        mock="mock_tts",
        description="VoxCPM2: timbre from wav + style instruction; voice design; LoRA",
        homepage="https://github.com/OpenBMB/VoxCPM",
    ),
]

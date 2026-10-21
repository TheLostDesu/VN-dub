"""Higgs TTS 3 4B (Boson AI, research & non-commercial, optional), own ``higgs`` env.

Richest tag vocabulary (``<|emotion:anger|>``, ``<|style:whisper|>``,
``<|sfx:laughter|>``). Higgs v3 is distributed for SGLang-Omni (Linux), so the
adapter is a client of a server on 127.0.0.1 (WSL2/Docker on the same machine).
Fit on 8 GB (4-bit) is UNVERIFIED.
"""

from vndub.backends.base import BackendSpec, WeightSpec

SPECS = [
    BackendSpec(
        name="higgs_tts3",
        role="tts",
        env="higgs",
        adapter="higgs_tts3",
        license="Boson Research & Non-Commercial License",
        commercial_ok=False,
        vram_estimate_gb=6.0,
        weights=(WeightSpec("hf", repo="bosonai/higgs-audio-v3-tts-4b", size_gb=8.5),),
        # weights are served by a local SGLang-Omni server (WSL2/Docker); the env is a loopback client
        options={"server_url": "http://127.0.0.1:8000", "temperature": 0.7, "max_new_tokens": 1024},
        caps={"instruct": "none", "tags": "higgs", "knobs": "none", "duration": False, "stress_format": "plain", "needs_ref_text": True, "sr": 24000, "langs": ["ru", "ja", "en", "zh"]},
        profiles=("optional",),
        mock="mock_tts",
        description="Higgs TTS 3: quality ceiling for A/B tests (non-commercial)",
        homepage="https://huggingface.co/bosonai/higgs-audio-v3-tts-4b",
    ),
]

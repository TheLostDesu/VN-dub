"""Fish Audio S2 Pro via audio.cpp GGUF Q8_0 (research license, optional).

Runs the audio.cpp CLI (``audiocpp_cli``, path in ``tools.yaml: audiocpp``) from the ``gguf`` env adapter; emotion
via free-form inline tags (``[angry] [laughing] [whisper]``).

Weights come from the official audio.cpp hub ``audio-cpp/audio.cpp-gguf`` (``Fish-Audio-S2-Pro-GGUF/``): one
self-contained file per precision (backbone + codec + tokenizer), bf16 and Q8_0 only. No Q4_K/Q6_K file is published
anywhere; the repo this module first named (``fishaudio/s2-pro-gguf``) never existed. A smaller file is made locally
with ``audiocpp_gguf`` (audio.cpp ``docs/proposals/fish_audio_quantization_recipe.md``); drop it next to the Q8_0 file
and point ``options.model_glob`` at it in ``pipeline.yaml``. Q8_0 (6.3 GB) was measured by that recipe on an 8 GB
RTX 4070 laptop GPU.

``args`` is the documented audio.cpp command line (``docs/models/fish_audio.md``): ``--task clon --family fish_audio``.
It is not run on the build machine (no GPU).
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
        weights=(
            WeightSpec(
                "hf_file",
                repo="audio-cpp/audio.cpp-gguf",
                allow=("Fish-Audio-S2-Pro-GGUF/fish-audio-s2-pro-q8_0.gguf",),
                subdir="s2-pro-gguf",
                size_gb=6.4,
            ),
        ),
        options={
            "tool": "audiocpp",
            "quant": "Q8_0",
            "model_glob": "fish-audio-s2-pro-q8_0.gguf",
            "args": [
                "--task", "clon", "--family", "fish_audio", "--model", "{model}", "--backend", "best",
                "--text", "{text}", "--voice-ref", "{ref}", "--reference-text", "{ref_text}", "--seed", "{seed}", "--out", "{out}",
            ],
        },
        caps={"instruct": "none", "tags": "fish", "knobs": "none", "duration": False, "stress_format": "plain", "needs_ref_text": True, "sr": 44100, "langs": ["ja", "ru", "en", "zh"]},
        profiles=("optional",),
        mock="mock_tts",
        description="Most expressive tag-directed model that fits 8 GB (GGUF Q8_0); research only",
        homepage="https://huggingface.co/fishaudio/s2-pro",
    ),
]

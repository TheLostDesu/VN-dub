"""MOSS-TTS v1.5 (8B delay-pattern) through audio.cpp (Apache-2.0, optional), ``gguf`` env.

The only Russian model with explicit duration control (a budget in codec frames, 12.5 per second).

There is no official v1.5 GGUF: ``OpenMOSS-Team/MOSS-TTS-v1.5-GGUF`` (named by the first version of this module) does not
exist, and the official ``OpenMOSS-Team/MOSS-TTS-GGUF`` is the older v1.0 for OpenMOSS's own Python + llama.cpp pipeline
(gated, plus separate embedding/LM-head/ONNX-tokenizer downloads, no CLI). The weights used here are the audio.cpp
package of v1.5 that audio.cpp's own ``model_specs/moss_tts_v15.json`` installs from:
``christopherthompson81/MOSS-TTS-v1.5-GGUF`` (community conversion of the Apache-2.0 ``OpenMOSS-Team/MOSS-TTS-v1.5``;
one self-contained file with the backbone, the encoder+decoder codec and the tokenizer). Q4_K is 9.3 GB on disk and its
voice-cloning peak is 14.1 GiB of VRAM (audio.cpp ``docs/models/moss_tts_v15.md``, measured on an RTX 3090: cloning keeps
both halves of the codec resident): it fits neither an 8 GB nor a 12 GB card, hence ``vram_estimate_gb`` 14.5 (an estimate
of 9.5 let a 12 GB card pass the fit check and spill or run out of memory). Q8_0 (13.2 GB, 17.9 GiB peak) and bf16
(20.5 GB, does not clone on 24 GiB) are in the same repo.

``args`` is the documented audio.cpp command line (``docs/models/moss_tts_v15.md``): ``--task clon --family moss_tts_v15``,
``duration_frames`` = the duration budget. audio.cpp lists en/zh as the tested languages and the adapter passes no
``--language``; not run on the build machine (no GPU).
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
        vram_estimate_gb=14.5,  # q4_k CUDA cloning peak 14.1 GiB (audio.cpp docs/models/moss_tts_v15.md)
        weights=(
            WeightSpec(
                "hf_file",
                repo="christopherthompson81/MOSS-TTS-v1.5-GGUF",
                allow=("moss_tts_v15_q4_k_codec_f16.gguf",),
                subdir="MOSS-TTS-v1.5-GGUF",
                size_gb=9.3,
            ),
        ),
        options={
            "tool": "audiocpp",
            "quant": "Q4_K",
            "tokens_per_second": 12.5,
            "model_glob": "moss_tts_v15_q4_k_codec_f16.gguf",
            "args": [
                "--task", "clon", "--family", "moss_tts_v15", "--model", "{model_dir}", "--backend", "best",
                "--text", "{text}", "--voice-ref", "{ref}", "--request-option", "duration_frames={tokens}", "--seed", "{seed}", "--out", "{out}",
            ],
        },
        caps={"instruct": "none", "tags": "none", "knobs": "none", "duration": True, "stress_format": "plain", "needs_ref_text": True, "sr": 24000, "langs": ["ru", "ja", "en", "zh"]},
        profiles=("optional",),
        mock="mock_tts",
        description="MOSS-TTS: duration control via token count",
        homepage="https://github.com/OpenMOSS/MOSS-TTS",
    ),
]

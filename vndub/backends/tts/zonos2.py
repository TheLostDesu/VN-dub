"""Zonos2 through zonos2.cpp Q4_K (optional), ``gguf`` env.

Emotion sliders + valence/arousal mapped directly from the A/D/V passport.
License: README says MIT, blog says Apache-2.0 (both permissive).

``args`` is the command line of ``zonos2-cli`` (zonos2.cpp ``src/main.cpp`` and README):
``zonos2-cli <model.gguf> --tts "<text>" <out.wav> [--clone ref.wav] [--emotion happy=1,...]
[--emotion-valence/-arousal/-strength X] [--seed N] [--gpu]``. It exits 1 on an unknown flag. ``dac.gguf`` and
``spk-encoder.gguf`` sit next to the backbone and are found by the CLI itself. Cloning decodes the reference through
ffmpeg (``$ZONOS2_FFMPEG``, set from the configured ffmpeg via ``tool_dir``). Not run on the build machine (no GPU).
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
        options={
            "tool": "zonos2_cli",
            "tool_dir": "ffmpeg",  # registry.inject_tools -> options.ffmpeg (the configured ffmpeg) -> $ZONOS2_FFMPEG of the CLI
            "accurate_mode": True,  # the CLI default (``--inaccurate`` turns it off)
            "model_glob": "zonos2-q4_k.gguf",
            "args": [
                "{model}", "--tts", "{text}", "{out}", "--clone", "{ref}",
                "--emotion", "{emotion}", "--emotion-valence", "{valence}", "--emotion-arousal", "{arousal}", "--emotion-strength", "{strength}",
                "--emotion-dir", "{emotion_dir}", "--seed", "{seed}", "{gpu}",
            ],
        },
        caps={"instruct": "none", "tags": "none", "knobs": "zonos", "duration": False, "stress_format": "plain", "needs_ref_text": False, "sr": 44100, "langs": ["ru", "ja", "en"]},
        profiles=("optional",),
        mock="mock_tts",
        description="Zonos2: emotion / valence / arousal sliders",
        homepage="https://github.com/Zyphra/zonos2.cpp",
    ),
]

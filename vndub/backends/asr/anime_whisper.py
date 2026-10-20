"""litagin/anime-whisper: Japanese galgame ASR (MIT).

Runs in the ``asr`` env through faster-whisper after a one-time CTranslate2
conversion (``models pull`` performs it). Per the model card: no initial
prompt, ``no_repeat_ngram_size=5``, no conditioning on previous text.
"""

from vndub.backends.base import BackendSpec, WeightSpec

SPECS = [
    BackendSpec(
        name="anime_whisper",
        role="asr",
        env="asr",
        adapter="faster_whisper",
        license="MIT",
        commercial_ok=True,
        vram_estimate_gb=2.5,
        cpu_ok=True,
        weights=(WeightSpec("hf", repo="litagin/anime-whisper", post="ct2_convert", size_gb=3.2),),
        options={
            "compute_type": "int8",
            "beam_size": 5,
            "no_repeat_ngram_size": 5,
            "condition_on_previous_text": False,
            "initial_prompt": None,
            "language": "ja",
            "vad_filter": False,
        },
        fallback="whisper_large_v3_turbo",
        mock="mock_asr",
        description="Japanese galgame ASR (CER 13.0 on unseen galgames); line<->voice verification",
        homepage="https://huggingface.co/litagin/anime-whisper",
    ),
]

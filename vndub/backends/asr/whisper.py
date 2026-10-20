"""Generic Whisper checkpoints through faster-whisper (``asr`` env, MIT)."""

from vndub.backends.base import BackendSpec, WeightSpec

_COMMON = dict(role="asr", env="asr", adapter="faster_whisper", license="MIT", commercial_ok=True, cpu_ok=True)

SPECS = [
    BackendSpec(
        name="whisper_large_v3_turbo",
        vram_estimate_gb=1.8,
        weights=(WeightSpec("hf", repo="mobiuslabsgmbh/faster-whisper-large-v3-turbo", size_gb=1.6),),
        options={"compute_type": "int8", "beam_size": 5, "condition_on_previous_text": False, "no_repeat_ngram_size": 5},
        mock="mock_asr",
        description="Whisper large-v3-turbo int8: fallback / second-opinion ASR",
        homepage="https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        **_COMMON,
    ),
    BackendSpec(
        name="whisper_ru",
        vram_estimate_gb=3.0,
        weights=(WeightSpec("hf", repo="antony66/whisper-large-v3-russian", post="ct2_convert", size_gb=6.2),),
        options={"compute_type": "int8", "beam_size": 5, "language": "ru", "condition_on_previous_text": False},
        mock="mock_asr",
        description="Russian fine-tune of Whisper large-v3: CER fallback for scoring",
        homepage="https://huggingface.co/antony66/whisper-large-v3-russian",
        **_COMMON,
    ),
    BackendSpec(
        name="whisper_tiny",
        vram_estimate_gb=0.3,
        weights=(WeightSpec("hf", repo="Systran/faster-whisper-tiny", size_gb=0.08),),
        options={"compute_type": "int8", "beam_size": 1, "condition_on_previous_text": False},
        profiles=("test",),
        mock="mock_asr",
        description="Whisper tiny multilingual: CPU test profile (link + CER)",
        homepage="https://huggingface.co/Systran/faster-whisper-tiny",
        heavy=False,
        **_COMMON,
    ),
]

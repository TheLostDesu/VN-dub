"""Dispatch table of the ``asr`` env (faster-whisper, GigaAM, silero VAD, ruaccent)."""

TABLE = {
    "faster_whisper": "adapters.asr.whisper_impl:FasterWhisperASR",
    "gigaam": "adapters.asr.gigaam_impl:GigaAMASR",
    "silero_vad": "adapters.asr.vad_impl:SileroVAD",
    "ruaccent": "adapters.asr.stress_impl:RuAccentStress",
}

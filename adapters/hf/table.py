"""Dispatch table of the ``hf`` env (Transformers-based models)."""

TABLE = {
    "anime_ser": "adapters.hf.ser_impl:AnimeSER",
    "audeering_adv": "adapters.hf.ser_impl:AudeeringADV",
    "qwen3_aligner": "adapters.hf.align_impl:Qwen3Aligner",
    "ctc_align": "adapters.hf.align_impl:CTCAligner",
    "qwen3_asr": "adapters.hf.asr_impl:Qwen3ASR",
    "qwen2_audio": "adapters.hf.caption_impl:Qwen2AudioCaptioner",
    "qwen3_tts": "adapters.hf.qwen3_tts_impl:Qwen3TTS",
    "qwen3_tts_vd": "adapters.hf.qwen3_tts_impl:Qwen3TTSVoiceDesign",
    "voxcpm2": "adapters.hf.voxcpm_impl:VoxCPM2TTS",
}

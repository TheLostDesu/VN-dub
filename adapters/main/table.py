"""Dispatch table of the ``main`` env.

One environment (``envs/main.txt``) runs the speech-recognition, Transformers and scoring adapters that used to
live in the ``asr``, ``hf`` and ``score`` envs: they share torch 2.8.0 cu128 and numpy 2.x, so one venv means
one torch download and one copy on disk. The implementation modules keep their packages (``adapters.asr``,
``adapters.hf``, ``adapters.score``); only the interpreter is shared.
"""

TABLE = {
    # speech recognition, VAD, stress (adapters/asr)
    "faster_whisper": "adapters.asr.whisper_impl:FasterWhisperASR",
    "gigaam": "adapters.asr.gigaam_impl:GigaAMASR",
    "silero_vad": "adapters.asr.vad_impl:SileroVAD",
    "ruaccent": "adapters.asr.stress_impl:RuAccentStress",
    # Transformers models (adapters/hf)
    "anime_ser": "adapters.hf.ser_impl:AnimeSER",
    "audeering_adv": "adapters.hf.ser_impl:AudeeringADV",
    "qwen3_aligner": "adapters.hf.align_impl:Qwen3Aligner",
    "ctc_align": "adapters.hf.align_impl:CTCAligner",
    "qwen3_asr": "adapters.hf.asr_impl:Qwen3ASR",
    "qwen2_audio": "adapters.hf.caption_impl:Qwen2AudioCaptioner",
    "qwen3_tts": "adapters.hf.qwen3_tts_impl:Qwen3TTS",
    "qwen3_tts_vd": "adapters.hf.qwen3_tts_impl:Qwen3TTSVoiceDesign",
    "voxcpm2": "adapters.hf.voxcpm_impl:VoxCPM2TTS",
    # quality scoring (adapters/score)
    "utmosv2": "adapters.score.mos_impl:UTMOSv2",
    "utmos22": "adapters.score.mos_impl:UTMOS22",
    "audiobox": "adapters.score.aesthetics_impl:AudioboxAesthetics",
    "ecapa": "adapters.score.speaker_impl:EcapaEmbedder",
    "wavlm_sv": "adapters.score.speaker_impl:WavLMSV",
    "torchcrepe": "adapters.score.f0_impl:TorchCrepeF0",
}

"""Dispatch table of the ``funasr`` env."""

TABLE = {
    "emotion2vec": "adapters.funasr.e2v_impl:Emotion2Vec",
    "sensevoice": "adapters.funasr.sensevoice_impl:SenseVoice",
    "cosyvoice3": "adapters.funasr.cosyvoice_impl:CosyVoice3TTS",
}

"""Dispatch table of the ``score`` env."""

TABLE = {
    "utmosv2": "adapters.score.mos_impl:UTMOSv2",
    "utmos22": "adapters.score.mos_impl:UTMOS22",
    "audiobox": "adapters.score.aesthetics_impl:AudioboxAesthetics",
    "ecapa": "adapters.score.speaker_impl:EcapaEmbedder",
    "wavlm_sv": "adapters.score.speaker_impl:WavLMSV",
    "torchcrepe": "adapters.score.f0_impl:TorchCrepeF0",
}

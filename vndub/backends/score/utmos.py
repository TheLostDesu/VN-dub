"""Naturalness MOS predictors (``score`` env, MIT)."""

from vndub.backends.base import BackendSpec, WeightSpec

SPECS = [
    BackendSpec(
        name="utmosv2",
        role="mos",
        env="main",
        adapter="utmosv2",
        license="MIT",
        commercial_ok=True,
        vram_estimate_gb=1.5,
        cpu_ok=True,
        weights=(WeightSpec("env", subdir="utmosv2", post="utmosv2_pretrained", size_gb=0.6),),
        fallback="utmos22",
        mock="mock_mos",
        description="UTMOSv2 (VoiceMOS 2024 winner); trained on synthetic speech",
        homepage="https://github.com/sarulab-speech/UTMOSv2",
    ),
    BackendSpec(
        name="utmos22",
        role="mos",
        env="main",
        adapter="utmos22",
        license="MIT",
        commercial_ok=True,
        vram_estimate_gb=0.5,
        cpu_ok=True,
        weights=(WeightSpec("env", subdir="speechmos", post="speechmos_utmos22", size_gb=0.4),),
        profiles=("default", "test"),
        mock="mock_mos",
        description="UTMOS22 strong via tarepan/SpeechMOS (torch.hub); CPU test profile",
        homepage="https://github.com/tarepan/SpeechMOS",
        heavy=False,
    ),
]

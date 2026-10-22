"""Light analysis backends: F0, VAD, Russian stress.

Remote ones run in existing envs (torchcrepe in ``score``, silero VAD and
ruaccent in ``asr``); local ones are pure core DSP (always available).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from vndub.audio import features, vad
from vndub.backends.base import BackendSpec, LocalBackend, WeightSpec

SPECS = [
    BackendSpec(
        name="torchcrepe",
        role="f0",
        env="main",
        adapter="torchcrepe",
        license="MIT",
        commercial_ok=True,
        vram_estimate_gb=0.5,
        cpu_ok=True,
        weights=(WeightSpec("builtin"),),
        options={"model": "full", "fmin": 50.0, "fmax": 1100.0, "periodicity_threshold": 0.21},
        fallback="librosa_f0",
        mock="librosa_f0",
        description="CREPE F0 (pip torchcrepe; weights bundled)",
        homepage="https://github.com/maxrmorrison/torchcrepe",
    ),
    BackendSpec(
        name="librosa_f0",
        role="f0",
        env=None,
        adapter="librosa_f0",
        license="ISC (librosa)",
        commercial_ok=True,
        vram_estimate_gb=0.0,
        heavy=False,
        cpu_ok=True,
        profiles=("default", "test"),
        local_impl="vndub.backends.dsp:LibrosaF0",
        description="pYIN F0 on CPU (dependency-free fallback)",
    ),
    BackendSpec(
        name="silero",
        role="vad",
        env="main",
        adapter="silero_vad",
        license="MIT",
        commercial_ok=True,
        vram_estimate_gb=0.0,
        heavy=False,
        cpu_ok=True,
        weights=(WeightSpec("builtin"),),
        options={"threshold": 0.5, "min_silence_duration_ms": 120, "speech_pad_ms": 20},
        fallback="energy_vad",
        mock="energy_vad",
        description="silero-vad (ONNX bundled with faster-whisper)",
        homepage="https://github.com/snakers4/silero-vad",
    ),
    BackendSpec(
        name="energy_vad",
        role="vad",
        env=None,
        adapter="energy_vad",
        license="MIT",
        commercial_ok=True,
        vram_estimate_gb=0.0,
        heavy=False,
        cpu_ok=True,
        profiles=("default", "test"),
        local_impl="vndub.backends.dsp:EnergyVAD",
        description="Energy-threshold VAD (core fallback)",
    ),
    BackendSpec(
        name="ruaccent",
        role="stress",
        env="main",
        adapter="ruaccent",
        license="MIT",
        commercial_ok=True,
        vram_estimate_gb=0.0,
        heavy=False,
        cpu_ok=True,
        weights=(WeightSpec("env", subdir="ruaccent", post="ruaccent_models", size_gb=0.6),),
        options={"model": "turbo", "use_dictionary": True},
        profiles=("default", "test"),
        mock="mock_stress",
        description="Russian stress placement + yo-fication (ONNX, CPU)",
        homepage="https://github.com/Den4ikAI/ruaccent",
    ),
]


class LibrosaF0(LocalBackend):
    def op_f0(self, item: dict[str, Any], **_: Any) -> dict[str, Any]:
        y, sr = features.load(item["wav"], sr=16000)
        f0 = features.f0_pyin(y, sr)
        out = Path(item["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, f0)
        return {"path": str(out), "hop_s": features.HOP_S, "n": int(f0.size)}


class EnergyVAD(LocalBackend):
    def op_vad(self, item: dict[str, Any], **_: Any) -> dict[str, Any]:
        y, sr = features.load(item["wav"], sr=16000)
        segs = vad.energy_vad(y, sr)
        return {"segments": [list(s) for s in segs], "duration": len(y) / sr}

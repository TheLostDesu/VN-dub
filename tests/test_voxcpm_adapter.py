"""adapters/hf/voxcpm_impl.py against the real ``VoxCPM.generate`` signature (voxcpm 2.0.3).

``generate(text, prompt_wav_path, prompt_text, reference_wav_path, cfg_value, inference_timesteps, min_len, max_len,
normalize, denoise, retry_badcase*)`` has no ``seed`` and no ``**kwargs``: the adapter used to pass ``seed=item["seed"]``
and every item (synth always sets an int seed) failed with TypeError. The stub below has exactly that signature, so any
keyword the adapter invents raises the same TypeError here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from adapters.hf.voxcpm_impl import VoxCPM2TTS


class _StubVoxCPM:
    """Same parameters as voxcpm 2.0.3 ``VoxCPM.generate`` (streaming is added by the real wrapper)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        text: str,
        prompt_wav_path: str | None = None,
        prompt_text: str | None = None,
        reference_wav_path: str | None = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        min_len: int = 2,
        max_len: int = 4096,
        normalize: bool = False,
        denoise: bool = False,
        retry_badcase: bool = True,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
    ) -> np.ndarray:
        self.calls.append({"text": text, "prompt_wav_path": prompt_wav_path, "prompt_text": prompt_text, "reference_wav_path": reference_wav_path})
        return np.zeros(480, dtype="float32")


def _impl(**options: Any) -> tuple[VoxCPM2TTS, _StubVoxCPM]:
    impl = VoxCPM2TTS({"name": "voxcpm2", "options": options, "device": "cpu"})
    impl.model, impl.sr = _StubVoxCPM(), 48000
    return impl, impl.model


def test_generate_passes_no_seed() -> None:
    impl, model = _impl()
    wav, sr = impl.generate({"text": "Привет.", "timbre_ref": "ref.wav", "seed": 1234, "text_prefix": "(calm)"})
    assert sr == 48000 and len(wav) == 480
    assert model.calls == [{"text": "(calm)Привет.", "prompt_wav_path": None, "prompt_text": None, "reference_wav_path": "ref.wav"}]


def test_generate_ultimate_cloning_passes_prompt_pair_and_no_seed() -> None:
    impl, model = _impl(ultimate=True)
    impl.generate({"text": "Привет.", "timbre_ref": "ref.wav", "timbre_ref_text": "Это пример.", "seed": 7})
    assert model.calls[0]["prompt_wav_path"] == "ref.wav" and model.calls[0]["prompt_text"] == "Это пример."


def test_design_generate_passes_no_seed() -> None:
    impl, model = _impl()
    impl.design_generate({"text": "Привет.", "description": "(a young woman)", "seed": 99})
    assert model.calls[0]["text"] == "(a young woman)Привет." and model.calls[0]["reference_wav_path"] is None


def test_gen_forwards_only_voxcpm_keywords(monkeypatch) -> None:
    """The keyword set sent to the model is a subset of the real signature (guards against another invented keyword)."""
    import inspect

    impl, model = _impl(cfg_value=1.5, inference_timesteps=6, normalize=False)
    sent: dict[str, Any] = {}

    def spy(**kw: Any) -> np.ndarray:
        sent.update(kw)
        return np.zeros(8, dtype="float32")

    monkeypatch.setattr(model, "generate", spy)
    impl.generate({"text": "x", "timbre_ref": "r.wav", "seed": 3})
    allowed = set(inspect.signature(_StubVoxCPM.generate).parameters) - {"self"}
    assert set(sent) <= allowed and "seed" not in sent
    assert sent["cfg_value"] == 1.5 and sent["inference_timesteps"] == 6 and sent["normalize"] is False

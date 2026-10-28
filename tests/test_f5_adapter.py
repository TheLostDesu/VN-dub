"""adapters/f5/f5_impl.py: the Vocos vocoder comes from the pulled weight, never from the hub.

f5-tts 1.1.9 ``F5TTS(model, ckpt_file, vocab_file, ode_method, use_ema, vocoder_local_path, device, hf_cache_dir)``: without
``vocoder_local_path`` its ``load_vocoder`` calls ``hf_hub_download("charactr/vocos-mel-24khz")``, which cannot work with
``HF_HUB_OFFLINE=1`` (how every adapter runs), so f5_ru never loaded on a clean machine.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from adapters.common.rpc import WeightsMissingError
from adapters.f5 import f5_impl
from vndub.backends import registry
from vndub.backends.weights import MARKER, resolved


class _StubF5TTS:
    """Constructor signature of f5-tts 1.1.9 (an unknown keyword raises TypeError like the real class)."""

    last: dict[str, Any] = {}

    def __init__(self, model="F5TTS_v1_Base", ckpt_file="", vocab_file="", ode_method="euler", use_ema=True, vocoder_local_path=None, device=None, hf_cache_dir=None):
        type(self).last = {"model": model, "ckpt_file": ckpt_file, "vocab_file": vocab_file, "vocoder_local_path": vocoder_local_path, "device": device}


def _models(tmp_path: Path, pulled: tuple[str, ...]) -> Path:
    md = tmp_path / "models"
    ck = md / "F5-TTS_RUSSIAN"
    (ck / "F5TTS_v1_Base_v4_winter").mkdir(parents=True)
    (ck / "F5TTS_v1_Base").mkdir(parents=True)
    (ck / "F5TTS_v1_Base_v4_winter" / "model_212000.safetensors").write_bytes(b"x")
    (ck / "F5TTS_v1_Base" / "vocab.txt").write_text("a\n", encoding="utf-8")
    dirs = {"F5-TTS_RUSSIAN": ck}
    for name in pulled:
        dirs[name] = md / name
        dirs[name].mkdir(parents=True, exist_ok=True)
    for d in dirs.values():
        (d / MARKER).write_text("x", encoding="utf-8")
    return md


def _impl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pulled: tuple[str, ...]) -> f5_impl.F5Russian:
    spec = registry.get("f5_ru")
    params = {"name": "f5_ru", "options": dict(spec.options), "device": "cpu", "weights": resolved(_models(tmp_path, pulled), spec)}
    api = types.ModuleType("f5_tts.api")
    api.F5TTS = _StubF5TTS  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "f5_tts", types.ModuleType("f5_tts"))
    monkeypatch.setitem(sys.modules, "f5_tts.api", api)
    return f5_impl.F5Russian(params)


def test_load_passes_the_pulled_vocoder_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    impl = _impl(tmp_path, monkeypatch, ("vocos-mel-24khz",))
    info = impl.load()
    voc = Path(_StubF5TTS.last["vocoder_local_path"])
    assert voc == tmp_path / "models" / "vocos-mel-24khz" and info["vocoder"] == str(voc)
    assert _StubF5TTS.last["ckpt_file"].endswith("model_212000.safetensors") and _StubF5TTS.last["vocab_file"].endswith("vocab.txt")


def test_load_without_the_vocoder_weight_asks_for_a_pull(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    impl = _impl(tmp_path, monkeypatch, ())
    with pytest.raises(WeightsMissingError, match="vocos-mel-24khz"):
        impl.load()


def test_vocoder_dir_name_is_the_spec_weight_dir() -> None:
    from vndub.backends.weights import local_dir

    names = {local_dir(Path("m"), w, "f5_ru").name for w in registry.get("f5_ru").weights}
    assert f5_impl.VOCODER_DIR in names

"""adapters/chatterbox/chatterbox_impl.py: load order and error reporting, with stand-ins for the env's packages.

resemble-perth imports pkg_resources inside ``try: ... except ImportError: PerthImplicitWatermarker = None``, so an env
without it failed only at the end of the model build, with ``TypeError: 'NoneType' object is not callable``. The real
packages are exercised by ``tests/test_pkg_resources_compat.py`` when the env exists.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from adapters.chatterbox import chatterbox_impl
from adapters.chatterbox.chatterbox_impl import ChatterboxML

REPO = Path(__file__).resolve().parent.parent


def install_packages(monkeypatch: pytest.MonkeyPatch, from_local: Any, watermarker: Any = object) -> None:
    perth = types.ModuleType("perth")
    perth.PerthImplicitWatermarker = watermarker
    monkeypatch.setitem(sys.modules, "perth", perth)
    mtl = types.ModuleType("chatterbox.mtl_tts")

    class Model:
        pass

    Model.from_local = classmethod(from_local)
    mtl.ChatterboxMultilingualTTS = Model
    monkeypatch.setitem(sys.modules, "chatterbox", types.ModuleType("chatterbox"))
    monkeypatch.setitem(sys.modules, "chatterbox.mtl_tts", mtl)
    tokenizer = types.ModuleType("chatterbox.models.tokenizers.tokenizer")
    tokenizers = types.ModuleType("chatterbox.models.tokenizers")
    tokenizers.tokenizer = tokenizer
    monkeypatch.setitem(sys.modules, "chatterbox.models", types.ModuleType("chatterbox.models"))
    monkeypatch.setitem(sys.modules, "chatterbox.models.tokenizers", tokenizers)
    monkeypatch.setitem(sys.modules, "chatterbox.models.tokenizers.tokenizer", tokenizer)


def model_params(weights: Path) -> dict[str, Any]:
    return {
        "name": "chatterbox_ml", "model": "chatterbox_ml", "device": "cpu", "options": {},
        "weights": [{"kind": "hf", "local_dir": str(weights), "present": True}],
    }


def loaded_model(sr: int = 24000) -> Any:
    return types.SimpleNamespace(sr=sr)


def test_the_stand_in_comes_before_the_model_package(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    order: list[str] = []
    monkeypatch.setattr(chatterbox_impl, "ensure_pkg_resources", lambda: order.append("ensure"))

    def from_local(cls: Any, ckpt_dir: str, device: str) -> Any:
        order.append("from_local")
        return loaded_model()

    install_packages(monkeypatch, from_local)
    assert ChatterboxML(model_params(tmp_path)).load()["sr"] == 24000
    assert order == ["ensure", "from_local"]


def test_a_swallowed_import_error_is_raised_with_its_real_message(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The real cause reaches the log (rpc.serve reports ImportError with a hint), not the ``None`` it left behind."""
    monkeypatch.setattr(chatterbox_impl, "ensure_pkg_resources", lambda: False)
    calls: list[str] = []
    install_packages(monkeypatch, lambda cls, *a: calls.append("from_local"), watermarker=None)
    for name in ("perth.perth_net", "perth.perth_net.perth_net_implicit", "perth.perth_net.perth_net_implicit.perth_watermarker"):
        monkeypatch.setitem(sys.modules, name, None)  # importing it raises ModuleNotFoundError, as a failed import would
    with pytest.raises(ImportError, match="perth"):
        ChatterboxML(model_params(tmp_path)).load()
    assert calls == []  # before minutes of weight loading


def test_a_package_without_t3_model_gets_two_arguments_and_reports_v2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """chatterbox-tts 0.1.7: ``from_local(ckpt_dir, device)``; a TypeError inside it is not read as a missing argument."""
    (tmp_path / "t3_mtl23ls_v2.safetensors").touch()
    calls: list[tuple] = []

    def from_local(cls: Any, ckpt_dir: str, device: str) -> Any:
        calls.append((ckpt_dir, device))
        return loaded_model()

    install_packages(monkeypatch, from_local)
    info = ChatterboxML(model_params(tmp_path)).load()
    assert calls == [(str(tmp_path), "cpu")] and info["t3"] == "t3_mtl23ls_v2.safetensors"


def test_a_package_with_t3_model_gets_the_newest_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "t3_mtl23ls_v2.safetensors").touch()
    (tmp_path / "t3_mtl23ls_v3.safetensors").touch()
    calls: list[str | None] = []

    def from_local(cls: Any, ckpt_dir: str, device: str, t3_model: str | None = None) -> Any:
        calls.append(t3_model)
        return loaded_model()

    install_packages(monkeypatch, from_local)
    assert ChatterboxML(model_params(tmp_path)).load()["t3"] == "t3_mtl23ls_v3.safetensors"
    assert calls == ["t3_mtl23ls_v3.safetensors"]


def test_a_type_error_inside_the_load_is_not_retried(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "t3_mtl23ls_v3.safetensors").touch()
    calls: list[str | None] = []

    def from_local(cls: Any, ckpt_dir: str, device: str, t3_model: str | None = None) -> Any:
        calls.append(t3_model)
        raise TypeError("a real bug inside the model build")

    install_packages(monkeypatch, from_local)
    with pytest.raises(TypeError, match="real bug"):
        ChatterboxML(model_params(tmp_path)).load()
    assert calls == ["t3_mtl23ls_v3.safetensors"]


def test_the_vendor_stresser_leaves_the_marks_alone(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The pipeline sends ruaccent's marks: tokenizer.encode must not warn about a missing stresser, or stress the text again."""
    install_packages(monkeypatch, lambda cls, ckpt_dir, device: loaded_model())
    ChatterboxML(model_params(tmp_path)).load()
    assert sys.modules["chatterbox.models.tokenizers.tokenizer"].add_russian_stress("замо́к") == "замо́к"


def test_the_probe_raises_the_swallowed_import_error_without_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """Round 18: the import-level probe meets round 14's failure in seconds; no weight entry is needed."""
    monkeypatch.setattr(chatterbox_impl, "ensure_pkg_resources", lambda: False)
    install_packages(monkeypatch, lambda cls, *a: None, watermarker=None)
    for name in ("perth.perth_net", "perth.perth_net.perth_net_implicit", "perth.perth_net.perth_net_implicit.perth_watermarker"):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(ImportError, match="perth"):
        ChatterboxML.probe({"name": "chatterbox_ml", "options": {}, "weights": []})
    install_packages(monkeypatch, lambda cls, *a: None)
    assert ChatterboxML.probe({"name": "chatterbox_ml", "options": {}, "weights": []})["ok"] is True


def test_the_env_pins_a_setuptools_with_pkg_resources() -> None:
    assert "setuptools==80.10.2" in (REPO / "envs" / "chatterbox.txt").read_text(encoding="utf-8")

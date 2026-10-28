"""adapters/asr/stress_impl.py: ruaccent keeps its koziev morphology data inside its package folder, so an environment
recreated after ``models pull`` lost it and the next offline run failed with ``OfflineModeIsEnabled``. The pull now keeps
a copy next to the weights and load restores the package copy from it."""

from __future__ import annotations

from pathlib import Path

import pytest

from adapters.asr import stress_impl
from adapters.common.rpc import WeightsMissingError


def _fill(d: Path, files=stress_impl.KOZIEV_FILES) -> None:
    for f in files:
        p = d / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")


def _weights(tmp_path: Path, size: str = "turbo3.1") -> Path:
    work = tmp_path / "models" / "ruaccent"
    (work / "dictionary").mkdir(parents=True)
    (work / "nn" / "nn_omograph" / size).mkdir(parents=True)
    (work / ".vndub_complete").write_text("x", encoding="utf-8")
    return work


def test_restores_package_copy_from_workdir(tmp_path: Path) -> None:
    work, module = tmp_path / "models" / "ruaccent", tmp_path / "env" / "ruaccent" / "koziev"
    _fill(work / "koziev")
    (module / "rulemma").mkdir(parents=True)  # a half-downloaded package copy counts as missing
    (module / "rulemma" / "rulemma.dat").write_bytes(b"partial")
    assert stress_impl.sync_koziev(work, module) == "restored"
    assert all((module / f).is_file() for f in stress_impl.KOZIEV_FILES)
    assert (module / "rulemma" / "rulemma.dat").read_bytes() == b"x"
    assert stress_impl.sync_koziev(work, module) == "env"


def test_saves_workdir_copy_from_package(tmp_path: Path) -> None:
    work, module = tmp_path / "models" / "ruaccent", tmp_path / "env" / "ruaccent" / "koziev"
    _fill(module)
    assert stress_impl.sync_koziev(work, module) == "saved"
    assert all((work / "koziev" / f).is_file() for f in stress_impl.KOZIEV_FILES)
    assert stress_impl.sync_koziev(work, module) == "env"


def test_neither_copy_is_weights_missing(tmp_path: Path) -> None:
    work, module = tmp_path / "models" / "ruaccent", tmp_path / "env" / "ruaccent" / "koziev"
    with pytest.raises(WeightsMissingError) as ei:
        stress_impl.sync_koziev(work, module)
    assert "koziev" in str(ei.value) and "models pull ruaccent" in str(ei.value)


def test_load_checks_koziev_before_ruaccent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work = _weights(tmp_path)
    module = tmp_path / "env" / "ruaccent" / "koziev"
    monkeypatch.setattr(stress_impl, "_module_koziev", lambda: module)
    calls: list[tuple] = []
    monkeypatch.setattr(stress_impl, "_load", lambda *a: calls.append(a) or object())
    impl = stress_impl.RuAccentStress({"name": "ruaccent", "weights": [{"local_dir": str(work), "present": True}]})
    with pytest.raises(WeightsMissingError):
        impl.load()
    assert calls == [], "ruaccent must not be asked to load: it would reach for the hub"
    _fill(work / "koziev")
    info = impl.load()
    assert info["koziev"] == "restored" and len(calls) == 1 and calls[0][0] == work


def test_pull_keeps_a_copy_next_to_the_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work = tmp_path / "models" / "ruaccent"
    module = tmp_path / "env" / "ruaccent" / "koziev"
    monkeypatch.setattr(stress_impl, "_module_koziev", lambda: module)
    monkeypatch.setattr(stress_impl, "_load", lambda *a: _fill(module))  # what the online load does
    res = stress_impl.RuAccentStress.pull({"name": "ruaccent", "local_dir": str(work), "options": {"model": "turbo"}})
    assert res["ok"] and "turbo3.1" in res["detail"] and "saved" in res["detail"]
    assert all((work / "koziev" / f).is_file() for f in stress_impl.KOZIEV_FILES)

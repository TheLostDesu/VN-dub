"""JSON-RPC adapter protocol: a real subprocess running tests/fake_adapter.py."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from vndub.backends.base import BackendSpec, BackendUnavailable, WeightsMissing
from vndub.backends.runner import AdapterCrashed, AdapterError, AdapterProcess, AdapterTimeout, RemoteBackend

REPO = Path(__file__).resolve().parent.parent
TESTS = REPO / "tests"


def _proc(env: str, name: str, root: Path | None = None, log_path: Path | None = None, extra_env: dict | None = None) -> AdapterProcess:
    env_vars = dict(extra_env or {})
    env_vars["PYTHONPATH"] = os.pathsep.join([str(TESTS), str(REPO)])
    return AdapterProcess(env, name, root=REPO, log_path=log_path, extra_env=env_vars, python=Path(sys.executable), module="fake_adapter")


def _spec(adapter: str = "echo", **opts) -> BackendSpec:
    return BackendSpec(name=f"fake_{adapter}", role="tts", env="fake", adapter=adapter, license="MIT", commercial_ok=True, vram_estimate_gb=0.1, cpu_ok=True, options=opts)


def _backend(tmp_path: Path, adapter: str = "echo", **opts) -> RemoteBackend:
    return RemoteBackend(_spec(adapter, **opts), device="cpu", models_dir=tmp_path / "models", log_dir=tmp_path / "logs", root=REPO, process_factory=_proc)


def test_load_run_unload(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    info = b.load()
    assert info["loaded"] is True and info["device"] == "cpu"
    res = b.call("echo", [{"id": "a", "text": "привет"}, {"id": "b", "text": "こんにちは"}], lang="ru")
    assert [r["echo"] for r in res] == ["привет", "こんにちは"]
    assert res[0]["kw"] == {"lang": "ru"}
    assert b.peak_vram() is None  # no CUDA in the test interpreter
    b.unload()
    assert not b.proc.alive
    # adapter stderr (incl. the stray print) went to the log, not into the protocol
    log = (tmp_path / "logs" / "adapter_fake.log").read_text(encoding="utf-8", errors="replace")
    assert "noise on stdout" in log


def test_item_errors_do_not_fail_batch(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    res = b.call("fail", [{"id": "1", "text": "good"}, {"id": "2", "text": "bad"}])
    assert res[0] == {"id": "1", "ok": True}
    assert "bad item" in res[1]["error"]
    b.unload()


def test_unknown_op_is_unavailable(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    with pytest.raises(BackendUnavailable, match="does not support op"):
        b.call("nope", [{"id": "x"}])
    b.unload()


def test_weights_missing_maps_to_clear_error(tmp_path: Path) -> None:
    b = _backend(tmp_path, "nowts")
    with pytest.raises(WeightsMissing) as ei:
        b.load()
    assert "vndub models pull fake_nowts" in str(ei.value)
    b.unload()


def test_unavailable_maps_with_hint(tmp_path: Path) -> None:
    b = _backend(tmp_path, "unavail")
    with pytest.raises(BackendUnavailable) as ei:
        b.load()
    assert "needs CUDA" in str(ei.value) and "use a GPU" in str(ei.value)
    b.unload()


def test_unknown_adapter_key(tmp_path: Path) -> None:
    b = _backend(tmp_path, "missing_key")
    with pytest.raises(BackendUnavailable, match="has no adapter"):
        b.load()
    b.unload()


def test_crash_restarts_once_and_retries(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    b.load()
    pid0 = b.call("pid")["pid"]
    marker = tmp_path / "crash.marker"
    res = b.call("crash_once", [{"id": "c"}], marker=str(marker))
    assert res[0]["recovered"] is True
    assert res[0]["pid"] != pid0  # served by the restarted process
    assert marker.exists()
    b.unload()


def test_timeout_kills_process(tmp_path: Path) -> None:
    p = _proc("fake", "fake_echo")
    p.start()
    p.request("load", {"name": "fake_echo", "model": "echo", "options": {}, "device": "cpu"}, timeout=60)
    with pytest.raises(AdapterTimeout):
        p.request("run", {"op": "sleep", "items": [{"id": "s"}], "seconds": 5.0}, timeout=1.0)
    assert not p.alive
    with pytest.raises(AdapterCrashed):
        p.request("peak_vram", {}, timeout=5)
    p.close()


def test_selftest_over_rpc(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    res = b.selftest()
    assert res["ok"] is True and res["op"] == "echo"
    assert "load_s" in res and "peak_vram_gb" in res
    assert not b.proc.alive


def test_env_missing_is_unavailable(tmp_path: Path) -> None:
    spec = BackendSpec(name="x", role="tts", env="no_such_env", adapter="echo", license="MIT", commercial_ok=True, vram_estimate_gb=0.1)
    b = RemoteBackend(spec, device="cpu", root=tmp_path)
    with pytest.raises(BackendUnavailable, match="no_such_env"):
        b.load()


def test_protocol_errors(tmp_path: Path) -> None:
    p = _proc("fake", "fake_echo")
    p.start()
    with pytest.raises(BackendUnavailable, match="no model loaded"):
        p.request("run", {"op": "echo", "items": []}, timeout=30)
    with pytest.raises(AdapterError, match="unknown method"):
        p.request("bogus", {}, timeout=30)
    assert p.alive  # errors never kill the server
    p.close()

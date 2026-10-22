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


def test_adapter_log_lines_carry_the_time_and_the_adapter_name(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    import re
    import time

    caplog.set_level("INFO", logger="vndub.backends")
    b = _backend(tmp_path)
    b.load()
    b.call("echo", [{"id": "a", "text": "hi"}])
    b.unload()
    path = tmp_path / "logs" / "adapter_fake.log"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and "=== exited" not in path.read_text(encoding="utf-8", errors="replace"):
        time.sleep(0.05)  # the reader thread writes the exit line once the pipe closes
    lines = [l for l in path.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    assert lines and all(re.match(r"\d\d:\d\d:\d\d \[fake_echo\] ", l) for l in lines), lines
    body = [l.split("] ", 1)[1] for l in lines]
    assert re.fullmatch(r"=== started pid \d+: .+ -m fake_adapter", body[0])
    assert "noise on stdout must not break the protocol" in body  # the adapter's own stderr, one line each
    assert re.fullmatch(r"=== exited code 0 after \d+\.\ds", body[-1])
    # the same facts, one line each, in the app log (logger vndub.backends.runner)
    msgs = [r.getMessage() for r in caplog.records if r.name == "vndub.backends.runner"]
    assert any(re.fullmatch(r"adapter fake_echo started pid \d+: .+ -m fake_adapter \(env fake\)", m) for m in msgs), msgs
    assert any(re.fullmatch(r"adapter fake_echo exited code 0 after \d+\.\ds \(pid \d+\)", m) for m in msgs), msgs
    assert any(re.fullmatch(r"backend fake_echo loaded in \d+\.\ds \(device cpu\)", m) for m in msgs), msgs


def test_a_crash_is_logged_as_a_warning_and_an_abort_is_not(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    import logging
    import time

    caplog.set_level("INFO", logger="vndub.backends")
    b = _backend(tmp_path)
    b.load()
    b.call("crash_once", [{"id": "c"}], marker=str(tmp_path / "crash.marker"))
    b.abort()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not any("exited code 1" in r.getMessage() for r in caplog.records):
        time.sleep(0.05)
    b.unload()
    recs = [(r.levelno, r.getMessage()) for r in caplog.records if r.name == "vndub.backends.runner"]
    warns = [m for lv, m in recs if lv == logging.WARNING]
    assert any(m.startswith("adapter fake_echo crashed (") and "restarting it once" in m for m in warns), warns
    assert any(m.startswith("adapter fake_echo exited code 3 after ") for m in warns), warns  # the crash: os._exit(3)
    assert any(lv == logging.INFO and m.startswith("adapter fake_echo (pid ") and "abort: killing its process tree" in m for lv, m in recs), recs


class _CountingProc(AdapterProcess):
    starts: list[int] = []

    def start(self) -> None:
        was_alive = self.alive
        super().start()
        if not was_alive:
            _CountingProc.starts.append(self.proc.pid)  # type: ignore[union-attr]


def _counting_backend(tmp_path: Path) -> RemoteBackend:
    _CountingProc.starts = []

    def factory(env: str, name: str, root: Path | None = None, log_path: Path | None = None, extra_env: dict | None = None) -> AdapterProcess:
        env_vars = dict(extra_env or {})
        env_vars["PYTHONPATH"] = os.pathsep.join([str(TESTS), str(REPO)])
        return _CountingProc(env, name, root=REPO, log_path=log_path, extra_env=env_vars, python=Path(sys.executable), module="fake_adapter")

    return RemoteBackend(_spec("echo"), device="cpu", models_dir=tmp_path / "models", log_dir=tmp_path / "logs", root=REPO, process_factory=factory)


def test_abort_during_a_request_raises_aborted_and_starts_no_second_process(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    import threading
    import time

    from vndub.backends.runner import AdapterAborted

    caplog.set_level("INFO", logger="vndub.backends")
    b = _counting_backend(tmp_path)
    b.load()
    assert len(_CountingProc.starts) == 1
    out: dict = {}

    def slow() -> None:
        t0 = time.monotonic()
        try:
            b.call("sleep", [{"id": "s"}], seconds=60.0)
        except BaseException as e:
            out["exc"] = e
        out["waited"] = time.monotonic() - t0

    t = threading.Thread(target=slow, daemon=True)
    t.start()
    time.sleep(1.0)  # the request is in flight inside the adapter
    t_abort = time.monotonic()
    b.abort()
    t.join(15)
    assert not t.is_alive()
    try:
        assert isinstance(out.get("exc"), AdapterAborted), out
        assert isinstance(out["exc"], AdapterCrashed)  # callers that handle crashes still see one
        assert time.monotonic() - t_abort < 5  # fast: the 60 s request is not waited for
        assert len(_CountingProc.starts) == 1  # and no restart: nothing re-sent, nothing left behind
        assert any(r.getMessage().startswith("adapter fake_echo aborted (pid ") and r.getMessage().endswith("request run dropped") and r.levelname == "INFO" for r in caplog.records)
        # a call that arrives after the abort (same registry, same job) fails at once without starting a process
        t0 = time.monotonic()
        with pytest.raises(AdapterAborted):
            b.call("echo", [{"id": "x", "text": "late"}])
        with pytest.raises(AdapterAborted):
            b.load()
        assert time.monotonic() - t0 < 1 and len(_CountingProc.starts) == 1
    finally:
        b.unload()
    # unload() clears the abort: the backend works again, with a new process
    res = b.call("echo", [{"id": "y", "text": "again"}])
    assert res[0]["echo"] == "again" and len(_CountingProc.starts) == 2
    b.unload()


def test_a_real_crash_is_still_restarted_and_not_mistaken_for_an_abort(tmp_path: Path) -> None:
    b = _counting_backend(tmp_path)
    b.load()
    res = b.call("crash_once", [{"id": "c"}], marker=str(tmp_path / "crash.marker"))
    assert res[0]["recovered"] is True and len(_CountingProc.starts) == 2
    b.unload()


def test_abort_of_a_process_request_reports_aborted_and_a_new_start_clears_it(tmp_path: Path) -> None:
    import threading
    import time

    from vndub.backends.runner import AdapterAborted

    p = _proc("fake", "fake_echo")
    p.start()
    p.request("load", {"name": "fake_echo", "model": "echo", "options": {}, "device": "cpu"}, timeout=60)
    out: dict = {}

    def slow() -> None:
        try:
            p.request("run", {"op": "sleep", "items": [{"id": "s"}], "seconds": 60.0}, timeout=120)
        except BaseException as e:
            out["exc"] = e

    t = threading.Thread(target=slow, daemon=True)
    t.start()
    time.sleep(1.0)
    p.abort()
    t.join(15)
    assert isinstance(out.get("exc"), AdapterAborted) and p.aborted
    with pytest.raises(AdapterAborted):
        p.request("peak_vram", {}, timeout=5)  # not "adapter process is not running": the cause is known
    p.close()
    assert not p.aborted
    p.start()  # a process the caller decided to start is a clean slate
    assert p.alive and not p.aborted
    p.close()

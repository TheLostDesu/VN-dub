"""Child-process lines of the llama.cpp backend: the server log markers and the app log (no binary, no GPU)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from vndub.backends.llm import llamacpp as lc


class FakePopen:
    """Stands in for the llama-server process: writes the offload line into the log it was handed, like the real one."""

    last: "FakePopen | None" = None

    def __init__(self, args: list[str], stdout: Any = None, stderr: Any = None, **_: Any) -> None:
        self.args = args
        self.pid = 4242
        self.returncode: int | None = None
        self.killed = False
        if stdout not in (None, lc.subprocess.DEVNULL):
            stdout.write(b"load_tensors: offloaded 49/49 layers to GPU\n")
            stdout.flush()
        FakePopen.last = self

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = 1

    def wait(self, timeout: float | None = None) -> int:
        return int(self.returncode or 0)


class Health:
    status = 200

    def __enter__(self) -> "Health":
        return self

    def __exit__(self, *a: object) -> None:
        return None


class Tools:
    def __init__(self, binary: Path) -> None:
        self.binary = binary

    def resolve(self, name: str) -> Path:
        return self.binary


def _backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> lc.LlamaCppBackend:
    gguf = tmp_path / "x.gguf"
    gguf.write_bytes(b"gguf")
    monkeypatch.setattr(lc.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(lc.urllib.request, "urlopen", lambda *a, **k: Health())
    monkeypatch.setattr(lc, "server_help", lambda binary: "")
    ctx = {"device": "cuda:0", "log_dir": tmp_path / "logs", "tools": Tools(tmp_path / "llama-server.exe"), "models_dir": tmp_path}
    return lc.LlamaCppBackend(lc.SPECS[0], {"gguf": str(gguf), "port": 0}, ctx)


def test_server_log_gets_a_dated_header_and_a_stopped_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger="vndub.backends")
    b = _backend(tmp_path, monkeypatch)
    info = b.load()
    assert info["gpu_layers"] == "49/49"  # parse_offload / _read_from still read the log from the end of the header
    log = tmp_path / "logs" / "llama_server.log"
    text = log.read_text(encoding="utf-8")
    header = text.strip().splitlines()[0]
    assert re.fullmatch(r"=== \d{4}-\d\d-\d\d \d\d:\d\d:\d\d .*-m .*x\.gguf .*--port \d+.*", header), header
    assert lc.parse_offload(text) == (49, 49)
    b.unload()
    last = log.read_text(encoding="utf-8").rstrip().splitlines()[-1]
    assert re.fullmatch(r"=== stopped \d\d:\d\d:\d\d \(exit code 0\)", last), last
    assert lc.parse_offload(log.read_text(encoding="utf-8")) == (49, 49)  # the markers do not disturb the offload check
    msgs = [r.getMessage() for r in caplog.records if r.name == "vndub.backends.llm.llamacpp"]
    assert any(re.fullmatch(r"llama-server started pid 4242 port \d+ gguf .*x\.gguf ctx 16384 parallel 1 \(gpu\)", m) for m in msgs), msgs
    assert any(re.fullmatch(r"llama-server ready pid 4242 port \d+ gpu layers 49/49 in \d+\.\ds", m) for m in msgs), msgs
    assert "llama-server stopped pid 4242 exit code 0" in msgs


def test_a_second_start_appends_its_own_header(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    b = _backend(tmp_path, monkeypatch)
    b.load()
    b.unload()
    b.load()
    text = (tmp_path / "logs" / "llama_server.log").read_text(encoding="utf-8")
    assert len(re.findall(r"^=== \d{4}-\d\d-\d\d ", text, re.M)) == 2 and text.count("=== stopped") == 1
    b.unload()


def test_a_killed_server_is_logged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger="vndub.backends")
    b = _backend(tmp_path, monkeypatch)
    b.load()
    b.abort()
    assert FakePopen.last is not None and FakePopen.last.killed
    assert any(r.levelname == "WARNING" and r.getMessage() == "llama-server pid 4242 killed (job cancelled)" for r in caplog.records)
    b.unload()

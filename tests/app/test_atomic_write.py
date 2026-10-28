"""``write_json_atomic`` / ``replace_atomic``: a reader that has the file open must not make the writer fail (Windows sharing violation)."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest
import yaml

from vndub_app import appdirs, installer


def _hammer_reader(path: Path, stop: threading.Event, reads: list[int]) -> None:
    """Like ``load_settings`` on every request: open, read, close, over and over (the files are tiny, so ~every 5 ms)."""
    while not stop.is_set():
        try:
            path.read_text(encoding="utf-8")
            reads[0] += 1
        except OSError:
            pass  # the file is being replaced: a reader may miss it for a moment
        time.sleep(0.002)


def test_writes_succeed_while_another_thread_keeps_reading_the_file(tmp_path: Path) -> None:
    f = tmp_path / "settings.json"
    appdirs.write_json_atomic(f, {"n": -1})
    stop, reads = threading.Event(), [0]
    t = threading.Thread(target=_hammer_reader, args=(f, stop, reads))
    t.start()
    failures: list[str] = []
    try:
        for i in range(300):
            try:
                appdirs.write_json_atomic(f, {"n": i})
            except OSError as e:
                failures.append(f"{i}: {e}")
    finally:
        stop.set()
        t.join(10)
    assert reads[0] > 20 and failures == []
    assert json.loads(f.read_text(encoding="utf-8")) == {"n": 299}
    assert not list(tmp_path.glob(".*.tmp"))  # no stray temp file


def test_replace_atomic_gives_up_after_its_tries_and_removes_the_temp_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def locked(src: object, dst: object) -> None:
        calls.append(1)
        raise PermissionError(13, "Permission denied", str(dst))

    monkeypatch.setattr(os, "replace", locked)
    monkeypatch.setattr(appdirs, "REPLACE_PAUSE_S", 0.0)
    f = tmp_path / "x.json"
    with pytest.raises(PermissionError):
        appdirs.write_json_atomic(f, {"a": 1})
    assert len(calls) == appdirs.REPLACE_TRIES
    assert not f.exists() and not list(tmp_path.glob(".*.tmp"))  # the temp file is gone


def test_replace_atomic_succeeds_when_the_lock_goes_away(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = os.replace
    left = [3]

    def flaky(src: object, dst: object) -> None:
        if left[0] > 0:
            left[0] -= 1
            raise PermissionError(13, "sharing violation", str(dst))
        real(src, dst)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "replace", flaky)
    monkeypatch.setattr(appdirs, "REPLACE_PAUSE_S", 0.001)
    f = tmp_path / "x.json"
    appdirs.write_json_atomic(f, {"a": 1})
    assert json.loads(f.read_text(encoding="utf-8")) == {"a": 1} and left[0] == 0


def test_a_non_permission_error_is_not_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def broken(src: object, dst: object) -> None:
        calls.append(1)
        raise FileNotFoundError(2, "gone", str(dst))

    monkeypatch.setattr(os, "replace", broken)
    with pytest.raises(FileNotFoundError):
        appdirs.write_json_atomic(tmp_path / "x.json", {})
    assert calls == [1] and not list(tmp_path.glob(".*.tmp"))


def test_update_local_tools_waits_out_a_reader_too(app_env: Path) -> None:
    """The installer rewrites ``tools.local.yaml`` after a download; a config refresh reading it must not fail the install."""
    f = appdirs.tools_local_yaml()
    f.parent.mkdir(parents=True, exist_ok=True)
    installer.update_local_tools({"ffmpeg": "C:/x/ffmpeg.exe"})
    stop, reads = threading.Event(), [0]
    t = threading.Thread(target=_hammer_reader, args=(f, stop, reads))
    t.start()
    failures: list[str] = []
    try:
        for i in range(200):
            try:
                installer.update_local_tools({f"tool{i % 5}": f"C:/x/{i}.exe"})
            except OSError as e:
                failures.append(f"{i}: {e}")
    finally:
        stop.set()
        t.join(10)
    assert failures == [] and reads[0] > 20
    assert yaml.safe_load(f.read_text(encoding="utf-8"))["tools"]["ffmpeg"] == "C:/x/ffmpeg.exe"
    assert not list(f.parent.glob("*.tmp"))

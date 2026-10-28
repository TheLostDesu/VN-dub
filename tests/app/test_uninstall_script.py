"""The PowerShell script the Inno Setup uninstaller writes and runs to close a running VNDub (``StopRunningApp`` in packaging/VNDub.iss).

The script is cut out of the .iss and run against stand-ins for the app and its reaper (``VNDub.exe --reaper``, the helper that
outlives the app for a few seconds): the reaper must not make the uninstaller wait the full 15 s of the app's timeout and
must not be killed before the app is gone.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
windows_only = pytest.mark.skipif(not sys.platform.startswith("win") or shutil.which("powershell") is None, reason="needs Windows PowerShell")


def stop_script() -> str:
    """The text ``StopRunningApp`` saves as ``vndub-stop.ps1``: the Pascal string literals of ``Script :=`` joined with CR LF."""
    iss = (ROOT / "packaging" / "VNDub.iss").read_bytes().decode("utf-8-sig")
    block = iss[iss.index("  Script :=\n") : iss.index("  Ps := ExpandConstant")]
    lines = []
    for ln in block.splitlines()[1:]:
        m = re.match(r"\s+'(.*)'(?: \+ #13#10)?(?: \+|;)?\s*$", ln)
        assert m, f"not a string literal line: {ln!r}"
        lines.append(m.group(1).replace("''", "'"))
    return "\r\n".join(lines) + "\r\n"


def test_the_script_keeps_its_safety_properties() -> None:
    text = stop_script()
    assert "CloseMainWindow" in text and "ExecutablePath -eq $Exe" in text and "taskkill" not in text.lower()
    assert "--reaper" in text  # the reaper is told apart by its command line


@pytest.fixture(scope="module")
def fake_app(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """``<tmp>/app/Scripts/VNDub.exe``: the launcher of a venv under the name of the app (a python that can run a script)."""
    root = tmp_path_factory.mktemp("uninstall")
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(root / "app")], check=True, capture_output=True, timeout=120)
    exe = root / "app" / "Scripts" / "VNDub.exe"
    shutil.copy(root / "app" / "Scripts" / "python.exe", exe)
    return exe


@windows_only
def test_the_reaper_is_neither_waited_for_like_the_app_nor_killed_early(fake_app: Path, tmp_path: Path) -> None:
    script = tmp_path / "vndub-stop.ps1"
    script.write_text(stop_script(), encoding="utf-8")
    t0 = time.monotonic()
    app = subprocess.Popen([str(fake_app), "-c", "import time; time.sleep(2)"])  # closes by itself soon after the window is asked to
    reaper = subprocess.Popen([str(fake_app), "-c", "import time; time.sleep(5)", "--reaper", "--parent", "1"])  # ends a few seconds after the app
    time.sleep(1.0)
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(script), "-Exe", str(fake_app)],
            capture_output=True, text=True, timeout=120,
        )
        took = time.monotonic() - t0
        assert r.returncode == 0, r.stderr
        assert app.wait(5) == 0 and reaper.wait(5) == 0  # nobody was killed (Kill() would have made the exit code 1)
        assert took < 12, took  # the 15 s of the app's timeout were not spent on the reaper
        assert took >= 4.5  # the script waited for the reaper's own end (it had 5 s to live)
    finally:
        for p in (app, reaper):
            if p.poll() is None:
                p.kill()


@windows_only
def test_other_copies_of_the_app_are_left_alone(fake_app: Path, tmp_path: Path) -> None:
    other = tmp_path / "other" / "VNDub.exe"
    other.parent.mkdir()
    shutil.copy(fake_app, other)  # a launcher in a folder without its pyvenv.cfg may not start: only the path matters, so use the venv one
    script = tmp_path / "vndub-stop.ps1"
    script.write_text(stop_script(), encoding="utf-8")
    bystander = subprocess.Popen([str(fake_app), "-c", "import time; time.sleep(4)"])
    try:
        time.sleep(0.5)
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(script), "-Exe", str(other)],
            capture_output=True, text=True, timeout=120,
        )
        assert r.returncode == 0 and bystander.poll() is None  # its path is not the uninstalled one: not closed, not killed
    finally:
        bystander.kill()
        bystander.wait(10)
    assert os.path.isfile(other)

"""The built exe walks the whole wizard (``packaging/smoke_exe.py``); needs ``VNDUB_SMOKE_EXE=1`` and a build.

Run after ``packaging/build_app.ps1``. It catches what ``--selfcheck`` cannot: modules the frozen app imports only
along the way, data files PyInstaller missed, and paths that differ in the bundle (backend envs, adapters).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "dist" / "VNDub" / "VNDub.exe"

pytestmark = [pytest.mark.slow, pytest.mark.timeout(1800)]


@pytest.mark.skipif(not os.environ.get("VNDUB_SMOKE_EXE"), reason="set VNDUB_SMOKE_EXE=1 to drive the built dist/VNDub/VNDub.exe")
def test_built_exe_walks_the_wizard() -> None:
    if not EXE.is_file():
        pytest.skip("no build: run packaging/build_app.ps1")
    args = [sys.executable, str(ROOT / "packaging" / "smoke_exe.py"), str(EXE)]
    if not os.environ.get("VNDUB_NETWORK_TESTS"):
        args.append("--no-network")
    r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1500)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]

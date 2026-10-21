"""Open a folder / file in the system file manager, with one seam for tests.

``open_path`` prefers ``vndub_app.system.open_path`` (server task S) when that module exists and
falls back to the platform opener. Tests (and headless runs) replace the implementation with
``set_opener`` so nothing is ever opened for real.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

_OPENER: Callable[[Path], None] | None = None


def set_opener(fn: Callable[[Path], None] | None) -> None:
    """Replace the opener (tests); ``None`` restores the default."""
    global _OPENER
    _OPENER = fn


def _platform_open(path: Path) -> None:
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def open_path(path: Path | str) -> None:
    p = Path(path)
    if _OPENER is not None:
        _OPENER(p)
        return
    try:
        system = importlib.import_module("vndub_app.system")
        fn = getattr(system, "open_path", None)
    except Exception:
        fn = None
    if fn is not None:
        fn(p)
        return
    _platform_open(p)

"""Native OS helpers: file dialogs (pywebview), open in Explorer, PC shutdown.

The shell (``vndub_app/shell.py``) creates the pywebview window and calls
:func:`set_window`. Without a window (browser mode, tests) the dialogs report
``unsupported`` and the UI shows a plain path field instead.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger("vndub_app.system")

_WINDOW: Any = None
_LOCK = threading.Lock()


def set_window(window: Any) -> None:
    """Called by the shell once the pywebview window exists."""
    global _WINDOW
    with _LOCK:
        _WINDOW = window


def get_window() -> Any:
    with _LOCK:
        return _WINDOW


def has_window() -> bool:
    return get_window() is not None


def _dialog_constants() -> tuple[Any, Any]:
    import webview  # type: ignore[import-not-found]

    # pywebview >= 5.3 moved the constants to enums; older versions keep the module ints
    fd = getattr(webview, "FileDialog", None)
    if fd is not None:
        return fd.FOLDER, fd.OPEN
    return webview.FOLDER_DIALOG, webview.OPEN_DIALOG


def _first(result: Any) -> str | None:
    if not result:
        return None
    if isinstance(result, (list, tuple)):
        return str(result[0]) if result else None
    return str(result)


def pick_folder(purpose: str = "") -> dict[str, Any]:
    """Native folder dialog. ``{path: str | None}``; ``{path: None, unsupported: True}`` without a window."""
    win = get_window()
    if win is None:
        return {"path": None, "unsupported": True}
    try:
        folder, _ = _dialog_constants()
        return {"path": _first(win.create_file_dialog(folder))}
    except Exception as e:  # a broken dialog must not break the wizard
        log.warning("pick_folder(%s) failed: %s", purpose, e)
        return {"path": None, "unsupported": True}


def _file_types(filters: list[str] | None) -> tuple[str, ...]:
    if not filters:
        return ()
    patterns = ";".join(f if f.startswith("*") else f"*{f}" for f in filters)
    return (f"Files ({patterns})",)


def pick_file(purpose: str = "", filters: list[str] | None = None) -> dict[str, Any]:
    win = get_window()
    if win is None:
        return {"path": None, "unsupported": True}
    try:
        _, open_dialog = _dialog_constants()
        return {"path": _first(win.create_file_dialog(open_dialog, allow_multiple=False, file_types=_file_types(filters)))}
    except Exception as e:
        log.warning("pick_file(%s) failed: %s", purpose, e)
        return {"path": None, "unsupported": True}


def open_path(path: str | os.PathLike[str]) -> bool:
    """Open a file or folder in the system file manager / default app."""
    p = str(path)
    try:
        if sys.platform.startswith("win"):
            os.startfile(p)  # type: ignore[attr-defined]  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", p])  # noqa: S603,S607
        else:
            subprocess.Popen(["xdg-open", p])  # noqa: S603,S607
        return True
    except OSError as e:
        log.warning("cannot open %s: %s", p, e)
        return False


def reveal_dir(path: Path) -> bool:
    path.mkdir(parents=True, exist_ok=True)
    return open_path(path)


def shutdown_pc(delay_s: int = 60) -> bool:
    """Schedule a PC shutdown (Windows ``shutdown /s /t 60``; ``shutdown -h`` elsewhere).

    Never executed under the ``apptest`` profile (tests) or with ``VNDUB_NO_SHUTDOWN=1``.
    """
    if os.environ.get("VNDUB_APP_PROFILE") == "apptest" or os.environ.get("VNDUB_NO_SHUTDOWN"):
        log.warning("shutdown_pc skipped (test profile / VNDUB_NO_SHUTDOWN)")
        return False
    try:
        if sys.platform.startswith("win"):
            subprocess.run(["shutdown", "/s", "/t", str(int(delay_s))], check=False)  # noqa: S603,S607
        else:
            subprocess.run(["shutdown", "-h", f"+{max(1, int(delay_s) // 60)}"], check=False)  # noqa: S603,S607
        return True
    except OSError as e:
        log.warning("shutdown failed: %s", e)
        return False


def cancel_shutdown() -> bool:
    """Abort a pending shutdown (``shutdown /a``)."""
    try:
        if sys.platform.startswith("win"):
            subprocess.run(["shutdown", "/a"], check=False)  # noqa: S603,S607
        else:
            subprocess.run(["shutdown", "-c"], check=False)  # noqa: S603,S607
        return True
    except OSError:
        return False

"""The start and stop banners of ``<data>/logs/app.log``.

``log_startup()`` writes one line that says what is running and where (version, source or frozen, Python, folders,
profile, Job Object); ``server._lifespan`` calls it when the server starts, so it runs once per server start without
the shell having to do anything. ``log_shutdown(code, started_at)`` writes the last line (exit code, uptime): the
shell calls it just before ``logging.shutdown()``.

Every function swallows its own errors: a banner must never keep the app from starting or from closing.
"""

from __future__ import annotations

import logging
import platform
import sys
import time

log = logging.getLogger("vndub_app.applog")

_STARTED_AT = time.time()  # when this module was first imported: close to the start of the server
_LOGGED_FOR: str | None = None  # the app.log the startup banner went to (a new data folder gets its own)


def format_duration(seconds: float) -> str:
    """``42s`` / ``3m05s`` / ``2h07m`` / ``1d03h``."""
    s = max(0, int(round(seconds)))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    if s < 86400:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    return f"{s // 86400}d{(s % 86400) // 3600:02d}h"


def _job_object() -> str:
    try:
        from vndub_app import winproc

        st = winproc.job_object_status()
    except Exception:  # not Windows, or the module is not there
        return "n/a"
    if st.get("installed"):
        return "installed"
    return f"not installed ({st['error']})" if st.get("error") else "not installed"


def startup_line() -> str:
    """The text of the startup banner (separate so that tests can look at it)."""
    from vndub_app import __version__, appdirs

    parts = [
        f"VNDub {__version__} starting",
        "frozen" if getattr(sys, "frozen", False) else "source",
        f"python {platform.python_version()}",
        platform.platform(terse=True),
    ]
    s = appdirs.load_settings()
    parts.append(f"profile={s.profile}")
    parts.append(f"data={appdirs.data_dir()}")
    try:
        from vndub_app import installer

        parts.append(f"envs={installer.resolve_envs_dir()}")
    except Exception:
        parts.append(f"envs={appdirs.effective_envs_dir(s)}")
    parts.append(f"models={appdirs.effective_models_dir(s)}")
    parts.append(f"job_object={_job_object()}")
    parts.append(f"debug_log={'on' if s.debug_log else 'off'}")
    return ", ".join(parts)


def log_startup(force: bool = False) -> None:
    """Write the startup banner to app.log (once for each app.log, unless ``force``)."""
    global _LOGGED_FOR
    try:
        from vndub_app import appdirs

        key = str(appdirs.logs_dir() / "app.log")
        if _LOGGED_FOR == key and not force:
            return
        _LOGGED_FOR = key
        log.info("%s", startup_line())
    except Exception:
        log.exception("cannot write the startup banner")


def log_shutdown(code: object = 0, started_at: float | None = None) -> None:
    """Write the last line of app.log: the exit code and how long the app ran.

    ``started_at`` is ``time.time()`` taken when the app started (default: when this module was imported).
    """
    try:
        t0 = _STARTED_AT if started_at is None else float(started_at)
        log.info("VNDub exiting with code %s after %s", code, format_duration(time.time() - t0))
    except Exception:
        pass

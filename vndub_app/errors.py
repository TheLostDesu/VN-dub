"""User-facing errors of the desktop app.

Every error the UI shows has two parts: what happened (one sentence) and one concrete
thing to do. The backend never sends display text: it sends a ``code`` (looked up as
``errors.<code>.title`` / ``errors.<code>.hint`` in ``app/i18n/{ru,en}.json``), optional
``params`` for interpolation, an ``action`` (a button the UI knows how to perform, label
``actions.<action>``) and ``details`` (technical text behind "Показать подробности").

Wire format (HTTP error body and ``JobInfo.error``)::

    {"error": {"code": "tool_missing", "params": {"tool": "ffmpeg"},
               "action": "open_tools", "details": "ToolMissing: ffmpeg not found ..."}}
"""

from __future__ import annotations

import re
import traceback
from typing import Any

# Buttons the UI can perform for an error (labels: i18n ``actions.<name>``).
ACTIONS = (
    "retry",            # repeat the failed request / job
    "open_doctor",      # go to Diagnostics
    "open_models",      # go to the model manager
    "open_tools",       # go to tools / environments installer
    "open_settings",    # go to Settings
    "open_data_dir",    # open the data folder in Explorer
    "open_logs",        # open the logs folder
    "report_problem",   # build the problem-report zip
    "choose_game_dir",  # pick another game folder
    "open_casting",     # go to the casting step
    "open_translation", # go to the translation step
    "show_manual",      # show manual instructions (install step, KrkrDump ...)
    "none",
)


class AppError(Exception):
    """An error with a stable code the UI translates. ``details`` is technical text."""

    def __init__(
        self,
        code: str,
        *,
        params: dict[str, Any] | None = None,
        action: str | None = None,
        details: str | None = None,
        status: int = 400,
    ) -> None:
        super().__init__(f"{code}: {details or params or ''}")
        if action is not None and action not in ACTIONS:
            raise ValueError(f"unknown error action {action!r}")
        self.code = code
        self.params = params or {}
        self.action = action or "none"
        self.details = details or ""
        self.status = status

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "params": self.params, "action": self.action, "details": self.details}


def _tb(e: BaseException) -> str:
    return "".join(traceback.format_exception(type(e), e, e.__traceback__))


def error_from_exception(e: BaseException, *, stage: str | None = None) -> AppError:
    """Map any exception (pipeline or app) to an :class:`AppError` with a known code."""
    if isinstance(e, AppError):
        return e
    from vndub.backends.base import BackendUnavailable
    from vndub.stages.common import StageError
    from vndub.tools import ToolMissing

    from vndub_app.jobs import JobCancelled

    msg = str(e)
    details = f"{type(e).__name__}: {msg}\n\n{_tb(e)}"
    params: dict[str, Any] = {"stage": stage} if stage else {}
    if isinstance(e, JobCancelled):
        return AppError("cancelled", params=params, action="retry", details=msg, status=409)
    if isinstance(e, ToolMissing):
        m = re.search(r"tool[ '\"]+([\w.-]+)", msg) or re.search(r"^([\w.-]+)", msg)
        return AppError("tool_missing", params={**params, "tool": m.group(1) if m else ""}, action="open_tools", details=details)
    if "weights missing" in msg:
        m = re.search(r"models pull ([\w.-]+)", msg)
        return AppError("weights_missing", params={**params, "model": m.group(1) if m else ""}, action="open_models", details=details)
    if re.search(r"env '[\w+.-]+' not created", msg):
        m = re.search(r"env '([\w+.-]+)'", msg)
        return AppError("env_missing", params={**params, "env": m.group(1) if m else ""}, action="open_tools", details=details)
    if isinstance(e, BackendUnavailable):
        return AppError("backend_unavailable", params=params, action="open_models", details=details)
    if isinstance(e, StageError):
        return AppError("stage_failed", params=params, action="retry", details=details)
    if isinstance(e, FileNotFoundError):
        return AppError("file_not_found", params={**params, "path": getattr(e, "filename", "") or ""}, action="retry", details=details)
    if isinstance(e, PermissionError):
        return AppError("permission_denied", params={**params, "path": getattr(e, "filename", "") or ""}, action="open_data_dir", details=details)
    return AppError("internal", params=params, action="report_problem", details=details, status=500)


def install_handlers(app: Any) -> None:
    """Register FastAPI exception handlers that emit the wire format above."""
    from fastapi import Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse

    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content={"error": exc.to_dict()})

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        err = AppError("bad_request", action="none", details=str(exc.errors()), status=422)
        return JSONResponse(status_code=422, content={"error": err.to_dict()})

    @app.exception_handler(Exception)
    async def _any(_: Request, exc: Exception) -> JSONResponse:
        err = error_from_exception(exc)
        return JSONResponse(status_code=err.status, content={"error": err.to_dict()})

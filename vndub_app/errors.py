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

import json
import logging
import re
import traceback
from typing import Any

log = logging.getLogger("vndub_app.errors")

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
    # contract C6: the real cause, one line, next to the hint (the traceback stays behind "details")
    reason = _short(f"{msg}" if msg else type(e).__name__, REASON_MAX)
    low = msg.lower()
    if "winerror 1455" in low or "paging file is too small" in low or "файл подкачки" in low:
        return AppError("paging_file_small", params={**params, "reason": reason}, action="open_doctor", details=details)
    from vndub.stages.genutil import is_vram_error

    if is_vram_error(msg) and not isinstance(e, (MemoryError, BackendUnavailable)):  # "does not fit the VRAM budget" is a refusal, not an OOM
        return AppError("cuda_oom", params={**params, "reason": reason}, action="open_settings", details=details)
    env = _import_failure_env(e, msg)
    if env is not None:  # the model's code does not import in its environment: a broken env, not a missing download
        return AppError("env_import_failed", params={**params, "env": env, "reason": reason}, action="open_doctor", details=details)
    if isinstance(e, BackendUnavailable):
        return AppError("backend_unavailable", params={**params, "reason": reason}, action="open_models", details=details)
    if isinstance(e, StageError):
        return AppError("stage_failed", params={**params, "reason": reason}, action="retry", details=details)
    if isinstance(e, FileNotFoundError):
        return AppError("file_not_found", params={**params, "path": getattr(e, "filename", "") or ""}, action="retry", details=details)
    if isinstance(e, PermissionError):
        return AppError("permission_denied", params={**params, "path": getattr(e, "filename", "") or ""}, action="open_data_dir", details=details)
    return AppError("internal", params={**params, "reason": reason}, action="report_problem", details=details, status=500)


REASON_MAX = 300
_IMPORT_RE = re.compile(r"\b(ModuleNotFoundError|ImportError|ErrorDuringImport|No module named|cannot import name|DLL load failed|import failed in env)\b", re.I)


def _import_failure_env(e: BaseException, msg: str) -> str | None:
    """The environment whose adapter failed to import (``""`` when the env is not named), else ``None``.

    Only errors that come from an adapter (``AdapterError`` / ``BackendUnavailable``) or that name an env count: an
    ``ImportError`` of the app itself is an internal error, not a broken backend environment."""
    from vndub.backends.base import BackendUnavailable

    try:
        from vndub.backends.runner import AdapterError
    except Exception:  # pragma: no cover
        AdapterError = ()  # type: ignore[assignment]
    if not _IMPORT_RE.search(msg):
        return None
    m = re.search(r"env '([\w+.-]+)'", msg)
    if m:
        return m.group(1)
    if isinstance(e, (BackendUnavailable, AdapterError)):  # type: ignore[arg-type]
        backend = getattr(e, "backend", None) or getattr(e, "name", None) or (e.args[0] if e.args else "")
        try:
            from vndub.backends import registry

            return registry.get(str(backend)).env or ""
        except Exception:
            return ""
    return None


def _short(text: str | None, limit: int = 200) -> str:
    """The first non-empty line of ``text``, cut to ``limit`` characters."""
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return line if len(line) <= limit else line[: limit - 3] + "..."
    return ""


def log_http_error(method: str, path: str, err: AppError, exc: BaseException | None = None) -> None:
    """One log line for an error response: ``HTTP 404 GET /api/project -> no_project (first line of details)``.

    WARNING for a 4xx, ERROR for a 5xx. ``exc`` is the unexpected exception behind an error the app did not raise
    itself: its traceback goes with the line. Details and parameters are cut to one short line (no query string, no body).
    """
    why = _short(err.details)
    if not why and err.params:
        try:
            why = _short(json.dumps(err.params, ensure_ascii=False, default=str))
        except (TypeError, ValueError):
            why = ""
    log.log(
        logging.ERROR if err.status >= 500 else logging.WARNING,
        "HTTP %s %s %s -> %s%s", err.status, method, path, err.code, f" ({why})" if why else "",
        exc_info=(type(exc), exc, exc.__traceback__) if exc is not None else None,
    )


def install_handlers(app: Any) -> None:
    """Register FastAPI exception handlers that emit the wire format above (and log every error response)."""
    from fastapi import Request
    from fastapi.exception_handlers import http_exception_handler
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
    from starlette.exceptions import HTTPException as StarletteHTTPException

    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError) -> JSONResponse:
        log_http_error(request.method, request.url.path, exc)
        return JSONResponse(status_code=exc.status, content={"error": exc.to_dict()})

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        err = AppError("bad_request", action="none", details=str(exc.errors()), status=422)
        log_http_error(request.method, request.url.path, err)
        return JSONResponse(status_code=422, content={"error": err.to_dict()})

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> Any:  # a bad method / path that no route owns
        log.warning("HTTP %s %s %s -> %s", exc.status_code, request.method, request.url.path, _short(str(exc.detail)))
        return await http_exception_handler(request, exc)

    @app.exception_handler(Exception)
    async def _any(request: Request, exc: Exception) -> JSONResponse:
        err = error_from_exception(exc)
        log_http_error(request.method, request.url.path, err, exc)
        return JSONResponse(status_code=err.status, content={"error": err.to_dict()})

"""Per-process app session: the current project, pipeline contexts, URL helpers.

* The *current project* is process-wide (one window). ``POST /api/project/open`` sets it
  and remembers it in ``Settings.last_project``.
* :func:`read_ctx` returns a cached read-only :class:`StageContext` (cheap manifest
  access, never used to run a stage); :func:`job_ctx` builds a fresh one for a job,
  exactly as API.md section 1 describes.
* ``vndub.cli`` (typer, rich ...) costs ~1 s to import, so every pipeline import in this
  package is lazy; :func:`warm_up` imports them in a background thread after startup.
"""

from __future__ import annotations

import json
import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote

from vndub_app import appdirs
from vndub_app.errors import AppError

log = logging.getLogger("vndub_app.session")

_LOCK = threading.RLock()
#: held around every read-modify-write of ``app_state.json`` made by the app (request threads and job threads)
STATE_LOCK = threading.RLock()
_CURRENT: str | None = None
_READ_CTX: dict[str, tuple[tuple[Any, ...], Any]] = {}

AUDIO_EXTS = (".wav", ".ogg", ".mp3", ".flac", ".opus", ".m4a", ".weba")


# ----------------------------------------------------------------------------
# current project
# ----------------------------------------------------------------------------


def set_current(slug: str | None) -> None:
    global _CURRENT
    with _LOCK:
        _CURRENT = slug
    if slug:
        s = appdirs.load_settings()
        if s.last_project != slug:
            s.last_project = slug
            appdirs.save_settings(s)


def current_slug() -> str | None:
    with _LOCK:
        if _CURRENT:
            return _CURRENT
    last = appdirs.load_settings().last_project
    if last and (appdirs.projects_dir() / last / "project_meta.json").exists():
        return last
    return None


def current_handle() -> appdirs.ProjectHandle:
    slug = current_slug()
    if not slug:
        raise AppError("no_project", action="choose_game_dir", status=404)
    return appdirs.project_handle(slug)


def reset() -> None:
    """Forget the current project and every cached context (tests, data-dir change)."""
    global _CURRENT
    with _LOCK:
        _CURRENT = None
        _READ_CTX.clear()


# ----------------------------------------------------------------------------
# pipeline contexts
# ----------------------------------------------------------------------------


def _sets_list(sets: dict[str, Any] | list[str] | None) -> list[str]:
    if not sets:
        return []
    if isinstance(sets, list):
        return list(sets)
    return [f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in sets.items()]


def job_ctx(
    h: appdirs.ProjectHandle,
    *,
    only: list[str] | None = None,
    limit: int | None = None,
    force: bool = False,
    sets: dict[str, Any] | list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> Any:
    """Fresh pipeline context for a job (regenerates ``pipeline.yaml`` first). Caller closes ``ctx.models``."""
    from vndub.cli import make_ctx

    appdirs.write_project_config(h)
    return make_ctx(
        h.game_dir, h.config_file, force=force, only=",".join(only) if only else None, limit=limit,
        sets=_sets_list(sets), extra=dict(extra or {}),
    )


def read_ctx(h: appdirs.ProjectHandle) -> Any:
    """Cached context for reading manifests (invalidated when ``pipeline.yaml`` changes)."""
    if not h.config_file.exists():
        appdirs.write_project_config(h)
    try:
        key = (h.config_file.stat().st_mtime_ns, str(h.game_dir))
    except OSError:
        key = (0, str(h.game_dir))
    with _LOCK:
        hit = _READ_CTX.get(h.slug)
        if hit and hit[0] == key:
            return hit[1]
    from vndub.cli import make_ctx

    ctx = make_ctx(h.game_dir, h.config_file)
    with _LOCK:
        _READ_CTX[h.slug] = (key, ctx)
    return ctx


@contextmanager
def short_ctx(h: appdirs.ProjectHandle, **kw: Any) -> Iterator[Any]:
    """A fresh context for a short synchronous operation (casting helpers); models are closed on exit."""
    ctx = job_ctx(h, **kw)
    try:
        yield ctx
    finally:
        ctx.models.close()


def work_paths(h: appdirs.ProjectHandle) -> Any:
    """``WorkPaths`` of a project without building a whole pipeline context."""
    from vndub.paths import WorkPaths

    return WorkPaths(h.work.resolve(), h.out.resolve())


def warm_up() -> None:
    """Import the heavy pipeline modules once (background thread; startup must stay fast)."""
    try:
        import vndub.cli  # noqa: F401
        import vndub.stages.common  # noqa: F401
        import vndub.stages.genutil  # noqa: F401
    except Exception as e:  # never fatal
        log.warning("warm-up import failed: %s", e)


# ----------------------------------------------------------------------------
# audio urls
# ----------------------------------------------------------------------------


def _q(rel: str) -> str:
    return "/".join(quote(p, safe="") for p in rel.split("/") if p)


def audio_url(slug: str, rel: str | None) -> str | None:
    """URL of a work-relative file (posix path)."""
    if not rel:
        return None
    return f"/api/audio/{quote(slug, safe='')}/{_q(rel.replace(chr(92), '/'))}"


def out_url(slug: str, rel: str | None) -> str | None:
    if not rel:
        return None
    return f"/api/audio/{quote(slug, safe='')}/_out/{_q(rel.replace(chr(92), '/'))}"


def work_url(h: appdirs.ProjectHandle, p: str | Path | None) -> str | None:
    """URL for a path that lives in the project's work dir (relative or absolute)."""
    if not p:
        return None
    path = Path(p)
    if path.is_absolute():
        try:
            rel = path.resolve().relative_to(h.work.resolve()).as_posix()
        except ValueError:
            return None
    else:
        rel = path.as_posix()
    return audio_url(h.slug, rel)


def resolve_audio(slug: str, rel: str) -> Path:
    """Map ``/api/audio/<slug>/<rel>`` to a file, refusing anything outside work/out (or the voice library)."""
    rel = rel.replace("\\", "/")
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." or ":" in p for p in parts) or rel.startswith("/"):
        raise AppError("file_not_found", params={"path": rel}, status=404)
    if slug == "_voices":
        base = appdirs.voices_dir()
    else:
        h = appdirs.project_handle(slug)
        if parts[0] == "_out":
            base, parts = h.out, parts[1:]
        else:
            base = h.work
    if not parts:
        raise AppError("file_not_found", params={"path": rel}, status=404)
    target = base.joinpath(*parts)
    try:
        resolved = target.resolve(strict=True)
        resolved.relative_to(base.resolve())
    except (OSError, ValueError):
        raise AppError("file_not_found", params={"path": rel}, status=404) from None
    if not resolved.is_file() or resolved.suffix.lower() not in AUDIO_EXTS:
        raise AppError("file_not_found", params={"path": rel}, status=404)
    return resolved

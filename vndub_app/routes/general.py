"""General, settings, system, help and audio routes (API.md section 3 "General")."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body
from fastapi.responses import FileResponse

from vndub_app import __version__, appdirs, jobs, pipeline, session, system
from vndub_app.errors import AppError

log = logging.getLogger("vndub_app.routes.general")
router = APIRouter(prefix="/api")

MEDIA_TYPES = {".wav": "audio/wav", ".ogg": "audio/ogg", ".mp3": "audio/mpeg", ".flac": "audio/flac", ".opus": "audio/ogg", ".m4a": "audio/mp4", ".weba": "audio/webm"}

SETTINGS_KEYS = {"ui_lang", "gpu_budget_gb", "license_policy", "models_dir", "player_volume", "shutdown_when_done", "profile", "translation_llm"}


def _bad(field: str, value: Any = None) -> AppError:
    return AppError("bad_request", params={"field": field, "value": value}, status=422)


# ----------------------------------------------------------------------------
# health + settings
# ----------------------------------------------------------------------------


def _check_translation_llm(value: Any) -> str:
    """"" (the default model) or the name of a real, non-mock ``llm`` backend (built in or added by link)."""
    name = str(value or "").strip()
    if not name:
        return ""
    from vndub.backends import registry

    from vndub_app import hf_custom

    hf_custom.sync_env()
    try:
        spec = registry.get(name)
    except KeyError:
        raise _bad("translation_llm", name) from None
    if spec.role != "llm" or spec.name.startswith("mock"):
        raise _bad("translation_llm", name)
    return name


@router.get("/health")
def health() -> dict[str, Any]:
    s = appdirs.load_settings()
    return {"ok": True, "version": __version__, "profile": s.profile, "mock": s.profile == "apptest"}


def _settings_out() -> dict[str, Any]:
    d = appdirs.load_settings().model_dump(mode="json")
    d["data_dir"] = str(appdirs.data_dir())
    return d


@router.get("/settings")
def get_settings() -> dict[str, Any]:
    return _settings_out()


@router.put("/settings")
def put_settings(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    unknown = set(body) - SETTINGS_KEYS - {"data_dir", "last_project"}
    if unknown:
        raise _bad("settings", sorted(unknown))
    s = appdirs.load_settings()
    if "ui_lang" in body:
        if body["ui_lang"] not in ("ru", "en"):
            raise _bad("ui_lang", body["ui_lang"])
        s.ui_lang = body["ui_lang"]
    if "gpu_budget_gb" in body:
        try:
            v = float(body["gpu_budget_gb"])
        except (TypeError, ValueError):
            raise _bad("gpu_budget_gb", body["gpu_budget_gb"]) from None
        if not 1.0 <= v <= 96.0:
            raise _bad("gpu_budget_gb", v)
        s.gpu_budget_gb = v
    if "license_policy" in body:
        if body["license_policy"] not in ("research", "commercial"):
            raise _bad("license_policy", body["license_policy"])
        s.license_policy = body["license_policy"]
    if "models_dir" in body:
        s.models_dir = str(body["models_dir"] or "")
    if "player_volume" in body:
        if not isinstance(body["player_volume"], dict):
            raise _bad("player_volume")
        merged = dict(s.player_volume)
        for k, v in body["player_volume"].items():
            merged[str(k)] = max(0.0, min(1.0, float(v)))
        s.player_volume = merged
    if "shutdown_when_done" in body:
        s.shutdown_when_done = bool(body["shutdown_when_done"])
    if "profile" in body:
        if body["profile"] not in appdirs.PROFILES:
            raise _bad("profile", body["profile"])
        s.profile = body["profile"]
    if "translation_llm" in body:
        s.translation_llm = _check_translation_llm(body["translation_llm"])
    appdirs.save_settings(s)
    slug = session.current_slug()
    if slug:  # the generated pipeline.yaml carries models_dir / license / GPU budget
        try:
            appdirs.write_project_config(appdirs.project_handle(slug))
        except AppError:
            pass
    return _settings_out()


@router.post("/settings/data-dir")
def set_data_dir(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    raw = str(body.get("path") or "").strip()
    if not raw:
        raise _bad("path", raw)
    active = jobs.runner().list(active_only=True)
    if active:
        raise AppError("job_busy", params={"job_id": active[-1].id, "kind": active[-1].kind}, status=409)
    try:
        appdirs.set_data_dir(Path(raw))
    except OSError as e:
        raise AppError("permission_denied", params={"path": raw}, action="open_data_dir", details=str(e)) from e
    jobs.reset_runner(None)  # the next runner() binds to the new folder
    session.reset()
    from vndub_app import server

    server.setup_logging()
    server._sync_custom_models()  # noqa: SLF001  (the custom models list lives in the data folder)
    return _settings_out()


# ----------------------------------------------------------------------------
# system
# ----------------------------------------------------------------------------


@router.post("/system/pick-folder")
def pick_folder(body: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    return system.pick_folder(str(body.get("purpose") or ""))


@router.post("/system/pick-file")
def pick_file(body: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    filters = body.get("filters")
    return system.pick_file(str(body.get("purpose") or ""), [str(f) for f in filters] if isinstance(filters, list) else None)


def _patch_dir(h: appdirs.ProjectHandle) -> Path:
    """``out/<engine>`` when the project has been packed, else ``out/``."""
    try:
        d = h.out / session.read_ctx(h).engine().out_name
        if d.exists():
            return d
    except Exception:  # no project.yaml yet / unknown engine
        pass
    return h.out


@router.post("/system/open")
def open_something(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    kind = body.get("kind")
    if kind == "data_dir":
        target = appdirs.data_dir()
    elif kind == "logs":
        target = appdirs.logs_dir()
    elif kind == "backups":
        target = appdirs.backups_dir()
    elif kind in ("patch", "work"):
        h = session.current_handle()
        target = h.work if kind == "work" else _patch_dir(h)
        target.mkdir(parents=True, exist_ok=True)
    elif kind == "path":
        raw = str(body.get("path") or "")
        if not raw:
            raise _bad("path", raw)
        target = Path(raw)
        if not target.exists():
            raise AppError("file_not_found", params={"path": raw}, status=404)
    else:
        raise _bad("kind", kind)
    return {"ok": bool(system.open_path(target))}


@router.post("/system/report")
def make_report() -> dict[str, Any]:
    slug = session.current_slug()
    return pipeline.submit("report", slug, {"slug": slug} if slug else {}).info()


@router.post("/system/shutdown-pc")
def shutdown_pc() -> dict[str, Any]:
    """Internal: only when the user ticked "turn the computer off when finished"."""
    if not appdirs.load_settings().shutdown_when_done:
        raise AppError("permission_denied", params={"path": "shutdown"}, status=403)
    return {"ok": system.shutdown_pc()}


# ----------------------------------------------------------------------------
# help (bundled docs)
# ----------------------------------------------------------------------------

HELP_DOCS = (
    ("app", "docs/APP.md"),
    ("troubleshooting", "docs/TROUBLESHOOTING.md"),
    ("readme", "README.md"),
    ("pipeline", "docs/PIPELINE.md"),
    ("engines", "docs/ENGINES.md"),
    ("models", "docs/MODELS.md"),
)


def _help_path(doc_id: str) -> Path | None:
    for i, rel in HELP_DOCS:
        if i == doc_id:
            p = appdirs.resource(*rel.split("/"))
            return p if p.exists() else None
    return None


@router.get("/help")
def help_index() -> list[dict[str, Any]]:
    return [{"id": i, "title_key": f"help.{i}.title"} for i, _ in HELP_DOCS if _help_path(i) is not None]


@router.get("/help/{doc_id}")
def help_doc(doc_id: str) -> dict[str, Any]:
    p = _help_path(doc_id)
    if p is None:
        raise AppError("file_not_found", params={"path": doc_id}, status=404)
    return {"id": doc_id, "markdown": p.read_text(encoding="utf-8", errors="replace")}


# ----------------------------------------------------------------------------
# audio
# ----------------------------------------------------------------------------


@router.get("/audio/{slug}/{path:path}")
def audio(slug: str, path: str) -> FileResponse:
    """Audio of a project (``work/`` or ``_out/`` for ``out/``) or the voice library (slug ``_voices``).

    Nothing outside those folders is ever served; ``..`` and absolute paths are refused (404).
    Range requests are handled by ``FileResponse``.
    """
    f = session.resolve_audio(slug, path)
    return FileResponse(f, media_type=MEDIA_TYPES.get(f.suffix.lower(), "application/octet-stream"), headers={"Cache-Control": "no-cache"})

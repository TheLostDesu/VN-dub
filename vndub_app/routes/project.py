"""Project routes: list/open, state, resume, Step 1 import + summary, generic stage run."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, Body

from vndub_app import appdirs, jobs, pipeline, session, summary
from vndub_app.errors import AppError

log = logging.getLogger("vndub_app.routes.project")
router = APIRouter(prefix="/api")

# keys the SPA may not overwrite through PUT /api/project/state
PROTECTED_STATE_KEYS = {"config_overrides", "updated"}
PROTECTED_STATE_FIELDS = {"review": {"touched", "last_apply"}, "preview": {"line_ids"}}


def _engine_of(h: appdirs.ProjectHandle) -> str | None:
    p = h.work / "project.yaml"
    if not p.exists():
        return None
    try:
        return (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("engine")
    except (OSError, yaml.YAMLError):
        return None


def _project_yaml(h: appdirs.ProjectHandle) -> dict[str, Any] | None:
    p = h.work / "project.yaml"
    if not p.exists():
        return None
    try:
        d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    return d if isinstance(d, dict) else None


def _stage_status(h: appdirs.ProjectHandle) -> dict[str, Any]:
    p = h.work / "stage_status.json"
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def project_info(h: appdirs.ProjectHandle) -> dict[str, Any]:
    runner = jobs.runner()
    active = runner.active(h.slug)
    inter = runner.interrupted(h.slug)
    job = inter[-1] if inter else None
    st = appdirs.read_app_state(h)
    py = _project_yaml(h)
    return {
        "slug": h.slug, "name": h.name, "game_dir": str(h.game_dir), "engine": (py or {}).get("engine"),
        "project_yaml": py, "state": st, "stage_status": _stage_status(h),
        "active_job": active.info() if active else None,
        "resume": {
            "available": job is not None, "job": job.info() if job else None,
            "next_stage": (job.stage or (job.stages[0] if job.stages else None)) if job else None,
        },
    }


def _lite(h: appdirs.ProjectHandle) -> dict[str, Any]:
    st = appdirs.read_app_state(h)
    return {"slug": h.slug, "name": h.name, "game_dir": str(h.game_dir), "engine": _engine_of(h), "step": st.get("step"), "updated": st.get("updated")}


@router.get("/projects")
def list_projects() -> list[dict[str, Any]]:
    out = [_lite(h) for h in appdirs.list_projects()]
    out.sort(key=lambda p: p.get("updated") or "", reverse=True)
    return out


@router.post("/project/open")
def open_project(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    created = False
    if body.get("slug"):
        h = appdirs.project_handle(str(body["slug"]))
    elif body.get("game_dir"):
        g = Path(str(body["game_dir"])).expanduser()
        if not g.is_dir():
            raise AppError("game_dir_invalid", params={"path": str(g)}, action="choose_game_dir", status=400)
        created = appdirs.find_project_for_game(g) is None
        h = appdirs.create_project(g)
    else:
        raise AppError("bad_request", params={"field": "game_dir"}, status=422)
    previous = session.current_slug()
    session.set_current(h.slug)
    if previous and previous != h.slug:
        log.info("project closed: %s", previous)
    log.info("project %s: slug=%s game_dir=%s engine=%s", "created" if created else "opened", h.slug, h.game_dir, _engine_of(h) or "unknown")
    return project_info(h)


@router.get("/project")
def get_project() -> dict[str, Any]:
    return project_info(session.current_handle())


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@router.put("/project/state")
def put_state(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    h = session.current_handle()
    patch = {k: v for k, v in body.items() if k not in PROTECTED_STATE_KEYS}
    for section, keys in PROTECTED_STATE_FIELDS.items():  # owned by the backend: the review list and the preview run
        if isinstance(patch.get(section), dict):
            patch[section] = {k: v for k, v in patch[section].items() if k not in keys}
    with session.STATE_LOCK:
        st = appdirs.write_app_state(h, _deep_merge(appdirs.read_app_state(h), patch))
    appdirs.write_project_config(h)
    return st


@router.post("/project/resume")
def resume() -> dict[str, Any]:
    h = session.current_handle()
    return pipeline.resume_interrupted(h.slug).info()


@router.post("/project/import")
def import_game(body: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    h = session.current_handle()
    params = {k: body[k] for k in ("engine", "force") if body.get(k)}
    return pipeline.submit("import", h.slug, params).info()


@router.post("/project/recover-text")
def recover_text() -> dict[str, Any]:
    return pipeline.submit("recover_text", session.current_handle().slug, {}).info()


@router.post("/project/asr-check")
def asr_check() -> dict[str, Any]:
    return pipeline.submit("asr_check", session.current_handle().slug, {}).info()


@router.get("/project/summary")
def get_summary() -> dict[str, Any]:
    return summary.build(session.current_handle())


@router.post("/stages/{name}/run")
def run_stage(name: str, body: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    from vndub.stages.common import STAGES

    h = session.current_handle()
    if name not in STAGES:
        raise AppError("bad_request", params={"field": "name", "value": name}, status=422)
    only = body.get("only")
    if only is not None and not (isinstance(only, list) and all(isinstance(x, str) for x in only)):
        raise AppError("bad_request", params={"field": "only"}, status=422)
    sets = body.get("sets") or {}
    if not isinstance(sets, dict):
        raise AppError("bad_request", params={"field": "sets"}, status=422)
    if sets:  # validate against the config schema now, not in the middle of the job
        from vndub.config import load_config

        try:
            load_config(h.config_file, {str(k): v for k, v in sets.items()})
        except ValueError as e:
            raise AppError("bad_request", params={"field": "sets"}, details=str(e), status=422) from e
    params = {"name": name, "only": only or None, "limit": body.get("limit"), "force": bool(body.get("force")), "sets": sets or None}
    return pipeline.submit("stage", h.slug, params).info()

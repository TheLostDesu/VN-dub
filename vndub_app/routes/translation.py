"""Step 2 routes: translation settings, import preview, glossary, translate job, delivery cues."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body

from vndub_app import pipeline, session, translation
from vndub_app.errors import AppError

router = APIRouter(prefix="/api")


@router.get("/project/translation")
def get_translation() -> dict[str, Any]:
    return translation.get_settings(session.current_handle())


@router.put("/project/translation")
def put_translation(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    return translation.put_settings(session.current_handle(), body)


@router.post("/project/translation/preview-import")
def preview_import(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    h = session.current_handle()
    raw = str(body.get("path") or "")
    if not raw or not Path(raw).exists():
        raise AppError("file_not_found", params={"path": raw}, status=404)
    from vndub.stages import translate
    from vndub.stages.common import StageError

    fn = getattr(translate, "preview_import", None)
    if fn is None:
        raise AppError("feature_unavailable", params={"feature": "translate.preview_import"}, action="report_problem", status=501)
    ctx = session.read_ctx(h)
    try:
        return dict(fn(ctx, raw, n=5))
    except (ValueError, StageError) as e:
        raise AppError("bad_request", params={"field": "path"}, details=str(e), status=422) from e


@router.get("/project/glossary")
def get_glossary() -> dict[str, Any]:
    return translation.get_glossary(session.current_handle())


@router.put("/project/glossary")
def put_glossary(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    return translation.put_glossary(session.current_handle(), body)


@router.post("/project/translate")
def translate_job() -> dict[str, Any]:
    h = session.current_handle()
    return pipeline.submit("translate", h.slug, {}).info()


@router.get("/project/delivery-cues")
def get_cues() -> dict[str, Any]:
    return translation.get_cues(session.current_handle())


@router.put("/project/delivery-cues")
def put_cues(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    return translation.put_cues(session.current_handle(), body)

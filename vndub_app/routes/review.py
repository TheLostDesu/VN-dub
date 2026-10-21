"""Step 6 routes: review list, line detail, actions, prefer-route, apply."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Query

from vndub_app import pipeline, reviewdata, session
from vndub_app.errors import AppError

router = APIRouter(prefix="/api/review")


@router.get("/lines")
def lines(
    speaker: str | None = None, scene: str | None = None, status: str | None = None, q: str | None = None,
    min: float | None = Query(None, alias="min"), max: float | None = Query(None, alias="max"),  # noqa: A002
    limit: int | None = Query(None, ge=0), offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    if status not in (None, "", "ok", "check", "failed"):
        raise AppError("bad_request", params={"field": "status", "value": status}, status=422)
    return reviewdata.lines_view(
        session.current_handle(), speaker=speaker or None, scene=scene or None, status=status or None, q=q, lo=min, hi=max, limit=limit, offset=offset,
    )


@router.get("/lines/{line_id}")
def line(line_id: str) -> dict[str, Any]:
    return reviewdata.line_detail(session.current_handle(), line_id)


@router.post("/lines/{line_id}/action")
def action(line_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    return reviewdata.apply_action(session.current_handle(), line_id, str(body.get("action") or ""), body.get("payload"))


@router.post("/prefer-route")
def prefer_route(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    return reviewdata.prefer_route(session.current_handle(), str(body.get("route") or ""), body.get("speaker_id") or None)


@router.post("/apply")
def apply() -> dict[str, Any]:
    h = session.current_handle()
    ids = reviewdata.touched_ids(h)
    if not ids:
        raise AppError("nothing_to_review", action="none", status=400)
    return pipeline.submit("review_apply", h.slug, {"line_ids": ids}).info()

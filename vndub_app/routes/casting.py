"""Step 3 / 3b routes: characters, casting proposals, upload, choose, listen, voice library, tuning."""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, File, UploadFile

from vndub_app import appdirs, casting, pipeline, session, viewdata
from vndub_app.errors import AppError

log = logging.getLogger("vndub_app.routes.casting")
router = APIRouter(prefix="/api")

MAX_UPLOAD = 50 * 1024 * 1024


def _bad(field: str, value: Any = None) -> AppError:
    return AppError("bad_request", params={"field": field, "value": value}, status=422)


def _int(v: Any, default: int, field: str) -> int:
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        raise _bad(field, v) from None


def _voicebank(name: str) -> Any:
    from vndub.stages import voicebank

    fn = getattr(voicebank, name, None)
    if fn is None:
        raise AppError("feature_unavailable", params={"feature": f"voicebank.{name}"}, action="report_problem", details=f"vndub.stages.voicebank.{name} is not implemented in this build", status=501)
    return fn


def _check_route(route: Any) -> str:
    if route not in casting.ROUTES:
        raise _bad("route", route)
    return str(route)


# ----------------------------------------------------------------------------
# characters
# ----------------------------------------------------------------------------


@router.get("/characters")
def list_characters() -> list[dict[str, Any]]:
    return casting.characters_view(session.current_handle())


@router.put("/characters/{sid}")
def update_character(sid: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    h = session.current_handle()
    casting.update_character(
        h, sid, gender=body.get("gender"), age_hint=body.get("age_hint"), description=body.get("description"), name_tgt=body.get("name_tgt"),
    )
    return casting.character_view(h, sid)


# ----------------------------------------------------------------------------
# proposals
# ----------------------------------------------------------------------------


@router.get("/casting/{sid}/proposals")
def proposals(sid: str) -> list[dict[str, Any]]:
    h = session.current_handle()
    ctx = session.read_ctx(h)
    if sid not in viewdata.load_characters(ctx):
        raise AppError("bad_request", params={"field": "speaker_id", "value": sid}, status=404)
    return casting.designs_view(h, ctx, sid)


@router.post("/casting/{sid}/generate")
def generate(sid: str, body: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    h = session.current_handle()
    n = _int(body.get("n"), 3, "n")
    if not 1 <= n <= 12:
        raise _bad("n", n)
    params: dict[str, Any] = {"speaker_id": sid, "n": n}
    if body.get("description") is not None:
        params["description"] = str(body["description"])
    return pipeline.submit("casting_generate", h.slug, params).info()


_SAFE = re.compile(r"[^0-9A-Za-z._-]+")


@router.post("/casting/{sid}/upload")
async def upload(sid: str, file: UploadFile = File(...)) -> dict[str, Any]:
    from starlette.concurrency import run_in_threadpool

    h = session.current_handle()
    name = _SAFE.sub("_", Path(file.filename or "sample").name)[-80:] or "sample"
    dest_dir = h.root / "uploads" / _SAFE.sub("_", sid)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{uuid.uuid4().hex[:8]}_{name}"
    size = 0
    with open(dest, "wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD:
                f.close()
                dest.unlink(missing_ok=True)
                raise AppError("sample_too_long", params={"max": 20}, action="none", status=413)
            f.write(chunk)
    try:
        return await run_in_threadpool(_register_upload, h, sid, dest)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise


def _register_upload(h: appdirs.ProjectHandle, sid: str, path: Path) -> dict[str, Any]:
    with session.short_ctx(h) as ctx:
        if sid not in viewdata.load_characters(ctx):
            raise AppError("bad_request", params={"field": "speaker_id", "value": sid}, status=404)
        casting.check_extension(path)
        fn = _voicebank("add_reference_design")
        try:
            d = fn(ctx, sid, str(path))
        except ValueError as e:
            raise casting.map_value_error(e) from e
        d = dict(d) if not isinstance(d, dict) else d
        return casting.design_view(h, d, texts_src=None)


# ----------------------------------------------------------------------------
# choose / auto / listen
# ----------------------------------------------------------------------------


@router.post("/casting/{sid}/choose")
def choose(sid: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    h = session.current_handle()
    design_id = str(body.get("design_id") or "")
    if not design_id:
        raise _bad("design_id", design_id)
    route = _check_route(body.get("route"))
    fn = _voicebank("choose_design")
    with session.short_ctx(h) as ctx:
        if sid not in viewdata.load_characters(ctx):
            raise AppError("bad_request", params={"field": "speaker_id", "value": sid}, status=404)
        try:
            fn(ctx, sid, design_id, route)
        except ValueError as e:
            raise casting.map_choose_error(e, sid, design_id, route) from e
    if route == "B":
        pipeline.maybe_train_rvc(h, sid)
    return casting.character_view(h, sid)


@router.post("/casting/auto")
def auto_cast(body: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    """-> Job (casting_auto): designs for characters without any, then the top-ranked voice for everybody."""
    h = session.current_handle()
    params: dict[str, Any] = {}
    if body.get("n") is not None:
        params["n"] = _int(body.get("n"), 3, "n")
    return pipeline.submit("casting_auto", h.slug, params).info()


@router.post("/casting/{sid}/listen")
def listen(sid: str, body: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    h = session.current_handle()
    n = _int(body.get("n"), 3, "n")
    if not 1 <= n <= 10:
        raise _bad("n", n)
    return pipeline.submit("casting_listen", h.slug, {"speaker_id": sid, "n": n}).info()


# ----------------------------------------------------------------------------
# voice library
# ----------------------------------------------------------------------------


@router.get("/voices")
def list_voices() -> list[dict[str, Any]]:
    return casting.list_voices()


@router.post("/voices")
def save_voice(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    return casting.save_voice(session.current_handle(), str(body.get("speaker_id") or ""), str(body.get("name") or ""))


@router.post("/casting/{sid}/from-library")
def from_library(sid: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    return casting.import_voice(session.current_handle(), sid, str(body.get("voice_id") or ""))


# ----------------------------------------------------------------------------
# tuning
# ----------------------------------------------------------------------------


@router.get("/characters/{sid}/tuning")
def get_tuning(sid: str) -> dict[str, Any]:
    h = session.current_handle()
    chars = viewdata.load_characters(session.read_ctx(h))
    if sid not in chars:
        raise AppError("bad_request", params={"field": "speaker_id", "value": sid}, status=404)
    return casting.get_tuning(chars[sid])


@router.put("/characters/{sid}/tuning")
def put_tuning(sid: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    return casting.set_tuning(session.current_handle(), sid, body)


@router.post("/characters/{sid}/tuning/listen")
def tuning_listen(sid: str, body: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    h = session.current_handle()
    chars = viewdata.load_characters(session.read_ctx(h))
    if sid not in chars:
        raise AppError("bad_request", params={"field": "speaker_id", "value": sid}, status=404)
    tuning = casting.validate_tuning(body["tuning"]) if body.get("tuning") is not None else casting.get_tuning(chars[sid])
    return pipeline.submit("tuning_listen", h.slug, {"speaker_id": sid, "tuning": tuning}).info()


@router.post("/characters/{sid}/tuning/apply-similar")
def apply_similar(sid: str) -> dict[str, Any]:
    return {"updated": casting.apply_tuning_to_similar(session.current_handle(), sid)}


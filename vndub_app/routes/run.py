"""Step 4 (preview), Step 5 (full run) and GPU status routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body

from vndub_app import appdirs, gpuinfo, pipeline, session, viewdata
from vndub_app.errors import AppError

router = APIRouter(prefix="/api")

PREVIEW_MODES = ("first_scene", "first_n")


@router.post("/preview/run")
def preview_run(body: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    h = session.current_handle()
    mode = body.get("mode") or "first_scene"
    if mode not in PREVIEW_MODES:
        raise AppError("bad_request", params={"field": "mode", "value": mode}, status=422)
    try:
        n = 40 if body.get("n") is None else int(body["n"])
    except (TypeError, ValueError):
        raise AppError("bad_request", params={"field": "n", "value": body.get("n")}, status=422) from None
    if not 1 <= n <= 1000:
        raise AppError("bad_request", params={"field": "n", "value": n}, status=422)
    ids = pipeline.first_scene_ids(h) if mode == "first_scene" else pipeline.first_n_ids(h, n)
    if not ids:
        raise AppError("import_required", action="choose_game_dir", status=400)
    with session.STATE_LOCK:
        st = appdirs.read_app_state(h)
        st["preview"] = {**(st.get("preview") or {}), "mode": mode, "n": n, "line_ids": ids}
        appdirs.write_app_state(h, st)
    return pipeline.submit("preview", h.slug, {"mode": mode, "n": n, "line_ids": ids}).info()


@router.get("/preview/items")
def preview_items() -> dict[str, Any]:
    h = session.current_handle()
    ctx = session.read_ctx(h)
    ids = list(appdirs.read_app_state(h).get("preview", {}).get("line_ids") or [])
    snap = viewdata.snapshot(h, ctx)
    chars = viewdata.load_characters(ctx)
    trs = viewdata.translations(ctx)
    sel = viewdata.selection(ctx)
    fin = viewdata.final_index(ctx)
    items = []
    for lid in ids:
        l = snap.by_id.get(lid)
        if l is None:
            continue
        tr = trs.get(lid)
        s = sel.get(lid)
        dub = None
        f = fin.get(lid)
        if f is not None and ctx.paths.abs(f.final_wav).exists():
            dub = session.work_url(h, ctx.paths.abs(f.final_wav))
        elif s is not None and s.winner and ctx.paths.abs(s.winner).exists():
            dub = session.work_url(h, ctx.paths.abs(s.winner))
        items.append({
            "line_id": lid, "speaker_id": l.speaker_id, "name": viewdata.display_name(l.speaker_id, chars),
            "text_src": l.text_src, "text_tgt": tr.text_tgt if tr else "",
            "original_url": session.work_url(h, l.src_wav) if snap.is_voiced(lid) and l.src_wav else None,
            "dub_url": dub, "status": viewdata.status_of(s) if s is not None else "pending",
        })
    return {"line_ids": ids, "items": items}


@router.post("/run/full")
def run_full(body: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    h = session.current_handle()
    shutdown = bool(body.get("shutdown_when_done"))
    s = appdirs.load_settings()
    if s.shutdown_when_done != shutdown:
        s.shutdown_when_done = shutdown
        appdirs.save_settings(s)
    return pipeline.submit("full", h.slug, {"shutdown_when_done": shutdown}).info()


@router.get("/gpu")
def gpu() -> dict[str, Any]:
    return gpuinfo.query() or {"available": False}

"""Step 6 data: the review list, a line's candidates / verdicts / "why", and review actions.

Status mapping (see ``viewdata.status_of``): selection ``auto`` / ``override`` -> ``ok``;
``review`` / ``regen`` with a winner -> ``check``; no winner -> ``failed``. Only lines that
have a selection record are listed (nothing to review before ``score`` ran for them).
Candidate labels A, B, C ... follow the composite order (best first); the review shows the
top three plus the current winner.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from vndub_app import appdirs, casting, session, verdicts, viewdata
from vndub_app.errors import AppError

MAX_CANDIDATES = 3
ACTIONS = ("accept", "reject", "regen", "set_text", "set_ref", "keep_original", "set_src_text", "tuning")
STATUS_ORDER = {"check": 0, "failed": 1, "ok": 2}

_LOCK = threading.RLock()
_LABELS: dict[str, tuple[tuple[Any, ...], dict[str, str]]] = {}
_DELIVERY: dict[str, tuple[tuple[Any, ...], dict[str, dict[str, Any]]]] = {}


def label_for(i: int) -> str:
    return chr(ord("A") + i) if i < 26 else f"A{i}"


# ----------------------------------------------------------------------------
# scores of a line
# ----------------------------------------------------------------------------


def _rejected(ov: list[Any]) -> set[str]:
    return {str(o.payload.get("candidate") or o.payload.get("wav")) for o in ov if o.action == "reject"}


def ordered_candidates(ctx: Any, line_id: str, ov: list[Any]) -> list[Any]:
    """Scored candidates of a line without rejected ones, best composite first (stable by file name)."""
    from vndub.manifest.io import CandidateScore, read_jsonl

    p = ctx.paths.scores(line_id)
    if not p.exists():
        return []
    try:
        cands = read_jsonl(p, CandidateScore)
    except ValueError:
        return []
    rej = _rejected(ov)
    cands = [c for c in cands if c.wav not in rej]
    return sorted(cands, key=lambda c: (-(c.composite if c.composite is not None else -1.0), c.wav))


def label_map(h: appdirs.ProjectHandle, ctx: Any, sel: dict[str, Any], ovs: dict[str, list[Any]]) -> dict[str, str]:
    """line_id -> label of its winner among the ordered candidates (cached per selection/overrides state)."""
    key = (viewdata._mt(ctx.paths.selection), viewdata._mt(ctx.paths.overrides))
    with _LOCK:
        hit = _LABELS.get(h.slug)
        if hit and hit[0] == key:
            return hit[1]
    out: dict[str, str] = {}
    for lid, s in sel.items():
        if not s.winner:
            continue
        cands = ordered_candidates(ctx, lid, ovs.get(lid, []))
        for i, c in enumerate(cands):
            if c.wav == s.winner:
                out[lid] = label_for(i)
                break
    with _LOCK:
        _LABELS[h.slug] = (key, out)
    return out


def delivery_map(h: appdirs.ProjectHandle, ctx: Any, snap: Any) -> dict[str, dict[str, Any]]:
    """Unvoiced lines -> ``{source: text, layer}`` from their text passports (cached)."""
    st = viewdata.stage_status(ctx)
    key = ((st.get("text_emotion") or {}).get("finished"), len(snap.voiced))
    with _LOCK:
        hit = _DELIVERY.get(h.slug)
        if hit and hit[0] == key:
            return hit[1]
    out: dict[str, dict[str, Any]] = {}
    for l in snap.dubbable():
        if snap.is_voiced(l.line_id):
            continue
        pp = viewdata.passport(ctx, l.line_id)
        if pp and pp.get("source") == "text":
            out[l.line_id] = {"source": "text", "layer": (pp.get("delivery") or {}).get("layer")}
    with _LOCK:
        _DELIVERY[h.slug] = (key, out)
    return out


def touched_ids(h: appdirs.ProjectHandle) -> list[str]:
    return list(appdirs.read_app_state(h).get("review", {}).get("touched", []))


def add_touched(h: appdirs.ProjectHandle, ids: list[str]) -> list[str]:
    with _LOCK, session.STATE_LOCK:
        st = appdirs.read_app_state(h)
        cur = list(st.get("review", {}).get("touched", []))
        for i in ids:
            if i not in cur:
                cur.append(i)
        st.setdefault("review", {})["touched"] = cur
        appdirs.write_app_state(h, st)
    return cur


# ----------------------------------------------------------------------------
# list
# ----------------------------------------------------------------------------


def make_row(l: Any, s: Any, tr: Any, chars: dict[str, Any], winner_label: str | None, delivery: dict[str, Any] | None, touched: bool) -> dict[str, Any]:
    return {
        "line_id": l.line_id, "speaker_id": l.speaker_id, "name": viewdata.display_name(l.speaker_id, chars), "scene": l.scene,
        "status": viewdata.status_of(s), "composite": (s.scores or {}).get("composite"), "text_src": l.text_src,
        "text_tgt": tr.text_tgt if tr else "", "winner_label": winner_label, "flags": list(l.flags or []),
        "delivery": delivery, "touched": touched, "_order": l.order,
    }


def lines_view(
    h: appdirs.ProjectHandle, *, speaker: str | None = None, scene: str | None = None, status: str | None = None,
    q: str | None = None, lo: float | None = None, hi: float | None = None, limit: int | None = None, offset: int = 0,
) -> dict[str, Any]:
    ctx = session.read_ctx(h)
    snap = viewdata.snapshot(h, ctx)
    sel = viewdata.selection(ctx)
    if not sel or not snap.lines:
        return {"items": [], "facets": {"speakers": [], "scenes": []}, "counts": {"ok": 0, "check": 0, "failed": 0}, "total": 0}
    trs = viewdata.translations(ctx)
    chars = viewdata.load_characters(ctx)
    ovs = viewdata.overrides_by_line(ctx)
    labels = label_map(h, ctx, sel, ovs)
    delivery = delivery_map(h, ctx, snap)
    touched = set(touched_ids(h))
    needle = (q or "").strip().lower()

    rows: list[dict[str, Any]] = []
    sp_count: dict[str, int] = {}
    scene_count: dict[str, int] = {}
    counts = {"ok": 0, "check": 0, "failed": 0}
    for l in snap.lines:
        s = sel.get(l.line_id)
        if s is None:
            continue
        st = viewdata.status_of(s)
        counts[st] += 1
        sp_count[l.speaker_id] = sp_count.get(l.speaker_id, 0) + 1
        scene_count[l.scene] = scene_count.get(l.scene, 0) + 1
        comp = s.scores.get("composite") if s.scores else None
        if speaker and l.speaker_id != speaker:
            continue
        if scene and l.scene != scene:
            continue
        if status and st != status:
            continue
        if lo is not None and (comp is None or comp < lo):
            continue
        if hi is not None and (comp is None or comp > hi):
            continue
        tr = trs.get(l.line_id)
        if needle and needle not in (l.text_src or "").lower() and needle not in ((tr.text_tgt if tr else "") or "").lower() and needle not in l.line_id.lower():
            continue
        rows.append(make_row(l, s, tr, chars, labels.get(l.line_id), delivery.get(l.line_id) or ({"source": "audio", "layer": None} if snap.is_voiced(l.line_id) else None), l.line_id in touched))
    rows.sort(key=lambda r: (STATUS_ORDER[r["status"]], r["composite"] if r["composite"] is not None else -1.0 if r["status"] != "ok" else 0.0, r["_order"]))
    total = len(rows)
    if limit is not None:
        rows = rows[max(0, offset): max(0, offset) + max(0, limit)]
    for r in rows:
        r.pop("_order", None)
    speakers = [{"speaker_id": k, "name": viewdata.display_name(k, chars), "count": v} for k, v in sorted(sp_count.items(), key=lambda kv: (-kv[1], kv[0]))]
    scenes = [{"scene": k, "count": v} for k, v in scene_count.items()]
    return {"items": rows, "facets": {"speakers": speakers, "scenes": scenes}, "counts": counts, "total": total}


# ----------------------------------------------------------------------------
# detail
# ----------------------------------------------------------------------------


def _thresholds(ctx: Any) -> verdicts.Thresholds:
    from vndub.manifest.io import read_json

    calib = None
    try:
        calib = read_json(ctx.paths.spk_calibration)
    except Exception:
        calib = None
    return verdicts.Thresholds.from_config(ctx.cfg, calib)


def _translation_failed(tr: Any) -> bool:
    if tr is None:
        return True
    flags = set(tr.flags or [])
    return bool({"translate_failed", "translate_missing"} & flags) or not (tr.text_tts or "").strip()


def line_detail(h: appdirs.ProjectHandle, line_id: str) -> dict[str, Any]:
    from vndub.manifest.io import read_json

    ctx = session.read_ctx(h)
    snap = viewdata.snapshot(h, ctx)
    line = snap.by_id.get(line_id)
    sel_all = viewdata.selection(ctx)
    sel = sel_all.get(line_id)
    if line is None or sel is None:
        raise AppError("file_not_found", params={"path": line_id}, status=404)
    trs = viewdata.translations(ctx)
    tr = trs.get(line_id)
    chars = viewdata.load_characters(ctx)
    ovs = viewdata.overrides_by_line(ctx).get(line_id, [])
    th = _thresholds(ctx)
    interj = bool(tr and "interjection" in (tr.flags or []))
    ref = "original" if snap.is_voiced(line_id) else "performer"  # what spk_sim was measured against
    cands = ordered_candidates(ctx, line_id, ovs)

    shown = cands[:MAX_CANDIDATES]
    if sel.winner and not any(c.wav == sel.winner for c in shown):
        win = next((c for c in cands if c.wav == sel.winner), None)
        if win is not None:
            shown = [*shown[: MAX_CANDIDATES - 1], win]
    out_cands = []
    for c in shown:
        label = label_for(next(i for i, x in enumerate(cands) if x.wav == c.wav))
        out_cands.append({
            "id": c.wav, "label": label, "url": session.work_url(h, c.wav), "backend": c.backend, "route": c.route,
            "composite": c.composite, "passed": bool(c.passed), "is_winner": c.wav == sel.winner,
            "verdicts": verdicts.verdicts_for(c.metrics, th, route=c.route, interjection=interj, reference=ref),
        })
    if sel.winner and not any(c["is_winner"] for c in out_cands):  # e.g. keep-original without a score record
        wp = ctx.paths.abs(sel.winner)
        out_cands.append({
            "id": sel.winner, "label": label_for(len(cands)), "url": session.work_url(h, wp) if wp.exists() else None,
            "backend": sel.backend or "orig", "route": sel.route or "orig", "composite": (sel.scores or {}).get("composite"), "passed": True,
            "is_winner": True, "verdicts": verdicts.verdicts_for(sel.scores or {}, th, route=sel.route or "", interjection=interj, reference=ref),
        })

    best = next((c for c in cands if c.wav == sel.winner), None) or (cands[0] if cands else None)
    why = verdicts.why_items(
        metrics=best.metrics if best else (sel.scores or {}), gates=best.gates if best else {}, th=th, interjection=interj,
        has_candidates=bool(cands) or bool(sel.winner), selection_status=sel.status, selection_reason=sel.reason,
        line_flags=[f for f in (line.flags or []) if not (f == "text_from_asr" and "text_edited" in (line.flags or []))],
        translation_failed=_translation_failed(tr), voice_reference=ref,
    )
    if sel.status == "regen" and (sel.reason or "").startswith("override"):
        why.append({"gate": "review", "key": "pending_change", "params": {}})
    if not sel.winner and not any(w["key"] in ("no_candidates", "translation_failed") for w in why):
        why.append({"gate": "synth", "key": "no_candidates", "params": {"detail": sel.reason}})  # a failed line always says why

    style_prompt = ""
    wsrc = sel.winner or (best.wav if best else None)
    if wsrc:
        side = read_json(ctx.paths.abs(wsrc).with_suffix(".json"), {}) or {}
        style_prompt = str(side.get("style_prompt") or "")

    refs = []
    for r in read_json(ctx.paths.vb_refs_json(line.speaker_id), []) or []:
        refs.append({"path": r.get("path"), "url": session.work_url(h, r.get("path")), "emotion": r.get("emotion", "neutral")})

    tuning_override = None
    for o in ovs:
        if o.action == "tuning":
            tuning_override = o.payload
    pp = viewdata.passport(ctx, line_id)
    if pp and pp.get("source") == "text":
        delivery: dict[str, Any] | None = {"source": "text", "layer": (pp.get("delivery") or {}).get("layer")}
    else:
        delivery = {"source": "audio", "layer": None} if snap.is_voiced(line_id) else None
    winner_label = next((label_for(i) for i, c in enumerate(cands) if c.wav == sel.winner), None)
    row = make_row(line, sel, tr, chars, winner_label, delivery, line_id in touched_ids(h))
    row.pop("_order", None)
    return {
        "line": row,
        "original_url": session.work_url(h, line.src_wav) if snap.is_voiced(line_id) and line.src_wav else None,
        "candidates": out_cands, "why": why, "style_prompt": style_prompt,
        "text_tts": (tr.text_tts or tr.text_tgt) if tr else "", "refs": refs, "tuning_override": tuning_override,
        "selection": {"status": sel.status, "route": sel.route, "backend": sel.backend},
    }


# ----------------------------------------------------------------------------
# actions
# ----------------------------------------------------------------------------


def _resolve_candidate(ctx: Any, line_id: str, payload: dict[str, Any], *, default_winner: bool) -> str:
    ov = viewdata.overrides_by_line(ctx).get(line_id, [])
    cands = ordered_candidates(ctx, line_id, ov)
    sel = viewdata.selection(ctx).get(line_id)
    want = payload.get("candidate") or payload.get("wav")
    if not want and payload.get("label"):
        lab = str(payload["label"]).upper()
        hit = next((c for i, c in enumerate(cands) if label_for(i) == lab), None)
        want = hit.wav if hit else None
    if not want and default_winner and sel is not None and sel.winner:
        want = sel.winner
    if not want or (not any(c.wav == want for c in cands) and not (sel is not None and sel.winner == want)):
        raise AppError("bad_request", params={"field": "candidate", "value": want}, status=422)
    return str(want)


def apply_action(h: appdirs.ProjectHandle, line_id: str, action: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    from vndub.review.app import write_override

    if action not in ACTIONS:
        raise AppError("bad_request", params={"field": "action", "value": action}, status=422)
    payload = dict(payload or {})
    ctx = session.read_ctx(h)
    snap = viewdata.snapshot(h, ctx)
    if line_id not in snap.by_id:
        raise AppError("file_not_found", params={"path": line_id}, status=404)

    writes: list[tuple[str, dict[str, Any]]] = []
    if action in ("accept", "reject"):
        cand = _resolve_candidate(ctx, line_id, payload, default_winner=action == "accept")
        writes.append((action, {"candidate": cand}))
    elif action == "regen":
        text, style = str(payload.get("text") or "").strip(), str(payload.get("style_prompt") or "")
        if text:
            writes.append(("set_text", {"text": text, "style_prompt": style}))
        writes.append(("regen", {k: v for k, v in {"style_prompt": style, "text": text}.items() if v}))
    elif action == "set_text":
        text = str(payload.get("text") or "").strip()
        if not text:
            raise AppError("bad_request", params={"field": "text"}, status=422)
        writes.append(("set_text", {"text": text, **({"style_prompt": str(payload["style_prompt"])} if payload.get("style_prompt") else {})}))
    elif action == "set_src_text":
        text = str(payload.get("text") or "").strip()
        if not text:
            raise AppError("bad_request", params={"field": "text"}, status=422)
        writes.append(("set_src_text", {"text": text}))
    elif action == "set_ref":
        ref = str(payload.get("ref") or "").replace("\\", "/")
        if not ref or ".." in ref.split("/") or Path(ref).is_absolute() or not (h.work / ref).is_file():
            raise AppError("bad_request", params={"field": "ref", "value": ref}, status=422)
        writes.append(("set_ref", {"ref": ref}))
    elif action == "keep_original":
        writes.append(("keep_original", {}))
    elif action == "tuning":
        writes.append(("tuning", casting.validate_tuning(payload)))
    for act, pl in writes:
        write_override(ctx.paths, line_id, act, pl)
    viewdata.clear_cache(h.slug)
    touched = add_touched(h, [line_id])
    return {"ok": True, "touched": touched, "touched_count": len(touched)}


def prefer_route(h: appdirs.ProjectHandle, route: str, speaker_id: str | None = None) -> dict[str, Any]:
    """Accept the best candidate of ``route`` that passed its gates, for every line in scope whose winner is of another route."""
    from vndub.review.app import write_override

    if route not in ("A", "B", "C", "P"):
        raise AppError("bad_request", params={"field": "route", "value": route}, status=422)
    ctx = session.read_ctx(h)
    snap = viewdata.snapshot(h, ctx)
    sel = viewdata.selection(ctx)
    ovs = viewdata.overrides_by_line(ctx)
    updated: list[str] = []
    for l in snap.lines:
        if speaker_id and l.speaker_id != speaker_id:
            continue
        s = sel.get(l.line_id)
        if s is None or s.status == "override" or s.route == route:
            continue
        mine = [c for c in ordered_candidates(ctx, l.line_id, ovs.get(l.line_id, [])) if c.route == route and c.passed]
        if not mine:
            continue
        write_override(ctx.paths, l.line_id, "accept", {"candidate": mine[0].wav, "prefer_route": route})
        updated.append(l.line_id)
    if updated:
        add_touched(h, updated)
        viewdata.clear_cache(h.slug)
    return {"updated": len(updated), "lines": updated}

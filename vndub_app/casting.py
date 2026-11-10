"""Step 3 / 3b data: characters, casting (voice designs), tuning, the voice library.

The designs themselves come from ``vndub.stages.voicebank`` (task P3); this module decorates
them with audio URLs, reads/writes the casting fields of ``characters.yaml`` and implements
the app-side parts: three original clips per character, RVC chip state, the tuning panel,
the saved-voice library and the "listen" line picking.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import math
import random
import shutil
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml

from vndub.util import now_iso
from vndub_app import appdirs, jobs, session, viewdata
from vndub_app.errors import AppError

log = logging.getLogger("vndub_app.casting")

_CHAR_LOCK = threading.RLock()
_CLIP_CACHE: dict[str, tuple[tuple[Any, ...], dict[str, dict[str, Any]]]] = {}

ROUTES = ("A", "B", "C", "P")
GENDERS = ("m", "f", "unknown")

TUNING_DEFAULT: dict[str, Any] = {"pitch_st": 0.0, "speed": 1.0, "expressiveness": 0.5, "energy": 0.0, "timbre_preset": "", "character_note": ""}
TUNING_RANGES = {"pitch_st": (-6.0, 6.0), "speed": (0.8, 1.2), "expressiveness": (0.0, 1.0), "energy": (-1.0, 1.0)}
TIMBRE_PRESETS = ("", "younger", "older", "softer", "sharper")


# ----------------------------------------------------------------------------
# characters.yaml access
# ----------------------------------------------------------------------------


@contextmanager
def edit_characters(h: appdirs.ProjectHandle) -> Iterator[dict[str, Any]]:
    """Read-modify-write ``characters.yaml`` (atomic write; app-side lock)."""
    from vndub.manifest.io import load_characters, save_characters

    path = session.work_paths(h).characters
    with _CHAR_LOCK:
        chars = load_characters(path) if path.exists() else {}
        yield chars
        save_characters(path, chars)


def _char_or_404(chars: dict[str, Any], sid: str) -> Any:
    if sid not in chars:
        raise AppError("bad_request", params={"field": "speaker_id", "value": sid}, status=404)
    return chars[sid]


def update_character(h: appdirs.ProjectHandle, sid: str, *, gender: str | None = None, age_hint: str | None = None,
                     description: str | None = None, name_tgt: str | None = None) -> None:
    if gender is not None and gender not in GENDERS:
        raise AppError("bad_request", params={"field": "gender", "value": gender}, status=422)
    with edit_characters(h) as chars:
        c = _char_or_404(chars, sid)
        if gender is not None:
            c.gender = gender
            setattr(c, "gender_source", "user")  # contract C3: the translate stage's guess no longer applies to it
        if age_hint is not None:
            c.age_hint = str(age_hint)
        if description is not None:
            c.description = str(description)
        if name_tgt is not None:
            c.names = {**c.names, "tgt": str(name_tgt)}
    from vndub_app import translation

    translation.write_glossary(h)


# ----------------------------------------------------------------------------
# tuning
# ----------------------------------------------------------------------------


def validate_tuning(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate ranges and fill defaults; ``AppError(bad_request)`` names the offending field."""
    out = dict(TUNING_DEFAULT)
    for k, v in (raw or {}).items():
        if k not in TUNING_DEFAULT:
            raise AppError("bad_request", params={"field": k}, status=422)
        if k in TUNING_RANGES:
            try:
                f = float(v)
            except (TypeError, ValueError):
                raise AppError("bad_request", params={"field": k, "value": v}, status=422) from None
            lo, hi = TUNING_RANGES[k]
            if math.isnan(f) or not lo <= f <= hi:
                raise AppError("bad_request", params={"field": k, "value": f, "min": lo, "max": hi}, status=422)
            out[k] = round(f, 4)
        elif k == "timbre_preset":
            if v not in TIMBRE_PRESETS:
                raise AppError("bad_request", params={"field": k, "value": v}, status=422)
            out[k] = v
        else:  # character_note: the style prompt keeps only NOTE_MAX_CHARS of it, so a longer one is refused, not cut silently
            from vndub.text.style_prompt import NOTE_MAX_CHARS

            note = str(v or "").strip()
            if len(note) > NOTE_MAX_CHARS:
                raise AppError("bad_request", params={"field": k, "value": len(note), "max": NOTE_MAX_CHARS}, status=422)
            out[k] = note
    return out


def get_tuning(c: Any) -> dict[str, Any]:
    t = (c.performer or {}).get("tuning") or {}
    out = dict(TUNING_DEFAULT)
    out.update({k: v for k, v in t.items() if k in TUNING_DEFAULT})
    return out


def set_tuning(h: appdirs.ProjectHandle, sid: str, tuning: dict[str, Any]) -> dict[str, Any]:
    t = validate_tuning(tuning)
    with edit_characters(h) as chars:
        c = _char_or_404(chars, sid)
        perf = dict(c.performer or {})
        perf["tuning"] = t
        c.performer = perf
    return t


def apply_tuning_to_similar(h: appdirs.ProjectHandle, sid: str) -> list[str]:
    """Copy the tuning of ``sid`` to characters with the same known gender and the same age_hint. A character of unknown
    gender matches nobody: with everything ``unknown`` one character's tuning would land on the whole cast."""
    updated: list[str] = []
    with edit_characters(h) as chars:
        src = _char_or_404(chars, sid)
        if src.gender not in ("m", "f"):
            return updated
        t = get_tuning(src)
        for other, c in chars.items():
            if other in (sid, viewdata.NARRATOR) or c.gender != src.gender or (c.age_hint or "") != (src.age_hint or ""):
                continue
            perf = dict(c.performer or {})
            perf["tuning"] = dict(t)
            c.performer = perf
            updated.append(other)
    return updated


@contextmanager
def temporary_tuning(h: appdirs.ProjectHandle, sid: str, tuning: dict[str, Any] | None) -> Iterator[None]:
    """Render with ``tuning`` without persisting it: the stored value is restored afterwards."""
    if tuning is None:
        yield
        return
    _missing = object()
    with edit_characters(h) as chars:
        c = _char_or_404(chars, sid)
        perf = dict(c.performer or {})
        old = perf.get("tuning", _missing)
        perf["tuning"] = tuning
        c.performer = perf
    try:
        yield
    finally:
        with edit_characters(h) as chars:
            c = chars.get(sid)
            if c is not None:
                perf = dict(c.performer or {})
                if old is _missing:
                    perf.pop("tuning", None)
                else:
                    perf["tuning"] = old
                c.performer = perf


# ----------------------------------------------------------------------------
# original clips + emotion histogram (from the expression passports)
# ----------------------------------------------------------------------------


def _passport_features(pp: dict[str, Any]) -> dict[str, Any]:
    e = pp.get("emotion") or {}
    probs = ((e.get("e2v") or {}).get("probs")) or {}
    style = pp.get("style") or {}
    label = style.get("emotion") or (e.get("e2v") or {}).get("label") or "neutral"
    arousal = (e.get("adv") or {}).get("arousal")
    neutral = probs.get("neutral")
    if arousal is None and pp.get("source") == "text":
        arousal = (pp.get("delivery") or {}).get("intensity")
    emo_score = arousal if arousal is not None else (1.0 - neutral if neutral is not None else 0.0)
    return {
        "label": str(label).lower(), "neutral": neutral if neutral is not None else (1.0 if str(label).lower() == "neutral" else 0.0),
        "emo": float(emo_score), "dur": float(pp.get("duration_s") or 0.0), "source": pp.get("source", "audio"),
    }


def _pick_clips(feats: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    chosen: list[tuple[str, dict[str, Any]]] = []
    used: set[str] = set()

    def pick(kind: str, key: Any) -> None:
        pool = [f for f in feats if f["id"] not in used]  # fewer than three voiced lines: fewer clips, none twice
        if pool:
            best = max(pool, key=key)
            used.add(best["id"])
            chosen.append((kind, best))

    pick("neutral", lambda f: (f["neutral"], -f["emo"]))
    pick("emotional", lambda f: (f["emo"], 1.0 - f["neutral"]))
    pick("longest", lambda f: f["dur"])
    return chosen


def compute_clips(h: appdirs.ProjectHandle, ctx: Any, snap: Any, sid: str, voiced: bool) -> dict[str, Any]:
    lines = [l for l in snap.lines if l.speaker_id == sid]
    feats: list[dict[str, Any]] = []
    hist: dict[str, int] = {}
    for l in lines:
        is_v = snap.is_voiced(l.line_id)
        if voiced != is_v:
            continue
        pp = viewdata.passport(ctx, l.line_id)
        if not pp:
            continue
        f = _passport_features(pp)
        f.update(id=l.line_id, line=l)
        hist[f["label"]] = hist.get(f["label"], 0) + 1
        if is_v and f["source"] != "text" and l.src_wav:
            feats.append(f)
    clips = []
    for kind, f in _pick_clips(feats):
        l = f["line"]
        clips.append({"kind": kind, "line_id": l.line_id, "url": session.work_url(h, l.src_wav), "text_src": l.text_src, "emotion": f["label"]})
    total = sum(hist.values()) or 1
    return {"clips": clips, "hist": {k: round(v / total, 3) for k, v in sorted(hist.items(), key=lambda kv: -kv[1])}}


def clips_for(h: appdirs.ProjectHandle, ctx: Any, snap: Any, sid: str, voiced: bool) -> dict[str, Any]:
    """Cached per (project, analysis state); passports of a big game are read once."""
    st = viewdata.stage_status(ctx)
    key = (
        (st.get("analyze") or {}).get("finished"), (st.get("text_emotion") or {}).get("finished"),
        (st.get("link") or {}).get("finished"), len(snap.voiced),
    )
    with _CHAR_LOCK:
        hit = _CLIP_CACHE.get(h.slug)
        if hit and hit[0] == key and sid in hit[1]:
            return hit[1][sid]
        store = hit[1] if hit and hit[0] == key else {}
    val = compute_clips(h, ctx, snap, sid, voiced)
    with _CHAR_LOCK:
        store[sid] = val
        _CLIP_CACHE[h.slug] = (key, store)
    return val


# ----------------------------------------------------------------------------
# character view
# ----------------------------------------------------------------------------


def rvc_state(h: appdirs.ProjectHandle, c: Any, sid: str) -> dict[str, Any]:
    js = [j for j in jobs.runner().list(h.slug) if j.kind == "rvc_train" and j.params.get("speaker_id") == sid]
    last = js[-1] if js else None
    if last is not None and last.state in ("queued", "paused"):
        return {"state": "queued", "progress": 0.0, "job_id": last.id}
    if last is not None and last.state == "running":
        prog = (last.done / last.total) if last.total else 0.0
        return {"state": "training", "progress": round(float(prog), 3), "job_id": last.id}
    model = (c.rvc or {}).get("model")
    if model and (h.work / model).exists():
        return {"state": "done", "progress": 1.0, "job_id": None}
    if last is not None and last.state == "failed":
        return {"state": "failed", "progress": 0.0, "job_id": last.id}
    reason = (last.result or {}).get("reason_code") if last is not None and last.state == "done" else None
    return {"state": "none", "progress": 0.0, "job_id": None, **({"reason_code": reason} if reason and reason != "none" else {})}


def character_dict(h: appdirs.ProjectHandle, ctx: Any, snap: Any, sid: str, c: Any, counts: dict[str, tuple[int, int]]) -> dict[str, Any]:
    lines_n, voiced_n = counts.get(sid, (0, 0))
    mode = "voiced" if voiced_n and voiced_n >= lines_n else ("unvoiced" if not voiced_n else "mixed")
    perf = c.performer or {}
    minutes = float(c.stats.total_sec or 0.0) / 60.0
    clips = {"clips": [], "hist": {}}
    if voiced_n or ctx.paths.analysis.exists():
        clips = clips_for(h, ctx, snap, sid, voiced=bool(voiced_n))
    hist = dict(c.stats.emotion_hist or {}) or clips["hist"]
    chosen = None
    if perf.get("design_id"):
        chosen = {"design_id": perf.get("design_id"), "route": c.route, "kind": perf.get("kind")}
    return {
        "speaker_id": sid, "name_src": c.names.get("src", "") or sid, "name_tgt": c.names.get("tgt", ""),
        "gender": c.gender, "age_hint": c.age_hint, "description": c.description,
        "lines": lines_n, "voiced_lines": voiced_n, "minutes": round(minutes, 2), "mode": mode,
        "emotion_hist": hist, "clips": clips["clips"], "chosen": chosen, "route": c.route,
        "tuning": get_tuning(c), "rvc": rvc_state(h, c, sid),
    }


def line_counts(snap: Any) -> dict[str, tuple[int, int]]:
    counts: dict[str, list[int]] = {}
    for l in snap.lines:
        c = counts.setdefault(l.speaker_id, [0, 0])
        c[0] += 1
        if snap.is_voiced(l.line_id):
            c[1] += 1
    return {k: (v[0], v[1]) for k, v in counts.items()}


def characters_view(h: appdirs.ProjectHandle) -> list[dict[str, Any]]:
    ctx = session.read_ctx(h)
    chars = viewdata.load_characters(ctx)
    if not chars:
        return []
    snap = viewdata.snapshot(h, ctx)
    counts = line_counts(snap)
    out = [character_dict(h, ctx, snap, sid, c, counts) for sid, c in chars.items() if sid != viewdata.NARRATOR]
    out.sort(key=lambda d: (-d["lines"], d["speaker_id"]))
    return out


def character_view(h: appdirs.ProjectHandle, sid: str) -> dict[str, Any]:
    ctx = session.read_ctx(h)
    chars = viewdata.load_characters(ctx)
    c = _char_or_404(chars, sid)
    snap = viewdata.snapshot(h, ctx)
    return character_dict(h, ctx, snap, sid, c, line_counts(snap))


def effective_route(h: appdirs.ProjectHandle, ctx: Any, sid: str) -> str:
    """The route synth will use for a character: its chosen route, else B (has original voice) / P (none)."""
    chars = viewdata.load_characters(ctx)
    c = chars.get(sid)
    if c is not None and c.route:
        return c.route
    snap = viewdata.snapshot(h, ctx)
    has_voice = any(l.speaker_id == sid and snap.is_voiced(l.line_id) for l in snap.lines)
    return "B" if has_voice else "P"


# ----------------------------------------------------------------------------
# designs
# ----------------------------------------------------------------------------


def raw_designs(ctx: Any, sid: str) -> list[dict[str, Any]]:
    """Designs of a character: ``voicebank.list_designs`` (task P3), else ``voicebank/<sid>/designs.json``."""
    try:
        from vndub.stages import voicebank

        fn = getattr(voicebank, "list_designs", None)
        if fn is not None:
            return [d if isinstance(d, dict) else dict(d) for d in fn(ctx, sid)]
    except ImportError:
        pass
    from vndub.manifest.io import read_json

    data = read_json(ctx.paths.vb(sid) / "designs.json", []) or []
    if isinstance(data, dict):
        data = data.get("designs", [])
    return list(data)


def design_view(h: appdirs.ProjectHandle, d: dict[str, Any], *, chosen_id: str | None = None, texts_src: dict[str, str] | None = None) -> dict[str, Any]:
    out = dict(d)
    samples = []
    for s in d.get("samples") or []:
        s2 = dict(s)
        s2["url"] = session.work_url(h, s.get("wav"))
        s2["text_tgt"] = s.get("text_tgt") or s.get("text") or ""
        if texts_src and s.get("line_id") in texts_src:
            s2["text_src"] = texts_src[s["line_id"]]
        samples.append(s2)
    out["samples"] = samples
    out["chosen"] = bool(chosen_id and d.get("design_id") == chosen_id)
    return out


def designs_view(h: appdirs.ProjectHandle, ctx: Any, sid: str) -> list[dict[str, Any]]:
    chars = viewdata.load_characters(ctx)
    c = chars.get(sid)
    chosen_id = (c.performer or {}).get("design_id") if c is not None else None
    snap = viewdata.snapshot(h, ctx)
    texts = {l.line_id: l.text_src for l in snap.lines if l.speaker_id == sid}
    return [design_view(h, d, chosen_id=chosen_id, texts_src=texts) for d in raw_designs(ctx, sid)]


UPLOAD_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


def check_extension(path: Path) -> None:
    """Cheap pre-check of an uploaded sample; length and decoding are validated by ``voicebank.add_reference_design``."""
    if path.suffix.lower() not in UPLOAD_EXTS:
        raise AppError("sample_bad_format", params={"file": path.name}, action="none")


def map_value_error(e: ValueError) -> AppError:
    """P3's ``add_reference_design`` raises ValueError whose message starts with an error code."""
    msg = str(e)
    for code in ("sample_too_short", "sample_too_long", "sample_bad_format"):
        if code in msg:
            return AppError(code, details=msg, action="none")
    return AppError("bad_request", params={"field": "design", "reason": msg}, details=msg, status=422)


def map_choose_error(e: ValueError, sid: str, design_id: str, route: str) -> AppError:
    """``voicebank.choose_design`` raises ValueError with a code as its message."""
    code = str(e)
    if code == "design_not_found":
        return AppError("bad_request", params={"field": "design_id", "value": design_id}, details=code, status=404)
    if code == "unknown_character":
        return AppError("bad_request", params={"field": "speaker_id", "value": sid}, details=code, status=404)
    if code in ("bad_route", "route_mismatch"):
        return AppError("bad_request", params={"field": "route", "value": route, "reason": code}, details=code, status=422)
    return AppError("bad_request", details=code, status=422)


# ----------------------------------------------------------------------------
# lines for "listen"
# ----------------------------------------------------------------------------


def _speaker_lines(h: appdirs.ProjectHandle, ctx: Any, sid: str) -> list[Any]:
    snap = viewdata.snapshot(h, ctx)
    trs = viewdata.translations(ctx)
    sel = viewdata.selection(ctx)
    out = []
    for l in snap.dubbable():
        if l.speaker_id != sid:
            continue
        tr = trs.get(l.line_id)
        if tr is None or not (tr.text_tgt or "").strip() or {"translate_failed", "translate_missing"} & set(tr.flags or []):
            continue
        s = sel.get(l.line_id)
        if s is not None and s.status == "override":
            continue
        out.append(l)
    return out


def pick_lines(h: appdirs.ProjectHandle, sid: str, n: int, *, mode: str = "random", seed: int | None = None) -> list[str]:
    ctx = session.read_ctx(h)
    lines = _speaker_lines(h, ctx, sid)
    if not lines:
        raise AppError("translation_missing", params={"speaker_id": sid}, action="open_translation", status=400)
    long = [l for l in lines if len((l.text_src or "").strip()) >= 6] or lines
    rnd = random.Random(seed)
    return [l.line_id for l in sorted(rnd.sample(long, min(max(1, n), len(long))), key=lambda l: l.order)]


def pick_tuning_lines(h: appdirs.ProjectHandle, sid: str) -> list[tuple[str, str]]:
    """One neutral and one emotional line of the character (by the expression passports)."""
    ctx = session.read_ctx(h)
    lines = _speaker_lines(h, ctx, sid)
    if not lines:
        raise AppError("translation_missing", params={"speaker_id": sid}, action="open_translation", status=400)
    pool = [l for l in lines if len((l.text_src or "").strip()) >= 6] or lines
    scored = []
    for l in pool:
        pp = viewdata.passport(ctx, l.line_id)
        scored.append((_passport_features(pp)["emo"] if pp else None, l))
    if all(s is None for s, _ in scored):
        neutral, emo = pool[0], pool[-1]
    else:
        ranked = sorted(scored, key=lambda t: (t[0] if t[0] is not None else 0.5, t[1].order))
        neutral, emo = ranked[0][1], ranked[-1][1]
    picks = [(neutral.line_id, "neutral")]
    if emo.line_id != neutral.line_id:
        picks.append((emo.line_id, "emotional"))
    return picks


def text_tgt_of(h: appdirs.ProjectHandle, line_id: str) -> str:
    tr = viewdata.translations(session.read_ctx(h)).get(line_id)
    return tr.text_tgt if tr else ""


def listen_items(h: appdirs.ProjectHandle, ids: list[str], rendered: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ctx = session.read_ctx(h)
    snap = viewdata.snapshot(h, ctx)
    trs = viewdata.translations(ctx)
    out = []
    for lid in ids:
        l = snap.by_id.get(lid)
        tr = trs.get(lid)
        out.append({
            "line_id": lid, "text_src": l.text_src if l else "", "text_tgt": tr.text_tgt if tr else "",
            "original_url": session.work_url(h, l.src_wav) if l is not None and snap.is_voiced(lid) and l.src_wav else None,
            "result_url": (rendered.get(lid) or {}).get("url"),
        })
    return out


@functools.lru_cache(maxsize=1)
def routes_accept_p() -> bool:
    """Does the pipeline config allow route ``P`` in ``synth.routes`` (added by the casting task)?"""
    try:
        from vndub.config import PipelineConfig

        PipelineConfig.model_validate({"synth": {"routes": {"P": True}}})
        return True
    except Exception:
        return False


def route_sets(route: str) -> dict[str, Any]:
    sets: dict[str, Any] = {f"synth.routes.{r}": r == route for r in ("A", "B", "C")}
    if route == "P" and routes_accept_p():
        sets["synth.routes.P"] = True
    return sets


def select_listen_winners(ctx: Any, line_ids: list[str], route: str) -> None:
    """After a one-candidate synth: make that candidate the line's winner so ``master`` renders it.

    No scoring models are loaded (the listen target is a few seconds); the next real ``score``
    run recomputes the selection of these lines from scratch.
    """
    from vndub.manifest.io import SelectionRecord, write_jsonl
    from vndub.stages import genutil as gu

    sel = viewdata.selection(ctx)
    for lid in line_ids:
        best: tuple[str, Any, Any] | None = None
        for wav, side in gu.candidate_files(ctx, lid):
            if side is None or (side.model_extra or {}).get("error") or side.route == "orig":
                continue
            created = str((side.model_extra or {}).get("created", "")) or f"{wav.stat().st_mtime_ns:020d}"
            prefer = "1" if side.route == route else "0"
            key = prefer + created
            if best is None or key > best[0]:
                best = (key, wav, side)
        if best is None:
            continue
        _, wav, side = best
        sel[lid] = SelectionRecord(
            line_id=lid, winner=ctx.paths.rel(wav), backend=side.backend, route=side.route, scores={}, status="auto",
            reason="app listen (not scored)", round=side.round, n_candidates=1,
        )
    order = {l.line_id: i for i, l in enumerate(ctx.lines())}
    write_jsonl(ctx.paths.selection, sorted(sel.values(), key=lambda s: order.get(s.line_id, 1 << 30)))


# ----------------------------------------------------------------------------
# voice library  (<data>/voices/<id>.yaml + <id>.wav)
# ----------------------------------------------------------------------------


def _voice_files() -> list[Path]:
    return sorted(appdirs.voices_dir().glob("*.yaml"))


def _read_voice(p: Path) -> dict[str, Any] | None:
    try:
        d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    d.setdefault("id", p.stem)
    return d if isinstance(d, dict) else None


def voice_entry(d: dict[str, Any]) -> dict[str, Any]:
    wav = appdirs.voices_dir() / f"{d['id']}.wav"
    return {
        "id": d["id"], "name": d.get("name", d["id"]), "gender": d.get("gender", "unknown"), "description": d.get("description", ""),
        "sample_url": f"/api/audio/_voices/{d['id']}.wav" if wav.exists() else None, "source_project": d.get("source_project"),
    }


def list_voices() -> list[dict[str, Any]]:
    out = []
    for p in _voice_files():
        d = _read_voice(p)
        if d is not None:
            out.append(voice_entry(d))
    out.sort(key=lambda v: v["name"].lower())
    return out


def get_voice(voice_id: str) -> dict[str, Any]:
    if not voice_id or any(c in voice_id for c in "/\\.") or voice_id.startswith("_"):
        raise AppError("file_not_found", params={"path": voice_id}, status=404)
    p = appdirs.voices_dir() / f"{voice_id}.yaml"
    d = _read_voice(p) if p.exists() else None
    if d is None:
        raise AppError("file_not_found", params={"path": voice_id}, status=404)
    return d


def save_voice(h: appdirs.ProjectHandle, sid: str, name: str) -> dict[str, Any]:
    """Save the chosen design of a character into the voice library."""
    name = (name or "").strip()
    if not name:
        raise AppError("bad_request", params={"field": "name"}, status=422)
    with session.short_ctx(h) as ctx:
        chars = viewdata.load_characters(ctx)
        c = _char_or_404(chars, sid)
        design_id = (c.performer or {}).get("design_id")
        design = next((d for d in raw_designs(ctx, sid) if d.get("design_id") == design_id), None) if design_id else None
        if design is None:
            raise AppError("no_design_chosen", params={"speaker_id": sid}, action="open_casting", status=400)
        if design.get("kind") == "clone":
            raise AppError("bad_request", params={"field": "design", "reason": "clone"}, status=422)
        wavs = [ctx.paths.abs(s["wav"]) for s in design.get("samples") or [] if s.get("wav") and ctx.paths.abs(s["wav"]).exists()]
        base = appdirs.slugify(name)
        vid = base
        while (appdirs.voices_dir() / f"{vid}.yaml").exists():
            vid = f"{base}-{hashlib.sha1(f'{name}{now_iso()}'.encode()).hexdigest()[:4]}"
        data = {
            "id": vid, "name": name, "gender": c.gender, "age_hint": c.age_hint, "description": design.get("description") or c.description,
            "source_project": h.slug, "created": now_iso(),
            "design": {k: v for k, v in design.items() if k not in ("samples", "chosen")},
            "samples": [{"line_id": s.get("line_id"), "text": s.get("text") or s.get("text_tgt", "")} for s in design.get("samples") or []],
        }
        if wavs:
            shutil.copyfile(wavs[0], appdirs.voices_dir() / f"{vid}.wav")
        (appdirs.voices_dir() / f"{vid}.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return voice_entry(data)


def import_voice(h: appdirs.ProjectHandle, sid: str, voice_id: str) -> dict[str, Any]:
    """``voicebank.import_design`` of a saved voice -> the new design of the character."""
    d = get_voice(voice_id)
    wav = appdirs.voices_dir() / f"{voice_id}.wav"
    from vndub.stages import voicebank

    fn = getattr(voicebank, "import_design", None)
    if fn is None:
        raise AppError("feature_unavailable", params={"feature": "voicebank.import_design"}, action="report_problem", status=501)
    with session.short_ctx(h) as ctx:
        try:
            design = fn(ctx, sid, {**(d.get("design") or {}), "name": d.get("name", ""), "samples": d.get("samples") or []}, [str(wav)] if wav.exists() else [])
        except ValueError as e:
            raise map_value_error(e) from e
        design = dict(design) if not isinstance(design, dict) else design
        chars = viewdata.load_characters(ctx)
        chosen_id = (chars[sid].performer or {}).get("design_id") if sid in chars else None
        return design_view(h, design, chosen_id=chosen_id)

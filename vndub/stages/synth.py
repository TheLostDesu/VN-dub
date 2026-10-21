"""synth: N candidates per voiced line x enabled backend x route.

Plan -> execute, grouped by model (one acquire per backend, all lines under it):

1. TTS for routes A and B per backend (route B writes ``<cand>.prevc.wav``),
2. route C per (backend, LoRA adapter),
3. voice conversion of route-B pre-VC wavs (RVC when the character has a
   trained model, else the zero-shot fallback; no VC available -> the
   candidate fails with that reason, it is never passed through unconverted),
4. the original audio as an ``orig`` candidate for interjections.

Route P (performer voice only, no VC) serves speaker lines without original audio
(``synth.voice_unvoiced``, characters without voice): the same performer TTS as route B, written
straight to the candidate path. Their style comes from the text passport (``source: "text"``, stage
``text_emotion``) or is neutral when there is none.

Per-character tuning (``characters.yaml performer.tuning``) and the per-line ``tuning`` override go through
``text/style_prompt.apply_tuning`` / ``tuning_knobs`` / ``tuning_plan``; the tuning is part of every
candidate's cache key, so changing it regenerates only that character's (or that line's) candidates, and
candidates of an older tuning that are no longer planned are removed.

Every candidate gets ``synth/<line_id>/<backend>_<route>_<k>.wav`` and a sidecar
``.json`` (:class:`SynthSidecar`); failures get a sidecar with ``error`` and
never stop the stage. Regen rounds number candidates ``k = round*100 + i`` so
earlier candidates stay available to the scorer.
"""

from __future__ import annotations

import re
import shutil
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vndub.backends.base import BackendSpec, BackendUnavailable
from vndub.manifest.io import LineRecord, load_overrides, write_json
from vndub.stages import genutil as gu
from vndub.stages.common import StageError
from vndub.text.style_prompt import StyleSpec, apply_tuning, build_style, normalize_tuning, render, to_emotion_ref, tuning_key, tuning_knobs, tuning_plan

NAME = "synth"
VERSION = "1"
CONFIG_SECTIONS = ("synth", "project")


@dataclass
class Job:
    line_id: str
    speaker_id: str
    backend: str  # resolved backend name
    route: str
    k: int
    round: int
    seed: int
    text: str  # text sent to the backend (prefix + text)
    text_plain: str  # CER reference
    text_variant: str
    style: StyleSpec
    rendered: dict[str, Any]
    item: dict[str, Any]
    out: Path
    vc: dict[str, Any] | None = None  # route B: VC plan
    tuning: dict[str, Any] | None = None  # normalized tuning, None when default
    plan: dict[str, Any] | None = None  # tuning_plan(): what master / score still need to know
    options: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    elapsed: float = 0.0
    peak_vram: float | None = None
    cache_key: str = ""

    @property
    def cid(self) -> str:
        return f"{self.line_id}|{self.backend}|{self.route}|{self.k}"

    @property
    def prevc(self) -> Path:
        return self.out.with_name(self.out.stem + ".prevc.wav")


def _is_short(text_tgt: str, src_dur: float) -> bool:
    return gu.count_syllables(text_tgt) <= 2 or (0 < src_dur < 1.0)


def _latest_override(overrides: dict[str, list[Any]], line_id: str, actions: tuple[str, ...]) -> Any | None:
    for o in reversed(overrides.get(line_id, [])):
        if o.action in actions:
            return o
    return None


def _prev_dur_ratio(ctx: Any, line_id: str) -> float | None:
    vals = [s.metrics.get("dur_ratio") for s in gu.load_scores(ctx, line_id) if s.route != "orig"]
    vals = [float(v) for v in vals if v is not None]
    return float(np.median(vals)) if vals else None


def run(ctx: Any) -> dict[str, Any]:
    cfg = ctx.cfg
    sc = cfg.synth
    tgt = cfg.project.tgt_lang
    lines = ctx.dub_lines()
    if not lines:
        raise StageError("no voiced lines to synthesize (run parse/extract/link first)")
    trs = gu.load_translations(ctx)
    if not trs:
        raise StageError(f"{ctx.paths.translation(tgt)} not found: run `vndub translate` first")
    overrides = load_overrides(ctx.paths.overrides)
    regen = bool(ctx.extra.get("regen"))
    if regen:
        sel = gu.load_selection(ctx)
        todo = {lid for lid, s in sel.items() if s.status == "regen"}
        if not todo:
            return {"lines": 0, "note": "no lines with status regen"}
        lines = [l for l in lines if l.line_id in todo]

    def line_round(line_id: str) -> int:
        """0 for a first pass; previous round + 1 in regen mode or when a review
        override (regen / set_text / set_ref) is newer than the line's candidates."""
        sides = [s for _, s in gu.candidate_files(ctx, line_id) if s is not None]
        prev = max([s.round for s in sides], default=-1)
        if regen:
            return max(1, prev + 1)
        newest = max([str((s.model_extra or {}).get("created", "")) for s in sides], default="")
        ov = _latest_override(overrides, line_id, ("regen", "set_text", "set_ref"))
        if ov is not None and sides and ov.ts > newest:
            return prev + 1
        return 0

    chars = ctx.characters()
    summary: dict[str, Any] = {"lines": len(lines), "candidates": 0, "cached": 0, "failed": 0}
    rounds: list[int] = []
    unavailable: dict[str, str] = {}
    skipped_routes: dict[str, str] = {}
    no_text: list[str] = []

    # ---- resolve backends (dedupe on the backend actually used) --------------
    backends: list[tuple[str, BackendSpec, dict[str, Any]]] = []
    requested: dict[str, str] = {}  # resolved name -> configured name (acquire records substitutions)
    seen: set[str] = set()
    for name in cfg.enabled_backends():
        try:
            spec, sub = ctx.models.resolve(name)
        except BackendUnavailable as e:
            unavailable[name] = e.reason
            ctx.log.warning("synth backend %s unavailable: %s", name, e)
            continue
        except KeyError as e:
            unavailable[name] = str(e)
            continue
        if spec.name in seen:
            ctx.log.info("backend %s resolves to %s which is already scheduled; skipped", name, spec.name)
            continue
        seen.add(spec.name)
        backends.append((spec.name, spec, gu.backend_options(cfg, name)))
        requested[spec.name] = name
    if not backends:
        raise StageError(f"no usable TTS backend: {unavailable or 'none enabled in synth.backends'}")

    # ---- per-speaker voicebank info -------------------------------------------
    spk_info: dict[str, dict[str, Any]] = {}

    def info(sid: str) -> dict[str, Any]:
        if sid not in spk_info:
            refs = gu.ref_dicts(ctx, gu.load_refs(ctx, sid))
            perf = gu.load_performer(ctx, sid)
            prefs = []
            if perf:
                for r in perf.get("refs", []):
                    p = ctx.paths.abs(r["path"])
                    if p.exists():
                        prefs.append({**r, "path": str(p)})
            c = chars.get(sid)
            route_override = c.route if c is not None else None
            spk_info[sid] = {
                "tuning": normalize_tuning(((c.performer or {}).get("tuning")) if c is not None else None),
                "refs": refs,
                "performer": perf,
                "performer_refs": prefs,
                "rvc": gu.load_rvc_meta(ctx, sid),
                "lora": gu.load_lora_meta(ctx, sid),
                "stats": gu.char_stats(ctx, chars, sid),
                "gender": c.gender if c is not None else "unknown",
                "route": route_override,
            }
        return spk_info[sid]

    enabled_routes = [r for r, on in sc.routes.items() if on]
    cache = ctx.cache(NAME, VERSION, "synth", "project")
    jobs: list[Job] = []
    orig_jobs: list[tuple[LineRecord, int]] = []
    interjection_lex = _interjections(ctx)

    for line in lines:
        tr = trs.get(line.line_id)
        if not _usable(tr):
            # translate_failed keeps the *source* text in text_tgt: never synthesize it
            no_text.append(line.line_id)
            continue
        sid = line.speaker_id
        si = info(sid)
        rnd = line_round(line.line_id)
        rounds.append(rnd)
        passport = gu.load_passport(ctx, line.line_id) or {}
        src_dur = gu.src_duration(ctx, line, passport)
        set_text = _latest_override(overrides, line.line_id, ("set_text",))
        set_ref = _latest_override(overrides, line.line_id, ("set_ref",))
        regen_ov = _latest_override(overrides, line.line_id, ("regen",))
        tuning_ov = _latest_override(overrides, line.line_id, ("tuning",))
        notes = tr.notes
        base_spec = _base_style(passport, si["stats"], notes, cfg)
        # the line's own tuning replaces the character's (it is a full tuning, not a delta)
        tuning = normalize_tuning(tuning_ov.payload) if tuning_ov is not None else si["tuning"]
        tkey = tuning_key(tuning)
        spec = apply_tuning(base_spec, tuning)
        style_override = ""
        for ov in (regen_ov, set_text):
            if ov is not None and ov.payload.get("style_prompt"):
                style_override = str(ov.payload["style_prompt"])

        # text variant
        interj = "interjection" in tr.flags or _is_interjection(line.text_src, interjection_lex, sc.interjection_max_syllables)
        variant = "interjection" if interj else "full"
        prev_ratio = _prev_dur_ratio(ctx, line.line_id) if rnd > 0 else None
        use_short = prev_ratio is not None and prev_ratio > sc.use_short_text_if_ratio_above and bool(tr.text_tts_short)
        if use_short:
            variant = "short"
        # voice sustain: the head line's voice file also covers its continuation lines
        cont_ids = (line.model_extra or {}).get("sustain_lines") or []
        cont = [trs[c] for c in cont_ids if _usable(trs.get(c))]
        if len(cont) < len(cont_ids):
            ctx.log.warning("%s: %d sustain continuation line(s) without a usable translation", line.line_id, len(cont_ids) - len(cont))

        def texts(plus: bool) -> tuple[str, str]:
            if set_text is not None and set_text.payload.get("text"):
                t = str(set_text.payload["text"])
                return t, gu.strip_stress(t)
            return join_texts([tr, *cont], use_short, plus)

        high = spec.intensity >= sc.high_arousal
        n = sc.n_candidates_short if (interj or _is_short(tr.text_tgt, src_dur) or high) else sc.n_candidates
        if si["route"]:
            routes = [si["route"]]
        elif not line.has_voice and not si["refs"]:
            routes = ["P"]  # no original voice to clone or convert to: the performer speaks as is
        else:
            routes = enabled_routes
        planned: set[Path] = set()
        if interj and sc.keep_original_for_interjections and line.src_wav and not list(ctx.paths.synth_dir(line.line_id).glob("orig_orig_*.wav")):
            orig_jobs.append((line, rnd))

        for route in routes:
            if route in ("B", "P") and not si["performer_refs"]:
                skipped_routes[f"{sid}:{route}"] = "no performer (run voicebank, or choose a design)"
                continue
            if route == "A" and not (si["refs"] or line.src_wav):
                skipped_routes[f"{sid}:A"] = "no original audio to clone"
                continue
            if route == "C" and not si["lora"]:
                skipped_routes[f"{sid}:C"] = "no LoRA adapter (voicebank.route_c)"
                continue
            route_backends = backends
            if route == "C":
                lname = si["lora"].get("backend")
                route_backends = [(b, s, o) for b, s, o in backends if b == lname] or []
                if not route_backends:
                    try:
                        lspec, _ = ctx.models.resolve(lname)
                        route_backends = [(lspec.name, lspec, gu.backend_options(cfg, lname))]
                        requested.setdefault(lspec.name, lname)
                    except (BackendUnavailable, KeyError) as e:
                        skipped_routes[f"{sid}:C"] = f"LoRA backend {lname} unavailable: {e}"
                        continue
            for bname, bspec, bopts in route_backends:
                caps = bspec.caps or {}
                plus = caps.get("stress_format") == "plus"
                text, plain = texts(plus)
                rendered = render(spec, caps, bopts, cross_lingual=(route == "A"))
                plan = tuning_plan(tuning, caps, route) if tkey else None
                if style_override:
                    rendered["instruct"] = style_override if caps.get("instruct", "none") not in ("none",) else rendered["instruct"]
                    rendered["style_prompt"] = style_override
                text_used = (rendered["text_prefix"] + " " + text).strip() if rendered["text_prefix"] else text
                knobs = dict(rendered["knobs"] or {})
                if tkey:
                    knobs.update(tuning_knobs(tuning, caps, knobs))
                if bspec.env is None:  # local backends (mock / sapi) also take generic hints
                    gen = {"tempo": spec.tempo, "loudness": spec.loudness, "intensity": spec.intensity, "arc": spec.arc, "emotion": spec.emotion}
                    for k2, v2 in gen.items():
                        knobs.setdefault(k2, v2)
                timbre_ref = timbre_text = emotion_ref = None
                ref_paths: list[str] = []
                if route == "A":
                    pool = si["refs"]
                    if set_ref is not None and set_ref.payload.get("ref"):
                        timbre_ref = str(ctx.paths.abs(set_ref.payload["ref"]))
                    elif pool:
                        timbre_ref = to_emotion_ref(spec, pool, usage="timbre")
                    else:
                        timbre_ref = str(ctx.paths.abs(line.src_wav)) if line.src_wav else None
                    timbre_text = next((r.get("text") for r in pool if r["path"] == timbre_ref), None) or (line.text_src if timbre_ref and line.src_wav and timbre_ref == str(ctx.paths.abs(line.src_wav)) else None)
                    emotion_ref = to_emotion_ref(spec, pool) if pool else timbre_ref
                    if bspec.env is None:
                        knobs.setdefault("f0_hz", si["stats"].get("f0_median_hz"))
                        knobs.setdefault("gender", si["gender"])
                elif route in ("B", "P"):
                    prefs = si["performer_refs"]
                    timbre_ref = to_emotion_ref(spec, prefs) if prefs else None
                    timbre_text = next((r.get("text") for r in prefs if r["path"] == timbre_ref), None)
                    emotion_ref = timbre_ref
                    if bspec.env is None:
                        knobs.setdefault("f0_hz", (si["performer"] or {}).get("f0_median_hz"))
                        knobs.setdefault("gender", si["gender"])
                ref_paths = [p for p in dict.fromkeys([timbre_ref, emotion_ref]) if p]
                options = dict(bopts)
                if route == "C":
                    options["lora"] = str(ctx.paths.abs(si["lora"]["adapter"]))
                target_duration = round(src_dur * (plan["target_duration_scale"] if plan else 1.0), 3) if caps.get("duration") and src_dur > 0 else None
                vc0 = _vc_plan(ctx, si) if route == "B" else None
                if vc0 is not None and plan and plan["vc_f0_transpose"] and vc0.get("backend"):
                    vc0["f0_transpose"] = int(vc0.get("f0_transpose", 0)) + int(plan["vc_f0_transpose"])
                for i in range(n):
                    k = rnd * 100 + i
                    seed = sc.seed_base + gu.stable_seed(line.line_id, bname, route, k, rnd) % 1_000_000
                    out = ctx.paths.candidate(line.line_id, bname, route, k)
                    item = {
                        "id": f"{line.line_id}|{bname}|{route}|{k}",
                        "text": text_used,
                        "lang": tgt,
                        "timbre_ref": timbre_ref,
                        "timbre_ref_text": timbre_text,
                        "emotion_ref": emotion_ref,
                        "instruct": rendered["instruct"],
                        "knobs": knobs,
                        "seed": seed,
                        "target_duration": target_duration,
                        "out": str(out.with_name(out.stem + ".prevc.wav") if route == "B" else out),
                    }
                    vc = dict(vc0) if vc0 is not None else None
                    job = Job(
                        line_id=line.line_id, speaker_id=sid, backend=bname, route=route, k=k, round=rnd, seed=seed,
                        text=text_used, text_plain=plain, text_variant=variant if set_text is None else "override",
                        style=spec, rendered=rendered, item=item, out=out, vc=vc, options=options, tuning=tkey, plan=plan,
                    )
                    parts: list[Any] = [
                        text_used, spec.to_dict(), rendered, [Path(p).name for p in ref_paths], options, seed, tgt,
                        item["target_duration"], vc,
                    ]
                    if tkey:  # default tuning keeps the pre-tuning key
                        parts.append({"tuning": tkey, "plan": plan})
                    job.cache_key = cache.key(*parts, files=ref_paths)
                    planned.add(out)
                    jobs.append(job)
        _purge_stale(ctx, line.line_id, tkey, planned)

    # ---- execute ------------------------------------------------------------
    todo = [j for j in jobs if not cache.fresh(j.cid, j.cache_key, [j.out])]
    summary["cached"] = len(jobs) - len(todo)
    progress = _Progress(ctx, len(jobs) + sum(1 for j in todo if j.route == "B"), len(jobs) - len(todo))
    groups: dict[tuple[str, str], list[Job]] = {}
    for j in todo:
        groups.setdefault((j.backend, j.options.get("lora", "")), []).append(j)
    for (bname, lora), js in groups.items():
        try:
            backend = ctx.models.acquire(requested.get(bname, bname), js[0].options or None)
        except BackendUnavailable as e:
            for j in js:
                j.error = f"backend unavailable: {e}"
                if j.route != "B":
                    _finish(ctx, cache, j)
            unavailable[bname] = str(e)
            continue
        for i in range(0, len(js), 16):
            part = js[i : i + 16]
            t0 = time.perf_counter()
            res = gu.run_batched(backend, "synthesize", [j.item for j in part], chunk=16)
            el = (time.perf_counter() - t0) / max(1, len(part))
            peak = None
            try:
                peak = backend.peak_vram()
            except Exception:
                pass
            by = {r.get("id"): r for r in res}
            for j in part:
                r = by.get(j.item["id"], {})
                target = Path(j.item["out"])
                if r.get("error"):
                    j.error = str(r["error"])
                elif not target.exists():
                    j.error = "backend returned no audio"
                j.elapsed, j.peak_vram = el, peak
                if j.route != "B":
                    _finish(ctx, cache, j)
                progress(j.line_id)

    # ---- VC for route B -------------------------------------------------------
    vc_groups: dict[str, list[Job]] = {}
    for j in todo:
        if j.route != "B" or j.error:
            if j.route == "B" and j.error:
                _finish(ctx, cache, j)
            continue
        if not j.vc or not j.vc.get("backend"):
            j.error = (j.vc or {}).get("reason") or "no voice conversion available"
            _finish(ctx, cache, j)
            continue
        vc_groups.setdefault(j.vc.get("request") or j.vc["backend"], []).append(j)
    for vname, js in vc_groups.items():
        try:
            vb = ctx.models.acquire(vname)
        except BackendUnavailable as e:
            for j in js:
                j.error = f"VC backend unavailable: {e}"
                _finish(ctx, cache, j)
            unavailable[vname] = str(e)
            continue
        for i in range(0, len(js), 16):  # chunks of 16: progress between them
            part = js[i : i + 16]
            items = []
            for j in part:
                v = j.vc
                items.append({
                    "id": j.cid, "src": str(j.prevc), "out": str(j.out), "target_ref": v.get("target_ref"),
                    "model": v.get("model"), "index": v.get("index"), "f0_transpose": v.get("f0_transpose", 0), "params": v.get("params", {}),
                })
            t0 = time.perf_counter()
            res = {r.get("id"): r for r in gu.run_batched(vb, "convert", items, chunk=16)}
            el = (time.perf_counter() - t0) / max(1, len(items))
            for j in part:
                r = res.get(j.cid, {})
                if r.get("error"):
                    j.error = f"vc: {r['error']}"
                elif not j.out.exists():
                    j.error = "vc returned no audio"
                j.elapsed += el
                _finish(ctx, cache, j)
                progress(j.line_id)

    # ---- original audio candidates (interjections) ------------------------------
    for line, r in orig_jobs:
        out = ctx.paths.candidate(line.line_id, "orig", "orig", r * 100)
        if not out.exists():
            src = ctx.paths.abs(line.src_wav)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
            _write_sidecar(ctx, out, {
                "line_id": line.line_id, "backend": "orig", "route": "orig", "k": r * 100, "seed": None,
                "text_used": line.text_src, "text_variant": "orig", "text_plain": "", "round": r,
                "ref_paths": [ctx.paths.rel(src)], "style_prompt": "", "created": gu.ts_now(),
            })
        summary.setdefault("orig_candidates", 0)
        summary["orig_candidates"] += 1

    failed = [j for j in todo if j.error]
    summary["candidates"] = len(jobs) - len(failed)
    summary["generated"] = len(todo) - len(failed)
    summary["failed"] = len(failed)
    if failed:
        errs: dict[str, int] = {}
        for j in failed:
            key = f"{j.backend}/{j.route}: {j.error[:80]}"
            errs[key] = errs.get(key, 0) + 1
        summary["errors"] = errs
    cache.save()
    summary["backends"] = [b for b, _, _ in backends]
    summary["round"] = max(rounds, default=0)
    if unavailable:
        summary["unavailable"] = unavailable
    if skipped_routes:
        summary["skipped_routes"] = skipped_routes
    if no_text:
        summary["no_translation"] = len(no_text)
        ctx.log.warning("%d line(s) without a usable translation were skipped: %s", len(no_text), no_text[:20])
    return summary


class _Progress:
    """``ctx.progress`` over the planned candidates (a route-B candidate counts twice: TTS, then VC);
    candidates that are already cached count as done from the start (a fully cached run reports n / n)."""

    def __init__(self, ctx: Any, total: int, done: int = 0) -> None:
        self.ctx, self.total, self.done = ctx, total, min(done, total)
        if total:
            ctx.progress(self.done, total, None)

    def __call__(self, item: str | None = None) -> None:
        self.done = min(self.total, self.done + 1)
        self.ctx.progress(self.done, self.total, item)


def _base_style(passport: dict[str, Any], stats: dict[str, Any], notes: str, cfg: Any) -> StyleSpec:
    """The line's StyleSpec before tuning: built from an audio passport; taken as is from a text passport
    (``source: "text"``, written by ``text_emotion``); neutral when there is no passport."""
    if passport.get("source") == "text":
        st = passport.get("style")
        spec = StyleSpec.from_dict(st) if isinstance(st, dict) and st else StyleSpec()
        joined = "; ".join(x for x in (spec.notes, notes) if x)
        spec.notes = joined
        return spec
    return build_style(passport, stats, notes, cfg.analyze.e2v_weight, cfg.analyze.anime_weight, cfg.analyze.caption)


def _purge_stale(ctx: Any, line_id: str, tkey: dict[str, Any] | None, planned: set[Path]) -> None:
    """Delete candidates rendered under another tuning that this run does not regenerate in place
    (e.g. extra regen-round candidates): they would compete with the tuned ones in score."""
    for wav, side in gu.candidate_files(ctx, line_id):
        if side is None or side.route == "orig" or wav in planned:
            continue
        old = ((side.params or {}).get("tuning") or {}).get("values")
        if old == tkey:
            continue
        for p in (wav, wav.with_suffix(".json"), wav.with_name(wav.stem + ".prevc.wav")):
            p.unlink(missing_ok=True)


def _usable(tr: Any) -> bool:
    """A translation synth may speak: present, not failed/missing, non-empty text_tts."""
    return tr is not None and not ({"translate_failed", "translate_missing"} & set(tr.flags)) and bool((tr.text_tts or "").strip())


def join_texts(recs: list[Any], short: bool, plus: bool) -> tuple[str, str]:
    """(TTS text, CER reference) for a head line + its sustain continuations, in line order.

    short -> text_tts_short / text_plain_short; plus -> the '+' stress variants when present.
    """
    parts, plains = [], []
    for rec in recs:
        ex = rec.model_extra or {}
        if short:
            body = (ex.get("text_tts_short_plus") if plus else None) or rec.text_tts_short or rec.text_tts
            plain = ex.get("text_plain_short") or gu.strip_stress(rec.text_tgt_short or body)
        else:
            body = (ex.get("text_tts_plus") if plus else None) or rec.text_tts
            plain = rec.text_plain or gu.strip_stress(rec.text_tgt)
        parts.append(body.strip())
        plains.append(plain.strip())
    return " ".join(p for p in parts if p), " ".join(p for p in plains if p)


def _vc_plan(ctx: Any, si: dict[str, Any]) -> dict[str, Any]:
    vc = ctx.cfg.synth.vc
    perf_f0 = (si["performer"] or {}).get("f0_median_hz")
    char_f0 = si["stats"].get("f0_median_hz")
    f0_t = int(round(12 * np.log2(char_f0 / perf_f0))) if char_f0 and perf_f0 else 0
    reasons = []
    if si["rvc"] and vc.primary:
        try:
            spec, _ = ctx.models.resolve(vc.primary)
            meta = si["rvc"]
            return {
                "backend": spec.name, "request": vc.primary, "model": str(ctx.paths.abs(meta["model"])),
                "index": str(ctx.paths.abs(meta["index"])) if meta.get("index") else None,
                "f0_transpose": int(meta.get("f0_transpose", f0_t)), "params": dict(vc.rvc), "target_ref": None,
            }
        except (BackendUnavailable, KeyError) as e:
            reasons.append(f"{vc.primary}: {e}")
    elif not si["rvc"]:
        reasons.append("no RVC model for this character")
    if vc.fallback:
        try:
            spec, _ = ctx.models.resolve(vc.fallback)
            timbre = [r for r in si["refs"] if r.get("usage") in ("timbre", "both")] or si["refs"]
            target = max(timbre, key=lambda r: float(r.get("snr") or 0))["path"] if timbre else None
            if target is None:
                reasons.append("no character reference clip for zero-shot VC")
            else:
                return {"backend": spec.name, "request": vc.fallback, "target_ref": target, "f0_transpose": f0_t, "params": dict(vc.seedvc), "model": None, "index": None}
        except (BackendUnavailable, KeyError) as e:
            reasons.append(f"{vc.fallback}: {e}")
    return {"backend": None, "reason": "no voice conversion available (" + "; ".join(reasons) + ")"}


def _interjections(ctx: Any) -> dict[str, str]:
    try:
        from vndub.text.ja import load_interjections

        return load_interjections(ctx.cfg.resolve(ctx.cfg.translate.interjections))
    except Exception:
        return {}


_JA_PUNCT = re.compile(r"[\s、。，．・！？!?…‥ー〜~「」『』（）()\[\]【】]")


def _is_interjection(text: str, lex: dict[str, str], max_morae: int) -> bool:
    """fork C's text.ja.is_interjection when present, else lexicon / length heuristic."""
    try:
        from vndub.text.ja import is_interjection

        return bool(is_interjection(text, lex, max_morae))
    except Exception:
        pass
    norm = _JA_PUNCT.sub("", unicodedata.normalize("NFKC", text or ""))
    return bool(norm) and (norm in lex or len(norm) <= max_morae)


def _write_sidecar(ctx: Any, wav: Path, data: dict[str, Any]) -> None:
    data = dict(data)
    data["wav"] = ctx.paths.rel(wav)
    write_json(wav.with_suffix(".json"), data)


def _finish(ctx: Any, cache: Any, j: Job) -> None:
    data = {
        "line_id": j.line_id, "backend": j.backend, "route": j.route, "k": j.k, "seed": j.seed,
        "params": {
            "options": j.options, "knobs": j.item.get("knobs"), "instruct": j.item.get("instruct"), "target_duration": j.item.get("target_duration"), "vc": j.vc,
            **({"tuning": {"values": j.tuning, "plan": j.plan}} if j.tuning else {}),
        },
        "style_prompt": j.rendered.get("style_prompt", ""), "style": j.style.to_dict(),
        "ref_paths": [ctx.paths.rel(p) for p in [j.item.get("timbre_ref"), j.item.get("emotion_ref")] if p],
        "text_used": j.text, "text_plain": j.text_plain, "text_variant": j.text_variant,
        "elapsed_s": round(j.elapsed, 3), "peak_vram_gb": j.peak_vram, "round": j.round, "created": gu.ts_now(),
    }
    if j.route == "B" and j.prevc.exists():
        data["pre_vc_wav"] = ctx.paths.rel(j.prevc)
    if j.error:
        data["error"] = j.error
        if j.out.exists():
            j.out.unlink()
        write_json(j.out.with_suffix(".json"), {**data, "wav": ""})
        return
    _write_sidecar(ctx, j.out, data)
    cache.put(j.cid, j.cache_key)

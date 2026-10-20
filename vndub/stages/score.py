"""score: metrics for every candidate, gates, composite, selection.

Metrics (grouped by model, one acquire each, all candidates under it):

==========  ==========================================================  ===============
metric      source                                                      composite term
==========  ==========================================================  ===============
utmos       MOS predictor (UTMOSv2, fallback UTMOS22)                   (utmos-1)/4
pq          Audiobox Aesthetics PQ                                      pq/10
spk_sim     speaker embedder: max cosine to the character's timbre refs clip(0..1)
emo_sim     emotion2vec+ embedding cosine to the original line          0.75*emo_sim + 0.25*(1-min(1,adv_delta))
adv_delta   |dA| + |dV| (A/D/V model) original vs candidate             (inside emo term)
cer         ASR of the candidate vs the stress-free target text         1-cer
dur_ratio   candidate / source duration                                 1-min(1,|ln r|/ln 2)
prosody     mean of (f0_corr+1)/2, (energy_corr+1)/2, pause_match       as is
==========  ==========================================================  ===============

The composite is the weighted mean over the terms that are available (weights
renormalized); missing terms are listed per candidate. Every raw metric is
persisted in ``scores/<line_id>.jsonl`` so ``vndub score --reweight`` re-derives
composites and the selection without touching any model.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from vndub.audio import features, vad
from vndub.backends.score import prosody as pros
from vndub.backends.score.cer import cer as cer_fn
from vndub.backends.score.speaker_sim import cosine
from vndub.manifest.io import CandidateScore, LineRecord, SelectionRecord, SynthSidecar, load_overrides, read_json, write_jsonl
from vndub.stages import genutil as gu
from vndub.stages.common import StageError
from vndub.util import file_fingerprint, hash_obj

NAME = "score"
VERSION = "1"
CONFIG_SECTIONS = ("score",)

METRIC_GROUPS = {
    "mos": ["utmos"],
    "aesthetics": ["pq", "ce", "cu", "pc"],
    "speaker": ["spk_sim"],
    "emotion": ["emo_sim"],
    "adv": ["adv_delta"],
    "cer": ["cer"],
    "prosody": ["f0_corr", "energy_corr", "pause_match", "prosody"],
}


# ----------------------------------------------------------------------------
# pure scoring functions (unit-tested)
# ----------------------------------------------------------------------------


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(min(hi, max(lo, x)))


def terms(m: dict[str, Any]) -> dict[str, float | None]:
    """Normalized [0, 1] composite terms from raw metrics (None = unavailable)."""
    t: dict[str, float | None] = {k: None for k in ("utmos", "pq", "spk_sim", "emo_sim", "cer", "dur", "prosody")}
    if m.get("utmos") is not None:
        t["utmos"] = _clip((float(m["utmos"]) - 1.0) / 4.0)
    if m.get("pq") is not None:
        t["pq"] = _clip(float(m["pq"]) / 10.0)
    if m.get("spk_sim") is not None:
        t["spk_sim"] = _clip(float(m["spk_sim"]))
    emo, adv = m.get("emo_sim"), m.get("adv_delta")
    if emo is not None and adv is not None:
        t["emo_sim"] = 0.75 * _clip(float(emo)) + 0.25 * (1.0 - _clip(float(adv)))
    elif emo is not None:
        t["emo_sim"] = _clip(float(emo))
    elif adv is not None:
        t["emo_sim"] = 1.0 - _clip(float(adv))
    if m.get("cer") is not None:
        t["cer"] = 1.0 - _clip(float(m["cer"]))
    r = m.get("dur_ratio")
    if r is not None and float(r) > 0:
        t["dur"] = 1.0 - min(1.0, abs(math.log(float(r))) / math.log(2.0))
    if m.get("prosody") is not None:
        t["prosody"] = _clip(float(m["prosody"]))
    return t


def composite(m: dict[str, Any], weights: dict[str, float]) -> tuple[float | None, list[str]]:
    t = terms(m)
    num = den = 0.0
    missing = []
    for k, w in weights.items():
        if w <= 0:
            continue
        v = t.get(k)
        if v is None:
            missing.append(k)
            continue
        num += w * v
        den += w
    return (round(num / den, 4) if den > 0 else None), missing


def evaluate_gates(m: dict[str, Any], gates: Any, spk_threshold: float, cer_max: float | None = None, route: str = "") -> dict[str, bool]:
    """Gate -> passed. Gates whose metric is unavailable are not evaluated (absent)."""
    out: dict[str, bool] = {}
    if m.get("utmos") is not None:
        out["utmos"] = float(m["utmos"]) >= gates.utmos_min
    if route == "orig":
        out["cer"] = True  # original audio is the source language by definition
        out["dur"] = True
    else:
        if m.get("cer") is not None:
            out["cer"] = float(m["cer"]) <= (gates.cer_max if cer_max is None else cer_max)
        if m.get("dur_ratio") is not None:
            lo, hi = gates.dur_ratio
            out["dur"] = lo <= float(m["dur_ratio"]) <= hi
    if m.get("spk_sim") is not None:
        out["spk_sim"] = float(m["spk_sim"]) >= spk_threshold
    return out


def describe_failures(m: dict[str, Any], g: dict[str, bool], gates: Any, spk_threshold: float, cer_max: float) -> str:
    parts = []
    for k, ok in g.items():
        if ok:
            continue
        if k == "utmos":
            parts.append(f"utmos {m['utmos']:.2f}<{gates.utmos_min}")
        elif k == "cer":
            parts.append(f"cer {m['cer']:.2f}>{cer_max}")
        elif k == "dur":
            parts.append(f"dur_ratio {m['dur_ratio']:.2f} outside {list(gates.dur_ratio)}")
        elif k == "spk_sim":
            parts.append(f"spk_sim {m['spk_sim']:.2f}<{spk_threshold:.2f}")
    return ", ".join(parts)


def translation_problem(tr: Any) -> str:
    """'' when the line has a speakable translation, else the review reason."""
    if tr is None or "translate_missing" in tr.flags:
        return "translation missing"
    if "translate_failed" in tr.flags or not (tr.text_tts or "").strip():
        return "translation failed"
    return ""


def cer_reference(side: SynthSidecar | None, tr: Any, cont: list[Any]) -> str:
    """CER reference: the sidecar's text_plain (what synth actually used), else rebuilt
    from the translation (text_plain_short for the short variant; sustain continuations joined)."""
    ref = (side.model_extra or {}).get("text_plain") if side is not None else None
    if ref:
        return str(ref)
    short = side is not None and side.text_variant == "short"
    parts = []
    for rec in [tr, *cont]:
        if rec is None:
            continue
        ex = rec.model_extra or {}
        p = (ex.get("text_plain_short") if short else None) or rec.text_plain or gu.strip_stress(rec.text_tgt)
        parts.append(p.strip())
    return " ".join(p for p in parts if p)


def spk_threshold(cfg: Any, calibration: dict[str, Any] | None) -> tuple[float, str]:
    g = cfg.score.gates
    if isinstance(g.spk_sim_min, (int, float)):
        return float(g.spk_sim_min), "configured"
    if calibration and calibration.get("threshold") is not None:
        return float(calibration["threshold"]), f"calibrated ({calibration.get('embedder')}, EER {calibration.get('eer')})"
    return float(g.spk_sim_default), "default (no calibration)"


def apply_scores(c: CandidateScore, cfg: Any, thr: float, interjection: bool) -> CandidateScore:
    comp, missing = composite(c.metrics, cfg.score.weights)
    c.composite = comp
    gates = cfg.score.gates
    cer_max = cfg.score.interjection_cer_max if interjection else gates.cer_max
    g = evaluate_gates(c.metrics, gates, thr, cer_max, c.route)
    c.gates = g
    c.passed = bool(g) and all(g.values()) and comp is not None
    setattr(c, "missing", sorted(set(missing)))
    setattr(c, "failures", describe_failures(c.metrics, g, gates, thr, cer_max))
    return c


def select(
    line_id: str,
    cands: list[CandidateScore],
    sidecars: dict[str, SynthSidecar | None],
    overrides: list[Any],
    regen_rounds: int,
    unavailable: dict[str, str] | None = None,
    src_wav: str | None = None,
    n_failed: int = 0,
) -> SelectionRecord:
    rejected = {str(o.payload.get("candidate") or o.payload.get("wav")) for o in overrides if o.action == "reject"}
    pool = [c for c in cands if c.wav not in rejected]
    created = [s.model_extra.get("created", "") for s in sidecars.values() if s is not None and s.model_extra]
    newest = max(created) if created else ""
    rnd = max([s.round for s in sidecars.values() if s is not None] or [0])
    decisive = [o for o in overrides if o.action in ("accept", "keep_original", "regen", "set_text", "set_ref")]
    unavail_note = ("; metrics unavailable: " + ", ".join(f"{k} ({v[:80]})" for k, v in unavailable.items())) if unavailable else ""

    def rec(winner: CandidateScore | None, status: str, reason: str) -> SelectionRecord:
        scores = dict(winner.metrics) if winner else {}
        if winner is not None:
            scores["composite"] = winner.composite
        return SelectionRecord(
            line_id=line_id, winner=winner.wav if winner else None, backend=winner.backend if winner else None,
            route=winner.route if winner else None, scores=scores, status=status, reason=reason, round=rnd, n_candidates=len(pool),
        )

    if decisive:
        last = decisive[-1]
        if last.action == "accept":
            want = str(last.payload.get("candidate") or last.payload.get("wav") or "")
            hit = next((c for c in pool if c.wav == want), None)
            if hit is not None:
                return rec(hit, "override", f"accepted in review ({last.ts})")
        elif last.action == "keep_original":
            orig = next((c for c in cands if c.route == "orig"), None)
            if orig is None and src_wav:
                orig = CandidateScore(line_id=line_id, wav=src_wav, backend="orig", route="orig", metrics={}, composite=None)
            if orig is not None:
                return rec(orig, "override", "keep original (review)")
        elif last.ts and last.ts > newest:
            best = max(pool, key=lambda c: c.composite if c.composite is not None else -1, default=None)
            return rec(best, "regen", f"override {last.action} pending")
    if not pool:
        return rec(None, "review", f"no candidates{f' ({n_failed} failed generations)' if n_failed else ''}{unavail_note}")
    passing = [c for c in pool if c.passed]
    if passing:
        best = max(passing, key=lambda c: c.composite or 0.0)
        return rec(best, "auto", f"best of {len(passing)} passing{unavail_note}")
    best = max(pool, key=lambda c: c.composite if c.composite is not None else -1.0)
    why = (best.model_extra or {}).get("failures") or "no gate could be evaluated"
    if best.metrics.get("dur_ratio") is not None and float(best.metrics["dur_ratio"]) > 1.25:
        why += "; try text_tgt_short"
    if rnd < regen_rounds:
        return rec(best, "regen", f"no candidate passes the gates (best fails: {why}){unavail_note}")
    return rec(best, "review", f"no candidate passes the gates after {rnd} regen round(s) (best fails: {why}){unavail_note}")


# ----------------------------------------------------------------------------
# metric computation
# ----------------------------------------------------------------------------


class Cand:
    def __init__(self, ctx: Any, line: LineRecord, wav: Path, side: SynthSidecar | None, passport: dict[str, Any], tr: Any, record: CandidateScore) -> None:
        self.line, self.wav, self.side, self.passport, self.tr, self.rec = line, wav, side, passport, tr, record
        self.cid = f"{line.line_id}|{wav.stem}"
        self.emb_dir = ctx.paths.scores_dir / "emb" / line.line_id
        self.feat_dir = ctx.paths.scores_dir / "feat" / line.line_id

    def needs(self, metric: str) -> bool:
        return self.rec.metrics.get(metric) is None and metric not in self.na

    @property
    def na(self) -> list[str]:
        return list((self.rec.model_extra or {}).get("na") or [])


def _metric_key(cfg: Any) -> str:
    s = cfg.score
    return hash_obj([VERSION, s.mos, s.mos_fallback, s.aesthetics, s.speaker, s.emotion, s.adv, s.cer_asr, s.cer_asr_fallback, cfg.analyze.f0])


def run(ctx: Any) -> dict[str, Any]:
    cfg = ctx.cfg
    lines = ctx.voiced_lines()
    if not lines:
        raise StageError("no voiced lines")
    trs = gu.load_translations(ctx)
    overrides = load_overrides(ctx.paths.overrides)
    calib = read_json(ctx.paths.spk_calibration)
    thr, thr_src = spk_threshold(cfg, calib)
    reweight = bool(ctx.extra.get("reweight"))
    mkey = _metric_key(cfg)
    unavailable: dict[str, str] = {}
    if cfg.score.emotion != cfg.analyze.e2v:
        ctx.log.warning("score.emotion (%s) differs from analyze.e2v (%s): emotion similarity compares different embedding spaces", cfg.score.emotion, cfg.analyze.e2v)

    cands: list[Cand] = []
    per_line: dict[str, list[Cand]] = {}
    sidecars: dict[str, dict[str, SynthSidecar | None]] = {}
    failed_counts: dict[str, int] = {}
    for line in lines:
        old = {c.wav: c for c in gu.load_scores(ctx, line.line_id)}
        passport = gu.load_passport(ctx, line.line_id) or {}
        tr = trs.get(line.line_id)
        lst = []
        sidecars[line.line_id] = {}
        n_failed = 0
        for side_path in sorted(ctx.paths.synth_dir(line.line_id).glob("*.json")) if ctx.paths.synth_dir(line.line_id).exists() else []:
            d = read_json(side_path) or {}
            if d.get("error"):
                n_failed += 1
        failed_counts[line.line_id] = n_failed
        for wav, side in gu.candidate_files(ctx, line.line_id):
            if side is None or (side.model_extra or {}).get("error"):
                continue
            rel = ctx.paths.rel(wav)
            fp = file_fingerprint(wav)
            prev = old.get(rel)
            if prev is not None and (reweight or ((prev.model_extra or {}).get("fp") == fp and (prev.model_extra or {}).get("mkey") == mkey)):
                record = prev
            else:
                record = CandidateScore(line_id=line.line_id, wav=rel, backend=side.backend, route=side.route, metrics={})
                setattr(record, "na", ["cer"] if side.route == "orig" else [])
            setattr(record, "fp", fp)
            setattr(record, "mkey", mkey)
            setattr(record, "round", side.round)
            c = Cand(ctx, line, wav, side, passport, tr, record)
            lst.append(c)
            sidecars[line.line_id][rel] = side
        per_line[line.line_id] = lst
        cands += lst

    if not reweight and cands:
        _compute(ctx, cands, unavailable)

    # ---- composite, gates, selection ------------------------------------------
    selections: list[SelectionRecord] = []
    for line in lines:
        lst = per_line.get(line.line_id, [])
        tr = trs.get(line.line_id)
        interj = bool(tr and "interjection" in tr.flags)
        for c in lst:
            apply_scores(c.rec, cfg, thr, interj)
        write_jsonl(ctx.paths.scores(line.line_id), [c.rec for c in lst])
        tr_problem = translation_problem(tr)
        if tr_problem and not any(c.rec.route == "orig" for c in lst):
            selections.append(SelectionRecord(line_id=line.line_id, status="review", reason=tr_problem, n_candidates=len(lst)))
            continue
        sel = select(
            line.line_id, [c.rec for c in lst], sidecars.get(line.line_id, {}), overrides.get(line.line_id, []),
            cfg.score.regen_rounds, unavailable, line.src_wav, failed_counts.get(line.line_id, 0),
        )
        selections.append(sel)

    # keep selections of lines outside --only/--limit
    if ctx.only or ctx.limit is not None:
        prev = gu.load_selection(ctx)
        mine = {s.line_id for s in selections}
        selections = [*selections, *[s for lid, s in prev.items() if lid not in mine]]
    write_jsonl(ctx.paths.selection, selections)

    status: dict[str, int] = {}
    for s in selections:
        status[s.status] = status.get(s.status, 0) + 1
    per_br: dict[str, list[float]] = {}
    for c in cands:
        if c.rec.composite is not None:
            per_br.setdefault(f"{c.rec.backend}/{c.rec.route}", []).append(c.rec.composite)
    summary: dict[str, Any] = {
        "lines": len(lines),
        "candidates": len(cands),
        "status": status,
        "mean_composite": {k: round(float(np.mean(v)), 3) for k, v in sorted(per_br.items())},
        "spk_threshold": f"{thr:.3f} {thr_src}",
        "reweight": reweight,
    }
    if unavailable:
        summary["metrics_unavailable"] = unavailable
    return summary


def _acquire(ctx: Any, names: list[str], group: str, unavailable: dict[str, str]):
    b, used, reason = gu.acquire_first(ctx, names)
    if b is None:
        unavailable[group] = reason
    return b, used


def _compute(ctx: Any, cands: list[Cand], unavailable: dict[str, str]) -> None:
    cfg = ctx.cfg
    s = cfg.score
    tgt = cfg.project.tgt_lang

    # MOS
    todo = [c for c in cands if c.needs("utmos")]
    if todo:
        b, _ = _acquire(ctx, [s.mos, s.mos_fallback], "mos", unavailable)
        if b is not None:
            by = {c.cid: c for c in todo}
            for r in gu.run_batched(b, "mos", [{"id": c.cid, "wav": str(c.wav)} for c in todo]):
                v = r.get("mos", r.get("utmos", r.get("score")))
                if v is not None and not r.get("error") and r.get("id") in by:
                    by[r["id"]].rec.metrics["utmos"] = round(float(v), 4)
    # aesthetics
    if s.aesthetics:
        todo = [c for c in cands if c.needs("pq")]
        if todo:
            b, _ = _acquire(ctx, [s.aesthetics], "aesthetics", unavailable)
            if b is not None:
                by = {c.cid: c for c in todo}
                for r in gu.run_batched(b, "aesthetics", [{"id": c.cid, "wav": str(c.wav)} for c in todo]):
                    if not r.get("error") and r.get("id") in by:
                        for k in ("pq", "ce", "cu", "pc"):
                            if r.get(k) is not None:
                                by[r["id"]].rec.metrics[k] = round(float(r[k]), 4)
    else:
        for c in cands:
            if "pq" not in c.na:
                c.rec.na = [*c.na, "pq"]  # type: ignore[attr-defined]
    # speaker similarity
    todo = [c for c in cands if c.needs("spk_sim")]
    if todo:
        b, name = _acquire(ctx, [s.speaker], "speaker", unavailable)
        if b is not None:
            jobs = []
            refs_for: dict[str, list[str]] = {}
            for c in todo:
                sid = c.line.speaker_id
                if sid not in refs_for:
                    refs = [r for r in gu.load_refs(ctx, sid) if r.usage in ("timbre", "both")] or gu.load_refs(ctx, sid)
                    keys = []
                    for r in refs:
                        k = f"ref|{sid}|{r.path}"
                        jobs.append((k, ctx.paths.abs(r.path), ctx.paths.vb(sid) / "emb" / f"ref_{Path(r.path).stem}.{name}.npy"))
                        keys.append(k)
                    refs_for[sid] = keys
                if not refs_for[sid] and (c.line.src_wav16 or c.line.src_wav):
                    k = f"orig|{c.line.line_id}"
                    jobs.append((k, ctx.paths.abs(c.line.src_wav16 or c.line.src_wav), c.emb_dir / f"orig.{name}.npy"))
                jobs.append((f"cand|{c.cid}", c.wav, c.emb_dir / f"{c.wav.stem}.{name}.npy"))
            vecs = gu.embed_paths(ctx, b, list(dict.fromkeys(jobs)))
            for c in todo:
                v = vecs.get(f"cand|{c.cid}")
                if v is None:
                    continue
                keys = refs_for.get(c.line.speaker_id) or [f"orig|{c.line.line_id}"]
                refs = [vecs[k] for k in keys if k in vecs]
                if refs:
                    c.rec.metrics["spk_sim"] = round(max(cosine(v, r) for r in refs), 4)
                    if not refs_for.get(c.line.speaker_id):
                        c.rec.notes = [*c.rec.notes, "spk_sim vs the original line (no voicebank refs)"]
        if s.speaker_extra:
            b2, name2 = _acquire(ctx, [s.speaker_extra], "speaker_extra", unavailable)
            if b2 is not None:
                jobs = []
                for c in todo:
                    if c.line.src_wav16 or c.line.src_wav:
                        jobs.append((f"orig|{c.line.line_id}", ctx.paths.abs(c.line.src_wav16 or c.line.src_wav), c.emb_dir / f"orig.{name2}.npy"))
                    jobs.append((f"cand|{c.cid}", c.wav, c.emb_dir / f"{c.wav.stem}.{name2}.npy"))
                vecs = gu.embed_paths(ctx, b2, list(dict.fromkeys(jobs)))
                for c in todo:
                    a, o = vecs.get(f"cand|{c.cid}"), vecs.get(f"orig|{c.line.line_id}")
                    if a is not None and o is not None:
                        c.rec.metrics["spk_sim_extra"] = round(cosine(a, o), 4)
    # emotion similarity (same emotion2vec+ model as analyze)
    todo = [c for c in cands if c.needs("emo_sim")]
    if todo:
        b, name = _acquire(ctx, [s.emotion], "emotion", unavailable)
        if b is not None:
            items = []
            orig_out: dict[str, Path] = {}
            for c in todo:
                lid = c.line.line_id
                emb = ((c.passport.get("emotion") or {}).get("e2v") or {}).get("embedding_path")
                same_model = (c.passport.get("backends") or {}).get("e2v") in (None, name)
                if emb and same_model and ctx.paths.abs(emb).exists():
                    orig_out[lid] = ctx.paths.abs(emb)
                elif lid not in orig_out and (c.line.src_wav16 or c.line.src_wav):
                    out = c.emb_dir / f"orig.{name}.e2v.npy"
                    orig_out[lid] = out
                    if not out.exists():
                        items.append({"id": f"orig|{lid}", "wav": str(ctx.paths.abs(c.line.src_wav16 or c.line.src_wav)), "out": str(out)})
                items.append({"id": c.cid, "wav": str(c.wav), "out": str(c.emb_dir / f"{c.wav.stem}.{name}.e2v.npy")})
            for it in items:
                Path(it["out"]).parent.mkdir(parents=True, exist_ok=True)
            res = {r["id"]: r for r in gu.run_batched(b, "emotion", items)}
            for c in todo:
                r = res.get(c.cid, {})
                p = r.get("embedding_path") or (c.emb_dir / f"{c.wav.stem}.{name}.e2v.npy")
                o = orig_out.get(c.line.line_id)
                if r.get("error") or not Path(p).exists() or o is None or not Path(o).exists():
                    continue
                c.rec.metrics["emo_sim"] = round(cosine(np.load(p), np.load(o)), 4)
                if r.get("label"):
                    setattr(c.rec, "emo_label", r["label"])
                    orig_label = (c.passport.get("style") or {}).get("emotion")
                    if orig_label:
                        c.rec.metrics["emo_match"] = 1.0 if r["label"] == orig_label else 0.0
    # A/D/V delta
    todo = [c for c in cands if c.needs("adv_delta")]
    if todo and s.adv:
        b, _ = _acquire(ctx, [s.adv], "adv", unavailable)
        if b is not None:
            orig_items = {}
            for c in todo:
                if not (c.passport.get("emotion") or {}).get("adv") and (c.line.src_wav16 or c.line.src_wav):
                    orig_items[c.line.line_id] = {"id": f"orig|{c.line.line_id}", "wav": str(ctx.paths.abs(c.line.src_wav16 or c.line.src_wav))}
            res = {r["id"]: r for r in gu.run_batched(b, "adv", [{"id": c.cid, "wav": str(c.wav)} for c in todo] + list(orig_items.values()))}
            for c in todo:
                r = res.get(c.cid, {})
                o = (c.passport.get("emotion") or {}).get("adv") or res.get(f"orig|{c.line.line_id}", {})
                if r.get("error") or r.get("arousal") is None or o.get("arousal") is None:
                    continue
                c.rec.metrics["adv_delta"] = round(abs(float(r["arousal"]) - float(o["arousal"])) + abs(float(r.get("valence", 0.5)) - float(o.get("valence", 0.5))), 4)
    # CER
    todo = [c for c in cands if c.needs("cer")]
    if todo:
        b, _ = _acquire(ctx, [s.cer_asr, s.cer_asr_fallback], "cer", unavailable)
        if b is not None:
            res = {r["id"]: r for r in gu.run_batched(b, "transcribe", [{"id": c.cid, "wav": str(c.wav), "lang": tgt} for c in todo], lang=tgt)}
            trs = gu.load_translations(ctx)
            for c in todo:
                r = res.get(c.cid, {})
                if r.get("error"):
                    continue
                cont = [trs.get(x) for x in ((c.line.model_extra or {}).get("sustain_lines") or [])]
                ref = cer_reference(c.side, c.tr, [x for x in cont if x is not None])
                hyp = r.get("text", "")
                c.rec.metrics["cer"] = round(cer_fn(ref or "", hyp), 4)
                setattr(c.rec, "asr_text", hyp)
    # duration + prosody (core DSP, F0 via the configured f0 backend)
    for c in cands:
        if c.rec.metrics.get("dur_ratio") is None:
            src_d = gu.src_duration(ctx, c.line, c.passport)
            if src_d > 0:
                c.rec.metrics["dur_ratio"] = round(features.duration(c.wav) / src_d, 4)
                c.rec.metrics["duration_s"] = round(features.duration(c.wav), 3)
    todo = [c for c in cands if c.needs("prosody")]
    if todo:
        f0b, _, _ = gu.acquire_first(ctx, [cfg.analyze.f0])
        jobs = [(c.cid, c.wav, c.feat_dir / f"{c.wav.stem}.f0.npy") for c in todo]
        for c in todo:
            pf = (c.passport.get("prosody") or {}).get("f0_contour_path")
            if not (pf and ctx.paths.abs(pf).exists()) and (c.line.src_wav16 or c.line.src_wav):
                jobs.append((f"orig|{c.line.line_id}", ctx.paths.abs(c.line.src_wav16 or c.line.src_wav), c.feat_dir / "orig.f0.npy"))
        f0s = gu.f0_of(ctx, f0b, list(dict.fromkeys(jobs)))
        for c in todo:
            p = c.passport.get("prosody") or {}
            if p.get("f0_contour_path") and ctx.paths.abs(p["f0_contour_path"]).exists():
                f0_src = np.load(ctx.paths.abs(p["f0_contour_path"]))
            else:
                f0_src = f0s.get(f"orig|{c.line.line_id}")
            f0_c = f0s.get(c.cid)
            y, sr = features.load(c.wav, sr=16000)
            e_c = features.rms_db(y, sr)
            if p.get("energy_contour_path") and ctx.paths.abs(p["energy_contour_path"]).exists():
                e_src = np.load(ctx.paths.abs(p["energy_contour_path"]))
            elif c.line.src_wav16 or c.line.src_wav:
                ys, srs = features.load(ctx.paths.abs(c.line.src_wav16 or c.line.src_wav), sr=16000)
                e_src = features.rms_db(ys, srs)
            else:
                e_src = None
            fc = pros.f0_corr(f0_src, f0_c) if f0_src is not None and f0_c is not None else None
            ec = pros.energy_corr(e_src, e_c) if e_src is not None else None
            prof = vad.silence_profile(vad.energy_vad(y, sr), len(y) / sr)
            src_pauses = p.get("pauses")
            src_dur = gu.src_duration(ctx, c.line, c.passport)
            pm = pros.pause_match(src_pauses, src_dur, prof["pauses"], len(y) / sr) if src_pauses is not None else None
            c.rec.metrics["f0_corr"] = None if fc is None else round(fc, 4)
            c.rec.metrics["energy_corr"] = None if ec is None else round(ec, 4)
            c.rec.metrics["pause_match"] = None if pm is None else round(pm, 4)
            ps = pros.prosody_score(fc, ec, pm)
            c.rec.metrics["prosody"] = None if ps is None else round(ps, 4)
            if ps is None:
                c.rec.na = [*c.na, "prosody"]  # type: ignore[attr-defined]
    for group, reason in unavailable.items():
        for c in cands:
            note = f"{group} unavailable: {reason[:120]}"
            if note not in c.rec.notes:
                c.rec.notes = [*c.rec.notes, note]

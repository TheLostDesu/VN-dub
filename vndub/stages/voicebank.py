"""voicebank: per-character reference clips, native performer voice, RVC model, LoRA.

Order of work (one model resident at a time):

1. refs (no model): 5-10 clean clips covering the character's top emotions
   -> ``voicebank/<sid>/refs/*.wav`` + ``refs.json`` + ``characters.yaml refs``.
2. performer designs (design backend, all characters): ``n_designs`` voices x
   ``test_sentences`` native target-language sentences.
3. speaker embedder (all characters): refs + clean clips (character centroid,
   EER calibration) + performer designs (selection by similarity).
4. F0 of designs (f0 backend / core pYIN): pick the design closest to the
   character whose median F0 is within ``max_f0_delta_semitones``.
5. performer emotion variants (design backend again, if it takes style
   instructions or knobs) -> ``performer.yaml``.
6. RVC training per character with enough clean audio (rvc backend).
7. route C (optional): LoRA on route-B winners that passed the gates.
"""

from __future__ import annotations

import math
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from vndub.backends.base import BackendUnavailable
from vndub.backends.score.speaker_sim import calibrate_eer, centroid, cosine
from vndub.manifest.io import Character, LineRecord, RefClip, read_json, save_characters, write_json, write_yaml
from vndub.stages import genutil as gu
from vndub.stages.common import StageError, counter
from vndub.text.style_prompt import StyleSpec, render

NAME = "voicebank"
VERSION = "1"
CONFIG_SECTIONS = ("voicebank", "score", "project")

TEST_SENTENCES = {
    "ru": [
        "Привет! Сегодня отличный день, правда?",
        "Я не знаю, что сказать... Давай подумаем вместе.",
        "Подожди меня, пожалуйста, я сейчас вернусь.",
        "Это просто невероятно! Как ты это сделал?",
        "Ну и ладно. Мне всё равно.",
    ],
    "en": [
        "Hi! It's a wonderful day today, isn't it?",
        "I don't know what to say... Let's think it over together.",
        "Please wait for me, I'll be right back.",
        "That's simply incredible! How did you do that?",
        "Fine, whatever. I don't care.",
    ],
}

_SENTENCE_END = re.compile(r"[。！？!?…」』\.]\s*$")
N_CLIPS_FOR_CENTROID = 10


# ----------------------------------------------------------------------------
# refs
# ----------------------------------------------------------------------------


def _emotion_of(passport: dict[str, Any]) -> str:
    style = passport.get("style") or {}
    if style.get("emotion"):
        return str(style["emotion"])
    probs = ((passport.get("emotion") or {}).get("e2v") or {}).get("probs") or {}
    return max(probs, key=probs.get) if probs else "neutral"


def select_refs(
    lines: list[LineRecord],
    passports: dict[str, dict[str, Any]],
    *,
    min_snr_db: float,
    min_sec: float,
    max_sec: float,
    per_emotion: int,
    max_total: int,
    min_refs: int,
) -> tuple[list[dict[str, Any]], bool]:
    """Pick reference clips. Returns (chosen candidates, relaxed criteria?)."""

    def cands(strict: bool) -> list[dict[str, Any]]:
        out = []
        for l in lines:
            p = passports.get(l.line_id)
            if not p or not l.src_wav:
                continue
            dur = float(p.get("duration_s") or 0.0)
            snr = p.get("snr_db")
            if p.get("bgm"):
                if strict:
                    continue
            if strict and not (min_sec <= dur <= max_sec):
                continue
            if strict and (snr is None or float(snr) < min_snr_db):
                continue
            full = bool(_SENTENCE_END.search(l.text_src or ""))
            adv = (p.get("emotion") or {}).get("adv") or {}
            # prefer clean, full sentences, durations inside the window
            in_window = min_sec <= dur <= max_sec
            score = float(snr if snr is not None else 0.0) + (3.0 if full else 0.0) + (5.0 if in_window else 0.0) - (10.0 if p.get("bgm") else 0.0)
            out.append(
                {
                    "line": l,
                    "emotion": _emotion_of(p),
                    "snr": None if snr is None else float(snr),
                    "duration": dur,
                    "arousal": adv.get("arousal"),
                    "valence": adv.get("valence"),
                    "score": score,
                }
            )
        return out

    relaxed = False
    pool = cands(strict=True)
    if len(pool) < min_refs:
        relaxed = True
        pool = cands(strict=False)
    if not pool:
        return [], relaxed
    hist = counter([c["emotion"] for c in pool])
    by_emo: dict[str, list[dict[str, Any]]] = {}
    for c in sorted(pool, key=lambda c: -c["score"]):
        by_emo.setdefault(c["emotion"], []).append(c)
    chosen: list[dict[str, Any]] = []
    # round-robin over emotions ordered by frequency
    for rnd in range(per_emotion):
        for emo in hist:
            lst = by_emo.get(emo, [])
            if rnd < len(lst) and len(chosen) < max_total:
                chosen.append(lst[rnd])
    if len(chosen) < min(min_refs, len(pool)):
        rest = [c for c in sorted(pool, key=lambda c: -c["score"]) if c not in chosen]
        chosen += rest[: min(min_refs, len(pool)) - len(chosen)]
    # usage: best clip of each emotion and the overall cleanest neutral-ish clip -> both
    seen: set[str] = set()
    for c in chosen:
        c["usage"] = "both" if c["emotion"] not in seen else "emotion"
        seen.add(c["emotion"])
    if len(chosen) <= 2:
        for c in chosen:
            c["usage"] = "both"
    neutral = [c for c in chosen if c["emotion"] == "neutral"] or chosen
    best = max(neutral, key=lambda c: c["score"])
    best["usage"] = "timbre" if len(chosen) > 2 and best["emotion"] != "neutral" else "both"
    return chosen, relaxed


def _write_refs(ctx: Any, sid: str, chosen: list[dict[str, Any]]) -> list[RefClip]:
    d = ctx.paths.vb_refs(sid)
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    refs: list[RefClip] = []
    n_by: dict[str, int] = {}
    for c in chosen:
        l: LineRecord = c["line"]
        n_by[c["emotion"]] = n_by.get(c["emotion"], 0) + 1
        dst = d / f"{c['emotion']}_{n_by[c['emotion']]:02d}.wav"
        src = ctx.paths.abs(l.src_wav)
        shutil.copy2(src, dst)
        refs.append(
            RefClip(
                path=ctx.paths.rel(dst),
                line_id=l.line_id,
                emotion=c["emotion"],
                arousal=c["arousal"],
                valence=c["valence"],
                snr=c["snr"],
                duration=round(c["duration"], 3),
                usage=c["usage"],
                text=l.text_src,
            )
        )
    write_json(ctx.paths.vb_refs_json(sid), [r.model_dump(mode="json") for r in refs])
    return refs


# ----------------------------------------------------------------------------
# performer description
# ----------------------------------------------------------------------------


def pitch_word(f0: float | None, gender: str) -> str:
    if not f0:
        return "medium pitch"
    hi, lo = {"f": (280.0, 200.0), "m": (160.0, 110.0)}.get(gender, (240.0, 150.0))
    if f0 >= hi:
        return "high pitch"
    if f0 < lo:
        return "low pitch"
    return "medium pitch"


_AGE = {"child": "child", "teen": "young", "young": "young", "adult": "adult", "middle": "middle-aged", "old": "elderly", "elderly": "elderly"}
_EMO_WORD = {
    "happy": "cheerful and bright", "sad": "soft and melancholic", "angry": "sharp and forceful", "neutral": "calm and even",
    "surprised": "lively", "fearful": "timid", "embarrassed": "shy", "disgusted": "dry",
}


def performer_description(char: Character, lang: str) -> str:
    gender = {"f": "female", "m": "male"}.get(char.gender, "")
    age = _AGE.get((char.age_hint or "").lower(), "")
    who = " ".join(x for x in (age, gender) if x) or "neutral"
    hist = char.stats.emotion_hist or {}
    top = max(hist, key=hist.get) if hist else "neutral"
    lang_name = {"ru": "Russian", "en": "English", "ja": "Japanese"}.get(lang, lang)
    parts = [
        f"A {who} voice, {pitch_word(char.stats.f0_median_hz, char.gender)}",
        f"{_EMO_WORD.get(top, 'expressive')} character",
        f"native {lang_name} speaker with clear standard pronunciation",
    ]
    return ", ".join(parts) + "."


# ----------------------------------------------------------------------------
# stage
# ----------------------------------------------------------------------------


def run(ctx: Any) -> dict[str, Any]:
    vb = ctx.cfg.voicebank
    lang = ctx.cfg.project.tgt_lang
    lines = ctx.voiced_lines(apply_only=False)
    if not lines:
        raise StageError("no voiced lines: run parse/extract (and analyze) first")
    chars = ctx.characters()
    by_spk: dict[str, list[LineRecord]] = {}
    for l in lines:
        if l.speaker_id == "narrator":
            continue
        by_spk.setdefault(l.speaker_id, []).append(l)
    if ctx.only:
        sel = {s for s in by_spk if s in ctx.only}
        if sel:
            by_spk = {s: by_spk[s] for s in sel}
    passports: dict[str, dict[str, Any]] = {}
    for spk_lines in by_spk.values():
        for l in spk_lines:
            p = gu.load_passport(ctx, l.line_id)
            if p:
                passports[l.line_id] = p
    if not passports:
        ctx.log.warning("no passports found: refs use durations only (run analyze for emotion/SNR-aware selection)")

    cache = ctx.cache(NAME, VERSION, "voicebank", "score", "project")
    summary: dict[str, Any] = {"characters": 0, "refs": 0, "performers": 0, "rvc": 0, "lora": 0, "skipped": {}}
    eligible: dict[str, list[LineRecord]] = {}
    minutes: dict[str, float] = {}
    audio = gu.load_audio_index(ctx)
    for sid, ls in by_spk.items():
        mins = sum(gu.src_duration(ctx, l, passports.get(l.line_id), audio.get(l.audio_id or "")) for l in ls) / 60.0
        minutes[sid] = mins
        if mins + 1e-9 < vb.min_minutes_for_voicebank:
            summary["skipped"][sid] = f"{mins:.1f} min < {vb.min_minutes_for_voicebank} min"
            continue
        eligible[sid] = ls
        chars.setdefault(sid, Character(names={"src": ls[0].speaker_raw or sid, "tgt": ""}))
    summary["characters"] = len(eligible)

    # 1. refs -------------------------------------------------------------------
    refs: dict[str, list[RefClip]] = {}
    for sid, ls in eligible.items():
        key = cache.key("refs", [(l.line_id, l.src_wav) for l in ls], [passports.get(l.line_id, {}).get("snr_db") for l in ls], vb.refs.model_dump())
        if cache.fresh(f"refs:{sid}", key, [ctx.paths.vb_refs_json(sid)]):
            refs[sid] = gu.load_refs(ctx, sid)
        else:
            chosen, relaxed = select_refs(
                ls, passports,
                min_snr_db=vb.refs.min_snr_db, min_sec=vb.refs.min_sec, max_sec=vb.refs.max_sec,
                per_emotion=vb.refs.per_emotion, max_total=vb.refs.max_total, min_refs=vb.refs.min_refs,
            )
            if relaxed:
                ctx.log.warning("%s: fewer than %d clips meet the ref criteria; relaxed SNR/duration filters", sid, vb.refs.min_refs)
            refs[sid] = _write_refs(ctx, sid, chosen)
            cache.put(f"refs:{sid}", key)
        chars[sid].refs = [r.path for r in refs[sid]]
        summary["refs"] += len(refs[sid])

    # character F0 (stats from analyze, else core pYIN over the refs)
    char_f0: dict[str, float | None] = {}
    for sid in eligible:
        f0 = chars[sid].stats.f0_median_hz
        if not f0 and refs[sid]:
            res = gu.f0_of(ctx, None, [(r.path, ctx.paths.abs(r.path), ctx.paths.vb(sid) / "feat" / f"{Path(r.path).stem}.f0.npy") for r in refs[sid]])
            vals = [m for m in (gu.median_f0(v) for v in res.values()) if m]
            f0 = float(np.median(vals)) if vals else None
        char_f0[sid] = f0

    # 2. performer designs ------------------------------------------------------
    perf_kind = vb.performer.kind
    sentences = TEST_SENTENCES.get(lang, TEST_SENTENCES["en"])[: max(1, vb.performer.test_sentences)]
    designs: dict[str, list[dict[str, Any]]] = {}  # sid -> [{d, seed, description, clips:[(path,text)]}]
    performer_reason: dict[str, str] = {}
    design_backend_name = None
    design_caps: dict[str, Any] = {}
    need_perf = [sid for sid in eligible if not _performer_fresh(ctx, cache, sid, vb, char_f0, chars)]
    if perf_kind == "voice_design" and need_perf:
        backend, design_backend_name, reason = gu.acquire_first(ctx, [vb.performer.backend])
        if backend is None:
            for sid in need_perf:
                performer_reason[sid] = f"design backend unavailable: {reason}"
        else:
            design_caps = dict(backend.spec.caps)
            items = []
            for sid in need_perf:
                c = chars[sid]
                desc = performer_description(c, lang)
                designs[sid] = []
                for d in range(max(1, vb.performer.n_designs)):
                    seed = ctx.cfg.synth.seed_base + gu.stable_seed("performer", sid, d)
                    factor = 1.0 if vb.performer.n_designs <= 1 else 0.85 + 0.3 * d / (vb.performer.n_designs - 1)
                    entry = {"d": d, "seed": seed, "description": desc, "clips": []}
                    for t, text in enumerate(sentences):
                        out = ctx.paths.vb(sid) / "performer" / "designs" / f"d{d}_s{t}.wav"
                        it = {"id": f"{sid}|{d}|{t}", "text": text, "description": desc, "instruct": desc, "lang": lang, "seed": seed, "out": str(out)}
                        if backend.spec.env is None:  # local backends (mock/sapi) take pitch hints
                            it["knobs"] = {"f0_hz": (char_f0[sid] or 200.0) * factor, "gender": c.gender}
                        items.append(it)
                        entry["clips"].append((out, text))
                    designs[sid].append(entry)
            results = gu.run_batched(backend, "design", items, chunk=8)
            failed = {r["id"] for r in results if r.get("error")}
            for r in results:
                if r.get("error"):
                    ctx.log.warning("design %s failed: %s", r["id"], r["error"])
            for sid in list(designs):
                for entry in designs[sid]:
                    entry["clips"] = [(p, t) for i, (p, t) in enumerate(entry["clips"]) if f"{sid}|{entry['d']}|{i}" not in failed and p.exists()]
                designs[sid] = [e for e in designs[sid] if e["clips"]]
                if not designs[sid]:
                    performer_reason[sid] = "all performer designs failed"
    elif perf_kind == "reference" and need_perf:
        for sid in need_perf:
            src_dir = Path(ctx.cfg.resolve(vb.performer.reference_dir)) / sid if vb.performer.reference_dir else None
            clips = sorted(src_dir.glob("*.wav")) if src_dir and src_dir.exists() else []
            if not clips:
                performer_reason[sid] = f"no native reference clips in {src_dir}"
                continue
            designs[sid] = [{"d": 0, "seed": None, "description": f"reference clips from {src_dir}", "clips": [(p, "") for p in clips], "reference": True}]

    # 3. embeddings: refs + clean clips (centroid, calibration) + designs ---------
    emb_backend, emb_name, emb_reason = gu.acquire_first(ctx, [ctx.cfg.score.speaker])
    centroids: dict[str, np.ndarray] = {}
    ref_vecs: dict[str, list[np.ndarray]] = {}
    design_vecs: dict[str, dict[int, np.ndarray]] = {}
    if emb_backend is not None:
        jobs = []
        for sid in eligible:
            edir = ctx.paths.vb(sid) / "emb"
            for r in refs[sid]:
                jobs.append((f"ref|{sid}|{r.path}", ctx.paths.abs(r.path), edir / f"ref_{Path(r.path).stem}.{emb_name}.npy"))
            clean = sorted(
                (l for l in eligible[sid] if l.src_wav16 or l.src_wav),
                key=lambda l: -float((passports.get(l.line_id) or {}).get("snr_db") or 0.0),
            )[:N_CLIPS_FOR_CENTROID]
            for l in clean:
                wav = ctx.paths.abs(l.src_wav16 or l.src_wav)
                jobs.append((f"clip|{sid}|{l.line_id}", wav, edir / f"clip_{l.line_id}.{emb_name}.npy"))
            for entry in designs.get(sid, []):
                for p, _ in entry["clips"]:
                    jobs.append((f"design|{sid}|{entry['d']}|{p.name}", p, p.with_suffix(f".{emb_name}.npy")))
        vecs = gu.embed_paths(ctx, emb_backend, jobs)
        for sid in eligible:
            rv = [v for k, v in vecs.items() if k.startswith(f"ref|{sid}|")]
            cv = [v for k, v in vecs.items() if k.startswith(f"clip|{sid}|")]
            ref_vecs[sid] = rv + cv
            if ref_vecs[sid]:
                centroids[sid] = centroid(ref_vecs[sid])
                np.save(ctx.paths.vb(sid) / f"centroid.{emb_name}.npy", centroids[sid])
            design_vecs[sid] = {}
            for entry in designs.get(sid, []):
                dv = [v for k, v in vecs.items() if k.startswith(f"design|{sid}|{entry['d']}|")]
                if dv:
                    design_vecs[sid][entry["d"]] = centroid(dv)
        if vb.calibrate_spk:
            summary["calibration"] = _calibrate(ctx, emb_name, ref_vecs)
    else:
        ctx.log.warning("speaker embedder unavailable (%s): no calibration, performer chosen by F0 only", emb_reason)
        summary["calibration"] = f"unavailable: {emb_reason}"

    # 4. F0 of designs + performer choice ----------------------------------------
    f0_backend = None
    if designs:
        f0_backend, _, _ = gu.acquire_first(ctx, [ctx.cfg.analyze.f0])
    chosen_design: dict[str, dict[str, Any]] = {}
    for sid, entries in designs.items():
        jobs = [(f"{entry['d']}|{p.name}", p, p.with_suffix(".f0.npy")) for entry in entries for p, _ in entry["clips"]]
        f0s = gu.f0_of(ctx, f0_backend, jobs)
        scored = []
        for entry in entries:
            meds = [m for m in (gu.median_f0(f0s.get(f"{entry['d']}|{p.name}")) for p, _ in entry["clips"]) if m]
            entry["f0_median_hz"] = float(np.median(meds)) if meds else None
            entry["f0_delta_st"] = gu.semitone_delta(entry["f0_median_hz"], char_f0.get(sid))
            sim = None
            if sid in centroids and entry["d"] in design_vecs.get(sid, {}):
                sim = cosine(design_vecs[sid][entry["d"]], centroids[sid])
            entry["spk_sim"] = sim
            scored.append(entry)
        within = [e for e in scored if e["f0_delta_st"] is None or abs(e["f0_delta_st"]) <= vb.performer.max_f0_delta_semitones]
        pool = within or scored
        if not within:
            ctx.log.warning("%s: no performer design within %.1f st of the character F0; using the closest overall", sid, vb.performer.max_f0_delta_semitones)
        best = max(pool, key=lambda e: (e["spk_sim"] if e["spk_sim"] is not None else -1.0, -abs(e["f0_delta_st"] or 0.0)))
        best["warning"] = "" if within else "f0 outside window"
        chosen_design[sid] = best

    # 5. performer refs (+ emotion variants) -> performer.yaml --------------------
    variant_items: list[dict[str, Any]] = []
    can_vary = design_caps.get("instruct", "none") != "none" or design_caps.get("knobs", "none") != "none"
    for sid, best in chosen_design.items():
        pdir = ctx.paths.vb(sid) / "performer"
        prefs = []
        for i, (p, text) in enumerate(best["clips"]):
            dst = pdir / f"neutral_{i + 1:02d}.wav"
            shutil.copy2(p, dst)
            prefs.append({"path": ctx.paths.rel(dst), "emotion": _ref_emotion_from_name(p.stem) if best.get("reference") else "neutral", "text": text})
        best["refs"] = prefs
        if can_vary and not best.get("reference"):
            emos = [e for e in dict.fromkeys(r.emotion for r in refs[sid]) if e not in ("neutral", "other", "unknown")]
            for emo in emos[:5]:
                spec = StyleSpec(emotion=emo, intensity=0.65)
                rendered = render(spec, design_caps)
                instr = rendered.get("instruct") or rendered.get("style_prompt") or ""
                out = pdir / f"{emo}_01.wav"
                it = {
                    "id": f"{sid}|{emo}", "text": sentences[0], "lang": lang, "seed": best["seed"], "out": str(out),
                    "description": f"{best['description']} {instr}".strip(), "instruct": f"{best['description']} {instr}".strip(),
                }
                knobs = dict(rendered.get("knobs") or {})
                if design_backend_name and gu_is_local(ctx, design_backend_name):
                    knobs.update({"f0_hz": best.get("f0_median_hz") or char_f0.get(sid) or 200.0, "gender": chars[sid].gender})
                if knobs:
                    it["knobs"] = knobs
                variant_items.append(it)
    if variant_items and design_backend_name:
        try:
            backend = ctx.models.acquire(design_backend_name)
            for r in gu.run_batched(backend, "design", variant_items, chunk=8):
                sid, emo = r["id"].split("|", 1)
                out = ctx.paths.vb(sid) / "performer" / f"{emo}_01.wav"
                if not r.get("error") and out.exists():
                    chosen_design[sid]["refs"].append({"path": ctx.paths.rel(out), "emotion": emo, "text": sentences[0]})
                elif r.get("error"):
                    ctx.log.warning("performer variant %s failed: %s", r["id"], r["error"])
        except BackendUnavailable as e:
            ctx.log.warning("performer emotion variants skipped: %s", e)
    for sid, best in chosen_design.items():
        perf = {
            "kind": perf_kind,
            "backend": design_backend_name or ("reference" if best.get("reference") else ""),
            "description": best["description"],
            "seed": best["seed"],
            "design_index": best["d"],
            "refs": best["refs"],
            "f0_median_hz": None if best.get("f0_median_hz") is None else round(best["f0_median_hz"], 2),
            "spk_sim": None if best.get("spk_sim") is None else round(best["spk_sim"], 4),
            "f0_delta_st": None if best.get("f0_delta_st") is None else round(best["f0_delta_st"], 2),
            "embedder": emb_name,
            "warnings": [w for w in [best.get("warning", "")] if w],
        }
        write_yaml(ctx.paths.vb_performer(sid), perf)
        chars[sid].performer = {k: perf[k] for k in ("kind", "backend", "description", "f0_median_hz", "spk_sim", "f0_delta_st")}
        chars[sid].performer["path"] = ctx.paths.rel(ctx.paths.vb_performer(sid))
        cache.put(f"performer:{sid}", _performer_key(cache, sid, vb, char_f0, chars))
    for sid in eligible:
        if ctx.paths.vb_performer(sid).exists():
            summary["performers"] += 1
            if not chars[sid].performer:
                perf = gu.load_performer(ctx, sid) or {}
                chars[sid].performer = {k: perf.get(k) for k in ("kind", "backend", "description", "f0_median_hz", "spk_sim", "f0_delta_st")}
    if performer_reason:
        summary["performer_missing"] = performer_reason
        for sid, reason in performer_reason.items():
            ctx.log.warning("%s: no performer (%s): route B will be skipped for this character", sid, reason)

    # 6. RVC --------------------------------------------------------------------
    if vb.rvc.enabled:
        summary["rvc"] = _train_rvc(ctx, cache, eligible, passports, refs, chars, char_f0, summary)

    # 7. route C ------------------------------------------------------------------
    if vb.route_c.enabled:
        summary["lora"] = _train_lora(ctx, cache, eligible, chars, summary)

    cache.save()
    save_characters(ctx.paths.characters, chars)
    summary["minutes"] = {s: round(m, 2) for s, m in minutes.items()}
    if not summary["skipped"]:
        summary.pop("skipped")
    return summary


def gu_is_local(ctx: Any, name: str) -> bool:
    from vndub.backends import registry

    try:
        return registry.get(name).env is None
    except KeyError:
        return False


def _ref_emotion_from_name(stem: str) -> str:
    from vndub.text.style_prompt import ALL_EMOTIONS

    head = stem.split("_", 1)[0].lower()
    return head if head in ALL_EMOTIONS else "neutral"


def _performer_key(cache: Any, sid: str, vb: Any, char_f0: dict[str, Any], chars: dict[str, Character]) -> str:
    c = chars[sid]
    return cache.key("performer", vb.performer.model_dump(), char_f0.get(sid), c.gender, c.age_hint, c.stats.emotion_hist)


def _performer_fresh(ctx: Any, cache: Any, sid: str, vb: Any, char_f0: dict[str, Any], chars: dict[str, Character]) -> bool:
    perf = gu.load_performer(ctx, sid)
    if not perf or not all(ctx.paths.abs(r["path"]).exists() for r in perf.get("refs", [])):
        return False
    return cache.fresh(f"performer:{sid}", _performer_key(cache, sid, vb, char_f0, chars), [ctx.paths.vb_performer(sid)])


def _stats(xs: list[float]) -> dict[str, float]:
    a = np.asarray(xs, dtype=np.float64)
    return {"n": int(a.size), "mean": round(float(a.mean()), 4), "std": round(float(a.std()), 4), "min": round(float(a.min()), 4), "max": round(float(a.max()), 4)}


def _calibrate(ctx: Any, emb_name: str | None, vecs: dict[str, list[np.ndarray]]) -> Any:
    same: list[float] = []
    diff: list[float] = []
    sids = [s for s, v in vecs.items() if v]
    for s in sids:
        v = vecs[s]
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                same.append(cosine(v[i], v[j]))
    for i, a in enumerate(sids):
        for b in sids[i + 1 :]:
            for x in vecs[a]:
                for y in vecs[b]:
                    diff.append(cosine(x, y))
    if not same or not diff:
        reason = "need >= 2 characters with >= 2 clips each"
        ctx.log.warning("speaker calibration skipped: %s", reason)
        return f"skipped: {reason}"
    thr, eer = calibrate_eer(same, diff)
    data = {
        "embedder": emb_name,
        "threshold": round(thr, 4),
        "eer": round(eer, 4),
        "same": _stats(same),
        "diff": _stats(diff),
        "n_same": len(same),
        "n_diff": len(diff),
    }
    write_json(ctx.paths.spk_calibration, data)
    return f"threshold={thr:.3f} eer={eer:.3f}"


def _train_rvc(ctx: Any, cache: Any, eligible: dict[str, list[LineRecord]], passports: dict[str, dict[str, Any]], refs: dict[str, list[RefClip]], chars: dict[str, Character], char_f0: dict[str, Any], summary: dict[str, Any]) -> int:
    rv = ctx.cfg.voicebank.rvc
    vbref = ctx.cfg.voicebank.refs
    plans: dict[str, list[str]] = {}
    reasons: dict[str, str] = {}
    for sid, ls in eligible.items():
        clean = []
        for l in ls:
            p = passports.get(l.line_id) or {}
            snr = p.get("snr_db")
            if p.get("bgm") or snr is None or float(snr) < vbref.min_snr_db or float(p.get("duration_s") or 0) < 1.0:
                continue
            clean.append((l, p))
        mins = sum(float(p.get("duration_s") or 0) for _, p in clean) / 60.0
        if mins + 1e-9 < rv.min_minutes:
            reasons[sid] = f"{mins:.1f} min clean audio < {rv.min_minutes} min"
            continue
        # emotion-coverage round robin until max_minutes
        by_emo: dict[str, list[tuple[LineRecord, dict[str, Any]]]] = {}
        for l, p in sorted(clean, key=lambda lp: -float(lp[1].get("snr_db") or 0)):
            by_emo.setdefault(_emotion_of(p), []).append((l, p))
        picked: list[str] = []
        total = 0.0
        idx = 0
        while total < rv.max_minutes * 60 and any(idx < len(v) for v in by_emo.values()):
            for v in by_emo.values():
                if idx < len(v) and total < rv.max_minutes * 60:
                    l, p = v[idx]
                    picked.append(str(ctx.paths.abs(l.src_wav)))
                    total += float(p.get("duration_s") or 0)
            idx += 1
        plans[sid] = picked
    n = 0
    backend = None
    for sid, clips in plans.items():
        out_dir = ctx.paths.vb_rvc(sid)
        perf = gu.load_performer(ctx, sid) or {}
        f0_t = 0
        if char_f0.get(sid) and perf.get("f0_median_hz"):
            f0_t = int(round(12 * math.log2(float(char_f0[sid]) / float(perf["f0_median_hz"]))))
        key = cache.key("rvc", rv.model_dump(), sorted(clips))
        meta = read_json(out_dir / "meta.json")
        if meta and cache.fresh(f"rvc:{sid}", key, [out_dir / "meta.json"]) and ctx.paths.abs(meta.get("model", "missing")).exists():
            if meta.get("f0_transpose") != f0_t:
                meta["f0_transpose"] = f0_t
                write_json(out_dir / "meta.json", meta)
            chars[sid].rvc = meta
            n += 1
            continue
        if backend is None:
            backend, _, reason = gu.acquire_first(ctx, [rv.backend])
            if backend is None:
                for s in plans:
                    reasons[s] = f"rvc backend unavailable: {reason}"
                break
        job = {
            "speaker_id": sid, "clips": clips, "out_dir": str(out_dir), "sample_rate": rv.sample_rate, "epochs": rv.epochs,
            "batch_size": rv.batch_size, "f0_method": rv.f0_method, "overfit_patience": rv.overfit_patience,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            res = backend.train(job) or {}
        except BackendUnavailable:
            raise
        except Exception as e:
            reasons[sid] = f"rvc training failed: {type(e).__name__}: {e}"
            ctx.log.error("%s: %s", sid, reasons[sid])
            continue
        if res.get("error") or not res.get("model"):
            reasons[sid] = f"rvc training failed: {res.get('error', 'no model returned')}"
            continue
        meta = {
            "model": ctx.paths.rel(res["model"]),
            "index": ctx.paths.rel(res["index"]) if res.get("index") else None,
            "f0_transpose": f0_t,
            "minutes": round(sum(gu_duration(c) for c in clips) / 60.0, 2),
            "epochs": res.get("epochs"),
            "sample_rate": rv.sample_rate,
            "backend": backend.spec.name,
        }
        write_json(out_dir / "meta.json", meta)
        chars[sid].rvc = meta
        cache.put(f"rvc:{sid}", key)
        n += 1
    if reasons:
        summary["rvc_skipped"] = reasons
    return n


def gu_duration(path: str) -> float:
    from vndub.audio.features import duration

    try:
        return duration(path)
    except Exception:
        return 0.0


def _train_lora(ctx: Any, cache: Any, eligible: dict[str, list[LineRecord]], chars: dict[str, Character], summary: dict[str, Any]) -> int:
    rc = ctx.cfg.voicebank.route_c
    sel = gu.load_selection(ctx)
    trs = gu.load_translations(ctx)
    if not sel:
        summary["lora_skipped"] = "no selection.jsonl yet: run synth + score on route B first"
        return 0
    reasons: dict[str, str] = {}
    n = 0
    backend = None
    sentences = TEST_SENTENCES.get(ctx.cfg.project.tgt_lang, TEST_SENTENCES["en"])
    for sid, ls in eligible.items():
        pairs = []
        total = 0.0
        for l in ls:
            s = sel.get(l.line_id)
            tr = trs.get(l.line_id)
            if not s or not tr or s.status not in ("auto", "override") or not s.winner or s.route not in ("B",):
                continue
            if (s.scores.get("composite") or 0) < rc.composite_threshold:
                continue
            wav = ctx.paths.abs(s.winner)
            if not wav.exists():
                continue
            d = gu_duration(str(wav))
            if total + d > rc.max_minutes * 60:
                break
            pairs.append((l.line_id, tr.text_tts or tr.text_tgt, wav))
            total += d
        if total / 60.0 + 1e-9 < rc.min_minutes:
            reasons[sid] = f"{total / 60:.1f} min of gated route-B winners < {rc.min_minutes} min"
            continue
        syn = ctx.paths.vb_synthetic(sid)
        syn.mkdir(parents=True, exist_ok=True)
        recs = []
        for lid, text, wav in pairs:
            dst = syn / f"{lid}.wav"
            shutil.copy2(wav, dst)
            recs.append({"text": text, "wav": str(dst)})
        write_json(syn / "pairs.json", recs)
        key = cache.key("lora", rc.model_dump(), [(r["text"], Path(r["wav"]).name) for r in recs])
        meta = read_json(ctx.paths.vb_lora(sid) / "meta.json")
        if meta and cache.fresh(f"lora:{sid}", key, [ctx.paths.vb_lora(sid) / "meta.json"]):
            chars[sid].lora = meta
            n += 1
            continue
        if backend is None:
            backend, _, reason = gu.acquire_first(ctx, [rc.backend])
            if backend is None:
                for s in eligible:
                    reasons.setdefault(s, f"route-C backend unavailable: {reason}")
                break
        out_dir = ctx.paths.vb_lora(sid)
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            res = backend.call("train_lora", None, job={"speaker_id": sid, "pairs": recs, "out_dir": str(out_dir)}) or {}
        except NotImplementedError:
            reasons[sid] = f"backend {backend.spec.name} has no train_lora op"
            continue
        except Exception as e:
            reasons[sid] = f"LoRA training failed: {type(e).__name__}: {e}"
            continue
        if res.get("error") or not res.get("adapter"):
            reasons[sid] = f"LoRA training failed: {res.get('error', 'no adapter returned')}"
            continue
        # 5-sentence selftest with the adapter
        st_items = [{"id": f"st{i}", "text": t, "lang": ctx.cfg.project.tgt_lang, "seed": 1, "out": str(out_dir / "selftest" / f"s{i}.wav")} for i, t in enumerate(sentences)]
        st_ok = []
        try:
            lb = ctx.models.acquire(rc.backend, {"lora": res["adapter"]})
            st_ok = [r["id"] for r in gu.run_batched(lb, "synthesize", st_items) if not r.get("error")]
        except Exception as e:
            ctx.log.warning("%s: LoRA selftest failed: %s", sid, e)
        meta = {
            "adapter": ctx.paths.rel(res["adapter"]),
            "backend": rc.backend,
            "minutes": round(total / 60.0, 2),
            "n_pairs": len(recs),
            "selftest": [ctx.paths.rel(it["out"]) for it in st_items if it["id"] in st_ok],
        }
        write_json(out_dir / "meta.json", meta)
        chars[sid].lora = meta
        cache.put(f"lora:{sid}", key)
        n += 1
        backend = None  # selftest acquired another instance; re-acquire for the next character
    if reasons:
        summary["lora_skipped"] = reasons
    return n

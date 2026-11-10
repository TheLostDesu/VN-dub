"""voicebank: per-character reference clips, voice designs, performer voice, RVC model, LoRA.

Order of work (one model resident at a time):

1. refs (no model): 5-10 clean clips covering the character's top emotions
   -> ``voicebank/<sid>/refs/*.wav`` + ``refs.json`` + ``characters.yaml refs``.
2. voice designs (design backend, all characters): ``n_designs`` voices x ``test_sentences`` clips
   (the first ``performer.sample_lines`` are the character's own translated lines). Every design is
   kept in ``voicebank/<sid>/designs.json`` (see :mod:`vndub.stages.designs`); optionally a ``clone``
   design (route A) when ``performer.offer_clone``. Characters without original audio (mode B, only
   with ``synth.voice_unvoiced``) get designs from the character sheet: gender, age, names,
   ``Character.description`` and the tuning's character note.
3. speaker embedder (all characters): refs + clean clips (character centroid, EER calibration)
   + design clips (similarity).
4. F0 of designs (f0 backend / core pYIN), ranking (:func:`vndub.stages.designs.rank_score`).
5. the performer: the design the user chose (``characters.yaml performer.design_id``) or the
   top-ranked one -> ``performer.yaml`` (+ emotion variants from the design backend if it takes style
   instructions or knobs).
6. RVC training per character with enough clean audio (rvc backend; not for routes A / P).
7. route C (optional): LoRA on route-B winners that passed the gates.

The casting screen drives the same code through the public functions at the end of this module
(:func:`list_designs`, :func:`propose_designs`, :func:`add_reference_design`, :func:`import_design`,
:func:`choose_design`, :func:`auto_cast`, :func:`sample_lines`, :func:`train_rvc`, :func:`set_tuning`).
"""

from __future__ import annotations

import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vndub.backends.base import BackendUnavailable
from vndub.backends.score.speaker_sim import calibrate_eer, centroid, cosine
from vndub.manifest.io import Character, LineRecord, RefClip, read_json, save_characters, write_json, write_yaml
from vndub.stages import designs as dz
from vndub.stages import genutil as gu
from vndub.stages.common import StageError, counter
from vndub.stages.designs import TEST_SENTENCES  # noqa: F401  (re-exported: tests and docs refer to it here)
from vndub.util import hash_obj
from vndub.text.style_prompt import StyleSpec, design_words, normalize_tuning, render

NAME = "voicebank"
VERSION = "1"
CONFIG_SECTIONS = ("voicebank", "score", "project")

_SENTENCE_END = re.compile(r"[。！？!?…」』\.]\s*$")
N_CLIPS_FOR_CENTROID = 10
HARD_MIN_SEC = 1.5  # select_refs: the limits that hold even when the usual criteria are relaxed
HARD_MAX_SEC = 15.0
MAX_PROPOSE = 8


# ----------------------------------------------------------------------------
# refs
# ----------------------------------------------------------------------------


def voicebank_hash(cfg: Any) -> str:
    """What the voicebank reads from the config: its own section, ``project`` without the folders, the speaker embedder (and its
    options) and the F0 backend. The rest of ``score`` (gates, weights, regen limits, the CER ASR) is not an input of a reference, a
    design or an RVC model: it used to be in the key, so a gate tweak after casting re-rendered the designs of every uncast
    character and re-trained every RVC model."""
    dump = cfg.model_dump(mode="json", by_alias=True)
    project = {k: v for k, v in (dump.get("project") or {}).items() if k not in gu.PROJECT_PATH_KEYS}
    score = dump.get("score") or {}
    return hash_obj({
        "voicebank": dump.get("voicebank"), "project": project,
        "score": {"speaker": score.get("speaker"), "speaker_extra": score.get("speaker_extra")},
        "analyze": {"f0": (dump.get("analyze") or {}).get("f0")},
    })


def _cache(ctx: Any) -> Any:
    """The stage cache keyed by :func:`voicebank_hash`; entries stored under the older keys (the whole ``score`` section, with and
    without the folders) are still honoured and migrated, so a 1.1.0 project keeps its designs and trained RVC models."""
    cache = gu.stage_cache(ctx, NAME, VERSION, "voicebank", "score", "project")
    cache.rekey(voicebank_hash(ctx.cfg))
    return cache


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
            bgm = bool(p.get("bgm"))
            in_window = min_sec <= dur <= max_sec
            reasons: list[str] = []
            if not in_window:
                reasons.append(f"duration {dur:.1f} s outside {min_sec:g}-{max_sec:g} s")
            if snr is not None and float(snr) < min_snr_db:
                reasons.append(f"snr {float(snr):.0f} dB < {min_snr_db:g}")
            # snr None = no noise floor to measure (a tightly trimmed clip): unknown, not bad (and not the old 40 dB "perfect")
            if strict and (bgm or reasons):
                continue
            if not strict and (bgm or not HARD_MIN_SEC <= dur <= HARD_MAX_SEC):
                continue  # music under the voice, a half-second "eh?" or a monologue is never a reference, however few clips there are
            full = bool(_SENTENCE_END.search(l.text_src or ""))
            adv = (p.get("emotion") or {}).get("adv") or {}
            # prefer clean, full sentences, durations inside the window
            score = float(snr if snr is not None else min_snr_db) + (3.0 if full else 0.0) + (5.0 if in_window else 0.0) - (10.0 if bgm else 0.0)
            out.append(
                {
                    "line": l,
                    "emotion": _emotion_of(p),
                    "snr": None if snr is None else float(snr),
                    "duration": dur,
                    "arousal": adv.get("arousal"),
                    "valence": adv.get("valence"),
                    "score": score,
                    "bgm": bgm,
                    "relaxed_reasons": reasons,
                }
            )
        return out

    def round_robin(pool: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        hist = counter([c["emotion"] for c in pool])
        by_emo: dict[str, list[dict[str, Any]]] = {}
        for c in sorted(pool, key=lambda c: -c["score"]):
            by_emo.setdefault(c["emotion"], []).append(c)
        picked: list[dict[str, Any]] = []
        for rnd in range(per_emotion):  # round-robin over emotions ordered by frequency
            for emo in hist:
                lst = by_emo.get(emo, [])
                if rnd < len(lst) and len(picked) < limit:
                    picked.append(lst[rnd])
        return picked

    pool = cands(strict=True)
    relaxed = len(pool) < min_refs
    chosen = round_robin(pool, max_total)
    if relaxed:
        # strict clips first, then the best relaxed ones up to min_refs (not every clip of the character, as the old fallback did)
        extra = [c for c in cands(strict=False) if c["line"].line_id not in {x["line"].line_id for x in chosen}]
        chosen += round_robin(extra, max(0, min(min_refs, max_total) - len(chosen)))
        pool = [*pool, *extra]
    if not pool:
        return [], relaxed
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


def _forget_vectors(ctx: Any, sid: str, *globs: tuple[str, str]) -> None:
    """Delete cached embeddings / F0 of clips that are about to be replaced. They are keyed by file stem
    (``neutral_01``), so a new clip under an old name would otherwise be scored with the old clip's vector."""
    root = ctx.paths.vb(sid)
    for sub, pattern in globs:
        for p in (root / sub).glob(pattern):
            try:
                p.unlink()
            except OSError:
                pass


def _write_refs(ctx: Any, sid: str, chosen: list[dict[str, Any]]) -> list[RefClip]:
    d = ctx.paths.vb_refs(sid)
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    _forget_vectors(ctx, sid, ("emb", "ref_*.npy"), ("feat", "*.f0.npy"))  # the clips below are different audio under the same names
    refs: list[RefClip] = []
    n_by: dict[str, int] = {}
    for c in chosen:
        l: LineRecord = c["line"]
        n_by[c["emotion"]] = n_by.get(c["emotion"], 0) + 1
        dst = d / f"{c['emotion']}_{n_by[c['emotion']]:02d}.wav"
        src = ctx.paths.abs(l.src_wav)
        shutil.copyfile(src, dst)  # not copy2: that keeps the source's older mtime and the embedding cache (npy newer than wav) would be reused
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
                bgm=bool(c.get("bgm")),
                relaxed=bool(c.get("relaxed_reasons")),
                relaxed_reasons=list(c.get("relaxed_reasons") or []),
            )
        )
    write_json(ctx.paths.vb_refs_json(sid), [r.model_dump(mode="json") for r in refs])
    return refs


def _passport_inputs(p: dict[str, Any] | None) -> list[Any]:
    """Everything :func:`select_refs` and :func:`_write_refs` read from a passport (all of it is part of the refs cache key:
    after analyze is run again, e.g. with the ADV model newly available, refs.json must not keep the old emotion or ``arousal: null``)."""
    p = p or {}
    adv = (p.get("emotion") or {}).get("adv") or {}
    return [p.get("snr_db"), p.get("duration_s"), bool(p.get("bgm")), _emotion_of(p) if p else None, adv.get("arousal"), adv.get("valence")]


def _ensure_refs(ctx: Any, cache: Any, sid: str, ls: list[LineRecord], passports: dict[str, dict[str, Any]], chars: dict[str, Character]) -> list[RefClip]:
    """Reference clips of one character (cached by the clips, their passport inputs and the refs config)."""
    vb = ctx.cfg.voicebank
    # the effective source text is part of the key: refs.json carries it (timbre_ref_text), and a set_src_text
    # override / ASR-recovered text changes it without touching any audio
    key = cache.key("refs", "r18", [(l.line_id, l.src_wav) for l in ls], [_passport_inputs(passports.get(l.line_id)) for l in ls], vb.refs.model_dump(), [l.text_src for l in ls])
    if cache.fresh(f"refs:{sid}", key, [ctx.paths.vb_refs_json(sid)]):
        refs = gu.load_refs(ctx, sid)
    else:
        # a line without a passport (analyze not run, or run on a subset) is still a candidate by its length: select_refs
        # skips lines it knows nothing about, which left no refs at all while the log said "refs use durations only"
        passports = {**{l.line_id: {"duration_s": gu.src_duration(ctx, l)} for l in ls if l.line_id not in passports}, **passports}
        chosen, relaxed = select_refs(
            ls, passports,
            min_snr_db=vb.refs.min_snr_db, min_sec=vb.refs.min_sec, max_sec=vb.refs.max_sec,
            per_emotion=vb.refs.per_emotion, max_total=vb.refs.max_total, min_refs=vb.refs.min_refs,
        )
        if relaxed:
            ctx.log.warning("%s: fewer than %d clips meet the ref criteria; relaxed SNR/duration filters (never music, under %g s or over %g s)", sid, vb.refs.min_refs, HARD_MIN_SEC, HARD_MAX_SEC)
        if not chosen:
            ctx.log.warning("%s: no usable reference clip (every clip has music under it or is outside %g-%g s)", sid, HARD_MIN_SEC, HARD_MAX_SEC)
        refs = _write_refs(ctx, sid, chosen)
        cache.put(f"refs:{sid}", key)
    chars[sid].refs = [r.path for r in refs]
    return refs


def _char_f0(ctx: Any, chars: dict[str, Character], sid: str, refs: list[RefClip]) -> float | None:
    """Character F0: analyze's stats, else core pYIN over the reference clips."""
    f0 = chars[sid].stats.f0_median_hz
    if not f0 and refs:
        res = gu.f0_of(ctx, None, [(r.path, ctx.paths.abs(r.path), ctx.paths.vb(sid) / "feat" / f"{Path(r.path).stem}.f0.npy") for r in refs])
        vals = [m for m in (gu.median_f0(v) for v in res.values()) if m]
        f0 = float(np.median(vals)) if vals else None
    return f0


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


def design_prompt(char: Character, lang: str, description: str | None = None, index: int = 0) -> str:
    """Voice-design prompt: the character sheet (gender, age, pitch, emotion profile), the free-text
    description, the tuning's timbre preset / character note, and a small variation hint per design
    index (the first design of a character has none)."""
    parts = [performer_description(char, lang)]
    extra = (char.description if description is None else description or "").strip()
    if extra:
        parts.append(f"Character: {extra.rstrip('.')}.")
    words = design_words((char.performer or {}).get("tuning"))
    if words:
        parts.append(f"{words}.")
    hint = dz.VARIANT_HINTS[index % len(dz.VARIANT_HINTS)]
    if hint:
        parts.append(f"Voice: {hint}.")
    return " ".join(parts)


# ----------------------------------------------------------------------------
# progress
# ----------------------------------------------------------------------------


class _Ticker:
    """Monotonic progress over the stage's work units (``ctx.progress(done, total, item)``)."""

    def __init__(self, ctx: Any, total: int) -> None:
        self.ctx, self.total, self.done = ctx, max(1, int(total)), 0

    def __call__(self, item: str | None = None) -> None:
        self.done = min(self.total, self.done + 1)
        self.ctx.progress(self.done, self.total, item)


# ----------------------------------------------------------------------------
# rendering designs
# ----------------------------------------------------------------------------


@dataclass
class _DesignJob:
    sid: str
    design_id: str
    index: int
    seed: int
    prompt: str
    description: str
    audition: list[dict[str, str]]
    extra: list[dict[str, str]]
    knobs: dict[str, Any] | None = None
    origin: str = "stage"


def _f0_hint(char: Character, char_f0: float | None) -> float:
    if char_f0:
        return float(char_f0)
    return {"f": 230.0, "m": 120.0}.get(char.gender, 200.0)


def _plan_designs(
    ctx: Any, sid: str, char: Character, existing: list[dict[str, Any]], n: int, char_f0: float | None,
    description: str | None = None, line_ids: list[str] | None = None, origin: str = "stage", audition_only: bool = False,
) -> list[_DesignJob]:
    """Jobs for ``n`` new voice designs of one character (ids and seeds continue after ``existing``)."""
    vb = ctx.cfg.voicebank.performer
    lang = ctx.cfg.project.tgt_lang
    total = max(1, vb.test_sentences)
    n_aud = min(max(0, vb.sample_lines), total) or min(1, total)
    audition = dz.sample_items(ctx, sid, n_aud, line_ids)
    # audition_only: casting of a character with nothing to compare with renders just the clips the user listens to
    extra = [] if audition_only else dz.test_sentence_items(lang, total - len(audition), skip={a["text"] for a in audition})
    prior = [d for d in existing if d.get("kind") == "voice_design" and d.get("design_index") is not None]
    base = max([int(d["design_index"]) for d in prior], default=-1) + 1
    jobs: list[_DesignJob] = []
    taken = list(existing)
    for j in range(n):
        idx = base + j
        if base == 0:
            factor = 1.0 if n <= 1 else 0.85 + 0.3 * j / (n - 1)
        else:  # later batches: low-discrepancy spread so local (mock / SAPI) voices differ too
            factor = 0.85 + 0.3 * ((idx * 0.6180339887) % 1.0)
        did = dz.next_design_id(ctx, sid, taken, "vd")
        taken.append({"design_id": did})
        jobs.append(
            _DesignJob(
                sid=sid, design_id=did, index=idx, seed=ctx.cfg.synth.seed_base + gu.stable_seed("performer", sid, idx),
                prompt=design_prompt(char, lang, description, idx),
                description=(char.description if description is None else description or "").strip(),
                audition=audition, extra=extra, knobs={"f0_hz": _f0_hint(char, char_f0) * factor, "gender": char.gender}, origin=origin,
            )
        )
    return jobs


def _render_designs(ctx: Any, backend: Any, jobs: list[_DesignJob], tick: Any = None) -> list[dict[str, Any]]:
    """Render the clips of each job (one design backend call per design); returns the new Design dicts."""
    lang = ctx.cfg.project.tgt_lang
    spec = backend.spec
    out: list[dict[str, Any]] = []
    for job in jobs:
        ddir = dz.design_dir(ctx, job.sid, job.design_id)
        shutil.rmtree(ddir, ignore_errors=True)
        items, meta = [], {}
        for tag, group in (("s", job.audition), ("r", job.extra)):
            for t, row in enumerate(group):
                iid = f"{job.sid}|{job.design_id}|{tag}{t}"
                it = {"id": iid, "text": row["text"], "description": job.prompt, "instruct": job.prompt, "lang": lang, "seed": job.seed, "out": str(ddir / f"{tag}{t}.wav")}
                if spec.env is None and job.knobs:  # local backends (mock / SAPI) take pitch hints
                    it["knobs"] = dict(job.knobs)
                items.append(it)
                meta[iid] = (tag, row)
        results = gu.run_batched(backend, "design", items, chunk=max(1, len(items)))
        samples, refs_extra = [], []
        for it, r in zip(items, results):
            tag, row = meta[it["id"]]
            wav = Path(it["out"])
            if r.get("error") or not wav.exists():
                ctx.log.warning("design %s failed: %s", it["id"], r.get("error", "no audio"))
                continue
            if tag == "s":
                samples.append({"line_id": row["line_id"], "wav": ctx.paths.rel(wav), "text": row["text"]})
            else:
                refs_extra.append({"wav": ctx.paths.rel(wav), "text": row["text"]})
        if tick is not None:
            tick(job.design_id)
        if not samples and not refs_extra:
            shutil.rmtree(ddir, ignore_errors=True)
            continue
        out.append(
            {
                "design_id": job.design_id, "kind": "voice_design", "backend": spec.name, "prompt": job.prompt,
                "description": job.description, "samples": samples, "ref_clips": refs_extra,
                "spk_sim": None, "f0_median_hz": None, "f0_delta_st": None, "rank_score": -1.0,
                "license": spec.license, "commercial_ok": bool(spec.commercial_ok), "created": gu.ts_now(),
                "origin": job.origin, "seed": job.seed, "design_index": job.index,
            }
        )
    return out


def _tts_backend(ctx: Any) -> tuple[Any, str, dict[str, Any]] | None:
    """The first usable synth TTS backend (the one that clones a reference): (handle, name, options)."""
    for name in ctx.cfg.enabled_backends():
        opts = gu.backend_options(ctx.cfg, name)
        b, used, _ = gu.acquire_first(ctx, [name], opts or None, group=f"TTS backend {name}", stage=NAME)
        if b is not None:
            return b, used or name, opts
    return None


def _render_cloned(
    ctx: Any, sid: str, design_id: str, kind: str, timbre_ref: Path, timbre_text: str | None, audition: list[dict[str, str]],
    char: Character, char_f0: float | None, cross_lingual: bool, origin: str, extra: dict[str, Any] | None = None, tick: Any = None,
) -> dict[str, Any] | None:
    """A design whose audition clips are the audition lines spoken by a zero-shot clone of ``timbre_ref``."""
    got = _tts_backend(ctx)
    if got is None:
        ctx.log.warning("%s: no usable TTS backend for the %s design", sid, kind)
        return None
    backend, bname, bopts = got
    caps = backend.spec.caps or {}
    lang = ctx.cfg.project.tgt_lang
    spec = StyleSpec()
    rendered = render(spec, caps, bopts, cross_lingual=cross_lingual)
    ddir = dz.design_dir(ctx, sid, design_id)
    ddir.mkdir(parents=True, exist_ok=True)
    items = []
    for t, row in enumerate(audition):
        text = (rendered["text_prefix"] + " " + row["text"]).strip() if rendered["text_prefix"] else row["text"]
        knobs = dict(rendered["knobs"] or {})
        if backend.spec.env is None:
            knobs.setdefault("f0_hz", _f0_hint(char, char_f0))
            knobs.setdefault("gender", char.gender)
        items.append(
            {
                "id": f"{sid}|{design_id}|s{t}", "text": text, "lang": lang, "timbre_ref": str(timbre_ref), "timbre_ref_text": timbre_text,
                "emotion_ref": str(timbre_ref), "instruct": rendered["instruct"], "knobs": knobs,
                "seed": ctx.cfg.synth.seed_base + gu.stable_seed("design", sid, design_id, t), "target_duration": None, "out": str(ddir / f"s{t}.wav"),
            }
        )
    results = gu.run_batched(backend, "synthesize", items, chunk=max(1, len(items)))
    samples = []
    for it, r, row in zip(items, results, audition):
        if r.get("error") or not Path(it["out"]).exists():
            ctx.log.warning("%s design %s failed: %s", kind, it["id"], r.get("error", "no audio"))
            continue
        samples.append({"line_id": row["line_id"], "wav": ctx.paths.rel(it["out"]), "text": row["text"]})
    if tick is not None:
        tick(design_id)
    if not samples:
        return None
    d = {
        "design_id": design_id, "kind": kind, "backend": bname, "prompt": "", "description": "", "samples": samples, "ref_clips": [],
        "spk_sim": None, "f0_median_hz": None, "f0_delta_st": None, "rank_score": -1.0,
        "license": backend.spec.license, "commercial_ok": bool(backend.spec.commercial_ok), "created": gu.ts_now(), "origin": origin,
    }
    d.update(extra or {})
    return d


def _clone_design(ctx: Any, sid: str, char: Character, refs: list[RefClip], char_f0: float | None, audition: list[dict[str, str]], origin: str, tick: Any = None) -> dict[str, Any] | None:
    """The route-A design: the audition lines in a zero-shot clone of the character's best timbre reference."""
    refs = [r for r in refs if not (r.model_extra or {}).get("bgm")]  # a clone repeats the music under its reference
    pool = [r for r in refs if r.usage in ("timbre", "both")] or refs
    if not pool:
        return None
    best = max(pool, key=lambda r: float(r.snr or 0.0))
    return _render_cloned(
        ctx, sid, "clone", "clone", ctx.paths.abs(best.path), best.text or None, audition, char, char_f0, cross_lingual=True, origin=origin,
        extra={"ref_line_id": best.line_id}, tick=tick,
    )


# ----------------------------------------------------------------------------
# scoring designs
# ----------------------------------------------------------------------------


def _ref_clip_jobs(ctx: Any, sid: str, refs: list[RefClip], ls: list[LineRecord], passports: dict[str, dict[str, Any]], emb_name: str) -> list[tuple[str, Path, Path]]:
    """Embedding jobs of the character itself: reference clips + the cleanest original clips."""
    edir = ctx.paths.vb(sid) / "emb"
    jobs = [(f"ref|{sid}|{r.path}", ctx.paths.abs(r.path), edir / f"ref_{Path(r.path).stem}.{emb_name}.npy") for r in refs]
    clean = sorted((l for l in ls if l.src_wav16 or l.src_wav), key=lambda l: -float((passports.get(l.line_id) or {}).get("snr_db") or 0.0))[:N_CLIPS_FOR_CENTROID]
    for l in clean:
        jobs.append((f"clip|{sid}|{l.line_id}", ctx.paths.abs(l.src_wav16 or l.src_wav), edir / f"clip_{l.line_id}.{emb_name}.npy"))
    return jobs


def _design_clip_jobs(ctx: Any, sid: str, ds: list[dict[str, Any]], emb_name: str) -> list[tuple[str, Path, Path]]:
    jobs = []
    for d in ds:
        for p in dz.clip_paths(ctx, d):
            if p.exists():
                jobs.append((f"design|{sid}|{d['design_id']}|{p.name}", p, p.with_suffix(f".{emb_name}.npy")))
    return jobs


def _score_designs(
    ctx: Any, sid: str, ds: list[dict[str, Any]], char: Character, char_f0: float | None, vecs: dict[str, np.ndarray],
    char_centroid: np.ndarray | None, emb_name: str | None, f0_backend: Any,
) -> None:
    """Fill spk_sim / F0 / rank_score of ``ds`` in place (``vecs``: embedding vectors by job key)."""
    max_delta = ctx.cfg.voicebank.performer.max_f0_delta_semitones
    # without a character F0 there is nothing to compare a design's pitch with: skip the (slow) pYIN over every clip
    jobs = [(f"{d['design_id']}|{p.name}", p, p.with_suffix(".f0.npy")) for d in ds for p in dz.clip_paths(ctx, d) if p.exists()] if char_f0 else []
    f0s = gu.f0_of(ctx, f0_backend, jobs)
    for d in ds:
        clips = [p for p in dz.clip_paths(ctx, d) if p.exists()]
        meds = [m for m in (gu.median_f0(f0s.get(f"{d['design_id']}|{p.name}")) for p in clips) if m]
        d["f0_median_hz"] = float(np.median(meds)) if meds else None
        d["f0_delta_st"] = gu.semitone_delta(d["f0_median_hz"], char_f0)
        dv = [vecs[k] for k in (f"design|{sid}|{d['design_id']}|{p.name}" for p in clips) if k in vecs]
        d["spk_sim"] = cosine(centroid(dv), char_centroid) if dv and char_centroid is not None else None
        if emb_name:
            d["embedder"] = emb_name
        d["rank_score"] = dz.rank_score(d["spk_sim"], d["f0_delta_st"], max_delta)


def _measure_new_designs(ctx: Any, sid: str, char: Character, new: list[dict[str, Any]], has_audio: bool) -> None:
    """API path (casting screen): embed + F0 + rank the designs just created for one character."""
    emb_backend, emb_name, emb_reason = gu.acquire_first(ctx, [ctx.cfg.score.speaker], group="speaker embedder", stage=NAME) if has_audio else (None, None, "no original audio")
    centroid_vec = None
    vecs: dict[str, np.ndarray] = {}
    char_f0 = char.stats.f0_median_hz
    if emb_backend is not None:
        jobs: list[tuple[str, Path, Path]] = []
        ref_keys: list[str] = []
        if has_audio:
            ls = [l for l in ctx.voiced_lines(apply_only=False) if l.speaker_id == sid]
            passports = {l.line_id: p for l in ls if (p := gu.load_passport(ctx, l.line_id))}
            refs = gu.load_refs(ctx, sid)
            if not refs and ls:
                cache = _cache(ctx)
                chars = {sid: char}
                refs = _ensure_refs(ctx, cache, sid, ls, passports, chars)
                cache.save()
            char_f0 = char_f0 or _char_f0(ctx, {sid: char}, sid, refs)
            jobs = _ref_clip_jobs(ctx, sid, refs, ls, passports, emb_name)
            ref_keys = [j[0] for j in jobs]
        jobs += _design_clip_jobs(ctx, sid, new, emb_name)
        vecs = gu.embed_paths(ctx, emb_backend, jobs)
        own = [vecs[k] for k in ref_keys if k in vecs]
        centroid_vec = centroid(own) if own else None
    elif has_audio:
        ctx.log.warning("speaker embedder unavailable (%s): designs are ranked by F0 only", emb_reason)
    # the F0 of a design only matters against a character's F0: without original audio the cheap core pYIN is enough
    f0_backend = gu.acquire_first(ctx, [ctx.cfg.analyze.f0], group="f0 backend", stage=NAME)[0] if has_audio else None
    _score_designs(ctx, sid, new, char, char_f0 if has_audio else None, vecs, centroid_vec, emb_name, f0_backend)


# ----------------------------------------------------------------------------
# performer (what synth reads: performer.yaml) from a design
# ----------------------------------------------------------------------------


def _ref_emotion_from_name(stem: str) -> str:
    from vndub.text.style_prompt import ALL_EMOTIONS

    head = stem.split("_", 1)[0].lower()
    return head if head in ALL_EMOTIONS else "neutral"


def _write_performer(ctx: Any, sid: str, d: dict[str, Any], chars: dict[str, Character], chosen_by: str) -> dict[str, Any]:
    """performer.yaml + performer/*.wav from a design, and the character's ``performer`` record."""
    c = chars[sid]
    pdir = ctx.paths.vb(sid) / "performer"
    pdir.mkdir(parents=True, exist_ok=True)
    for p in pdir.glob("*.wav"):  # the previous design's clips and emotion variants
        p.unlink()
    _forget_vectors(ctx, sid, ("emb", "perf_*.npy"))  # keyed by stem too: the new neutral_01 is another voice
    refs = []
    for i, clip in enumerate(dz.performer_clips(ctx, d)):
        dst = pdir / f"neutral_{i + 1:02d}.wav"
        shutil.copyfile(clip["src"], dst)  # fresh mtime (see _write_refs)
        refs.append({"path": ctx.paths.rel(dst), "emotion": clip.get("emotion", "neutral"), "text": clip["text"]})
    for v in d.get("variants") or []:
        src = ctx.paths.abs(v["wav"])
        if src.exists():
            dst = pdir / f"{v['emotion']}_01.wav"
            shutil.copyfile(src, dst)
            refs.append({"path": ctx.paths.rel(dst), "emotion": v["emotion"], "text": v.get("text", "")})
    f0, sim, delta = d.get("f0_median_hz"), d.get("spk_sim"), d.get("f0_delta_st")
    max_delta = ctx.cfg.voicebank.performer.max_f0_delta_semitones
    perf = {
        "kind": "reference" if d.get("kind") == "reference" else "voice_design",
        "backend": d.get("backend", ""),
        "description": d.get("prompt") or d.get("description") or "",
        "seed": d.get("seed"),
        "design_index": d.get("design_index"),
        "design_id": d["design_id"],
        "refs": refs,
        "f0_median_hz": None if f0 is None else round(float(f0), 2),
        "spk_sim": None if sim is None else round(float(sim), 4),
        "f0_delta_st": None if delta is None else round(float(delta), 2),
        "embedder": d.get("embedder"),
        "warnings": ["f0 outside window"] if delta is not None and abs(delta) > max_delta else [],
        "source": chosen_by,
        "variants_done": "variants" in d,
    }
    write_yaml(ctx.paths.vb_performer(sid), perf)
    tuning = (c.performer or {}).get("tuning")
    c.performer = {k: perf[k] for k in ("kind", "backend", "description", "f0_median_hz", "spk_sim", "f0_delta_st")}
    c.performer.update(path=ctx.paths.rel(ctx.paths.vb_performer(sid)), design_id=d["design_id"], chosen_by=chosen_by)
    if tuning:
        c.performer["tuning"] = tuning
    return perf


def _performer_key(cache: Any, sid: str, vb: Any, char_f0: dict[str, Any], chars: dict[str, Character]) -> str:
    c = chars[sid]
    parts: list[Any] = [vb.performer.model_dump(), char_f0.get(sid), c.gender, c.age_hint, c.stats.emotion_hist]
    words = design_words((c.performer or {}).get("tuning"))
    if c.description:
        parts.append(c.description)
    if words:
        parts.append(words)
    return cache.key("performer", *parts)


def _performer_refs_ok(ctx: Any, perf: dict[str, Any] | None) -> bool:
    return bool(perf) and bool(perf.get("refs")) and all(ctx.paths.abs(r["path"]).exists() for r in perf["refs"])


def _performer_state(ctx: Any, cache: Any, sid: str, vb: Any, char_f0: dict[str, Any], chars: dict[str, Character]) -> str:
    """``fresh`` | ``write`` (rewrite performer.yaml from the chosen design) | ``variants`` | ``regen`` (new designs).

    A design the user chose is never replaced by a config change: it is only (re)written from its design
    entry and given emotion variants when it has none. A stage-picked performer is recomputed when its key changes.
    """
    perf = gu.load_performer(ctx, sid)
    ch = chars[sid].performer or {}
    if ch.get("chosen_by") == "user" and ch.get("design_id"):
        d = dz.get_design(ctx, sid, ch["design_id"])
        if d is not None:
            if d["kind"] == "clone":
                return "fresh"  # route A: the performer is not used
            if not (_performer_refs_ok(ctx, perf) and perf.get("design_id") == d["design_id"]):
                return "write"
            if d["kind"] == "voice_design" and not perf.get("variants_done"):
                return "variants"
            return "fresh"
    if not _performer_refs_ok(ctx, perf):
        return "regen"
    return "fresh" if cache.fresh(f"performer:{sid}", _performer_key(cache, sid, vb, char_f0, chars), [ctx.paths.vb_performer(sid)]) else "regen"


# ----------------------------------------------------------------------------
# stage
# ----------------------------------------------------------------------------


def run(ctx: Any) -> dict[str, Any]:
    vb = ctx.cfg.voicebank
    lang = ctx.cfg.project.tgt_lang
    voiced = ctx.voiced_lines(apply_only=False)
    unvoiced_on = bool(ctx.cfg.synth.voice_unvoiced)
    dub = ctx.dub_lines(apply_only=False) if unvoiced_on else voiced
    if not voiced and not any(not l.has_voice for l in dub):
        raise StageError("no voiced lines: run parse/extract (and analyze) first")
    chars = ctx.characters()
    chars_at_start = gu.snapshot_characters(chars)  # the final save changes only what this run changed (training takes hours)
    by_spk: dict[str, list[LineRecord]] = {}
    for l in voiced:
        if l.speaker_id == "narrator":
            continue
        by_spk.setdefault(l.speaker_id, []).append(l)
    unvoiced_raw: dict[str, str] = {}
    if unvoiced_on:
        for l in dub:
            if not l.has_voice and l.speaker_id != "narrator":
                unvoiced_raw.setdefault(l.speaker_id, l.speaker_raw or l.speaker_id)
    if ctx.only:
        sel = {s for s in by_spk if s in ctx.only}
        only_u = {s for s in unvoiced_raw if s in ctx.only}
        if sel or only_u:  # `--only <a speaker with unvoiced lines only>` used to leave every voiced speaker in
            by_spk = {s: by_spk[s] for s in sel}
            unvoiced_raw = {s: r for s, r in unvoiced_raw.items() if s in only_u}
    passports: dict[str, dict[str, Any]] = {}
    for spk_lines in by_spk.values():
        for l in spk_lines:
            p = gu.load_passport(ctx, l.line_id)
            if p:
                passports[l.line_id] = p
    if not passports and by_spk:
        ctx.log.warning("no passports found: refs use durations only (run analyze for emotion/SNR-aware selection)")

    cache = _cache(ctx)
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
    # mode B: characters with dub lines but no usable original audio (designed from the character sheet)
    mode_b = [sid for sid in unvoiced_raw if sid not in eligible]
    for sid in mode_b:
        chars.setdefault(sid, Character(names={"src": unvoiced_raw[sid], "tgt": ""}))
        if chars[sid].route is None:
            chars[sid].route = "P"
    if mode_b:
        summary["mode_b"] = mode_b
    perf_chars = [*eligible, *mode_b]
    n_rvc = len(eligible) if vb.rvc.enabled else 0
    tick = _Ticker(ctx, len(eligible) + len(perf_chars) + n_rvc)

    # 1. refs -------------------------------------------------------------------
    refs: dict[str, list[RefClip]] = {}
    for sid, ls in eligible.items():
        refs[sid] = _ensure_refs(ctx, cache, sid, ls, passports, chars)
        summary["refs"] += len(refs[sid])
        if any((r.model_extra or {}).get("relaxed") for r in refs[sid]):
            summary.setdefault("refs_relaxed", {})[sid] = sorted({why for r in refs[sid] for why in (r.model_extra or {}).get("relaxed_reasons") or []})[:3]
        tick(sid)

    char_f0: dict[str, float | None] = {sid: _char_f0(ctx, chars, sid, refs[sid]) for sid in eligible}
    for sid in mode_b:
        char_f0[sid] = chars[sid].stats.f0_median_hz

    # 2. voice designs ----------------------------------------------------------
    perf_kind = vb.performer.kind
    state = {sid: _performer_state(ctx, cache, sid, vb, char_f0, chars) for sid in perf_chars}
    regen = [sid for sid in perf_chars if state[sid] == "regen"]
    performer_reason: dict[str, str] = {}
    design_backend_name = None
    design_caps: dict[str, Any] = {}
    kept: dict[str, list[dict[str, Any]]] = {}  # sid -> designs that stay (user's, imported, the chosen one)
    new: dict[str, list[dict[str, Any]]] = {}  # sid -> designs rendered in this run
    for sid in regen:
        chosen_id = (chars[sid].performer or {}).get("design_id") if (chars[sid].performer or {}).get("chosen_by") == "user" else None
        old = dz.load_designs(ctx, sid)
        kept[sid] = [d for d in old if d.get("origin") != "stage" or d["design_id"] == chosen_id]
        for d in old:
            if d not in kept[sid]:
                dz.remove_design_files(ctx, sid, d["design_id"])
        new[sid] = []
    if perf_kind == "voice_design" and regen:
        backend, design_backend_name, reason = gu.acquire_first(ctx, [vb.performer.backend], group="performer design backend", stage=NAME)
        if backend is None:
            for sid in regen:
                performer_reason[sid] = f"design backend unavailable: {reason}"
        else:
            design_caps = dict(backend.spec.caps)
            for sid in regen:
                # without original audio nothing ranks the designs (every rank_score is -1) and the stage keeps the first one: the
                # other designs and their clips (3 of 4, 15 of 20 by default) were rendered for nothing; "generate more" in the app adds them
                n_new = max(1, vb.performer.n_designs) if sid in eligible else 1
                jobs = _plan_designs(ctx, sid, chars[sid], kept[sid], n_new, char_f0.get(sid))
                new[sid] = _render_designs(ctx, backend, jobs)
                if not new[sid]:
                    performer_reason[sid] = "all performer designs failed"
    elif perf_kind == "reference" and regen:
        for sid in regen:
            src_dir = Path(ctx.cfg.resolve(vb.performer.reference_dir)) / sid if vb.performer.reference_dir else None
            clips = sorted(src_dir.glob("*.wav")) if src_dir and src_dir.exists() else []
            if not clips:
                performer_reason[sid] = f"no native reference clips in {src_dir}"
                continue
            did = dz.next_design_id(ctx, sid, kept[sid], "ref")
            ddir = dz.design_dir(ctx, sid, did)
            samples = []
            for i, p in enumerate(clips):
                dst = ddir / f"s{i}.wav"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst)
                samples.append({"line_id": "", "wav": ctx.paths.rel(dst), "text": "", "emotion": _ref_emotion_from_name(p.stem)})
            new[sid] = [
                {
                    "design_id": did, "kind": "reference", "backend": "reference", "prompt": "", "description": f"reference clips from {src_dir}",
                    "samples": samples, "ref_clips": [], "spk_sim": None, "f0_median_hz": None, "f0_delta_st": None, "rank_score": -1.0,
                    "license": "user-supplied", "commercial_ok": True, "created": gu.ts_now(), "origin": "stage", "seed": None, "design_index": 0,
                }
            ]
    if vb.performer.offer_clone and perf_kind == "voice_design":
        for sid in regen:
            if sid in eligible and not any(d["design_id"] == "clone" for d in kept[sid]) and sid not in performer_reason:
                aud = dz.sample_items(ctx, sid, max(1, vb.performer.sample_lines))
                clone = _clone_design(ctx, sid, chars[sid], refs[sid], char_f0.get(sid), aud, "stage")
                if clone is not None:
                    new[sid].append(clone)

    # 3. embeddings: refs + clean clips (centroid, calibration) + design clips ------
    # (characters without original audio have nothing to compare with: no embedder for them)
    emb_backend, emb_name, emb_reason = gu.acquire_first(ctx, [ctx.cfg.score.speaker], group="speaker embedder", stage=NAME) if eligible else (None, None, "no character with original audio")
    centroids: dict[str, np.ndarray] = {}
    ref_vecs: dict[str, list[np.ndarray]] = {}
    design_vecs: dict[str, dict[str, np.ndarray]] = {}
    if emb_backend is not None:
        for sid in eligible:
            jobs = _ref_clip_jobs(ctx, sid, refs[sid], eligible[sid], passports, emb_name)
            ref_keys = [j[0] for j in jobs]
            jobs += _design_clip_jobs(ctx, sid, new.get(sid, []), emb_name)
            vecs = gu.embed_paths(ctx, emb_backend, jobs)
            ref_vecs[sid] = [vecs[k] for k in ref_keys if k in vecs]
            if ref_vecs[sid]:
                centroids[sid] = centroid(ref_vecs[sid])
                np.save(ctx.paths.vb(sid) / f"centroid.{emb_name}.npy", centroids[sid])
            design_vecs[sid] = {k: v for k, v in vecs.items() if k.startswith("design|")}
        if vb.calibrate_spk:
            summary["calibration"] = _calibrate(ctx, emb_name, ref_vecs)
    elif eligible:
        ctx.log.warning("speaker embedder unavailable (%s): no calibration, performer chosen by F0 only", emb_reason)
        summary["calibration"] = f"unavailable: {emb_reason}"

    # 4. F0 + ranking + choice --------------------------------------------------
    f0_backend = None
    if any(new.get(sid) for sid in eligible):  # characters without audio use the cheap core pYIN
        f0_backend, _, _ = gu.acquire_first(ctx, [ctx.cfg.analyze.f0], group="f0 backend", stage=NAME)
    picked: dict[str, dict[str, Any]] = {}  # sid -> design the performer is written from
    for sid in regen:
        ds = new[sid]
        if ds:
            _score_designs(
                ctx, sid, ds, chars[sid], char_f0.get(sid) if sid in eligible else None, design_vecs.get(sid, {}), centroids.get(sid), emb_name, f0_backend,
            )
        allds = [*kept[sid], *ds]
        if not allds:
            if dz.load_designs(ctx, sid):  # the stage designs removed above must not stay listed with their files gone
                dz.save_designs(ctx, sid, [])
            tick(sid)
            continue
        dz.save_designs(ctx, sid, allds)
        chosen_id = (chars[sid].performer or {}).get("design_id") if (chars[sid].performer or {}).get("chosen_by") == "user" else None
        pool = [d for d in allds if d["kind"] != "clone"]
        pick = next((d for d in pool if d["design_id"] == chosen_id), None) if chosen_id else None
        if pick is None and pool:
            pick = max(pool, key=lambda d: d["rank_score"])  # max() keeps the first of equals: creation order
            max_delta = vb.performer.max_f0_delta_semitones
            if not any(d.get("f0_delta_st") is None or abs(d["f0_delta_st"]) <= max_delta for d in pool):
                ctx.log.warning("%s: no performer design within %.1f st of the character F0; using the closest overall", sid, max_delta)
        if pick is not None:
            picked[sid] = pick
        tick(sid)
    for sid in perf_chars:
        if state[sid] in ("write", "variants"):
            ch = chars[sid].performer or {}
            d = dz.get_design(ctx, sid, ch.get("design_id", ""))
            if d is not None:
                picked[sid] = d
            tick(sid)
        elif state[sid] == "fresh":
            tick(sid)

    # 5. emotion variants of the chosen designs, then performer.yaml -----------------
    variant_ids = [sid for sid, d in picked.items() if d["kind"] == "voice_design" and "variants" not in d]
    if variant_ids:
        if not design_caps:
            vb_backend, design_backend_name, _ = gu.acquire_first(ctx, [vb.performer.backend], group="performer design backend", stage=NAME)
            design_caps = dict(vb_backend.spec.caps) if vb_backend is not None else {}
        can_vary = design_caps.get("instruct", "none") != "none" or design_caps.get("knobs", "none") != "none"
        variant_items: list[dict[str, Any]] = []
        sentence0 = dz.TEST_SENTENCES.get(lang, dz.TEST_SENTENCES["en"])[0]
        for sid in variant_ids:
            best = picked[sid]
            if not design_backend_name:
                continue  # no design backend now: leave the variants undecided ("variants_done" false) so that a later run tries again
            best["variants"] = []  # a backend that takes no style instructions: no variants is the final answer
            if not can_vary:
                continue
            emos = [e for e in dict.fromkeys(r.emotion for r in refs.get(sid, [])) if e not in ("neutral", "other", "unknown")]
            vdir = dz.design_dir(ctx, sid, best["design_id"]) / "variants"
            for emo in emos[:5]:
                rendered = render(StyleSpec(emotion=emo, intensity=0.65), design_caps)
                instr = rendered.get("instruct") or rendered.get("style_prompt") or ""
                desc = f"{best['prompt']} {instr}".strip()
                it = {"id": f"{sid}|{emo}", "text": sentence0, "lang": lang, "seed": best.get("seed"), "out": str(vdir / f"{emo}_01.wav"), "description": desc, "instruct": desc}
                knobs = dict(rendered.get("knobs") or {})
                if design_backend_name and gu_is_local(ctx, design_backend_name):
                    knobs.update({"f0_hz": best.get("f0_median_hz") or char_f0.get(sid) or 200.0, "gender": chars[sid].gender})
                if knobs:
                    it["knobs"] = knobs
                variant_items.append(it)
        if variant_items:
            try:
                backend = ctx.models.acquire(design_backend_name)
                for r in gu.run_batched(backend, "design", variant_items, chunk=8):
                    sid, emo = r["id"].split("|", 1)
                    out = dz.design_dir(ctx, sid, picked[sid]["design_id"]) / "variants" / f"{emo}_01.wav"
                    if not r.get("error") and out.exists():
                        picked[sid]["variants"].append({"emotion": emo, "wav": ctx.paths.rel(out), "text": sentence0})
                    elif r.get("error"):
                        ctx.log.warning("performer variant %s failed: %s", r["id"], r["error"])
            except BackendUnavailable as e:
                ctx.log.warning("performer emotion variants skipped: %s", e)
                for sid in variant_ids:
                    picked[sid].pop("variants", None)  # not rendered: not "done" either
        for sid in variant_ids:  # persist the variants on the design entries
            if "variants" not in picked[sid]:
                continue
            allds = dz.load_designs(ctx, sid)
            for d in allds:
                if d["design_id"] == picked[sid]["design_id"]:
                    d["variants"] = picked[sid]["variants"]
            dz.save_designs(ctx, sid, allds)
    for sid, d in picked.items():
        if d["kind"] == "clone":
            continue
        chosen_by = "user" if (chars[sid].performer or {}).get("chosen_by") == "user" and (chars[sid].performer or {}).get("design_id") == d["design_id"] else "stage"
        _write_performer(ctx, sid, d, chars, chosen_by)
        if state[sid] == "regen":
            cache.put(f"performer:{sid}", _performer_key(cache, sid, vb, char_f0, chars))
    for sid in perf_chars:
        if ctx.paths.vb_performer(sid).exists():
            summary["performers"] += 1
            if not chars[sid].performer:
                perf = gu.load_performer(ctx, sid) or {}
                chars[sid].performer = {k: perf.get(k) for k in ("kind", "backend", "description", "f0_median_hz", "spk_sim", "f0_delta_st")}
    summary["designs"] = sum(len(dz.load_designs(ctx, sid)) for sid in perf_chars)
    if performer_reason:
        summary["performer_missing"] = performer_reason
        for sid, reason in performer_reason.items():
            ctx.log.warning("%s: no performer (%s): route B will be skipped for this character", sid, reason)

    # 6. RVC --------------------------------------------------------------------
    if vb.rvc.enabled:
        summary["rvc"] = _train_rvc(ctx, cache, eligible, passports, refs, chars, char_f0, summary, tick)

    # 7. route C ------------------------------------------------------------------
    if vb.route_c.enabled:
        summary["lora"] = _train_lora(ctx, cache, eligible, chars, summary)

    cache.save()
    gu.save_characters_merged(ctx, chars, chars_at_start)
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


# ----------------------------------------------------------------------------
# RVC (route B)
# ----------------------------------------------------------------------------


def _rvc_clips(ctx: Any, ls: list[LineRecord], passports: dict[str, dict[str, Any]]) -> tuple[list[str], str]:
    """Training clips of one character (emotion-coverage round robin up to ``max_minutes``) or ('', reason)."""
    rv = ctx.cfg.voicebank.rvc
    vbref = ctx.cfg.voicebank.refs
    clean = []
    for l in ls:
        p = passports.get(l.line_id) or {}
        snr = p.get("snr_db")
        if p.get("bgm") or (snr is not None and float(snr) < vbref.min_snr_db) or float(p.get("duration_s") or 0) < 1.0:  # snr None: unmeasured, not bad
            continue
        clean.append((l, p))
    mins = sum(float(p.get("duration_s") or 0) for _, p in clean) / 60.0
    if mins + 1e-9 < rv.min_minutes:
        return [], f"{mins:.1f} min clean audio < {rv.min_minutes} min"
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
    return picked, ""


def _rvc_f0_transpose(ctx: Any, sid: str, char_f0: float | None) -> int:
    perf = gu.load_performer(ctx, sid) or {}
    if char_f0 and perf.get("f0_median_hz"):
        return int(round(12 * math.log2(float(char_f0) / float(perf["f0_median_hz"]))))
    return 0


def _rvc_train_character(
    ctx: Any, cache: Any, sid: str, clips: list[str], chars: dict[str, Character], char_f0: float | None, holder: dict[str, Any], reasons: dict[str, str],
    progress: Any = None,
) -> str:
    """Train (or reuse) one character's RVC model. Returns ``done`` | ``cached`` | ``failed``."""
    rv = ctx.cfg.voicebank.rvc
    out_dir = ctx.paths.vb_rvc(sid)
    f0_t = _rvc_f0_transpose(ctx, sid, char_f0)
    key = cache.key("rvc", rv.model_dump(), sorted(clips))
    meta = read_json(out_dir / "meta.json")
    if meta and cache.fresh(f"rvc:{sid}", key, [out_dir / "meta.json"]) and ctx.paths.abs(meta.get("model", "missing")).exists():
        if meta.get("f0_transpose") != f0_t:
            meta["f0_transpose"] = f0_t
            write_json(out_dir / "meta.json", meta)
        chars[sid].rvc = meta
        return "cached"
    if holder.get("backend") is None:
        backend, _, reason = gu.acquire_first(ctx, [rv.backend], group="RVC training backend", stage=NAME)
        if backend is None:
            holder["unavailable"] = reason
            reasons[sid] = f"rvc backend unavailable: {reason}"
            return "failed"
        holder["backend"] = backend
    backend = holder["backend"]
    job = {
        "speaker_id": sid, "clips": clips, "out_dir": str(out_dir), "sample_rate": rv.sample_rate, "epochs": rv.epochs,
        "batch_size": rv.batch_size, "f0_method": rv.f0_method, "overfit_patience": rv.overfit_patience,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    if progress is not None:
        progress(1, 3, sid)
    try:
        # hours, not the per-item minutes: at the old 720 s the adapter was killed mid-training and the Applio
        # process it had started lived on, holding its VRAM next to the next model
        timeout_s = float((getattr(backend, "options", None) or {}).get("timeout_train_s", 6 * 3600))
        res = backend.train(job, timeout_s=timeout_s) or {}
    except BackendUnavailable:
        raise
    except Exception as e:
        reasons[sid] = f"rvc training failed: {type(e).__name__}: {e}"
        ctx.log.error("%s: %s", sid, reasons[sid])
        return "failed"
    if res.get("error") or not res.get("model"):
        reasons[sid] = f"rvc training failed: {res.get('error', 'no model returned')}"
        return "failed"
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
    return "done"


def _train_rvc(ctx: Any, cache: Any, eligible: dict[str, list[LineRecord]], passports: dict[str, dict[str, Any]], refs: dict[str, list[RefClip]], chars: dict[str, Character], char_f0: dict[str, Any], summary: dict[str, Any], tick: Any = None) -> int:
    plans: dict[str, list[str]] = {}
    reasons: dict[str, str] = {}
    for sid, ls in eligible.items():
        if chars[sid].route in ("A", "P"):  # no VC step on these routes: skip the training
            reasons[sid] = f"route {chars[sid].route}: no voice conversion"
            if tick is not None:
                tick(sid)
            continue
        clips, why = _rvc_clips(ctx, ls, passports)
        if why:
            reasons[sid] = why
            if tick is not None:
                tick(sid)
            continue
        plans[sid] = clips
    n = 0
    holder: dict[str, Any] = {}
    for sid, clips in plans.items():
        outcome = _rvc_train_character(ctx, cache, sid, clips, chars, char_f0.get(sid), holder, reasons)
        if outcome in ("done", "cached"):
            n += 1
        if holder.get("unavailable"):
            for s in plans:
                reasons[s] = f"rvc backend unavailable: {holder['unavailable']}"
            if tick is not None:
                tick(sid)
            break
        if tick is not None:
            tick(sid)
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
    sentences = dz.TEST_SENTENCES.get(ctx.cfg.project.tgt_lang, dz.TEST_SENTENCES["en"])
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
            backend, _, reason = gu.acquire_first(ctx, [rc.backend], group="route C training backend", stage=NAME)
            if backend is None:
                for s in eligible:
                    reasons.setdefault(s, f"route-C backend unavailable: {reason}")
                break
        out_dir = ctx.paths.vb_lora(sid)
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            # hours, like the RVC training above: without timeout_s the per-item default (120 s + 300 s) killed a
            # VoxCPM LoRA run that takes hours, and its training subprocess lived on holding the VRAM
            timeout_s = float((getattr(backend, "options", None) or {}).get("timeout_train_s", 6 * 3600))
            res = backend.call("train_lora", None, job={"speaker_id": sid, "pairs": recs, "out_dir": str(out_dir)}, timeout_s=timeout_s) or {}
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


# ----------------------------------------------------------------------------
# casting screen API (vndub_app/API.md 2.3)
# ----------------------------------------------------------------------------


def _enter(ctx: Any) -> None:
    """Progress callbacks are keyed by stage name; API calls run outside ``run_stage``."""
    if not ctx.stage:
        ctx.stage = NAME


def _character(ctx: Any, sid: str) -> tuple[dict[str, Character], Character]:
    chars = ctx.characters()
    if sid not in chars:
        raise ValueError("unknown_character")
    return chars, chars[sid]


def _has_original_audio(ctx: Any, sid: str) -> bool:
    return any(l.speaker_id == sid for l in ctx.voiced_lines(apply_only=False))


def list_designs(ctx: Any, speaker_id: str) -> list[dict[str, Any]]:
    """All designs of a character, best first (``rank_score``), with ``stars`` filled in.

    ``stars``: 1..5 from ``spk_sim`` with the speaker gate threshold as the 3-star point (formula in
    :mod:`vndub.stages.designs`); ``None`` for characters without original audio. No model is loaded.
    """
    return dz.ranked(ctx, dz.load_designs(ctx, speaker_id))


def sample_lines(ctx: Any, speaker_id: str, n: int = 2) -> list[str]:
    """Texts of ``n`` translated lines of the character (the audition lines); the configured test
    sentences fill the gap when there are not enough translated lines."""
    return [row["text"] for row in dz.sample_items(ctx, speaker_id, max(0, int(n)))]


def propose_designs(
    ctx: Any, speaker_id: str, n: int = 3, description: str | None = None, sample_line_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Add ``n`` voice designs for the character; returns the new designs (ranked best first).

    GPU work through the model registry: the design backend, then (characters with audio) the speaker
    embedder and the F0 backend. A character with original audio also gets its ``clone`` design (route A)
    the first time. ``description`` replaces ``Character.description`` (and is saved there). Raises
    ``BackendUnavailable`` when the design backend cannot run.
    """
    _enter(ctx)
    vb = ctx.cfg.voicebank
    n = min(MAX_PROPOSE, max(1, int(n)))
    chars, char = _character(ctx, speaker_id)
    if description is not None and description.strip() != char.description:
        char.description = description.strip()
        save_characters(ctx.paths.characters, chars)
    existing = dz.load_designs(ctx, speaker_id)
    has_audio = _has_original_audio(ctx, speaker_id)
    refs = gu.load_refs(ctx, speaker_id)
    ls = [l for l in ctx.voiced_lines(apply_only=False) if l.speaker_id == speaker_id] if has_audio else []
    if has_audio and not refs and ls:
        passports = {l.line_id: p for l in ls if (p := gu.load_passport(ctx, l.line_id))}
        cache = _cache(ctx)
        refs = _ensure_refs(ctx, cache, speaker_id, ls, passports, chars)
        cache.save()
        save_characters(ctx.paths.characters, chars)
    char_f0 = _char_f0(ctx, chars, speaker_id, refs) if has_audio else None
    want_clone = has_audio and bool(refs) and not any(d["design_id"] == "clone" for d in existing)
    tick = _Ticker(ctx, n + (1 if want_clone else 0) + 1)
    ctx.progress(0, tick.total, speaker_id)
    backend, _, reason = gu.acquire_first(ctx, [vb.performer.backend], group="performer design backend", stage=NAME)
    if backend is None:
        raise BackendUnavailable(vb.performer.backend, reason)
    jobs = _plan_designs(ctx, speaker_id, char, existing, n, char_f0, description, sample_line_ids, origin="user", audition_only=not has_audio)
    new = _render_designs(ctx, backend, jobs, tick)
    if want_clone:
        clone = _clone_design(ctx, speaker_id, char, refs, char_f0, dz.sample_items(ctx, speaker_id, max(1, vb.performer.sample_lines), sample_line_ids), "user", tick)
        if clone is not None:
            new.append(clone)
        else:
            tick("clone")
    if new:
        _measure_new_designs(ctx, speaker_id, char, new, has_audio)
        dz.save_designs(ctx, speaker_id, [*existing, *new])
    tick(speaker_id)
    p = dz.star_params(ctx)
    return sorted((dz.public(ctx, d, p) for d in new), key=lambda d: -float(d["rank_score"]))


def add_reference_design(ctx: Any, speaker_id: str, wav_path: str) -> dict[str, Any]:
    """A design from the user's own sample (wav / mp3, 5-20 s of native speech of the voice).

    Raises ``ValueError`` with the message ``sample_too_short`` / ``sample_too_long`` /
    ``sample_bad_format``. The sample is decoded to ``designs/<id>/ref.wav``; when a synth TTS backend
    is usable the audition lines are rendered as a zero-shot clone of it (else the sample itself is the
    only audition clip). The sample becomes the performer's first timbre reference when chosen.
    """
    _enter(ctx)
    chars, char = _character(ctx, speaker_id)
    existing = dz.load_designs(ctx, speaker_id)
    did = dz.next_design_id(ctx, speaker_id, existing, "ref")
    ddir = dz.design_dir(ctx, speaker_id, did)
    ref = ddir / "ref.wav"
    try:
        dur = dz.decode_sample(ctx, wav_path, ref)
    except Exception:
        shutil.rmtree(ddir, ignore_errors=True)
        raise
    has_audio = _has_original_audio(ctx, speaker_id)
    char_f0 = (char.stats.f0_median_hz or _char_f0(ctx, chars, speaker_id, gu.load_refs(ctx, speaker_id))) if has_audio else None
    audition = dz.sample_items(ctx, speaker_id, max(1, ctx.cfg.voicebank.performer.sample_lines))
    d = _render_cloned(
        ctx, speaker_id, did, "reference", ref, None, audition, char, char_f0, cross_lingual=False, origin="user",
        extra={"ref_wav": ctx.paths.rel(ref), "duration_s": round(dur, 2)},
    )
    if d is None:  # no TTS backend: the sample itself is the audition clip
        spec = None
        got = _tts_backend(ctx)
        if got is not None:
            spec = got[0].spec
        d = {
            "design_id": did, "kind": "reference", "backend": spec.name if spec else "reference", "prompt": "", "description": "",
            "samples": [{"line_id": "", "wav": ctx.paths.rel(ref), "text": ""}], "ref_clips": [], "spk_sim": None, "f0_median_hz": None,
            "f0_delta_st": None, "rank_score": -1.0, "license": spec.license if spec else "user-supplied",
            "commercial_ok": bool(spec.commercial_ok) if spec else True, "created": gu.ts_now(), "origin": "user",
            "ref_wav": ctx.paths.rel(ref), "duration_s": round(dur, 2),
        }
    _measure_new_designs(ctx, speaker_id, char, [d], has_audio)
    dz.save_designs(ctx, speaker_id, [*existing, d])
    return dz.public(ctx, d)


def import_design(ctx: Any, speaker_id: str, design: dict[str, Any], wav_paths: list[str]) -> dict[str, Any]:
    """Reuse a voice saved in the cross-project voice library.

    ``design``: the library entry (``kind``, ``backend``, ``prompt``, ``description``, ``name``,
    optional ``samples: [{"text"}]`` aligned with ``wav_paths``). The wavs are copied (decoded to wav) into
    a new design ``imp<NN>`` and become the performer's timbre references when it is chosen. Raises
    ``ValueError("sample_bad_format")`` when a file cannot be read, ``ValueError("design_empty")`` without wavs.
    """
    from vndub.audio import ffmpeg

    _enter(ctx)
    chars, char = _character(ctx, speaker_id)
    if not wav_paths:
        raise ValueError("design_empty")
    existing = dz.load_designs(ctx, speaker_id)
    did = dz.next_design_id(ctx, speaker_id, existing, "imp")
    ddir = dz.design_dir(ctx, speaker_id, did)
    texts = [str((s or {}).get("text", "")) for s in (design.get("samples") or [])]
    samples = []
    try:
        for i, src in enumerate(wav_paths):
            p = Path(src)
            if not p.is_file():
                raise ValueError("sample_bad_format")
            dst = ddir / f"s{i}.wav"
            try:
                ffmpeg.decode(ctx.tools, p, dst, mono=True)
            except Exception as e:  # noqa: BLE001
                from vndub.tools import ToolMissing

                if isinstance(e, ToolMissing) and p.suffix.lower() == ".wav":
                    shutil.copy2(p, dst)
                else:
                    raise ValueError("sample_bad_format") from e
            samples.append({"line_id": "", "wav": ctx.paths.rel(dst), "text": texts[i] if i < len(texts) else ""})
    except Exception:
        shutil.rmtree(ddir, ignore_errors=True)
        raise
    kind = "voice_design" if design.get("kind") == "voice_design" else "reference"
    from vndub.backends import registry

    try:
        spec = registry.get(str(design.get("backend") or ""))
        lic, comm = spec.license, bool(spec.commercial_ok)
    except KeyError:
        lic, comm = str(design.get("license") or "unknown"), bool(design.get("commercial_ok", False))
    d = {
        "design_id": did, "kind": kind, "backend": str(design.get("backend") or "library"), "prompt": str(design.get("prompt") or ""),
        "description": str(design.get("description") or ""), "samples": samples, "ref_clips": [], "spk_sim": None, "f0_median_hz": None,
        "f0_delta_st": None, "rank_score": -1.0, "license": lic, "commercial_ok": comm, "created": gu.ts_now(), "origin": "library",
        "imported": True, "source_name": str(design.get("name") or design.get("id") or ""),
    }
    _measure_new_designs(ctx, speaker_id, char, [d], _has_original_audio(ctx, speaker_id))
    dz.save_designs(ctx, speaker_id, [*existing, d])
    return dz.public(ctx, d)


def choose_design(ctx: Any, speaker_id: str, design_id: str, route: str) -> dict[str, Any]:
    """Make a design the character's performer and set its route (``A`` clone / ``B`` performer + VC /
    ``C`` / ``P`` performer only). Writes ``characters.yaml`` (``performer`` incl. ``design_id`` and
    ``chosen_by: user``, ``route``) and ``performer.yaml`` (+ ``performer/*.wav``, what synth reads).

    A ``clone`` design only goes with route ``A`` (``ValueError("route_mismatch")``) and leaves
    performer.yaml alone. Returns ``{"speaker_id", "design_id", "route", "kind", "performer"}``.
    Raises ``ValueError`` (``unknown_character`` / ``design_not_found`` / ``bad_route``). No model is loaded;
    the emotion variants of the performer are rendered by the next ``voicebank`` run.
    """
    _enter(ctx)
    chars, char = _character(ctx, speaker_id)
    d = dz.get_design(ctx, speaker_id, design_id)
    if d is None:
        raise ValueError("design_not_found")
    if route not in ("A", "B", "C", "P"):
        raise ValueError("bad_route")
    if d["kind"] == "clone" and route != "A":
        raise ValueError("route_mismatch")
    if d["kind"] == "clone":
        tuning = (char.performer or {}).get("tuning")
        char.performer = {"kind": "clone", "backend": d.get("backend", ""), "description": "", "design_id": design_id, "chosen_by": "user"}
        if tuning:
            char.performer["tuning"] = tuning
    else:
        _write_performer(ctx, speaker_id, d, chars, "user")
        meta_path = ctx.paths.vb_rvc(speaker_id) / "meta.json"
        meta = read_json(meta_path)
        if meta:  # the pitch offset to the character depends on the performer's F0
            f0_t = _rvc_f0_transpose(ctx, speaker_id, char.stats.f0_median_hz)
            if meta.get("f0_transpose") != f0_t:
                meta["f0_transpose"] = f0_t
                write_json(meta_path, meta)
                char.rvc = meta
    char.route = route  # type: ignore[assignment]
    save_characters(ctx.paths.characters, chars)
    return {"speaker_id": speaker_id, "design_id": design_id, "route": route, "kind": d["kind"], "performer": dict(char.performer)}


def auto_cast(ctx: Any, overwrite: bool = False) -> dict[str, Any]:
    """Choose the top-ranked design for every character that has designs; returns ``{"chosen": {speaker_id: design_id}, "kept": [speaker_id]}``.

    A character whose voice the user chose (``chosen_by: user`` and the design still exists) is kept as it is unless ``overwrite``:
    the "choose automatically" button is for the characters that have no voice yet, not for undoing a casting made by ear.

    The ``clone`` design is only taken when it is the character's only design (its similarity is high by
    construction, but it is route A with the accent risk). Route: ``B`` for characters with original audio
    (enough of it for ``voicebank.min_minutes_for_voicebank``: below that no RVC model is ever trained), ``P`` without
    or with too little, ``A`` for a clone. No model is loaded.
    """
    _enter(ctx)
    voiced_lines = ctx.voiced_lines(apply_only=False)
    voiced = {l.speaker_id for l in voiced_lines}
    min_minutes = ctx.cfg.voicebank.min_minutes_for_voicebank
    audio = gu.load_audio_index(ctx)
    seconds: dict[str, float] = {}
    for l in voiced_lines:
        seconds[l.speaker_id] = seconds.get(l.speaker_id, 0.0) + gu.src_duration(ctx, l, gu.load_passport(ctx, l.line_id), audio.get(l.audio_id or ""))
    out: dict[str, str] = {}
    kept: list[str] = []
    for sid, char in ctx.characters().items():
        if sid == "narrator":
            continue
        ds = list_designs(ctx, sid)
        pool = [d for d in ds if d["kind"] != "clone"] or ds
        if not pool:
            continue
        mine = char.performer or {}
        if not overwrite and mine.get("chosen_by") == "user" and any(d["design_id"] == mine.get("design_id") for d in ds):
            kept.append(sid)
            continue
        d = pool[0]
        route = "A" if d["kind"] == "clone" else ("B" if sid in voiced else "P")
        if route == "B" and seconds.get(sid, 0.0) / 60.0 + 1e-9 < min_minutes:
            # voice conversion needs the character's RVC model (or refs), which the stage only builds from enough audio
            ctx.log.info("%s: %.1f min of original audio < %s min (voicebank.min_minutes_for_voicebank): route P instead of B", sid, seconds.get(sid, 0.0) / 60.0, min_minutes)
            route = "P"
        choose_design(ctx, sid, d["design_id"], route)
        out[sid] = d["design_id"]
    return {"chosen": out, "kept": kept}


def set_tuning(ctx: Any, speaker_id: str, tuning: dict[str, Any] | None) -> dict[str, Any]:
    """Store ``characters.yaml.<sid>.performer.tuning`` (normalized: clamped, defaults filled); returns it."""
    chars, char = _character(ctx, speaker_id)
    t = normalize_tuning(tuning)
    perf = dict(char.performer or {})
    perf["tuning"] = t
    char.performer = perf
    save_characters(ctx.paths.characters, chars)
    return t


def train_rvc(ctx: Any, speaker_id: str) -> dict[str, Any]:
    """Train (or reuse) the character's RVC model for route B; reports ``ctx.progress`` (3 steps).

    Returns ``{"speaker_id", "state": "done" | "skipped" | "failed", "reason_code", "reason", "cached",
    "meta"}``; ``reason_code``: ``none`` | ``disabled`` | ``no_voiced_lines`` | ``not_enough_audio`` |
    ``training_failed``. ``meta`` is the ``rvc/meta.json`` content (model, index, f0_transpose, minutes,
    epochs, sample_rate, backend) once a model exists. Raises ``BackendUnavailable`` when the RVC backend
    cannot run. Cached on the clip set and the RVC config; ``f0_transpose`` follows the chosen performer.
    """
    _enter(ctx)
    rv = ctx.cfg.voicebank.rvc
    chars, char = _character(ctx, speaker_id)
    chars_at_start = gu.snapshot_characters(chars)

    def result(state: str, code: str = "none", reason: str = "", cached: bool = False) -> dict[str, Any]:
        return {"speaker_id": speaker_id, "state": state, "reason_code": code, "reason": reason, "cached": cached, "meta": dict(char.rvc) if char.rvc else None}

    if not rv.enabled:
        return result("skipped", "disabled", "voicebank.rvc.enabled is false")
    ls = [l for l in ctx.voiced_lines(apply_only=False) if l.speaker_id == speaker_id]
    if not ls:
        return result("skipped", "no_voiced_lines", "the character has no original audio")
    ctx.progress(0, 3, speaker_id)
    passports = {l.line_id: p for l in ls if (p := gu.load_passport(ctx, l.line_id))}
    clips, why = _rvc_clips(ctx, ls, passports)
    if why:
        ctx.progress(3, 3, speaker_id)
        return result("skipped", "not_enough_audio", why)
    char_f0 = _char_f0(ctx, chars, speaker_id, gu.load_refs(ctx, speaker_id))
    cache = _cache(ctx)
    reasons: dict[str, str] = {}
    holder: dict[str, Any] = {}
    outcome = _rvc_train_character(ctx, cache, speaker_id, clips, chars, char_f0, holder, reasons, progress=ctx.progress)
    ctx.save_caches()
    if holder.get("unavailable"):
        raise BackendUnavailable(rv.backend, holder["unavailable"])
    if outcome == "failed":
        return result("failed", "training_failed", reasons.get(speaker_id, ""))
    # training took hours: characters.yaml is read again and only this character's `rvc` is written onto it, so a casting
    # edit made meanwhile (a chosen design, tuning, a name) is not reverted by the snapshot loaded before the training
    gu.save_characters_merged(ctx, chars, chars_at_start)
    ctx.progress(3, 3, speaker_id)
    return {"speaker_id": speaker_id, "state": "done", "reason_code": "none", "reason": "", "cached": outcome == "cached", "meta": dict(chars[speaker_id].rvc)}

"""analyze: the expression passport of every voiced line.

Models run grouped by model (one ``acquire`` each, all lines under it):
emotion2vec+ (9-class probs + embedding), anime SER (10 classes), audeering
A/D/V, SenseVoice events, forced aligner (char timestamps), F0 (RMVPE /
torchcrepe / pYIN), VAD (silero / energy), optional Qwen2-Audio caption.
CPU features in core: energy contour, spectral tilt, LUFS, SNR / BGM flag,
pauses and leading/trailing silence, mora rate.

Then per character: ``characters.yaml`` stats (F0 median/IQR, energy, LUFS,
mora rate, emotion histogram) and reference-clip candidates; finally the
StyleSpec of every passport (it needs the character medians).

A part whose backend is unavailable is recorded in ``passport.missing`` with
the reason and the stage continues (the style builder degrades gracefully).

analyze never recognises speech: the forced aligner gets the *known* script text (alignment,
not recognition) and only backends of role ``align`` are accepted as aligner / fallback; a
configured name of another role (an ASR model) is refused and the line's timing falls back to
VAD speech time (``passport.backends["timing"] = "vad_only"``, reason in ``passport.missing["align"]``).
SenseVoice is used as an event tagger only; its recognised text is dropped before caching.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

from vndub.audio import features, loudness, vad
from vndub.backends import registry
from vndub.backends.base import BackendUnavailable
from vndub.backends.caption.qwen2_audio import CAPTION_PROMPT
from vndub.manifest.io import Character, CharacterStats, LineRecord, Passport, load_characters, read_json, save_characters, write_json
from vndub.stages._opcache import OpCache
from vndub.stages.common import StageContext
from vndub.stages.link import expected_texts
from vndub.text import ja
from vndub.text.style_prompt import build_style, emotion_vote

NAME = "analyze"
VERSION = "1"
CONFIG_SECTIONS = ("analyze",)

CHUNK = 32

_TAG = re.compile(r"<\|[^|]+\|>")


def event_tags(raw: str | None) -> str:
    """SenseVoice raw output reduced to its ``<|tag|>`` markers (the recognised text is dropped)."""
    return "".join(_TAG.findall(raw or ""))


def alignment_backends(names: list[str], lang: str = "") -> tuple[list[str], list[str]]:
    """(names usable for forced alignment, notes for names refused because they would transcribe).

    Only role ``align`` backends align text that is already known; anything else (an ASR model set
    as aligner or fallback) is refused so that analyze can never run speech recognition. A backend
    built for one language (``options.language``) is skipped for voices in another ``lang``.
    """
    ok: list[str] = []
    notes: list[str] = []
    for n in names:
        if not n:
            continue
        try:
            spec = registry.get(n)
        except KeyError:
            ok.append(n)  # unknown names are reported by the registry when resolved
            continue
        fixed = spec.options.get("language")
        if spec.role != "align":
            notes.append(f"{n}: role '{spec.role}' would transcribe, analyze only aligns known text")
        elif lang and fixed and fixed != lang:
            notes.append(f"{n}: a {fixed} aligner, the voices are {lang}")
        else:
            ok.append(n)
    return ok, notes


# ----------------------------------------------------------------------------
# generic grouped model pass
# ----------------------------------------------------------------------------


class PartResult:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}
        self.errors: dict[str, str] = {}
        self.backend = ""
        self.reason = ""


def run_part(
    ctx: StageContext,
    part: str,
    names: list[str],
    lines: list[LineRecord],
    make_item: Callable[[LineRecord], dict[str, Any]],
    op: str,
    op_kwargs: dict[str, Any] | None = None,
    outputs: Callable[[LineRecord], list[Path]] | None = None,
    value_of: Callable[[LineRecord, dict[str, Any]], Any] | None = None,
    stop_on: Callable[[str], bool] | None = None,
    on_items: Callable[[int, str | None], None] | None = None,
) -> PartResult:
    """Run ``op`` for every line with the first usable backend in ``names``.

    Results are cached per line (key = backend + op kwargs + item fields + wav content).
    A backend that becomes unavailable mid-way hands over to the next name for all lines,
    so one part never mixes two models (embeddings would not be comparable).
    """
    op_kwargs = op_kwargs or {}
    reasons: list[str] = []
    for name in [n for n in names if n]:
        res = PartResult()
        try:
            spec, _ = ctx.models.resolve(name)
        except BackendUnavailable as e:
            reasons.append(str(e))
            continue
        store = OpCache(ctx, f"analyze_{part}_{spec.name}")
        todo: list[tuple[LineRecord, dict[str, Any], str]] = []
        for l in lines:
            item = make_item(l)
            sig = {k: v for k, v in item.items() if k not in ("id", "out", "wav")}
            key = store.key(spec.name, op, op_kwargs, sig, files=[item["wav"]])
            hit = store.get(l.line_id, key, outputs(l) if outputs else ())
            if hit is not None:
                res.values[l.line_id] = hit
            else:
                todo.append((l, item, key))
        if on_items is not None and len(lines) > len(todo):
            on_items(len(lines) - len(todo), None)
        try:
            if todo:
                backend = ctx.models.acquire(name)
                stopped = False
                for i in range(0, len(todo), CHUNK):
                    chunk = todo[i : i + CHUNK]
                    out = backend.call(op, [it for _, it, _ in chunk], **op_kwargs)
                    by_id = {r.get("id"): r for r in out or []}
                    for l, _, key in chunk:
                        r = by_id.get(l.line_id) or {"error": "no result"}
                        if r.get("error"):
                            res.errors[l.line_id] = str(r["error"])
                            ctx.log.warning("%s failed for %s: %s", part, l.line_id, r["error"])
                            if stop_on and stop_on(str(r["error"])):
                                stopped = True
                            continue
                        val = value_of(l, r) if value_of else {k: v for k, v in r.items() if k != "id"}
                        res.values[l.line_id] = val
                        store.put(l.line_id, key, val)
                    store.save()
                    if on_items is not None:
                        on_items(len(chunk), chunk[-1][0].line_id)
                    if stopped:
                        res.reason = f"stopped after error: {next(iter(res.errors.values()))}"
                        break
        except BackendUnavailable as e:
            store.save()
            reasons.append(str(e))
            continue
        store.save()
        res.backend = spec.name
        return res
    res = PartResult()
    res.reason = "; ".join(reasons) or "not configured"
    return res


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def _rel(ctx: StageContext, p: str | Path | None) -> str | None:
    return ctx.paths.rel(p) if p else None


def core_features(ctx: StageContext, l: LineRecord, store: OpCache) -> dict[str, Any]:
    wav16 = ctx.paths.abs(l.src_wav16 or l.src_wav or "")
    wav = ctx.paths.abs(l.src_wav or l.src_wav16 or "")
    epath = ctx.paths.feat(l.line_id, "energy")
    key = store.key("core", files=[wav16, wav])
    hit = store.get(l.line_id, key, [epath])
    if hit is not None:
        return hit
    y16, sr16 = features.load(wav16, sr=16000)
    e = features.rms_db(y16, sr16)
    epath.parent.mkdir(parents=True, exist_ok=True)
    np.save(epath, e.astype(np.float32))
    y, sr = features.load(wav)
    val = {
        "duration_s": round(len(y16) / sr16, 3),
        "lufs": None if (lv := loudness.lufs(y, sr)) is None else round(lv, 2),
        "spectral_tilt_db_khz": round(features.spectral_tilt(y16, sr16), 3),
        "energy_contour_path": ctx.paths.rel(epath),
    }
    store.put(l.line_id, key, val)
    return val


def _median(xs: list[float]) -> float | None:
    v = [float(x) for x in xs if x is not None]
    return round(float(np.median(v)), 3) if v else None


def build_passport(
    ctx: StageContext,
    l: LineRecord,
    text: str,
    core: dict[str, Any],
    parts: dict[str, PartResult],
) -> Passport:
    p = Passport(line_id=l.line_id, duration_s=core["duration_s"])
    backends: dict[str, str] = {}
    missing: dict[str, str] = {}

    def get(part: str) -> Any:
        r = parts.get(part)
        if r is None:
            return None
        if l.line_id in r.values:
            backends[part] = r.backend
            return r.values[l.line_id]
        if l.line_id in r.errors:
            missing[part] = f"error: {r.errors[l.line_id]}"
        elif r.reason:
            missing[part] = r.reason
        return None

    emotion: dict[str, Any] = {}
    if (e2v := get("e2v")) is not None:
        emotion["e2v"] = e2v
    if (anime := get("anime")) is not None:
        emotion["anime"] = {"label": anime.get("label"), "probs": anime.get("probs", {})}
    if (adv := get("adv")) is not None:
        emotion["adv"] = {k: adv.get(k) for k in ("arousal", "dominance", "valence")}
    p.emotion = emotion
    ev = get("events")
    events = list((ev or {}).get("events") or [])

    # VAD -> silence profile (core energy VAD when the configured VAD is missing)
    seg_val = get("vad")
    if seg_val and seg_val.get("segments") is not None:
        segs = [tuple(s) for s in seg_val["segments"]]
    else:
        y16, sr16 = features.load(ctx.paths.abs(l.src_wav16 or l.src_wav or ""), sr=16000)
        segs = vad.energy_vad(y16, sr16)
        backends["vad"] = "energy_vad(core)"
    prof = vad.silence_profile(segs, p.duration_s)

    energy = np.load(ctx.paths.abs(core["energy_contour_path"]))
    mask = features.mask_from_segments(energy.size, segs)
    snr = features.snr_db(energy, mask)
    p.snr_db = None if snr is None else round(snr, 2)
    speech_e = energy[mask] if mask.any() else energy
    noise_e = energy[~mask] if (~mask).sum() >= 5 else np.array([])
    noise_floor = float(np.mean(noise_e)) if noise_e.size else None
    p.bgm = "bgm" in events or (p.snr_db is not None and p.snr_db < 12.0 and noise_floor is not None and noise_floor > -45.0)

    f0_val = get("f0")
    f0 = None
    f0_path = None
    hop = features.HOP_S
    if f0_val and f0_val.get("path"):
        fp = ctx.paths.abs(f0_val["path"])
        if fp.exists():
            f0 = np.load(fp).reshape(-1)
            f0_path = ctx.paths.rel(fp)
            hop = float(f0_val.get("hop_s") or hop)
    stats = features.f0_stats(f0) if f0 is not None else {"f0_median_hz": None, "f0_std_st": None, "f0_range_st": None, "f0_iqr_st": None, "voiced_ratio": None}

    al = get("align")
    chars = list((al or {}).get("chars") or [])
    p.align = {"chars": chars, "backend": backends.get("align", "")} if chars else {}
    if not chars:
        backends["timing"] = "vad_only"  # mora rate over the VAD speech time (no forced alignment for this line)
    span = (float(chars[-1]["t1"]) - float(chars[0]["t0"])) if len(chars) >= 2 else 0.0
    speech_s = span if span > 0.2 else (prof["speech_s"] or max(0.1, p.duration_s - prof["leading_silence_s"] - prof["trailing_silence_s"]))
    morae = ja.count_morae(text)
    rate = round(morae / speech_s, 3) if speech_s > 0 and morae else None

    p.events = events
    p.prosody = {
        **{k: v for k, v in stats.items()},
        "energy_db_mean": round(float(np.mean(speech_e)), 3) if speech_e.size else None,
        "energy_contour_path": core["energy_contour_path"],
        "f0_contour_path": f0_path,
        "f0_hop_s": hop,
        "rate_mora_per_s": rate,
        "morae": morae,
        "pauses": prof["pauses"],
        "leading_silence_s": prof["leading_silence_s"],
        "trailing_silence_s": prof["trailing_silence_s"],
        "speech_s": round(speech_s, 3),
        "arc": features.arc(f0, hop) if f0 is not None else "flat",
        "spectral_tilt_db_khz": core["spectral_tilt_db_khz"],
        "lufs": core["lufs"],
    }
    cap = get("caption")
    if cap:
        p.caption = cap.get("caption")
    if not ctx.cfg.analyze.caption:
        missing.pop("caption", None)
    p.backends = backends
    p.missing = missing
    return p


def vote_label(p: dict[str, Any], w_e2v: float, w_anime: float) -> str:
    emo = p.get("emotion") or {}
    return emotion_vote((emo.get("e2v") or {}).get("probs"), (emo.get("anime") or {}).get("probs"), w_e2v, w_anime)[0]


def aggregate(passports: list[dict[str, Any]], w_e2v: float, w_anime: float) -> dict[str, Any]:
    """Character stats over the passports of one speaker."""
    f0s = [(p.get("prosody") or {}).get("f0_median_hz") for p in passports]
    f0_med = _median(f0s)
    iqr = None
    if f0_med:
        st = [12 * np.log2(float(f) / f0_med) for f in f0s if f]
        if len(st) >= 2:
            q75, q25 = np.percentile(st, [75, 25])
            iqr = round(float(q75 - q25), 3)
    labels = Counter(vote_label(p, w_e2v, w_anime) for p in passports)
    n = len(passports)
    return {
        "n_lines": n,
        "total_sec": round(sum(float(p.get("duration_s") or 0) for p in passports), 2),
        "f0_median_hz": f0_med,
        "f0_iqr_st": iqr,
        "energy_db_median": _median([(p.get("prosody") or {}).get("energy_db_mean") for p in passports]),
        "lufs_median": _median([(p.get("prosody") or {}).get("lufs") for p in passports]),
        "rate_median": _median([(p.get("prosody") or {}).get("rate_mora_per_s") for p in passports]),
        "emotion_hist": {k: round(v / n, 3) for k, v in labels.most_common()} if n else {},
    }


def ref_candidates(
    items: list[tuple[LineRecord, dict[str, Any]]],
    min_snr: float,
    min_sec: float,
    max_sec: float,
    per_emotion: int,
    w_e2v: float,
    w_anime: float,
) -> list[str]:
    """Top clips by SNR (no BGM, duration window), ``per_emotion`` per emotion class."""
    by_class: dict[str, list[tuple[float, str]]] = {}
    for l, p in items:
        if p.get("bgm"):
            continue
        d = float(p.get("duration_s") or 0)
        snr = p.get("snr_db")
        if snr is None or snr < min_snr or not (min_sec <= d <= max_sec) or not l.src_wav:
            continue
        by_class.setdefault(vote_label(p, w_e2v, w_anime), []).append((float(snr), l.src_wav))
    out: list[str] = []
    for cls in sorted(by_class, key=lambda c: -len(by_class[c])):
        for _, path in sorted(by_class[cls], key=lambda x: -x[0])[:per_emotion]:
            out.append(path)
    return out


# ----------------------------------------------------------------------------
# stage
# ----------------------------------------------------------------------------


def run(ctx: StageContext) -> dict[str, Any]:
    a = ctx.cfg.analyze
    src_lang, audio_lang = ctx.src_lang, ctx.audio_lang
    all_voiced = ctx.voiced_lines(apply_only=False)
    lines = ctx.select(all_voiced)
    texts = expected_texts(ctx.lines())

    def wav16(l: LineRecord) -> str:
        return str(ctx.paths.abs(l.src_wav16 or l.src_wav or ""))

    lines = [l for l in lines if (l.src_wav16 or l.src_wav) and Path(wav16(l)).exists()]
    parts: dict[str, PartResult] = {}

    # progress: one unit per line and model pass, then one per passport
    n_parts = 1 + sum(1 for x in (a.anime_ser, a.adv, a.events, a.aligner, a.f0, a.vad, a.caption) if x)
    total = max(1, (n_parts + 1) * len(lines))
    done = 0

    def step(n: int, item: str | None) -> None:
        nonlocal done
        done = min(total, done + n)
        ctx.progress(done, total, item)

    parts["e2v"] = run_part(
        ctx, "e2v", [a.e2v], lines,
        lambda l: {"id": l.line_id, "wav": wav16(l), "out": str(ctx.paths.e2v_emb(l.line_id))},
        "emotion",
        on_items=step,
        outputs=lambda l: [ctx.paths.e2v_emb(l.line_id)],
        value_of=lambda l, r: {"probs": r.get("probs", {}), "label": r.get("label"), "embedding_path": ctx.paths.rel(ctx.paths.e2v_emb(l.line_id)) if ctx.paths.e2v_emb(l.line_id).exists() else _rel(ctx, r.get("embedding_path"))},
    )
    if a.anime_ser:
        parts["anime"] = run_part(ctx, "anime", [a.anime_ser], lines, lambda l: {"id": l.line_id, "wav": wav16(l)}, "emotion", on_items=step,
                                  value_of=lambda l, r: {"label": r.get("label"), "probs": r.get("probs", {})})
    if a.adv:
        parts["adv"] = run_part(ctx, "adv", [a.adv], lines, lambda l: {"id": l.line_id, "wav": wav16(l)}, "adv", on_items=step,
                                value_of=lambda l, r: {k: r.get(k) for k in ("arousal", "dominance", "valence")})
    if a.events:
        parts["events"] = run_part(ctx, "events", [a.events], lines, lambda l: {"id": l.line_id, "wav": wav16(l)}, "events", on_items=step,
                                   value_of=lambda l, r: {"events": list(r.get("events") or []), "raw": event_tags(r.get("raw", ""))})
    if a.aligner and audio_lang != src_lang:
        # a translated release keeps the original voices: the text says something else, nothing to align
        parts["align"] = PartResult()
        parts["align"].reason = f"the voices are {audio_lang} but the text is {src_lang}: nothing to align; timing from VAD speech time"
        ctx.log.warning("aligner skipped: %s", parts["align"].reason)
        step(len(lines), None)
    elif a.aligner:
        # forced alignment of the KNOWN text; a backend that would transcribe is refused (VAD-only timing instead)
        align_names, align_notes = alignment_backends([a.aligner, a.aligner_fallback], audio_lang)
        for note in align_notes:
            ctx.log.warning("aligner refused: %s", note)
        parts["align"] = run_part(ctx, "align", align_names, lines,
                                  lambda l: {"id": l.line_id, "wav": wav16(l), "text": texts.get(l.line_id, l.text_src), "lang": audio_lang}, "align", on_items=step,
                                  value_of=lambda l, r: {"chars": list(r.get("chars") or [])})
        if align_notes and not parts["align"].backend:
            parts["align"].reason = "; ".join(align_notes) + "; timing from VAD speech time"
    if a.f0:
        parts["f0"] = run_part(ctx, "f0", [a.f0, "librosa_f0"], lines,
                               lambda l: {"id": l.line_id, "wav": wav16(l), "out": str(ctx.paths.feat(l.line_id, "f0"))}, "f0", on_items=step,
                               outputs=lambda l: [ctx.paths.feat(l.line_id, "f0")],
                               value_of=lambda l, r: {"path": ctx.paths.rel(ctx.paths.feat(l.line_id, "f0")) if ctx.paths.feat(l.line_id, "f0").exists() else _rel(ctx, r.get("path")), "hop_s": r.get("hop_s", features.HOP_S)})
    if a.vad:
        parts["vad"] = run_part(ctx, "vad", [a.vad, "energy_vad"], lines, lambda l: {"id": l.line_id, "wav": wav16(l)}, "vad", on_items=step,
                                value_of=lambda l, r: {"segments": r.get("segments") or [], "duration": r.get("duration")})
    caption_note = ""
    if a.caption:
        parts["caption"] = run_part(ctx, "caption", [a.caption_backend], lines, lambda l: {"id": l.line_id, "wav": wav16(l)}, "caption",
                                    op_kwargs={"prompt": CAPTION_PROMPT}, on_items=step,
                                    value_of=lambda l, r: {"caption": r.get("caption", "")},
                                    stop_on=lambda err: "out of memory" in err.lower())
        if parts["caption"].reason:
            caption_note = f"captions disabled: {parts['caption'].reason}"
            ctx.log.warning(caption_note)
    ctx.models.release_heavy()

    # passports
    core_store = OpCache(ctx, "analyze_core")
    written = 0
    for l in lines:
        core = core_features(ctx, l, core_store)
        p = build_passport(ctx, l, texts.get(l.line_id, l.text_src), core, parts)
        prev = read_json(ctx.paths.passport(l.line_id))
        if prev and prev.get("style"):
            p.style = prev["style"]
        write_json(ctx.paths.passport(l.line_id), p)
        written += 1
        step(1, l.line_id)
    core_store.save()
    ctx.progress(total, total, None)

    # per-character aggregation over every passport on disk
    refs_cfg = ctx.cfg.voicebank.refs
    by_spk: dict[str, list[tuple[LineRecord, dict[str, Any]]]] = {}
    for l in all_voiced:
        d = read_json(ctx.paths.passport(l.line_id))
        if d:
            by_spk.setdefault(l.speaker_id, []).append((l, d))
    chars = load_characters(ctx.paths.characters)
    stats_of: dict[str, dict[str, Any]] = {}
    for sid, items in by_spk.items():
        st = aggregate([p for _, p in items], a.e2v_weight, a.anime_weight)
        stats_of[sid] = st
        c = chars.get(sid) or Character(names={"src": items[0][0].speaker_raw or sid, "tgt": ""})
        c.stats = CharacterStats(**st)
        c.ref_candidates = ref_candidates(items, refs_cfg.min_snr_db, refs_cfg.min_sec, refs_cfg.max_sec, a.ref_candidates_per_emotion, a.e2v_weight, a.anime_weight)
        chars[sid] = c
    save_characters(ctx.paths.characters, chars)

    # style pass (needs the character medians)
    emo_counts: Counter[str] = Counter()
    for sid, items in by_spk.items():
        for l, d in items:
            style = build_style(d, stats_of[sid], w_e2v=a.e2v_weight, w_anime=a.anime_weight, use_caption=a.caption).to_dict()
            emo_counts[style["emotion"]] += 1
            if d.get("style") != style:
                d["style"] = style
                write_json(ctx.paths.passport(l.line_id), d)

    missing = Counter()
    for l in lines:
        d = read_json(ctx.paths.passport(l.line_id)) or {}
        for part in (d.get("missing") or {}):
            missing[part] += 1
    summary: dict[str, Any] = {
        "lines": written,
        "characters": len(by_spk),
        "backends": {k: (v.backend or f"unavailable") for k, v in parts.items()},
        "emotions": dict(emo_counts.most_common()),
        "missing_parts": dict(missing),
        "ref_candidates": {sid: len(chars[sid].ref_candidates) for sid in by_spk},
    }
    unavailable = {k: v.reason for k, v in parts.items() if not v.backend}
    if unavailable:
        summary["unavailable"] = unavailable
    if caption_note:
        summary["caption"] = caption_note
    return summary

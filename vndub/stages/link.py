"""link: verify and repair the line <-> voice-file mapping.

1. ASR every bound voice clip (16 kHz) with ``link.asr`` (anime-whisper; no
   initial prompt), falling back to ``link.asr_fallback``.
2. Normalize (``text.ja.normalize_for_match``) and score RapidFuzz
   ``Levenshtein.normalized_similarity`` against the bound line and its
   +-``neighbour_window`` neighbours; >= ``accept_score`` ok, the review band
   ``[review_score, accept_score)`` review, below: global top-k search (blocked
   by speaker when known) and, with ``auto_repair``, move the voice to the line
   it clearly matches. Unbound voice files get the same global search.
3. ECAPA embeddings for every clip; agglomerative clustering (cosine);
   ``spk_consistency`` = cosine to the speaker centroid (leave-one-out);
   outliers and clips clustering with another speaker -> review.
4. ``link.jsonl`` (one record per judged line; ``audio_id`` = the audio the
   line now owns), ``review_queue.jsonl``, coverage stats.

When ASR is unavailable the stage does not fail: every line is marked review
with the reason.

Modes (``link.mode``):

* ``asr`` (CLI default, the behaviour above).
* ``no_asr`` ("intonation-check only", the app default): no ASR backend is acquired.
  The mapping is verified with cheap signals only -- speaker-embedding consistency per
  character, duration vs text length (morae of the script text, a plausible mora/s window
  plus silence tolerance), and voice-file prefix / index order. A line is ``ok`` when all
  signals agree and ``review`` otherwise, with machine-readable reasons prefixed by the
  signal (``spk_outlier:0.31``, ``spk_cluster:<speaker>``, ``duration_implausible:2.4``,
  ``order_gap``, ``prefix_mismatch:<prefix>``, ``audio_empty``); ``unmatched`` only when the audio is missing
  (``audio_missing``). Nothing is moved (no auto-repair). The summary gains ``confidence``
  (share of judged lines with all signals ok), ``outliers`` and ``suggest_asr_check``.

Mode C (``link.recover_text``): voiced lines whose script text is empty are transcribed
with ``link.recover_asr`` (fallback ``link.asr_fallback``) -- only those clips, in either
mode. The text is stored as ``text_recovered`` in ``link.jsonl`` and applied by
``StageContext.lines()`` (flag ``text_from_asr``); ``lines.jsonl`` is never rewritten.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Callable

import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from vndub.backends.base import BackendUnavailable
from vndub.backends.score.speaker_sim import agglomerative, cosine
from vndub.engines.base import VOICE_OVER, voice_name_parts
from vndub.manifest.io import AudioIndexRecord, Character, LineRecord, LinkRecord, iter_jsonl, read_jsonl, write_jsonl
from vndub.stages import genutil as gu
from vndub.stages._opcache import OpCache
from vndub.stages.common import StageContext, StageError
from vndub.text import ja
from vndub.text.lang import speech_units

NAME = "link"
VERSION = "2"
CONFIG_SECTIONS = ("link",)

#: a global match must beat the bound line by this much before a voice is moved
REPAIR_MARGIN = 0.15
CHUNK = 32
#: no_asr: fewer judged lines than this never suggest the ASR check (the share is too noisy)
SUGGEST_MIN_LINES = 5

#: (items finished, last item id) -- reports model-pass progress between chunks
ProgressFn = Callable[[int, Any], None]


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return float(Levenshtein.normalized_similarity(a, b))


# ----------------------------------------------------------------------------
# model passes
# ----------------------------------------------------------------------------


def run_asr(
    ctx: StageContext,
    items: list[tuple[str, str]],
    lang: str,
    names: list[str] | None = None,
    on_items: ProgressFn | None = None,
) -> tuple[dict[str, str | None], str, str]:
    """items: (audio_id, abs wav16). Returns ({audio_id: text|None}, backend used, failure reason).

    ``names``: backends to try in order (default ``link.asr``, ``link.asr_fallback``).
    ``on_items(n, last_id)`` is called after each cache scan / finished chunk (never mid-write).
    """
    cfg = ctx.cfg.link
    names = [n for n in (names if names is not None else [cfg.asr, cfg.asr_fallback]) if n]
    reasons: list[str] = []
    for name in names:
        try:
            spec, _ = gu.resolve_backend(ctx, name)
        except BackendUnavailable as e:
            reasons.append(str(e))
            continue
        fixed = spec.options.get("language")
        if fixed and lang and fixed != lang:  # anime-whisper is a Japanese model: useless for voices in another language
            reasons.append(f"{spec.name}: a {fixed} model, the voices are {lang}")
            continue
        options: dict[str, Any] = {"compute_type": cfg.asr_compute} if spec.adapter == "faster_whisper" else {}
        store = OpCache(ctx, f"link_asr_{spec.name}")
        out: dict[str, str | None] = {}
        todo: list[tuple[str, str, str]] = []
        for aid, wav in items:
            key = store.key(spec.name, options, lang, files=[wav])
            hit = store.get(aid, key)
            if hit is not None:
                out[aid] = hit.get("text")
            else:
                todo.append((aid, wav, key))
        if on_items is not None and len(items) > len(todo):
            on_items(len(items) - len(todo), None)
        try:
            if todo:
                backend = ctx.models.acquire(name, options)
                for i in range(0, len(todo), CHUNK):
                    chunk = todo[i : i + CHUNK]
                    res = backend.transcribe([{"id": aid, "wav": wav, "lang": lang} for aid, wav, _ in chunk], lang=lang)
                    by_id = {r.get("id"): r for r in res or []}
                    for aid, _, key in chunk:
                        r = by_id.get(aid) or {"error": "no result"}
                        if r.get("error"):
                            out[aid] = None
                            ctx.log.warning("asr failed for %s: %s", aid, r["error"])
                        else:
                            out[aid] = r.get("text") or ""
                            store.put(aid, key, {"text": out[aid]})
                    store.save()
                    if on_items is not None:
                        on_items(len(chunk), chunk[-1][0])
        except BackendUnavailable as e:
            reasons.append(str(e))
            store.save()
            continue
        store.save()
        return out, spec.name, ""
    return {}, "", "; ".join(reasons) or "no ASR backend configured"


def run_embed(ctx: StageContext, items: list[tuple[str, str]], on_items: ProgressFn | None = None) -> tuple[dict[str, np.ndarray], str, str]:
    name = ctx.cfg.link.speaker_embedder
    try:
        spec, _ = gu.resolve_backend(ctx, name)
    except BackendUnavailable as e:
        return {}, "", str(e)
    store = OpCache(ctx, f"link_spk_{spec.name}")
    out: dict[str, np.ndarray] = {}
    todo: list[tuple[str, str, str, str]] = []
    for aid, wav in items:
        dst = ctx.paths.spk_emb(aid)
        key = store.key(spec.name, files=[wav])
        if store.get(aid, key, outputs=[dst]) is not None:
            out[aid] = np.load(dst).reshape(-1)
        else:
            todo.append((aid, wav, str(dst), key))
    if on_items is not None and len(items) > len(todo):
        on_items(len(items) - len(todo), None)
    try:
        if todo:
            backend = ctx.models.acquire(name)
            for i in range(0, len(todo), CHUNK):
                chunk = todo[i : i + CHUNK]
                res = backend.embed([{"id": aid, "wav": wav, "out": dst} for aid, wav, dst, _ in chunk])
                by_id = {r.get("id"): r for r in res or []}
                for aid, _, dst, key in chunk:
                    r = by_id.get(aid) or {"error": "no result"}
                    if r.get("error"):
                        ctx.log.warning("embedding failed for %s: %s", aid, r["error"])
                        continue
                    out[aid] = np.load(r.get("path") or dst).reshape(-1)
                    store.put(aid, key, {"path": ctx.paths.rel(dst)})
                store.save()
                if on_items is not None:
                    on_items(len(chunk), chunk[-1][0])
    except BackendUnavailable as e:
        store.save()
        return out, spec.name, str(e)
    store.save()
    return out, spec.name, ""


# ----------------------------------------------------------------------------
# matching
# ----------------------------------------------------------------------------


def expected_texts(lines: list[LineRecord]) -> dict[str, str]:
    """Text each line's voice should contain (sustain heads include their continuation lines)."""
    by_id = {l.line_id: l for l in lines}
    out = {}
    for l in lines:
        text = l.text_src or ""
        for cid in getattr(l, "sustain_lines", None) or []:
            if cid in by_id:
                text += by_id[cid].text_src or ""
        out[l.line_id] = text
    return out


class Matcher:
    def __init__(self, lines: list[LineRecord], window: int, topk: int) -> None:
        self.lines = lines
        self.idx = {l.line_id: i for i, l in enumerate(lines)}
        self.norm = {k: ja.normalize_for_match(v) for k, v in expected_texts(lines).items()}
        self.window = window
        self.topk = topk
        # continuation lines of a voice sustain are covered by their head
        self.targets = [l for l in lines if self.norm[l.line_id] and "voice_sustain" not in l.flags]
        self.by_speaker: dict[str, dict[str, str]] = {}
        for l in self.targets:
            self.by_speaker.setdefault(l.speaker_id, {})[l.line_id] = self.norm[l.line_id]
        self.all_choices = {l.line_id: self.norm[l.line_id] for l in self.targets}

    def bound_score(self, line_id: str, asr_norm: str) -> float:
        return similarity(asr_norm, self.norm.get(line_id, ""))

    def neighbours(self, line_id: str, asr_norm: str) -> tuple[str | None, float]:
        i = self.idx[line_id]
        best, best_s = None, 0.0
        for j in range(max(0, i - self.window), min(len(self.lines), i + self.window + 1)):
            other = self.lines[j]
            if j == i or not self.norm[other.line_id] or "voice_sustain" in other.flags:
                continue
            s = similarity(asr_norm, self.norm[other.line_id])
            if s > best_s:
                best, best_s = other.line_id, s
        return best, best_s

    def global_search(self, asr_norm: str, speaker: str | None, exclude: str | None = None) -> list[tuple[str, float]]:
        choices = self.all_choices
        if speaker and speaker != "narrator" and self.by_speaker.get(speaker):
            choices = self.by_speaker[speaker]
        hits = process.extract(asr_norm, choices, scorer=Levenshtein.normalized_similarity, limit=self.topk + 1)
        return [(key, float(score)) for _, score, key in hits if key != exclude][: self.topk]


def judge(
    line: LineRecord,
    asr_text: str | None,
    matcher: Matcher,
    accept: float,
    review: float,
    auto_repair: bool,
    asr_note: str = "",
) -> tuple[LinkRecord, tuple[str, float] | None]:
    """LinkRecord for one bound line and an optional relink proposal (target line, score)."""
    rec = LinkRecord(line_id=line.line_id, audio_id=line.audio_id)
    if asr_text is None:
        rec.status = "review"
        rec.reasons = ["asr failed for this clip"]
        return rec, None
    rec.asr_text = asr_text
    an = ja.normalize_for_match(asr_text)
    if not an:
        rec.status = "review"
        rec.reasons = ["asr returned empty text" + (f" ({asr_note})" if asr_note else "")]
        return rec, None
    s0 = matcher.bound_score(line.line_id, an)
    rec.match_score = round(s0, 4)
    alt, alt_s = matcher.neighbours(line.line_id, an)
    rec.best_alt_line_id, rec.best_alt_score = alt, round(alt_s, 4)
    if s0 >= accept:
        rec.status = "ok"
        return rec, None
    if s0 < review:
        for gid, gs in matcher.global_search(an, line.speaker_id, exclude=line.line_id):
            if gs > rec.best_alt_score:
                rec.best_alt_line_id, rec.best_alt_score = gid, round(gs, 4)
            break
    proposal = None
    if auto_repair and rec.best_alt_line_id and rec.best_alt_score >= accept and rec.best_alt_score - s0 >= REPAIR_MARGIN:
        proposal = (rec.best_alt_line_id, rec.best_alt_score)
    if s0 >= review:
        rec.status = "review"
        rec.reasons = [f"match {s0:.2f} in review band [{review:.2f}, {accept:.2f})"]
    else:
        rec.status = "unmatched"
        rec.reasons = [f"no match for the bound line ({s0:.2f}); best {rec.best_alt_line_id} ({rec.best_alt_score:.2f})"]
    return rec, proposal


def apply_proposals(
    records: dict[str, LinkRecord],
    proposals: list[tuple[str, str, float]],
    owner: dict[str, str | None],
) -> int:
    """Move voices to the lines they match. ``owner``: line_id -> audio_id (mutated).

    A target that still owns a voice is a conflict (reported, not moved). Several
    passes resolve chains/swaps where the target's own voice moves away first.
    """
    pending = sorted(proposals, key=lambda p: -p[2])
    moved = 0
    # 2-cycles: two voices bound to each other's lines -> swap them
    by_src = {s: (t, sc) for s, t, sc in pending}
    swapped: set[str] = set()
    for s, (t, sc) in by_src.items():
        back = by_src.get(t)
        if back is None or back[0] != s or s in swapped or t in swapped or not owner.get(s) or not owner.get(t):
            continue
        a_s, a_t = owner[s], owner[t]
        r_s, r_t = records[s], records[t]
        owner[s], owner[t] = a_t, a_s
        records[t] = LinkRecord(line_id=t, audio_id=a_s, asr_text=r_s.asr_text, match_score=round(sc, 4), status="ok", relinked_from=s, reasons=[f"voice {a_s} swapped from {s}"])
        records[s] = LinkRecord(line_id=s, audio_id=a_t, asr_text=r_t.asr_text, match_score=round(back[1], 4), status="ok", relinked_from=t, reasons=[f"voice {a_t} swapped from {t}"])
        swapped |= {s, t}
        moved += 2
    pending = [p for p in pending if p[0] not in swapped]
    for _ in range(3):
        rest = []
        for src, tgt, score in pending:
            audio = owner.get(src)
            if audio is None:
                continue
            if owner.get(tgt):
                rest.append((src, tgt, score))
                continue
            owner[src] = None
            owner[tgt] = audio
            src_rec = records[src]
            src_rec.reasons = [*src_rec.reasons, f"voice {audio} moved to {tgt}"]
            src_rec.audio_id = None
            src_rec.status = "unmatched"
            prev = records.get(tgt)
            records[tgt] = LinkRecord(
                line_id=tgt,
                audio_id=audio,
                asr_text=src_rec.asr_text,
                match_score=round(score, 4),
                status="ok",
                relinked_from=src,
                reasons=([*prev.reasons] if prev else []) + [f"voice {audio} moved from {src}"],
            )
            moved += 1
        if len(rest) == len(pending):
            break
        pending = rest
    for src, tgt, score in pending:
        if owner.get(src):
            records[src].reasons = [*records[src].reasons, f"best match {tgt} ({score:.2f}) already has a voice"]
    return moved


def speaker_checks(
    records: dict[str, LinkRecord],
    owner: dict[str, str | None],
    speaker_of: dict[str, str],
    embs: dict[str, np.ndarray],
    distance: float,
    outlier: float,
    judged: set[str],
    machine: bool = False,
) -> dict[str, Any]:
    """Cluster all owned clips; annotate records of judged lines (mutated).

    ``machine``: reasons are ``spk_outlier:<consistency>`` / ``spk_cluster:<speaker>`` (no_asr mode)
    instead of the human-readable sentences.
    """
    lines = [lid for lid, aid in owner.items() if aid and aid in embs]
    if len(lines) < 2:
        return {"clusters": len(lines)}
    mat = np.stack([embs[owner[lid]] for lid in lines]).astype(np.float64)
    labels = agglomerative(mat, distance)
    lab = dict(zip(lines, labels.tolist()))
    groups: dict[str, list[str]] = {}
    for lid in lines:
        groups.setdefault(speaker_of.get(lid, "narrator"), []).append(lid)
    cluster_members: dict[int, list[str]] = {}
    for lid, c in lab.items():
        cluster_members.setdefault(c, []).append(lid)
    flagged = 0
    for lid in lines:
        if lid not in judged and lid not in records:
            continue
        rec = records.get(lid)
        if rec is None:
            continue
        rec.spk_cluster = int(lab[lid])
        spk = speaker_of.get(lid, "narrator")
        peers = [p for p in groups.get(spk, []) if p != lid]
        if len(peers) >= 1:
            vecs = [embs[owner[p]] / (np.linalg.norm(embs[owner[p]]) or 1.0) for p in peers]
            cen = np.mean(vecs, axis=0)
            rec.spk_consistency = round(cosine(embs[owner[lid]], cen), 4)
            if rec.spk_consistency < outlier:
                rec.reasons = [*rec.reasons, f"spk_outlier:{rec.spk_consistency:.2f}" if machine else f"speaker outlier: consistency {rec.spk_consistency:.2f} < {outlier:.2f}"]
                if rec.status == "ok":
                    rec.status = "review"
                flagged += 1
        members = cluster_members[lab[lid]]
        if len(members) >= 3:
            maj, cnt = Counter(speaker_of.get(m, "narrator") for m in members).most_common(1)[0]
            if maj != spk and cnt / len(members) > 0.6:
                rec.reasons = [*rec.reasons, f"spk_cluster:{maj}" if machine else f"voice clusters with speaker '{maj}' ({cnt}/{len(members)})"]
                if rec.status == "ok":
                    rec.status = "review"
                flagged += 1
    return {"clusters": int(len(cluster_members)), "speaker_flags": flagged}


# ----------------------------------------------------------------------------
# no_asr: cheap signals (duration vs text, voice-file prefix / index order)
# ----------------------------------------------------------------------------

def duration_check(text: str, duration_s: float, rate_min: float, rate_max: float, tolerance_s: float, lang: str = "ja") -> tuple[float | None, int, bool]:
    """(ratio, morae, implausible) for a clip of ``duration_s`` seconds that should say ``text``.

    ratio = duration / (morae / typical rate) with the typical rate = sqrt(rate_min * rate_max), so 1.0 is
    a typical speaking rate, 2.0 is twice as long as expected and 0.5 twice as short. The clip is
    implausible when it is shorter than morae / rate_max or longer than morae / rate_min (at least 1.5 s)
    plus ``tolerance_s`` (leading / trailing silence and pauses are part of the file). Texts without
    morae (punctuation only, empty) and empty clips give ``(None, morae, False)``. For a text in another
    language than Japanese ``morae`` are mora equivalents (:func:`vndub.text.lang.speech_units`).
    """
    morae = speech_units(text, lang)
    if morae < 1 or duration_s <= 0:
        return None, morae, False
    typical = (rate_min * rate_max) ** 0.5
    ratio = duration_s / (morae / typical)
    too_fast = duration_s < morae / rate_max
    too_slow = duration_s > max(morae / rate_min, 1.5) + tolerance_s
    return round(ratio, 2), morae, bool(too_fast or too_slow)


def _file_seconds(path: str) -> float:
    """Length of a wav file in seconds; 0.0 when it holds no samples, and also when it cannot be read (not told apart here:
    an unreadable voice file is a review reason as well)."""
    from vndub.audio import features

    try:
        return features.duration(path)
    except Exception:
        return 0.0


def order_gaps(lines: list[LineRecord]) -> dict[str, str]:
    """line_id -> voice prefix for voiced lines whose voice index runs backwards inside a scene.

    Inside one scene the files of one prefix are numbered in script order. When a line's index is
    lower than the previous line's index of the same prefix, both lines are flagged (either could be
    the swapped one). Equal indexes (a voice reused on purpose) are fine.
    """
    last: dict[tuple[str, str], tuple[str, int]] = {}
    out: dict[str, str] = {}
    for l in lines:
        if not (l.has_voice and l.audio_id):
            continue
        parts = voice_name_parts(l.voice_file)
        if parts is None:
            continue
        key = (l.scene, parts[0])
        prev = last.get(key)
        if prev is not None and parts[1] < prev[1]:
            out[l.line_id] = parts[0]
            out[prev[0]] = parts[0]
        last[key] = (l.line_id, parts[1])
    return out


def prefix_expectations(lines: list[LineRecord], chars: dict[str, Character]) -> dict[str, set[str]]:
    """speaker_id -> accepted voice prefixes (``characters.yaml`` ``voice_prefix`` when set, else the
    speaker's majority prefix).

    Empty when prefixes do not identify speakers in this game (``ch01_0001`` style names, hashed names,
    one prefix for everybody): then a "wrong prefix" says nothing. The narrator and the voice-over
    (voiced lines without a name whose prefix names nobody) are never checked.
    """
    pairs: list[tuple[str, str]] = []
    for l in lines:
        if l.has_voice and l.audio_id:
            parts = voice_name_parts(l.voice_file)
            if parts is not None:
                pairs.append((l.speaker_id, parts[0]))
    if len(pairs) < 4:
        return {}
    by_prefix: dict[str, Counter[str]] = {}
    by_speaker: dict[str, Counter[str]] = {}
    for sid, p in pairs:
        by_prefix.setdefault(p, Counter())[sid] += 1
        by_speaker.setdefault(sid, Counter())[p] += 1
    purity = sum(max(c.values()) for c in by_prefix.values()) / len(pairs)
    if len(by_prefix) < 2 or purity < 0.8:
        return {}
    out: dict[str, set[str]] = {}
    for sid, c in by_speaker.items():
        if sid in ("narrator", VOICE_OVER):
            continue
        declared = {x.lower() for x in re.split(r"[,\s|/]+", (chars[sid].voice_prefix if sid in chars else "") or "") if x}
        if declared:
            out[sid] = declared
            continue
        total = sum(c.values())
        maj, n = c.most_common(1)[0]
        if total >= 3 and n / total >= 0.6:
            out[sid] = {maj}
    return out


# ----------------------------------------------------------------------------
# mode C helpers
# ----------------------------------------------------------------------------


def no_text_lines(ctx: StageContext) -> list[str]:
    """line_ids of voiced lines (after link rebindings) that have no source text.

    "Text" is the effective text of :meth:`StageContext.lines`: script text, text recovered by a
    previous mode C run, or a user edit. These are the lines mode C (``link.recover_text``) would
    transcribe; after a successful recovery they drop out of this list.
    """
    lines = ctx.lines()
    texts = expected_texts(lines)
    voiced = {l.line_id for l in ctx.voiced_lines(apply_only=False)}
    return [l.line_id for l in lines if l.line_id in voiced and "voice_sustain" not in l.flags and not texts[l.line_id].strip()]


def text_recovered_lines(ctx: StageContext) -> list[str]:
    """line_ids whose source text comes from ASR (flag ``text_from_asr``; edited ones carry ``text_edited`` too)."""
    return [l.line_id for l in ctx.lines() if "text_from_asr" in l.flags]


class _Units:
    """Stage-wide progress: one unit per clip transcribed / embedded / judged."""

    def __init__(self, ctx: StageContext, total: int) -> None:
        self.ctx = ctx
        self.total = max(1, total)
        self.done = 0

    def step(self, n: int = 1, item: Any = None) -> None:
        self.done = min(self.total, self.done + max(0, n))
        self.ctx.progress(self.done, self.total, item)

    def finish(self) -> None:
        self.done = self.total
        self.ctx.progress(self.total, self.total, None)


# ----------------------------------------------------------------------------
# stage
# ----------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return round(float(np.mean(values)), 3) if values else 0.0


def run(ctx: StageContext) -> dict[str, Any]:
    cfg = ctx.cfg.link
    no_asr = cfg.mode == "no_asr"
    # a forced mode C run transcribes again: previously recovered texts are not applied to the lines
    redo_recovery = bool(ctx.force and cfg.recover_text)
    lines = ctx.lines(recovered=not redo_recovery)
    if not ctx.paths.audio_index.exists():
        raise StageError(f"{ctx.paths.audio_index} not found: run `vndub extract` first")
    audio = {a.id: a for a in read_jsonl(ctx.paths.audio_index, AudioIndexRecord)}
    src_lang, audio_lang = ctx.src_lang, ctx.audio_lang
    lang_note = ""
    if not no_asr and audio_lang != src_lang:
        # a translated release with the original voices: what the ASR hears is not the text, cheap signals only
        lang_note = f"the voices are {audio_lang} but the script text is {src_lang}: ASR text cannot be compared, cheap signals only"
        ctx.log.warning(lang_note)
        no_asr = True

    def wav16(aid: str) -> str:
        return str(ctx.paths.abs(audio[aid].wav16 or audio[aid].wav))

    def usable(aid: str | None) -> bool:
        a = audio.get(aid or "")
        return a is not None and not a.error and bool(a.wav16 or a.wav) and ctx.paths.abs(a.wav16 or a.wav).is_file()

    voiced = [l for l in lines if l.has_voice and l.audio_id and l.audio_id in audio]
    selected = ctx.select(voiced)
    bound = {l.audio_id for l in voiced}
    orphans = [] if (ctx.only or ctx.limit is not None) else [a for a in audio.values() if a.id not in bound and a.wav16 and not a.error]

    # lines whose audio is gone: unmatched (no_asr only; the asr mode keeps its behaviour)
    missing_audio: list[LineRecord] = []
    if no_asr:
        missing_audio = ctx.select([l for l in lines if ("voice_missing" in l.flags) or (l.has_voice and l.audio_id and not usable(l.audio_id))], voiced_only=False)
        gone = {l.line_id for l in missing_audio}
        voiced = [l for l in voiced if l.line_id not in gone]
        selected = [l for l in selected if l.line_id not in gone]
    selected_ids = {l.line_id for l in selected}

    # ---- mode C: transcribe only the voiced lines that have no text -------------------------------------
    texts_all = expected_texts(lines)
    fix = [l for l in selected if cfg.recover_text and usable(l.audio_id) and not texts_all[l.line_id].strip()]
    fix_audio = sorted({l.audio_id for l in fix if l.audio_id})
    # progress units: clips transcribed + lines judged + clips embedded
    n_judge = len(selected) - (0 if no_asr else len(fix))
    n_units = len(fix_audio) + n_judge + len({l.audio_id for l in voiced})
    if not no_asr:
        n_units += n_judge + len(orphans)  # one ASR item per judged clip and orphan
    units = _Units(ctx, n_units)

    recovered_now: dict[str, str] = {}
    recover_backend, recover_reason = "", ""
    if fix:
        rtexts, recover_backend, recover_reason = run_asr(
            ctx, [(aid, wav16(aid)) for aid in fix_audio], audio_lang, names=[cfg.recover_asr, cfg.asr_fallback], on_items=units.step
        )
        for l in fix:
            t = (rtexts.get(l.audio_id or "") or "").strip()
            if t:
                recovered_now[l.line_id] = t
                l.text_src = t
                if "text_from_asr" not in l.flags:
                    l.flags = [*l.flags, "text_from_asr"]
        if recover_reason:
            ctx.log.warning("text recovery unavailable: %s", recover_reason)
        elif len(recovered_now) < len(fix):
            ctx.log.warning("text recovery: %d of %d clips gave no text", len(fix) - len(recovered_now), len(fix))
        texts_all = expected_texts(lines)

    # texts recovered by earlier runs stay attached to their records (only while the script still has no text)
    prior_recovered: dict[str, str] = {}
    if ctx.paths.link.exists() and not redo_recovery:
        raw_empty = {l.line_id for l in ctx.lines(recovered=False, edits=False) if not (l.text_src or "").strip()}
        for row in iter_jsonl(ctx.paths.link):
            t = row.get("text_recovered")
            if isinstance(t, str) and t.strip() and row.get("line_id") in raw_empty:
                prior_recovered[row["line_id"]] = t
    recovered_all = {**prior_recovered, **recovered_now}

    records: dict[str, LinkRecord] = {}
    proposals: list[tuple[str, str, float]] = []
    asr_backend, asr_reason = "", ""
    relinked = orphans_bound = 0
    owner: dict[str, str | None] = {l.line_id: l.audio_id for l in voiced}

    if not no_asr:
        # ---- asr mode ---------------------------------------------------------------------------------
        matcher = Matcher(lines, cfg.neighbour_window, cfg.global_topk)
        recovered_ids = set(recovered_now)
        asr_items = [(l.audio_id, wav16(l.audio_id)) for l in selected if l.line_id not in recovered_ids] + [(a.id, wav16(a.id)) for a in orphans]
        texts, asr_backend, asr_reason = run_asr(ctx, asr_items, audio_lang, on_items=units.step) if asr_items else ({}, "", "")
        asr_note = "mock ASR cannot transcribe original clips" if asr_backend.startswith("mock") else ""
        for l in selected:
            if l.line_id in recovered_ids:  # the ASR text IS the line's text: nothing to compare it with
                records[l.line_id] = LinkRecord(line_id=l.line_id, audio_id=l.audio_id, asr_text=l.text_src, match_score=1.0, status="ok", reasons=[f"text recovered by ASR ({recover_backend})"])
                continue
            if asr_reason:
                records[l.line_id] = LinkRecord(line_id=l.line_id, audio_id=l.audio_id, status="review", reasons=[f"asr unavailable: {asr_reason}"])
                continue
            rec, prop = judge(l, texts.get(l.audio_id), matcher, cfg.accept_score, cfg.review_score, cfg.auto_repair, asr_note)
            records[l.line_id] = rec
            if prop is not None:
                proposals.append((l.line_id, prop[0], prop[1]))
            units.step(1, l.line_id)
        relinked = apply_proposals(records, proposals, owner) if proposals else 0

        # unbound voice files: bind to an unvoiced line they clearly match
        if cfg.auto_repair and not asr_reason:
            for a in orphans:
                t = texts.get(a.id)
                an = ja.normalize_for_match(t or "")
                if not an:
                    continue
                hits = matcher.global_search(an, None)
                if not hits:
                    continue
                tgt, score = hits[0]
                if score >= cfg.accept_score and not owner.get(tgt):
                    owner[tgt] = a.id
                    records[tgt] = LinkRecord(line_id=tgt, audio_id=a.id, asr_text=t or "", match_score=round(score, 4), status="ok", relinked_from=f"orphan:{a.id}", reasons=[f"unbound voice {a.relpath} matched"])
                    orphans_bound += 1
    else:
        # ---- no_asr: duration, order and prefix signals (speaker consistency follows below) ------------
        chars = ctx.characters() if ctx.paths.characters.exists() else {}
        gaps = order_gaps(voiced)
        expected_prefix = prefix_expectations(voiced, chars)
        for l in selected:
            reasons: list[str] = []
            text = texts_all[l.line_id]
            seconds = float(audio[l.audio_id or ""].duration or 0.0)
            if seconds <= 0:  # the index may simply not carry the length: ask the file before calling the clip empty
                seconds = _file_seconds(wav16(l.audio_id or ""))
            ratio, morae, implausible = duration_check(text, seconds, cfg.rate_min, cfg.rate_max, cfg.silence_tolerance_s, src_lang)
            signals: dict[str, Any] = {"dur_ratio": ratio, "morae": morae}
            if not text.strip():
                signals["no_text"] = True
            if seconds <= 0:
                # duration_check cannot judge an empty clip (it answers "plausible"): without this the mapping was accepted
                # with no reason although the voice file holds no sound
                reasons.append("audio_empty")
                signals["audio_empty"] = True
            if implausible:
                reasons.append(f"duration_implausible:{ratio}")
            parts = voice_name_parts(l.voice_file)
            if parts is not None:
                signals["voice_prefix"], signals["voice_index"] = parts
            if l.line_id in gaps:
                reasons.append("order_gap")
            want = expected_prefix.get(l.speaker_id)
            if parts is not None and want and parts[0] not in want:
                reasons.append(f"prefix_mismatch:{parts[0]}")
            records[l.line_id] = LinkRecord(line_id=l.line_id, audio_id=l.audio_id, status="review" if reasons else "ok", reasons=reasons, signals=signals)
            units.step(1, l.line_id)
        for l in missing_audio:
            ref = l.voice_file or l.audio_id or ""
            records[l.line_id] = LinkRecord(line_id=l.line_id, audio_id=l.audio_id, status="unmatched", reasons=[f"audio_missing:{ref}" if ref else "audio_missing"])

    # speaker embeddings over every owned clip
    owned = sorted({aid for aid in owner.values() if aid and (usable(aid) or not no_asr)})
    embs, emb_backend, emb_reason = run_embed(ctx, [(aid, wav16(aid)) for aid in owned], on_items=units.step) if owned else ({}, "", "")
    speaker_of = {l.line_id: l.speaker_id for l in lines}
    spk_summary: dict[str, Any] = {}
    if embs:
        spk_summary = speaker_checks(records, owner, speaker_of, embs, cfg.cluster_distance, cfg.outlier_consistency, selected_ids, machine=no_asr)
    if emb_reason:
        ctx.log.warning("speaker check incomplete: %s", emb_reason)

    # texts recovered by mode C travel with the record of their line
    for lid, t in recovered_all.items():
        rec = records.get(lid)
        if rec is not None:
            rec.text_recovered = t  # type: ignore[attr-defined]

    # merge with records of lines not judged this run
    existing = {r.line_id: r for r in read_jsonl(ctx.paths.link, LinkRecord)} if ctx.paths.link.exists() else {}
    existing.update(records)
    order = {l.line_id: i for i, l in enumerate(lines)}
    gone_lines = [lid for lid in existing if lid not in order]
    if gone_lines:  # a line that a re-parse removed must not stay in the review queue and the counts
        ctx.log.info("link: dropping %d record(s) of lines that no longer exist", len(gone_lines))
        for lid in gone_lines:
            del existing[lid]
    merged = sorted(existing.values(), key=lambda r: order.get(r.line_id, 1 << 30))
    write_jsonl(ctx.paths.link, merged)
    queue: list[dict[str, Any]] = []
    for r in merged:
        if r.status == "ok":
            continue
        q: dict[str, Any] = {"line_id": r.line_id, "status": r.status, "reasons": r.reasons, "match_score": r.match_score, "asr_text": r.asr_text, "best_alt_line_id": r.best_alt_line_id, "audio_id": r.audio_id}
        signals_of = (r.model_extra or {}).get("signals")
        if signals_of:
            q["signals"] = signals_of
        queue.append(q)
    write_jsonl(ctx.paths.review_queue, queue)
    units.finish()

    judged_ids = selected_ids | {m.line_id for m in missing_audio}
    judged = [records[lid] for lid in judged_ids if lid in records]
    counts = Counter(r.status for r in judged)
    scores = [r.match_score for r in judged if r.asr_text]
    coverage = round(counts.get("ok", 0) / len(judged), 3) if judged else 0.0
    outliers = counts.get("review", 0) + counts.get("unmatched", 0)
    suggest = bool(no_asr and len(judged) >= SUGGEST_MIN_LINES and outliers / len(judged) >= cfg.suggest_asr_share)
    if no_asr:
        asr_label = f"{recover_backend} (text recovery only)" if recover_backend else "skipped (mode no_asr)"
    else:
        asr_label = asr_backend or f"unavailable: {asr_reason}"
    summary: dict[str, Any] = {
        "mode": "no_asr" if (no_asr and cfg.mode != "no_asr") else cfg.mode,
        "src_lang": src_lang,
        "audio_lang": audio_lang,
        **({"note": lang_note} if lang_note else {}),
        "lines_judged": len(judged),
        "ok": counts.get("ok", 0),
        "review": counts.get("review", 0),
        "unmatched": counts.get("unmatched", 0),
        "coverage_ok": coverage,
        "confidence": coverage,
        "outliers": outliers,
        "suggest_asr_check": suggest,
        "mean_match": _mean(scores),
        "relinked": relinked,
        "orphans": len(orphans),
        "orphans_bound": orphans_bound,
        "asr": asr_label,
        "embedder": emb_backend or (f"unavailable: {emb_reason}" if emb_reason else "-"),
        **spk_summary,
        "review_queue": len(queue),
    }
    if no_asr:
        flagged = Counter(reason.split(":", 1)[0] for rec in judged for reason in rec.reasons)
        summary["signals"] = dict(flagged.most_common())
    if cfg.recover_text:
        summary["recover"] = {
            "asr": recover_backend or (f"unavailable: {recover_reason}" if recover_reason else "-"),
            "candidates": len(fix),
            "recovered": len(recovered_now),
            "no_text_left": len(no_text_lines(ctx)),
        }
    return summary

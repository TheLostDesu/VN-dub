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
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from vndub.backends.base import BackendUnavailable
from vndub.backends.score.speaker_sim import agglomerative, cosine
from vndub.manifest.io import AudioIndexRecord, LineRecord, LinkRecord, read_jsonl, write_jsonl
from vndub.stages._opcache import OpCache
from vndub.stages.common import StageContext, StageError
from vndub.text import ja

NAME = "link"
VERSION = "1"
CONFIG_SECTIONS = ("link",)

#: a global match must beat the bound line by this much before a voice is moved
REPAIR_MARGIN = 0.15
CHUNK = 32


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return float(Levenshtein.normalized_similarity(a, b))


# ----------------------------------------------------------------------------
# model passes
# ----------------------------------------------------------------------------


def run_asr(ctx: StageContext, items: list[tuple[str, str]], lang: str) -> tuple[dict[str, str | None], str, str]:
    """items: (audio_id, abs wav16). Returns ({audio_id: text|None}, backend used, failure reason)."""
    cfg = ctx.cfg.link
    names = [n for n in [cfg.asr, cfg.asr_fallback] if n]
    reasons: list[str] = []
    for name in names:
        try:
            spec, _ = ctx.models.resolve(name)
        except BackendUnavailable as e:
            reasons.append(str(e))
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
        except BackendUnavailable as e:
            reasons.append(str(e))
            store.save()
            continue
        store.save()
        return out, spec.name, ""
    return {}, "", "; ".join(reasons) or "no ASR backend configured"


def run_embed(ctx: StageContext, items: list[tuple[str, str]]) -> tuple[dict[str, np.ndarray], str, str]:
    name = ctx.cfg.link.speaker_embedder
    try:
        spec, _ = ctx.models.resolve(name)
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
) -> dict[str, Any]:
    """Cluster all owned clips; annotate records of judged lines (mutated)."""
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
                rec.reasons = [*rec.reasons, f"speaker outlier: consistency {rec.spk_consistency:.2f} < {outlier:.2f}"]
                if rec.status == "ok":
                    rec.status = "review"
                flagged += 1
        members = cluster_members[lab[lid]]
        if len(members) >= 3:
            maj, cnt = Counter(speaker_of.get(m, "narrator") for m in members).most_common(1)[0]
            if maj != spk and cnt / len(members) > 0.6:
                rec.reasons = [*rec.reasons, f"voice clusters with speaker '{maj}' ({cnt}/{len(members)})"]
                if rec.status == "ok":
                    rec.status = "review"
                flagged += 1
    return {"clusters": int(len(cluster_members)), "speaker_flags": flagged}


# ----------------------------------------------------------------------------
# stage
# ----------------------------------------------------------------------------


def run(ctx: StageContext) -> dict[str, Any]:
    cfg = ctx.cfg.link
    lines = ctx.lines()
    if not ctx.paths.audio_index.exists():
        raise StageError(f"{ctx.paths.audio_index} not found: run `vndub extract` first")
    audio = {a.id: a for a in read_jsonl(ctx.paths.audio_index, AudioIndexRecord)}
    src_lang = ctx.cfg.project.src_lang
    proj = ctx.project
    if proj is not None:
        src_lang = proj.src_lang or src_lang

    voiced = [l for l in lines if l.has_voice and l.audio_id and l.audio_id in audio]
    selected = ctx.select(voiced)
    selected_ids = {l.line_id for l in selected}
    bound = {l.audio_id for l in voiced}
    orphans = [] if (ctx.only or ctx.limit is not None) else [a for a in audio.values() if a.id not in bound and a.wav16 and not a.error]

    def wav16(aid: str) -> str:
        return str(ctx.paths.abs(audio[aid].wav16 or audio[aid].wav))

    asr_items = [(l.audio_id, wav16(l.audio_id)) for l in selected] + [(a.id, wav16(a.id)) for a in orphans]
    texts, asr_backend, asr_reason = run_asr(ctx, asr_items, src_lang) if asr_items else ({}, "", "")
    asr_note = "mock ASR cannot transcribe original clips" if asr_backend.startswith("mock") else ""

    matcher = Matcher(lines, cfg.neighbour_window, cfg.global_topk)
    records: dict[str, LinkRecord] = {}
    proposals: list[tuple[str, str, float]] = []
    for l in selected:
        if asr_reason:
            records[l.line_id] = LinkRecord(line_id=l.line_id, audio_id=l.audio_id, status="review", reasons=[f"asr unavailable: {asr_reason}"])
            continue
        rec, prop = judge(l, texts.get(l.audio_id), matcher, cfg.accept_score, cfg.review_score, cfg.auto_repair, asr_note)
        records[l.line_id] = rec
        if prop is not None:
            proposals.append((l.line_id, prop[0], prop[1]))

    owner: dict[str, str | None] = {l.line_id: l.audio_id for l in voiced}
    relinked = apply_proposals(records, proposals, owner) if proposals else 0

    # unbound voice files: bind to an unvoiced line they clearly match
    orphans_bound = 0
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

    # speaker embeddings over every owned clip
    owned = sorted({aid for aid in owner.values() if aid})
    embs, emb_backend, emb_reason = run_embed(ctx, [(aid, wav16(aid)) for aid in owned]) if owned else ({}, "", "")
    speaker_of = {l.line_id: l.speaker_id for l in lines}
    spk_summary: dict[str, Any] = {}
    if embs:
        spk_summary = speaker_checks(records, owner, speaker_of, embs, cfg.cluster_distance, cfg.outlier_consistency, selected_ids)
    if emb_reason:
        ctx.log.warning("speaker check incomplete: %s", emb_reason)

    # merge with records of lines not judged this run
    existing = {r.line_id: r for r in read_jsonl(ctx.paths.link, LinkRecord)} if ctx.paths.link.exists() else {}
    existing.update(records)
    order = {l.line_id: i for i, l in enumerate(lines)}
    merged = sorted(existing.values(), key=lambda r: order.get(r.line_id, 1 << 30))
    write_jsonl(ctx.paths.link, merged)
    queue = [
        {"line_id": r.line_id, "status": r.status, "reasons": r.reasons, "match_score": r.match_score, "asr_text": r.asr_text, "best_alt_line_id": r.best_alt_line_id, "audio_id": r.audio_id}
        for r in merged
        if r.status != "ok"
    ]
    write_jsonl(ctx.paths.review_queue, queue)

    judged = [records[l] for l in selected_ids if l in records]
    counts = Counter(r.status for r in judged)
    scores = [r.match_score for r in judged if r.asr_text]
    summary: dict[str, Any] = {
        "lines_judged": len(judged),
        "ok": counts.get("ok", 0),
        "review": counts.get("review", 0),
        "unmatched": counts.get("unmatched", 0),
        "coverage_ok": round(counts.get("ok", 0) / len(judged), 3) if judged else 0.0,
        "mean_match": round(float(np.mean(scores)), 3) if scores else 0.0,
        "relinked": relinked,
        "orphans": len(orphans),
        "orphans_bound": orphans_bound,
        "asr": asr_backend or f"unavailable: {asr_reason}",
        "embedder": emb_backend or (f"unavailable: {emb_reason}" if emb_reason else "-"),
        **spk_summary,
        "review_queue": len(queue),
    }
    return summary

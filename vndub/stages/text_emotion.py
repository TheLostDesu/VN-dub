"""text_emotion: delivery from text for lines without original audio (mode B).

For every speaker line that has no original voice (``has_voice`` false after link rebindings; narrator
lines, ``voice_sustain`` continuations and lines with audio are skipped) it writes a *synthetic expression
passport* ``analysis/<line_id>.json`` (schema ``manifest.io.Passport``)::

    {"line_id", "source": "text", "duration_s": 0.0,
     "delivery": {"emotion", "intensity", "tempo", "loudness", "events", "confidence",
                  "layer": "rule"|"cue"|"llm"|"neutral", "cues": [...], "arc", "notes", "layers": {...}},
     "style": <StyleSpec.to_dict()>, "backends": {...}, "missing": {"audio": ...}}

Passports of lines that have original audio (written by ``analyze``) are never touched.

Evidence layers (``vndub/text/delivery.py``), merged rule -> cue -> llm; a later layer overrides an
earlier one only when its confidence is >= ``text_emotion.min_confidence``, otherwise the line stays
generic (neutral, intensity 0.3, normal tempo and loudness):

1. rule  -- punctuation and markers of the line (and its translation when there is one);
2. cue   -- speech verbs / manner adverbs in the narration lines around the line
            (``configs/delivery_cues.yaml``, attribution below);
3. llm   -- the ``delivery`` object ``translate`` asked the local LLM for (``TranslationRecord.delivery``).
            This stage never loads a model.

Cue attribution (a cue describes ONE dialogue line): narration lines within ``narration_window`` lines of a
dialogue line, same scene, are scanned in the source language and, when a translation of the narration
exists, the target language. A narration line's cues apply to
  1. the dialogue line of the speaker whose name occurs in the narration (the name mentioned first wins among
     speakers who are around; a narration that names only characters who are not around is ignored);
     among that speaker's lines, and without any name among all dialogue lines, to
  2. the NEAREST dialogue line. On a tie the cue's ``position`` decides (``after`` = tag after the speech:
     the preceding line, ``before`` = lead-in: the following line); ``any`` and no hint: the preceding line.
     A cue whose position argues against the line it ended up on loses 0.05 (0.1 when that was the only
     line around).
Distance (0.05 per line beyond the first), ambiguity (a tie with no name: 0.1) and the name bonus (+0.1) adjust
the cue's confidence; it is judged against ``min_confidence`` after the adjustment. Cues are attributed to
speaker lines of any kind (voiced ones included) so that a cue about a voiced heroine never leaks onto the
unvoiced protagonist standing next to her.

Runs without any translation file (the app runs it before ``translate``); item-level cached.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence

from vndub.manifest.io import Character, LineRecord, Passport, TranslationRecord, read_json, read_jsonl, write_json
from vndub.stages.common import StageContext, StageError
from vndub.text import ja
from vndub.text.delivery import (  # noqa: F401  (load_cues / save_cues are this stage's public helpers)
    CompiledCues,
    CueHit,
    Delivery,
    compile_cues,
    cue_layer,
    default_cues_path,
    llm_layer,
    load_cues,
    merge_layers,
    rule_pass,
    save_cues,
    scan_cues,
    to_style,
    validate_cues,
    validate_delivery,
)
from vndub.util import hash_obj

NAME = "text_emotion"
VERSION = "1"
CONFIG_SECTIONS = ("text_emotion",)

NARRATOR = "narrator"
PASSPORT_MISSING = {"audio": "no original audio: delivery derived from text"}


# ----------------------------------------------------------------------------
# which lines
# ----------------------------------------------------------------------------


def is_dialogue(l: LineRecord) -> bool:
    """A line somebody speaks (non-narrator speaker with text)."""
    return l.speaker_id != NARRATOR and bool((l.text_src or "").strip()) and "menu" not in l.flags


def is_narration(l: LineRecord) -> bool:
    return l.speaker_id == NARRATOR and bool((l.text_src or "").strip()) and "menu" not in l.flags and not l.choices


def unvoiced_speaker_lines(lines: Sequence[LineRecord], voiced_ids: set[str]) -> list[LineRecord]:
    """Speaker lines nobody voices: the lines text_emotion describes and (optionally) translate adds.

    Narrator lines, ``voice_sustain`` continuations (covered by the head line's audio) and empty lines are excluded.
    """
    return [l for l in lines if is_dialogue(l) and l.line_id not in voiced_ids and "voice_sustain" not in l.flags]


def cues_path(cfg: Any) -> Path:
    """Cue table path from ``text_emotion.cues``; falls back to the bundled default when that file is missing."""
    p = cfg.resolve(cfg.text_emotion.cues)
    return p if p.exists() else default_cues_path()


def _tgt_text(rec: TranslationRecord | None, src: str) -> str:
    if rec is None or not rec.text_tgt:
        return ""
    if "translate_failed" in rec.flags or "translate_missing" in rec.flags or rec.text_tgt == src:
        return ""
    return rec.text_tgt


def load_translations(ctx: StageContext, tgt: str) -> dict[str, TranslationRecord]:
    p = ctx.paths.translation(tgt)
    return {r.line_id: r for r in read_jsonl(p, TranslationRecord)} if p.exists() else {}


# ----------------------------------------------------------------------------
# narration cues -> dialogue lines
# ----------------------------------------------------------------------------


def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "").lower()


def _name_ok(n: str) -> bool:
    """A usable name: two characters, or one kanji (蓮, 凛); a single kana would match everywhere."""
    n = n.strip()
    return len(n) >= 2 or (len(n) == 1 and ja._is_kanji(n))


class _Names:
    """Speaker names as they may appear in narration (source: substring; target: word stem prefix)."""

    def __init__(self, lines: Sequence[LineRecord], chars: Mapping[str, Character]) -> None:
        src: dict[str, set[str]] = {}
        tgt: dict[str, set[str]] = {}
        for sid, c in chars.items():
            if sid == NARRATOR:
                continue
            if c.names.get("src"):
                src.setdefault(sid, set()).add(c.names["src"])
            if c.names.get("tgt"):
                tgt.setdefault(sid, set()).add(c.names["tgt"])
        for l in lines:
            if l.speaker_id != NARRATOR and l.speaker_raw:
                src.setdefault(l.speaker_id, set()).add(l.speaker_raw)
        self.src = {sid: sorted({_nfkc(n) for n in ns if _name_ok(n)}) for sid, ns in src.items()}
        self.tgt = {sid: sorted({_nfkc(n) for n in ns if len(n.strip()) >= 3}) for sid, ns in tgt.items()}

    def first_mentions(self, text_src: str, text_tgt: str) -> dict[str, int]:
        """speaker_id -> position of the first mention (earliest wins) in the narration."""
        s, t = _nfkc(text_src), _nfkc(text_tgt).replace("ё", "е")
        found: dict[str, int] = {}
        for sid, names in self.src.items():
            pos = [s.find(n) for n in names if n and s.find(n) >= 0]
            if pos:
                found[sid] = min(pos)
        for sid, names in self.tgt.items():
            if not t:
                break
            for n in names:
                stem = n.replace("ё", "е")
                stem = stem[: max(3, len(stem) - 1)]  # declined forms: Юи -> Юи, Сакура -> Сакур(ы/е/у)
                i = t.find(stem)
                while i >= 0 and i > 0 and t[i - 1].isalpha():
                    i = t.find(stem, i + 1)
                if i >= 0:
                    found[sid] = min(found.get(sid, 1 << 30), i)
        return found


def _line_position(hits: Sequence[CueHit]) -> str:
    pos = {h.entry.get("position") or "any" for h in hits} - {"any"}
    return next(iter(pos)) if len(pos) == 1 else "any"


def attribute_cues(
    lines: Sequence[LineRecord],
    chars: Mapping[str, Character],
    cues: CompiledCues,
    src_lang: str = "ja",
    tgt_lang: str = "ru",
    tgt_texts: Mapping[str, str] | None = None,
    window: int = 3,
    targets: Collection[str] | None = None,
) -> dict[str, list[CueHit]]:
    """Cue hits of the narration lines, attributed to the dialogue line each one describes (see module doc).

    ``targets``: only narration within ``window`` lines of one of these line ids is scanned (the others cannot
    matter to them); ``None`` scans all narration.
    """
    tgt_texts = tgt_texts or {}
    target_set = set(targets) if targets is not None else None
    names = _Names(lines, chars)
    runs: list[list[LineRecord]] = []
    for l in lines:
        if not runs or runs[-1][0].scene != l.scene:
            runs.append([])
        runs[-1].append(l)
    out: dict[str, list[CueHit]] = {}
    for run in runs:
        near_idx = {j for j, l in enumerate(run) if target_set is None or l.line_id in target_set}
        for i, n in enumerate(run):
            if not is_narration(n):
                continue
            if target_set is not None and not any(j in near_idx for j in range(i - window, i + window + 1)):
                continue
            hits = scan_cues(n.text_src, src_lang, cues)
            tt = tgt_texts.get(n.line_id, "")
            if tt and tgt_lang != src_lang:
                hits += scan_cues(tt, tgt_lang, cues)
            if not hits:
                continue
            cands = [(j, run[j]) for j in range(max(0, i - window), min(len(run), i + window + 1)) if j != i and is_dialogue(run[j])]
            if not cands:
                continue
            mentions = names.first_mentions(n.text_src, tt)
            by_name = False
            if mentions:
                around = [sid for sid in sorted(mentions, key=mentions.get) if any(l.speaker_id == sid for _, l in cands)]
                if not around:
                    continue  # the narration is about somebody who is not speaking here
                cands = [(j, l) for j, l in cands if l.speaker_id == around[0]]
                by_name = True
            pref = _line_position(hits)
            prev = sorted((c for c in cands if c[0] < i), key=lambda c: i - c[0])
            nxt = sorted((c for c in cands if c[0] > i), key=lambda c: c[0] - i)
            penalty = 0.0
            if prev and nxt:
                dp, dn = i - prev[0][0], nxt[0][0] - i
                if dp != dn:
                    pick = prev[0] if dp < dn else nxt[0]
                    if (pref == "after" and pick is nxt[0]) or (pref == "before" and pick is prev[0]):
                        penalty = 0.05  # the nearer line is on the side the cue argues against
                else:  # equally near: the cue's position decides, and without a name that is still a guess
                    pick = nxt[0] if pref == "before" else prev[0]
                    penalty = 0.0 if (by_name or pref != "any") else 0.1
            else:
                pick = (prev or nxt)[0]
                if (pref == "after" and not prev) or (pref == "before" and not nxt):
                    penalty = 0.1
            j, target = pick
            dist = abs(j - i)
            adj = (0.1 if by_name else 0.0) - penalty - 0.05 * (dist - 1)
            for h in hits:
                h.line_id, h.distance, h.by_name = n.line_id, dist, by_name
                h.confidence = round(max(0.0, min(0.95, float(h.entry["confidence"]) + adj)), 3)
                out.setdefault(target.line_id, []).append(h)
    return out


# ----------------------------------------------------------------------------
# per-line delivery
# ----------------------------------------------------------------------------


def build_delivery(
    line: LineRecord,
    hits: Sequence[CueHit],
    tgt_text: str = "",
    llm_delivery: Mapping[str, Any] | None = None,
    lexicon: Mapping[str, str] | None = None,
    min_confidence: float = 0.6,
) -> Delivery:
    """rule -> cue -> llm for one line (see :func:`vndub.text.delivery.merge_layers`)."""
    layers = [rule_pass(line.text_src, tgt_text, lexicon), cue_layer(hits), llm_layer(llm_delivery)]
    return merge_layers(layers, min_confidence)


def text_passport(line: LineRecord, d: Delivery, note: str = "") -> Passport:
    """The synthetic passport. ``note`` is extra style notes; synth adds the translation's own ``notes`` itself."""
    layers = [k for k, v in d.layers.items() if v.get("accepted")]
    return Passport(
        line_id=line.line_id,
        duration_s=0.0,
        style=to_style(d, note).to_dict(),
        backends={"text_emotion": "+".join(layers) or "neutral"},
        missing=dict(PASSPORT_MISSING),
        source="text",
        delivery=d.to_dict(),
    )


def window_context(lines: Sequence[LineRecord], idx: int, window: int, tgt_texts: Mapping[str, str]) -> list[Any]:
    """Cache-key view of the neighbourhood of ``lines[idx]``: scene lines within ``window``, narration with its texts."""
    scene = lines[idx].scene
    out: list[Any] = []
    for j in range(max(0, idx - window), min(len(lines), idx + window + 1)):
        l = lines[j]
        if j == idx or l.scene != scene:
            continue
        out.append([j - idx, l.speaker_id, l.text_src if l.speaker_id == NARRATOR else "", tgt_texts.get(l.line_id, "") if l.speaker_id == NARRATOR else ""])
    return out


# ----------------------------------------------------------------------------
# stage
# ----------------------------------------------------------------------------


def run(ctx: StageContext) -> dict[str, Any]:
    te = ctx.cfg.text_emotion
    if not te.enabled:
        return {"enabled": False, "lines": 0}
    lines = ctx.lines()
    voiced_ids = {l.line_id for l in ctx.voiced_lines(apply_only=False)}
    chars = ctx.characters()
    proj = ctx.project
    src_lang = (proj.src_lang if proj else None) or ctx.cfg.project.src_lang
    tgt_lang = (proj.tgt_lang if proj else None) or ctx.cfg.project.tgt_lang

    cpath = cues_path(ctx.cfg)
    try:
        table = load_cues(cpath)
    except FileNotFoundError:
        ctx.log.warning("delivery cue table %s not found: narration cues are off", cpath)
        table = {}
    except ValueError as e:
        raise StageError(f"delivery cue table is invalid: {e}") from e
    cues = compile_cues(table)
    lexicon = ja.load_interjections(ctx.cfg.resolve(ctx.cfg.translate.interjections))

    targets = unvoiced_speaker_lines(lines, voiced_ids)
    if not te.only_unvoiced:
        # lines with audio but no passport (analyze did not run / failed) get a text passport as a stop-gap
        targets = [l for l in lines if is_dialogue(l) and "voice_sustain" not in l.flags and (l.line_id not in voiced_ids or not ctx.paths.passport(l.line_id).exists())]
    targets = ctx.select(targets, voiced_only=False)

    trs = load_translations(ctx, tgt_lang)
    src_of = {l.line_id: l.text_src for l in lines}
    pos_of = {l.line_id: i for i, l in enumerate(lines)}
    tgt_texts = {lid: _tgt_text(r, src_of.get(lid, "")) for lid, r in trs.items()}
    use_llm = te.llm != "off"
    hits_by_line = attribute_cues(lines, chars, cues, src_lang, tgt_lang, tgt_texts, te.narration_window, {l.line_id for l in targets})
    lex_hash = hash_obj(lexicon) if lexicon else ""
    cache = ctx.cache(NAME, VERSION, *CONFIG_SECTIONS)

    layer_counts: Counter[str] = Counter()
    emotions: Counter[str] = Counter()
    written = cached = kept_audio = llm_have = 0
    total = len(targets)
    for i, line in enumerate(targets):
        ctx.progress(i, total, line.line_id)
        lid = line.line_id
        pp = ctx.paths.passport(lid)
        prev = read_json(pp) if pp.exists() else None
        if prev is not None and prev.get("source") != "text":
            kept_audio += 1  # an audio passport is never overwritten
            continue
        rec = trs.get(lid)
        tgt_text = tgt_texts.get(lid, "")
        llm_d = ((rec.model_extra or {}).get("delivery") if (rec is not None and use_llm) else None)
        llm_d = validate_delivery(llm_d, lid) if llm_d else None
        llm_have += llm_d is not None
        hits = hits_by_line.get(lid, [])
        key = cache.key(
            line.text_src, line.speaker_id, tgt_text, llm_d,
            [[h.line_id, h.lang, h.cue, h.confidence] for h in hits],
            window_context(lines, pos_of[lid], te.narration_window, tgt_texts), cues.hash,
            lex_hash,
        )
        d: Delivery | None = None
        if prev is not None and cache.fresh(lid, key, [pp]):
            cached += 1
            d = Delivery.from_dict(prev.get("delivery") or {})
        else:
            d = build_delivery(line, hits, tgt_text, llm_d, lexicon, te.min_confidence)
            write_json(pp, text_passport(line, d))
            cache.put(lid, key)
            written += 1
        layer_counts[d.layer] += 1
        emotions[d.emotion] += 1
    ctx.progress(total, total, None)
    narration_cued = len({h.line_id for hs in hits_by_line.values() for h in hs})
    return {
        "lines": total,
        "written": written,
        "cached": cached,
        "kept_audio_passport": kept_audio,
        "layers": dict(layer_counts.most_common()),
        "emotions": dict(emotions.most_common()),
        "narration_lines_with_cues": narration_cued,
        "llm": "off" if not use_llm else f"{llm_have} of {total - kept_audio} lines have an llm delivery",
        "cues": str(cpath),
    }

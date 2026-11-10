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
            (words only: laughter is taken out of both sides)
dur_ratio   candidate / source duration                                 1-min(1,|ln r|/ln 2)
prosody     mean of (f0_corr+1)/2, (energy_corr+1)/2, pause_match       as is
==========  ==========================================================  ===============

The composite is the weighted mean over the terms that are available (weights
renormalized); missing terms are listed per candidate. Every raw metric is
persisted in ``scores/<line_id>.jsonl`` so ``vndub score --reweight`` re-derives
composites and the selection without touching any model.

A metric group whose backend cannot be acquired (VRAM taken by an orphan process, env or weights gone) is reported
through ``ctx.models.record_unavailable("score", group, reason)`` and noted on every candidate that lacks it. When it is a
GATE group (``mos`` -> utmos, ``speaker`` -> spk_sim, ``cer`` -> cer) the gate cannot be evaluated, and a candidate that
passed only because that gate was skipped must not be accepted on its own: the line's selection becomes ``review`` with
the reason (see :func:`unchecked_gates`; it also works from the stored notes, so ``--reweight`` keeps the verdict). Among the
passing takes the fully checked ones rank first (a take without a CER has a composite renormalised without that term and must not
outrank one that passed every gate); a line whose best take has no gate at all that could be evaluated goes to review at once, a new
take could not be judged either. A take with no reference voice to compare with (no voicebank clip, performer clip or original line
could be embedded) has its speaker gate unchecked too (``speaker failed: no reference voice to compare with``). ``--force`` computes
every metric again (the embedding files on disk, keyed by the audio's mtime, are still reused).

Laughter: an ASR model transcribes a laugh as nothing or as random syllables, so the CER is computed over the words
only: :func:`cer_words` takes the laughs (``vndub.text.laughter``) out of the reference and of the transcript
(:func:`cer_over_words` also keeps the plain reading as a floor, for a laugh the ASR heard as a stray syllable). A
line that is nothing but laughter ("Haha!", "Хе-хе.") has no words to check: its candidates get ``cer`` in ``na`` (like the
original recording) and the note :data:`LAUGHTER_NOTE`, the ASR is not asked, no CER gate exists for them and they can pass
on the other gates (``regen_hopeless`` never fires on them: there is no CER). A line with words whose transcript holds no
words (only a laugh, or nothing) keeps the "the ASR heard no words" verdict, an unchecked gate. CER values stored before
this rule (no ``cer_rule`` on the record) of a line that has laughter are recomputed on the next run.

Comparable sides (CER rule 3): the reference is spelled out by translate, GigaAM writes digits, so :func:`comparable` runs the
Russian number / abbreviation expansion on both sides and then turns every run of number words into one digit token (compared by
value: "тремя" and "3" agree); Latin words are transliterated (``ru.translit_latin``) or dropped; a laugh syllable the ASR wrote
for a laugh taken out of the reference is dropped. A reference under ``score.cer_short_chars`` letters tolerates one wrong
character (:func:`cer_gate`). A CER stored under an older rule is recomputed once from the stored transcript (``asr_text``): no
ASR, no synthesis. A take the ASR heard no words in does not pass when another take of the line has a measured CER
(:func:`demote_unheard`); without a passing take the provisional winner is the one closest to passing (:func:`_provisional_rank`).
A take whose sidecar text is not a text the line can have now (:func:`stale_take`) is skipped.

Non-speech sounds (``synth.sounds``, ``vndub.text.sounds``): the same rule as laughter. :func:`cer_reference` and
:func:`cer_words` strip sounds ("Achoo!", "Хр-р-р", ``*yawns*``) from both sides, so a stand-in inside speech never counts against
the words; a line that is nothing but a sound has no CER gate (``cer`` in ``na``, note :data:`SOUND_NOTE`, never hopeless). A
candidate of the ``sample`` backend (a recorded clip that replaced the TTS, ``synth.sounds.only: sample``) is not measured at
all: every metric is ``na``, the note is :data:`SAMPLE_NOTE`, it passes and is selected ``auto``. A line that synth left out
on purpose (``synth.sounds.only: silent``) has no candidates and no selection: it is neither a failure nor a review item (``pack``
lists it as ``sound_silent``).

Resume: the metrics of a candidate whose scoring has not finished are kept in ``scores/_partial.json`` (see
:class:`_Partial`: keyed by candidate id, valid for its file fingerprint and the metric key), written after every metric
group and at most every ``PARTIAL_EVERY_S`` seconds while a group runs; ``run`` starts from them and ``Cand.needs``
therefore asks the models only for what is still missing. The file is removed when the final ``scores/<line_id>.jsonl``
files are written. The emotion group also skips candidates whose ``.e2v.npy`` embedding already exists.

Lines without original audio (``synth.voice_unvoiced``, route P) have nothing to compare the
delivery to: ``emo_sim``, ``adv_delta``, ``prosody`` and ``dur_ratio`` are not computed (their terms
are missing, so the composite is renormalized over utmos / pq / spk_sim / cer) and no gate needs them.
``spk_sim`` is then measured against the performer's reference clips (consistency with the chosen
design) instead of the character's original clips -- the same threshold applies.
"""

from __future__ import annotations

import math
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from vndub.audio import features, vad
from vndub.backends.base import BackendUnavailable
from vndub.backends.score import prosody as pros
from vndub.backends.score.cer import cer as cer_fn
from vndub.backends.score.cer import normalize_for_cer, numbers_to_digits
from vndub.backends.score.speaker_sim import cosine
from vndub.manifest.io import CandidateScore, LineRecord, SelectionRecord, SynthSidecar, load_overrides, read_json, write_json, write_jsonl
from vndub.stages import genutil as gu
from vndub.stages.common import StageError
from vndub.text import laughter, ru, sounds
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

# metric groups that feed a GATE (the others only feed the composite): group -> the metric the gate reads
GATE_GROUPS = {"mos": "utmos", "speaker": "spk_sim", "cer": "cer"}
GATE_LABELS = {"mos": "naturalness (utmos)", "speaker": "speaker similarity (spk_sim)", "cer": "speech intelligibility (cer)"}
# the note of a candidate whose line is nothing but laughter: its CER gate is not applicable (``na``), not unchecked
LAUGHTER_NOTE = "laughter: no words to check"
# the note of a candidate whose line is nothing but a non-speech sound (snoring, a kiss ...): same treatment as laughter
SOUND_NOTE = "sound: no words to check"
# a recorded clip that replaced the TTS (backend "sample"): no metric applies, it passes unchecked
SAMPLE_NOTE = "recorded sample"
SAMPLE_NA = ["utmos", "pq", "spk_sim", "emo_sim", "adv_delta", "cer", "dur_ratio", "prosody"]
# the rule a stored CER was computed under; absent = the old one (laughter counted as text); "2" = laughter out;
# "3" = both sides made comparable (numbers by value, Latin words, laugh syllables): an older value is recomputed from the
# stored transcript (``asr_text``), no ASR and no synthesis needed
CER_RULE = "3"
# the note of a take the ASR heard no words in: its CER gate is unchecked
NO_WORDS = "the ASR heard no words in this take"
NO_WORDS_NOTE = f"cer failed: {NO_WORDS}"
PARTIAL_FILE = "_partial.json"
PARTIAL_EVERY_S = 15.0  # at most this often while a metric group runs (it is also written when a group ends)


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


def unchecked_gates(c: CandidateScore, unavailable: dict[str, str] | None = None) -> dict[str, str]:
    """Gate groups (``mos`` / ``speaker`` / ``cer``) that could not be evaluated for this candidate: its metric is missing
    (and not "not applicable") because the group's backend was unavailable or failed for it. The reason comes from this
    run's ``unavailable`` or from the candidate's stored notes (``"<group> unavailable: ..."`` / ``"<group> failed: ..."``),
    so the verdict survives ``--reweight``."""
    out: dict[str, str] = {}
    na = list((c.model_extra or {}).get("na") or [])
    for group, metric in GATE_GROUPS.items():
        if c.metrics.get(metric) is not None or metric in na:
            continue
        reason = (unavailable or {}).get(group)
        if reason is None:
            for n in c.notes:
                if n.startswith(f"{group} unavailable: ") or n.startswith(f"{group} failed: "):
                    reason = n.split(": ", 1)[1]
                    break
        if reason is not None:
            out[group] = reason
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


def silent_sound_line(cfg: Any, tr: Any, line: Any = None) -> bool:
    from vndub.stages.synth import silent_sound_line as _silent

    return _silent(cfg, tr, line)


def cer_reference(side: SynthSidecar | None, tr: Any, cont: list[Any]) -> str:
    """CER reference: the sidecar's text_plain (what synth actually used), else rebuilt
    from the translation (text_plain_short for the short variant; sustain continuations joined). Non-speech sounds are
    taken out (:func:`raw_cer_reference` keeps them)."""
    raw = raw_cer_reference(side, tr, cont)
    return " ".join(sounds.strip_sounds(raw, "auto").split()) if sounds.has_sound(raw) else raw


def raw_cer_reference(side: SynthSidecar | None, tr: Any, cont: list[Any]) -> str:
    """:func:`cer_reference` before the sounds are taken out."""
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


def cer_words(ref: str, hyp: str) -> tuple[str, str]:
    """(reference, transcript) with the laughter taken out: the CER is about the words, a laugh is not text an ASR can spell
    (``"Haha, no way!"`` -> ``"no way!"``). Both sides are stripped in every script (``lang="auto"``)."""
    return _strip_nonspeech(ref or ""), _strip_nonspeech(hyp or "")


def _strip_nonspeech(text: str) -> str:
    out = laughter.strip_laughter(text, "auto")
    return " ".join(sounds.strip_sounds(out, "auto").split()) if sounds.has_sound(out) else out


_LATIN_WORD = re.compile(r"\b[A-Za-z]+(?:'[A-Za-z]+)?\b")  # a whole Latin word ("дa" with a Latin "a" is not one)
_CYR = re.compile(r"[а-яёА-ЯЁ]")
# a laugh syllable as the ASR writes it once normalized ("ха", "хаха", "хех", "hah")
_LAUGH_SYLLABLE = re.compile(r"^(?:[хxh][аеиоэыaeio])+[хxh]?$")


def comparable(text: str, lang: str = "ru") -> str:
    """A text as the CER compares it, the same transform on the reference and on the transcript.

    For Russian: abbreviations and numbers are expanded (the reference was spelled out by translate, GigaAM writes "16",
    "10:30", "100%"), then every run of number words becomes one digit token (:func:`numbers_to_digits`: "тремя" and the
    ASR's "3" -> "три" then compare by value, not by case ending), and a Latin word is transliterated (``ru.translit_latin``)
    or, without it, dropped from both sides (a name the translator left in Latin, the ASR writes in Cyrillic)."""
    t = text or ""
    if lang != "ru":
        return normalize_for_cer(t)
    try:
        t = ru.normalize_numbers(ru.expand_abbreviations(t))
    except Exception:  # never let a normalization quirk end the scoring: compare what there is
        pass
    if _LATIN_WORD.search(t):
        translit = getattr(ru, "translit_latin", None)
        if translit is not None:
            t = translit(t)
        elif _CYR.search(_LATIN_WORD.sub(" ", t)):  # a line that is all Latin keeps its words: there would be nothing left
            t = _LATIN_WORD.sub(" ", t)
    return numbers_to_digits(normalize_for_cer(t))


def cer_texts(ref: str, hyp: str, lang: str = "ru") -> tuple[str, str]:
    """(reference, transcript) exactly as the CER compares them: laughs and sounds out (:func:`cer_words`), both made
    :func:`comparable`, and when a laugh was taken out of the reference, the laugh syllables the ASR wrote for it ("ха")."""
    ref_w, hyp_w = cer_words(ref, hyp)
    r, h = comparable(ref_w, lang), comparable(hyp_w, lang)
    if laughter.has_laughter(ref or ""):
        h = " ".join(w for w in h.split() if not _LAUGH_SYLLABLE.match(w))
    if lang == "ru" and getattr(ru, "translit_latin", None) is None and _CYR.search(_LATIN_WORD.sub(" ", ref_w)):
        # no transliteration: the Latin words left the reference, so as many transcript words that the reference does not
        # have (the ASR wrote the name in Cyrillic) leave the transcript; at most that many, so a misread elsewhere still counts
        k = len(_LATIN_WORD.findall(ref_w))
        known = set(r.split())
        out = []
        for w in h.split():
            if k > 0 and w not in known:
                k -= 1
                continue
            out.append(w)
        h = " ".join(out)
    return r, h


def cer_over_words(ref: str, hyp: str, words: tuple[str, str] | None = None, lang: str = "ru") -> float:
    """CER over the words of the line, never worse than the CER of the same texts with the laughs kept.

    ``words`` is the pair :func:`cer_texts` returns. The laughs are taken out of both sides; an ASR that heard a laugh as a
    stray syllable ("ха да ну" for "Ха-ха, да ну!") has that syllable dropped too. The reading that keeps the laughs is the
    floor, so a laugh is never held against the line."""
    r, h = words or cer_texts(ref, hyp, lang)
    value = cer_fn(r, h)
    if value > 0:
        value = min(value, cer_fn(comparable(ref, lang), comparable(hyp, lang)))
    return value


def cer_gate(cer_max: float, ref: str, short_chars: int) -> float:
    """The CER gate of a line: a reference under ``short_chars`` letters tolerates one wrong character ("Да." with one
    letter off is 0.5, which no 0.10 gate passes)."""
    n = len(normalize_for_cer(ref).replace(" ", ""))
    if short_chars > 0 and 0 < n < short_chars:
        return max(cer_max, round(1.0 / n, 4))
    return cer_max


def is_laughter_line(c: CandidateScore) -> bool:
    """Is this candidate's line nothing but laughter (its CER gate is not applicable, see :data:`LAUGHTER_NOTE`)?"""
    return LAUGHTER_NOTE in c.notes or SOUND_NOTE in c.notes


def is_sample(c: CandidateScore) -> bool:
    """Is this candidate a recorded clip that replaced the TTS (``synth.sounds.only: sample``)?"""
    return c.backend == "sample" or SAMPLE_NOTE in c.notes


def spk_threshold(cfg: Any, calibration: dict[str, Any] | None) -> tuple[float, str]:
    g = cfg.score.gates
    if isinstance(g.spk_sim_min, (int, float)):
        return float(g.spk_sim_min), "configured"
    if calibration and calibration.get("threshold") is not None:
        return float(calibration["threshold"]), f"calibrated ({calibration.get('embedder')}, EER {calibration.get('eer')})"
    return float(g.spk_sim_default), "default (no calibration)"


def apply_scores(c: CandidateScore, cfg: Any, thr: float, interjection: bool) -> CandidateScore:
    if is_sample(c):  # a recording, not a synthesis: nothing to measure, nothing to gate
        c.composite, c.gates, c.passed = None, {}, True
        setattr(c, "missing", [])
        setattr(c, "failures", "")
        return c
    comp, missing = composite(c.metrics, cfg.score.weights)
    c.composite = comp
    gates = cfg.score.gates
    cer_max = cfg.score.interjection_cer_max if interjection else gates.cer_max
    ref = (c.model_extra or {}).get("cer_ref")
    if ref:
        cer_max = cer_gate(float(cer_max), str(ref), int(getattr(cfg.score, "cer_short_chars", 0) or 0))
    g = evaluate_gates(c.metrics, gates, thr, cer_max, c.route)
    c.gates = g
    c.passed = bool(g) and all(g.values()) and comp is not None
    setattr(c, "missing", sorted(set(missing)))
    setattr(c, "failures", describe_failures(c.metrics, g, gates, thr, cer_max))
    setattr(c, "deficit", gate_deficit(c.metrics, g, gates, thr, cer_max))
    return c


def gate_deficit(m: dict[str, Any], g: dict[str, bool], gates: Any, spk_thr: float, cer_max: float) -> float:
    """How far the worst failed gate misses, relative to its threshold (0 = every gate passes): ranks the takes of a line
    none of which passes, so the provisional winner is the one closest to passing, not the prettiest garble."""
    worst = 0.0
    for k, ok in g.items():
        if ok:
            continue
        if k == "utmos":
            d = (gates.utmos_min - float(m["utmos"])) / max(float(gates.utmos_min), 1e-6)
        elif k == "cer":
            d = (float(m["cer"]) - cer_max) / max(float(cer_max), 0.01)
        elif k == "spk_sim":
            d = (spk_thr - float(m["spk_sim"])) / max(abs(spk_thr), 1e-6)
        elif k == "dur":
            lo, hi = gates.dur_ratio
            r = float(m["dur_ratio"])
            d = (lo - r) / max(lo, 1e-6) if r < lo else (r - hi) / max(hi, 1e-6)
        else:
            continue
        worst = max(worst, d)
    return round(worst, 4)


def unheard(c: CandidateScore) -> bool:
    """Did the ASR hear no words in this take (its CER gate is unchecked, not measured)?"""
    return c.metrics.get("cer") is None and NO_WORDS_NOTE in c.notes


def demote_unheard(pool: list[CandidateScore]) -> None:
    """A take the ASR heard no words in does not pass when another take of the line has a measured CER: the empty transcript
    removed its CER gate, which made silence or garble beat a take that narrowly failed (and stopped the regen rounds)."""
    if not any(c.metrics.get("cer") is not None for c in pool):
        return  # every take is unchecked: the "gate not checked" review stays
    for c in pool:
        if c.passed and unheard(c) and not is_sample(c):
            c.passed = False
            prev = (c.model_extra or {}).get("failures") or ""
            setattr(c, "failures", ", ".join(x for x in (prev, "cer: the ASR heard no words") if x))


def _provisional_rank(c: CandidateScore) -> tuple[Any, ...]:
    """Order of the takes of a line that has no passing take: measured before unheard, judged before unjudged, then fewer
    failed gates, then the smaller worst deficit, the composite last."""
    g = c.gates or {}
    deficit = (c.model_extra or {}).get("deficit")
    return (not unheard(c), bool(g), -sum(1 for ok in g.values() if not ok), -float(deficit or 0.0), c.composite if c.composite is not None else -1.0)


def regen_hopeless(pool: list[CandidateScore], best: CandidateScore, cer_ceiling: float | None) -> str | None:
    """Why another synth round would not help this line, or None when it might.

    A new seed changes the delivery, not the words: when the best take fails the CER gate and even the most intelligible
    take of the line misreads the text by more than ``cer_ceiling`` (``score.regen_cer_max``, never below the line's
    gate), the text or the voice is the problem, and the line goes to a person at once instead of after
    ``regen_rounds`` rounds of the same. A laughter-only line has no CER gate (its candidates have no ``cer``), so it is
    never hopeless on that account.
    """
    if cer_ceiling is None or is_laughter_line(best):
        return None
    gates = getattr(best, "gates", None) or {}
    if gates.get("cer", True):
        return None
    cers = [float(c.metrics["cer"]) for c in pool if c.metrics.get("cer") is not None]
    if not cers or min(cers) <= cer_ceiling:
        return None
    return f"every take misreads the text (cer {min(cers):.2f}-{max(cers):.2f}, more than {cer_ceiling:.2f})"


def select(
    line_id: str,
    cands: list[CandidateScore],
    sidecars: dict[str, SynthSidecar | None],
    overrides: list[Any],
    regen_rounds: int,
    unavailable: dict[str, str] | None = None,
    src_wav: str | None = None,
    n_failed: int = 0,
    regen_cer_max: float | None = None,
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
    measured = any(c.metrics.get("cer") is not None for c in pool)  # (run() also marks the records: demote_unheard)
    passing = [c for c in pool if c.passed and not (measured and unheard(c) and not is_sample(c))]
    if passing:
        # a take whose gate could not be checked has a composite renormalised without that term, so it can outrank a take
        # that passed every gate: rank the fully checked takes first, the composite only decides among equals
        open_gates = {c.wav: unchecked_gates(c, unavailable) for c in passing}
        best = max(passing, key=lambda c: (not open_gates[c.wav], c.composite or 0.0))
        skipped = open_gates[best.wav]
        if skipped:
            # it passed the gates that ran; a gate that could not run (no backend) may have caught it: a person decides
            what = "; ".join(f"{GATE_LABELS[g]}: {r[:100]}" for g, r in skipped.items())
            return rec(best, "review", f"best of {len(passing)} passing, but not accepted automatically: gate not checked ({what})")
        return rec(best, "auto", f"best of {len(passing)} passing{unavail_note}")
    best = max(pool, key=_provisional_rank)
    if not best.gates:
        # not one gate ran (every metric group unavailable, no reference audio): a new take cannot be judged either
        return rec(best, "review", f"no gate could be evaluated: nothing to judge the takes by{unavail_note}")
    why = (best.model_extra or {}).get("failures") or "no gate could be evaluated"
    if is_laughter_line(best):
        why += " (laughter only, no words to check)" if LAUGHTER_NOTE in best.notes else " (only a sound, no words to check)"
    if best.metrics.get("dur_ratio") is not None and float(best.metrics["dur_ratio"]) > 1.25:
        why += "; try text_tgt_short"
    if rnd < regen_rounds:
        hopeless = regen_hopeless(pool, best, regen_cer_max)
        if hopeless is None:
            return rec(best, "regen", f"no candidate passes the gates (best fails: {why}){unavail_note}")
        return rec(best, "review", f"no candidate passes the gates and a new take would not help: {hopeless} (best fails: {why}){unavail_note}")
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

    @property
    def has_ref_audio(self) -> bool:
        """Is there an original recording of the line to compare the delivery with?"""
        return bool(self.line.src_wav or self.line.src_wav16) and self.passport.get("source") != "text"

    def applies(self, metric: str, off: Any = ()) -> bool:
        """Is ``metric`` part of this candidate's work at all (not switched off, not marked n/a, a reference exists)?"""
        if metric in off or metric in self.na:
            return False
        return not (metric in REFERENCE_METRICS and not self.has_ref_audio)

    def needs(self, metric: str) -> bool:
        """Is ``metric`` still to be computed? A record restored from ``scores/_partial.json`` already holds the metrics of the
        groups that finished before a kill, so only the missing ones are asked of the models."""
        return self.applies(metric) and self.rec.metrics.get(metric) is None

    @property
    def na(self) -> list[str]:
        return list((self.rec.model_extra or {}).get("na") or [])


# metrics that compare a candidate with the original recording of its line
REFERENCE_METRICS = ("emo_sim", "adv_delta", "prosody")
# the metrics that count as one unit of work each in the progress (a unit per candidate per metric)
UNIT_METRICS = ("utmos", "pq", "spk_sim", "emo_sim", "adv_delta", "cer", "prosody")
# group -> the metric a candidate lacks when the group did not run (for the notes)
GROUP_METRIC = {"mos": "utmos", "aesthetics": "pq", "speaker": "spk_sim", "emotion": "emo_sim", "adv": "adv_delta", "cer": "cer"}
_STALE_NOTE = re.compile(r"^[a-z_]+ (unavailable|failed): ")


def native_speed(side: SynthSidecar | None) -> float:
    """The speed multiplier the TTS backend itself applied (tuning), so the duration check expects it."""
    plan = (((side.params or {}).get("tuning") or {}).get("plan") or {}) if side is not None else {}
    return float(plan.get("speed") or 1.0) if plan.get("speed_native") else 1.0


class _Progress:
    """``ctx.progress`` over the metric computations: one unit per candidate per metric.

    ``total`` is ALL the units of the stage, the ones already computed (restored from the final scores or from the partial
    file) included, and ``done`` starts at those: the first report is ``(cached, total)``, then real work moves it. ``total``
    never shrinks to the remaining units (a resumed run used to report only what was left, which made the app's throughput
    store think the stage was tiny and fast)."""

    def __init__(self, ctx: Any, total: int, done: int = 0) -> None:
        self.ctx, self.total = ctx, max(0, int(total))
        self.done = min(max(0, int(done)), self.total)
        if self.total:
            ctx.progress(self.done, self.total, None)

    def add(self, n: int, item: str | None = None) -> None:
        if self.total and n > 0:
            self.done = min(self.total, self.done + n)
            self.ctx.progress(self.done, self.total, item)

    def chunk(self, part: list[dict[str, Any]]) -> None:
        """A chunk of backend items; the reference-side items (``orig|...``, ``ref|...``) are not units."""
        real = [it for it in part if not str(it.get("id", "")).startswith(("orig|", "ref|", "perf|"))]
        self.add(len(real), str(real[-1].get("id", "")).split("|")[0] if real else None)

    def finish(self) -> None:
        """Units nobody could compute (an unavailable group) are done too."""
        if self.total and self.done < self.total:
            self.add(self.total - self.done)


def _units(cands: list[Cand], off: set[str]) -> tuple[int, int]:
    """(all metric units of the stage, the ones already computed)."""
    total = done = 0
    for c in cands:
        for m in UNIT_METRICS:
            if c.applies(m, off):
                total += 1
                done += c.rec.metrics.get(m) is not None
    return total, done


class _Partial:
    """Metrics of candidates whose scoring was not finished, in ``scores/_partial.json``.

    One file for the stage: ``{"version": 1, "mkey": <metric key>, "items": {<candidate id>: <the candidate's score record>}}``.
    An entry is only used for the same file (``fp`` = size + mtime of the wav) and the same metric key, so a regenerated
    candidate or a changed model setting starts clean. It is written after every metric group and, while a group runs, at
    most every ``PARTIAL_EVERY_S`` seconds (a hard kill loses seconds, not the group). :meth:`forget` removes the entries
    of candidates whose final ``scores/<line_id>.jsonl`` has been written."""

    def __init__(self, path: Path, mkey: str, clock: Any = time.monotonic) -> None:
        self.path, self.mkey, self._clock = Path(path), mkey, clock
        self.items: dict[str, dict[str, Any]] = {}
        try:
            raw = read_json(self.path, {}) or {}
            if isinstance(raw, dict) and raw.get("mkey") == mkey and isinstance(raw.get("items"), dict):
                self.items = dict(raw["items"])
        except Exception:  # a torn or foreign file is just no resume data
            self.items = {}
        self._dirty: dict[str, Cand] = {}
        self._last = clock()

    @classmethod
    def open(cls, ctx: Any) -> "_Partial":
        return cls(ctx.paths.scores_dir / PARTIAL_FILE, _metric_key(ctx.cfg))

    def lookup(self, cid: str, fp: str) -> dict[str, Any] | None:
        d = self.items.get(cid)
        return d if d and d.get("fp") == fp and d.get("mkey") == self.mkey else None

    def touch(self, c: Cand) -> None:
        self._dirty[c.cid] = c

    def checkpoint(self, force: bool = False) -> None:
        if not self._dirty or (not force and self._clock() - self._last < PARTIAL_EVERY_S):
            return
        for cid, c in self._dirty.items():
            self.items[cid] = c.rec.model_dump(mode="json")
        self._dirty.clear()
        write_json(self.path, {"version": 1, "mkey": self.mkey, "items": self.items}, indent=False)
        self._last = self._clock()

    def forget(self, cids: Any) -> None:
        gone = [cid for cid in cids if cid in self.items]
        for cid in cids:
            self._dirty.pop(cid, None)
        for cid in gone:
            del self.items[cid]
        if not self.items:
            self.path.unlink(missing_ok=True)
        elif gone:
            write_json(self.path, {"version": 1, "mkey": self.mkey, "items": self.items}, indent=False)


def _metric_key(cfg: Any) -> str:
    s = cfg.score
    return hash_obj([VERSION, s.mos, s.mos_fallback, s.aesthetics, s.speaker, s.emotion, s.adv, s.cer_asr, s.cer_asr_fallback, cfg.analyze.f0])


def run(ctx: Any) -> dict[str, Any]:
    cfg = ctx.cfg
    lines = ctx.dub_lines()
    if not lines:
        raise StageError("no voiced lines")
    trs = gu.load_translations(ctx)
    overrides = load_overrides(ctx.paths.overrides)
    calib = read_json(ctx.paths.spk_calibration)
    thr, thr_src = spk_threshold(cfg, calib)
    reweight = bool(ctx.extra.get("reweight"))
    force = bool(ctx.force) and not reweight  # --force: every metric is computed again (a reweight never looks at a model)
    mkey = _metric_key(cfg)
    unavailable: dict[str, str] = {}
    if cfg.score.emotion != cfg.analyze.e2v:
        ctx.log.warning("score.emotion (%s) differs from analyze.e2v (%s): emotion similarity compares different embedding spaces", cfg.score.emotion, cfg.analyze.e2v)
    partial = _Partial.open(ctx)

    cands: list[Cand] = []
    per_line: dict[str, list[Cand]] = {}
    sidecars: dict[str, dict[str, SynthSidecar | None]] = {}
    failed_counts: dict[str, int] = {}
    restored = stale = 0
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
        cont = [trs[x] for x in ((line.model_extra or {}).get("sustain_lines") or []) if trs.get(x) is not None]
        set_texts = [o for o in overrides.get(line.line_id, []) if o.action == "set_text" and o.payload.get("text")]
        cur_text = gu.strip_stress(str(set_texts[-1].payload["text"])) if set_texts else None
        for wav, side in gu.candidate_files(ctx, line.line_id):
            if side is None or (side.model_extra or {}).get("error"):
                continue
            if stale_take(side, tr, cont, cur_text):
                stale += 1  # a take of an older text: never measured, never selected (synth deletes it on its next run)
                continue
            rel = ctx.paths.rel(wav)
            fp = file_fingerprint(wav)
            prev = old.get(rel)
            saved = None if (reweight or force) else partial.lookup(f"{line.line_id}|{wav.stem}", fp)
            if saved is not None:  # a killed run got this far (it started from the final record, if there was one)
                record = CandidateScore.model_validate(saved)
                restored += 1
            elif prev is not None and (reweight or (not force and (prev.model_extra or {}).get("fp") == fp and (prev.model_extra or {}).get("mkey") == mkey)):
                record = prev
            else:
                record = CandidateScore(line_id=line.line_id, wav=rel, backend=side.backend, route=side.route, metrics={})
                setattr(record, "na", ["cer"] if side.route == "orig" else [])
            if side.backend == "sample":
                setattr(record, "na", list(SAMPLE_NA))
                if SAMPLE_NOTE not in record.notes:
                    record.notes = [*record.notes, SAMPLE_NOTE]
            setattr(record, "fp", fp)
            setattr(record, "mkey", mkey)
            setattr(record, "round", side.round)
            c = Cand(ctx, line, wav, side, passport, tr, record)
            lst.append(c)
            sidecars[line.line_id][rel] = side
        per_line[line.line_id] = lst
        cands += lst
    if restored:
        ctx.log.info("score: %d candidate(s) resumed from %s", restored, PARTIAL_FILE)
    if stale:
        ctx.log.warning("score: %d take(s) of an older text skipped (not scored, not selectable)", stale)

    if not reweight and cands:
        _compute(ctx, cands, unavailable, partial)
    elif cands:  # no model: an older CER is still re-derived from its stored transcript
        _refresh_cer(cands, trs, cfg.project.tgt_lang, partial, drop=False)

    # ---- composite, gates, selection ------------------------------------------
    selections: list[SelectionRecord] = []
    for line in lines:
        lst = per_line.get(line.line_id, [])
        tr = trs.get(line.line_id)
        if not lst and silent_sound_line(cfg, tr, line) and not any(o.action in ("set_text", "accept", "keep_original") for o in overrides.get(line.line_id, [])):
            continue  # synth.sounds.only: silent -- left out on purpose: no selection, no review item (pack lists it)
        interj = bool(tr and "interjection" in tr.flags)
        for c in lst:
            apply_scores(c.rec, cfg, thr, interj)
        demote_unheard([c.rec for c in lst])
        write_jsonl(ctx.paths.scores(line.line_id), [c.rec for c in lst])
        tr_problem = translation_problem(tr)
        if tr_problem and not any(c.rec.route == "orig" for c in lst):
            selections.append(SelectionRecord(line_id=line.line_id, status="review", reason=tr_problem, n_candidates=len(lst)))
            continue
        line_cer_max = cfg.score.interjection_cer_max if interj else cfg.score.gates.cer_max
        ceiling = max(float(cfg.score.regen_cer_max), float(line_cer_max)) if cfg.score.regen_cer_max > 0 else None
        sel = select(
            line.line_id, [c.rec for c in lst], sidecars.get(line.line_id, {}), overrides.get(line.line_id, []),
            cfg.score.regen_rounds, unavailable, line.src_wav, failed_counts.get(line.line_id, 0), regen_cer_max=ceiling,
        )
        selections.append(sel)

    # keep selections of lines outside --only/--limit
    if ctx.only or ctx.limit is not None:
        prev = gu.load_selection(ctx)
        mine = {s.line_id for s in selections}
        selections = [*selections, *[s for lid, s in prev.items() if lid not in mine]]
    write_jsonl(ctx.paths.selection, selections)
    if not reweight:  # (a reweight did not look at the partial records: they may hold more than the final ones)
        partial.forget([c.cid for c in cands])  # the final records are on disk

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
    if restored:
        summary["resumed_candidates"] = restored
    if stale:
        summary["stale_candidates_skipped"] = stale
    if unavailable:
        summary["metrics_unavailable"] = unavailable
    unchecked = sum(1 for x in selections if x.status == "review" and ("gate not checked" in x.reason or "no gate could be evaluated" in x.reason))
    if unchecked:
        summary["lines_with_unchecked_gates"] = unchecked
    hopeless = sum(1 for x in selections if x.status == "review" and "a new take would not help" in x.reason)
    if hopeless:
        summary["review_without_regen"] = hopeless
    return summary


def _mark_unavailable(ctx: Any, unavailable: dict[str, str], group: str, reason: str) -> None:
    """A metric group could not run: remembered for the selection and the notes, told to the registry (-> the job's warnings)."""
    unavailable.setdefault(group, reason)
    gu.record_unavailable(ctx, NAME, group, reason)


def _acquire(ctx: Any, names: list[str], group: str, unavailable: dict[str, str]):
    b, used, reason = gu.acquire_first(ctx, names)
    if b is None:
        _mark_unavailable(ctx, unavailable, group, reason)
    return b, used


@contextmanager
def _guard(ctx: Any, unavailable: dict[str, str], group: str) -> Iterator[None]:
    """A backend that disappears while a group runs (``BackendUnavailable`` out of ``run_batched``) ends that group, not the stage."""
    try:
        yield
    except BackendUnavailable as e:
        _mark_unavailable(ctx, unavailable, group, e.reason or str(e))


def _emb_fresh(out: Path, wav: Path) -> bool:
    """Is the embedding file there and not older than its audio (the rule ``genutil.embed_paths`` uses)?"""
    try:
        return out.exists() and out.stat().st_mtime >= Path(wav).stat().st_mtime
    except OSError:
        return False


def line_cer_reference(c: Cand, trs: dict[str, Any]) -> str:
    """The CER reference of a candidate: what synth said, or the translation of the line and of its sustain continuations."""
    cont = [trs.get(x) for x in ((c.line.model_extra or {}).get("sustain_lines") or [])]
    return cer_reference(c.side, c.tr, [x for x in cont if x is not None])


def _set_cer(c: Cand, ref: str, hyp: str, lang: str) -> bool:
    """Store the CER of ``hyp`` (the ASR's transcript) against ``ref`` under the current rule; False when the transcript holds
    no words to compare (nothing stored)."""
    ref_w, hyp_w = cer_words(ref, hyp)  # the words only: an ASR cannot spell a laugh
    words = cer_texts(ref, hyp, lang)
    if ref_w.strip() and not words[1].strip():
        return False
    c.rec.metrics["cer"] = round(cer_over_words(ref, hyp, words, lang), 4)
    setattr(c.rec, "asr_text", hyp)
    setattr(c.rec, "cer_ref", ref_w)
    if hyp_w != " ".join(hyp.split()):
        setattr(c.rec, "cer_hyp", hyp_w)  # the transcript as compared, when a laugh was taken out of it
    elif c.rec.model_extra:
        c.rec.model_extra.pop("cer_hyp", None)
    setattr(c.rec, "cer_rule", CER_RULE)
    return True


def _refresh_cer(cands: list[Cand], trs: dict[str, Any], lang: str, partial: _Partial, drop: bool) -> int:
    """CER values stored under an older rule are recomputed from the stored transcript (no ASR, no synthesis); one without a
    transcript is dropped so the ASR runs again (``drop``; a reweight never asks a model and keeps it). Returns how many changed."""
    n = 0
    for c in cands:
        ex = c.rec.model_extra or {}
        if c.rec.route == "orig" or is_sample(c.rec) or "cer" in c.na or c.rec.metrics.get("cer") is None or ex.get("cer_rule") == CER_RULE:
            continue
        hyp = ex.get("asr_text")
        if hyp is None:
            if drop:
                c.rec.metrics.pop("cer")
                partial.touch(c)
                n += 1
            continue
        if not _set_cer(c, line_cer_reference(c, trs), str(hyp), lang):
            c.rec.metrics.pop("cer")
            if NO_WORDS_NOTE not in c.rec.notes:
                c.rec.notes = [*c.rec.notes, NO_WORDS_NOTE]
        partial.touch(c)
        n += 1
    return n


def _text_sig(text: str | None) -> str:
    return normalize_for_cer(_strip_nonspeech(gu.strip_stress(text or "")))


def stale_take(side: SynthSidecar | None, tr: Any, cont: list[Any], set_text: str | None) -> bool:
    """Is this take of an older text (the translation was edited, a review ``set_text`` came or went) than any reference the
    line can have now: the full or the short text of the current translation (with and without its sustain continuations), or
    the current ``set_text``? Synth purges such takes; this keeps one that survived from competing (it would be measured against
    its own old text, pass and win). Whatever cannot be told safely is kept: no text in the sidecar (``orig``, samples), a sound
    stand-in (its reference is rendered), no usable translation."""
    if side is None or side.route == "orig" or side.backend == "sample":
        return False
    ex = side.model_extra or {}
    plain = ex.get("text_plain")
    if not plain or side.text_variant in ("sound", "orig") or ex.get("sound") or sounds.has_sound(str(plain)):
        return False
    allowed: list[str] = []
    if set_text:
        allowed.append(set_text)
    if tr is not None and translation_problem(tr) == "":
        if sounds.has_sound(tr.text_plain or tr.text_tgt or ""):
            return False
        for recs in ([tr, *cont], [tr]) if cont else ([tr],):
            for short in (False, True):
                parts = []
                for r in recs:
                    rex = r.model_extra or {}
                    if short:
                        parts.append(rex.get("text_plain_short") or gu.strip_stress(r.text_tgt_short or r.text_tts_short or r.text_tts or ""))
                    else:
                        parts.append(r.text_plain or gu.strip_stress(r.text_tgt or ""))
                allowed.append(" ".join(p.strip() for p in parts if p and p.strip()))
    sigs = {_text_sig(a) for a in allowed if a and a.strip()}
    return bool(sigs) and _text_sig(str(plain)) not in sigs


def line_raw_cer_reference(c: Cand, trs: dict[str, Any]) -> str:
    """:func:`line_cer_reference` with the sounds still in it."""
    cont = [trs.get(x) for x in ((c.line.model_extra or {}).get("sustain_lines") or [])]
    return raw_cer_reference(c.side, c.tr, [x for x in cont if x is not None])


def _laughter_lines(cands: list[Cand], trs: dict[str, Any], partial: _Partial) -> None:
    """Decide the CER of every candidate whose text has laughter, before any model runs.

    * the text is nothing but laughter: no words for an ASR to check, so ``cer`` is not applicable (``na``, like the original
      recording), the ASR is never asked, and a CER stored by an older run is dropped;
    * the line had been marked that way and has words now: the mark goes;
    * the text has words and laughter and its CER was computed before laughter was taken out (no ``cer_rule``): recomputed.
    """
    for c in cands:
        if c.rec.route == "orig" or is_sample(c.rec):
            continue
        ref = line_cer_reference(c, trs)
        changed = False
        note = LAUGHTER_NOTE if laughter.is_laughter_only(ref) else (SOUND_NOTE if sounds.is_sound_only(line_raw_cer_reference(c, trs)) else None)
        if note is not None:
            if "cer" not in c.na:
                c.rec.na = [*c.na, "cer"]  # type: ignore[attr-defined]
                changed = True
            stale = [c.rec.metrics.pop("cer", None), (c.rec.model_extra or {}).pop("asr_text", None)]
            changed = changed or any(x is not None for x in stale)
            stale_notes = [n for n in c.rec.notes if n in (LAUGHTER_NOTE, SOUND_NOTE) and n != note]
            if stale_notes:
                c.rec.notes = [n for n in c.rec.notes if n not in stale_notes]
                changed = True
            if note not in c.rec.notes:
                c.rec.notes = [*c.rec.notes, note]
                changed = True
        elif is_laughter_line(c.rec):
            c.rec.na = [x for x in c.na if x != "cer"]  # type: ignore[attr-defined]
            c.rec.notes = [n for n in c.rec.notes if n not in (LAUGHTER_NOTE, SOUND_NOTE)]
            changed = True
        elif c.rec.metrics.get("cer") is not None and (c.rec.model_extra or {}).get("cer_rule") != CER_RULE and (laughter.has_laughter(ref) or sounds.has_sound(line_raw_cer_reference(c, trs))):
            c.rec.metrics.pop("cer")
            changed = True
        if changed:
            partial.touch(c)


def _compute(ctx: Any, cands: list[Cand], unavailable: dict[str, str], partial: _Partial | None = None) -> None:
    """Fill the missing metrics of ``cands`` group by group (one model at a time, all candidates under it).

    The metrics found in ``partial`` are already in the records, so only the missing groups run. The partial file is written
    after every group and at most every ``PARTIAL_EVERY_S`` seconds inside one, also when the stage dies half way."""
    partial = partial or _Partial.open(ctx)
    for c in cands:  # reasons from an earlier run are rewritten below when they still hold
        c.rec.notes = [n for n in c.rec.notes if not _STALE_NOTE.match(n)]
    trs = gu.load_translations(ctx)
    _refresh_cer(cands, trs, ctx.cfg.project.tgt_lang, partial, drop=True)  # before the laughter rule: it reads the new values
    _laughter_lines(cands, trs, partial)
    off: set[str] = set()
    if not ctx.cfg.score.aesthetics:
        off.add("pq")
    if not ctx.cfg.score.adv:
        off.add("adv_delta")
    total, done = _units(cands, off)
    prog = _Progress(ctx, total, done)
    try:
        _compute_groups(ctx, cands, unavailable, partial, prog)
    finally:
        partial.checkpoint(force=True)


def _compute_groups(ctx: Any, cands: list[Cand], unavailable: dict[str, str], partial: _Partial, prog: _Progress) -> None:
    cfg = ctx.cfg
    s = cfg.score
    tgt = cfg.project.tgt_lang

    def failed(c: Cand, group: str, err: Any) -> None:
        """This item got no result from a group that ran: a note (the gate verdict reads it), and a loud report when the GPU is full."""
        err = str(err or "no result")
        note = f"{group} failed: {err[:120]}"
        if note not in c.rec.notes:
            c.rec.notes = [*c.rec.notes, note]
        partial.touch(c)
        if gu.is_vram_error(err):
            _mark_unavailable(ctx, unavailable, group, f"out of memory: {err[:160]}")

    # MOS
    todo = [c for c in cands if c.needs("utmos")]
    if todo:
        b, _ = _acquire(ctx, [s.mos, s.mos_fallback], "mos", unavailable)
        if b is not None:
            by = {c.cid: c for c in todo}

            def got_mos(part: list[dict[str, Any]], res: list[dict[str, Any]]) -> None:
                for r in res:
                    c = by.get(r.get("id"))
                    if c is None:
                        continue
                    v = r.get("mos", r.get("utmos", r.get("score")))
                    if v is not None and not r.get("error"):
                        c.rec.metrics["utmos"] = round(float(v), 4)
                        partial.touch(c)
                    else:
                        failed(c, "mos", r.get("error"))
                prog.chunk(part)
                partial.checkpoint()

            with _guard(ctx, unavailable, "mos"):
                gu.run_batched(b, "mos", [{"id": c.cid, "wav": str(c.wav)} for c in todo], on_results=got_mos)
        partial.checkpoint(force=True)
    # aesthetics
    if s.aesthetics:
        todo = [c for c in cands if c.needs("pq")]
        if todo:
            b, _ = _acquire(ctx, [s.aesthetics], "aesthetics", unavailable)
            if b is not None:
                by = {c.cid: c for c in todo}

                def got_aes(part: list[dict[str, Any]], res: list[dict[str, Any]]) -> None:
                    for r in res:
                        c = by.get(r.get("id"))
                        if c is None:
                            continue
                        if not r.get("error"):
                            for k in ("pq", "ce", "cu", "pc"):
                                if r.get(k) is not None:
                                    c.rec.metrics[k] = round(float(r[k]), 4)
                            partial.touch(c)
                        else:
                            failed(c, "aesthetics", r.get("error"))
                    prog.chunk(part)
                    partial.checkpoint()

                with _guard(ctx, unavailable, "aesthetics"):
                    gu.run_batched(b, "aesthetics", [{"id": c.cid, "wav": str(c.wav)} for c in todo], on_results=got_aes)
            partial.checkpoint(force=True)
    else:
        for c in cands:
            if "pq" not in c.na:
                c.rec.na = [*c.na, "pq"]  # type: ignore[attr-defined]
    # speaker similarity: vs the character's timbre refs; route P (and characters without refs) vs the
    # performer's reference clips; last resort the original line
    todo = [c for c in cands if c.needs("spk_sim")]
    if todo:
        b, name = _acquire(ctx, [s.speaker], "speaker", unavailable)
        if b is not None:
            jobs = []
            refs_cache: dict[tuple[str, str], tuple[list[str], str]] = {}

            def ref_keys(sid: str, route: str) -> tuple[list[str], str]:
                """(embedding job keys of the reference clips, which kind they are: char | perf)."""
                want = "perf" if route == "P" else "char"
                if (sid, want) not in refs_cache:
                    keys: list[str] = []
                    kind = want
                    if want == "char":
                        refs = [r for r in gu.load_refs(ctx, sid) if r.usage in ("timbre", "both")] or gu.load_refs(ctx, sid)
                        for r in refs:
                            if not ctx.paths.abs(r.path).exists():
                                continue  # (embed_paths stats the wav even when the cached .npy is there: a gone clip must not end the stage)
                            k = f"ref|{sid}|{r.path}"
                            jobs.append((k, ctx.paths.abs(r.path), ctx.paths.vb(sid) / "emb" / f"ref_{Path(r.path).stem}.{name}.npy"))
                            keys.append(k)
                    if not keys:
                        kind = "perf"
                        for r in (gu.load_performer(ctx, sid) or {}).get("refs", []):
                            p = ctx.paths.abs(r["path"])
                            if p.exists():
                                k = f"perf|{sid}|{r['path']}"
                                jobs.append((k, p, ctx.paths.vb(sid) / "emb" / f"perf_{Path(r['path']).stem}.{name}.npy"))
                                keys.append(k)
                    refs_cache[(sid, want)] = (keys, kind)
                return refs_cache[(sid, want)]

            for c in todo:
                keys, _ = ref_keys(c.line.speaker_id, c.rec.route)
                if not keys and (c.line.src_wav16 or c.line.src_wav) and ctx.paths.abs(c.line.src_wav16 or c.line.src_wav).exists():
                    k = f"orig|{c.line.line_id}"
                    jobs.append((k, ctx.paths.abs(c.line.src_wav16 or c.line.src_wav), c.emb_dir / f"orig.{name}.npy"))
                jobs.append((f"cand|{c.cid}", c.wav, c.emb_dir / f"{c.wav.stem}.{name}.npy"))
            vecs: dict[str, Any] = {}
            with _guard(ctx, unavailable, "speaker"):
                vecs = gu.embed_paths(ctx, b, list(dict.fromkeys(jobs)))  # the npy files survive a kill: a rerun only re-reads them
            for c in todo:
                v = vecs.get(f"cand|{c.cid}")
                if v is None:
                    if "speaker" not in unavailable:  # (when the backend vanished, the group is reported as a whole)
                        failed(c, "speaker", "no embedding")
                    continue
                keys, kind = ref_keys(c.line.speaker_id, c.rec.route)
                refs = [vecs[k] for k in (keys or [f"orig|{c.line.line_id}"]) if k in vecs]
                if refs:
                    c.rec.metrics["spk_sim"] = round(max(cosine(v, r) for r in refs), 4)
                    if not keys:
                        c.rec.notes = [*c.rec.notes, "spk_sim vs the original line (no voicebank refs)"]
                    elif kind == "perf":
                        c.rec.notes = [*c.rec.notes, "spk_sim vs the performer's sample (consistency; no original voice)"]
                    partial.touch(c)
                else:
                    # nothing to compare the take with (no reference clip, or none could be embedded): the gate is unchecked and the
                    # line goes to a person, instead of the take passing without a speaker check
                    failed(c, "speaker", "no reference voice to compare with")
            prog.add(len(todo), todo[-1].line.line_id)
        if s.speaker_extra:
            b2, name2 = _acquire(ctx, [s.speaker_extra], "speaker_extra", unavailable)
            if b2 is not None:
                jobs = []
                for c in todo:
                    if c.line.src_wav16 or c.line.src_wav:
                        jobs.append((f"orig|{c.line.line_id}", ctx.paths.abs(c.line.src_wav16 or c.line.src_wav), c.emb_dir / f"orig.{name2}.npy"))
                    jobs.append((f"cand|{c.cid}", c.wav, c.emb_dir / f"{c.wav.stem}.{name2}.npy"))
                vecs2: dict[str, Any] = {}
                with _guard(ctx, unavailable, "speaker_extra"):
                    vecs2 = gu.embed_paths(ctx, b2, list(dict.fromkeys(jobs)))
                for c in todo:
                    a, o = vecs2.get(f"cand|{c.cid}"), vecs2.get(f"orig|{c.line.line_id}")
                    if a is not None and o is not None:
                        c.rec.metrics["spk_sim_extra"] = round(cosine(a, o), 4)
                        partial.touch(c)
        partial.checkpoint(force=True)
    # emotion similarity (same emotion2vec+ model as analyze)
    todo = [c for c in cands if c.needs("emo_sim")]
    if todo:
        b, name = _acquire(ctx, [s.emotion], "emotion", unavailable)
        if b is not None:
            items = []
            orig_out: dict[str, Path] = {}
            have = 0  # candidates whose embedding file survived an interrupted run: no model call for them
            for c in todo:
                lid = c.line.line_id
                emb = ((c.passport.get("emotion") or {}).get("e2v") or {}).get("embedding_path")
                same_model = (c.passport.get("backends") or {}).get("e2v") in (None, name)
                if emb and same_model and ctx.paths.abs(emb).exists():
                    orig_out[lid] = ctx.paths.abs(emb)
                elif lid not in orig_out and (c.line.src_wav16 or c.line.src_wav):
                    out = c.emb_dir / f"orig.{name}.e2v.npy"
                    orig_out[lid] = out
                    osrc = ctx.paths.abs(c.line.src_wav16 or c.line.src_wav)
                    if not _emb_fresh(out, osrc):
                        items.append({"id": f"orig|{lid}", "wav": str(osrc), "out": str(out)})
                cout = c.emb_dir / f"{c.wav.stem}.{name}.e2v.npy"
                if _emb_fresh(cout, c.wav):
                    have += 1
                    continue
                items.append({"id": c.cid, "wav": str(c.wav), "out": str(cout)})
            for it in items:
                Path(it["out"]).parent.mkdir(parents=True, exist_ok=True)
            prog.add(have, todo[-1].line.line_id)
            res: dict[str, dict[str, Any]] = {}

            def got_emo(part: list[dict[str, Any]], got: list[dict[str, Any]]) -> None:
                res.update({r["id"]: r for r in got if r.get("id") is not None})
                prog.chunk(part)

            with _guard(ctx, unavailable, "emotion"):
                gu.run_batched(b, "emotion", items, on_results=got_emo)
            for c in todo:
                r = res.get(c.cid, {})
                p = r.get("embedding_path") or (c.emb_dir / f"{c.wav.stem}.{name}.e2v.npy")
                o = orig_out.get(c.line.line_id)
                if r.get("error") or not Path(p).exists() or o is None or not Path(o).exists():
                    if r.get("error"):
                        failed(c, "emotion", r["error"])
                    continue
                c.rec.metrics["emo_sim"] = round(cosine(np.load(p), np.load(o)), 4)
                partial.touch(c)
                if r.get("label"):
                    setattr(c.rec, "emo_label", r["label"])
                    orig_label = (c.passport.get("style") or {}).get("emotion")
                    if orig_label:
                        c.rec.metrics["emo_match"] = 1.0 if r["label"] == orig_label else 0.0
        partial.checkpoint(force=True)
    # A/D/V delta
    todo = [c for c in cands if c.needs("adv_delta")]
    if todo and s.adv:
        b, _ = _acquire(ctx, [s.adv], "adv", unavailable)
        if b is not None:
            orig_items = {}
            for c in todo:
                if not (c.passport.get("emotion") or {}).get("adv") and (c.line.src_wav16 or c.line.src_wav):
                    orig_items[c.line.line_id] = {"id": f"orig|{c.line.line_id}", "wav": str(ctx.paths.abs(c.line.src_wav16 or c.line.src_wav))}
            by = {c.cid: c for c in todo}
            orig_res: dict[str, dict[str, Any]] = {}

            def got_adv(part: list[dict[str, Any]], res_: list[dict[str, Any]]) -> None:
                for r in res_:  # the original-line items come first, so their result is known when the candidates arrive
                    rid = str(r.get("id"))
                    if rid.startswith("orig|"):
                        orig_res[rid] = r
                        continue
                    c = by.get(rid)
                    if c is None:
                        continue
                    o = (c.passport.get("emotion") or {}).get("adv") or orig_res.get(f"orig|{c.line.line_id}", {})
                    if r.get("error") or r.get("arousal") is None or o.get("arousal") is None:
                        failed(c, "adv", r.get("error") or "no arousal value")
                        continue
                    c.rec.metrics["adv_delta"] = round(abs(float(r["arousal"]) - float(o["arousal"])) + abs(float(r.get("valence", 0.5)) - float(o.get("valence", 0.5))), 4)
                    partial.touch(c)
                prog.chunk(part)
                partial.checkpoint()

            with _guard(ctx, unavailable, "adv"):
                gu.run_batched(b, "adv", list(orig_items.values()) + [{"id": c.cid, "wav": str(c.wav)} for c in todo], on_results=got_adv)
        partial.checkpoint(force=True)
    # CER
    todo = [c for c in cands if c.needs("cer")]
    if todo:
        b, _ = _acquire(ctx, [s.cer_asr, s.cer_asr_fallback], "cer", unavailable)
        if b is not None:
            trs = gu.load_translations(ctx)
            by = {c.cid: c for c in todo}

            def got_cer(part: list[dict[str, Any]], res_: list[dict[str, Any]]) -> None:
                for r in res_:
                    c = by.get(r.get("id"))
                    if c is None:
                        continue
                    if r.get("error"):
                        failed(c, "cer", r["error"])
                        continue
                    ref = line_cer_reference(c, trs)
                    hyp = str(r.get("text", "") or "")
                    if not _set_cer(c, ref, hyp, tgt):
                        # the model heard no words (a clip too short or too odd for it, or only a laugh): not a CER of 1.0 for
                        # regen rounds to chase, but a gate that could not be checked (and a take that loses to a measured one)
                        failed(c, "cer", NO_WORDS)
                        continue
                    partial.touch(c)
                prog.chunk(part)
                partial.checkpoint()

            with _guard(ctx, unavailable, "cer"):
                gu.run_batched(b, "transcribe", [{"id": c.cid, "wav": str(c.wav), "lang": tgt} for c in todo], on_results=got_cer, lang=tgt)
        partial.checkpoint(force=True)
    # duration + prosody (core DSP, F0 via the configured f0 backend)
    for c in cands:
        if c.rec.metrics.get("dur_ratio") is None and "dur_ratio" not in c.na:
            src_d = gu.src_duration(ctx, c.line, c.passport) / native_speed(c.side)  # a speed the backend applied is expected
            if src_d > 0:
                c.rec.metrics["dur_ratio"] = round(features.duration(c.wav) / src_d, 4)
                c.rec.metrics["duration_s"] = round(features.duration(c.wav), 3)
    todo = [c for c in cands if c.needs("prosody")]
    if todo:
        # no F0 backend: the core pYIN does it (slower, rougher); said so, the prosody term is still computed
        f0b, _, _ = gu.acquire_first(ctx, [cfg.analyze.f0], group="f0 backend (core pYIN used instead)", stage=NAME)
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
            partial.touch(c)
            prog.add(1, c.line.line_id)
            partial.checkpoint()
        partial.checkpoint(force=True)
    prog.finish()
    for group, reason in unavailable.items():
        metric = GROUP_METRIC.get(group)
        for c in cands:
            if metric is not None and (c.rec.metrics.get(metric) is not None or metric in c.na):
                continue  # this candidate has what the group would have given
            note = f"{group} unavailable: {reason[:120]}"
            if note not in c.rec.notes:
                c.rec.notes = [*c.rec.notes, note]
                partial.touch(c)

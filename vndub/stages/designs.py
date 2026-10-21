"""Persisted voice designs: the store behind the casting screen (helpers shared by ``voicebank``).

Every design a character is offered is kept in ``voicebank/<sid>/designs.json`` with its wavs under
``voicebank/<sid>/designs/<design_id>/``::

    {"version": 1, "speaker_id": "sakura", "designs": [Design, ...]}

``Design`` (all paths work-relative, see ``vndub_app/API.md`` 2.3)::

    design_id    "vd01" (generated) | "ref01" (user sample) | "clone" | "imp01" (library)
    kind         "voice_design" (text prompt) | "reference" (user sample / reference_dir clips) | "clone" (route A)
    backend      backend that rendered the samples
    prompt       the text sent to the design backend ("" for reference / clone)
    description  free text that went into the prompt (Character.description or the one passed in)
    samples      [{"line_id", "wav", "text"}]  audition clips: two translated lines of the character
    ref_clips    [{"wav", "text"}]             extra neutral clips (configured test sentences); with the
                                               samples they become the performer's timbre refs
    spk_sim      cosine to the character centroid (None without original audio)
    f0_delta_st  semitones from the character's median F0 (None without original audio)
    stars        1..5 or None  (see :func:`star_rating`; computed on read, never trusted from disk)
    rank_score   ordering key, higher is better (see :func:`rank_score`)
    license, commercial_ok, created, origin ("stage" | "user" | "library"), seed, design_index, ...

Stars: ``stars = clamp(floor(3 + 2 * (spk_sim - thr) / w + 0.5), 1, 5)`` where ``thr`` is the speaker-similarity
gate threshold (the calibrated EER threshold when ``voicebank`` calibrated one, else the configured / default
one) and ``w`` the half width ``clamp((same_mean - diff_mean) / 2, 0.10, 0.35)`` taken from the calibration
(0.20 without it). That is ``1 + 4 * (spk_sim - thr_lo) / (thr_hi - thr_lo)`` with ``thr_lo = thr - w`` and
``thr_hi = thr + w``: the threshold is the 3-star point, ``thr - w`` and below is 1 star, ``thr + w`` and above 5.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any

from vndub.manifest.io import read_json, write_json
from vndub.stages import genutil as gu

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

KINDS = ("voice_design", "reference", "clone")
SAMPLE_MIN_S = 5.0
SAMPLE_MAX_S = 20.0
SAMPLE_EXTS = (".wav", ".mp3")
STAR_HALF_WIDTH = 0.20
STAR_HALF_WIDTH_RANGE = (0.10, 0.35)
VARIANT_HINTS = [
    "",
    "slightly higher and brighter",
    "slightly lower and warmer",
    "softer and a little breathy",
    "crisper and more energetic",
    "calmer and more mature",
]
VARIANT_FACTORS = (1.0, 1.08, 0.92, 1.04, 0.96)  # F0 factors of local (mock / SAPI) design backends


# ----------------------------------------------------------------------------
# store
# ----------------------------------------------------------------------------


def designs_path(ctx: Any, sid: str) -> Path:
    return ctx.paths.vb(sid) / "designs.json"


def design_dir(ctx: Any, sid: str, design_id: str) -> Path:
    return ctx.paths.vb(sid) / "designs" / design_id


def load_designs(ctx: Any, sid: str) -> list[dict[str, Any]]:
    data = read_json(designs_path(ctx, sid), {}) or {}
    rows = data.get("designs") if isinstance(data, dict) else data
    return [dict(d) for d in (rows or []) if isinstance(d, dict) and d.get("design_id")]


def save_designs(ctx: Any, sid: str, designs: list[dict[str, Any]]) -> None:
    write_json(designs_path(ctx, sid), {"version": 1, "speaker_id": sid, "designs": designs})


def get_design(ctx: Any, sid: str, design_id: str) -> dict[str, Any] | None:
    return next((d for d in load_designs(ctx, sid) if d["design_id"] == design_id), None)


def next_design_id(ctx: Any, sid: str, designs: list[dict[str, Any]], prefix: str) -> str:
    """``vd01`` / ``ref01`` / ``imp01`` ...: the next free number for ``prefix`` (``clone`` has a fixed id)."""
    if prefix == "clone":
        return "clone"
    used = {d["design_id"] for d in designs}
    root = ctx.paths.vb(sid) / "designs"
    if root.exists():
        used |= {p.name for p in root.iterdir() if p.is_dir()}
    n = 1
    while f"{prefix}{n:02d}" in used:
        n += 1
    return f"{prefix}{n:02d}"


def remove_design_files(ctx: Any, sid: str, design_id: str) -> None:
    shutil.rmtree(design_dir(ctx, sid, design_id), ignore_errors=True)


def clip_paths(ctx: Any, d: dict[str, Any]) -> list[Path]:
    """The wavs similarity and F0 are measured on (a user's sample counts as itself, not its clones)."""
    if d.get("ref_wav"):
        return [ctx.paths.abs(d["ref_wav"])]
    wavs = [s["wav"] for s in d.get("samples", [])] + [c["wav"] for c in d.get("ref_clips", [])]
    return [ctx.paths.abs(w) for w in wavs]


def performer_clips(ctx: Any, d: dict[str, Any]) -> list[dict[str, Any]]:
    """Clips that become the performer's timbre references: ``[{"src": Path, "text": str}]`` in priority order."""
    out: list[dict[str, Any]] = []
    if d.get("ref_wav"):
        out.append({"src": ctx.paths.abs(d["ref_wav"]), "text": ""})
    out += [{"src": ctx.paths.abs(s["wav"]), "text": s.get("text", "")} for s in d.get("samples", [])]
    out += [{"src": ctx.paths.abs(c["wav"]), "text": c.get("text", "")} for c in d.get("ref_clips", [])]
    return [c for c in out if c["src"].exists()]


# ----------------------------------------------------------------------------
# ranking and stars
# ----------------------------------------------------------------------------


def rank_score(spk_sim: float | None, f0_delta_st: float | None, max_f0_delta: float) -> float:
    """Higher is better. Designs inside the F0 window beat those outside it; within a group the
    speaker similarity decides (a design without a measurement ranks below any measured one) and the
    F0 distance only breaks ties -- the same order the single-best selection used before designs were kept.
    """
    outside = f0_delta_st is not None and abs(f0_delta_st) > max_f0_delta
    return (-10.0 if outside else 0.0) + (float(spk_sim) if spk_sim is not None else -1.0) - 1e-7 * abs(f0_delta_st or 0.0)


def star_rating(spk_sim: float | None, threshold: float, half_width: float = STAR_HALF_WIDTH) -> int | None:
    if spk_sim is None:
        return None
    x = 3.0 + 2.0 * (float(spk_sim) - float(threshold)) / float(half_width)
    return int(min(5, max(1, math.floor(x + 0.5 + 1e-9))))  # half-up (the epsilon absorbs float error at x.5)


def star_params(ctx: Any) -> dict[str, Any]:
    """``{"threshold", "half_width", "source"}`` for :func:`star_rating` (gate threshold + calibration width)."""
    from vndub.stages.score import spk_threshold

    calib = read_json(ctx.paths.spk_calibration)
    thr, src = spk_threshold(ctx.cfg, calib)
    w = STAR_HALF_WIDTH
    try:
        same, diff = float(calib["same"]["mean"]), float(calib["diff"]["mean"])
        w = min(STAR_HALF_WIDTH_RANGE[1], max(STAR_HALF_WIDTH_RANGE[0], (same - diff) / 2.0))
    except (TypeError, KeyError, ValueError):
        pass
    return {"threshold": thr, "half_width": round(w, 4), "source": src}


def public(ctx: Any, d: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
    """A design as the API returns it: stars filled in from the current threshold."""
    p = params or star_params(ctx)
    out = dict(d)
    out["stars"] = star_rating(d.get("spk_sim"), p["threshold"], p["half_width"])
    out.setdefault("rank_score", rank_score(d.get("spk_sim"), d.get("f0_delta_st"), ctx.cfg.voicebank.performer.max_f0_delta_semitones))
    return out


def ranked(ctx: Any, designs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Best first (stable for ties: creation order)."""
    p = star_params(ctx)
    return sorted((public(ctx, d, p) for d in designs), key=lambda d: -float(d["rank_score"]))


# ----------------------------------------------------------------------------
# sample lines
# ----------------------------------------------------------------------------


def _speakable(tr: Any) -> bool:
    return tr is not None and not ({"translate_failed", "translate_missing"} & set(tr.flags)) and bool((tr.text_tts or "").strip())


def sample_items(ctx: Any, sid: str, n: int = 2, line_ids: list[str] | None = None) -> list[dict[str, str]]:
    """``[{"line_id", "text"}]``: up to ``n`` translated lines of the character, then test sentences.

    Preference: not an interjection, 3+ words, under 220 characters, closest to 14 syllables (long enough
    to judge a voice, short enough to be a good timbre reference); script order breaks ties. Missing lines
    are filled with the configured test sentences (``line_id`` ``test_<i>``).
    """
    lang = ctx.cfg.project.tgt_lang
    trs = gu.load_translations(ctx)
    picked: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(line_id: str, text: str) -> None:
        text = gu.strip_stress(text).strip()
        if text and text not in seen and len(picked) < n:
            seen.add(text)
            picked.append({"line_id": line_id, "text": text})

    if line_ids:
        for lid in line_ids:
            tr = trs.get(lid)
            if _speakable(tr):
                add(lid, tr.text_tts or tr.text_tgt)
    if len(picked) < n:
        try:
            lines = [l for l in ctx.lines() if l.speaker_id == sid]
        except Exception:
            lines = []
        cands = []
        for l in lines:
            tr = trs.get(l.line_id)
            if not _speakable(tr) or "interjection" in tr.flags:
                continue
            text = gu.strip_stress(tr.text_tts or tr.text_tgt).strip()
            if len(text.split()) < 3 or len(text) > 220:
                continue
            syl = int(tr.syllables or 0) or gu.count_syllables(text)
            cands.append((abs(syl - 14), l.order, l.line_id, tr.text_tts or tr.text_tgt))
        for _, _, lid, text in sorted(cands):
            add(lid, text)
    tests = TEST_SENTENCES.get(lang, TEST_SENTENCES["en"])
    i = 0
    while len(picked) < n and i < len(tests):
        add(f"test_{i}", tests[i])
        i += 1
    return picked


def test_sentence_items(lang: str, n: int, skip: set[str] | None = None) -> list[dict[str, str]]:
    """The first ``n`` configured test sentences not in ``skip`` (extra neutral clips for performer refs)."""
    tests = TEST_SENTENCES.get(lang, TEST_SENTENCES["en"])
    return [{"line_id": f"test_{i}", "text": t} for i, t in enumerate(tests) if t not in (skip or set())][: max(0, n)]


# ----------------------------------------------------------------------------
# user samples
# ----------------------------------------------------------------------------


def decode_sample(ctx: Any, src: str | Path, dst: Path) -> float:
    """Validate a user sample (wav / mp3, 5..20 s) and write it to ``dst`` as mono 16-bit wav.

    Raises ``ValueError`` whose message is exactly ``sample_bad_format`` / ``sample_too_short`` /
    ``sample_too_long`` (the app maps these codes). Returns the duration in seconds.
    """
    from vndub.audio import features, ffmpeg
    from vndub.tools import ToolMissing

    src = Path(src)
    if not src.is_file() or src.suffix.lower() not in SAMPLE_EXTS:
        raise ValueError("sample_bad_format")
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        ffmpeg.decode(ctx.tools, src, dst, mono=True)
    except ToolMissing:
        if src.suffix.lower() != ".wav":
            raise
        try:  # no ffmpeg: a plain wav is still readable
            y, sr = features.load(src, mono=True)
            features.save(dst, y, sr)
        except Exception as e:  # noqa: BLE001
            raise ValueError("sample_bad_format") from e
    except Exception as e:  # noqa: BLE001  (AudioToolError: not decodable)
        raise ValueError("sample_bad_format") from e
    try:
        dur = float(features.duration(dst))
    except Exception as e:  # noqa: BLE001
        dst.unlink(missing_ok=True)
        raise ValueError("sample_bad_format") from e
    if dur < SAMPLE_MIN_S:
        dst.unlink(missing_ok=True)
        raise ValueError("sample_too_short")
    if dur > SAMPLE_MAX_S:
        dst.unlink(missing_ok=True)
        raise ValueError("sample_too_long")
    return dur

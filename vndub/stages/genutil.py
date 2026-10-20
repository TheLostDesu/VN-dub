"""Helpers shared by the generation-side stages (voicebank, synth, score, master, report).

Loading of translations / passports / voicebank files, backend acquisition with
fallbacks and recorded reasons, embedding/F0 helpers with on-disk caching, and
stable seeds. Everything here is read-only with respect to manifests owned by
other stages.
"""

from __future__ import annotations

import re
import zlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from vndub.backends.base import Backend, BackendUnavailable
from vndub.manifest.io import (
    AudioIndexRecord,
    CandidateScore,
    LineRecord,
    RefClip,
    SelectionRecord,
    SynthSidecar,
    TranslationRecord,
    read_json,
    read_jsonl,
    read_yaml,
)

# ----------------------------------------------------------------------------
# text helpers (fork C owns vndub.text.ru / ja; keep working without them)
# ----------------------------------------------------------------------------

_STRESS = re.compile("[+́]")
_RU_VOWELS = re.compile(r"[аеёиоуыэюяaeiouy]", re.I)


def strip_stress(text: str) -> str:
    try:
        from vndub.text.ru import strip_stress as _s

        return _s(text)
    except Exception:
        return _STRESS.sub("", text or "")


def count_syllables(text: str) -> int:
    try:
        from vndub.text.ru import count_syllables as _c

        return int(_c(text))
    except Exception:
        return len(_RU_VOWELS.findall(text or ""))


def ts_now() -> str:
    """Local ISO timestamp with microseconds (orders overrides vs. candidates within a second)."""
    from datetime import datetime

    return datetime.now().astimezone().isoformat(timespec="microseconds")


def stable_seed(*parts: Any) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode("utf-8")) & 0x7FFFFFFF


# ----------------------------------------------------------------------------
# manifest loading
# ----------------------------------------------------------------------------


def load_translations(ctx: Any) -> dict[str, TranslationRecord]:
    p = ctx.paths.translation(ctx.cfg.project.tgt_lang)
    return {r.line_id: r for r in read_jsonl(p, TranslationRecord)} if p.exists() else {}


def load_passport(ctx: Any, line_id: str) -> dict[str, Any] | None:
    return read_json(ctx.paths.passport(line_id))


def load_audio_index(ctx: Any) -> dict[str, AudioIndexRecord]:
    p = ctx.paths.audio_index
    return {a.id: a for a in read_jsonl(p, AudioIndexRecord)} if p.exists() else {}


def load_refs(ctx: Any, speaker_id: str) -> list[RefClip]:
    data = read_json(ctx.paths.vb_refs_json(speaker_id)) or []
    return [RefClip.model_validate(r) for r in data]


def ref_dicts(ctx: Any, refs: Iterable[RefClip]) -> list[dict[str, Any]]:
    """RefClips as dicts with absolute paths (for style_prompt.to_emotion_ref)."""
    out = []
    for r in refs:
        d = r.model_dump()
        d["path"] = str(ctx.paths.abs(r.path))
        out.append(d)
    return out


def load_performer(ctx: Any, speaker_id: str) -> dict[str, Any] | None:
    return read_yaml(ctx.paths.vb_performer(speaker_id))


def load_rvc_meta(ctx: Any, speaker_id: str) -> dict[str, Any] | None:
    meta = read_json(ctx.paths.vb_rvc(speaker_id) / "meta.json")
    if meta and meta.get("model") and ctx.paths.abs(meta["model"]).exists():
        return meta
    return None


def load_lora_meta(ctx: Any, speaker_id: str) -> dict[str, Any] | None:
    meta = read_json(ctx.paths.vb_lora(speaker_id) / "meta.json")
    if meta and meta.get("adapter") and ctx.paths.abs(meta["adapter"]).exists():
        return meta
    return None


def load_selection(ctx: Any) -> dict[str, SelectionRecord]:
    p = ctx.paths.selection
    return {s.line_id: s for s in read_jsonl(p, SelectionRecord)} if p.exists() else {}


def load_scores(ctx: Any, line_id: str) -> list[CandidateScore]:
    p = ctx.paths.scores(line_id)
    return read_jsonl(p, CandidateScore) if p.exists() else []


def candidate_files(ctx: Any, line_id: str) -> list[tuple[Path, SynthSidecar | None]]:
    """Synth candidates of a line: (wav, sidecar) excluding pre-VC intermediates."""
    d = ctx.paths.synth_dir(line_id)
    if not d.exists():
        return []
    out = []
    for wav in sorted(d.glob("*.wav")):
        if wav.name.endswith(".prevc.wav"):
            continue
        side = wav.with_suffix(".json")
        sc = None
        if side.exists():
            try:
                sc = SynthSidecar.model_validate(read_json(side))
            except Exception:
                sc = None
        out.append((wav, sc))
    return out


def src_duration(ctx: Any, line: LineRecord, passport: dict[str, Any] | None = None, audio: AudioIndexRecord | None = None) -> float:
    if passport and passport.get("duration_s"):
        return float(passport["duration_s"])
    if audio and audio.duration:
        return float(audio.duration)
    wav = line.src_wav or line.src_wav16
    if wav and ctx.paths.abs(wav).exists():
        from vndub.audio.features import duration

        return duration(ctx.paths.abs(wav))
    return 0.0


def char_stats(ctx: Any, characters: dict[str, Any], speaker_id: str) -> dict[str, Any]:
    c = characters.get(speaker_id)
    return c.stats.model_dump() if c is not None else {}


# ----------------------------------------------------------------------------
# backends
# ----------------------------------------------------------------------------


def acquire_first(ctx: Any, names: Iterable[str], options: dict[str, Any] | None = None) -> tuple[Backend | None, str | None, str]:
    """Acquire the first usable backend among ``names`` (empty names skipped).

    Returns (backend, name used, reason-if-none). Never raises for unavailability.
    """
    reasons = []
    for n in names:
        if not n:
            continue
        try:
            b = ctx.models.acquire(n, options)
            return b, b.spec.name, ""
        except BackendUnavailable as e:
            reasons.append(str(e))
            ctx.log.warning("backend %s unavailable: %s", n, e)
        except KeyError as e:
            reasons.append(f"{n}: {e}")
    return None, None, "; ".join(reasons) or "no backend configured"


def run_batched(backend: Backend, op: str, items: list[dict[str, Any]], chunk: int = 16, **kw: Any) -> list[dict[str, Any]]:
    """Call ``op`` in chunks; a failing chunk yields per-item errors instead of raising."""
    out: list[dict[str, Any]] = []
    for i in range(0, len(items), chunk):
        part = items[i : i + chunk]
        try:
            res = backend.call(op, part, **kw) or []
        except BackendUnavailable:
            raise
        except Exception as e:  # adapter crash / timeout: record and continue
            res = [{"id": it.get("id"), "error": f"{type(e).__name__}: {e}"} for it in part]
        by_id = {r.get("id"): r for r in res if isinstance(r, dict)}
        for it in part:
            out.append(by_id.get(it.get("id"), {"id": it.get("id"), "error": "no result"}))
    return out


def embed_paths(ctx: Any, backend: Backend, jobs: list[tuple[str, Path, Path]]) -> dict[str, np.ndarray]:
    """Embed (id, wav, out_npy) jobs, reusing existing npy files. Returns id -> vector."""
    vecs: dict[str, np.ndarray] = {}
    todo = []
    for jid, wav, out in jobs:
        if out.exists() and out.stat().st_mtime >= Path(wav).stat().st_mtime:
            vecs[jid] = np.load(out).astype(np.float32).reshape(-1)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            todo.append({"id": jid, "wav": str(wav), "out": str(out)})
    if todo:
        for r in run_batched(backend, "embed", todo):
            if r.get("error"):
                ctx.log.warning("embed %s failed: %s", r.get("id"), r["error"])
                continue
            path = r.get("path") or next((t["out"] for t in todo if t["id"] == r.get("id")), None)
            if path and Path(path).exists():
                vecs[r["id"]] = np.load(path).astype(np.float32).reshape(-1)
    return vecs


def f0_of(ctx: Any, backend: Backend | None, jobs: list[tuple[str, Path, Path]]) -> dict[str, np.ndarray]:
    """F0 contours (Hz, 0 = unvoiced) via the f0 backend or core pYIN; cached as npy."""
    from vndub.audio import features

    res: dict[str, np.ndarray] = {}
    todo = []
    for jid, wav, out in jobs:
        if out.exists() and out.stat().st_mtime >= Path(wav).stat().st_mtime:
            res[jid] = np.load(out).reshape(-1)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            todo.append({"id": jid, "wav": str(wav), "out": str(out)})
    if todo and backend is not None:
        for r in run_batched(backend, "f0", todo):
            p = r.get("path") or next((t["out"] for t in todo if t["id"] == r.get("id")), None)
            if not r.get("error") and p and Path(p).exists():
                res[r["id"]] = np.load(p).reshape(-1)
    for t in todo:
        if t["id"] in res:
            continue
        y, sr = features.load(t["wav"], sr=16000)
        f0 = features.f0_pyin(y, sr)
        np.save(t["out"], f0)
        res[t["id"]] = f0
    return res


def median_f0(f0: np.ndarray | None) -> float | None:
    if f0 is None:
        return None
    v = f0[f0 > 0]
    return float(np.median(v)) if v.size >= 3 else None


def semitone_delta(f_a: float | None, f_b: float | None) -> float | None:
    if not f_a or not f_b:
        return None
    return float(12.0 * np.log2(f_a / f_b))


def backend_options(cfg: Any, name: str) -> dict[str, Any]:
    opts = cfg.synth.backends.get(name)
    if opts is None:
        return {}
    d = opts.model_dump()
    d.pop("enabled", None)
    return d

"""master: winners -> game-ready voice files.

Per selected line: trim leading/trailing silence to the original's profile,
time-stretch toward the source duration (only outside ``target_window`` and only
within ``max_stretch``), loudness-normalize to the character's median LUFS with a
true-peak limiter, resample to the source sample rate / channel count, and
encode to the source codec under exactly the original ``voice_file`` name in
``work/final/encoded/``. ``orig`` winners copy the original file bytes.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

from vndub.audio import features, ffmpeg, loudness, stretch, vad
from vndub.manifest.io import AudioIndexRecord, FinalIndexRecord, LineRecord, SelectionRecord, read_json, write_json, write_jsonl
from vndub.stages import genutil as gu
from vndub.stages.common import StageError

NAME = "master"
VERSION = "1"
CONFIG_SECTIONS = ("master",)

# ffprobe codec name -> encoder key in vndub.audio.ffmpeg
_CODEC_MAP = {
    "vorbis": "vorbis", "opus": "opus", "mp3": "mp3", "flac": "flac", "aac": "aac",
    "pcm_s16le": "pcm_s16le", "pcm_s24le": "pcm_s16le", "pcm_f32le": "pcm_s16le", "pcm_u8": "pcm_s16le",
}


def trim_to_profile(y: np.ndarray, sr: int, lead_s: float, trail_s: float) -> np.ndarray:
    """Cut/pad silence so the candidate starts/ends like the original."""
    segs = vad.energy_vad(y, sr)
    if not segs:
        return y
    a = max(0, int((segs[0][0]) * sr))
    b = min(len(y), int(np.ceil(segs[-1][1] * sr)))
    body = y[a:b]
    # 10 ms fades at the cut points avoid clicks
    f = min(len(body) // 4, int(0.01 * sr))
    if f > 0:
        body = body.copy()
        body[:f] *= np.linspace(0, 1, f)
        body[-f:] *= np.linspace(1, 0, f)
    return np.concatenate([np.zeros(int(max(0.0, lead_s) * sr), dtype=y.dtype), body, np.zeros(int(max(0.0, trail_s) * sr), dtype=y.dtype)])


def target_lufs(ctx: Any, sid: str, chars: dict[str, Any]) -> tuple[float, str]:
    t = ctx.cfg.master.loudness.target
    if not isinstance(t, str):
        return float(t), "configured"
    try:
        return float(t), "configured"
    except ValueError:
        pass
    c = chars.get(sid)
    if t == "character_median" and c is not None and c.stats.lufs_median is not None:
        return float(c.stats.lufs_median), "character median"
    return float(ctx.cfg.master.loudness.default_lufs), "default"


def _bitrate(audio: AudioIndexRecord | None, project: Any) -> int | None:
    if project is not None and project.voice_codec.bitrate_k and (audio is None or not audio.codec or audio.codec == project.voice_codec.codec):
        return int(project.voice_codec.bitrate_k)
    if audio is not None and audio.bytes and audio.duration:
        return max(32, int(round(audio.bytes * 8 / audio.duration / 1000 / 16)) * 16)
    return None


def run(ctx: Any) -> dict[str, Any]:
    cfg = ctx.cfg.master
    sel = gu.load_selection(ctx)
    if not sel:
        raise StageError(f"{ctx.paths.selection} not found: run `vndub score` first")
    lines = {l.line_id: l for l in ctx.voiced_lines()}
    audio = gu.load_audio_index(ctx)
    chars = ctx.characters()
    project = ctx.project
    cache = ctx.cache(NAME, VERSION, "master")
    records: list[FinalIndexRecord] = []
    summary: dict[str, Any] = {"lines": 0, "encoded": 0, "cached": 0, "stretched": 0, "skipped": {}, "errors": {}}
    for lid, line in lines.items():
        s: SelectionRecord | None = sel.get(lid)
        if s is None or not s.winner:
            summary["skipped"]["no winner"] = summary["skipped"].get("no winner", 0) + 1
            continue
        status = s.status
        if status == "review":
            if cfg.include_review_lines != "provisional":
                summary["skipped"]["review"] = summary["skipped"].get("review", 0) + 1
                continue
            status = "provisional"
        elif status == "regen":
            summary["skipped"]["regen"] = summary["skipped"].get("regen", 0) + 1
            continue
        summary["lines"] += 1
        a = audio.get(line.audio_id or "")
        try:
            rec, cached, stretched = _master_line(ctx, cache, line, s, a, chars, project, status)
        except Exception as e:  # keep going: one broken line must not stop the patch
            ctx.log.error("master %s failed: %s", lid, e)
            summary["errors"][lid] = f"{type(e).__name__}: {e}"[:200]
            continue
        records.append(rec)
        summary["encoded"] += 1
        summary["cached"] += int(cached)
        summary["stretched"] += int(stretched)
    if ctx.only or ctx.limit is not None:
        from vndub.manifest.io import read_jsonl

        prev = {r.line_id: r for r in read_jsonl(ctx.paths.final_index, FinalIndexRecord)} if ctx.paths.final_index.exists() else {}
        mine = {r.line_id for r in records}
        records += [r for lid, r in prev.items() if lid not in mine]
    cache.save()
    write_jsonl(ctx.paths.final_index, records)
    for k in ("skipped", "errors"):
        if not summary[k]:
            summary.pop(k)
    provisional = sum(1 for r in records if r.status == "provisional")
    if provisional:
        summary["provisional"] = provisional
    return summary


def _master_line(ctx: Any, cache: Any, line: LineRecord, s: SelectionRecord, a: AudioIndexRecord | None, chars: dict[str, Any], project: Any, status: str) -> tuple[FinalIndexRecord, bool, bool]:
    cfg = ctx.cfg.master
    voice_file = line.voice_file or (a.relpath if a else f"{line.line_id}.wav")
    encoded = ctx.paths.final_encoded / voice_file
    final_wav = ctx.paths.final_wav(line.line_id)
    winner = ctx.paths.abs(s.winner)
    passport = gu.load_passport(ctx, line.line_id) or {}
    pros = passport.get("prosody") or {}
    sr_out = a.sr if a and a.sr else (project.voice_codec.sr if project else 44100)
    ch_out = a.ch if a and a.ch else (project.voice_codec.ch if project else 1)
    codec = _CODEC_MAP.get((a.codec if a else "") or (project.voice_codec.codec if project else "vorbis"), None)
    lufs_t, _ = target_lufs(ctx, line.speaker_id, chars)
    src_dur = gu.src_duration(ctx, line, passport, a)
    key = cache.key(s.winner, s.route, cfg.model_dump(), lufs_t, sr_out, ch_out, codec, voice_file, files=[winner])
    if cache.fresh(line.line_id, key, [encoded, final_wav]):
        meta = read_json(final_wav.with_suffix(".json")) or {}
        return FinalIndexRecord(
            line_id=line.line_id, voice_file=voice_file, final_wav=ctx.paths.rel(final_wav), encoded=ctx.paths.rel(encoded),
            source=s.winner if s.route != "orig" else "orig", status=status, stretch=float(meta.get("stretch", 1.0)),
            lufs=meta.get("lufs"), duration_s=float(meta.get("duration_s", 0.0)),
        ), True, False

    encoded.parent.mkdir(parents=True, exist_ok=True)
    final_wav.parent.mkdir(parents=True, exist_ok=True)
    if s.route == "orig":
        src_file = Path(a.src_path) if a and a.src_path else winner
        shutil.copy2(src_file, encoded)
        y, sr = features.load(ctx.paths.abs(line.src_wav) if line.src_wav else winner, mono=True)
        features.save(final_wav, y, sr)
        meta = {"stretch": 1.0, "lufs": loudness.lufs(y, sr), "duration_s": round(len(y) / sr, 3), "source": "orig"}
        write_json(final_wav.with_suffix(".json"), meta)
        cache.put(line.line_id, key)
        return FinalIndexRecord(line_id=line.line_id, voice_file=voice_file, final_wav=ctx.paths.rel(final_wav), encoded=ctx.paths.rel(encoded), source="orig", status=status, stretch=1.0, lufs=meta["lufs"], duration_s=meta["duration_s"]), False, False

    y, sr = features.load(winner, mono=True)
    if cfg.match_silence_profile:
        y = trim_to_profile(y, sr, float(pros.get("leading_silence_s", 0.05) or 0.0), float(pros.get("trailing_silence_s", 0.1) or 0.0))
    factor = 1.0
    stretched = False
    st = cfg.stretch
    if st.enabled and src_dur > 0:
        plan = stretch.plan(len(y) / sr, src_dur, tuple(st.target_window), st.max_stretch)
        if plan.apply:
            tmp_in = final_wav.with_name(final_wav.stem + ".prestretch.wav")
            tmp_out = final_wav.with_name(final_wav.stem + ".stretched.wav")
            features.save(tmp_in, y, sr)
            stretch.stretch(ctx.tools, tmp_in, tmp_out, plan.factor, st.engine)
            y, sr = features.load(tmp_out, mono=True)
            for p in (tmp_in, tmp_out):
                p.unlink(missing_ok=True)
            factor, stretched = plan.factor, True
        else:
            ctx.log.info("%s: no stretch (%s)", line.line_id, plan.reason)
    if sr != sr_out:
        import librosa

        y = librosa.resample(y, orig_sr=sr, target_sr=sr_out).astype(np.float32)
        sr = sr_out
    y, info = loudness.normalize(y, sr, lufs_t, ctx.cfg.master.loudness.true_peak_dbtp)
    out = np.stack([y, y], axis=1) if ch_out == 2 else y
    features.save(final_wav, out, sr)
    if codec is None:
        raise StageError(f"cannot re-encode to source codec '{a.codec if a else '?'}' for {voice_file}; write a converter or set the codec in project.yaml")
    ffmpeg.encode(ctx.tools, final_wav, encoded, codec, sr_out, ch_out, _bitrate(a, project))
    meta = {"stretch": round(factor, 4), "lufs": info.get("lufs_after"), "duration_s": round(len(y) / sr, 3), "target_lufs": lufs_t, "limiter_db": info.get("limiter_db")}
    write_json(final_wav.with_suffix(".json"), meta)
    cache.put(line.line_id, key)
    return FinalIndexRecord(
        line_id=line.line_id, voice_file=voice_file, final_wav=ctx.paths.rel(final_wav), encoded=ctx.paths.rel(encoded),
        source=s.winner, status=status, stretch=round(factor, 4), lufs=None if info.get("lufs_after") is None else round(info["lufs_after"], 2),
        duration_s=meta["duration_s"],
    ), False, stretched


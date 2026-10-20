"""Tiny synthetic projects for link / analyze / translate tests (no fixtures, no ffmpeg)."""

from __future__ import annotations

from typing import Any

import numpy as np
import soundfile as sf

from vndub.engines.base import build_characters
from vndub.manifest.io import AudioIndexRecord, LineRecord, save_characters, write_jsonl


def tone(path, seconds: float, f0: float, sr: int = 16000, seed: int = 0, n_syl: int = 4, amp: float = 0.3, noise: float = 0.0004, lead: float = 0.15, trail: float = 0.2) -> None:
    """Speech-like signal: harmonics with an F0 wobble, syllable envelope, silences at both ends."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    f = f0 * 2 ** ((1.5 * np.sin(2 * np.pi * 1.1 * t + seed)) / 12)
    ph = 2 * np.pi * np.cumsum(f) / sr
    y = sum((0.6 / k) * np.sin(k * ph) for k in range(1, 8))
    env = np.zeros(n)
    body = max(0.05, seconds - lead - trail)
    for s in range(n_syl):
        a = lead + body * s / n_syl
        b = lead + body * (s + 0.8) / n_syl
        m = (t >= a) & (t < b)
        env[m] = np.sin(np.pi * (t[m] - a) / (b - a)) ** 0.7
    y = amp * env * y / 2.5 + rng.normal(0, noise, n)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y.astype(np.float32), sr, subtype="PCM_16")


def build_project(ctx, rows: list[dict[str, Any]]) -> list[LineRecord]:
    """rows: {line_id, speaker_id, text, f0?, seconds?, voiced?, scene?, flags?, extra?, speaker_raw?}."""
    paths = ctx.paths
    lines: list[LineRecord] = []
    audio: list[AudioIndexRecord] = []
    for i, r in enumerate(rows):
        lid = r["line_id"]
        voiced = r.get("voiced", True)
        l = LineRecord(
            line_id=lid,
            scene=r.get("scene", "s01"),
            order=i,
            speaker_raw=r.get("speaker_raw", r["speaker_id"]),
            speaker_id=r["speaker_id"],
            text_src=r["text"],
            flags=list(r.get("flags", [])),
            source_file=r.get("source_file"),
            **r.get("extra", {}),
        )
        if voiced:
            aid = r.get("audio_id", f"{r['speaker_id'][:3]}_{i:04d}")
            src = paths.audio_src / f"{aid}.wav"
            src16 = paths.audio_src16 / f"{aid}.wav"
            secs = r.get("seconds", 1.5)
            tone(src, secs, r.get("f0", 220.0), sr=24000, seed=i, n_syl=r.get("n_syl", 4), amp=r.get("amp", 0.3))
            tone(src16, secs, r.get("f0", 220.0), sr=16000, seed=i, n_syl=r.get("n_syl", 4), amp=r.get("amp", 0.3))
            rel = f"voice/{aid}.ogg"
            audio.append(AudioIndexRecord(id=aid, relpath=rel, src_path=str(src), duration=secs, sr=24000, ch=1, codec="vorbis", wav=paths.rel(src), wav16=paths.rel(src16)))
            l.voice_file, l.audio_id, l.has_voice = rel, aid, True
            l.src_wav, l.src_wav16 = paths.rel(src), paths.rel(src16)
        lines.append(l)
    write_jsonl(paths.lines, lines)
    write_jsonl(paths.audio_index, audio)
    save_characters(paths.characters, build_characters(lines))
    return lines


def add_orphan(ctx, aid: str, seconds: float = 1.0, f0: float = 200.0) -> AudioIndexRecord:
    from vndub.manifest.io import read_jsonl

    paths = ctx.paths
    src = paths.audio_src / f"{aid}.wav"
    src16 = paths.audio_src16 / f"{aid}.wav"
    tone(src, seconds, f0, sr=24000, seed=99)
    tone(src16, seconds, f0, sr=16000, seed=99)
    rec = AudioIndexRecord(id=aid, relpath=f"voice/{aid}.ogg", src_path=str(src), duration=seconds, sr=24000, ch=1, codec="vorbis", wav=paths.rel(src), wav16=paths.rel(src16))
    audio = read_jsonl(paths.audio_index, AudioIndexRecord) + [rec]
    write_jsonl(paths.audio_index, audio)
    return rec

"""Synthetic project builder for generation-side tests (voicebank/synth/score/master/report/review).

Writes, under a StageContext's work dir, everything the generation stages read:
project.yaml, audio_index.jsonl + decoded wavs, lines.jsonl, characters.yaml,
analysis passports (computed with the mock analyzers so embeddings match the
mock scorers), and a Russian translation. Audio is numpy-synthesized speech-like
signal, so no fixtures or downloads are needed.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from vndub.audio import features, loudness, vad
from vndub.backends import registry
from vndub.backends.mock import MockADV, MockSER
from vndub.config import ProjectYaml, VoiceCodec, save_project_yaml
from vndub.manifest.io import (
    AudioIndexRecord,
    Character,
    CharacterStats,
    LineRecord,
    Passport,
    TranslationRecord,
    save_characters,
    write_json,
    write_jsonl,
)
from vndub.text.style_prompt import build_style

SPEAKERS = {"sakura": ("f", 260.0), "haruto": ("m", 130.0)}
RU = ["Привет, это тест.", "Подожди немного!", "Что случилось?", "Сегодня хороший день."]


def speechlike(path: Path, dur: float, f0: float, sr: int = 24000, seed: int = 0, amp: float = 0.3, lead: float = 0.08, trail: float = 0.12, stereo: bool = False) -> Path:
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    t = np.arange(n) / sr
    f = f0 * 2 ** ((1.5 * np.sin(2 * np.pi * 1.1 * t + seed)) / 12)
    ph = 2 * np.pi * np.cumsum(f) / sr
    y = sum((0.5 / k) * np.sin(k * ph) for k in range(1, 8))
    body = (t >= lead) & (t <= dur - trail)
    syl = np.clip(np.sin(2 * np.pi * 3.0 * (t - lead)), 0, None) ** 0.5
    y = amp * y * syl * body / 2.0 + rng.normal(0, 0.002, n)
    y = y.astype(np.float32)
    if stereo:
        y = np.stack([y, y], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    features.save(path, y, sr)
    return path


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def build(ctx: Any, n_per: int = 3, codec: str = "wav", stereo_for: tuple[str, ...] = (), interjection: bool = True, src_sr: int = 24000) -> dict[str, Any]:
    p = ctx.paths
    game = Path(ctx.project_dir)
    game.mkdir(parents=True, exist_ok=True)
    ext = ".ogg" if codec == "vorbis" else ".wav"
    save_project_yaml(
        p.project_yaml,
        ProjectYaml(game_dir=str(game), engine="renpy", voice_codec=VoiceCodec(container=ext.strip("."), codec=codec if codec != "wav" else "pcm_s16le", sr=src_sr, ch=1, bitrate_k=64)),
    )
    ser = MockSER(registry.get("mock_ser"))
    adv = MockADV(registry.get("mock_adv"))
    audio, lines, trs = [], [], []
    chars: dict[str, Character] = {}
    order = 0
    for si, (sid, (gender, f0)) in enumerate(SPEAKERS.items()):
        stereo = sid in stereo_for
        entries = [(f"テストです{i}。", RU[i % len(RU)], False) for i in range(n_per)]
        if interjection and si == 0:
            entries.append(("えへへ。", "Хе-хе.", True))
        durs, lufs_vals = [], []
        for i, (ja, ru, interj) in enumerate(entries):
            order += 1
            aid = f"{sid[:3]}_{i + 1:04d}"
            lid = f"s01_{order:06d}"
            dur = 0.9 if interj else 1.6 + 0.3 * i
            raw = game / "game" / "voice" / f"{aid}{ext}"
            wav = p.audio_src / f"{aid}.wav"
            speechlike(wav, dur, f0 * (1 + 0.03 * i), sr=src_sr, seed=order, stereo=False)
            if stereo:
                speechlike(p.audio_src_stereo / f"{aid}.wav", dur, f0 * (1 + 0.03 * i), sr=src_sr, seed=order, stereo=True)
            y16, _ = features.load(wav, sr=16000)
            wav16 = p.audio_src16 / f"{aid}.wav"
            features.save(wav16, y16, 16000)
            raw.parent.mkdir(parents=True, exist_ok=True)
            src_for_raw = p.audio_src_stereo / f"{aid}.wav" if stereo else wav
            if codec == "vorbis":
                subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src_for_raw), "-c:a", "libvorbis", "-b:a", "64k", str(raw)], check=True)
            else:
                shutil.copy2(src_for_raw, raw)
            ch = 2 if stereo else 1
            audio.append(
                AudioIndexRecord(
                    id=aid, relpath=f"voice/{aid}{ext}", src_path=str(raw), duration=dur, sr=src_sr, ch=ch,
                    codec="vorbis" if codec == "vorbis" else "pcm_s16le", bytes=raw.stat().st_size,
                    wav=p.rel(wav), wav16=p.rel(wav16), wav_stereo=p.rel(p.audio_src_stereo / f"{aid}.wav") if stereo else "",
                )
            )
            lines.append(
                LineRecord(
                    line_id=lid, scene="s01", order=order, speaker_raw=sid, speaker_id=sid, text_src=ja,
                    voice_file=f"voice/{aid}{ext}", has_voice=True, audio_id=aid, src_wav=p.rel(wav), src_wav16=p.rel(wav16),
                )
            )
            # passport via the mock analyzers (embedding space == mock scorer's)
            emb = p.e2v_emb(lid)
            e = ser.op_emotion({"wav": str(wav16), "out": str(emb)})
            a = adv.op_adv({"wav": str(wav16)})
            y, sr = features.load(wav16, sr=16000)
            f0c = features.f0_pyin(y, sr)
            en = features.rms_db(y, sr)
            p.feat(lid, "f0").parent.mkdir(parents=True, exist_ok=True)
            np.save(p.feat(lid, "f0"), f0c)
            np.save(p.feat(lid, "energy"), en)
            prof = vad.silence_profile(vad.energy_vad(y, sr), len(y) / sr)
            st = features.f0_stats(f0c)
            ln = loudness.lufs(features.load(wav)[0], src_sr)
            lufs_vals.append(ln)
            durs.append(dur)
            pp = {
                "line_id": lid, "duration_s": dur, "snr_db": 30.0, "bgm": False,
                "emotion": {"e2v": {"probs": e["probs"], "label": e["label"], "embedding_path": p.rel(emb)}, "anime": {"label": "Happy", "probs": {"Happy": 0.6, "Neutral": 0.4}}, "adv": {k: a[k] for k in ("arousal", "dominance", "valence")}},
                "events": [],
                "prosody": {
                    "f0_median_hz": st["f0_median_hz"], "f0_std_st": st["f0_std_st"], "voiced_ratio": st["voiced_ratio"],
                    "energy_db_mean": float(np.mean(en)), "f0_contour_path": p.rel(p.feat(lid, "f0")), "energy_contour_path": p.rel(p.feat(lid, "energy")),
                    "rate_mora_per_s": 7.0, "pauses": prof["pauses"], "leading_silence_s": prof["leading_silence_s"],
                    "trailing_silence_s": prof["trailing_silence_s"], "speech_s": prof["speech_s"], "arc": "flat", "lufs": ln,
                },
                "backends": {"e2v": "mock_ser", "adv": "mock_adv"},
            }
            pp["style"] = build_style(pp, {"f0_median_hz": f0, "rate_median": 7.0, "energy_db_median": float(np.mean(en))}).to_dict()
            write_json(p.passport(lid), Passport.model_validate(pp).model_dump(mode="json"))
            flags = ["interjection"] if interj else []
            trs.append(TranslationRecord(line_id=lid, text_tgt=ru, text_tgt_short=ru, text_tts=ru, text_plain=ru, source="test", flags=flags))
        chars[sid] = Character(
            names={"src": sid, "tgt": sid.capitalize()}, gender=gender, age_hint="teen",
            stats=CharacterStats(n_lines=len(entries), total_sec=sum(durs), f0_median_hz=f0, energy_db_median=-25.0, lufs_median=float(np.median([v for v in lufs_vals if v is not None])), rate_median=7.0, emotion_hist={"happy": 0.6, "neutral": 0.4}),
        )
    write_jsonl(p.audio_index, audio)
    write_jsonl(p.lines, lines)
    save_characters(p.characters, chars)
    write_jsonl(p.translation(ctx.cfg.project.tgt_lang), trs)
    return {"lines": lines, "audio": audio, "chars": chars, "translations": trs}

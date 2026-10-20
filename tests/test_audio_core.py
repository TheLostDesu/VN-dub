"""Core audio helpers: atempo chains, stretch planning, loudness + limiter, VAD, features, prosody/CER metrics."""

from __future__ import annotations

import numpy as np
import pytest

from vndub.audio import features, loudness, stretch, vad
from vndub.audio.ffmpeg import atempo_chain
from vndub.backends.score import cer, prosody

SR = 16000


def _tone(sec: float, f0: float = 200.0, amp: float = 0.3, sr: int = SR) -> np.ndarray:
    t = np.arange(int(sec * sr)) / sr
    return (amp * np.sin(2 * np.pi * f0 * t)).astype(np.float32)


def test_atempo_chain_bounds() -> None:
    for r in (0.3, 0.5, 0.9, 1.0, 1.7, 3.5):
        stages = [float(s.split("=")[1]) for s in atempo_chain(r).split(",")]
        assert all(0.5 <= s <= 2.0 for s in stages)
        assert np.prod(stages) == pytest.approx(r, rel=1e-4)


def test_stretch_plan_window_and_limit() -> None:
    assert not stretch.plan(2.0, 2.0).apply
    p = stretch.plan(2.4, 2.0)  # ratio 1.2 -> stretch to the 1.1 edge
    assert p.apply and p.factor == pytest.approx(1.1 / 1.2)
    p = stretch.plan(3.0, 2.0)  # ratio 1.5 needs >15 %
    assert not p.apply and "max" in p.reason
    p = stretch.plan(1.6, 2.0)  # ratio 0.8 -> stretch up to 0.9
    assert p.apply and p.factor == pytest.approx(0.9 / 0.8)


def test_loudness_normalize_and_true_peak() -> None:
    y = _tone(2.0, amp=0.05)
    out, info = loudness.normalize(y, SR, target_lufs=-20.0, ceiling_dbtp=-1.0)
    assert info["lufs_after"] == pytest.approx(-20.0, abs=1.0)
    assert loudness.true_peak_db(out, SR) <= -1.0 + 0.05
    loud = _tone(2.0, amp=0.9)
    out2, info2 = loudness.normalize(loud, SR, target_lufs=-1.0, ceiling_dbtp=-1.0)  # gain pushes peaks over the ceiling
    assert loudness.true_peak_db(out2, SR) <= -1.0 + 0.05
    assert info2["limiter_db"] < 0


def test_energy_vad_and_silence_profile() -> None:
    y = np.concatenate([np.zeros(int(0.3 * SR)), _tone(0.5), np.zeros(int(0.4 * SR)), _tone(0.6), np.zeros(int(0.2 * SR))]).astype(np.float32)
    segs = vad.energy_vad(y, SR)
    assert len(segs) == 2
    prof = vad.silence_profile(segs, len(y) / SR)
    assert prof["leading_silence_s"] == pytest.approx(0.3, abs=0.05)
    assert prof["trailing_silence_s"] == pytest.approx(0.2, abs=0.05)
    assert len(prof["pauses"]) == 1 and prof["pauses"][0]["d"] == pytest.approx(0.4, abs=0.06)


def test_f0_stats_and_arc() -> None:
    f0 = np.full(200, 220.0)
    st = features.f0_stats(f0)
    assert st["f0_median_hz"] == pytest.approx(220.0) and st["f0_std_st"] == pytest.approx(0.0)
    rising = np.concatenate([np.full(140, 200.0), 200.0 * 2 ** (np.linspace(0, 6, 60) / 12)])
    assert features.arc(rising) == "rising_end"
    assert features.arc(rising[::-1].copy()) in ("flat", "rising_end", "falling_end")
    falling = np.concatenate([np.full(140, 250.0), 250.0 * 2 ** (-np.linspace(0, 6, 60) / 12)])
    assert features.arc(falling) == "falling_end"
    assert features.arc(np.zeros(50)) == "flat"


def test_pyin_on_tone() -> None:
    f0 = features.f0_pyin(_tone(1.0, f0=180.0), SR)
    voiced = f0[f0 > 0]
    assert voiced.size > 50 and np.median(voiced) == pytest.approx(180.0, rel=0.03)


def test_prosody_metrics() -> None:
    a = 200 * 2 ** (np.sin(np.linspace(0, 3 * np.pi, 120)) / 12)
    b = 230 * 2 ** (np.sin(np.linspace(0, 3 * np.pi, 90)) / 12)  # same shape, other speed/offset
    assert prosody.f0_corr(a, b) > 0.9
    ramp = 200 * 2 ** (np.linspace(0, 8, 100) / 12)
    assert prosody.f0_corr(ramp, ramp[::-1].copy()) < 0.5  # DTW cannot turn a fall into a rise
    assert prosody.f0_corr(np.zeros(10), b) is None
    assert prosody.pause_match([], 2.0, [], 2.5) == 1.0
    assert prosody.pause_match([{"t": 1.0, "d": 0.3}], 2.0, [{"t": 1.2, "d": 0.3}], 2.4) == 1.0
    assert prosody.pause_match([{"t": 0.2, "d": 0.2}], 2.0, [{"t": 1.6, "d": 0.2}], 2.0) == 0.0
    assert prosody.prosody_score(1.0, None, 1.0) == 1.0


def test_cer_normalization() -> None:
    assert cer.normalize_for_cer("Всё в поря+дке, Ёжик!") == "все в порядке ежик"
    assert cer.cer("Привет, мир!", "привет мир") == 0.0
    assert cer.cer("абв", "") == 1.0
    assert 0 < cer.cer("кошка", "кошки") < 0.5

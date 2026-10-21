"""CPU audio features: loading, RMS energy, F0 (pYIN fallback), F0 stats, SNR, tilt, arc."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

HOP_S = 0.01


def load(path: str | Path, sr: int | None = None, mono: bool = True) -> tuple[np.ndarray, int]:
    y, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1) if mono else y.T
    if sr and sr != file_sr:
        y = resample(y, file_sr, sr) if mono else np.stack([resample(c, file_sr, sr) for c in y])
        file_sr = sr
    return np.ascontiguousarray(y, dtype=np.float32), file_sr


def resample(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase resampling (scipy). librosa's default soxr_hq measured ~150x slower on the
    build machine (1.2 s vs 8 ms per 3 s clip) with no audible benefit for speech."""
    if sr_in == sr_out:
        return y.astype(np.float32, copy=False)
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(int(sr_in), int(sr_out))
    return resample_poly(y, int(sr_out) // g, int(sr_in) // g, window=("kaiser", 8.0)).astype(np.float32)


def save(path: str | Path, y: np.ndarray, sr: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = y.T if y.ndim == 2 and y.shape[0] < y.shape[1] else y
    sf.write(str(path), data, sr, subtype="PCM_16")


def duration(path: str | Path) -> float:
    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate or 1)


def rms_db(y: np.ndarray, sr: int, hop_s: float = HOP_S, win_s: float = 0.025) -> np.ndarray:
    import librosa

    hop = max(1, int(sr * hop_s))
    win = max(hop, int(sr * win_s))
    if y.size < win:
        y = np.pad(y, (0, win - y.size))
    rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop, center=True)[0]
    return 20.0 * np.log10(np.maximum(rms, 1e-6))


def f0_pyin(y: np.ndarray, sr: int, fmin: float = 60.0, fmax: float = 1000.0, hop_s: float = HOP_S, resolution: float = 0.25) -> np.ndarray:
    """F0 in Hz per hop with 0 for unvoiced frames (librosa pYIN, dependency-free).

    ``resolution`` is the pitch-bin width in semitones; 0.25 is ~4x faster than
    librosa's 0.1 default and ample for contour statistics. The first call in a
    process pays a one-time numba compilation (~20 s).
    """
    import librosa

    target_sr = 16000
    if sr != target_sr:
        y = resample(y, sr, target_sr)
        sr = target_sr
    hop = int(sr * hop_s)
    if y.size < 2048:
        y = np.pad(y, (0, 2048 - y.size))
    f0, voiced, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=1024, hop_length=hop, resolution=resolution)
    f0 = np.where(voiced & np.isfinite(f0), f0, 0.0)
    return f0.astype(np.float32)


def semitones(f: np.ndarray | float, ref: float) -> np.ndarray | float:
    return 12.0 * np.log2(np.asarray(f) / ref)


def f0_stats(f0: np.ndarray) -> dict[str, float | None]:
    v = f0[f0 > 0]
    if v.size < 3:
        return {"f0_median_hz": None, "f0_std_st": None, "f0_range_st": None, "f0_iqr_st": None, "voiced_ratio": float(v.size / max(1, f0.size))}
    med = float(np.median(v))
    st = semitones(v, med)
    p5, p25, p75, p95 = np.percentile(st, [5, 25, 75, 95])
    return {
        "f0_median_hz": round(med, 2),
        "f0_std_st": round(float(np.std(st)), 3),
        "f0_range_st": round(float(p95 - p5), 3),
        "f0_iqr_st": round(float(p75 - p25), 3),
        "voiced_ratio": round(float(v.size / max(1, f0.size)), 3),
    }


def arc(f0: np.ndarray, hop_s: float = HOP_S, threshold_st_per_s: float = 4.0) -> str:
    """Intonation arc from the slope of the last voiced third (semitones/second)."""
    idx = np.nonzero(f0 > 0)[0]
    if idx.size < 6:
        return "flat"
    tail = idx[len(idx) * 2 // 3 :]
    if tail.size < 3:
        return "flat"
    med = float(np.median(f0[idx]))
    t = tail * hop_s
    st = semitones(f0[tail], med)
    slope = float(np.polyfit(t, st, 1)[0])
    if slope > threshold_st_per_s:
        return "rising_end"
    if slope < -threshold_st_per_s:
        return "falling_end"
    return "flat"


def snr_db(energy_db: np.ndarray, speech_mask: np.ndarray) -> float | None:
    """Speech-vs-non-speech energy ratio (dB). None when there is no non-speech region."""
    if speech_mask.size != energy_db.size:
        n = min(speech_mask.size, energy_db.size)
        speech_mask, energy_db = speech_mask[:n], energy_db[:n]
    sp = energy_db[speech_mask]
    ns = energy_db[~speech_mask]
    if sp.size == 0:
        return None
    if ns.size < 5:
        return 40.0  # no measurable noise floor: tightly trimmed clean clip
    p_sp = np.mean(10 ** (sp / 10))
    p_ns = np.mean(10 ** (ns / 10))
    return float(10 * np.log10(max(p_sp, 1e-12) / max(p_ns, 1e-12)))


def spectral_tilt(y: np.ndarray, sr: int) -> float:
    """Slope (dB/kHz) of the average log-magnitude spectrum; breathy/whisper speech is flatter."""
    if y.size < 512:
        return 0.0
    spec = np.abs(np.fft.rfft(y * np.hanning(y.size)))
    freqs = np.fft.rfftfreq(y.size, 1.0 / sr)
    m = (freqs > 100) & (freqs < min(8000, sr / 2))
    if m.sum() < 10:
        return 0.0
    db = 20 * np.log10(np.maximum(spec[m], 1e-9))
    return float(np.polyfit(freqs[m] / 1000.0, db, 1)[0])


def mask_from_segments(n_frames: int, segments: list[tuple[float, float]], hop_s: float = HOP_S) -> np.ndarray:
    mask = np.zeros(n_frames, dtype=bool)
    for t0, t1 in segments:
        a, b = int(t0 / hop_s), int(np.ceil(t1 / hop_s))
        mask[max(0, a) : min(n_frames, b)] = True
    return mask

"""Time stretch toward the source duration: Rubber Band CLI, ffmpeg atempo fallback.

``master`` stretches only when the duration ratio is outside the target window
and the required change is within ``max_stretch``; otherwise it keeps the
candidate as is (and records why).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from vndub.audio import ffmpeg
from vndub.tools import Tools


@dataclass
class StretchPlan:
    apply: bool
    factor: float  # output_duration / input_duration
    reason: str


def plan(cand_dur: float, src_dur: float, window: tuple[float, float] = (0.9, 1.1), max_stretch: float = 0.15) -> StretchPlan:
    if cand_dur <= 0 or src_dur <= 0:
        return StretchPlan(False, 1.0, "zero duration")
    ratio = cand_dur / src_dur
    lo, hi = window
    if lo <= ratio <= hi:
        return StretchPlan(False, 1.0, f"ratio {ratio:.3f} within window")
    # stretch to the nearest window edge, not all the way to 1.0
    target_ratio = hi if ratio > hi else lo
    factor = target_ratio / ratio
    if abs(1.0 - factor) > max_stretch:
        return StretchPlan(False, 1.0, f"ratio {ratio:.3f} needs {abs(1 - factor):.1%} > max {max_stretch:.0%}")
    return StretchPlan(True, factor, f"ratio {ratio:.3f} -> {target_ratio:.2f}")


def stretch(tools: Tools, src: Path, dst: Path, factor: float, engine: str = "rubberband") -> str:
    """Write ``dst`` with duration ``factor`` x input. Returns the engine actually used."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if engine == "rubberband":
        rb = tools.resolve("rubberband")
        if rb is not None:
            r = subprocess.run(
                [str(rb), "-q", "--fine", "-t", f"{factor:.6f}", str(src), str(dst)],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            if r.returncode == 0 and dst.exists():
                return "rubberband"
    # atempo: tempo = 1/factor
    ffmpeg.atempo(tools, src, dst, 1.0 / factor)
    return "atempo"


def retune(tools: Tools, src: Path, dst: Path, factor: float = 1.0, pitch_st: float = 0.0, engine: str = "rubberband") -> str:
    """Write ``dst`` with duration ``factor`` x the input and the pitch moved by ``pitch_st`` semitones.

    ``master``'s last-resort pass for the voice tuning (backends with no native pitch / speed knob).
    Rubber Band does both in one pass (``-t`` and ``-p``); without it ffmpeg shifts pitch and speed
    together (``asetrate`` + ``aresample``) and ``atempo`` then restores the duration and applies ``factor``.
    Returns the engine actually used.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if engine == "rubberband":
        rb = tools.resolve("rubberband")
        if rb is not None:
            cmd = [str(rb), "-q", "--fine", "-t", f"{factor:.6f}"]
            if pitch_st:
                cmd += ["-p", f"{pitch_st:.4f}"]
            r = subprocess.run([*cmd, str(src), str(dst)], capture_output=True, text=True, encoding="utf-8", errors="replace")
            if r.returncode == 0 and dst.exists():
                return "rubberband"
    import soundfile as sf

    ratio = 2.0 ** (pitch_st / 12.0)
    sr = int(sf.info(str(src)).samplerate)
    filters = []
    if pitch_st:
        filters += [f"asetrate={sr * ratio:.3f}", f"aresample={sr}"]
    filters.append(ffmpeg.atempo_chain(1.0 / (ratio * factor)))
    ff = tools.require("ffmpeg", "pitch / tempo pass")
    ffmpeg._run([str(ff), "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-filter:a", ",".join(filters), "-c:a", "pcm_s16le", str(dst)])
    return "atempo"

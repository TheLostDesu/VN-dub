"""ffmpeg / ffprobe / vgmstream wrappers (subprocess; paths from tools.yaml)."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from vndub.tools import Tools


class AudioToolError(RuntimeError):
    pass


@dataclass
class Probe:
    duration: float
    sr: int
    ch: int
    codec: str
    container: str
    bitrate_k: int | None


def _run(args: list[str], timeout: float = 300) -> subprocess.CompletedProcess[str]:
    try:
        r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired as e:  # one error type for every tool failure: callers catch AudioToolError per file
        raise AudioToolError(f"{Path(args[0]).name} timed out after {timeout:.0f} s") from e
    if r.returncode != 0:
        raise AudioToolError(f"{Path(args[0]).name} failed ({r.returncode}): {r.stderr.strip()[-800:]}")
    return r


def probe(tools: Tools, path: Path) -> Probe:
    ffprobe = tools.require("ffprobe", "probing audio")
    r = _run([str(ffprobe), "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)])
    data = json.loads(r.stdout or "{}")
    streams = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
    if not streams:
        raise AudioToolError(f"no audio stream in {path}")
    s = streams[0]
    fmt = data.get("format", {})
    dur = float(s.get("duration") or fmt.get("duration") or 0.0)
    br = s.get("bit_rate") or fmt.get("bit_rate")
    return Probe(
        duration=dur,
        sr=int(s.get("sample_rate") or 0),
        ch=int(s.get("channels") or 0),
        codec=str(s.get("codec_name") or ""),
        container=str(fmt.get("format_name") or "").split(",")[0],
        bitrate_k=int(int(br) / 1000) if br else None,
    )


def decode(tools: Tools, src: Path, dst: Path, sr: int | None = None, mono: bool = True) -> None:
    """Decode anything ffmpeg reads into 16-bit PCM WAV."""
    ffmpeg = tools.require("ffmpeg", "decoding audio")
    dst.parent.mkdir(parents=True, exist_ok=True)
    args = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-i", str(src)]
    if mono:
        args += ["-ac", "1"]
    if sr:
        args += ["-ar", str(sr)]
    args += ["-c:a", "pcm_s16le", str(dst)]
    _run(args)


def decode_vgmstream(tools: Tools, src: Path, dst: Path) -> None:
    vgm = tools.require("vgmstream_cli", "decoding proprietary game audio")
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run([str(vgm), "-o", str(dst), str(src)])


_CODEC_ARGS = {
    "vorbis": ["-c:a", "libvorbis"],
    "opus": ["-c:a", "libopus"],
    "mp3": ["-c:a", "libmp3lame"],
    "pcm_s16le": ["-c:a", "pcm_s16le"],
    "flac": ["-c:a", "flac"],
    "aac": ["-c:a", "aac"],
}


def encode(tools: Tools, src_wav: Path, dst: Path, codec: str, sr: int, ch: int, bitrate_k: int | None = None) -> None:
    """Encode to the original codec/sr/channels. The container follows ``dst``'s extension."""
    ffmpeg = tools.require("ffmpeg", "encoding audio")
    dst.parent.mkdir(parents=True, exist_ok=True)
    codec_args = _CODEC_ARGS.get(codec)
    if codec_args is None:
        raise AudioToolError(f"no encoder mapping for codec '{codec}'; add it to vndub.audio.ffmpeg._CODEC_ARGS")
    args = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-i", str(src_wav), "-ar", str(sr), "-ac", str(ch), *codec_args]
    if bitrate_k and codec in ("vorbis", "opus", "mp3", "aac"):
        args += ["-b:a", f"{bitrate_k}k"]
    # ffmpeg picks the muxer from the extension; .ogg + opus needs an explicit muxer
    if dst.suffix.lower() == ".ogg":
        args += ["-f", "ogg"]
    args.append(str(dst))
    _run(args)


def atempo_chain(ratio: float) -> str:
    """ffmpeg atempo filter chain for a tempo factor (each stage within [0.5, 2])."""
    if ratio <= 0:
        raise ValueError("ratio must be > 0")
    parts = []
    r = ratio
    while r > 2.0:
        parts.append(2.0)
        r /= 2.0
    while r < 0.5:
        parts.append(0.5)
        r /= 0.5
    parts.append(r)
    return ",".join(f"atempo={p:.6f}" for p in parts)


def atempo(tools: Tools, src: Path, dst: Path, tempo: float) -> None:
    """Change duration by 1/tempo keeping pitch (ffmpeg atempo)."""
    ffmpeg = tools.require("ffmpeg", "atempo stretch")
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run([str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-filter:a", atempo_chain(tempo), "-c:a", "pcm_s16le", str(dst)])


def copy_or_convert(tools: Tools, src: Path, dst: Path) -> None:
    if src.suffix.lower() == dst.suffix.lower():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    else:
        decode(tools, src, dst, mono=False)


def tmp_wav(suffix: str = ".wav") -> Path:
    fd, p = tempfile.mkstemp(suffix=suffix)
    import os

    os.close(fd)
    return Path(p)

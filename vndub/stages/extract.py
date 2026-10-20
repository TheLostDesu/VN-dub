"""extract: engine extraction, then decode every voice file to WAV.

Outputs (work-relative): ``raw/extract.json`` (scripts, voice roots),
``audio/src/<id>.wav`` (original sample rate, mono), ``audio/src_stereo/<id>.wav``
(stereo originals, when ``extract.keep_stereo_copy``), ``audio/src16/<id>.wav``
(16 kHz mono, analysis), and ``audio_index.jsonl``. ffmpeg decodes common
formats, ``vgmstream-cli`` the proprietary ones (``extract.vgmstream_exts``).
Items are cached by file content hash; failures are recorded per file.
"""

from __future__ import annotations

import collections
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any

from vndub.audio import ffmpeg
from vndub.config import VoiceCodec, save_project_yaml
from vndub.engines.base import ExtractResult
from vndub.manifest.io import AudioIndexRecord, read_jsonl, write_jsonl
from vndub.stages.common import StageContext

NAME = "extract"
VERSION = "1"
CONFIG_SECTIONS = ("extract",)

_ID_RE = re.compile(r"[^0-9A-Za-z_\-\.()]+")


def assign_ids(relpaths: list[str]) -> dict[str, str]:
    """relpath -> audio id: the stem, prefixed with parent dirs only where stems collide."""
    stems = collections.Counter(PurePosixPath(r).stem.lower() for r in relpaths)
    out: dict[str, str] = {}
    used: set[str] = set()
    for r in relpaths:
        p = PurePosixPath(r)
        if stems[p.stem.lower()] == 1:
            cand = p.stem
        else:
            cand = "__".join([*p.parent.parts, p.stem]) if p.parent.parts else p.stem
        cand = _ID_RE.sub("_", cand) or "audio"
        base, k = cand, 1
        while cand.lower() in used:
            k += 1
            cand = f"{base}_{k}"
        used.add(cand.lower())
        out[r] = cand
    return out


def _decode_one(ctx: StageContext, src: Path, rec: AudioIndexRecord, keep_stereo: bool, sr16: int, use_vgm: bool) -> AudioIndexRecord:
    paths = ctx.paths
    wav = paths.audio_src / f"{rec.id}.wav"
    wav16 = paths.audio_src16 / f"{rec.id}.wav"
    input_path = src
    tmp: Path | None = None
    try:
        if use_vgm:
            tmp = paths.audio_src / f".{rec.id}.vgm.wav"
            ffmpeg.decode_vgmstream(ctx.tools, src, tmp)
            input_path = tmp
        pr = ffmpeg.probe(ctx.tools, input_path)
        rec.duration, rec.sr, rec.ch = round(pr.duration, 4), pr.sr, pr.ch
        rec.codec = pr.codec if not use_vgm else f"vgmstream:{src.suffix.lstrip('.')}"
        ffmpeg.decode(ctx.tools, input_path, wav, sr=None, mono=True)
        ffmpeg.decode(ctx.tools, input_path, wav16, sr=sr16, mono=True)
        rec.wav = paths.rel(wav)
        rec.wav16 = paths.rel(wav16)
        if pr.ch > 1 and keep_stereo:
            st = paths.audio_src_stereo / f"{rec.id}.wav"
            ffmpeg.decode(ctx.tools, input_path, st, sr=None, mono=False)
            rec.wav_stereo = paths.rel(st)
        if rec.duration <= 0:
            import soundfile as sf

            info = sf.info(str(wav))
            rec.duration = round(info.frames / float(info.samplerate or 1), 4)
        rec.error = ""
    except Exception as e:
        rec.error = f"{type(e).__name__}: {e}"[:500]
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()
    return rec


def run(ctx: StageContext) -> dict[str, Any]:
    proj = ctx.require_project()
    eng = ctx.engine()
    res: ExtractResult = eng.extract(ctx)
    ctx.paths.raw.mkdir(parents=True, exist_ok=True)
    res.save(ctx.paths.raw / "extract.json")
    for w in res.warnings:
        ctx.log.warning(w)
    ctx.tools.require("ffmpeg", "decoding voices")
    ctx.tools.require("ffprobe", "probing voices")
    vgm_exts = {e.lower() for e in ctx.cfg.extract.vgmstream_exts}
    files = [f for f in res.voice_files if f.exists()]
    rels = [res.relpath(f) for f in files]
    ids = assign_ids(rels)
    cache = ctx.cache(NAME, VERSION, "extract")
    previous = {r.id: r for r in read_jsonl(ctx.paths.audio_index, AudioIndexRecord)} if ctx.paths.audio_index.exists() else {}
    todo: list[tuple[Path, AudioIndexRecord, str, bool]] = []
    records: dict[str, AudioIndexRecord] = {}
    for f, rel in zip(files, rels):
        aid = ids[rel]
        rec = AudioIndexRecord(id=aid, relpath=rel, src_path=str(f.resolve()), bytes=f.stat().st_size)
        use_vgm = f.suffix.lower() in vgm_exts
        key = cache.key(rel, use_vgm, ctx.cfg.extract.analysis_sr, ctx.cfg.extract.keep_stereo_copy, files=[f])
        outs = [ctx.paths.audio_src / f"{aid}.wav", ctx.paths.audio_src16 / f"{aid}.wav"]
        if cache.fresh(aid, key, outs) and aid in previous and not previous[aid].error:
            records[aid] = previous[aid]
            continue
        todo.append((f, rec, key, use_vgm))
        records[aid] = rec
    if todo and any(u for *_, u in todo) and ctx.tools.resolve("vgmstream_cli") is None:
        ctx.log.warning("vgmstream-cli not configured: proprietary formats will fail to decode (tools.yaml: vgmstream_cli)")
    errors = 0
    with ThreadPoolExecutor(max_workers=max(1, ctx.cfg.extract.workers)) as pool:
        futs = {pool.submit(_decode_one, ctx, f, rec, ctx.cfg.extract.keep_stereo_copy, ctx.cfg.extract.analysis_sr, vgm): (rec, key) for f, rec, key, vgm in todo}
        for fut in as_completed(futs):
            rec, key = futs[fut]
            out = fut.result()
            records[out.id] = out
            if out.error:
                errors += 1
                ctx.log.warning("decode failed %s: %s", out.relpath, out.error)
            else:
                cache.put(out.id, key)
    ordered = [records[ids[r]] for r in rels]
    write_jsonl(ctx.paths.audio_index, ordered)
    ok = [r for r in ordered if not r.error]
    # codec summary -> project.yaml when detect could not probe (archived voices)
    if ok:
        cnt = collections.Counter((r.codec, r.sr, r.ch) for r in ok)
        (codec, sr, ch), _ = cnt.most_common(1)[0]
        ext = PurePosixPath(ok[0].relpath).suffix.lstrip(".").lower()
        if not getattr(proj, "voice_codec_detected", False):
            proj.voice_codec = VoiceCodec(container=ext or proj.voice_codec.container, codec=codec, sr=sr, ch=ch, bitrate_k=proj.voice_codec.bitrate_k)
            proj.voice_codec_detected = True  # type: ignore[attr-defined]
            save_project_yaml(ctx.paths.project_yaml, proj)
    if not files:
        ctx.log.warning("no voice files found")
    total_s = sum(r.duration for r in ok)
    return {
        "engine": proj.engine,
        "scripts": len(res.script_files),
        "voice_files": len(files),
        "decoded": len(ok),
        "decoded_now": len(todo) - errors,
        "cached": len(files) - len(todo),
        "errors": errors,
        "total_audio_min": round(total_s / 60, 2),
        "warnings": res.warnings[:5],
    }

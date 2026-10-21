"""detect: identify the engine (signatures/extensions) and write ``work/project.yaml``.

All engine plugins are asked; the highest confidence wins unless ``--engine``
(or ``parse.engine`` != auto) forces one. Encryption heuristics and DRM
markers become warnings. Re-running keeps user-edited fields of an existing
project.yaml (languages, notes, codec) unless ``--force``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vndub.audio import ffmpeg
from vndub.config import ProjectYaml, VoiceCodec, load_project_yaml, save_project_yaml
from vndub.engines import ENGINES, all_engines
from vndub.engines.base import DetectResult
from vndub.stages.common import StageContext, StageError

NAME = "detect"
VERSION = "1"
CONFIG_SECTIONS = ("parse",)

DRM_MARKERS = {
    "steam": ["steam_api.dll", "steam_api64.dll", "steam_appid.txt"],
    "softdenchi": ["SoftDenchi", "sdl.dll", "SDProtect"],
    "dmm": ["dmmgame.cnf", "DMMGamePlayer"],
    "dlsite_playdrm": ["PlayDRM", "DLsiteDRM"],
    "enigma": ["enigma_ide.dll"],
}

KEEP_FIELDS = ("src_lang", "tgt_lang", "notes", "voice_codec")


def find_drm(game: Path) -> list[str]:
    hits: list[str] = []
    try:
        names = [p.name.lower() for p in game.iterdir()]
    except OSError:
        return hits
    for kind, markers in DRM_MARKERS.items():
        for m in markers:
            if any(m.lower() in n for n in names):
                hits.append(kind)
                break
    return hits


def probe_codec(ctx: StageContext, sample: Path | None) -> dict[str, Any] | None:
    if sample is None or not sample.exists():
        return None
    try:
        pr = ffmpeg.probe(ctx.tools, sample)
    except Exception as e:  # tool missing or unreadable sample
        ctx.log.warning("codec probe failed on %s: %s", sample, e)
        return None
    container = {"ogg": "ogg", "wav": "wav", "mp3": "mp3", "flac": "flac", "mov": "m4a"}.get(pr.container, pr.container or sample.suffix.lstrip("."))
    return {"container": container, "codec": pr.codec, "sr": pr.sr, "ch": pr.ch, "bitrate_k": pr.bitrate_k or 96}


def run(ctx: StageContext) -> dict[str, Any]:
    game = ctx.project_dir
    if not game.exists():
        raise StageError(f"game dir not found: {game}")
    forced = ctx.extra.get("engine") or (ctx.cfg.parse.engine if ctx.cfg.parse.engine != "auto" else None)
    if forced and forced not in ENGINES:
        raise StageError(f"unknown engine {forced!r}; known: {', '.join(ENGINES)}")
    results: list[DetectResult] = []
    engines = list(all_engines())
    for i, eng in enumerate(engines, 1):
        try:
            r = eng.detect(game)
        except Exception as e:  # a broken detector must not stop detection
            ctx.log.warning("detector %s failed: %s", eng.name, e)
            r = None
        if r is not None:
            results.append(r)
        ctx.progress(i, len(engines), eng.name)
    results.sort(key=lambda r: -r.confidence)
    chosen: DetectResult | None = None
    if forced:
        chosen = next((r for r in results if r.engine == forced), None) or DetectResult(forced, 0.0, notes=["forced by --engine"])
    elif results:
        chosen = results[0]
    if chosen is None:
        chosen = DetectResult("generic", 0.0, notes=["no engine signature found; using generic (VNTextPatch import + prefix voices)"])
    warnings: list[str] = [n for n in chosen.notes if "encrypt" in n or "EXPERIMENTAL" in n or "unreadable" in n]
    if chosen.encrypted:
        warnings.append("archives look encrypted: extraction needs a runtime dump (see docs/ENGINES.md for this engine)")
    drm = find_drm(game)
    for d in drm:
        warnings.append(f"DRM marker: {d} (injection dumpers may not work; use an unprotected copy you own)")
    codec = chosen.voice_codec or probe_codec(ctx, chosen.voice_sample)
    proj = ProjectYaml(
        game_dir=str(game.resolve()),
        engine=chosen.engine,
        engine_version=chosen.version,
        src_lang=ctx.cfg.project.src_lang,
        tgt_lang=ctx.cfg.project.tgt_lang,
        voice_codec=VoiceCodec(**codec) if codec else VoiceCodec(),
        encrypted=chosen.encrypted,
        drm=drm,
        archives=chosen.archives,
        warnings=warnings,
        notes="",
    )
    proj.voice_codec_detected = codec is not None  # type: ignore[attr-defined]
    proj.detect_notes = list(chosen.notes)  # type: ignore[attr-defined]
    for n in chosen.notes:
        ctx.log.info(n)
    if ctx.paths.project_yaml.exists() and not ctx.force:
        old = load_project_yaml(ctx.paths.project_yaml)
        if Path(old.game_dir).resolve() == game.resolve():
            for f in KEEP_FIELDS:
                if f == "voice_codec":
                    if not codec:
                        proj.voice_codec = old.voice_codec
                        proj.voice_codec_detected = bool(getattr(old, "voice_codec_detected", False))  # type: ignore[attr-defined]
                else:
                    setattr(proj, f, getattr(old, f))
    save_project_yaml(ctx.paths.project_yaml, proj)
    for w in warnings:
        ctx.log.warning(w)
    return {
        "engine": chosen.engine,
        "confidence": round(chosen.confidence, 2),
        "version": chosen.version,
        "alternatives": [f"{r.engine}:{r.confidence:.2f}" for r in results if r.engine != chosen.engine][:4],
        "encrypted": chosen.encrypted,
        "drm": drm,
        "voice_codec": f"{proj.voice_codec.container}/{proj.voice_codec.codec} {proj.voice_codec.sr}Hz {proj.voice_codec.ch}ch" if codec else "unknown (probed after extract)",
        "warnings": warnings[:4],
        "notes": chosen.notes[:4],
        "project_yaml": str(ctx.paths.project_yaml),
    }

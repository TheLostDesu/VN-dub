"""Step 1 summary: what the import found (engine, voices, characters, mapping, problems)."""

from __future__ import annotations

from typing import Any

from vndub_app import appdirs, casting, session, viewdata

# engine id -> proper name shown as is (generic has no name: the UI uses i18n ``engines.generic``)
ENGINE_LABELS = {
    "renpy": "Ren'Py", "kirikiri": "KiriKiri / KAG", "nscripter": "NScripter", "siglus": "SiglusEngine",
    "bgi": "BGI / Ethornell", "catsystem2": "CatSystem2", "artemis": "Artemis", "yuris": "YU-RIS",
    "majiro": "Majiro", "unity": "Unity", "tyrano": "TyranoScript",
}

PROBLEM_FILES_MAX = 500


def warning_codes(proj: Any) -> list[str]:
    """Machine-readable warning codes of a project (i18n ``summary.warning.<code>``; ``drm:<kind>``)."""
    codes: list[str] = []
    if proj.encrypted:
        codes.append("encrypted_archives")
    for d in proj.drm or []:
        codes.append(f"drm:{d}")
    for w in proj.warnings or []:
        wl = w.lower()
        if "experimental" in wl:
            codes.append("experimental")
        elif "unreadable" in wl:
            codes.append("unreadable")
        elif "encrypt" in wl or "drm marker" in wl:
            continue  # already covered above
        else:
            codes.append("other")
    return list(dict.fromkeys(codes))


def engine_tools(engine: str, proj: Any) -> list[str]:
    """Tools an engine needs to extract/pack (besides ffmpeg), given what ``detect`` saw."""
    archives = bool(getattr(proj, "archives", None))
    table = {
        "renpy": ["unrpa"] if archives else [],
        "kirikiri": ["krkrdump"] if getattr(proj, "encrypted", False) else [],
        "siglus": ["siglus_ssu"], "bgi": ["bgi_extractor"], "catsystem2": ["catsystem2_cli"],
        "artemis": ["pfs_unpacker"], "yuris": ["yrt"], "majiro": ["majiro_tools"], "unity": ["assetripper"],
    }
    return table.get(engine, [])


MANUAL_KEYS = {"krkrdump": "krkrdump"}


def tools_missing(ctx: Any, proj: Any | None) -> list[dict[str, Any]]:
    from vndub.tools import TOOL_INFO

    engine = proj.engine if proj is not None else None
    need = [n for n, i in TOOL_INFO.items() if i.required]
    if proj is not None:
        need += engine_tools(engine or "", proj)
    out = []
    for name in dict.fromkeys(need):
        info = TOOL_INFO.get(name)
        if info is None or ctx.tools.resolve(name) is not None:
            continue
        row: dict[str, Any] = {"tool": name, "url": info.url}
        if info.pip:
            row["pip"] = info.pip
        if name in MANUAL_KEYS:
            row["manual_key"] = MANUAL_KEYS[name]
        out.append(row)
    return out


def _audio_minutes(ctx: Any, snap: Any) -> dict[str, float]:
    """Seconds of original audio per speaker from the audio index (when characters.yaml has no stats yet)."""
    from vndub.manifest.io import AudioIndexRecord, read_jsonl

    idx = {a.id: a for a in read_jsonl(ctx.paths.audio_index, AudioIndexRecord)} if ctx.paths.audio_index.exists() else {}
    sec: dict[str, float] = {}
    for l in snap.voiced:
        a = idx.get(l.audio_id or "")
        if a is not None:
            sec[l.speaker_id] = sec.get(l.speaker_id, 0.0) + float(a.duration or 0.0)
    return sec


def no_text_count(h: appdirs.ProjectHandle) -> int:
    return len(no_text_lines(h))


def no_text_lines(h: appdirs.ProjectHandle) -> list[Any]:
    """Voiced lines without script text (mode C): same definition as ``link.no_text_lines``."""
    snap = viewdata.snapshot(h)
    return [l for l in snap.voiced if "voice_sustain" not in (l.flags or []) and not (l.text_src or "").strip()]


def _link_counts(ctx: Any) -> dict[str, int]:
    from vndub.manifest.io import LinkRecord, read_jsonl

    c = {"ok": 0, "review": 0, "unmatched": 0}
    if ctx.paths.link.exists():
        for r in read_jsonl(ctx.paths.link, LinkRecord):
            c[r.status] = c.get(r.status, 0) + 1
    return c


def build(h: appdirs.ProjectHandle) -> dict[str, Any]:
    from vndub.manifest.io import read_jsonl

    ctx = session.read_ctx(h)
    proj = ctx.project
    status = viewdata.stage_status(ctx)
    snap = viewdata.snapshot(h, ctx)
    chars = viewdata.load_characters(ctx)

    engine = {
        "id": proj.engine if proj else None,
        "label": ENGINE_LABELS.get(proj.engine) if proj else None,
        "version": (proj.engine_version if proj else "") or "",
        "encrypted": bool(proj.encrypted) if proj else False,
        "drm": list(proj.drm) if proj else [],
        "warnings": warning_codes(proj) if proj else [],
    }

    counts = casting.line_counts(snap)
    sec_by_speaker: dict[str, float] | None = None
    char_rows = []
    for sid in sorted(set(chars) | set(counts), key=lambda s: (-counts.get(s, (0, 0))[0], s)):
        if sid == viewdata.NARRATOR:
            continue
        n, v = counts.get(sid, (0, 0))
        c = chars.get(sid)
        secs = float(c.stats.total_sec) if c is not None and c.stats.total_sec else 0.0
        if not secs and v and snap.voiced:
            sec_by_speaker = sec_by_speaker if sec_by_speaker is not None else _audio_minutes(ctx, snap)
            secs = sec_by_speaker.get(sid, 0.0)
        char_rows.append({
            "speaker_id": sid, "name_src": (c.names.get("src") if c is not None else "") or sid, "name_tgt": (c.names.get("tgt") if c is not None else "") or "",
            "lines": n, "voiced_lines": v, "unvoiced_lines": n - v, "minutes": round(secs / 60.0, 2),
            "mode": "voiced" if v and v >= n else ("unvoiced" if not v else "mixed"),
            # contract C10: the SPA marks a gender the translation model guessed ("guess") or the user set ("user")
            "gender": (c.gender if c is not None else "unknown") or "unknown",
            "gender_source": ((c.model_extra or {}).get("gender_source") if c is not None else None) or None,
        })

    link_s = viewdata.stage_summary(status, "link")
    lc = {k: int(link_s[k]) for k in ("ok", "review", "unmatched") if k in link_s} or _link_counts(ctx)
    judged = sum(lc.values())
    conf = link_s.get("confidence")
    if conf is None and judged:
        conf = round(lc.get("ok", 0) / judged, 3)
    mapping = {
        "mode": link_s.get("mode") or ctx.cfg.link.mode, "ok": lc.get("ok", 0), "review": lc.get("review", 0),
        "unmatched": lc.get("unmatched", 0), "confidence": conf, "suggest_asr_check": bool(link_s.get("suggest_asr_check", False)),
        "outliers": int(link_s.get("outliers", 0) or 0),
    }

    no_text = no_text_lines(h)
    problems = []
    if ctx.paths.review_queue.exists():
        for r in read_jsonl(ctx.paths.review_queue)[:PROBLEM_FILES_MAX]:
            lid = r.get("line_id")
            line = snap.by_id.get(lid)
            problems.append({"line_id": lid, "status": r.get("status", "review"), "reasons": list(r.get("reasons") or []),
                         "voice_file": line.voice_file if line else None, "signals": r.get("signals")})

    extract_s = viewdata.stage_summary(status, "extract")
    de = extract_s.get("decode_errors")  # {"count", "samples", "missing_tools"} (absent before round 18)
    de = de if isinstance(de, dict) else {"count": de} if isinstance(de, int) else {}
    try:
        decode_errors = int(de.get("count") or 0)
    except (TypeError, ValueError):
        decode_errors = 0
    if decode_errors:  # voice files the extract stage could not decode: those lines have no original voice
        engine["warnings"] = [*engine["warnings"], "decode_errors"]
    voice_files = extract_s.get("voice_files")
    if voice_files is None:
        voice_files = len(read_jsonl(ctx.paths.audio_index)) if ctx.paths.audio_index.exists() else 0
    analyze_s = viewdata.stage_summary(status, "analyze")
    text_s = viewdata.stage_summary(status, "text_emotion")
    audio_pp = analyze_s.get("lines")
    if audio_pp is None:
        audio_pp = sum(1 for f in ctx.paths.analysis.glob("*.json")) if ctx.paths.analysis.exists() else 0
    text_pp = text_s.get("lines", text_s.get("written", 0)) or 0

    from vndub_app.translation import source_language

    src = source_language(h)
    proj = ctx.project  # source_language may have written the detected code into project.yaml
    languages = {
        "src": src["effective"] or None,
        "src_detected": bool(src["detected"]),
        "audio": ((proj.audio_lang if proj else "") or ctx.cfg.project.audio_lang or src["effective"]) or None,
        "tgt": (proj.tgt_lang if proj else "") or ctx.cfg.project.tgt_lang,
    }

    return {
        "engine": engine,
        "languages": languages,
        "tools_missing": tools_missing(ctx, proj),
        "voice_files": int(voice_files or 0),
        "decode_errors": {"count": decode_errors, "samples": [str(x) for x in de.get("samples") or []][:3], "missing_tools": [str(x) for x in de.get("missing_tools") or []]},
        "lines_total": len(snap.lines),
        "voiced": len(snap.voiced),
        "unvoiced": len(snap.lines) - len(snap.voiced),
        "characters": char_rows,
        "mapping": mapping,
        "no_text_lines": len(no_text),
        "recover_offer": bool(no_text) or (len(snap.lines) == 0 and int(voice_files or 0) > 0),  # mode C / a voice-only game
        "no_text_sample": [{"line_id": l.line_id, "voice_file": l.voice_file} for l in no_text[:20]],
        "text_from_asr_lines": sum(1 for l in snap.lines if "text_from_asr" in (l.flags or [])),
        "problem_files": problems,
        "analysis": {"audio_passports": int(audio_pp or 0), "text_passports": int(text_pp)},
    }


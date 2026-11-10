"""parse: scripts -> ``manifest/lines.jsonl`` + ``manifest/characters.yaml``.

The engine plugin parses its scripts and binds voices through the audio index
(``VoiceResolver``). ``--import-vntextpatch DIR`` (or ``parse.import_vntextpatch``)
switches any engine to the generic VNTextPatch import with prefix voices.
Existing ``characters.yaml`` user edits (target names, gender, age, route,
notes, performer/rvc/lora) are kept. Audio files no line refers to are
listed in ``manifest/unbound_audio.jsonl`` (link can search them globally).

Mode C (``link.recover_text``): when the script yields no lines at all (unsupported or
encrypted script, generic engine without a text import) but voice files exist, parse still
writes one voiced line with empty text per file (``flags: [voice_only]``) so ``link`` can
recover the text from the audio. With the option off the engine's own error is raised as before.

Voiced lines the script shows without a speaker name (``parse.hidden_speakers: auto``): an off-screen
voice in the narrator box, or a name such as ``？？？``, goes to the character its voice-file prefix
names (flag ``speaker_from_voice``); narrator-box lines no prefix identifies go to the ``voice_over``
speaker (flag ``offscreen_voice``), which is cast and dubbed like any character. Unvoiced narration is
never touched. See :func:`vndub.engines.base.resolve_hidden_speakers`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vndub.engines.base import VOICE_OVER, VOICE_OVER_NAMES, ExtractResult, ParseResult, build_characters, resolve_hidden_speakers
from vndub.engines.generic_prefix import import_vntextpatch, textless_result
from vndub.manifest.io import AudioIndexRecord, Character, load_characters, read_jsonl, save_characters, write_jsonl
from vndub.stages.common import StageContext, StageError, resolve_src_lang
from vndub.text.lang import detect_language, language_name

NAME = "parse"
VERSION = "5"  # 5: Ren'Py voice kept over show / play / $, voice_stmt, narration characters, init-block characters, {p} as a space, no music as voice; 4: Ren'Py ids follow the translate blocks exactly (voice / nvl clear statements, unicode names, scripts from archives); 3: ids computed as Ren'Py does. The app tells a project imported with an older version to import again
CONFIG_SECTIONS = ("parse",)

# gender_source ("user" | "guess", an extra): who set the gender, so a guess stays marked as one after a re-import
USER_FIELDS = ("gender", "gender_source", "age_hint", "route", "notes", "performer", "rvc", "lora", "refs", "description")


def merge_characters(new: dict[str, Character], old: dict[str, Character]) -> dict[str, Character]:
    out: dict[str, Character] = {}
    for sid, c in new.items():
        prev = old.get(sid)
        if prev is not None:
            if prev.names.get("tgt"):
                c.names["tgt"] = prev.names["tgt"]
            for f in USER_FIELDS:
                v = getattr(prev, f, None)
                if v not in (None, "", [], {}, "unknown"):
                    setattr(c, f, v)
            if prev.voice_prefix:
                c.voice_prefix = prev.voice_prefix
        out[sid] = c
    for sid, c in old.items():  # keep characters the user added by hand
        out.setdefault(sid, c)
    return out


def run(ctx: StageContext) -> dict[str, Any]:
    proj = ctx.require_project()
    ex_path = ctx.paths.raw / "extract.json"
    if not ex_path.exists():
        raise StageError(f"{ex_path} not found: run `vndub extract` first")
    extracted = ExtractResult.load(ex_path)
    audio = [a for a in read_jsonl(ctx.paths.audio_index, AudioIndexRecord) if not a.error] if ctx.paths.audio_index.exists() else []
    imp = ctx.extra.get("import_vntextpatch") or ctx.cfg.parse.import_vntextpatch
    if imp:
        result: ParseResult = import_vntextpatch(ctx, Path(imp), audio)
        source = f"vntextpatch:{imp}"
    else:
        try:
            result = ctx.engine().parse(ctx, extracted, audio)
        except StageError as e:
            if not (ctx.cfg.link.recover_text and audio):
                raise
            ctx.log.warning("script not parsed (%s): voice-only lines for text recovery", e)
            result = textless_result(ctx, audio, str(e)[:120])
        source = proj.engine
    if ctx.cfg.link.recover_text and audio and not result.lines:
        result = textless_result(ctx, audio, "the script gave no lines")
        source = f"{source}+voice_only"
    lines = result.lines
    seen: set[str] = set()
    n_units = max(1, 2 * len(lines))
    for i, l in enumerate(lines, 1):
        if l.line_id in seen:
            raise StageError(f"duplicate line_id {l.line_id} from the {source} parser")
        seen.add(l.line_id)
        if i % 100 == 0:
            ctx.progress(i, n_units, l.line_id)
    hidden = {"from_voice": 0, "voice_over": 0}
    if ctx.cfg.parse.hidden_speakers == "auto":
        before = {l.line_id: l.speaker_id for l in lines}
        hidden = resolve_hidden_speakers(lines)
    old = load_characters(ctx.paths.characters) if ctx.paths.characters.exists() else {}
    if any(hidden.values()):
        result.characters = _hidden_characters(lines, before, result.characters, ctx.cfg.project.tgt_lang)
    chars = merge_characters(result.characters, old)
    if any(hidden.values()):
        live = {l.speaker_id for l in lines if l.has_voice or l.speaker_id != "narrator"}
        for sid in set(before.values()) - live:  # the narrator / a ？？？ speaker whose voiced lines all moved
            chars.pop(sid, None)
    write_jsonl(ctx.paths.lines, lines)
    save_characters(ctx.paths.characters, chars)
    # the language of the text: detected over the new lines when project.yaml says auto (or holds an earlier
    # detection); a code the user set is kept, with a warning when the text clearly disagrees
    src_lang = resolve_src_lang(ctx, lines, redetect=True)
    lang_detected = bool(getattr(ctx.project, "src_lang_detected", None))
    if not lang_detected:
        guess = detect_language(l.text_src for l in lines)
        if guess.lang and guess.lang != src_lang and guess.confidence >= 0.8:
            ctx.log.warning(
                "project.src_lang is %s but the script text looks like %s (%s, confidence %.2f): set project.src_lang: auto to detect it",
                src_lang, guess.lang, language_name(guess.lang), guess.confidence,
            )
    bound = {l.audio_id for l in lines if l.audio_id}
    unbound = [a for a in audio if a.id not in bound]
    write_jsonl(ctx.paths.manifest_dir / "unbound_audio.jsonl", [{"id": a.id, "relpath": a.relpath, "duration": a.duration} for a in unbound])
    for n in result.notes:
        ctx.log.info(n)
    voiced = [l for l in lines if l.has_voice]
    missing = [l for l in lines if "voice_missing" in l.flags]
    speakers: dict[str, int] = {}
    for i, l in enumerate(voiced, 1):
        speakers[l.speaker_id] = speakers.get(l.speaker_id, 0) + 1
        if i % 100 == 0:
            ctx.progress(len(lines) + i, n_units, l.line_id)
    ctx.progress(n_units, n_units, None)
    coverage = len(bound) / len(audio) if audio else 0.0
    return {
        "source": source,
        "lines": len(lines),
        "voiced": len(voiced),
        "unvoiced": len(lines) - len(voiced),
        "audio_files": len(audio),
        "bound_audio": len(bound),
        "voice_coverage": round(coverage, 3),
        "unbound_audio": len(unbound),
        "voice_missing_refs": len(missing),
        "sustain_lines": sum(1 for l in lines if "voice_sustain" in l.flags),
        "speakers": speakers,
        "characters": len(chars),
        "hidden_speakers": hidden,
        "src_lang": src_lang,
        "src_lang_detected": lang_detected,
        "notes": result.notes[:4],
    }


def _hidden_characters(lines: list, before: dict[str, str], chars: dict[str, Character], tgt_lang: str) -> dict[str, Character]:
    """Characters after :func:`resolve_hidden_speakers`: counts follow the moved lines, ``voice_over`` exists."""
    out = dict(chars)
    for sid in set(before.values()) - {l.speaker_id for l in lines}:
        out.pop(sid, None)
    if not any(l.has_voice and l.speaker_id == "narrator" for l in lines):
        out.pop("narrator", None)
    for c in out.values():
        c.stats.n_lines = 0
    out = build_characters(lines, existing=out)
    vo = out.get(VOICE_OVER)
    if vo is not None and vo.names.get("src") == VOICE_OVER:
        vo.names = {"src": "", "tgt": VOICE_OVER_NAMES.get(tgt_lang, "")}
    return out

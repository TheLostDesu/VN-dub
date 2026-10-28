"""pack: final voice files -> the engine's patch layout in ``out/<engine>/``.

Always writes ``INSTALL.md`` and ``manifest_patch.json`` (file, sha256,
line_id, backend, route, status) next to the patch.

Both say what the patch covers (:func:`patch_coverage`): how many of the dubbed lines have a file, how many of those
are ``provisional`` (no take passed the quality gates, the best one is installed: ``master.include_review_lines``),
how many were LEFT OUT and why (a line without a file is silent in the game), how many narration lines are never
dubbed, and (Ren'Py) how many lines are not installable because they have a ``voice`` statement of their own.

A full pack (not ``--only``) also removes what an earlier pack left in ``out/<engine>`` that this one does not hold
(:func:`_remove_stale`): the installer ships the folder whole, so a file the manifest does not list would be installed anyway.

Lines that had no original voice (``synth.voice_unvoiced``; ``FinalIndexRecord.unvoiced``) have no
file name in the game to take over, so they are placed apart from the engine's normal layout:

* Ren'Py: a voice file named by the line's dialogue identifier (``<dir>/<id>.<ext>``) and, unless the game
  already defines ``config.auto_voice`` (then its pattern decides the directory and nothing else is needed),
  a generated ``game/vndub_auto_voice.rpy`` that enables ``config.auto_voice`` for that directory. Ren'Py plays
  such a file for a say statement that has no ``voice`` statement of its own. Narration is not dubbed.
* every other engine: ``out/<engine>/unvoiced/<file>`` and a note in INSTALL.md. The patch cannot put them
  into the game (there is no voice slot to replace), and nothing here claims that it does.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

from vndub.engines.base import BaseEngine, write_install
from vndub.manifest.io import FinalIndexRecord, SelectionRecord, read_json, read_jsonl, write_json
from vndub.stages.common import StageContext, StageError
from vndub.util import hash_file

NAME = "pack"
VERSION = "1"
CONFIG_SECTIONS = ("pack",)

RENPY_RPY = "vndub_auto_voice.rpy"
RENPY_DEFAULT_DIR = "voice_dub"
LEFT_OUT_LISTED = 300  # lines named one by one in manifest_patch.json (the counts are always complete)
AUDIO_SUFFIXES = {".ogg", ".wav", ".mp3", ".opus", ".flac", ".m4a", ".aac", ".mp2"}
# the folder under out/<engine>/ that an engine copies <voice_file> into ("" = out/<engine> itself). An engine that builds an
# archive or a script (kirikiri, unity, catsystem2) writes a new one every time and is not listed
_LOOSE_ROOT = {"renpy": "game", "nscripter": "", "tyrano": "data", "siglus": "koe_dump", "artemis": "files"}


def _is_unvoiced(f: FinalIndexRecord) -> bool:
    return bool((f.model_extra or {}).get("unvoiced"))


# ----------------------------------------------------------------------------
# what the patch covers
# ----------------------------------------------------------------------------


def _left_out_why(s: SelectionRecord | None) -> tuple[str, str]:
    """(category, reason) of a dubbed line that has no file in the patch."""
    if s is None:
        return "not scored", "no selection record: score has not run for this line"
    if not s.winner:
        return "no winner", s.reason or "no take could be made"
    if s.status in ("review", "regen"):
        return s.status, s.reason
    return "not mastered", "mastering failed or its file is missing (see the master log)"


def patch_coverage(ctx: StageContext, finals: list[FinalIndexRecord], sel: dict[str, SelectionRecord] | None = None) -> dict[str, Any]:
    """What a patch built from ``finals`` (the records whose file exists) holds and what it leaves out.

    * ``dub_lines``: lines the pipeline dubs under this config (``ctx.dub_lines``); without a lines manifest, the files
    * ``installed``: lines with a file in the patch; ``provisional`` of them were mastered from a take that did not
      pass the gates (``provisional_review`` / ``provisional_regen`` by the selection status they had)
    * ``left_out``: dubbed lines without a file (silent in the game) with ``left_out_reasons`` by category and
      ``left_out_lines`` (the first :data:`LEFT_OUT_LISTED`) as ``{line_id, status, reason}``
    * ``narration``: narration lines without original voice, which are never dubbed
    * ``own_voice`` (Ren'Py): speaker lines that are NOT INSTALLABLE because they have a ``voice`` statement of their own
      whose file is missing or computed at run time: ``config.auto_voice`` is never asked for them, so they are not dubbed;
      ``own_voice_lines`` names the first :data:`LEFT_OUT_LISTED`
    """
    if sel is None:
        sel = {s.line_id: s for s in read_jsonl(ctx.paths.selection, SelectionRecord)} if ctx.paths.selection.exists() else {}
    have = {f.line_id: f for f in finals}
    dub: list[Any] = []
    others: list[Any] = []
    own: list[Any] = []
    if ctx.paths.lines.exists():
        try:
            dub = ctx.dub_lines(apply_only=False)
            dub_ids = {l.line_id for l in dub}
            others = [l for l in ctx.lines() if l.line_id not in dub_ids]
            own = ctx.own_voice_lines()
        except Exception as e:  # a half-built project must not stop the pack
            ctx.log.warning("patch coverage: cannot read the lines (%s)", e)
    n_prov = {"review": 0, "regen": 0}
    for f in finals:
        if f.status == "provisional":
            st = str((f.model_extra or {}).get("selection_status") or (sel[f.line_id].status if f.line_id in sel else "review"))
            n_prov[st if st in n_prov else "review"] += 1
    left: list[dict[str, str]] = []
    reasons: dict[str, int] = {}
    for l in dub:
        if l.line_id in have:
            continue
        cat, why = _left_out_why(sel.get(l.line_id))
        reasons[cat] = reasons.get(cat, 0) + 1
        left.append({"line_id": l.line_id, "status": cat, "reason": why[:300]})
    narration = sum(1 for l in others if l.speaker_id == "narrator" and (l.text_src or "").strip() and not ({"menu", "voice_sustain"} & set(l.flags)))
    return {
        "dub_lines": len(dub) if dub else len(have),
        "installed": len(have),
        "provisional": n_prov["review"] + n_prov["regen"],
        "provisional_review": n_prov["review"],
        "provisional_regen": n_prov["regen"],
        "left_out": len(left),
        "left_out_reasons": reasons,
        "left_out_lines": left[:LEFT_OUT_LISTED],
        "narration": narration,
        "own_voice": len(own),
        "own_voice_lines": [l.line_id for l in own[:LEFT_OUT_LISTED]],
        "include_review_lines": ctx.cfg.master.include_review_lines,
    }


def _coverage_notes(cov: dict[str, Any]) -> list[str]:
    """INSTALL.md section for :func:`patch_coverage`."""
    n_ok = cov["installed"] - cov["provisional"]
    notes = ["", "## What this patch covers", ""]
    notes.append(f"- {cov['dub_lines']} line(s) are dubbed; {cov['installed']} have a voice file in this patch ({n_ok} with a take that passed the quality checks).")
    if cov["provisional"]:
        notes.append(
            f"- {cov['provisional']} line(s) are PROVISIONAL: no take of them passed every quality check, so the best take is installed instead of "
            "leaving the line silent. Listen to them in Review (status \"check\"), replace the ones that sound wrong and pack again. "
            "They carry `\"status\": \"provisional\"` and the reason in `manifest_patch.json`."
        )
    if cov["left_out"]:
        why = ", ".join(f"{k}: {v}" for k, v in sorted(cov["left_out_reasons"].items()))
        notes.append(
            f"- {cov['left_out']} line(s) are LEFT OUT and stay silent in the game ({why}). "
            + (
                "`master.include_review_lines` is `skip`: set it to `provisional` to install the best take of such lines. "
                if cov.get("include_review_lines") == "skip" and (cov["left_out_reasons"].get("review") or cov["left_out_reasons"].get("regen"))
                else ""
            )
            + "Each line is listed with its reason under `coverage.left_out_lines` in `manifest_patch.json`."
        )
    if cov["narration"]:
        notes.append(f"- {cov['narration']} narration line(s) (no speaker) are never dubbed and stay silent in the game.")
    if cov.get("own_voice"):
        notes.append(
            f"- {cov['own_voice']} line(s) are NOT INSTALLABLE: own voice statement. The script gives each of them a `voice` statement whose file is not in the game "
            "(or whose name is computed while the game runs), and Ren'Py never asks `config.auto_voice` for a line with a voice statement, so a dub could not play. "
            "They are not dubbed and stay as they are in the game; they are listed under `coverage.own_voice_lines` in `manifest_patch.json`."
        )
    return notes


# ----------------------------------------------------------------------------
# Ren'Py: lines without original voice
# ----------------------------------------------------------------------------


def renpy_game_auto_voice(ctx: StageContext) -> str | None:
    """The game's own ``config.auto_voice`` pattern (``pack.renpy.auto_voice`` / ``parse...auto_voice`` override,
    else read from the game's scripts); ``None`` when it defines none."""
    override = str(ctx.cfg.pack.renpy.get("auto_voice") or ctx.cfg.parse.engine_opts.renpy.auto_voice or "")
    if override:
        return override
    from vndub.engines.base import read_text_any
    from vndub.engines.renpy import AUTO_VOICE, _literal

    roots = [ctx.game_dir / "game", ctx.paths.raw / "renpy"]
    for root in roots:
        if not root.exists():
            continue
        for rpy in root.rglob("*.rpy"):
            try:
                text = read_text_any(rpy)
            except OSError:
                continue
            if "auto_voice" not in text:
                continue
            for line in text.splitlines():
                m = AUTO_VOICE.search(line.strip())
                if m and not line.lstrip().startswith("#"):
                    val = _literal(m.group("val"))
                    if val:
                        return val
    return None


def _usable_pattern(pattern: str | None) -> bool:
    return bool(pattern) and pattern.count("{id}") == 1 and not re.search(r"\{(?!id\})", pattern)  # type: ignore[arg-type]


def renpy_auto_voice_rpy(directory: str, ext: str) -> str:
    """Text of the generated ``game/vndub_auto_voice.rpy`` (valid Ren'Py: one ``init python`` block)."""
    return (
        "# Generated by vndub: voices for lines that had no original voice.\n"
        "# Ren'Py plays <dialogue identifier>" + ext + " from game/" + directory + "/ for a say statement without its own voice statement.\n"
        "init 1000 python:\n"
        "    # a game that never had voice may have switched the voice channel off (config.has_voice = False): the patch needs it\n"
        "    config.has_voice = True\n"
        f'    config.auto_voice = "{directory}/{{id}}{ext}"\n'
    )


def _pack_unvoiced_renpy(ctx: StageContext, unvoiced: list[FinalIndexRecord]) -> tuple[dict[str, str], dict[str, Any]]:
    """Copy the unvoiced files named by dialogue identifier; returns ({line_id: path relative to game/}, info)."""
    out = ctx.paths.out_engine("renpy") / "game"
    pattern = renpy_game_auto_voice(ctx)
    own_pattern = _usable_pattern(pattern)
    placed: dict[str, str] = {}
    by_id = {l.line_id: l for l in ctx.lines()}
    unnamed: list[str] = []
    owner: dict[str, str] = {}  # destination -> the line that owns it
    clashes: list[str] = []
    for f in unvoiced:
        src = ctx.paths.abs(f.encoded)
        name = PurePosixPath(f.voice_file).name  # <dialogue id | line id>.<ext>
        stem, ext = PurePosixPath(name).stem, PurePosixPath(name).suffix
        line = by_id.get(f.line_id)
        if not str(((line.model_extra or {}).get("dialogue_id") if line is not None else "") or ""):
            unnamed.append(f.line_id)  # named by line id: Ren'Py cannot find it by dialogue identifier
        rel = pattern.replace("{id}", stem) if own_pattern else f"{ctx.cfg.pack.renpy.get('unvoiced_dir') or RENPY_DEFAULT_DIR}/{stem}{ext}"
        if rel in owner:  # two lines with one dialogue identifier: Ren'Py plays one file for both
            clashes.append(f.line_id)
            ctx.log.warning("pack: lines %s and %s have the same dialogue identifier (%s): one voice file serves both", owner[rel], f.line_id, stem)
        owner.setdefault(rel, f.line_id)
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        placed[f.line_id] = f"game/{rel}"
    info: dict[str, Any] = {"auto_voice": pattern if own_pattern else None, "unnamed": len(unnamed), "unnamed_ids": unnamed, "clashes": len(clashes)}
    if not own_pattern:
        d = str(ctx.cfg.pack.renpy.get("unvoiced_dir") or RENPY_DEFAULT_DIR).strip("/")
        ext = PurePosixPath(next(iter(placed.values()))).suffix if placed else ".ogg"
        rpy = out / RENPY_RPY
        rpy.write_text(renpy_auto_voice_rpy(d, ext), encoding="utf-8")
        info["rpy"] = f"game/{RENPY_RPY}"
        info["dir"] = d
    return placed, info


def _loose_root(eng: Any, out: Path) -> Path | None:
    """The folder under ``out`` that ``eng.pack`` copies ``<voice_file>`` into, None for an engine that builds an archive or a script."""
    if eng.name in _LOOSE_ROOT:
        return out / _LOOSE_ROOT[eng.name]
    return out if type(eng).pack is BaseEngine.pack else None


def _remove_stale(ctx: StageContext, eng: Any, out: Path, regular: list[FinalIndexRecord], placed: dict[str, str], uinfo: dict[str, Any]) -> int:
    """After a full pack: delete what an earlier pack left in ``out`` and this one does not hold. The installer ships the whole folder,
    so a file the new ``manifest_patch.json`` does not list would be installed all the same (a rejected line, ``include_review_lines``
    switched to ``skip``, an import under another parse version that renamed the Ren'Py ids ...). Touched are only the files the previous
    ``manifest_patch.json`` listed, the audio files next to files of the patch (the engine's voice folders), and the ``vndub_auto_voice.rpy``
    this stage generated when it is no longer needed; nothing else. Returns the number of files removed."""
    root = _loose_root(eng, out)
    if root is None:
        return 0
    keep = {root / f.voice_file for f in regular} | {out / rel for rel in placed.values()}
    old: set[Path] = set()
    try:
        prev = read_json(out / "manifest_patch.json", {})
    except (OSError, ValueError):
        prev = {}
    for e in (prev.get("files") or []) if isinstance(prev, dict) else []:
        if not isinstance(e, dict):
            continue
        unvoiced = bool(e.get("unvoiced"))  # packed_as of those is relative to out/<engine>, the others' file is relative to the engine's root
        rel = e.get("packed_as") if unvoiced else e.get("file")
        if isinstance(rel, str) and rel:
            old.add((out if unvoiced else root) / rel)
    base = out.resolve()

    def inside(p: Path) -> bool:
        try:
            p.resolve().relative_to(base)
            return True
        except (ValueError, OSError):
            return False

    dirs = {p.parent for p in keep | old if inside(p)}
    if eng.name == "renpy":
        dirs.add(root / str(ctx.cfg.pack.renpy.get("unvoiced_dir") or RENPY_DEFAULT_DIR).strip("/"))
    stale = {p for p in old if p not in keep and inside(p)}
    for d in dirs:
        if d.is_dir():
            stale |= {p for p in d.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES and p not in keep}
    n = 0
    for p in sorted(stale):
        try:
            p.unlink()
            n += 1
        except FileNotFoundError:
            pass
        except OSError as e:
            ctx.log.warning("pack: cannot remove the stale file %s (%s)", p, e)
    if eng.name == "renpy" and "rpy" not in uinfo:  # the generated auto_voice script of an earlier pack
        rpy = root / RENPY_RPY
        try:
            if rpy.is_file() and rpy.read_text(encoding="utf-8").startswith("# Generated by vndub"):
                rpy.unlink()
                n += 1
        except OSError as e:
            ctx.log.warning("pack: cannot remove the stale %s (%s)", rpy, e)
    for d in sorted(dirs, key=lambda x: len(x.parts), reverse=True):  # folders the removal emptied
        try:
            if d not in (root, out) and d.is_dir() and not any(d.iterdir()):
                d.rmdir()
        except OSError:
            pass
    if n:
        ctx.log.info("pack: removed %d file(s) an earlier pack left in %s that this patch does not hold", n, out)
    return n


def _install_notes(engine: str, n: int, info: dict[str, Any]) -> list[str]:
    if engine == "renpy":
        if info.get("auto_voice"):
            how = (
                f"The game defines `config.auto_voice = \"{info['auto_voice']}\"`, so the {n} voice file(s) for lines that had no original voice "
                "are named by dialogue identifier under that pattern (inside `game/`): copying the `game` folder is enough."
            )
        else:
            how = (
                f"{n} line(s) had no original voice. Their files are `game/{info.get('dir')}/<dialogue identifier>.*` and `game/{RENPY_RPY}` "
                f"enables `config.auto_voice` for that folder: copy both with the rest of the `game` folder. Needs Ren'Py 7.4 or newer; "
                "a line that has its own `voice` statement keeps that voice."
            )
        notes = ["", "## Lines without original voice", "", how]
        if info.get("unnamed"):
            notes += ["", f"{info['unnamed']} of them have no Ren'Py dialogue identifier and cannot be found automatically (see manifest_patch.json, `renpy_auto_voice: false`)."]
        if info.get("clashes"):
            notes += ["", f"{info['clashes']} of them share a dialogue identifier with another line, so one file serves both."]
        return notes
    return [
        "",
        "## Lines without original voice (manual)",
        "",
        f"{n} line(s) had no original voice, so there is no voice file in the game for them to replace. Their audio is in `unvoiced/`, named by line id "
        "(`manifest_patch.json` lists them with `unvoiced: true`). This patch does NOT install them automatically: playing them needs a change to the game's script "
        "(a voice command before the line) or a tool of the engine; see docs/ENGINES.md.",
    ]


def run(ctx: StageContext) -> dict[str, Any]:
    proj = ctx.require_project()
    if not ctx.paths.final_index.exists():
        raise StageError(f"{ctx.paths.final_index} not found: run `vndub master` first")
    all_finals = read_jsonl(ctx.paths.final_index, FinalIndexRecord)
    on_disk = [f for f in all_finals if ctx.paths.abs(f.encoded).exists()]
    if len(on_disk) < len(all_finals):
        ctx.log.warning("%d final files listed in final_index.jsonl are missing on disk", len(all_finals) - len(on_disk))
    sel = {s.line_id: s for s in read_jsonl(ctx.paths.selection, SelectionRecord)} if ctx.paths.selection.exists() else {}
    cov = patch_coverage(ctx, on_disk, sel)  # of the whole project, also when --only limits what is copied now
    present = [f for f in on_disk if f.line_id in ctx.only] if ctx.only else on_disk
    regular = [f for f in present if not _is_unvoiced(f)]
    unvoiced = [f for f in present if _is_unvoiced(f)]
    eng = ctx.engine()
    out = ctx.paths.out_engine(eng.out_name)
    unvoiced_only = bool(unvoiced) and not regular
    res = {"mode": "unvoiced only", "files": 0, "out": str(out)} if unvoiced_only else eng.pack(ctx, regular)
    if unvoiced_only or not (out / "INSTALL.md").exists():
        # the engine's own pack rewrote INSTALL.md; when it did not run, rebuild it here, so that the sections
        # appended below never pile up over repeated packs
        if eng.name == "renpy":
            from vndub.engines.renpy import install_steps

            steps = install_steps(ctx)
        else:
            steps = ["Copy the files in this folder into the game directory, keeping relative paths."]
        write_install(out, eng.name, steps)
    with open(out / "INSTALL.md", "a", encoding="utf-8") as fh:
        fh.write("\n".join(_coverage_notes(cov)) + "\n")
    placed: dict[str, str] = {}
    uinfo: dict[str, Any] = {}
    if unvoiced:
        if eng.name == "renpy":
            placed, uinfo = _pack_unvoiced_renpy(ctx, unvoiced)
        else:
            for f in unvoiced:
                dst = out / f.voice_file
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ctx.paths.abs(f.encoded), dst)
                placed[f.line_id] = f.voice_file
        with open(out / "INSTALL.md", "a", encoding="utf-8") as fh:
            fh.write("\n".join(_install_notes(eng.name, len(unvoiced), uinfo)) + "\n")
    removed = 0 if ctx.only else _remove_stale(ctx, eng, out, regular, placed, uinfo)  # a partial pack (--only) leaves the rest of the folder alone
    entries = []
    for i, f in enumerate(present):
        ctx.progress(i, len(present), f.line_id)
        src = ctx.paths.abs(f.encoded)
        s = sel.get(f.line_id)
        entry = {
            "file": f.voice_file,
            "packed_as": PurePosixPath(f.voice_file).name if eng.name == "kirikiri" else f.voice_file,
            "sha256": hash_file(src),
            "line_id": f.line_id,
            "backend": (s.backend if s else None) or ("orig" if f.source == "orig" else None),
            "route": s.route if s else None,
            "status": f.status,
        }
        if f.status == "provisional":  # why a person should listen to it
            entry["reason"] = str((f.model_extra or {}).get("reason") or (s.reason if s else ""))
        if _is_unvoiced(f):
            entry["unvoiced"] = True
            entry["packed_as"] = placed.get(f.line_id, f.voice_file)
            if eng.name == "renpy":
                entry["renpy_auto_voice"] = f.line_id not in set(uinfo.get("unnamed_ids", []))
            else:
                entry["manual"] = True
        entries.append(entry)
    if present:
        ctx.progress(len(present), len(present), None)
    write_json(out / "manifest_patch.json", {"engine": eng.name, "game_dir": proj.game_dir, "coverage": cov, "files": entries})
    summary = {**res, "engine": eng.name, "manifest": str(out / "manifest_patch.json")}
    if unvoiced:
        summary["unvoiced"] = len(unvoiced)
        summary["unvoiced_install"] = "renpy auto_voice" if eng.name == "renpy" else "manual"
        if uinfo.get("unnamed"):
            summary["unnamed"] = uinfo["unnamed"]
    if cov["provisional"]:
        summary["provisional"] = cov["provisional"]
    if removed:
        summary["removed_stale"] = removed
    if cov["own_voice"]:
        summary["own_voice"] = cov["own_voice"]
        ctx.log.warning(
            "pack: %d line(s) have a voice statement of their own with no usable file; Ren'Py never asks config.auto_voice for them, so they are not dubbed", cov["own_voice"]
        )
    if cov["left_out"]:
        summary["left_out"] = cov["left_out"]
        ctx.log.warning("pack: %d dubbed line(s) have no file in the patch and stay silent in the game: %s", cov["left_out"], cov["left_out_reasons"])
    return summary

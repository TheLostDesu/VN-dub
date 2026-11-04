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

The voice track (Ren'Py only): the dub plays on the ``voice`` channel, which has a mixer of its own only when
``config.has_voice`` is set early enough, and the music is not lowered under it unless the player turned ducking on. So a
generated ``game/vndub_voice_mix.rpy`` (:func:`renpy_voice_mix_rpy`) switches the channel on, lowers the music and sound
effects while a line plays and adds a panel (voice volume, ducking strength; a hotkey and a button on the preferences
screen). ``pack.renpy.voice_mix`` configures it (``enabled: auto | true | false``, auto = on when the patch holds lines that
had no original voice; ``duck``; ``key``); ``manifest_patch.json`` carries ``voice_mix`` (``{rpy, key, keysym, duck,
enabled}`` or ``null``), INSTALL.md the section "The voice track in the game", the summary ``voice_mix: true | false``.
A full pack removes the script it generated earlier when the feature is now off.

The translated text (Ren'Py only): ``game/tl/<language>/vndub_dialogue.rpy`` (:func:`build_text_patch`, :func:`renpy_text_rpy`: a
``translate <language> <dialogue id>:`` block per translated say line and a ``strings`` block for menu choices and captions) and
``game/vndub_language.rpy`` (:func:`renpy_language_rpy`, ``config.language``). ``pack.renpy.text`` configures it (``enabled: auto |
true | false``, auto = the project has a translation file; ``language``; ``font``); ``manifest_patch.json`` carries ``text``
(``{rpy, language_rpy, language, lines, strings, skipped, ..., enabled}`` or ``null``), INSTALL.md the section "The translated text".
"""

from __future__ import annotations

import math
import re
import shutil
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from vndub.engines.base import BaseEngine, write_install
from vndub.manifest.io import FinalIndexRecord, SelectionRecord, TranslationRecord, read_json, read_jsonl, write_json
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


def _silent_sound_ids(ctx: StageContext, lines: list[Any]) -> set[str]:
    """Lines synth left silent on purpose (``synth.sounds.only: silent``, the line is only a sound): see ``synth.silent_sound_line``."""
    so = ctx.cfg.synth.sounds
    if not lines or not so.enabled or so.only != "silent":
        return set()
    try:
        from vndub.stages import genutil as gu
        from vndub.stages.synth import silent_sound_line

        trs = gu.load_translations(ctx)
        return {l.line_id for l in lines if silent_sound_line(ctx.cfg, trs.get(l.line_id), l)}
    except Exception as e:  # coverage is a report: it must not stop the pack
        ctx.log.warning("patch coverage: cannot tell which lines are only a sound (%s)", e)
        return set()


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
    silent_sounds = _silent_sound_ids(ctx, [l for l in dub if l.line_id not in have])
    for l in dub:
        if l.line_id in have:
            continue
        if l.line_id in silent_sounds:
            cat, why = "sound_silent", "the line is only a sound (snoring, a kiss ...) and synth.sounds.only is silent"
        else:
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
    n_snd = cov["left_out_reasons"].get("sound_silent", 0)
    if n_snd:
        notes.append(
            f"- {n_snd} line(s) are only a sound (snoring, a kiss ...) and were left silent: `synth.sounds.only` is `silent`. "
            "Set it to `spell` (a stand-in spelling) or `sample` (a recorded clip per sound in `synth.sounds.samples`) to voice them."
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


# ----------------------------------------------------------------------------
# Ren'Py: the voice track in the game (mixer, ducking, in-game panel)
# ----------------------------------------------------------------------------

RENPY_MIX_RPY = "vndub_voice_mix.rpy"
# pack.renpy.voice_mix: enabled = auto (on when the patch holds lines that had no original voice: the game was never
# mixed for voice) | true | false; duck = volume of the music and sound effects while a line plays (plain amplitude,
# 0.3 = about -10 dB, 1.0 = no ducking); key = the Ren'Py key that opens the panel in the game
VOICE_MIX_DEFAULTS: dict[str, Any] = {"enabled": "auto", "duck": 0.3, "key": "K_F8"}
_KEYSYM = re.compile(r"[A-Za-z0-9_]+")  # Ren'Py keysyms: K_F8, shift_K_F8, mousedown_4 ...

_MIX_STRINGS: dict[str, dict[str, str]] = {
    "ru": {
        "title": "Озвучка",
        "hint": "Голос идёт отдельной дорожкой: громкость музыки на него не влияет.",
        "voice_volume": "Громкость голоса",
        "duck": "Музыка и звуки во время реплик: [vndub_pct]%",
        "duck_on": "Приглушать музыку во время реплик",
        "mute": "Выключить озвучку",
        "test": "Проверить голос",
        "close": "Закрыть",
        "button": "Озвучка",
        "key_hint": "Клавиша {key} открывает эту панель в любой момент игры.",
    },
    "en": {
        "title": "Dub",
        "hint": "The voice plays on its own track: the music volume does not affect it.",
        "voice_volume": "Voice volume",
        "duck": "Music and sounds while a line plays: [vndub_pct]%",
        "duck_on": "Lower the music while a line plays",
        "mute": "Mute the dub",
        "test": "Test the voice",
        "close": "Close",
        "button": "Dub",
        "key_hint": "The {key} key opens this panel at any time in the game.",
    },
}


def voice_mix_opts(ctx: StageContext) -> dict[str, Any]:
    """``pack.renpy.voice_mix`` over :data:`VOICE_MIX_DEFAULTS` (a bare ``true`` / ``false`` is ``enabled``)."""
    raw = ctx.cfg.pack.renpy.get("voice_mix")
    opts = dict(VOICE_MIX_DEFAULTS)
    if isinstance(raw, dict):
        opts.update({k: v for k, v in raw.items() if k in opts and v is not None})
    elif isinstance(raw, (bool, str)):
        opts["enabled"] = raw
    key = str(opts["key"] or "").strip()
    if not _KEYSYM.fullmatch(key):  # empty or not a Ren'Py keysym: the key goes into a quoted string of the generated script
        key = VOICE_MIX_DEFAULTS["key"]
    if re.fullmatch(r"[fF]\d{1,2}", key):
        key = key.upper()  # "f9" -> "F9" (K_F9; the letter keys are lower case: K_a)
    opts["key"] = key if key.startswith("K_") or "_" in key else f"K_{key}"  # "F8" -> "K_F8"; "shift_K_F8" stays
    try:
        duck = float(opts["duck"])
    except (TypeError, ValueError):
        duck = float(VOICE_MIX_DEFAULTS["duck"])
    if not math.isfinite(duck):
        duck = float(VOICE_MIX_DEFAULTS["duck"])
    opts["duck"] = max(0.0, min(1.0, duck))
    return opts


def _mix_mode(enabled: Any) -> str:
    """``pack.renpy.voice_mix.enabled`` as ``on`` / ``off`` / ``auto`` (YAML gives a bool; a string is accepted too)."""
    if enabled is True:
        return "on"
    if enabled is False:
        return "off"
    v = str(enabled or "").strip().lower()
    if v in ("true", "on", "yes"):
        return "on"
    if v in ("false", "off", "no"):
        return "off"
    return "auto"


def voice_mix_key_name(keysym: str) -> str:
    """``K_F8`` -> ``F8`` (what the game shows and INSTALL.md says)."""
    return keysym[2:] if keysym.startswith("K_") else keysym


def _py_str(s: str) -> str:
    """``s`` as the inside of a double-quoted Python string literal (a file name in an action or an assignment)."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _rpy_text(s: str) -> str:
    """``s`` as displayed Ren'Py text: a string literal whose ``[`` and ``{`` are not interpolation."""
    return _py_str(s).replace("[", "[[").replace("{", "{{")


def renpy_voice_mix_rpy(duck: float, keysym: str, sample: str, lang: str) -> str:
    """Text of the generated ``game/vndub_voice_mix.rpy``.

    The dub plays on Ren'Py's ``voice`` channel, which has a mixer of its own; the channel gets that mixer only when
    ``config.has_voice`` is True when ``00mixers.rpy`` binds the mixers (``init 1600``), so the flag is set at
    ``init 1000``, after the game's ``options.rpy``. Ducking (``config.emphasize_audio_*``) lowers every other channel
    while the voice channel plays, but it is a player preference that is off by default: the script turns it on once
    (``persistent.vndub_voice_setup``) and keeps the strength in ``persistent.vndub_duck``, applied at every interaction
    (``config.emphasize_audio_volume`` is a plain amplitude once ``init 1500`` has run, on every Ren'Py version).
    ``config.overlay_screens`` are hidden in the game menu, so the button over the preferences screen is a
    ``config.always_shown_screens`` screen (Ren'Py 7.5 / 8.0+; older versions get it from an interact callback), and
    the hotkey is a ``config.underlay`` keymap, which works in the game and in the menus. ``sample`` is a voice file
    relative to ``game/`` for the test buttons (also ``config.sample_voice`` of the standard preferences screen).
    """
    s = _MIX_STRINGS.get((lang or "").split("-")[0].lower()) or _MIX_STRINGS["en"]
    key = voice_mix_key_name(keysym)
    t = {k: _rpy_text(v.replace("{key}", key)).replace("[[vndub_pct]", "[vndub_pct]") for k, v in s.items()}
    sample_q = _py_str(sample)
    duck_s = f"{duck:.3f}".rstrip("0").rstrip(".") or "0"
    return (
        "# Generated by vndub: the voice track of the dub.\n"
        "# The dub plays on Ren'Py's \"voice\" channel, which has a mixer of its own: it is never mixed into the music.\n"
        "# This file switches that channel on (config.has_voice), lowers the music and sound effects while a line plays\n"
        f"# (config.emphasize_audio_*, {duck_s} = full volume x {duck_s}) and adds a panel with the voice volume and the ducking\n"
        f"# strength: press {key} in the game, or the \"{t['button']}\" button in the top right corner of the preferences screen.\n"
        "# To undo all of it, delete this file and vndub_voice_mix.rpyc next to it.\n"
        "\n"
        "init 1000 python:\n"
        "    # before Ren'Py binds the mixers (init 1600): a voice channel switched on later would stay on the sfx mixer\n"
        "    config.has_voice = True\n"
        + (
            "    if not config.sample_voice:\n"
            f'        config.sample_voice = "{sample_q}"\n'
            if sample
            else ""
        )
        + "    if \"voice\" not in config.emphasize_audio_channels:\n"
        "        config.emphasize_audio_channels = list(config.emphasize_audio_channels) + [\"voice\"]\n"
        "    config.emphasize_audio_time = 0.3\n"
        f'    config.keymap["vndub_voice_panel"] = ["{keysym}"]\n'
        "    config.underlay = list(config.underlay) + [renpy.Keymap(vndub_voice_panel=ToggleScreen(\"vndub_voice_panel\"))]\n"
        "    if persistent.vndub_duck is None:\n"
        f"        persistent.vndub_duck = {duck_s}\n"
        "    if not persistent.vndub_voice_setup:\n"
        "        # ducking is a player preference that Ren'Py leaves off: switch it on once, the panel can switch it off\n"
        "        persistent.vndub_voice_setup = True\n"
        "        _preferences.emphasize_audio = True\n"
        "\n"
        "    def vndub_voice_apply():\n"
        "        # config.emphasize_audio_volume is a plain amplitude after init 1500 (the dB conversion of newer versions ran)\n"
        "        d = persistent.vndub_duck\n"
        "        if d is None:\n"
        f"            d = {duck_s}\n"
        "        config.emphasize_audio_volume = max(0.0, min(1.0, float(d)))\n"
        "\n"
        "    config.interact_callbacks.append(vndub_voice_apply)\n"
        "\n"
        "    # renpy.config is the plain module: the store's config proxy raises on an unknown name instead of returning False\n"
        "    if hasattr(renpy.config, \"always_shown_screens\"):\n"
        "        config.always_shown_screens = list(config.always_shown_screens) + [\"vndub_voice_hook\"]\n"
        "    else:\n"
        "        # Ren'Py before 7.5: no always_shown_screens, and overlay screens are hidden in the game menu\n"
        "        def vndub_voice_hook_interact():\n"
        "            try:\n"
        "                want = renpy.get_screen(\"preferences\") is not None\n"
        "                have = renpy.get_screen(\"vndub_voice_hook\") is not None\n"
        "                if want and not have:\n"
        "                    renpy.show_screen(\"vndub_voice_hook\")\n"
        "                elif have and not want:\n"
        "                    renpy.hide_screen(\"vndub_voice_hook\")\n"
        "            except Exception:\n"
        "                pass\n"
        "\n"
        "        config.interact_callbacks.append(vndub_voice_hook_interact)\n"
        "\n"
        "# the button on the preferences screen (empty everywhere else)\n"
        "screen vndub_voice_hook():\n"
        "    zorder 100\n"
        "    if renpy.get_screen(\"preferences\"):\n"
        f'        textbutton "{t["button"]}":\n'
        "            action ToggleScreen(\"vndub_voice_panel\")\n"
        "            text_font \"DejaVuSans.ttf\"\n"
        "            xalign 1.0\n"
        "            yalign 0.0\n"
        "            xoffset -24\n"
        "            yoffset 24\n"
        "\n"
        f"# the panel: {key} in the game, or that button\n"
        "screen vndub_voice_panel():\n"
        "    zorder 150\n"
        "    modal True\n"
        "    key \"game_menu\" action Hide(\"vndub_voice_panel\")\n"
        "    $ vndub_pct = int(round((persistent.vndub_duck or 0.0) * 100))\n"
        "    frame:\n"
        "        xalign 0.5\n"
        "        yalign 0.5\n"
        "        xsize 0.6\n"
        "        padding (32, 24)\n"
        "        vbox:\n"
        "            spacing 12\n"
        "            xfill True\n"
        f'            text "{t["title"]}" size 30 font "DejaVuSans.ttf"\n'
        f'            text "{t["hint"]}" size 18 font "DejaVuSans.ttf"\n'
        "            null height 8\n"
        f'            text "{t["voice_volume"]}" font "DejaVuSans.ttf"\n'
        "            bar value Preference(\"voice volume\") style \"slider\" xfill True\n"
        "            null height 8\n"
        f'            text "{t["duck"]}" font "DejaVuSans.ttf"\n'
        "            bar value FieldValue(persistent, \"vndub_duck\", range=1.0, style=\"slider\") xfill True\n"
        f'            textbutton "{t["duck_on"]}" action Preference("emphasize audio", "toggle") text_font "DejaVuSans.ttf"\n'
        f'            textbutton "{t["mute"]}" action Preference("voice mute", "toggle") text_font "DejaVuSans.ttf"\n'
        "            null height 8\n"
        "            hbox:\n"
        "                spacing 24\n"
        + (f'                textbutton "{t["test"]}" action Play("voice", "{sample_q}") text_font "DejaVuSans.ttf"\n' if sample else "")
        + f'                textbutton "{t["close"]}" action Hide("vndub_voice_panel") text_font "DejaVuSans.ttf"\n'
        f'            text "{t["key_hint"]}" size 16 font "DejaVuSans.ttf"\n'
    )


# ----------------------------------------------------------------------------
# Ren'Py: the translated text (game/tl/<language>/vndub_dialogue.rpy + game/vndub_language.rpy)
# ----------------------------------------------------------------------------

RENPY_TEXT_NAME = "vndub_dialogue.rpy"
RENPY_LANG_RPY = "vndub_language.rpy"
TEXT_DEFAULTS: dict[str, Any] = {"enabled": "auto", "language": "", "font": ""}
GENERATED = "# Generated by vndub"
# project.tgt_lang -> the lowercase language name Ren'Py's own tools and the launcher use for game/tl/<name>/
RENPY_LANGUAGES = {
    "ru": "russian", "en": "english", "ja": "japanese", "zh": "chinese", "ko": "korean", "de": "german", "fr": "french",
    "es": "spanish", "it": "italian", "pt": "portuguese", "uk": "ukrainian",
}
_LANG_NAME = re.compile(r"[a-z][a-z0-9_]*")
# what a translated text must keep from the original string: [interpolation], %(old)s substitution; of the {tags} only the ones that
# change the flow or the content (the translator sees the text without tags, so {i} / {b} / {color} are dropped, the line still reads)
_TOKEN = re.compile(r"\[\[|\{\{|%%|%\([^)]*\)[#0 +\-]*\d*(?:\.\d+)?[sdifr]|\[[^\[\]]+\]|\{[^{}]+\}")
_ESCAPED = {"[[", "{{", "%%"}
_KEEP_SCAN = re.compile(r"%\([^)]*\)[#0 +\-]*\d*(?:\.\d+)?[sdifr]|\[[^\[\]]+\]|\{[^{}]+\}")
_BEHAVIOR_TAGS = {"w", "p", "nw", "fast", "done", "cps", "clear", "image", "a", "rb", "rt", "space", "vspace"}
_TL_ID = re.compile(r"^\s*translate\s+(\w+)\s+(\w+)\s*:", re.M)
_TL_OLD = re.compile(r'^\s*old\s+("(?:[^"\\]|\\.)*")\s*$', re.M)
_FONT_EXT = {".ttf", ".otf", ".ttc"}


def renpy_language(tgt_lang: str, override: str = "") -> str:
    """Ren'Py's conventional language name for a project language code (``ru`` -> ``russian``); ``override`` wins when it is a valid
    name, an unknown code is its own name."""
    o = str(override or "").strip().lower()
    if o and _LANG_NAME.fullmatch(o):
        return o
    code = (tgt_lang or "").split("-")[0].split("_")[0].strip().lower()
    name = RENPY_LANGUAGES.get(code) or re.sub(r"[^a-z0-9_]", "", code)
    return name if _LANG_NAME.fullmatch(name) else "translated"


def text_opts(ctx: StageContext) -> dict[str, Any]:
    """``pack.renpy.text`` over :data:`TEXT_DEFAULTS` (a bare ``true`` / ``false`` is ``enabled``) plus the resolved ``lang`` name."""
    raw = ctx.cfg.pack.renpy.get("text")
    opts = dict(TEXT_DEFAULTS)
    if isinstance(raw, dict):
        opts.update({k: v for k, v in raw.items() if k in opts and v is not None})
    elif isinstance(raw, (bool, str)):
        opts["enabled"] = raw
    opts["lang"] = renpy_language(ctx.cfg.project.tgt_lang, str(opts["language"] or ""))
    opts["font"] = str(opts["font"] or "").strip().replace("\\", "/")
    return opts


def _markup_esc(s: str) -> str:
    return s.replace("%", "%%").replace("[", "[[").replace("{", "{{")


def _real_tokens(s: str) -> list[str]:
    return [t for t in _TOKEN.findall(s) if t not in _ESCAPED]


def _required_tokens(orig: str) -> list[str]:
    """The tokens of the original string that the translation has to carry: interpolations and the tags that change the flow."""
    out = []
    for t in _real_tokens(orig):
        if t.startswith(("[", "%(")):
            out.append(t)
        else:
            m = re.match(r"\{/?([a-z]+)", t)
            if m and m.group(1) in _BEHAVIOR_TAGS:
                out.append(t)
    return out


def translated_string(orig: str, tr: str) -> str | None:
    """``tr`` as Ren'Py string content (``[`` ``{`` ``%`` escaped except the original's own tokens), or None when the translation
    dropped an interpolation / flow tag of ``orig`` ("interpolation lost")."""
    need = Counter(_required_tokens(orig))
    have = Counter(_KEEP_SCAN.findall(tr))
    if any(have[t] < n for t, n in need.items()):
        return None
    keep = set(_real_tokens(orig))
    out: list[str] = []
    pos = 0
    for m in _KEEP_SCAN.finditer(tr):
        out.append(_markup_esc(tr[pos : m.start()]))
        out.append(m.group(0) if m.group(0) in keep else _markup_esc(m.group(0)))
        pos = m.end()
    out.append(_markup_esc(tr[pos:]))
    return "".join(out)


def existing_translations(ctx: StageContext, language: str) -> tuple[set[str], set[str]]:
    """(dialogue identifiers, ``old`` string literals) the game's own ``game/tl/<language>/*.rpy`` already translate: Ren'Py refuses a
    second block for either. The files this stage generated earlier are not counted."""
    ids: set[str] = set()
    olds: set[str] = set()
    gd = ctx.game_dir
    for base in (gd / "game" / "tl" / language, gd / "tl" / language):
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*.rpy")):
            try:
                text = p.read_text(encoding="utf-8-sig", errors="replace")
            except OSError:
                continue
            if text.startswith(GENERATED):
                continue
            ids |= {m.group(2) for m in _TL_ID.finditer(text) if m.group(1) == language and m.group(2) != "strings"}
            olds |= set(_TL_OLD.findall(text))
    return ids, olds


def _say_parts(rec: Any) -> tuple[str, str, str] | None:
    """``(head, tail, what)`` of a say line's raw statement, or None when it cannot be read: ``head`` = speaker, attributes and
    ``@ temporary`` words, ``tail`` = the clauses after the text, ``what`` = the original string as Ren'Py holds it (tags and all)."""
    from vndub.engines import renpy as rp

    raw = (rec.raw or "").strip()
    m = rp.SAY_RE.match(raw) if raw else None
    if not m:
        return None
    tok = m.group("what")
    head = " ".join(
        [w for w in [m.group("who") or ""] if w] + (m.group("attrs") or "").split()
        + (["@"] + (m.group("tmp") or "").replace("@", " ").split() if m.group("tmp") else [])
    )
    tail = m.group("rest").strip()
    index = 0
    if tok[:3] in ('"""', "'''"):
        paras = [p for _o, p in rp.triple_paragraphs(tok)]
        want = (rec.text_src or "").strip()
        hits = [i for i, p in enumerate(paras) if rp.strip_text_tags(p) == want]
        if not hits:
            return None
        index = hits[0]
        what = paras[index]
        if index:
            tail = re.sub(r"\bid\s+\S+\s*", "", tail).strip()  # the explicit identifier belongs to the first paragraph
    else:
        what = rp.dequote(tok)
    return head, tail, what


def build_text_patch(ctx: StageContext, language: str, font: str, tl: dict[str, Any]) -> dict[str, Any]:
    """The translated-text patch of a Ren'Py project: ``{text, lines, strings, skipped, skipped_count, reasons}``.

    One block per translated say line (``text_tgt`` of ``lines.<tgt>.jsonl``, never ``text_tts``), a ``strings`` block for menu
    choices (``choices_tgt`` of the menu's translation record) and menu captions. What is left out is listed in ``skipped``."""
    from vndub.engines import renpy as rp

    taken_ids, taken_old = existing_translations(ctx, language)
    skipped: list[dict[str, str]] = []
    reasons: Counter[str] = Counter()

    def skip(rec: Any, why: str, ident: str = "") -> None:
        reasons[why] += 1
        if len(skipped) < LEFT_OUT_LISTED:
            skipped.append({"line_id": rec.line_id, "id": ident, "reason": why})

    blocks: list[str] = []
    strings: list[str] = []
    seen_ids: set[str] = set()
    seen_old: dict[str, str] = {}
    n_lines = n_strings = 0

    def add_string(rec: Any, old_raw: str, new_content: str) -> None:
        nonlocal n_strings
        old_lit = rp.encode_say_string(old_raw)
        new_lit = rp.encode_say_string(new_content)
        if old_lit in taken_old:
            skip(rec, "string already translated by the game", old_raw)
            return
        if old_lit in seen_old:
            if seen_old[old_lit] != new_lit:
                skip(rec, "same string translated twice", old_raw)
            return
        seen_old[old_lit] = new_lit
        strings.append(f"    # {_loc(rec)}\n    old {old_lit}\n    new {new_lit}\n")
        n_strings += 1

    for rec in ctx.lines(recovered=False, edits=False):
        flags = set(rec.flags or [])
        extra = rec.model_extra or {}
        tr = tl.get(rec.line_id)
        if "menu" in flags:
            choices = [str(c) for c in (rec.choices or [])]
            ctr = getattr(tr, "choices_tgt", None) if tr is not None else None
            if not choices:
                continue
            if not isinstance(ctr, list) or len(ctr) != len(choices) or not all(isinstance(c, str) for c in ctr):
                skip(rec, "choices not translated")
                continue
            for old, new in zip(choices, ctr):
                new = new.strip()
                if not new:
                    skip(rec, "choices not translated", old)
                elif re.search(r"[\[\]{}%]", old):
                    skip(rec, "markup in the choice", old)  # the parser stripped its tags: the script string cannot be restored
                else:
                    add_string(rec, old, _markup_esc(new))
            continue
        if "menu_caption" in flags:
            parts = _say_parts(rec)
            text = (tr.text_tgt or "").strip() if tr is not None else ""
            if parts is None or not text:
                skip(rec, "not translated" if not text else "source line unreadable")
                continue
            new = translated_string(parts[2], text)
            if new is None:
                skip(rec, "interpolation lost")
            else:
                add_string(rec, parts[2], new)
            continue
        ident = str(extra.get("dialogue_id") or "")
        if not ident:
            continue
        text = (tr.text_tgt or "").strip() if tr is not None else ""
        if not text:
            skip(rec, "not translated", ident)
            continue
        if ident in taken_ids:
            skip(rec, "identifier already translated by the game", ident)
            continue
        if ident in seen_ids:
            skip(rec, "duplicate identifier", ident)
            continue
        parts = _say_parts(rec)
        if parts is None:
            skip(rec, "source line unreadable", ident)
            continue
        head, tail, what = parts
        new = translated_string(what, text)
        if new is None:
            skip(rec, "interpolation lost", ident)
            continue
        voice = ""
        if "voice_sustain" in flags:
            voice = "voice sustain"
        elif "voice_expression" in flags or "voice_missing" in flags:
            skip(rec, "own voice statement", ident)  # a translate block replaces the voice statement of its group: it cannot be rebuilt
            continue
        elif rec.has_voice and "auto_voice" not in flags and rec.voice_file:
            voice = f"voice {rp.encode_say_string(rec.voice_file)}"
        if "nointeract" not in tail.split() and ("menu_say" in flags or _menu_say_by_digest(rec, ident, voice)):
            tail = ("nointeract " + tail).strip()  # a menu's own say: the script never writes it, the menu provides the interaction
        seen_ids.add(ident)
        code = " ".join(x for x in (head, rp.encode_say_string(new), tail) if x)
        comment = f"    # {rec.raw.strip()}\n" if rec.raw and "\n" not in rec.raw else ""
        body = (f"    {voice}\n" if voice else "") + f"    {code}\n"
        blocks.append(f"# {_loc(rec)}\ntranslate {language} {ident}:\n\n{comment}{body}")
        n_lines += 1
    text = renpy_text_rpy(language, blocks, strings, font)
    return {
        "text": text if (blocks or strings) else "", "lines": n_lines, "strings": n_strings,
        "skipped": skipped, "skipped_count": sum(reasons.values()), "reasons": dict(reasons),
    }


_ID_DIGEST = re.compile(r"_([0-9a-f]{8})(?:_\d+)?$")


def _menu_say_by_digest(rec: Any, ident: str, voice: str) -> bool:
    """Records parsed before the ``menu_say`` flag existed: True when the dialogue identifier's digest matches this say's code WITH
    ``nointeract`` and not without it (the id of a menu's own say is computed over the code of the ``nointeract`` say)."""
    from vndub.engines import renpy as rp

    dm = _ID_DIGEST.search(ident)
    m = rp.SAY_RE.match((rec.raw or "").strip())
    if not dm or not m or m.group("what")[:3] in ('"""', "'''"):
        return False
    interact, explicit_id, args, with_ = rp.parse_say_tail(m.group("rest").strip())
    if not interact:
        return False
    attrs = (m.group("attrs") or "").split()
    tmp = (m.group("tmp") or "").replace("@", " ").split()
    what = rp.dequote(m.group("what"))
    prefixes = [[], [voice]] if voice else [[]]

    def hit(inter: bool) -> bool:
        code = rp.say_code(m.group("who"), what, attrs, tmp, inter, explicit_id, args, with_)
        return any(rp.block_digest([*p, code]) == dm.group(1) for p in prefixes)

    return hit(False) and not hit(True)


def _loc(rec: Any) -> str:
    f = str(rec.source_file or "").replace("\\", "/")
    if f and not f.startswith("game/"):
        f = "game/" + f
    return f"{f}:{rec.source_line}" if f and rec.source_line else (f or rec.line_id)


def renpy_text_rpy(language: str, blocks: list[str], strings: list[str], font: str = "") -> str:
    """Text of the generated ``game/tl/<language>/vndub_dialogue.rpy``: a ``translate <language> <id>:`` block per say line (the
    shape ``renpy translate`` writes), then ``translate <language> strings:`` for menu choices and captions, then (``font``)
    a ``translate <language> python:`` block that switches the game's fonts."""
    head = (
        f"{GENERATED}: the translated text of the dub.\n"
        f"# One translate block per say statement (the identifier is the one \"renpy translate\" writes) and one strings block for menu\n"
        f"# choices and captions. Ren'Py uses them while the game language is \"{language}\" (game/{RENPY_LANG_RPY} sets it).\n"
        f"# To undo it, delete this file and {PurePosixPath(RENPY_TEXT_NAME).stem}.rpyc next to it.\n"
    )
    out = [head]
    if font:
        out.append(
            f"translate {language} python:\n"
            f'    gui.text_font = "{_py_str(font)}"\n'
            f'    gui.name_text_font = "{_py_str(font)}"\n'
            f'    gui.interface_text_font = "{_py_str(font)}"\n'
        )
    out.extend(blocks)
    if strings:
        out.append(f"translate {language} strings:\n\n" + "\n".join(strings))
    return "\n".join(out)


def renpy_language_rpy(language: str) -> str:
    """Text of the generated ``game/vndub_language.rpy``: the game starts in the translated language."""
    return (
        f"{GENERATED}: the game starts in the language of the dub.\n"
        f"# config.language beats the player's saved language choice at every start. The text itself is in\n"
        f"# game/tl/{language}/{RENPY_TEXT_NAME}.\n"
        f"# To undo it, delete this file and {PurePosixPath(RENPY_LANG_RPY).stem}.rpyc next to it (the player can then pick the language in the preferences).\n"
        f'define config.language = "{language}"\n'
    )


def _text_notes(info: dict[str, Any]) -> list[str]:
    """INSTALL.md section for the generated translation script (``info``: the manifest's ``text`` block)."""
    lang = info["language"]
    rpyc = PurePosixPath(info["rpy"]).stem + ".rpyc"
    return [
        "",
        "## The translated text",
        "",
        f"The dub also carries the translated text. `{info['rpy']}` holds {info['lines']} `translate {lang} <dialogue identifier>:` block(s) "
        f"(one per say statement, the shape `renpy translate` writes)"
        + (f" and a `translate {lang} strings:` block with {info['strings']} menu choice(s) and caption(s)" if info["strings"] else "")
        + f", and `{info['language_rpy']}` makes the game start in the language \"{lang}\" (`config.language`, which beats the player's saved choice). "
        + (f"{info['skipped_count']} line(s) stay in the original language (see `text.skipped` in manifest_patch.json). " if info["skipped_count"] else "")
        + f"To undo it, delete `{info['language_rpy']}`, `{info['rpy']}` and their `.rpyc` twins (`{rpyc}`) from the game folder. "
        "`pack.renpy.text` configures it: `enabled: auto | true | false` (auto = on when the project has a translation), `language` (the folder name under `game/tl/`), "
        "`font` (a .ttf / .otf inside `game/` for a game whose own font lacks the alphabet).",
    ]


def _pack_text_renpy(ctx: StageContext, out: Path) -> tuple[dict[str, Any] | None, int]:
    """Write ``game/tl/<language>/vndub_dialogue.rpy`` and ``game/vndub_language.rpy``; returns (the manifest's ``text`` block or None
    when the feature is off or has nothing to write, the number of lines left out)."""
    opts = text_opts(ctx)
    mode = _mix_mode(opts["enabled"])
    tgt = ctx.cfg.project.tgt_lang
    p = ctx.paths.translation(tgt)
    if mode == "off" or (mode == "auto" and not p.exists()):
        return None, 0
    if not p.exists():
        ctx.log.warning("pack: pack.renpy.text is on but %s does not exist: run `vndub translate` first; no text patch", p)
        return None, 0
    lang = opts["lang"]
    font = opts["font"]
    if font:
        rel = font[len("game/") :] if font.startswith("game/") else font
        if PurePosixPath(rel).suffix.lower() not in _FONT_EXT or ".." in PurePosixPath(rel).parts or rel.startswith("/") or re.match(r"[A-Za-z]:", rel):
            ctx.log.warning("pack: pack.renpy.text.font %r is not a .ttf / .otf path inside game/: ignored", font)
            rel = ""
        font = rel
    tl = {r.line_id: r for r in read_jsonl(p, TranslationRecord)}
    res = build_text_patch(ctx, lang, font, tl)
    if not res["text"]:
        ctx.log.warning("pack: no line of %s could be written as Ren'Py text (%s)", p, res["reasons"] or "no say statements with a dialogue identifier")
        return None, res["skipped_count"]
    rel_rpy = f"game/tl/{lang}/{RENPY_TEXT_NAME}"
    (out / "game" / "tl" / lang).mkdir(parents=True, exist_ok=True)
    for name, text in ((out / rel_rpy, res["text"]), (out / "game" / RENPY_LANG_RPY, renpy_language_rpy(lang))):
        with open(name, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    info = {
        "rpy": rel_rpy, "language_rpy": f"game/{RENPY_LANG_RPY}", "language": lang, "lines": res["lines"], "strings": res["strings"],
        "skipped": res["skipped"], "skipped_count": res["skipped_count"], "skipped_reasons": res["reasons"], "font": font or None, "enabled": True,
    }
    ctx.log.info("pack: text patch %s: %d line(s), %d string(s), %d left in the original language %s", rel_rpy, res["lines"], res["strings"], res["skipped_count"], res["reasons"] or "")
    if res["skipped_count"]:
        ids = [s["id"] or s["line_id"] for s in res["skipped"] if s["reason"] == "interpolation lost"]
        if ids:
            ctx.log.warning("pack: %d line(s) lost an interpolation or flow tag in the translation and keep the original text: %s", len(ids), ", ".join(ids[:20]))
    return info, res["skipped_count"]


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


def _remove_stale(
    ctx: StageContext, eng: Any, out: Path, regular: list[FinalIndexRecord], placed: dict[str, str], uinfo: dict[str, Any], mix_on: bool,
    text_keep: set[Path] | None = None,
) -> int:
    """After a full pack: delete what an earlier pack left in ``out`` and this one does not hold. The installer ships the whole folder,
    so a file the new ``manifest_patch.json`` does not list would be installed all the same (a rejected line, ``include_review_lines``
    switched to ``skip``, an import under another parse version that renamed the Ren'Py ids ...). Touched are only the files the previous
    ``manifest_patch.json`` listed, the audio files next to files of the patch (the engine's voice folders), and the ``vndub_auto_voice.rpy``
    and ``vndub_voice_mix.rpy`` this stage generated when they are no longer needed, and so are the translation scripts
    (``vndub_language.rpy``, ``tl/<language>/vndub_dialogue.rpy``; ``text_keep`` = the ones this pack wrote); nothing else. Returns the number of files removed."""
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
    text_files: list[Path] = []
    if eng.name == "renpy":  # the generated translation scripts of any language (a changed language leaves the old folder behind)
        text_files = [root / RENPY_LANG_RPY, *sorted(root.glob(f"tl/*/{RENPY_TEXT_NAME}"))]
        dirs |= {p.parent for p in text_files[1:]} | {root / "tl"}
    n = 0
    for p in sorted(stale):
        try:
            p.unlink()
            n += 1
        except FileNotFoundError:
            pass
        except OSError as e:
            ctx.log.warning("pack: cannot remove the stale file %s (%s)", p, e)
    if eng.name == "renpy":  # the generated scripts of an earlier pack, when this one does not need them
        keep_text = text_keep or set()
        gen = [(root / RENPY_RPY, "rpy" in uinfo), (root / RENPY_MIX_RPY, mix_on)] + [(p, p in keep_text) for p in text_files]
        for rpy, needed in gen:
            try:
                if not needed and rpy.is_file() and rpy.read_text(encoding="utf-8").startswith("# Generated by vndub"):
                    rpy.unlink()
                    n += 1
            except (OSError, UnicodeDecodeError) as e:
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


def _voice_mix_notes(info: dict[str, Any]) -> list[str]:
    """INSTALL.md section for the generated ``vndub_voice_mix.rpy`` (``info``: the manifest's ``voice_mix`` block + the ``button`` text)."""
    pct = round(float(info["duck"]) * 100)
    key = voice_mix_key_name(str(info["keysym"]))
    return [
        "",
        "## The voice track in the game",
        "",
        "The dub plays on Ren'Py's \"voice\" channel, which has a mixer of its own: the music volume does not touch it. "
        f"`{info['rpy']}` switches that channel on, lowers the music and sound effects to {pct}% while a line plays, "
        f"and adds a panel with the voice volume and the ducking strength: press {key} in the game, or the \"{info['button']}\" button in the "
        "top right corner of the game's preferences screen (a standard preferences screen also gains a \"Voice Volume\" slider with a Test button). "
        f"To undo it, delete that file and `{PurePosixPath(info['rpy']).stem}.rpyc` next to it. "
        "`pack.renpy.voice_mix` configures it: `enabled: auto | true | false` (auto = on when the patch holds lines that had no original voice), "
        "`duck` (0.3 = music at 30 % while a line plays, 1 = no ducking), `key` (default `K_F8`).",
    ]


def _voice_mix_sample(placed: dict[str, str], regular: list[FinalIndexRecord], on_disk: list[FinalIndexRecord]) -> str:
    """A voice file of the patch relative to ``game/`` for the panel's test button: the first unvoiced file placed now, else the first
    regular one (of this pack, else of the project: an earlier pack put it there), else nothing."""
    for rel in placed.values():
        return rel[len("game/") :] if rel.startswith("game/") else rel
    for f in [*regular, *on_disk]:
        if not _is_unvoiced(f):
            return f.voice_file
    return ""


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
    mix_on = False
    mix_info: dict[str, Any] | None = None
    if eng.name == "renpy":
        mix = voice_mix_opts(ctx)
        mode = _mix_mode(mix["enabled"])
        # auto: the game was never mixed for voice when the patch holds lines that had none (the whole project, so --only packs agree)
        mix_on = mode == "on" or (mode == "auto" and any(_is_unvoiced(f) for f in on_disk))
        if mix_on:
            lang = ctx.cfg.project.tgt_lang
            (out / "game").mkdir(parents=True, exist_ok=True)
            sample = _voice_mix_sample(placed, regular, on_disk)
            (out / "game" / RENPY_MIX_RPY).write_text(renpy_voice_mix_rpy(mix["duck"], mix["key"], sample, lang), encoding="utf-8")
            mix_info = {"rpy": f"game/{RENPY_MIX_RPY}", "key": voice_mix_key_name(mix["key"]), "keysym": mix["key"], "duck": mix["duck"], "enabled": True}
            strings = _MIX_STRINGS.get((lang or "").split("-")[0].lower()) or _MIX_STRINGS["en"]
            with open(out / "INSTALL.md", "a", encoding="utf-8") as fh:
                fh.write("\n".join(_voice_mix_notes({**mix_info, "button": strings["button"]})) + "\n")
    text_info: dict[str, Any] | None = None
    text_keep: set[Path] = set()
    text_skipped = 0
    if eng.name == "renpy":
        text_info, text_skipped = _pack_text_renpy(ctx, out)
        if text_info:
            text_keep = {out / "game" / RENPY_LANG_RPY, out / text_info["rpy"]}
            with open(out / "INSTALL.md", "a", encoding="utf-8") as fh:
                fh.write("\n".join(_text_notes(text_info)) + "\n")
    removed = 0 if ctx.only else _remove_stale(ctx, eng, out, regular, placed, uinfo, mix_on, text_keep)  # a partial pack (--only) leaves the rest of the folder alone
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
    manifest: dict[str, Any] = {"engine": eng.name, "game_dir": proj.game_dir, "coverage": cov, "files": entries}
    if eng.name == "renpy":
        manifest["voice_mix"] = mix_info
        manifest["text"] = text_info
    write_json(out / "manifest_patch.json", manifest)
    summary = {**res, "engine": eng.name, "manifest": str(out / "manifest_patch.json")}
    if eng.name == "renpy":
        summary["voice_mix"] = mix_on
        summary["text"] = bool(text_info)
        if text_info:
            summary["text_lines"] = text_info["lines"]
            summary["text_strings"] = text_info["strings"]
        if text_skipped:
            summary["text_skipped"] = text_skipped
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

"""pack layouts per engine with fake final_index records."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from vndub.config import ProjectYaml, save_project_yaml
from vndub.engines import xp3
from vndub.manifest.io import FinalIndexRecord, SelectionRecord, read_jsonl, write_jsonl
from vndub.stages import pack as pack_stage
from vndub.stages.common import StageError, run_stage


def _setup(make_ctx, tmp_path: Path, engine: str, voice_files: list[str], **proj_extra):
    game = tmp_path / "game"
    game.mkdir(parents=True, exist_ok=True)
    ctx = make_ctx(game)
    save_project_yaml(ctx.paths.project_yaml, ProjectYaml(game_dir=str(game), engine=engine, **proj_extra))
    finals, sel = [], []
    for i, vf in enumerate(voice_files):
        lid = f"s_{i:06d}"
        enc = ctx.paths.final_encoded / vf
        enc.parent.mkdir(parents=True, exist_ok=True)
        enc.write_bytes(b"OggS" + bytes([i]) * 64)
        finals.append(FinalIndexRecord(line_id=lid, voice_file=vf, final_wav=f"final/{lid}.wav", encoded=ctx.paths.rel(enc), source=f"synth/{lid}/mock_tts_A_0.wav"))
        sel.append(SelectionRecord(line_id=lid, winner=f"synth/{lid}/mock_tts_A_0.wav", backend="mock_tts", route="A", status="auto"))
    write_jsonl(ctx.paths.final_index, finals)
    write_jsonl(ctx.paths.selection, sel)
    return ctx, game


def _manifest(out: Path) -> dict:
    return json.loads((out / "manifest_patch.json").read_text(encoding="utf-8"))


def test_pack_renpy(make_ctx, tmp_path: Path) -> None:
    ctx, _ = _setup(make_ctx, tmp_path, "renpy", ["voice/sak_0001.ogg", "voice/har_0001.ogg"])
    s = run_stage("pack", ctx)
    out = ctx.paths.out_engine("renpy")
    assert (out / "game" / "voice" / "sak_0001.ogg").read_bytes().startswith(b"OggS")
    assert s["files"] == 2
    assert "force_archives" in (out / "INSTALL.md").read_text(encoding="utf-8")
    m = _manifest(out)
    assert [f["file"] for f in m["files"]] == ["voice/sak_0001.ogg", "voice/har_0001.ogg"]
    assert m["files"][0]["backend"] == "mock_tts" and m["files"][0]["route"] == "A" and len(m["files"][0]["sha256"]) == 64
    # a game with voice of its own is already mixed for it: no generated voice-mix script unless asked for
    assert not (out / "game" / "vndub_voice_mix.rpy").exists() and m["voice_mix"] is None and s["voice_mix"] is False
    assert "The voice track in the game" not in (out / "INSTALL.md").read_text(encoding="utf-8")


def test_pack_kirikiri_patch_xp3(make_ctx, tmp_path: Path) -> None:
    ctx, game = _setup(make_ctx, tmp_path, "kirikiri", ["voice/nen_0001.ogg", "nen_0002.ogg"], encrypted=True)
    (game / "patch.xp3").write_bytes(b"")  # existing patch -> next number
    s = run_stage("pack", ctx)
    target = ctx.paths.out_engine("kirikiri") / "patch2.xp3"
    assert s["out"] == str(target) and s["writer"] == "builtin"
    with xp3.XP3Reader(target) as r:
        assert sorted(e.name for e in r.entries) == ["nen_0001.ogg", "nen_0002.ogg"]  # flat root entries
        assert r.read("nen_0001.ogg").startswith(b"OggS")
    text = (ctx.paths.out_engine("kirikiri") / "INSTALL.md").read_text(encoding="utf-8")
    assert "version.dll" in text and "patch2.xp3" in text
    assert _manifest(ctx.paths.out_engine("kirikiri"))["files"][0]["packed_as"] == "nen_0001.ogg"


def test_pack_kirikiri_subdir_and_unencrypted(make_ctx, tmp_path: Path) -> None:
    ctx, _ = _setup(make_ctx, tmp_path, "kirikiri", ["voice/nen_0001.ogg"])
    ctx.cfg.pack.kirikiri["subdir"] = "voice"
    run_stage("pack", ctx)
    with xp3.XP3Reader(ctx.paths.out_engine("kirikiri") / "patch.xp3") as r:
        assert [e.name for e in r.entries] == ["voice/nen_0001.ogg"]
    assert "addAutoPath" in (ctx.paths.out_engine("kirikiri") / "INSTALL.md").read_text(encoding="utf-8")
    ctx.cfg.pack.kirikiri.update(mode="unencrypted_dir", subdir="")
    s = run_stage("pack", ctx)
    assert (ctx.paths.out_engine("kirikiri") / "unencrypted" / "nen_0001.ogg").exists() and s["mode"] == "unencrypted_dir"


def test_pack_nscripter_and_tyrano(make_ctx, tmp_path: Path) -> None:
    ctx, _ = _setup(make_ctx, tmp_path, "nscripter", ["voice/mio001.wav"])
    run_stage("pack", ctx)
    assert (ctx.paths.out_engine("nscripter") / "voice" / "mio001.wav").exists()
    ctx2, _ = _setup(make_ctx, tmp_path / "t", "tyrano", ["sound/akane_1.ogg"])
    run_stage("pack", ctx2)
    assert (ctx2.paths.out_engine("tyrano") / "data" / "sound" / "akane_1.ogg").exists()


def test_pack_unity_script(make_ctx, tmp_path: Path) -> None:
    ctx, _ = _setup(make_ctx, tmp_path, "unity", ["Voice/kohaku_01.wav"])
    s = run_stage("pack", ctx)
    out = ctx.paths.out_engine("unity")
    assert s["experimental"] is True
    assert (out / "clips" / "kohaku_01.wav").exists()
    script = (out / "replace_audioclips.py").read_text(encoding="utf-8")
    compile(script, "replace_audioclips.py", "exec")
    assert json.loads((out / "clips.json").read_text(encoding="utf-8"))["replacements"] == {"kohaku_01": "kohaku_01.wav"}
    assert "EXPERIMENTAL" in (out / "INSTALL.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("engine", ["siglus", "bgi", "artemis", "catsystem2", "generic", "yuris", "majiro"])
def test_pack_other_engines_write_install(make_ctx, tmp_path: Path, engine: str) -> None:
    ctx, _ = _setup(make_ctx, tmp_path, engine, ["voice/a_0001.ogg"])
    s = run_stage("pack", ctx)
    out = ctx.paths.out_engine(engine)
    assert (out / "INSTALL.md").exists() and (out / "manifest_patch.json").exists()
    assert s["files"] == 1


def test_pack_without_master(make_ctx, tmp_path: Path) -> None:
    game = tmp_path / "game"
    game.mkdir()
    ctx = make_ctx(game)
    save_project_yaml(ctx.paths.project_yaml, ProjectYaml(game_dir=str(game), engine="renpy"))
    with pytest.raises(StageError, match="master"):
        run_stage("pack", ctx)


def test_pack_states_what_it_covers_without_a_lines_manifest(make_ctx, tmp_path: Path) -> None:
    ctx, _ = _setup(make_ctx, tmp_path, "renpy", ["voice/sak_0001.ogg", "voice/har_0001.ogg"])
    finals = read_jsonl(ctx.paths.final_index, FinalIndexRecord)
    finals[1].status = "provisional"  # master marks a line mastered from a take that failed the gates
    finals[1].reason = "no candidate passes the gates after 2 regen round(s)"  # type: ignore[attr-defined]
    write_jsonl(ctx.paths.final_index, finals)
    s = run_stage("pack", ctx)
    out = ctx.paths.out_engine("renpy")
    cov = _manifest(out)["coverage"]
    assert (cov["dub_lines"], cov["installed"], cov["provisional"], cov["left_out"]) == (2, 2, 1, 0)
    assert s["provisional"] == 1 and "left_out" not in s
    m = _manifest(out)["files"]
    assert "reason" not in m[0] and m[1]["status"] == "provisional" and "regen round" in m[1]["reason"]
    text = (out / "INSTALL.md").read_text(encoding="utf-8")
    assert "What this patch covers" in text and "1 line(s) are PROVISIONAL" in text and "LEFT OUT" not in text
    run_stage("pack", ctx)
    assert (out / "INSTALL.md").read_text(encoding="utf-8").count("What this patch covers") == 1


# ----------------------------------------------------------------------------
# Ren'Py: the voice track (vndub_voice_mix.rpy)
# ----------------------------------------------------------------------------


def _mix_text(out: Path) -> str:
    return (out / "game" / "vndub_voice_mix.rpy").read_text(encoding="utf-8")


def test_pack_renpy_voice_mix_can_be_forced_on_for_a_voiced_game(make_ctx, tmp_path: Path) -> None:
    ctx, _ = _setup(make_ctx, tmp_path, "renpy", ["voice/sak_0001.ogg", "voice/har_0001.ogg"])
    ctx.cfg.pack.renpy["voice_mix"] = True  # the bare form is `enabled`
    s = run_stage("pack", ctx)
    out = ctx.paths.out_engine("renpy")
    text = _mix_text(out)
    assert s["voice_mix"] is True and text.startswith("# Generated by vndub")
    assert 'config.keymap["vndub_voice_panel"] = ["K_F8"]' in text and "persistent.vndub_duck = 0.3" in text
    assert 'Play("voice", "voice/sak_0001.ogg")' in text  # the sample is the first voice file of the patch
    assert (out / "game" / "voice" / "sak_0001.ogg").exists()
    assert _manifest(out)["voice_mix"] == {"rpy": "game/vndub_voice_mix.rpy", "key": "F8", "keysym": "K_F8", "duck": 0.3, "enabled": True}
    install = (out / "INSTALL.md").read_text(encoding="utf-8")
    assert install.count("## The voice track in the game") == 1 and "press F8 in the game" in install and "to 30% while a line plays" in install
    assert '"Озвучка" button' in install  # tgt_lang is ru
    run_stage("pack", ctx)  # the second pack rebuilds INSTALL.md, the section does not pile up
    assert (out / "INSTALL.md").read_text(encoding="utf-8").count("The voice track in the game") == 1


def test_pack_renpy_voice_mix_dict_form_sets_the_key_and_the_ducking(make_ctx, tmp_path: Path) -> None:
    ctx, _ = _setup(make_ctx, tmp_path, "renpy", ["voice/sak_0001.ogg", "voice/har_0001.ogg"])
    ctx.cfg.pack.renpy["voice_mix"] = {"enabled": True, "duck": 0.5, "key": "F9"}
    s = run_stage("pack", ctx)
    out = ctx.paths.out_engine("renpy")
    text = _mix_text(out)
    assert 'config.keymap["vndub_voice_panel"] = ["K_F9"]' in text and "persistent.vndub_duck = 0.5" in text and "d = 0.5" in text
    vm = _manifest(out)["voice_mix"]
    assert vm == {"rpy": "game/vndub_voice_mix.rpy", "key": "F9", "keysym": "K_F9", "duck": 0.5, "enabled": True} and s["voice_mix"] is True
    install = (out / "INSTALL.md").read_text(encoding="utf-8")
    assert "to 50% while a line plays" in install and "press F9 in the game" in install
    first = _manifest(out)["files"][0]["file"]
    assert f'Play("voice", "{first}")' in text and f'config.sample_voice = "{first}"' in text


@pytest.mark.parametrize(
    "value, expected",
    [(True, True), ("true", True), ("on", True), ("ON", True), (False, False), ("false", False), ("off", False), ("auto", False), (None, False), ("whatever", False)],
)
def test_pack_renpy_voice_mix_enabled_spellings(make_ctx, tmp_path: Path, value, expected: bool) -> None:
    """YAML gives a bool or a string; "auto" and anything unknown mean auto, which is off for a game that was voiced already."""
    ctx, _ = _setup(make_ctx, tmp_path, "renpy", ["voice/sak_0001.ogg"])
    ctx.cfg.pack.renpy["voice_mix"] = {"enabled": value}
    s = run_stage("pack", ctx)
    out = ctx.paths.out_engine("renpy")
    assert s["voice_mix"] is expected and (out / "game" / "vndub_voice_mix.rpy").exists() is expected
    assert (_manifest(out)["voice_mix"] is not None) is expected


def test_pack_other_engines_have_no_voice_mix(make_ctx, tmp_path: Path) -> None:
    ctx, _ = _setup(make_ctx, tmp_path, "nscripter", ["voice/mio001.wav"])
    ctx.cfg.pack.renpy["voice_mix"] = True
    s = run_stage("pack", ctx)
    out = ctx.paths.out_engine("nscripter")
    assert "voice_mix" not in s and "voice_mix" not in _manifest(out) and not list(out.rglob("vndub_voice_mix.rpy"))


# ---- the helpers


def test_voice_mix_opts_normalizes_the_key_the_ducking_and_the_shorthand(make_ctx, tmp_path: Path) -> None:
    ctx, _ = _setup(make_ctx, tmp_path, "renpy", [])

    def opts(value):
        ctx.cfg.pack.renpy["voice_mix"] = value
        return pack_stage.voice_mix_opts(ctx)

    ctx.cfg.pack.renpy.pop("voice_mix", None)
    assert pack_stage.voice_mix_opts(ctx) == {"enabled": "auto", "duck": 0.3, "key": "K_F8"}  # the defaults
    assert opts({"key": "F8"})["key"] == "K_F8" and opts({"key": "K_F8"})["key"] == "K_F8" and opts({"key": "shift_K_F8"})["key"] == "shift_K_F8"
    assert opts({"key": "f9"})["key"] == "K_F9" and opts({"key": "a"})["key"] == "K_a"  # function keys are upper case, letter keys lower case
    assert opts({"key": ""})["key"] == "K_F8" and opts({"key": None})["key"] == "K_F8"
    assert opts({"key": 'K_F8"] + evil("'})["key"] == "K_F8"  # not a keysym: it would break the quoted string of the generated script
    assert opts({"duck": 2})["duck"] == 1.0 and opts({"duck": -1})["duck"] == 0.0 and opts({"duck": "0.45"})["duck"] == 0.45
    assert opts({"duck": "loud"})["duck"] == 0.3 and opts({"duck": float("nan")})["duck"] == 0.3 and opts({"duck": None})["duck"] == 0.3
    assert opts(True)["enabled"] is True and opts(False)["enabled"] is False and opts("off")["enabled"] == "off"
    assert opts({"enabled": False, "duck": 0.1})["enabled"] is False and opts({"unknown": 1}) == pack_stage.VOICE_MIX_DEFAULTS
    assert opts(12) == pack_stage.VOICE_MIX_DEFAULTS  # nonsense falls back to the defaults


def test_voice_mix_enabled_modes() -> None:
    mode = pack_stage._mix_mode
    assert [mode(v) for v in (True, "true", "True", "on", "yes")] == ["on"] * 5
    assert [mode(v) for v in (False, "false", "off", "no")] == ["off"] * 4
    assert [mode(v) for v in ("auto", "", None, "maybe", 1, 0.5)] == ["auto"] * 6


def test_voice_mix_key_name() -> None:
    assert pack_stage.voice_mix_key_name("K_F8") == "F8" and pack_stage.voice_mix_key_name("shift_K_F8") == "shift_K_F8"


def test_voice_mix_rpy_escapes_the_sample_path_and_keeps_the_interpolation() -> None:
    sample = r'voice/a"b\c.ogg'  # a quote and a backslash in the file name
    quoted = r'voice/a\"b\\c.ogg'
    assert pack_stage._py_str(sample) == quoted
    text = pack_stage.renpy_voice_mix_rpy(0.3, "K_F8", sample, "en")
    assert f'config.sample_voice = "{quoted}"' in text and f'Play("voice", "{quoted}")' in text
    assert 'a"b' not in text  # the bare quote never reaches a string literal
    code = "\n".join(l[4:] for l in text.split("init 1000 python:\n", 1)[1].split("\n# the button", 1)[0].splitlines())
    ast.parse(code)  # still valid Python
    assert "[vndub_pct]%" in text and "[[vndub_pct]" not in text  # the only interpolation is the percentage
    assert pack_stage._rpy_text('a [b] {c} "d"') == r'a [[b] {{c} \"d\"'  # a displayed string: [ and { are doubled, the quote is escaped
    assert pack_stage._rpy_text("100%") == "100%"


def test_voice_mix_rpy_language_and_unknown_language_fallback() -> None:
    ru = pack_stage.renpy_voice_mix_rpy(0.3, "K_F8", "voice/a.ogg", "ru")
    assert "Озвучка" in ru and "Громкость голоса" in ru and "Voice volume" not in ru
    for lang in ("xx", "", None, "ja-JP"):
        text = pack_stage.renpy_voice_mix_rpy(0.3, "K_F8", "voice/a.ogg", lang)
        assert "Voice volume" in text and "Озвучка" not in text, lang
    assert "Озвучка" in pack_stage.renpy_voice_mix_rpy(0.3, "K_F8", "voice/a.ogg", "RU-ru")  # region and case do not matter
    assert "The F8 key opens this panel" in pack_stage.renpy_voice_mix_rpy(0.3, "K_F8", "voice/a.ogg", "en")
    assert "persistent.vndub_duck = 1\n" in pack_stage.renpy_voice_mix_rpy(1.0, "K_F8", "voice/a.ogg", "en")  # no trailing zeros
    assert "persistent.vndub_duck = 0\n" in pack_stage.renpy_voice_mix_rpy(0.0, "K_F8", "voice/a.ogg", "en")


def test_voice_mix_notes_text() -> None:
    info = {"rpy": "game/vndub_voice_mix.rpy", "key": "F8", "keysym": "K_F8", "duck": 0.3, "enabled": True, "button": "Dub"}
    notes = pack_stage._voice_mix_notes(info)
    assert notes[:3] == ["", "## The voice track in the game", ""]
    body = notes[3]
    assert "lowers the music and sound effects to 30% while a line plays" in body and "press F8 in the game" in body and 'the "Dub" button' in body
    assert "delete that file and `vndub_voice_mix.rpyc` next to it" in body and "`key` (default `K_F8`)" in body


def test_voice_mix_rpy_without_a_sample_has_no_test_button() -> None:
    text = pack_stage.renpy_voice_mix_rpy(0.5, "K_F9", "", "en")
    assert "config.sample_voice" not in text and 'Play("voice"' not in text and "Test the voice" not in text
    assert 'textbutton "Close" action Hide("vndub_voice_panel")' in text and 'config.keymap["vndub_voice_panel"] = ["K_F9"]' in text
    assert 'hasattr(renpy.config, "always_shown_screens")' in text  # the store's config proxy raises on an unknown name

"""Ren'Py parser, round 18: a voice statement that survives ``show`` / ``play`` / ``$``, the voice statement on the record
(contract C1), the engine's narration characters, characters defined in ``init`` / ``python`` blocks, ``{p}`` / ruby
text, and music taken for voice in a game that never had any."""

from __future__ import annotations

from pathlib import Path

from engine_helpers import detect_and_parse
from vndub.engines import renpy


def _game(tmp_path: Path, script: str, files: tuple[str, ...] = ()) -> Path:
    g = tmp_path / "g"
    (g / "game").mkdir(parents=True)
    (g / "game" / "script.rpy").write_text(script, encoding="utf-8")
    for rel in files:
        p = g / "game" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"OggS")
    return g


def _parse(text: str) -> renpy.RpyParse:
    st = renpy.RpyParse()
    renpy.parse_rpy_text(text, "t.rpy", st, renpy.IdAllocator(), [None, None])
    return st


VOICE_SHOW = """define e = Character("Eileen")
label start:
    voice "voice/e_0001.ogg"
    show eileen happy
    e "Hello there."
    voice "voice/e_0002.ogg"
    play sound "door.ogg"
    $ x = 1
    scene bg room
    e "Second line."
    voice "voice/e_0003.ogg"
    e "Third line."
    e "No voice."
"""


def test_voice_survives_non_interacting_statements_until_the_next_say(tmp_path: Path, make_ctx) -> None:
    g = _game(tmp_path, VOICE_SHOW, ("voice/e_0001.ogg", "voice/e_0002.ogg", "voice/e_0003.ogg"))
    _, _, _, res = detect_and_parse(make_ctx(g))
    by_text = {l.text_src: l for l in res.lines}
    first, second, third, none = (by_text[t] for t in ("Hello there.", "Second line.", "Third line.", "No voice."))
    assert first.has_voice and first.voice_file == "voice/e_0001.ogg"
    assert second.has_voice and second.voice_file == "voice/e_0002.ogg"
    assert "voice_outside_group" in first.flags and "voice_outside_group" in second.flags
    assert third.has_voice and "voice_outside_group" not in third.flags
    assert not none.has_voice and "voice_stmt" not in (none.model_extra or {})
    # C1: the whole voice statement as the script wrote it
    assert first.model_extra["voice_stmt"] == 'voice "voice/e_0001.ogg"'
    assert third.model_extra["voice_stmt"] == 'voice "voice/e_0003.ogg"'


def test_voice_outside_group_keeps_the_ids_of_the_say_alone() -> None:
    st = _parse(VOICE_SHOW)
    says = [s for s in st.says if s["kind"] == "say"]
    alone = renpy.block_digest([renpy.say_code("e", "Hello there.")])
    grouped = renpy.block_digest(['voice "voice/e_0003.ogg"', renpy.say_code("e", "Third line.")])
    assert says[0]["id"] == f"start_{alone}"
    assert says[2]["id"] == f"start_{grouped}"


def test_voice_is_dropped_by_other_statements_and_by_another_block() -> None:
    st = _parse(
        'label start:\n    voice "a.ogg"\n    with dissolve\n    e "One."\n'
        '    voice "b.ogg"\n    show x\n    if y:\n        e "Two."\n    e "Three."\n'
    )
    says = [s for s in st.says if s["kind"] == "say"]
    assert [s["voice"] for s in says] == [None, None, None]


def test_voice_expression_and_sustain_keep_their_flags(tmp_path: Path, make_ctx) -> None:
    g = _game(
        tmp_path,
        'define e = Character("Eileen")\nlabel start:\n    voice "voice/a.ogg"\n    e "One."\n    voice sustain\n    show x\n    e "Two."\n'
        '    voice ("voice/" + n)\n    show y\n    e "Three."\n',
        ("voice/a.ogg",),
    )
    _, _, _, res = detect_and_parse(make_ctx(g))
    one, two, three = res.lines
    assert "sustain_head" in one.flags and "voice_sustain" in two.flags
    assert "voice_expression" in three.flags and three.model_extra["voice_stmt"] == 'voice ("voice/" + n)'


NARR = """define e = Character("Eileen")
label start:
    centered "Chapter One"
    vcentered "Later that day"
    narrator "Plain narrator."
    nvl_narrator "nvl narr"
    adv "adv line"
    e "Hi."
    extend " there."
    centered "Two"
    extend " more"
"""


def test_builtin_narration_characters_are_narration(tmp_path: Path, make_ctx) -> None:
    _, _, _, res = detect_and_parse(make_ctx(_game(tmp_path, NARR)))
    sids = [l.speaker_id for l in res.lines]
    assert sids == ["narrator"] * 5 + ["eileen", "eileen", "narrator", "narrator"]


def test_a_game_character_named_like_a_builtin_speaks(tmp_path: Path, make_ctx) -> None:
    _, _, _, res = detect_and_parse(make_ctx(_game(tmp_path, 'define centered = Character("Cent")\nlabel start:\n    centered "Hi."\n')))
    assert res.lines[0].speaker_id != "narrator" and res.lines[0].speaker_raw == "Cent"


CHARS = """init:
    $ e = Character('Eileen', color="#c8ffc8")
init python:
    m = Character("Mia")
python early:
    q = DynamicCharacter("q_name")
init -1:
    define k = Character("Ken")
translate russian python:
    e = Character("Эйлин")
$ z = Character("Zed")
label start:
    e "One."
    m "Two."
    k "Three."
    z "Four."
"""


def test_characters_defined_in_init_and_python_blocks_keep_their_names(tmp_path: Path, make_ctx) -> None:
    _, _, _, res = detect_and_parse(make_ctx(_game(tmp_path, CHARS)))
    assert [l.speaker_raw for l in res.lines] == ["Eileen", "Mia", "Ken", "Zed"]
    st = _parse(CHARS)
    assert st.chars["q"].dynamic and st.chars["e"].name == "Eileen"  # a translate block does not rename the source


def test_strip_text_tags_keeps_page_breaks_as_spaces_and_drops_ruby() -> None:
    assert renpy.strip_text_tags("So...{w=0.5} anyway.{p}New page") == "So... anyway. New page"
    assert renpy.strip_text_tags("A line\nwith newline") == "A line with newline"
    assert renpy.strip_text_tags("End. {p=1.0} Next") == "End. Next"
    assert renpy.strip_text_tags("{rb}東京{/rb}{rt}とうきょう{/rt}へ") == "東京へ"
    assert renpy.strip_text_tags("{{not a tag}") == "{not a tag}"


def test_music_is_not_voice_in_a_game_without_a_voice_folder(tmp_path: Path, make_ctx) -> None:
    g = _game(
        tmp_path,
        'define e = Character("Eileen")\ndefine audio.knock = "audio/door_knock.ogg"\nlabel start:\n    play music "audio/theme_song.ogg"\n'
        '    queue sound knock\n    voice "audio/e_line1.ogg"\n    e "Hi."\n',
        ("audio/theme_song.ogg", "audio/door_knock.ogg", "audio/bgm/calm.ogg", "gui/click.ogg", "audio/e_line1.ogg", "audio/stray.ogg"),
    )
    _, _, audio, res = detect_and_parse(make_ctx(g))
    assert [a.relpath for a in audio] == ["audio/e_line1.ogg"]
    assert res.lines[0].has_voice


def test_auto_voice_pattern_keeps_its_files_without_a_voice_folder(tmp_path: Path, make_ctx) -> None:
    g = _game(
        tmp_path,
        'define config.auto_voice = "vo/{id}.ogg"\ndefine e = Character("Eileen")\nlabel start:\n    play music "music.ogg"\n    e "Hi."\n',
        ("vo/start_aaaaaaaa.ogg", "music.ogg"),
    )
    _, _, audio, _ = detect_and_parse(make_ctx(g))
    assert [a.relpath for a in audio] == ["vo/start_aaaaaaaa.ogg"]


def test_gui_is_not_a_voice_dir() -> None:
    from vndub.engines.base import NON_VOICE_DIRS

    assert "gui" in NON_VOICE_DIRS

"""Ren'Py parser: unicode names, backslash-u escapes, ``kind=`` inheritance, auto_voice pairing, the tlid default by version.

Each test pins one way a line used to be dropped, mis-attributed or installed over the wrong file.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from engine_helpers import detect_and_parse
from vndub.engines import renpy

BS = chr(92)  # a backslash, spelled out so the escapes below stay readable


def _parse(text: str) -> renpy.RpyParse:
    st = renpy.RpyParse()
    renpy.parse_rpy_text(text, "t.rpy", st, renpy.IdAllocator(), [None, None])
    return st


def _says(st: renpy.RpyParse) -> list[dict]:
    return [s for s in st.says if s["kind"] == "say"]


def _md5(*codes: str) -> str:
    return hashlib.md5("".join(c + "\r\n" for c in codes).encode("utf-8")).hexdigest()[:8]


# ----------------------------------------------------------------------------
# unicode words (Lexer.word / image_name_component accept U+00A0..U+FFFD)
# ----------------------------------------------------------------------------


def test_cyrillic_and_cjk_names_and_image_attributes_are_statements() -> None:
    script = (
        'define Аня = Character("Аня", voice_tag="anya")\n'
        'define 我 = Character("我")\n'
        "label start:\n"
        '    Аня "Привет"\n'
        '    我 "你好"\n'
        '    e 微笑 "x"\n'
        '    Аня улыбка @ смех "Ещё"\n'
        "    call sub from вызов_1\n"
        '    Аня "После"\n'
    )
    st = _parse(script)
    assert st.chars["Аня"].name == "Аня" and st.chars["Аня"].voice_tag == "anya" and st.chars["我"].name == "我"
    says = _says(st)
    assert [(s["who"], s["what"]) for s in says] == [("Аня", "Привет"), ("我", "你好"), ("e", "x"), ("Аня", "Ещё"), ("Аня", "После")]
    assert says[0]["id"] == "start_" + _md5('Аня "Привет"')
    assert says[1]["id"] == "start_" + _md5('我 "你好"')
    assert says[2]["id"] == "start_" + _md5('e 微笑 "x"')  # the attribute is part of the code
    assert says[3]["id"] == "start_" + _md5('Аня улыбка @ смех "Ещё"')
    assert says[4]["id"] == "вызов_1_" + _md5('Аня "После"')  # a unicode ``from`` label moves the translation label


def test_unicode_speakers_reach_the_lines(tmp_path: Path, make_ctx) -> None:
    game = tmp_path / "g"
    (game / "game").mkdir(parents=True)
    (game / "game" / "script.rpy").write_text(
        'define Аня = Character("Аня", voice_tag="anya")\ndefine 我 = Character("我")\nlabel start:\n    Аня "Привет"\n    我 "你好"\n', encoding="utf-8"
    )
    _, _, _, res = detect_and_parse(make_ctx(game))
    assert [(l.speaker_raw, l.speaker_id, l.text_src) for l in res.lines][0] == ("Аня", "anya", "Привет")
    assert res.lines[1].speaker_raw == "我" and res.lines[1].speaker_id.startswith("u_")
    assert res.characters["anya"].names["src"] == "Аня"


# ----------------------------------------------------------------------------
# backslash-u without hex digits
# ----------------------------------------------------------------------------


def test_backslash_u_without_hex_digits_is_an_empty_replacement() -> None:
    assert renpy.dequote('"C:' + BS + "users" + BS + 'me"') == "C:sersme"  # the lexer's dequote returns None, re.sub drops it
    assert renpy.dequote('"' + BS + 'unlock"') == "nlock"
    assert renpy.dequote('"end' + BS + 'u"') == "end"
    assert renpy.dequote('"caf' + BS + "u00e9 " + BS + 'u41"') == "café A"  # one to four hex digits still make a character
    st = _parse('label start:\n    e "Path C:' + BS + "users" + BS + 'me"\n    e "Next."\n')  # one such string no longer aborts the parse
    assert [s["what"] for s in _says(st)] == ["Path C:sersme", "Next."]


# ----------------------------------------------------------------------------
# Character(kind=...)
# ----------------------------------------------------------------------------

KIND_SCRIPT = """
define e = Character("Eileen", voice_tag="eileen")
define e2 = Character(kind=e)
define e3 = Character("Eileen Alt", kind=e)
define e4 = Character(kind=e2)
define n = Character(None, kind=nvl)
define n2 = Character(kind=nvl)
define n3 = Character()
define p = Character(player_name)
define q = Character(kind=unknown_kind)
define loop_a = Character(kind=loop_b)
define loop_b = Character(kind=loop_a)
label start:
    e2 "A."
    e3 "B."
    e4 "C."
    n "D."
    n2 "E."
    n3 "F."
    p "G."
    q "H."
"""


def test_resolve_character_follows_the_kind_chain() -> None:
    chars = _parse(KIND_SCRIPT).chars
    r = {v: renpy.resolve_character(chars, v) for v in chars}
    assert r["e"] == ("Eileen", "eileen", True)
    assert r["e2"] == ("Eileen", "eileen", True) and r["e4"] == ("Eileen", "eileen", True)  # name and voice_tag come from kind
    assert r["e3"] == ("Eileen Alt", "eileen", True)  # an own name wins, the voice_tag is still inherited
    assert r["n"][2] is False and r["n2"][2] is False and r["n3"][2] is False  # Character(None), kind=nvl, no arguments: narration
    assert r["p"] == (None, None, True)  # a name we cannot read is still a name: the character speaks
    assert r["q"] == (None, None, True)
    assert r["loop_a"][2] is False  # a cycle ends


def test_kind_characters_are_dubbed_not_narration(tmp_path: Path, make_ctx) -> None:
    game = tmp_path / "g"
    (game / "game").mkdir(parents=True)
    (game / "game" / "script.rpy").write_text(KIND_SCRIPT, encoding="utf-8")
    _, _, _, res = detect_and_parse(make_ctx(game))
    got = [(l.speaker_raw, l.speaker_id) for l in res.lines]
    assert got == [
        ("Eileen", "eileen"), ("Eileen Alt", "eileen"), ("Eileen", "eileen"),  # A, B, C
        ("", "narrator"), ("", "narrator"), ("", "narrator"),  # D, E, F
        ("p", "p"), ("q", "q"),  # G, H: the variable's name is all we have
    ]
    assert set(res.characters) >= {"eileen", "p", "q"}


# ----------------------------------------------------------------------------
# auto_voice fallback: pair only what is unambiguous
# ----------------------------------------------------------------------------


def _auto_voice_game(tmp_path: Path, script_lines: str, files: list[str]) -> Path:
    game = tmp_path / "g"
    (game / "game" / "voice").mkdir(parents=True)
    (game / "game" / "script.rpy").write_text('define e = Character("E")\ndefine config.auto_voice = "voice/{id}.ogg"\n' + script_lines, encoding="utf-8")
    for n in files:
        (game / "game" / "voice" / f"{n}.ogg").write_bytes(b"OggS")
    return game


def test_auto_voice_fallback_pairs_only_the_last_file_and_line_of_a_label(tmp_path: Path, make_ctx) -> None:
    # ids we cannot reproduce: one unclaimed file, one line without a voice in the label -> unambiguous
    game = _auto_voice_game(tmp_path, 'label start:\n    e "Only."\n', ["start_00000001"])
    _, _, _, res = detect_and_parse(make_ctx(game))
    assert [l.voice_file for l in res.lines] == ["voice/start_00000001.ogg"] and "auto_voice_order" in res.lines[0].flags
    assert any("1 line(s) bound" in n for n in res.notes)


def test_auto_voice_fallback_leaves_several_files_to_link(tmp_path: Path, make_ctx) -> None:
    # two files named by hashes: their name order says nothing about the script, so neither line is bound by position
    game = _auto_voice_game(tmp_path, 'label start:\n    e "One."\n    e "Two."\nlabel next:\n    e "Alone."\n', ["start_00000002", "start_00000001", "next_00000009"])
    _, _, _, res = detect_and_parse(make_ctx(game))
    by_text = {l.text_src: l for l in res.lines}
    assert by_text["One."].voice_file is None and by_text["Two."].voice_file is None
    assert by_text["Alone."].voice_file == "voice/next_00000009.ogg"  # the other label has exactly one of each
    (warning,) = [n for n in res.notes if "NOT paired" in n]
    assert "start (2 unclaimed file(s), 2 line(s) without a voice)" in warning and "next" not in warning


def test_auto_voice_fallback_skips_lines_with_a_voice_statement_of_their_own(tmp_path: Path, make_ctx) -> None:
    game = _auto_voice_game(tmp_path, 'label start:\n    voice "voice/gone.ogg"\n    e "Own voice, file missing."\n', ["start_00000001"])
    _, _, _, res = detect_and_parse(make_ctx(game))
    (line,) = res.lines
    assert "voice_missing" in line.flags and "auto_voice_order" not in line.flags  # Ren'Py never asks auto_voice for it


# ----------------------------------------------------------------------------
# config.tlid_only_considers_say by the Ren'Py version of the game
# ----------------------------------------------------------------------------

DOCS_LINE = "Thank you for taking a look at the Ren'Py translation framework."
VOICED = 'label start:\n    voice "voice/a.ogg"\n    e "' + DOCS_LINE + '"\n'


def _versioned_game(tmp_path: Path, files: dict[str, bytes]) -> Path:
    game = tmp_path / "g"
    (game / "game").mkdir(parents=True)
    (game / "game" / "script.rpy").write_text('define e = Character("Eileen")\n' + VOICED, encoding="utf-8")
    for rel, data in files.items():
        (game / "renpy" / rel).parent.mkdir(parents=True, exist_ok=True)
        (game / "renpy" / rel).write_bytes(data)
    return game


def _first_id(res) -> str:
    return res.lines[0].dialogue_id  # type: ignore[attr-defined]


def test_detect_renpy_version_from_the_engine_folder(tmp_path: Path) -> None:
    g = tmp_path / "a"
    assert renpy.detect_renpy_version(g) is None
    (g / "renpy").mkdir(parents=True)
    (g / "renpy" / "__init__.py").write_text("version_tuple = (8, 3, 4, vc_version)\n", encoding="utf-8")
    assert renpy.detect_renpy_version(g) == (8, 3, 4)
    (g / "renpy" / "vc_version.py").write_text('branch = "main"\nversion = "8.6.0.26010100"\nversion_name = "x"\n', encoding="utf-8")
    assert renpy.detect_renpy_version(g) == (8, 6, 0)  # 8.5+ keeps the number in vc_version.py
    b = tmp_path / "b"
    (b / "renpy" / "__pycache__").mkdir(parents=True)
    (b / "renpy" / "__pycache__" / "vc_version.cpython-39.pyc").write_bytes(b"\x00\x01\xe3\x00" + b"8.5.4.25010101" + b"\x00")
    assert renpy.detect_renpy_version(b) == (8, 5, 4)  # compiled only


def test_ids_ignore_voice_statements_from_renpy_8_6(tmp_path: Path, make_ctx) -> None:
    game = _versioned_game(tmp_path, {"vc_version.py": b'version = "8.6.0.26010100"\n'})
    _, _, _, res = detect_and_parse(make_ctx(game))
    assert _first_id(res) == "start_636ae3f5"  # the say alone: the documentation example
    assert any("8.6.0" in n and "ignore" in n for n in res.notes)


def test_ids_ignore_voice_statements_from_renpy_8_5_1(tmp_path: Path, make_ctx) -> None:
    """Round 18, checked in the 8.5.3 SDK: `renpy translate` writes the id of the say alone for a voice + say pair."""
    game = _versioned_game(tmp_path, {"vc_version.py": b'version = "8.5.3.26051504"\n'})
    _, _, _, res = detect_and_parse(make_ctx(game))
    assert _first_id(res) == "start_636ae3f5"
    assert any("8.5.3" in n and "ignore" in n for n in res.notes)


def test_ids_include_voice_statements_before_renpy_8_6(tmp_path: Path, make_ctx) -> None:
    game = _versioned_game(tmp_path, {"__init__.py": b"version_tuple = (8, 3, 4, vc_version)\n"})
    _, _, _, res = detect_and_parse(make_ctx(game))
    assert _first_id(res) == "start_" + _md5('voice "voice/a.ogg"', 'e "' + DOCS_LINE + '"')
    assert any("8.3.4" in n and "include" in n for n in res.notes)


def test_unknown_renpy_version_keeps_voice_in_the_ids_and_says_so(tmp_path: Path, make_ctx) -> None:
    game = _versioned_game(tmp_path, {})
    _, _, _, res = detect_and_parse(make_ctx(game))
    assert _first_id(res) != "start_636ae3f5"
    assert any("version not found" in n and "tlid_only_considers_say" in n for n in res.notes)


def test_explicit_option_beats_the_detected_version(tmp_path: Path, make_ctx) -> None:
    game = _versioned_game(tmp_path, {"vc_version.py": b'version = "8.6.0.26010100"\n'})
    _, _, _, res = detect_and_parse(make_ctx(game, sets={"parse.engine_opts.renpy.tlid_only_considers_say": False}))
    assert _first_id(res) == "start_" + _md5('voice "voice/a.ogg"', 'e "' + DOCS_LINE + '"')
    assert not any("found in the game folder" in n for n in res.notes)
    game2 = _versioned_game(tmp_path / "two", {"__init__.py": b"version_tuple = (8, 3, 4, vc_version)\n"})
    _, _, _, res2 = detect_and_parse(make_ctx(game2, sets={"parse.engine_opts.renpy.tlid_only_considers_say": True}))
    assert _first_id(res2) == "start_636ae3f5"

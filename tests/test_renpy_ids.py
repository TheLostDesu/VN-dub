"""Ren'Py dialogue identifiers (config.auto_voice) and the .rpy parser.

Verified examples:
* Ren'Py documentation (translation.html): under ``label start``,
  ``e "Thank you for taking a look at the Ren'Py translation framework."``
  has the identifier ``start_636ae3f5``.
* Collisions: identical statements under one label get ``_1``, ``_2`` suffixes
  (``Translator.unique_identifier``).
* ``voice`` statements are translatable and share the say statement's block, so
  they change the digest (the reason Ren'Py 8.5.1 added
  ``config.tlid_only_considers_say``).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from vndub.engines import renpy


def test_docs_example_identifier() -> None:
    code = renpy.say_code("e", "Thank you for taking a look at the Ren'Py translation framework.")
    assert code == 'e "Thank you for taking a look at the Ren\'Py translation framework."'
    assert renpy.dialogue_identifier("start", [code]) == "start_636ae3f5"


def test_encode_say_string() -> None:
    assert renpy.encode_say_string('say "hi"') == '"say \\"hi\\""'
    assert renpy.encode_say_string("a\\b") == '"a\\\\b"'
    assert renpy.encode_say_string("line1\nline2") == '"line1\\nline2"'
    assert renpy.encode_say_string("a  b") == '"a \\ b"'  # a space after a space is escaped


def test_dequote_matches_lexer() -> None:
    assert renpy.dequote('"He said \\"hi\\""') == 'He said "hi"'
    assert renpy.dequote("'single \"quoted\"'") == 'single "quoted"'
    assert renpy.dequote('"a   b"') == "a b"  # whitespace runs collapse
    assert renpy.dequote('"x\\ny"') == "x\ny"
    assert renpy.dequote('"100\\%"') == "100%%"


def test_collision_suffix_and_label_dots() -> None:
    alloc = renpy.IdAllocator()
    code = renpy.say_code("e", "Hello.")
    d = renpy.block_digest([code])
    assert alloc.allocate("start", d) == f"start_{d}"
    assert alloc.allocate("start", d) == f"start_{d}_1"
    assert alloc.allocate("start", d) == f"start_{d}_2"
    assert alloc.allocate("chapter1.local", d) == f"chapter1_local_{d}"
    assert alloc.allocate(None, d) == d


def test_say_code_variants() -> None:
    assert renpy.say_code("e", "Hi", ["happy"], ["sad"], interact=False, explicit_id="x1") == 'e happy @ sad "Hi" nointeract id x1'
    assert renpy.say_code(None, "narration") == '"narration"'


def _parse(text: str) -> renpy.RpyParse:
    st = renpy.RpyParse()
    renpy.parse_rpy_text(text, "t.rpy", st, renpy.IdAllocator(), [None, None])
    return st


SCRIPT = '''
define e = Character("Eileen", voice_tag="eileen")
define n = Character(None, kind=nvl)
define t = Character(_("トモ"), color="#fff")
define config.auto_voice = "voice/{id}.ogg"

label start:
    e "Thank you for taking a look at the Ren'Py translation framework."
    e happy "Second line."
    "Narration here."
    voice "voice/explicit.ogg"
    e "Voiced explicitly."
    extend " And more."
    t "日本語の台詞。" (multiple=2)
label .sub:
    e "In a local label."
label _hidden:
    e "Hidden label keeps the previous translation label."
init python:
    x = 1
    e("not dialogue")
'''


def test_parser_statements() -> None:
    st = _parse(SCRIPT)
    assert st.auto_voice == "voice/{id}.ogg"
    assert st.chars["e"].name == "Eileen" and st.chars["e"].voice_tag == "eileen"
    assert st.chars["n"].name is None and st.chars["n"].kind == "nvl"
    assert st.chars["t"].name == "トモ"
    says = [s for s in st.says if s["kind"] == "say"]
    assert [s["what"] for s in says] == [
        "Thank you for taking a look at the Ren'Py translation framework.",
        "Second line.",
        "Narration here.",
        "Voiced explicitly.",
        " And more.",
        "日本語の台詞。",
        "In a local label.",
        "Hidden label keeps the previous translation label.",
    ]
    assert says[0]["id"] == "start_636ae3f5"
    assert says[1]["id"] == "start_" + hashlib.md5(('e happy "Second line."' + "\r\n").encode()).hexdigest()[:8]
    # voice statement is part of the translate block
    voiced = says[3]
    assert voiced["voice"] == "voice/explicit.ogg"
    exp = hashlib.md5(('voice "voice/explicit.ogg"\r\n' + 'e "Voiced explicitly."\r\n').encode()).hexdigest()[:8]
    assert voiced["id"] == f"start_{exp}"
    assert says[4]["who"] == "extend"
    assert says[5]["id"].startswith("start_")
    assert says[6]["id"].startswith("start_sub_")
    assert says[7]["id"].startswith("start_sub_")  # _hidden does not change the label


def test_only_say_option_ignores_voice() -> None:
    st = renpy.RpyParse()
    renpy.parse_rpy_text('label start:\n    voice "v.ogg"\n    e "Thank you for taking a look at the Ren\'Py translation framework."\n', "x.rpy", st, renpy.IdAllocator(), [None, None], only_say=True)
    assert [s["id"] for s in st.says if s["kind"] == "say"] == ["start_636ae3f5"]


def test_auto_voice_binding_end_to_end(tmp_path: Path, make_ctx) -> None:
    game = tmp_path / "g" / "game"
    (game / "voice").mkdir(parents=True)
    script = 'define e = Character("Eileen", voice_tag="eileen")\ndefine config.auto_voice = "voice/{id}.ogg"\nlabel start:\n    e "Thank you for taking a look at the Ren\'Py translation framework."\n    e "Unvoiced line."\n'
    (game / "script.rpy").write_text(script, encoding="utf-8")
    (game / "voice" / "start_636ae3f5.ogg").write_bytes(b"OggS")
    from engine_helpers import detect_and_parse

    ctx = make_ctx(tmp_path / "g")
    _, _, _, res = detect_and_parse(ctx)
    l1, l2 = [l for l in res.lines]
    assert l1.has_voice and l1.voice_file == "voice/start_636ae3f5.ogg" and "auto_voice" in l1.flags
    assert not l2.has_voice


def test_dialogue_tab_overrides_identifier(tmp_path: Path, make_ctx) -> None:
    game = tmp_path / "g" / "game"
    (game / "voice").mkdir(parents=True)
    (game / "script.rpy").write_text('define e = Character("E")\ndefine config.auto_voice = "voice/{id}.ogg"\nlabel start:\n    e "Line."\n', encoding="utf-8")
    (game / "voice" / "start_deadbeef.ogg").write_bytes(b"OggS")
    tab = tmp_path / "dialogue.tab"
    tab.write_text("Identifier\tCharacter\tDialogue\tFilename\tLine Number\nstart_deadbeef\te\tLine.\tgame/script.rpy\t4\n", encoding="utf-8")
    from engine_helpers import detect_and_parse

    ctx = make_ctx(tmp_path / "g", sets={"parse.engine_opts.renpy.dialogue_tab": str(tab)})
    _, _, _, res = detect_and_parse(ctx)
    (l,) = res.lines
    assert l.voice_file == "voice/start_deadbeef.ogg"
    assert "dialogue_tab_id" in l.flags


def test_auto_voice_order_fallback(tmp_path: Path, make_ctx) -> None:
    game = tmp_path / "g" / "game"
    (game / "voice").mkdir(parents=True)
    (game / "script.rpy").write_text('define e = Character("E")\ndefine config.auto_voice = "voice/{id}.ogg"\nlabel start:\n    e "One."\n    e "Two."\n', encoding="utf-8")
    for n in ("start_00000001", "start_00000002"):  # ids we cannot reproduce
        (game / "voice" / f"{n}.ogg").write_bytes(b"OggS")
    from engine_helpers import detect_and_parse

    _, _, _, res = detect_and_parse(make_ctx(tmp_path / "g"))
    assert [l.voice_file for l in res.lines] == ["voice/start_00000001.ogg", "voice/start_00000002.ogg"]
    assert all("auto_voice_order" in l.flags for l in res.lines)
    assert any("fallback" in n for n in res.notes)

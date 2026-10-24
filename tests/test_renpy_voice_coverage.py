"""Route P on a Ren'Py game: every dubbed line gets exactly one voice file that Ren'Py will ask for, and the patch says
which lines it holds, which are provisional and which it leaves out.

The user's report: in the game one line speaks and the next ones stay silent. Two causes are pinned here:

* ``master.include_review_lines`` used to be ``skip``: with most lines of a route-P project in review (the TTS + ASR pair
  fails the CER gate), the patch held a third of the files and said nothing about the rest;
* the dialogue identifiers of statements after the first in a label must be the ones Ren'Py computes. The reference
  below is a TRANSLITERATION of ``renpy/translation/__init__.py`` (``Restructurer.callback`` / ``create_translate`` /
  ``unique_identifier``, Ren'Py master) over a hand-written statement tree; the statement trees and their ``get_code``
  strings are written by hand from ``renpy/parser.py`` / ``renpy/ast.py``, independently of ``vndub.engines.renpy``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from tests.conftest import requires_ffmpeg
from tests.engine_helpers import detect_and_parse
from tests.genproj import speechlike
from vndub.engines import renpy
from vndub.manifest.io import FinalIndexRecord, LineRecord, SelectionRecord, read_jsonl, save_characters, write_jsonl
from vndub.stages import master
from vndub.stages.common import run_stage

# ----------------------------------------------------------------------------
# reference: renpy.translation.Restructurer, transliterated
# ----------------------------------------------------------------------------


class Node:
    kind = "plain"
    translatable = False
    hide = False
    code = ""
    blocks: list[list["Node"]] = []

    def __init__(self, *blocks: list["Node"]) -> None:
        self.blocks = list(blocks)


class Plain(Node):
    pass


class Voice(Node):  # ``voice "x"`` is registered translatable: it joins the next say's translate block
    translatable = True

    def __init__(self, code: str) -> None:
        super().__init__()
        self.code = code


class Say(Node):
    kind = "say"

    def __init__(self, who: str | None, code: str) -> None:
        super().__init__()
        self.who, self.code = who, code  # ``who``: a tag for the test (None = narration); ``code``: Say.get_code() by hand


class Label(Node):
    kind = "label"

    def __init__(self, name: str, *block: "Node", hide: bool = False) -> None:
        super().__init__(list(block))
        self.name, self.hide = name, hide


class Menu(Node):  # Menu.restructure: callback(block) for every choice block
    pass


class If(Node):  # If.restructure: callback(block) for every entry
    pass


class Restructurer:
    def __init__(self) -> None:
        self.label: str | None = None
        self.alternate: str | None = None
        self.identifiers: set[str] = set()
        self.said: list[tuple[str | None, str]] = []  # (who tag, identifier) in the order the translate blocks are made

    def unique_identifier(self, label: str | None, digest: str) -> str:
        base = digest if label is None else label.replace(".", "_") + "_" + digest
        i, suffix = 0, ""
        while base + suffix in self.identifiers:
            i += 1
            suffix = f"_{i}"
        return base + suffix

    def create_translate(self, block: list[Node]) -> None:
        md5 = hashlib.md5()
        for i in block:
            md5.update((i.code + "\r\n").encode("utf-8"))
        digest = md5.hexdigest()[:8]
        identifier = self.unique_identifier(self.label, digest)
        alternate = self.unique_identifier(self.alternate, digest) if self.alternate is not None else None
        self.identifiers.add(identifier)
        if alternate is not None:
            self.identifiers.add(alternate)
        for i in block:
            if isinstance(i, Say):
                self.said.append((i.who, identifier))

    def callback(self, children: list[Node]) -> None:
        group: list[Node] = []
        for i in children:
            if isinstance(i, Label) and not i.hide:
                if i.name.startswith("_"):
                    self.alternate = i.name
                else:
                    self.label, self.alternate = i.name, None
            for block in i.blocks:  # i.restructure(self.callback)
                self.callback(block)
            if isinstance(i, Say):
                group.append(i)
                self.create_translate(group)
                group = []
            elif i.translatable:
                group.append(i)
            else:
                if group:
                    self.create_translate(group)
                    group = []
        if group:
            self.create_translate(group)


def reference_ids(tree: list[Node]) -> list[tuple[str | None, str]]:
    r = Restructurer()  # one per file
    r.callback(tree)
    return r.said


def parsed_ids(script: str, only_say: bool = False) -> list[str]:
    st = renpy.RpyParse()
    renpy.parse_rpy_text(script, "t.rpy", st, renpy.IdAllocator(), [None, None], only_say)
    return [s["id"] for s in st.says if s["kind"] == "say" and s["id"]]


# ----------------------------------------------------------------------------
# a route-P script and its statement tree
# ----------------------------------------------------------------------------

SCRIPT = '''
define s = Character("Sakuya", voice_tag="sakuya")
define r = Character("Remilia")

label start:
    scene bg room
    "The night was quiet."
    s "Good evening, master."
    s "Did you sleep well?"
    s happy "I made tea."
    extend " Would you like some?"
    s "Wait{w} for it{p}here it is."
    "Flandre" "A string speaker."
    s "Good evening, master."
    s "Ouch!" with vpunch
    s """
    First paragraph.

    Second # not a comment
    paragraph.
    """
    menu choices1:
        s "Which way?"
        "Left":
            s "Left it is."
        "Right" if lucky:
            r "Right it is."
            s "Good evening, master."
    call sub from _call_sub_1
    s "Back again."
    call sub2 from after_sub2
    r "After the named call."
    if lucky:
        s "Lucky."
    else:
        s "Unlucky."
    $ x = 1
    s "Last one." nointeract
label .local:
    s "In a local label."
label _helper:
    s "In a hidden label."
label other hide:
    s "Hidden too."
'''


def script_tree() -> list[Node]:
    return [
        Label(
            "start",
            Plain(),  # scene
            Say(None, '"The night was quiet."'),
            Say("s", 's "Good evening, master."'),
            Say("s", 's "Did you sleep well?"'),
            Say("s", 's happy "I made tea."'),
            Say("s", 'extend " Would you like some?"'),
            Say("s", 's "Wait{w} for it{p}here it is."'),
            Say("flandre", '"Flandre" "A string speaker."'),
            Say("s", 's "Good evening, master."'),  # the same statement again: _1
            Say("s", 's "Ouch!" with vpunch'),
            Say("s", 's "First paragraph."'),  # monologue mode: one Say per paragraph
            Say("s", 's "Second # not a comment paragraph."'),
            Label("choices1"),  # ``menu choices1:`` makes a Label node
            Say("s", 's "Which way?" nointeract'),  # a menu's own say is a nointeract one
            Menu(
                [Say("s", 's "Left it is."')],
                [Say("r", 'r "Right it is."'), Say("s", 's "Good evening, master."')],  # under another label: no suffix
            ),
            Plain(),  # call
            Label("_call_sub_1"),  # ``call ... from _x`` makes a Label too, but a "_" one only sets the alternate label
            Plain(),  # pass
            Say("s", 's "Back again."'),
            Plain(),
            Label("after_sub2"),  # a "from" label without "_" does move the translation label
            Plain(),
            Say("r", 'r "After the named call."'),
            If([Say("s", 's "Lucky."')], [Say("s", 's "Unlucky."')]),
            Plain(),  # $ x = 1
            Say("s", 's "Last one." nointeract'),
        ),
        Label("choices1.local", Say("s", 's "In a local label."')),  # ``menu choices1:`` declared the global label, so ".local" hangs on it
        Label("_helper", Say("s", 's "In a hidden label."')),
        Label("other", Say("s", 's "Hidden too."'), hide=True),
    ]


def test_every_statement_gets_the_identifier_renpy_computes():
    ref = reference_ids(script_tree())
    got = parsed_ids(SCRIPT)
    assert got == [i for _, i in ref]
    assert len(set(got)) == len(got)  # every statement has its own, none shared
    # a few of them spelled out, so a drift of the reference itself shows up
    assert got[0].startswith("start_") and got[7].endswith("_1")  # the repeated line
    assert got[13].startswith("choices1_") and got[14].startswith("choices1_") and not got[14].endswith("_1")
    assert any(i.startswith("after_sub2_") for i in got) and any(i.startswith("choices1_local_") for i in got)
    assert not any(i.startswith("_call") or i.startswith("_helper") or i.startswith("other_") for i in got)


def test_voice_statement_joins_the_following_say_only():
    script = 'label start:\n    voice "a.ogg"\n    e "One."\n    e "Two."\n    voice sustain\n    e "Three."\n'
    tree = [Label("start", Voice('voice "a.ogg"'), Say("e", 'e "One."'), Say("e", 'e "Two."'), Voice("voice sustain"), Say("e", 'e "Three."'))]
    assert parsed_ids(script) == [i for _, i in reference_ids(tree)]


def test_collision_suffixes_and_labels_start_over_in_every_file(tmp_path: Path):
    a, b = tmp_path / "a.rpy", tmp_path / "b.rpy"
    for f in (a, b):  # a say before any label (valid, unreachable) and a repeat: per file, not per game
        f.write_text('e "Hi."\ne "Hi."\n', encoding="utf-8")
    st = renpy.parse_rpy_files([a, b])
    ids = [s["id"] for s in st.says if s["kind"] == "say"]
    d = renpy.block_digest(['e "Hi."'])
    assert ids == [d, f"{d}_1", d, f"{d}_1"]
    ref = reference_ids([Say("e", 'e "Hi."'), Say("e", 'e "Hi."')])
    assert [i for _, i in ref] == ids[:2]


def test_menu_caption_is_not_a_say_statement():
    script = 'label start:\n    menu:\n        "Only a caption"\n        "Go":\n            e "Went."\n    e "Only a caption"\n'
    st = renpy.RpyParse()
    renpy.parse_rpy_text(script, "t.rpy", st, renpy.IdAllocator(), [None, None])
    says = [s for s in st.says if s["kind"] == "say"]
    assert [bool(s.get("caption")) for s in says] == [True, False, False]
    assert says[0]["id"] == ""
    tree = [Label("start", Menu([Say("e", 'e "Went."')]), Say("e", 'e "Only a caption"'))]  # the caption is a menu item, not a Say node
    assert [s["id"] for s in says[1:]] == [i for _, i in reference_ids(tree)]
    assert not says[2]["id"].endswith("_1")  # the caption did not take the narration's identifier


def test_triple_quoted_paragraph_lines_and_clauses():
    script = 'label start:\n    e """\n    One\n    two.\n\n\n    Three.\n    """ with dissolve\n'
    st = renpy.RpyParse()
    renpy.parse_rpy_text(script, "t.rpy", st, renpy.IdAllocator(), [None, None])
    says = [s for s in st.says if s["kind"] == "say"]
    assert [s["what"] for s in says] == ["One two.", "Three."]
    assert [s["line"] for s in says] == [3, 7]  # the launcher's dialogue.tab lists each paragraph at its own line
    tree = [Label("start", Say("e", 'e "One two." with dissolve'), Say("e", 'e "Three." with dissolve'))]
    assert [s["id"] for s in says] == [i for _, i in reference_ids(tree)]


def test_arguments_are_normalized_like_argumentinfo_get_code():
    got = parsed_ids('label start:\n    e "Hi." ( multiple = 2 ,  x )\n')
    assert got == [i for _, i in reference_ids([Label("start", Say("e", 'e "Hi." (multiple=2, x)'))])]


def test_comment_hash_and_apostrophes_in_python_do_not_swallow_dialogue():
    script = (
        "init python:\n    # it's a comment with an apostrophe\n    x = 'it\\'s'\n"
        "label start:\n    e \"One.\"  # trailing comment\n    e \"Two # is text.\"\n"
    )
    st = renpy.RpyParse()
    renpy.parse_rpy_text(script, "t.rpy", st, renpy.IdAllocator(), [None, None])
    assert [s["what"] for s in st.says if s["kind"] == "say"] == ["One.", "Two # is text."]


# ----------------------------------------------------------------------------
# the whole route: parse -> master -> pack
# ----------------------------------------------------------------------------


def _route_p_project(make_ctx: Any, tmp_path: Path, **sets: Any) -> tuple[Any, list[Any]]:
    game = tmp_path / "gamedir"
    (game / "game").mkdir(parents=True)
    (game / "game" / "script.rpy").write_text(SCRIPT, encoding="utf-8")
    ctx = make_ctx(game, sets={"synth.voice_unvoiced": True, **sets})
    _, _, _, res = detect_and_parse(ctx)
    write_jsonl(ctx.paths.lines, res.lines)
    save_characters(ctx.paths.characters, res.characters)
    return ctx, res.lines


def _select(ctx: Any, statuses: list[str], no_winner_last: bool = True) -> list[Any]:
    """Selection records for the dubbed lines: statuses cycle; the last line has no winner (nothing could be synthesized)."""
    dub = ctx.dub_lines()
    sel = []
    for i, line in enumerate(dub):
        if no_winner_last and i == len(dub) - 1:
            sel.append(SelectionRecord(line_id=line.line_id, status="review", reason="no candidates"))
            continue
        cand = ctx.paths.synth_dir(line.line_id) / "mock_tts_P_0.wav"
        speechlike(cand, 1.2 + 0.05 * (i % 5), 190.0, sr=24000, seed=300 + i, amp=0.2)
        status = statuses[i % len(statuses)]
        reason = {"auto": "best of 3 passing", "review": "no candidate passes the gates after 2 regen round(s) (best fails: cer 0.22 > 0.10)", "regen": "no candidate passes the gates"}[status]
        sel.append(SelectionRecord(line_id=line.line_id, winner=ctx.paths.rel(cand), backend="qwen3_tts", route="P", status=status, reason=reason, scores={"composite": 0.6}))
    write_jsonl(ctx.paths.selection, sel)
    return sel


def _manifest(ctx: Any) -> dict[str, Any]:
    return json.loads((ctx.paths.out_engine("renpy") / "manifest_patch.json").read_text(encoding="utf-8"))


@requires_ffmpeg
def test_route_p_every_mastered_line_has_one_file_named_by_the_id_renpy_asks_for(make_ctx, tmp_path):
    ctx, lines = _route_p_project(make_ctx, tmp_path)
    expected = [i for who, i in reference_ids(script_tree()) if who is not None]  # narration is not dubbed
    dub = ctx.dub_lines()
    assert [l.model_extra["dialogue_id"] for l in dub] == expected  # script order, the ids Ren'Py computes
    assert len(expected) == len(set(expected)) == 22
    sel = _select(ctx, ["auto", "review", "regen", "auto"])
    mastered = [l for l, s in zip(dub, sel) if s.winner]

    m = run_stage("master", ctx)
    assert m["encoded"] == len(mastered) == 21 and "errors" not in m
    assert m["skipped"] == {"no winner": 1}
    n_prov = sum(1 for s in sel if s.winner and s.status != "auto")
    assert m["provisional"] == n_prov and m["provisional_regen"] == sum(1 for s in sel if s.winner and s.status == "regen")
    finals = {f.line_id: f for f in read_jsonl(ctx.paths.final_index, FinalIndexRecord)}
    assert {f.status for f in finals.values()} == {"auto", "provisional"}
    prov = next(f for f in finals.values() if f.status == "provisional")
    assert prov.model_extra["selection_status"] in ("review", "regen") and prov.model_extra["reason"]

    p = run_stage("pack", ctx)
    out = ctx.paths.out_engine("renpy") / "game"
    files = sorted(f.name for f in (out / "voice_dub").iterdir())
    assert files == sorted(f"{l.model_extra['dialogue_id']}.ogg" for l in mastered)  # one file per mastered line, no more, no less
    assert p["unvoiced"] == 21 and p["provisional"] == n_prov and p["left_out"] == 1 and "unnamed" not in p

    # the generated rpy is valid Ren'Py (an init python block) and its pattern resolves to those files
    rpy = (out / "vndub_auto_voice.rpy").read_text(encoding="utf-8")
    block = rpy.split("init 1000 python:\n", 1)[1]
    code = "\n".join(l[4:] for l in block.splitlines())
    class _Cfg:  # config.* assignments of the block
        pass

    ns: dict[str, Any] = {"config": _Cfg()}
    exec(compile(ast.parse(code), "vndub_auto_voice.rpy", "exec"), ns)  # noqa: S102 - the file under test, trivial assignments
    assert ns["config"].has_voice is True
    assert ns["config"].auto_voice.format(id=expected[0]) == f"voice_dub/{expected[0]}.ogg"
    assert f"{expected[0]}.ogg" in files

    # what the patch says about itself
    man = _manifest(ctx)
    cov = man["coverage"]
    assert cov["dub_lines"] == 22 and cov["installed"] == 21 and cov["provisional"] == n_prov and cov["left_out"] == 1
    assert cov["left_out_reasons"] == {"no winner": 1}
    assert cov["left_out_lines"][0]["line_id"] == dub[-1].line_id and cov["narration"] == 1
    assert sum(1 for f in man["files"] if f["status"] == "provisional") == n_prov
    assert all(f["reason"] for f in man["files"] if f["status"] == "provisional")
    text = (ctx.paths.out_engine("renpy") / "INSTALL.md").read_text(encoding="utf-8")
    assert "What this patch covers" in text and "PROVISIONAL" in text and "LEFT OUT" in text and "narration" in text
    run_stage("pack", ctx)  # a second pack rebuilds INSTALL.md, the sections do not pile up
    text2 = (ctx.paths.out_engine("renpy") / "INSTALL.md").read_text(encoding="utf-8")
    assert text2.count("What this patch covers") == 1 and text2.count("Lines without original voice") == 1
    assert re.findall(r"(\d+) line\(s\) are dubbed; (\d+) have", text2) == [("22", "21")]


@requires_ffmpeg
def test_skip_mode_leaves_review_lines_out_and_says_so(make_ctx, tmp_path):
    ctx, _ = _route_p_project(make_ctx, tmp_path, **{"master.include_review_lines": "skip"})
    sel = _select(ctx, ["auto", "review", "regen", "auto"], no_winner_last=False)
    n_review = sum(1 for s in sel if s.status == "review")
    n_regen = sum(1 for s in sel if s.status == "regen")
    m = run_stage("master", ctx)
    assert m["skipped"] == {"review": n_review, "regen": n_regen} and "provisional" not in m
    assert "LEFT OUT" in ctx.paths.log("master").read_text(encoding="utf-8")  # the stage log says it too
    p = run_stage("pack", ctx)
    assert p["left_out"] == n_review + n_regen
    cov = _manifest(ctx)["coverage"]
    assert cov["left_out_reasons"] == {"review": n_review, "regen": n_regen} and cov["provisional"] == 0
    assert all(x["reason"] for x in cov["left_out_lines"]) and cov["include_review_lines"] == "skip"
    text = (ctx.paths.out_engine("renpy") / "INSTALL.md").read_text(encoding="utf-8")
    assert "LEFT OUT" in text and "master.include_review_lines" in text and "provisional" in text
    out = ctx.paths.out_engine("renpy") / "game" / "voice_dub"
    assert len(list(out.iterdir())) == len(sel) - n_review - n_regen


def test_review_and_regen_lines_are_installed_by_default():
    from vndub.config import PipelineConfig, load_config

    assert PipelineConfig().master.include_review_lines == "provisional"
    repo = Path(__file__).resolve().parent.parent
    for name in ("pipeline.yaml", "pipeline.test.yaml", "pipeline.apptest.yaml"):
        assert load_config(repo / "configs" / name).master.include_review_lines == "provisional", name


def test_unvoiced_file_name_keeps_a_non_ascii_label():
    line = LineRecord(line_id="s01_000001", scene="s01", order=1, speaker_id="s", dialogue_id="сцена_1_636ae3f5")
    assert master.unvoiced_name(line, "vorbis") == "unvoiced/сцена_1_636ae3f5.ogg"
    assert master.unvoiced_name(LineRecord(line_id="a b", scene="s", order=1), None) == "unvoiced/a_b.ogg"

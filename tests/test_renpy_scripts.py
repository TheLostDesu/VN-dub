"""Ren'Py extract: which script files are read and what the records call them.

* the engine-folder filter judged the ABSOLUTE path, so everything unrpa / unrpyc produced (``work/raw/renpy/...``), a game
  installed under a folder called ``renpy`` and every ``00*.rpy`` of the game itself vanished without a word;
* unrpyc output was flat (``decompiled/<name>``): ``game/ch1/script.rpyc`` and ``game/ch2/script.rpyc`` overwrote each other,
  a stale ``.rpy`` counted as success, and the records carried only the bare file name, which collides in the
  (file, line) lookup of the launcher's dialogue.tab.

unrpa / unrpyc are stubbed on the context's tool runner: the "archive" holds the files the stub writes, a ".rpyc" is a text
file that the stub copies to the ``.rpy`` next to it (what a decompiler does with the script it finds).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from engine_helpers import detect_and_parse
from vndub.engines import renpy


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _done(args: list[str], code: int = 0, err: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args, code, "", err)


def stub_tools(ctx: Any, archive: dict[str, str] | None = None, fail: tuple[str, ...] = ()) -> list[tuple[str, list[str]]]:
    """Make ``ctx.tools`` find unrpa / unrpyc and run the stubs below; returns the list of calls (tool, args)."""
    calls: list[tuple[str, list[str]]] = []

    def resolve(name: str) -> Path | None:
        return Path(name) if name in ("unrpa", "unrpyc") else None

    def run(name: str, args: list[str], cwd: Any = None, timeout: Any = None, check: bool = True) -> subprocess.CompletedProcess[str]:
        calls.append((name, list(args)))
        if name == "unrpa":  # unrpa -mp <dest> <archive>
            dest = Path(args[args.index("-mp") + 1])
            for rel, text in (archive or {}).items():
                _write(dest / rel, text)
            return _done(args)
        src = Path(next(a for a in args if not a.startswith("-")))  # unrpyc [--clobber] <file.rpyc>
        if src.name in fail or src.parent.name in fail:
            return _done(args, 1, "boom")
        rpy = src.with_suffix(".rpy")
        if rpy.exists() and "--clobber" not in args and "-c" not in args:
            return _done(args)  # unrpyc leaves an existing output alone without --clobber and still exits 0
        rpy.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        return _done(args)

    ctx.tools.resolve = resolve  # type: ignore[method-assign]
    ctx.tools.run = run  # type: ignore[method-assign]
    return calls


def _names(ctx: Any, ex: Any) -> list[str]:
    roots = [*ex.all_roots(), ctx.paths.raw / "renpy" / "decompiled"]
    return sorted(renpy.script_name(p, roots) for p in ex.script_files)


def test_archive_scripts_game_folder_named_renpy_and_00_files_are_all_read(tmp_path: Path, make_ctx) -> None:
    game = tmp_path / "renpy" / "MyGame"  # the game is installed under a folder called "renpy"
    _write(game / "game" / "script.rpy", 'label start:\n    e "From the loose script."\n')
    _write(game / "game" / "00_defines.rpy", 'define e = Character("Eileen")\n')  # a game file, not the engine's
    _write(game / "game" / "001_prologue.rpy", 'label prologue:\n    e "From 001."\n')
    (game / "game" / "scripts.rpa").write_bytes(b"RPA-3.0 stub")
    _write(game / "game" / "renpy" / "common" / "00console.rpy", 'label engine_dump:\n    e "engine"\n')  # the engine folder dumped under game/
    _write(game / "renpy" / "common" / "00nvl_mode.rpy", 'label engine:\n    e "engine too"\n')  # <game dir>/renpy: the engine, never scanned
    ctx = make_ctx(game)
    stub_tools(
        ctx,
        archive={"ch2/extra.rpy": 'label extra:\n    e "From the archive."\n', "ch2/compiled.rpyc": 'label compiled:\n    e "From a compiled script."\n'},
    )
    _, ex, _, res = detect_and_parse(ctx)
    assert _names(ctx, ex) == sorted(["00_defines.rpy", "001_prologue.rpy", "ch2/compiled.rpy", "ch2/extra.rpy", "script.rpy"])
    assert sorted(l.text_src for l in res.lines) == ["From 001.", "From a compiled script.", "From the archive.", "From the loose script."]
    assert {l.speaker_id for l in res.lines} == {"eileen"}  # the character defined in 00_defines.rpy is known
    assert not ex.warnings


def test_same_named_compiled_scripts_in_two_folders_stay_apart(tmp_path: Path, make_ctx) -> None:
    game = tmp_path / "game_dir"
    _write(game / "game" / "defs.rpy", 'define e = Character("Eileen")\n')
    _write(game / "game" / "ch1" / "script.rpyc", 'label ch1:\n    e "Chapter one."\n')
    _write(game / "game" / "ch2" / "script.rpyc", 'label ch2:\n    e "Chapter two."\n')
    ctx = make_ctx(game)
    calls = stub_tools(ctx)
    _, ex, _, res = detect_and_parse(ctx)
    dec = ctx.paths.raw / "renpy" / "decompiled"
    assert (dec / "ch1" / "script.rpy").read_text(encoding="utf-8").endswith('"Chapter one."\n')
    assert (dec / "ch2" / "script.rpy").read_text(encoding="utf-8").endswith('"Chapter two."\n')
    assert _names(ctx, ex) == ["ch1/script.rpy", "ch2/script.rpy", "defs.rpy"]
    unrpyc = [a for n, a in calls if n == "unrpyc"]
    assert len(unrpyc) == 2 and all(a[0] == "--clobber" for a in unrpyc)
    assert [(l.text_src, l.source_file) for l in res.lines] == [("Chapter one.", "ch1/script.rpy"), ("Chapter two.", "ch2/script.rpy")]


def test_a_stale_decompiled_script_is_not_a_success(tmp_path: Path, make_ctx) -> None:
    game = tmp_path / "game_dir"
    _write(game / "game" / "ch1" / "script.rpyc", 'label ch1:\n    e "New."\n')
    ctx = make_ctx(game)
    stale = _write(ctx.paths.raw / "renpy" / "decompiled" / "ch1" / "script.rpy", 'label ch1:\n    e "Stale."\n')
    stub_tools(ctx, fail=("ch1",))  # unrpyc fails on it
    from vndub.stages import detect

    detect.run(ctx)
    ex = ctx.engine().extract(ctx)
    assert ex.script_files == [] and not stale.exists()  # the leftover is gone, so the failure is not hidden behind it
    assert len(ex.warnings) == 1 and "ch1/script.rpyc" in ex.warnings[0]


def test_dialogue_tab_ids_follow_the_folder_of_the_file(tmp_path: Path, make_ctx) -> None:
    game = tmp_path / "game_dir"
    _write(game / "game" / "ch1" / "script.rpy", 'define e = Character("Eileen")\nlabel ch1:\n    e "Chapter one."\n')
    _write(game / "game" / "ch2" / "script.rpy", 'default seen = False\nlabel ch2:\n    e "Chapter two."\n')  # the say is on ch1's line number
    tab = _write(
        tmp_path / "dialogue.tab",
        "Identifier\tCharacter\tDialogue\tFilename\tLine Number\tSource Code\n"
        'ch1_one\te\tChapter one.\tgame/ch1/script.rpy\t3\te "[what]"\n'
        'ch2_two\te\tChapter two.\tgame/ch2/script.rpy\t3\te "[what]"\n',
    )
    ctx = make_ctx(game, sets={"parse.engine_opts.renpy.dialogue_tab": str(tab)})
    _, _, _, res = detect_and_parse(ctx)
    assert [(l.source_file, l.source_line, l.dialogue_id) for l in res.lines] == [  # type: ignore[attr-defined]
        ("ch1/script.rpy", 3, "ch1_one"),
        ("ch2/script.rpy", 3, "ch2_two"),
    ]
    assert all("dialogue_tab_id" in l.flags for l in res.lines)


def test_dialogue_tab_fields_starting_with_a_quote_are_text(tmp_path: Path) -> None:
    tab = _write(
        tmp_path / "dialogue.tab",
        "Identifier\tCharacter\tDialogue\tFilename\tLine Number\tSource Code\n"
        'start_a\te\t"Hello," she said.\tgame/script.rpy\t3\te "[what]"\n'
        'start_b\te\t"\tgame/script.rpy\t4\te "[what]"\n'  # a lone quote: a csv reader would swallow every row after it
        "start_c\te\tPlain.\tgame/script.rpy\t5\te \"[what]\"\n",
    )
    rows = renpy.read_dialogue_tab(tab)
    assert list(rows) == ["start_a", "start_b", "start_c"]
    assert rows["start_a"]["Dialogue"] == '"Hello," she said.' and rows["start_a"]["Line Number"] == "3"
    assert rows["start_b"]["Dialogue"] == '"' and rows["start_c"]["Filename"] == "game/script.rpy"

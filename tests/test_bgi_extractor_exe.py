"""The bgi-extractor.exe the app ships (``packaging/build_app.ps1`` compiles it) on a tiny BGI game made here.

The archive is a real BURIKO ARC20 with DSC-compressed BurikoCompiledScript files (``tests/bgi_archive.py``),
so the run covers the whole tool (archive index, DSC Huffman, bytecode parser, Shift-JIS -> UTF-8, JSON
writer) and the engine's reading of its output. Skipped where the exe has not been built.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bgi_archive import write_bgi_game
from engine_helpers import detect_and_parse

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = (
    ROOT / "dist" / "VNDub" / "bin" / "bgi-extractor" / "bgi-extractor.exe",
    ROOT / "packaging" / "_staging" / "bgi-extractor" / "bin" / "bgi-extractor.exe",
)
EXE = next((p for p in CANDIDATES if p.is_file()), None)

pytestmark = pytest.mark.skipif(EXE is None, reason="bgi-extractor.exe is not built here (packaging/build_app.ps1)")


def test_the_shipped_exe_reads_a_bgi_game_in_a_folder_with_japanese_and_cyrillic_names(tmp_path: Path, make_ctx) -> None:
    game = write_bgi_game(tmp_path / "ゲーム Игра")
    ctx = make_ctx(game, sets={"tools.bgi_extractor": str(EXE)})

    summary, ex, _audio, res = detect_and_parse(ctx, "bgi")

    assert summary["engine"] == "bgi"
    assert not ex.warnings, ex.warnings
    # one JSON per script, nothing merged: every line is read once
    assert sorted(p.relative_to(ctx.paths.raw / "bgi").as_posix() for p in ex.script_files) == [
        "data01701/01_scripts/data01701/sc01.json",
        "data01701/01_scripts/data01701/sc02.json",
    ]
    got = [(l.scene, l.speaker_raw, l.text_src, l.voice_file) for l in res.lines]
    assert got == [
        ("sc01", "由岐", "「間宮卓司……」", "voice/yuki_000003.ogg"),
        ("sc01", "", "街の喧噪の中に立っている時とは打って変わって。", None),
        ("sc01", "卓司", "「なんでこんな所に？」", "voice/taku_000001.ogg"),
        ("sc02", "由岐", "「行こう」", "voice/yuki_000004.ogg"),
    ]

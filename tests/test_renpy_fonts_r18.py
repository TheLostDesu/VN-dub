"""The cmap reader (``vndub.engines.renpy_fonts``): real fonts of this machine and hand-built minimal fonts."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from vndub.engines import renpy_fonts as rf

WINFONTS = Path(r"C:\Windows\Fonts")
SDK = Path(r"B:\VN-Dub-Research\tools\renpy-sdk\renpy-8.5.3-sdk")
ZERO2 = bytes(2)


def _sfnt(sub: bytes, plat: int = 3, enc: int = 1, magic: bytes = b"\x00\x01\x00\x00", shift: int = 0) -> bytes:
    cmap = struct.pack(">HHHHI", 0, 1, plat, enc, 12) + sub
    head = magic + struct.pack(">HHHH", 1, 16, 0, 0)
    rec = b"cmap" + struct.pack(">III", 0, shift + 12 + 16, len(cmap))  # table offsets are from the start of the FILE
    return head + rec + cmap


def _fmt4(segs: list[tuple[int, int]]) -> bytes:
    segs = [*segs, (0xFFFF, 0xFFFF)]
    n = len(segs)
    ends = b"".join(struct.pack(">H", e) for _s, e in segs)
    starts = b"".join(struct.pack(">H", s) for s, _e in segs)
    deltas = b"".join(struct.pack(">H", (1 - s) & 0xFFFF if s != 0xFFFF else 1) for s, _e in segs)  # glyph = cp + delta, never 0
    body = struct.pack(">HHHH", 2 * n, 0, 0, 0) + ends + ZERO2 + starts + deltas + ZERO2 * n
    return struct.pack(">HHH", 4, 6 + len(body), 0) + body


def _fmt12(groups: list[tuple[int, int, int]]) -> bytes:
    body = b"".join(struct.pack(">III", s, e, g) for s, e, g in groups)
    return struct.pack(">HHIII", 12, 0, 16 + len(body), 0, len(groups)) + body


def test_format4_all_and_missing(tmp_path: Path) -> None:
    full = tmp_path / "full.ttf"
    full.write_bytes(_sfnt(_fmt4([(0x20, 0x7E), (0x401, 0x401), (0x410, 0x44F), (0x451, 0x451)])))
    assert rf.has_cyrillic(full) is True
    part = tmp_path / "part.ttf"
    part.write_bytes(_sfnt(_fmt4([(0x20, 0x7E), (0x410, 0x44F)])))  # no yo
    assert rf.has_cyrillic(part) is False and rf.covered(part) == set(range(0x410, 0x450))
    latin = tmp_path / "latin.ttf"
    latin.write_bytes(_sfnt(_fmt4([(0x20, 0x7E)])))
    assert rf.has_cyrillic(latin) is False


def test_format12_and_collections(tmp_path: Path) -> None:
    f = tmp_path / "f12.otf"
    f.write_bytes(_sfnt(_fmt12([(0x20, 0x7E, 1), (0x401, 0x401, 100), (0x410, 0x44F, 200), (0x451, 0x451, 300)]), 3, 10, b"OTTO"))
    assert rf.has_cyrillic(f) is True
    g = tmp_path / "g12.ttf"
    g.write_bytes(_sfnt(_fmt12([(0x410, 0x42F, 200)]), 0, 4))
    assert rf.has_cyrillic(g) is False and rf.covered(g) == set(range(0x410, 0x430))
    # a .ttc: header "ttcf", version, count, offset of the first font; the first font stands for the collection
    c = tmp_path / "c.ttc"
    c.write_bytes(b"ttcf" + struct.pack(">III", 0x10000, 1, 16) + _sfnt(_fmt12([(0x401, 0x451, 5)]), shift=16))
    assert rf.has_cyrillic(c) is True


def test_garbage_is_none(tmp_path: Path) -> None:
    p = tmp_path / "x.ttf"
    p.write_bytes(b"not a font at all")
    assert rf.has_cyrillic(p) is None and rf.has_cyrillic(tmp_path / "missing.ttf") is None
    p.write_bytes(b"")
    assert rf.has_cyrillic(p) is None


@pytest.mark.skipif(not (WINFONTS / "arial.ttf").exists(), reason="no arial.ttf")
def test_real_arial_has_cyrillic() -> None:
    assert rf.has_cyrillic(WINFONTS / "arial.ttf") is True


@pytest.mark.skipif(not (SDK / "renpy" / "common" / "DejaVuSans.ttf").exists(), reason="no Ren'Py SDK")
def test_real_dejavu_has_cyrillic() -> None:
    assert rf.has_cyrillic(SDK / "renpy" / "common" / "DejaVuSans.ttf") is True


def test_a_real_font_without_cyrillic() -> None:
    # Office fonts with a plain Unicode cmap. A symbol font (webdings) has a symbol cmap the reader does not judge: it
    # answers None (unknown) for it, which is the safe answer and not this test's subject.
    cands = [p for n in ("AGENCYR.TTF", "ALGER.TTF", "BASKVILL.TTF") if (p := WINFONTS / n).exists()]
    if not cands:
        pytest.skip("no Latin-only font in C:/Windows/Fonts")
    assert any(rf.has_cyrillic(p) is False for p in cands)


# ----------------------------------------------------------------------------
# pack: the fonts the game names, and what the text patch does about a font without Cyrillic
# ----------------------------------------------------------------------------

from test_renpy_text_patch import _manifest, _project, _rpy  # noqa: E402

from vndub.stages.common import run_stage  # noqa: E402

GUI_SCRIPT = 'define gui.text_font = "fonts/Latin.ttf"\ndefine s = Character("Sakuya")\nlabel start:\n    s "Fine."\n'
OLD_SCRIPT = 'init python:\n    style.default.font = "fonts/Latin.ttf"\ndefine s = Character("Sakuya")\nlabel start:\n    s "Fine."\n'
TR1 = {"Fine.": "Хорошо."}


def _put_font(ctx, cyrillic: bool, name: str = "fonts/Latin.ttf") -> None:
    segs = [(0x20, 0x7E)] + ([(0x401, 0x401), (0x410, 0x44F), (0x451, 0x451)] if cyrillic else [])
    p = ctx.game_dir / "game" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(_sfnt(_fmt4(segs)))


def test_font_without_cyrillic_switches_to_the_engine_font(make_ctx, tmp_path: Path) -> None:
    from vndub.stages.pack import game_font_names

    assert game_font_names(['define gui.text_font = "a.ttf"', 'style x:\n    font "b/B.otf"', "# font \"c.ttf\"", 'Character("x", what_font="d.ttf", who_font = "e.ttf")', "$ style.default.font = 'f.ttf'"]) == [
        "a.ttf", "b/B.otf", "d.ttf", "e.ttf", "f.ttf"
    ]
    ctx, _ = _project(make_ctx, tmp_path, GUI_SCRIPT, TR1)
    _put_font(ctx, cyrillic=False)
    s = run_stage("pack", ctx)
    text = _rpy(ctx)
    assert 'translate russian python:\n    gui.text_font = "DejaVuSans.ttf"\n' in text and "style default" not in text  # a new GUI: the gui variables are enough
    fc = _manifest(ctx)["text"]["font_check"]
    assert fc["lacking"] == ["fonts/Latin.ttf"] and fc["mode"] == "auto" and "DejaVuSans.ttf" in fc["action"]
    assert _manifest(ctx)["text"]["font"] == "DejaVuSans.ttf" and "Latin.ttf" in s["text_font_warning"]
    assert "Font note" in (ctx.paths.out_engine("renpy") / "INSTALL.md").read_text(encoding="utf-8")


def test_old_gui_also_gets_a_style_default_block(make_ctx, tmp_path: Path) -> None:
    ctx, _ = _project(make_ctx, tmp_path, OLD_SCRIPT, TR1)
    _put_font(ctx, cyrillic=False)
    run_stage("pack", ctx)
    assert 'translate russian style default:\n    font "DejaVuSans.ttf"\n' in _rpy(ctx)


def test_font_with_cyrillic_or_unknown_font_changes_nothing(make_ctx, tmp_path: Path) -> None:
    ctx, _ = _project(make_ctx, tmp_path, GUI_SCRIPT, TR1)
    _put_font(ctx, cyrillic=True)
    s = run_stage("pack", ctx)
    assert "gui.text_font" not in _rpy(ctx) and "text_font_warning" not in s
    assert _manifest(ctx)["text"]["font_check"]["fonts"][0]["cyrillic"] is True and _manifest(ctx)["text"]["font"] is None
    (ctx.game_dir / "game" / "fonts" / "Latin.ttf").unlink()  # a font that is not there: nothing is known, nothing is replaced
    run_stage("pack", ctx)
    assert "gui.text_font" not in _rpy(ctx)
    assert _manifest(ctx)["text"]["font_check"]["fonts"][0]["cyrillic"] is None


def test_explicit_font_wins_and_empty_switches_the_check_off(make_ctx, tmp_path: Path) -> None:
    ctx, _ = _project(make_ctx, tmp_path, GUI_SCRIPT, TR1)
    _put_font(ctx, cyrillic=False)
    ctx.cfg.pack.renpy["text"] = {"font": "fonts/Mine.ttf"}
    run_stage("pack", ctx)
    assert 'gui.text_font = "fonts/Mine.ttf"' in _rpy(ctx) and "DejaVuSans" not in _rpy(ctx)
    for off in ("", False):
        ctx.cfg.pack.renpy["text"] = {"font": off}
        run_stage("pack", ctx)
        assert "gui.text_font" not in _rpy(ctx) and _manifest(ctx)["text"]["font_check"]["mode"] == "off"


def test_generated_font_blocks_are_valid_renpy() -> None:
    from vndub.stages.pack import renpy_text_rpy

    t = renpy_text_rpy("russian", [], [], "DejaVuSans.ttf", True)
    assert t.endswith('translate russian python:\n    gui.text_font = "DejaVuSans.ttf"\n    gui.name_text_font = "DejaVuSans.ttf"\n    gui.interface_text_font = "DejaVuSans.ttf"\n\ntranslate russian style default:\n    font "DejaVuSans.ttf"\n')

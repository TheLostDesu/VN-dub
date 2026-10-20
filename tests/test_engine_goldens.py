"""Golden parser outputs per fixture (regenerate: VNDUB_UPDATE_GOLDEN=1 uv run pytest tests/test_engine_goldens.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from engine_helpers import check_golden, detect_and_parse

CASES = [
    ("renpy_mini", "renpy", 20, 25),
    ("kag_mini", "kirikiri", 5, 7),
    ("nscripter_mini", "nscripter", 4, 6),
    ("siglus_mini", "siglus", 4, 4),
    ("bgi_mini", "bgi", 3, 4),
    ("tyrano_mini", "tyrano", 4, 6),
]


@pytest.mark.parametrize("fixture,engine,n_voiced,n_lines", CASES)
def test_detect_and_golden(fixture: str, engine: str, n_voiced: int, n_lines: int, make_ctx, fixtures_dir: Path) -> None:
    summary, ex, audio, res = detect_and_parse(make_ctx(fixtures_dir / fixture))
    assert summary["engine"] == engine, summary
    assert len(res.lines) == n_lines
    voiced = [l for l in res.lines if l.has_voice]
    assert len(voiced) == n_voiced
    bound = {l.audio_id for l in voiced}
    assert len(bound) == n_voiced  # no file bound twice
    assert all(l.audio_id in {a.id for a in audio} for l in voiced)
    check_golden(fixture, res.lines)


def test_renpy_fixture_details(make_ctx, fixtures_dir: Path) -> None:
    _, _, _, res = detect_and_parse(make_ctx(fixtures_dir / "renpy_mini"))
    by = {l.line_id: l for l in res.lines}
    menu = next(l for l in res.lines if "menu" in l.flags)
    assert menu.choices == ["励ます", "黙っている"]
    head = next(l for l in res.lines if "sustain_head" in l.flags)
    cont = by[head.sustain_lines[0]]  # type: ignore[attr-defined]
    assert "voice_sustain" in cont.flags and not cont.has_voice and cont.voice_file is None
    assert head.voice_file == "voice/har_0006.ogg"
    speakers = {l.speaker_id for l in res.lines}
    assert speakers == {"narrator", "sakura", "haruto", "yui"}
    assert {l.scene for l in res.lines} == {"start", "after_menu"}
    assert all(l.context_prev == [] or set(l.context_prev) <= set(by) for l in res.lines)
    assert res.characters["sakura"].names["src"] == "さくら" and res.characters["sakura"].voice_prefix == "sak"


def test_detect_forced_engine(make_ctx, fixtures_dir: Path) -> None:
    from vndub.stages import detect

    ctx = make_ctx(fixtures_dir / "renpy_mini", extra={"engine": "generic"})
    s = detect.run(ctx)
    assert s["engine"] == "generic"
    assert ctx.require_project().engine == "generic"


def test_detect_unknown_dir(tmp_path: Path, make_ctx) -> None:
    from vndub.stages import detect

    (tmp_path / "empty").mkdir()
    s = detect.run(make_ctx(tmp_path / "empty"))
    assert s["engine"] == "generic" and s["confidence"] == 0.0


def test_detect_drm_markers(tmp_path: Path, make_ctx) -> None:
    from vndub.stages import detect

    g = tmp_path / "g"
    (g / "game").mkdir(parents=True)
    (g / "game" / "script.rpy").write_text('label start:\n    "x"\n', encoding="utf-8")
    (g / "steam_api64.dll").write_bytes(b"")
    ctx = make_ctx(g)
    s = detect.run(ctx)
    assert s["drm"] == ["steam"]
    assert any("DRM" in w for w in ctx.require_project().warnings)

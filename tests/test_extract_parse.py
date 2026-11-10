"""detect -> extract (real ffmpeg decoding) -> parse on the fixtures, through the stage runner."""

from __future__ import annotations

from pathlib import Path

import soundfile as sf

from conftest import requires_ffmpeg
from vndub.manifest.io import AudioIndexRecord, load_characters, load_lines, read_json, read_jsonl
from vndub.stages.common import run_stage
from vndub.stages.extract import assign_ids


def test_assign_ids_disambiguates() -> None:
    ids = assign_ids(["voice/a/x01.ogg", "voice/b/x01.ogg", "voice/y02.ogg", "Y02.wav"])
    assert ids["voice/a/x01.ogg"] == "voice__a__x01"
    assert ids["voice/b/x01.ogg"] == "voice__b__x01"
    assert len(set(ids.values())) == 4


@requires_ffmpeg
def test_renpy_mini_stages(make_ctx, fixtures_dir: Path) -> None:
    ctx = make_ctx(fixtures_dir / "renpy_mini")
    run_stage("detect", ctx)
    s = run_stage("extract", ctx)
    assert s["voice_files"] == 20 and s["decoded"] == 20 and s["errors"] == 0
    audio = read_jsonl(ctx.paths.audio_index, AudioIndexRecord)
    a = audio[0]
    assert a.codec == "vorbis" and a.sr == 22050 and a.ch == 1 and a.duration > 0.3
    assert sf.info(str(ctx.paths.abs(a.wav16))).samplerate == 16000
    assert sf.info(str(ctx.paths.abs(a.wav))).samplerate == 22050
    ex = read_json(ctx.paths.raw / "extract.json")
    assert len(ex["script_files"]) == 1
    p = run_stage("parse", ctx)
    assert p["voiced"] == 20 and p["voice_coverage"] == 1.0 and p["sustain_lines"] == 1
    lines = load_lines(ctx.paths.lines)
    assert all(l.src_wav and ctx.paths.abs(l.src_wav).exists() for l in lines if l.has_voice)
    chars = load_characters(ctx.paths.characters)
    assert set(chars) == {"sakura", "haruto", "yui", "hero"}
    assert chars["hero"].names["src"] == "蓮" and chars["hero"].stats.n_lines == 0  # the unvoiced protagonist is a character too
    assert p["unvoiced"] == 21 and p["lines"] == 41
    proj = ctx.require_project()
    assert proj.engine == "renpy" and proj.voice_codec.codec == "vorbis" and proj.voice_codec.sr == 22050
    # second run hits the cache
    s2 = run_stage("extract", ctx)
    assert s2["decoded_now"] == 0 and s2["cached"] == 20


@requires_ffmpeg
def test_parse_keeps_user_character_edits(make_ctx, fixtures_dir: Path) -> None:
    from vndub.manifest.io import save_characters

    ctx = make_ctx(fixtures_dir / "renpy_mini")
    for st in ("detect", "extract", "parse"):
        run_stage(st, ctx)
    chars = load_characters(ctx.paths.characters)
    chars["sakura"].names["tgt"] = "Сакура"
    chars["sakura"].gender = "f"
    chars["sakura"].route = "B"
    save_characters(ctx.paths.characters, chars)
    run_stage("parse", ctx)
    again = load_characters(ctx.paths.characters)["sakura"]
    assert (again.names["tgt"], again.gender, again.route) == ("Сакура", "f", "B")


@requires_ffmpeg
def test_nscripter_archives_and_loose_override(tmp_path: Path, make_ctx) -> None:
    from scripts_helpers import tone_bytes  # type: ignore[import-not-found]
    from vndub.engines.nsa import encode_nscript, write_nsa

    game = tmp_path / "ns"
    game.mkdir()
    (game / "nscript.dat").write_bytes(encode_nscript('*start\ndwave 0,"voice\\a01.wav"\n【澪】「やあ」\\\n'))
    write_nsa(game / "arc.nsa", [("voice/a01.wav", tone_bytes(200))])
    (game / "voice").mkdir()
    (game / "voice" / "a01.wav").write_bytes(tone_bytes(400, seconds=0.6))  # loose wins
    ctx = make_ctx(game)
    for st in ("detect", "extract", "parse"):
        run_stage(st, ctx)
    audio = read_jsonl(ctx.paths.audio_index, AudioIndexRecord)
    assert len(audio) == 1 and abs(audio[0].duration - 0.6) < 0.05
    (line,) = load_lines(ctx.paths.lines)
    assert line.speaker_raw == "澪" and line.has_voice


@requires_ffmpeg
def test_kag_and_tyrano_stages(make_ctx, fixtures_dir: Path) -> None:
    for fx, n in (("kag_mini", 5), ("tyrano_mini", 4), ("siglus_mini", 4), ("bgi_mini", 3)):
        ctx = make_ctx(fixtures_dir / fx)
        for st in ("detect", "extract"):
            run_stage(st, ctx)
        p = run_stage("parse", ctx)
        assert p["voiced"] == n, (fx, p)


def test_parse_requires_extract(make_ctx, fixtures_dir: Path) -> None:
    import pytest

    from vndub.stages.common import StageError

    ctx = make_ctx(fixtures_dir / "renpy_mini")
    run_stage("detect", ctx)
    with pytest.raises(StageError, match="extract"):
        run_stage("parse", ctx)


@requires_ffmpeg
def test_extract_reports_decode_errors_and_stops_when_nothing_decodes(make_ctx, fixtures_dir: Path, monkeypatch) -> None:
    import pytest

    from vndub.audio import ffmpeg
    from vndub.stages.common import StageError

    ctx = make_ctx(fixtures_dir / "renpy_mini")
    run_stage("detect", ctx)
    real = ffmpeg.decode
    bad = {"n": 0}

    def flaky(tools, src, dst, sr=None, mono=True):
        if "_01" in src.name or bad["n"] < 2:
            bad["n"] += 1
            raise ffmpeg.AudioToolError("ffmpeg failed (1): Invalid data found")
        return real(tools, src, dst, sr=sr, mono=mono)

    monkeypatch.setattr(ffmpeg, "decode", flaky)
    s = run_stage("extract", ctx)
    de = s["decode_errors"]
    assert s["errors"] == de["count"] >= 1 and 1 <= len(de["samples"]) <= 3 and de["missing_tools"] == []

    def dead(*a, **k):
        raise ffmpeg.AudioToolError("tool 'vgmstream_cli' not found; set it in configs/tools.yaml")

    monkeypatch.setattr(ffmpeg, "decode", dead)
    ctx2 = make_ctx(fixtures_dir / "renpy_mini")
    ctx2.force = True
    with pytest.raises(StageError, match="none of the .*missing tool: vgmstream_cli"):
        run_stage("extract", ctx2)

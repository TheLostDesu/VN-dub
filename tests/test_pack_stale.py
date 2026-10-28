"""pack: the patch folder holds exactly what manifest_patch.json lists, and what cannot be installed is said so.

* ``out/<engine>`` used to be only ever added to, and the installer ships the whole folder: after a line was rejected,
  ``include_review_lines`` went to ``skip`` or a re-import under another parse version renamed the Ren'Py ids, the old
  ``voice_dub/<oldid>.ogg`` files stayed in the patch although the manifest and the coverage did not list them;
* a Ren'Py line that has a ``voice`` statement of its own was dubbed as an "unvoiced" line through ``config.auto_voice``
  and counted as installed, but Ren'Py never asks auto_voice for such a line.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.test_pack import _manifest, _setup
from vndub.config import ProjectYaml, save_project_yaml
from vndub.manifest.io import FinalIndexRecord, LineRecord, read_jsonl, write_jsonl
from vndub.stages.common import run_stage


def _files(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir()) if folder.is_dir() else []


def test_a_line_that_leaves_the_patch_leaves_the_folder(make_ctx, tmp_path: Path) -> None:
    ctx, _ = _setup(make_ctx, tmp_path, "renpy", ["voice/a.ogg", "voice/b.ogg", "voice/c.ogg"])
    run_stage("pack", ctx)
    out = ctx.paths.out_engine("renpy")
    voice = out / "game" / "voice"
    (voice / "readme.txt").write_text("not audio", encoding="utf-8")  # not a voice file: never touched
    (voice / "left_behind.ogg").write_bytes(b"OggS")  # a voice file of the folder that the patch does not hold
    (out / "game" / "other").mkdir()
    (out / "game" / "other" / "keep.ogg").write_bytes(b"OggS")  # a folder the patch does not write to
    write_jsonl(ctx.paths.final_index, read_jsonl(ctx.paths.final_index, FinalIndexRecord)[:2])  # the third line is rejected
    s = run_stage("pack", ctx)
    assert _files(voice) == ["a.ogg", "b.ogg", "readme.txt"] and s["removed_stale"] == 2  # c.ogg, left_behind.ogg
    assert (out / "game" / "other" / "keep.ogg").exists()
    assert [f["file"] for f in _manifest(out)["files"]] == ["voice/a.ogg", "voice/b.ogg"]


def test_a_partial_pack_leaves_the_rest_of_the_folder_alone(make_ctx, tmp_path: Path) -> None:
    ctx, _ = _setup(make_ctx, tmp_path, "renpy", ["voice/a.ogg", "voice/b.ogg"])
    run_stage("pack", ctx)
    ctx.only = {"s_000000"}
    s = run_stage("pack", ctx)
    assert _files(ctx.paths.out_engine("renpy") / "game" / "voice") == ["a.ogg", "b.ogg"] and "removed_stale" not in s


def test_other_loose_engines_are_cleaned_too_and_the_old_manifest_cannot_reach_outside(make_ctx, tmp_path: Path) -> None:
    ctx, _ = _setup(make_ctx, tmp_path, "nscripter", ["voice/m1.wav", "voice/m2.wav"])
    run_stage("pack", ctx)
    out = ctx.paths.out_engine("nscripter")
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"RIFF")
    man = _manifest(out)
    man["files"].append({"file": "../../outside.wav", "packed_as": "../../outside.wav", "line_id": "x"})  # a doctored manifest
    (out / "manifest_patch.json").write_text(json.dumps(man), encoding="utf-8")
    write_jsonl(ctx.paths.final_index, read_jsonl(ctx.paths.final_index, FinalIndexRecord)[:1])
    run_stage("pack", ctx)
    assert _files(out / "voice") == ["m1.wav"] and outside.exists()


def _unvoiced_finals(ctx, ids: list[str]) -> tuple[list[LineRecord], list[FinalIndexRecord]]:
    lines, finals = [], []
    for n, ident in enumerate(ids):
        lid = f"start_{n:06d}"
        enc = ctx.paths.final_encoded / "unvoiced" / f"{ident}.ogg"
        enc.parent.mkdir(parents=True, exist_ok=True)
        enc.write_bytes(b"OggS" + bytes([n]) * 32)
        lines.append(LineRecord(line_id=lid, scene="start", order=n, speaker_id="e", text_src="A line.", dialogue_id=ident))
        finals.append(
            FinalIndexRecord(line_id=lid, voice_file=f"unvoiced/{ident}.ogg", final_wav=f"final/{lid}.wav", encoded=ctx.paths.rel(enc), source="synth/x.wav", unvoiced=True)
        )
    return lines, finals


def test_renamed_ids_do_not_leave_voice_dub_files_or_the_generated_rpy_behind(make_ctx, tmp_path: Path) -> None:
    ctx, _ = _setup(make_ctx, tmp_path, "renpy", [])
    lines, finals = _unvoiced_finals(ctx, ["start_aaaaaaaa", "start_bbbbbbbb"])
    write_jsonl(ctx.paths.lines, lines)
    write_jsonl(ctx.paths.final_index, finals)
    s = run_stage("pack", ctx)
    out = ctx.paths.out_engine("renpy")
    dub = out / "game" / "voice_dub"
    assert _files(dub) == ["start_aaaaaaaa.ogg", "start_bbbbbbbb.ogg"] and (out / "game" / "vndub_auto_voice.rpy").exists() and "removed_stale" not in s

    # a re-import under another parse version renamed the ids: the old files are not installable names any more
    lines2, finals2 = _unvoiced_finals(ctx, ["start_cccccccc", "start_bbbbbbbb"])
    write_jsonl(ctx.paths.lines, lines2)
    write_jsonl(ctx.paths.final_index, finals2)
    s2 = run_stage("pack", ctx)
    assert _files(dub) == ["start_bbbbbbbb.ogg", "start_cccccccc.ogg"] and s2["removed_stale"] == 1
    listed = sorted(Path(f["packed_as"]).name for f in _manifest(out)["files"])
    assert listed == _files(dub)  # the folder and the manifest agree

    # nothing without a voice is left: the folder and the auto_voice script that pointed at it go
    ctx2, _ = _setup(make_ctx, tmp_path, "renpy", ["voice/x.ogg"])
    write_jsonl(ctx2.paths.lines, [])
    s3 = run_stage("pack", ctx2)
    assert not dub.exists() and not (out / "game" / "vndub_auto_voice.rpy").exists() and s3["removed_stale"] == 3
    assert _files(out / "game" / "voice") == ["x.ogg"]


def test_a_script_the_user_wrote_is_not_taken_for_the_generated_one(make_ctx, tmp_path: Path) -> None:
    ctx, _ = _setup(make_ctx, tmp_path, "renpy", ["voice/x.ogg"])
    mine = ctx.paths.out_engine("renpy") / "game" / "vndub_auto_voice.rpy"
    mine.parent.mkdir(parents=True, exist_ok=True)
    mine.write_text("# my own tweaks\n", encoding="utf-8")
    run_stage("pack", ctx)
    assert mine.read_text(encoding="utf-8") == "# my own tweaks\n"


# ----------------------------------------------------------------------------
# lines with a voice statement of their own
# ----------------------------------------------------------------------------


def _own_voice_project(make_ctx, tmp_path: Path, engine: str = "renpy"):
    game = tmp_path / "game"
    game.mkdir(parents=True, exist_ok=True)
    ctx = make_ctx(game, sets={"synth.voice_unvoiced": True})
    save_project_yaml(ctx.paths.project_yaml, ProjectYaml(game_dir=str(game), engine=engine))
    kw = dict(scene="start", speaker_id="e")
    write_jsonl(
        ctx.paths.lines,
        [
            LineRecord(line_id="s_1", order=1, text_src="Plain, no voice.", dialogue_id="start_11111111", **kw),
            LineRecord(line_id="s_2", order=2, text_src="Voice from an expression.", dialogue_id="start_22222222", flags=["voice_expression"], **kw),
            LineRecord(line_id="s_3", order=3, text_src="Voice file missing.", dialogue_id="start_33333333", flags=["voice_missing"], voice_file="voice/gone.ogg", **kw),
            LineRecord(line_id="s_4", order=4, text_src="Narration.", dialogue_id="start_44444444", scene="start", speaker_id="narrator", flags=["voice_missing"]),
            LineRecord(line_id="s_5", order=5, text_src="Continued.", dialogue_id="start_55555555", flags=["voice_sustain"], **kw),
        ],
    )
    return ctx


def test_lines_with_their_own_voice_statement_are_not_dubbed_through_auto_voice(make_ctx, tmp_path: Path) -> None:
    ctx = _own_voice_project(make_ctx, tmp_path)
    assert [l.line_id for l in ctx.dub_lines()] == ["s_1"]  # synth, score and master never see the others
    assert [l.line_id for l in ctx.own_voice_lines()] == ["s_2", "s_3"]  # narration and sustain continuations are not "lines to dub" either way
    ctx.cfg.synth.voice_unvoiced = False
    assert ctx.own_voice_lines() == []
    # another engine has no such rule: a missing voice file there is just a line without audio
    kirikiri = _own_voice_project(make_ctx, tmp_path / "k", engine="kirikiri")
    assert [l.line_id for l in kirikiri.dub_lines()] == ["s_1", "s_2", "s_3"] and kirikiri.own_voice_lines() == []


def test_pack_reports_the_lines_it_cannot_install(make_ctx, tmp_path: Path) -> None:
    ctx = _own_voice_project(make_ctx, tmp_path)
    lines, finals = _unvoiced_finals(ctx, ["start_11111111"])
    finals[0].line_id = "s_1"
    write_jsonl(ctx.paths.final_index, finals)
    s = run_stage("pack", ctx)
    out = ctx.paths.out_engine("renpy")
    cov = _manifest(out)["coverage"]
    assert (cov["dub_lines"], cov["installed"], cov["left_out"]) == (1, 1, 0)  # what is dubbed is installed...
    assert cov["own_voice"] == 2 and cov["own_voice_lines"] == ["s_2", "s_3"]  # ...and the rest is counted apart, not silently missing
    assert s["own_voice"] == 2
    text = (out / "INSTALL.md").read_text(encoding="utf-8")
    assert "2 line(s) are NOT INSTALLABLE: own voice statement" in text and "coverage.own_voice_lines" in text
    assert "2 line(s) are NOT INSTALLABLE" not in _own_voice_text_without(ctx)


def _own_voice_text_without(ctx) -> str:
    """INSTALL.md of a project with no such lines (the section must not appear)."""
    write_jsonl(ctx.paths.lines, [l for l in read_jsonl(ctx.paths.lines, LineRecord) if l.line_id == "s_1"])
    run_stage("pack", ctx)
    return (ctx.paths.out_engine("renpy") / "INSTALL.md").read_text(encoding="utf-8")

"""pack layouts per engine with fake final_index records."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vndub.config import ProjectYaml, save_project_yaml
from vndub.engines import xp3
from vndub.manifest.io import FinalIndexRecord, SelectionRecord, read_jsonl, write_jsonl
from vndub.stages.common import StageError, run_stage


def _setup(make_ctx, tmp_path: Path, engine: str, voice_files: list[str], **proj_extra):
    game = tmp_path / "game"
    game.mkdir(parents=True, exist_ok=True)
    ctx = make_ctx(game)
    save_project_yaml(ctx.paths.project_yaml, ProjectYaml(game_dir=str(game), engine=engine, **proj_extra))
    finals, sel = [], []
    for i, vf in enumerate(voice_files):
        lid = f"s_{i:06d}"
        enc = ctx.paths.final_encoded / vf
        enc.parent.mkdir(parents=True, exist_ok=True)
        enc.write_bytes(b"OggS" + bytes([i]) * 64)
        finals.append(FinalIndexRecord(line_id=lid, voice_file=vf, final_wav=f"final/{lid}.wav", encoded=ctx.paths.rel(enc), source=f"synth/{lid}/mock_tts_A_0.wav"))
        sel.append(SelectionRecord(line_id=lid, winner=f"synth/{lid}/mock_tts_A_0.wav", backend="mock_tts", route="A", status="auto"))
    write_jsonl(ctx.paths.final_index, finals)
    write_jsonl(ctx.paths.selection, sel)
    return ctx, game


def _manifest(out: Path) -> dict:
    return json.loads((out / "manifest_patch.json").read_text(encoding="utf-8"))


def test_pack_renpy(make_ctx, tmp_path: Path) -> None:
    ctx, _ = _setup(make_ctx, tmp_path, "renpy", ["voice/sak_0001.ogg", "voice/har_0001.ogg"])
    s = run_stage("pack", ctx)
    out = ctx.paths.out_engine("renpy")
    assert (out / "game" / "voice" / "sak_0001.ogg").read_bytes().startswith(b"OggS")
    assert s["files"] == 2
    assert "force_archives" in (out / "INSTALL.md").read_text(encoding="utf-8")
    m = _manifest(out)
    assert [f["file"] for f in m["files"]] == ["voice/sak_0001.ogg", "voice/har_0001.ogg"]
    assert m["files"][0]["backend"] == "mock_tts" and m["files"][0]["route"] == "A" and len(m["files"][0]["sha256"]) == 64


def test_pack_kirikiri_patch_xp3(make_ctx, tmp_path: Path) -> None:
    ctx, game = _setup(make_ctx, tmp_path, "kirikiri", ["voice/nen_0001.ogg", "nen_0002.ogg"], encrypted=True)
    (game / "patch.xp3").write_bytes(b"")  # existing patch -> next number
    s = run_stage("pack", ctx)
    target = ctx.paths.out_engine("kirikiri") / "patch2.xp3"
    assert s["out"] == str(target) and s["writer"] == "builtin"
    with xp3.XP3Reader(target) as r:
        assert sorted(e.name for e in r.entries) == ["nen_0001.ogg", "nen_0002.ogg"]  # flat root entries
        assert r.read("nen_0001.ogg").startswith(b"OggS")
    text = (ctx.paths.out_engine("kirikiri") / "INSTALL.md").read_text(encoding="utf-8")
    assert "version.dll" in text and "patch2.xp3" in text
    assert _manifest(ctx.paths.out_engine("kirikiri"))["files"][0]["packed_as"] == "nen_0001.ogg"


def test_pack_kirikiri_subdir_and_unencrypted(make_ctx, tmp_path: Path) -> None:
    ctx, _ = _setup(make_ctx, tmp_path, "kirikiri", ["voice/nen_0001.ogg"])
    ctx.cfg.pack.kirikiri["subdir"] = "voice"
    run_stage("pack", ctx)
    with xp3.XP3Reader(ctx.paths.out_engine("kirikiri") / "patch.xp3") as r:
        assert [e.name for e in r.entries] == ["voice/nen_0001.ogg"]
    assert "addAutoPath" in (ctx.paths.out_engine("kirikiri") / "INSTALL.md").read_text(encoding="utf-8")
    ctx.cfg.pack.kirikiri.update(mode="unencrypted_dir", subdir="")
    s = run_stage("pack", ctx)
    assert (ctx.paths.out_engine("kirikiri") / "unencrypted" / "nen_0001.ogg").exists() and s["mode"] == "unencrypted_dir"


def test_pack_nscripter_and_tyrano(make_ctx, tmp_path: Path) -> None:
    ctx, _ = _setup(make_ctx, tmp_path, "nscripter", ["voice/mio001.wav"])
    run_stage("pack", ctx)
    assert (ctx.paths.out_engine("nscripter") / "voice" / "mio001.wav").exists()
    ctx2, _ = _setup(make_ctx, tmp_path / "t", "tyrano", ["sound/akane_1.ogg"])
    run_stage("pack", ctx2)
    assert (ctx2.paths.out_engine("tyrano") / "data" / "sound" / "akane_1.ogg").exists()


def test_pack_unity_script(make_ctx, tmp_path: Path) -> None:
    ctx, _ = _setup(make_ctx, tmp_path, "unity", ["Voice/kohaku_01.wav"])
    s = run_stage("pack", ctx)
    out = ctx.paths.out_engine("unity")
    assert s["experimental"] is True
    assert (out / "clips" / "kohaku_01.wav").exists()
    script = (out / "replace_audioclips.py").read_text(encoding="utf-8")
    compile(script, "replace_audioclips.py", "exec")
    assert json.loads((out / "clips.json").read_text(encoding="utf-8"))["replacements"] == {"kohaku_01": "kohaku_01.wav"}
    assert "EXPERIMENTAL" in (out / "INSTALL.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("engine", ["siglus", "bgi", "artemis", "catsystem2", "generic", "yuris", "majiro"])
def test_pack_other_engines_write_install(make_ctx, tmp_path: Path, engine: str) -> None:
    ctx, _ = _setup(make_ctx, tmp_path, engine, ["voice/a_0001.ogg"])
    s = run_stage("pack", ctx)
    out = ctx.paths.out_engine(engine)
    assert (out / "INSTALL.md").exists() and (out / "manifest_patch.json").exists()
    assert s["files"] == 1


def test_pack_without_master(make_ctx, tmp_path: Path) -> None:
    game = tmp_path / "game"
    game.mkdir()
    ctx = make_ctx(game)
    save_project_yaml(ctx.paths.project_yaml, ProjectYaml(game_dir=str(game), engine="renpy"))
    with pytest.raises(StageError, match="master"):
        run_stage("pack", ctx)


def test_pack_states_what_it_covers_without_a_lines_manifest(make_ctx, tmp_path: Path) -> None:
    ctx, _ = _setup(make_ctx, tmp_path, "renpy", ["voice/sak_0001.ogg", "voice/har_0001.ogg"])
    finals = read_jsonl(ctx.paths.final_index, FinalIndexRecord)
    finals[1].status = "provisional"  # master marks a line mastered from a take that failed the gates
    finals[1].reason = "no candidate passes the gates after 2 regen round(s)"  # type: ignore[attr-defined]
    write_jsonl(ctx.paths.final_index, finals)
    s = run_stage("pack", ctx)
    out = ctx.paths.out_engine("renpy")
    cov = _manifest(out)["coverage"]
    assert (cov["dub_lines"], cov["installed"], cov["provisional"], cov["left_out"]) == (2, 2, 1, 0)
    assert s["provisional"] == 1 and "left_out" not in s
    m = _manifest(out)["files"]
    assert "reason" not in m[0] and m[1]["status"] == "provisional" and "regen round" in m[1]["reason"]
    text = (out / "INSTALL.md").read_text(encoding="utf-8")
    assert "What this patch covers" in text and "1 line(s) are PROVISIONAL" in text and "LEFT OUT" not in text
    run_stage("pack", ctx)
    assert (out / "INSTALL.md").read_text(encoding="utf-8").count("What this patch covers") == 1

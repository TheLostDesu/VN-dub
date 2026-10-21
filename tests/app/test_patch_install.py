"""Step 7: patch plan, backup / apply / rollback, zip for a friend, and the refusal for manual engines."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Iterator

import pytest
from o_helpers import client_for, drop_runner, fresh_runner, make_env, make_game_project, tree_snapshot, wait_job

from vndub_app import appdirs, jobs, openers, patch_install
from vndub_app.errors import AppError

ORIG_A = b"OggS original sak_0001"
ORIG_B = b"OggS original har_0001"
NEW_A = b"OggS dubbed sak_0001"
NEW_B = b"OggS dubbed har_0001"
NEW_C = b"OggS dubbed new_0001"
RPY = b"# generated\ninit 1000 python:\n    config.auto_voice = 'voice_dub/{id}.ogg'\n"

GAME = {"game/script.rpy": b"label start:\n    e 'hi'\n", "game/voice/sak_0001.ogg": ORIG_A, "game/voice/har_0001.ogg": ORIG_B, "game/options.rpy": b"define config.name = 'X'\n"}
PATCH = {
    "game/voice/sak_0001.ogg": NEW_A,
    "game/voice/har_0001.ogg": NEW_B,
    "game/voice_dub/dlg_1234abcd.ogg": NEW_C,  # a file the game does not have yet (a new folder)
    "game/vndub_auto_voice.rpy": RPY,
}


@pytest.fixture
def o_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    data = make_env(monkeypatch, tmp_path)
    fresh_runner()
    yield data
    openers.set_opener(None)
    drop_runner()


@pytest.fixture
def renpy(o_env: Path, tmp_path: Path) -> tuple[Any, Path, Path]:
    return make_game_project(tmp_path, engine="renpy", game_files=GAME, patch_files=PATCH)


def post(c: Any, path: str, slug: str, **json_body: Any) -> Any:
    return c.post(f"{path}?slug={slug}", json=json_body or None)


def run(c: Any, path: str, slug: str, **json_body: Any) -> dict[str, Any]:
    r = post(c, path, slug, **json_body)
    assert r.status_code == 200, r.text
    return wait_job(r.json())


# ---- plan -------------------------------------------------------------------------------------------------------------


def test_plan_describes_what_will_be_written(renpy: tuple[Any, Path, Path]) -> None:
    h, game, patch = renpy
    c = client_for(patch_install)
    plan = c.get(f"/api/install/plan?slug={h.slug}").json()
    assert plan["engine"] == "renpy" and plan["auto_possible"] is True and plan["manual_reason_key"] is None
    assert plan["patch_dir"] == str(patch) and plan["game_dir"] == str(game) and plan["game_dir_ok"] is True
    assert "# Install" in plan["install_md"]
    by = {f["target"]: f for f in plan["files"]}
    assert set(by) == set(PATCH)  # INSTALL.md and manifest_patch.json are not written into the game
    assert by["game/voice/sak_0001.ogg"]["exists"] is True and by["game/voice_dub/dlg_1234abcd.ogg"]["exists"] is False
    assert by["game/voice/sak_0001.ogg"]["source"] == "game/voice/sak_0001.ogg"
    assert plan["file_count"] == 4 and plan["replace_count"] == 2 and plan["add_count"] == 2 and plan["files_truncated"] is False
    assert plan["files"][0]["exists"] is True  # replaced files first
    assert plan["manifest"] == {"count": 4, "engine": "renpy", "backends": {"mock_tts": 4}, "routes": {"A": 4}}
    assert plan["backups"] == [] and plan["notes"] == []
    assert tree_snapshot(game) == {k: v for k, v in GAME.items()}  # planning never touches the game


def test_plan_uses_the_current_project(renpy: tuple[Any, Path, Path]) -> None:
    from vndub_app import session

    h, _, _ = renpy
    c = client_for(patch_install)
    assert c.get("/api/install/plan").status_code == 404  # no current project
    session.set_current(h.slug)
    try:
        assert c.get("/api/install/plan").json()["engine"] == "renpy"
    finally:
        session.reset()


def test_plan_without_a_finished_patch(o_env: Path, tmp_path: Path) -> None:
    h, game, patch = make_game_project(tmp_path, engine="renpy", game_files=GAME, patch_files={})
    c = client_for(patch_install)
    plan = c.get(f"/api/install/plan?slug={h.slug}").json()
    assert plan["auto_possible"] is False and plan["manual_reason_key"] == "install.manual.no_patch" and plan["files"] == []
    r = post(c, "/api/install/apply", h.slug, backup=True)
    assert r.status_code == 409 and r.json()["error"]["code"] == "nothing_to_install"


def test_plan_when_the_game_folder_is_gone(renpy: tuple[Any, Path, Path]) -> None:
    import shutil

    h, game, _ = renpy
    shutil.rmtree(game)
    c = client_for(patch_install)
    plan = c.get(f"/api/install/plan?slug={h.slug}").json()
    assert plan["game_dir_ok"] is False and plan["auto_possible"] is False and plan["manual_reason_key"] == "install.manual.game_missing"
    r = post(c, "/api/install/apply", h.slug, backup=True)
    assert r.status_code == 409 and r.json()["error"]["code"] == "game_dir_invalid" and r.json()["error"]["action"] == "choose_game_dir"


def test_unvoiced_folder_is_a_manual_note_not_copied(o_env: Path, tmp_path: Path) -> None:
    files = {"voice/sak_0001.ogg": NEW_A, "unvoiced/line_9.ogg": b"unvoiced", "_xp3pack/tmp.bin": b"x"}
    h, game, patch = make_game_project(tmp_path, engine="nscripter", game_files={"voice/sak_0001.ogg": ORIG_A, "nscript.dat": b"x"}, patch_files=files)
    c = client_for(patch_install)
    plan = c.get(f"/api/install/plan?slug={h.slug}").json()
    assert plan["auto_possible"] is True and [f["target"] for f in plan["files"]] == ["voice/sak_0001.ogg"]
    assert plan["notes"][0]["key"] == "install.note.unvoiced_manual" and plan["notes"][0]["params"]["path"].endswith("unvoiced")
    done = run(c, "/api/install/apply", h.slug, backup=True)
    assert done["state"] == "done" and not (game / "unvoiced").exists() and not (game / "_xp3pack").exists()


# ---- apply / rollback ------------------------------------------------------------------------------------------------------


def test_apply_with_backup_then_rollback_restores_everything(renpy: tuple[Any, Path, Path]) -> None:
    h, game, _ = renpy
    c = client_for(patch_install)
    done = run(c, "/api/install/apply", h.slug, backup=True)
    assert done["state"] == "done", done.get("error")
    res = done["result"]
    assert res["files"] == 4 and res["replaced"] == 2 and res["added"] == 2 and res["backup"] is True
    assert done["kind"] == "install_apply" and done["gpu"] is False and done["project"] == h.slug
    after = tree_snapshot(game)
    assert after["game/voice/sak_0001.ogg"] == NEW_A and after["game/voice/har_0001.ogg"] == NEW_B
    assert after["game/voice_dub/dlg_1234abcd.ogg"] == NEW_C and after["game/vndub_auto_voice.rpy"] == RPY
    assert after["game/script.rpy"] == GAME["game/script.rpy"] and after["game/options.rpy"] == GAME["game/options.rpy"]  # untouched
    assert not [p for p in game.rglob("*.vndub-tmp")]
    bdir = appdirs.backups_dir() / h.slug / res["backup_id"]
    man = json.loads((bdir / "manifest.json").read_text(encoding="utf-8"))
    assert man["state"] == "installed" and man["backup"] is True and man["engine"] == "renpy" and man["game_dir"] == str(game)
    assert (bdir / "files" / "game" / "voice" / "sak_0001.ogg").read_bytes() == ORIG_A  # originals
    assert (bdir / "files" / "game" / "voice" / "har_0001.ogg").read_bytes() == ORIG_B
    assert not (bdir / "files" / "game" / "script.rpy").exists() and not (bdir / "files" / "game" / "voice_dub").exists()
    assert {f["target"]: f["existed"] for f in man["files"]}["game/voice_dub/dlg_1234abcd.ogg"] is False
    assert man["created_dirs"] == ["game/voice_dub"]

    plan = c.get(f"/api/install/plan?slug={h.slug}").json()
    assert [b["id"] for b in plan["backups"]] == [res["backup_id"]] and plan["backups"][0]["files"] == 4 and plan["backups"][0]["replaced"] == 2
    assert plan["replace_count"] == 4  # everything is "replaced" now: the patch is installed

    back = run(c, "/api/install/rollback", h.slug, backup_id=res["backup_id"])
    assert back["state"] == "done", back.get("error")
    assert back["result"] == {"backup_id": res["backup_id"], "restored": 2, "removed": 2, "unrecoverable": 0}
    assert tree_snapshot(game) == dict(GAME)  # byte for byte, including the removed new folder
    assert not (game / "game" / "voice_dub").exists()
    assert c.get(f"/api/install/plan?slug={h.slug}").json()["backups"] == []  # rolled back: nothing left to roll back
    again = run(c, "/api/install/rollback", h.slug, backup_id=res["backup_id"])
    assert again["state"] == "failed" and again["error"]["code"] == "nothing_to_install"


def test_apply_twice_keeps_two_backups_and_rolls_back_in_order(renpy: tuple[Any, Path, Path]) -> None:
    h, game, patch = renpy
    c = client_for(patch_install)
    first = run(c, "/api/install/apply", h.slug, backup=True)["result"]["backup_id"]
    (patch / "game" / "voice" / "sak_0001.ogg").write_bytes(b"OggS second version")
    second = run(c, "/api/install/apply", h.slug, backup=True)["result"]["backup_id"]
    assert first != second
    assert [b["id"] for b in c.get(f"/api/install/plan?slug={h.slug}").json()["backups"]] == [second, first]  # newest first
    assert (game / "game/voice/sak_0001.ogg").read_bytes() == b"OggS second version"
    assert run(c, "/api/install/rollback", h.slug, backup_id=second)["state"] == "done"
    assert (game / "game/voice/sak_0001.ogg").read_bytes() == NEW_A  # back to the first install
    assert run(c, "/api/install/rollback", h.slug, backup_id=first)["state"] == "done"
    assert tree_snapshot(game) == dict(GAME)


def test_apply_without_backup_cannot_restore_replaced_files(renpy: tuple[Any, Path, Path]) -> None:
    h, game, _ = renpy
    c = client_for(patch_install)
    done = run(c, "/api/install/apply", h.slug, backup=False)
    assert done["state"] == "done" and done["result"]["backup"] is False
    bdir = appdirs.backups_dir() / h.slug / done["result"]["backup_id"]
    assert not (bdir / "files").exists() or not any((bdir / "files").rglob("*"))
    back = run(c, "/api/install/rollback", h.slug, backup_id=done["result"]["backup_id"])
    assert back["state"] == "done" and back["result"]["unrecoverable"] == 2 and back["result"]["removed"] == 2 and back["result"]["restored"] == 0
    snap = tree_snapshot(game)
    assert snap["game/voice/sak_0001.ogg"] == NEW_A  # the original is gone: reported, never hidden
    assert "game/voice_dub/dlg_1234abcd.ogg" not in snap


def test_a_failure_halfway_rolls_back_automatically(renpy: tuple[Any, Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    h, game, _ = renpy
    real = patch_install._copy_atomic
    calls = {"n": 0}

    def flaky(src: Path, dst: Path) -> None:
        calls["n"] += 1
        if calls["n"] == 3:  # the first two game files are already replaced
            raise PermissionError(13, "Access is denied", str(dst))
        real(src, dst)

    monkeypatch.setattr(patch_install, "_copy_atomic", flaky)
    c = client_for(patch_install)
    done = run(c, "/api/install/apply", h.slug, backup=True)
    assert done["state"] == "failed" and done["error"]["code"] == "permission_denied" and done["error"]["action"] == "show_manual"
    monkeypatch.setattr(patch_install, "_copy_atomic", real)
    assert tree_snapshot(game) == dict(GAME)  # nothing half-installed
    man = json.loads(next((appdirs.backups_dir() / h.slug).iterdir()).joinpath("manifest.json").read_text(encoding="utf-8"))
    assert man["state"] == "rolled_back"
    assert c.get(f"/api/install/plan?slug={h.slug}").json()["backups"] == []


def test_existing_files_are_never_replaced_by_a_corrupt_copy(renpy: tuple[Any, Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    """The copy goes through a temp file: an error mid-copy leaves the original in place."""
    h, game, _ = renpy
    import shutil

    real = shutil.copy2

    def broken(src: Any, dst: Any, **kw: Any) -> Any:
        if str(dst).endswith(".vndub-tmp"):
            Path(dst).write_bytes(b"half")
            raise OSError(28, "No space left on device")
        return real(src, dst, **kw)

    monkeypatch.setattr(patch_install.shutil, "copy2", broken)
    done = run(client_for(patch_install), "/api/install/apply", h.slug, backup=False)
    monkeypatch.setattr(patch_install.shutil, "copy2", real)
    assert done["state"] == "failed"
    snap = tree_snapshot(game)
    assert snap["game/voice/sak_0001.ogg"] == ORIG_A and not [k for k in snap if k.endswith(".vndub-tmp")]


def test_backup_names_are_validated(renpy: tuple[Any, Path, Path]) -> None:
    h, _, _ = renpy
    c = client_for(patch_install)
    for bad in ("../x", "a/b", "..", ".hidden"):
        done = run(c, "/api/install/rollback", h.slug, backup_id=bad)
        assert done["state"] == "failed" and done["error"]["code"] == "bad_request", bad
    missing = run(c, "/api/install/rollback", h.slug, backup_id="20200101-000000")
    assert missing["error"]["code"] == "file_not_found"


def test_targets_cannot_escape_the_game_folder(tmp_path: Path) -> None:
    base = tmp_path / "game"
    base.mkdir()
    assert patch_install._safe_target(base, Path("game/voice/a.ogg")) == base / "game/voice/a.ogg"
    with pytest.raises(AppError) as ei:
        patch_install._safe_target(base, Path("../outside.ogg"))
    assert ei.value.code == "permission_denied"


# ---- engines: automatic or manual, never a false success -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("engine", "reason", "files"),
    [
        ("unity", "install.manual.unity", {"clips/a.wav": b"w", "clips.json": b"{}"}),
        ("siglus", "install.manual.repack", {"koe_dump/KOE(1).ogg": b"o"}),
        ("artemis", "install.manual.repack", {"files/sound/a.ogg": b"o"}),
        ("catsystem2", "install.manual.repack", {"pcm/a.wav": b"o"}),
        ("bgi", "install.manual.loose_unverified", {"voice/a.ogg": b"o"}),
        ("generic", "install.manual.loose_unverified", {"voice/a.ogg": b"o"}),
    ],
)
def test_manual_engines_are_refused_and_never_touch_the_game(o_env: Path, tmp_path: Path, engine: str, reason: str, files: dict[str, bytes]) -> None:
    h, game, _ = make_game_project(tmp_path, engine=engine, game_files={"data/x.bin": b"orig"}, patch_files=files)
    before = tree_snapshot(game)
    c = client_for(patch_install)
    plan = c.get(f"/api/install/plan?slug={h.slug}").json()
    assert plan["auto_possible"] is False and plan["manual_reason_key"] == reason and plan["file_count"] == len(files) and plan["install_md"]
    r = post(c, "/api/install/apply", h.slug, backup=True)
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "install_manual_only" and err["action"] == "show_manual" and err["params"]["engine"] == engine and err["params"]["reason_key"] == reason
    assert jobs.runner().list() == []  # no job: nothing that could claim success
    assert tree_snapshot(game) == before and not (appdirs.backups_dir() / h.slug).exists()


def test_a_manual_engine_cannot_be_applied_directly_either(o_env: Path, tmp_path: Path) -> None:
    h, game, _ = make_game_project(tmp_path, engine="unity", game_files={"data/x.bin": b"orig"}, patch_files={"clips/a.wav": b"w"})
    r = jobs.runner().submit("install_apply", lambda ctx: patch_install.apply_patch(h.slug, True, ctx), gpu=False)
    done = wait_job(r.id)
    assert done["state"] == "failed" and done["error"]["code"] == "install_manual_only"
    assert tree_snapshot(game) == {"data/x.bin": b"orig"}


def test_encrypted_kirikiri_needs_version_dll(o_env: Path, tmp_path: Path) -> None:
    h, game, _ = make_game_project(tmp_path, engine="kirikiri", encrypted=True, game_files={"data.xp3": b"enc", "game.exe": b"MZ"}, patch_files={"patch2.xp3": b"XP3 patch"})
    c = client_for(patch_install)
    plan = c.get(f"/api/install/plan?slug={h.slug}").json()
    assert plan["auto_possible"] is False and plan["manual_reason_key"] == "install.manual.encrypted_kirikiri"
    assert plan["helpers"][0]["name"] == "version.dll" and plan["helpers"][0]["in_game"] is False
    assert any(n["key"] == "install.note.encrypted" for n in plan["notes"])
    assert post(c, "/api/install/apply", h.slug, backup=True).status_code == 409
    assert tree_snapshot(game) == {"data.xp3": b"enc", "game.exe": b"MZ"}
    (game / "version.dll").write_bytes(b"MZ kirikiritools")  # the user installed it
    plan = c.get(f"/api/install/plan?slug={h.slug}").json()
    assert plan["auto_possible"] is True and plan["helpers"][0]["in_game"] is True
    done = run(c, "/api/install/apply", h.slug, backup=True)
    assert done["state"] == "done" and (game / "patch2.xp3").read_bytes() == b"XP3 patch"
    assert run(c, "/api/install/rollback", h.slug, backup_id=done["result"]["backup_id"])["state"] == "done"
    assert not (game / "patch2.xp3").exists()


def test_plain_kirikiri_patch_xp3_installs_next_to_the_exe(o_env: Path, tmp_path: Path) -> None:
    h, game, _ = make_game_project(tmp_path, engine="kirikiri", game_files={"data.xp3": b"x", "patch.xp3": b"old patch"}, patch_files={"patch2.xp3": b"XP3 new"})
    c = client_for(patch_install)
    plan = c.get(f"/api/install/plan?slug={h.slug}").json()
    assert plan["auto_possible"] is True and [f["target"] for f in plan["files"]] == ["patch2.xp3"]
    assert run(c, "/api/install/apply", h.slug, backup=True)["state"] == "done"
    assert (game / "patch2.xp3").read_bytes() == b"XP3 new" and (game / "patch.xp3").read_bytes() == b"old patch"


def test_kirikiri_unencrypted_dir_mode_needs_version_dll(o_env: Path, tmp_path: Path) -> None:
    h, game, _ = make_game_project(tmp_path, engine="kirikiri", game_files={"data.xp3": b"x"}, patch_files={"unencrypted/a.ogg": b"o"})
    c = client_for(patch_install)
    assert c.get(f"/api/install/plan?slug={h.slug}").json()["manual_reason_key"] == "install.manual.encrypted_kirikiri"
    (game / "version.dll").write_bytes(b"MZ")
    plan = c.get(f"/api/install/plan?slug={h.slug}").json()
    assert plan["auto_possible"] is True and plan["files"][0]["target"] == "unencrypted/a.ogg"


def test_renpy_layout_is_checked(o_env: Path, tmp_path: Path) -> None:
    h, _, _ = make_game_project(tmp_path, engine="renpy", game_files={"readme.txt": b"no game folder here"}, patch_files={"game/voice/a.ogg": b"o"})
    plan = client_for(patch_install).get(f"/api/install/plan?slug={h.slug}").json()
    assert plan["auto_possible"] is False and plan["manual_reason_key"] == "install.manual.layout_mismatch"


def test_tyrano_installs_into_data_or_resources_app(o_env: Path, tmp_path: Path) -> None:
    h, game, _ = make_game_project(tmp_path, engine="tyrano", game_files={"resources/app/data/scenario/first.ks": b"ks", "resources/app/data/voice/a.ogg": b"orig"}, patch_files={"data/voice/a.ogg": b"new"})
    c = client_for(patch_install)
    plan = c.get(f"/api/install/plan?slug={h.slug}").json()
    assert plan["auto_possible"] is True and plan["files"][0] == {"target": "resources/app/data/voice/a.ogg", "source": "data/voice/a.ogg", "exists": True, "size": 3}
    assert run(c, "/api/install/apply", h.slug, backup=True)["state"] == "done"
    assert (game / "resources/app/data/voice/a.ogg").read_bytes() == b"new"
    packed, _, _ = make_game_project(tmp_path / "second", engine="tyrano", game_files={"resources/app.asar": b"asar"}, patch_files={"data/voice/a.ogg": b"new"})
    plan2 = c.get(f"/api/install/plan?slug={packed.slug}").json()
    assert plan2["auto_possible"] is False and plan2["manual_reason_key"] == "install.manual.tyrano_packed"


def test_large_plans_are_truncated_in_the_listing_only(o_env: Path, tmp_path: Path) -> None:
    files = {f"voice/v_{i:04d}.ogg": b"o" for i in range(patch_install.PLAN_FILE_LIMIT + 25)}
    h, game, _ = make_game_project(tmp_path, engine="nscripter", game_files={"nscript.dat": b"x"}, patch_files=files)
    c = client_for(patch_install)
    plan = c.get(f"/api/install/plan?slug={h.slug}").json()
    assert plan["files_truncated"] is True and len(plan["files"]) == patch_install.PLAN_FILE_LIMIT and plan["file_count"] == len(files)
    done = run(c, "/api/install/apply", h.slug, backup=True)
    assert done["result"]["files"] == len(files) and len(list((game / "voice").iterdir())) == len(files)


# ---- zip and folder ----------------------------------------------------------------------------------------------------------


def test_zip_for_a_friend_has_the_patch_and_instructions_only(renpy: tuple[Any, Path, Path]) -> None:
    h, game, patch = renpy
    (patch / "_xp3pack").mkdir()
    (patch / "_xp3pack" / "tmp.bin").write_bytes(b"scratch")
    (patch / "unvoiced").mkdir()
    (patch / "unvoiced" / "u.ogg").write_bytes(b"unvoiced audio")
    c = client_for(patch_install)
    done = run(c, "/api/install/zip", h.slug)
    assert done["state"] == "done" and done["kind"] == "install_zip", done.get("error")
    out = Path(done["result"]["path"])
    assert out.parent == h.out and out.suffix == ".zip" and done["result"]["files"] == 6 and done["result"]["size_mb"] >= 0
    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
        assert z.testzip() is None
        assert z.read("vndub-voice-patch/game/voice/sak_0001.ogg") == NEW_A and z.read("vndub-voice-patch/game/vndub_auto_voice.rpy") == RPY
        assert "# Install" in z.read("vndub-voice-patch/INSTALL.md").decode("utf-8")
    assert "vndub-voice-patch/unvoiced/u.ogg" in names  # the friend gets the unvoiced audio too (manual step)
    assert not any("manifest_patch.json" in n or "_xp3pack" in n for n in names)  # local paths / scratch stay home
    assert not any(n.endswith(("script.rpy", "options.rpy")) for n in names) and not any(ORIG_A in z_bytes for z_bytes in [Path(out).read_bytes()])  # no originals
    done2 = run(c, "/api/install/zip", h.slug)  # building it again replaces the file
    assert done2["state"] == "done" and Path(done2["result"]["path"]) == out


def test_zip_without_a_patch_is_an_error(o_env: Path, tmp_path: Path) -> None:
    h, _, patch = make_game_project(tmp_path, engine="renpy", game_files=GAME, patch_files={}, install_md="")  # type: ignore[arg-type]
    for f in patch.iterdir():
        f.unlink()
    done = run(client_for(patch_install), "/api/install/zip", h.slug)
    assert done["state"] == "failed" and done["error"]["code"] == "nothing_to_install"


def test_open_patch_opens_the_folder(renpy: tuple[Any, Path, Path]) -> None:
    h, _, patch = renpy
    opened: list[Path] = []
    openers.set_opener(opened.append)
    r = post(client_for(patch_install), "/api/install/open-patch", h.slug)
    assert r.status_code == 200 and r.json() == {"ok": True, "path": str(patch)} and opened == [patch]


def test_patch_files_helper_skips_metadata_and_temp_folders(tmp_path: Path) -> None:
    (tmp_path / "game").mkdir()
    for rel in ("INSTALL.md", "manifest_patch.json", "game/a.ogg", "_tmp/b.ogg", "unvoiced/c.ogg", ".hidden/d", "game/.keep", "game/x.ogg.vndub-tmp"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    assert [p.relative_to(tmp_path).as_posix() for p in patch_install.patch_files(tmp_path)] == ["game/a.ogg"]
    assert patch_install.patch_files(tmp_path / "missing") == []

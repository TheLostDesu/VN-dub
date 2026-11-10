"""Round 18: one active install per game, read-only targets, a running game, and the empty game folder guard."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

import pytest
from o_helpers import client_for, tree_snapshot
from test_patch_install import GAME, NEW_A, PATCH, RPY, o_env, post, renpy, run  # noqa: F401  (fixtures)

from vndub_app import appdirs, patch_install


@pytest.fixture(autouse=True)
def no_game_running(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(patch_install, "_running_game", lambda game_dir: [])


def shrink_patch(patch: Path) -> None:
    """A re-pack without the generated script and without the extra voice file."""
    (patch / "game" / "vndub_auto_voice.rpy").unlink()
    (patch / "game" / "voice_dub" / "dlg_1234abcd.ogg").unlink()
    (patch / "game" / "voice_dub").rmdir()


# ---- 1. one active install per game ------------------------------------------------------------------------------------


def test_second_smaller_install_leaves_no_stale_file(renpy: tuple[Any, Path, Path]) -> None:
    h, game, patch = renpy
    c = client_for(patch_install)
    assert run(c, "/api/install/apply", h.slug, backup=True)["state"] == "done"
    assert (game / "game/vndub_auto_voice.rpy").exists()
    shrink_patch(patch)
    (patch / "manifest_patch.json").write_text("{}", encoding="utf-8")
    second = run(c, "/api/install/apply", h.slug, backup=True)
    assert second["state"] == "done", second.get("error")
    assert not (game / "game/vndub_auto_voice.rpy").exists() and not (game / "game/voice_dub").exists()
    assert run(c, "/api/install/rollback", h.slug, backup_id=second["result"]["backup_id"])["state"] == "done"
    assert tree_snapshot(game) == dict(GAME)


def test_install_twice_then_rollback_gives_the_original_game(renpy: tuple[Any, Path, Path]) -> None:
    h, game, _ = renpy
    c = client_for(patch_install)
    run(c, "/api/install/apply", h.slug, backup=True)
    second = run(c, "/api/install/apply", h.slug, backup=True)["result"]["backup_id"]
    assert [b["id"] for b in patch_install.list_backups(h.slug)] == [second]
    assert run(c, "/api/install/rollback", h.slug, backup_id=second)["state"] == "done"
    assert tree_snapshot(game) == dict(GAME)


def test_wrong_order_rollback_cascades_newest_first(renpy: tuple[Any, Path, Path]) -> None:
    """Old data can hold two active backups: rolling back the older one undoes the newer one first, so the game ends as the original."""
    h, game, _ = renpy
    c = client_for(patch_install)
    first = run(c, "/api/install/apply", h.slug, backup=True)["result"]["backup_id"]
    # a second active backup made the way 1.1.0 did it: installed over the first install, without undoing it
    from vndub_app.jobs import JobContext  # noqa: F401

    class Ctx:
        def message(self, *a: Any, **k: Any) -> None: ...
        def progress(self, *a: Any, **k: Any) -> None: ...
        def checkpoint(self) -> None: ...
        def log(self, *a: Any, **k: Any) -> None: ...

    monkey_active = patch_install._active_backups
    patch_install._active_backups = lambda slug, game_dir=None: []  # type: ignore[assignment]
    try:
        second = patch_install.apply_patch(h.slug, True, Ctx())["backup_id"]  # type: ignore[arg-type]
    finally:
        patch_install._active_backups = monkey_active  # type: ignore[assignment]
    assert {b["id"] for b in patch_install.list_backups(h.slug)} == {first, second}
    done = run(c, "/api/install/rollback", h.slug, backup_id=first)
    assert done["state"] == "done", done.get("error")
    assert done["result"]["also_rolled_back"] == [second]
    assert tree_snapshot(game) == dict(GAME)
    assert patch_install.list_backups(h.slug) == []


@pytest.mark.parametrize("fail_at", [1, 3])
def test_failure_in_second_install_never_leaves_a_mix(renpy: tuple[Any, Path, Path], monkeypatch: pytest.MonkeyPatch, fail_at: int) -> None:
    h, game, patch = renpy
    c = client_for(patch_install)
    run(c, "/api/install/apply", h.slug, backup=True)
    after_first = tree_snapshot(game)
    real = patch_install._copy_atomic
    calls = {"n": 0}

    def flaky(src: Path, dst: Path) -> None:
        if patch in src.parents:  # only the install's own copies, not the restore of the first install
            calls["n"] += 1
            if calls["n"] == fail_at:
                raise OSError("disk went away")
        real(src, dst)

    monkeypatch.setattr(patch_install, "_copy_atomic", flaky)
    (patch / "game" / "voice" / "sak_0001.ogg").write_bytes(b"OggS second version")
    job = run(c, "/api/install/apply", h.slug, backup=True)
    assert job["state"] == "failed"
    snap = tree_snapshot(game)
    active = patch_install.list_backups(h.slug)
    # the game is the original (the first install was undone) or the first install; a dubbed game always has its backup
    assert snap == dict(GAME) or (snap == after_first and active)
    if snap != dict(GAME):
        assert active


def test_cancel_in_second_install_restores_the_original(renpy: tuple[Any, Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    from vndub_app.jobs import JobCancelled

    h, game, patch = renpy
    c = client_for(patch_install)
    run(c, "/api/install/apply", h.slug, backup=True)
    real = patch_install._copy_atomic
    n = {"i": 0}

    def cancelling(src: Path, dst: Path) -> None:
        if patch in src.parents:
            n["i"] += 1
            if n["i"] == 2:
                raise JobCancelled("cancelled by user")
        real(src, dst)

    monkeypatch.setattr(patch_install, "_copy_atomic", cancelling)
    job = run(c, "/api/install/apply", h.slug, backup=True)
    assert job["state"] in ("cancelled", "failed")
    assert tree_snapshot(game) == dict(GAME)
    assert patch_install.list_backups(h.slug) == []


def test_failed_undo_of_the_previous_install_stops_before_touching_anything(renpy: tuple[Any, Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    h, game, _ = renpy
    c = client_for(patch_install)
    run(c, "/api/install/apply", h.slug, backup=True)
    after_first = tree_snapshot(game)
    monkeypatch.setattr(patch_install, "_restore", lambda *a, **k: (_ for _ in ()).throw(PermissionError("nope")))
    job = run(c, "/api/install/apply", h.slug, backup=True)
    assert job["state"] == "failed" and job["error"]["code"] == "install_undo_failed"
    assert tree_snapshot(game) == after_first
    assert len(patch_install.list_backups(h.slug)) == 1


# ---- 2. read-only targets, running game -------------------------------------------------------------------------------


def test_read_only_target_is_replaced_and_the_attribute_comes_back(renpy: tuple[Any, Path, Path]) -> None:
    h, game, _ = renpy
    target = game / "game/voice/sak_0001.ogg"
    os.chmod(target, stat.S_IREAD)
    c = client_for(patch_install)
    try:
        done = run(c, "/api/install/apply", h.slug, backup=True)
        assert done["state"] == "done", done.get("error")
        assert target.read_bytes() == NEW_A
        man = appdirs.read_json(appdirs.backups_dir() / h.slug / done["result"]["backup_id"] / "manifest.json", {})
        assert {f["target"]: f.get("readonly") for f in man["files"]}["game/voice/sak_0001.ogg"] is True
        assert run(c, "/api/install/rollback", h.slug, backup_id=done["result"]["backup_id"])["state"] == "done"
        assert target.read_bytes() == GAME["game/voice/sak_0001.ogg"]
        assert not target.stat().st_mode & stat.S_IWRITE  # read-only again, as the game had it
    finally:
        os.chmod(target, stat.S_IWRITE | stat.S_IREAD)


def test_running_game_refuses_apply_and_rollback(renpy: tuple[Any, Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    h, game, _ = renpy
    c = client_for(patch_install)
    done = run(c, "/api/install/apply", h.slug, backup=True)
    after = tree_snapshot(game)
    monkeypatch.setattr(patch_install, "_running_game", lambda game_dir: ["Game.exe"])
    for path, body in (("/api/install/apply", {"backup": True}), ("/api/install/rollback", {"backup_id": done["result"]["backup_id"]})):
        job = run(c, path, h.slug, **body)
        assert job["state"] == "failed" and job["error"]["code"] == "game_running", job
    assert tree_snapshot(game) == after


def test_running_game_lookup_never_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from vndub_app import winproc

    def boom(roots: Any) -> Any:
        raise RuntimeError("no process list")

    monkeypatch.setattr(winproc, "list_processes", boom)
    assert patch_install._running_game(tmp_path) == []


def test_permission_error_while_the_game_runs_is_game_running(renpy: tuple[Any, Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    h, game, _ = renpy
    c = client_for(patch_install)
    state = {"seen": 0}

    def locked(game_dir: Path) -> list[str]:
        state["seen"] += 1
        return [] if state["seen"] == 1 else ["Game.exe"]  # started between the check and the copy

    monkeypatch.setattr(patch_install, "_running_game", locked)
    monkeypatch.setattr(patch_install, "_copy_atomic", lambda s, d: (_ for _ in ()).throw(PermissionError(5, "denied", str(d))))
    job = run(c, "/api/install/apply", h.slug, backup=True)
    assert job["state"] == "failed" and job["error"]["code"] == "game_running"
    assert tree_snapshot(game) == dict(GAME)


# ---- 3. empty game folder -----------------------------------------------------------------------------------------------


def test_empty_game_folder_is_never_the_current_folder(renpy: tuple[Any, Path, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from vndub.config import load_project_yaml, save_project_yaml

    h, game, patch = renpy
    f = h.work / "project.yaml"
    proj = load_project_yaml(f)
    proj.game_dir = ""
    save_project_yaml(f, proj)
    meta = appdirs.read_json(h.meta_file, {})
    meta["game_dir"] = ""
    appdirs.write_json_atomic(h.meta_file, meta)
    cwd = tmp_path / "cwd"
    (cwd / "game").mkdir(parents=True)
    monkeypatch.chdir(cwd)
    c = client_for(patch_install)
    plan = c.get(f"/api/install/plan?slug={h.slug}").json()
    assert plan["auto_possible"] is False and plan["game_dir_ok"] is False and plan["manual_reason_key"] == "install.manual.game_missing"
    r = post(c, "/api/install/apply", h.slug, backup=True)
    assert r.status_code == 409 and r.json()["error"]["code"] == "game_dir_invalid"
    assert not any((cwd / "game").iterdir())


def test_relative_game_dir_is_refused() -> None:
    assert patch_install._game_dir_ok(Path("")) is False
    assert patch_install._game_dir_ok(Path("game")) is False
    assert patch_install._game_dir_ok(Path(__file__).parent) is True
    assert patch_install._game_dir_ok(Path(__file__).parent / "nope") is False

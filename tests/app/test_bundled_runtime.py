"""Git, the Visual C++ Redistributable and the tools (bgi-extractor) that the installer carries: lookups, child environment, Doctor fixes, install job.

The packaging side (build script, VNDub.iss) is in ``test_packaging.py``; the Doctor rows are in ``test_doctor.py``.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Iterator

import pytest
from o_helpers import client_for, drop_runner, fresh_runner, make_env, wait_job

from vndub_app import appdirs, doctor, installer
from vndub_app.errors import AppError

GIT_EXE = "git.exe" if os.name == "nt" else "git"


@pytest.fixture
def o_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    data = make_env(monkeypatch, tmp_path)
    fresh_runner()
    yield data
    doctor.set_probes(None)
    drop_runner()


def fake_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, git: bool = True, redist: bool = True) -> Path:
    """A packaged-app layout: <app>/VNDub.exe, <app>/git/cmd/git.exe, <app>/redist/vc_redist.x64.exe; ``sys`` says frozen."""
    app = tmp_path / "app"
    app.mkdir()
    (app / "VNDub.exe").write_bytes(b"MZ")
    if git:
        (app / "git" / "cmd").mkdir(parents=True)
        (app / "git" / "cmd" / GIT_EXE).write_bytes(b"MZ")
    if redist:
        (app / "redist").mkdir()
        (app / "redist" / "vc_redist.x64.exe").write_bytes(b"MZ")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(app / "VNDub.exe"))
    return app.resolve()


def which_returns(monkeypatch: pytest.MonkeyPatch, git: Path | None) -> None:
    real = shutil.which
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: (str(git) if git else None) if name == "git" else real(name, *a, **k))


# ---- where git comes from -----------------------------------------------------------------------------------------


def test_git_on_path_wins_over_the_bundled_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = fake_app(tmp_path, monkeypatch)
    system = tmp_path / "system" / GIT_EXE
    system.parent.mkdir()
    system.write_bytes(b"MZ")
    which_returns(monkeypatch, system)
    assert installer.find_git() == system
    assert installer.bundled_git() == app / "git" / "cmd" / GIT_EXE  # still known, for the Doctor row


def test_bundled_git_is_used_when_path_has_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = fake_app(tmp_path, monkeypatch)
    which_returns(monkeypatch, None)
    assert installer.find_git() == app / "git" / "cmd" / GIT_EXE


def test_no_git_anywhere_is_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_app(tmp_path, monkeypatch, git=False)  # the folder next to the exe was deleted
    which_returns(monkeypatch, None)
    assert installer.find_git() is None and installer.bundled_git() is None


def test_a_dev_run_never_looks_for_a_bundled_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = fake_app(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "frozen", False, raising=False)  # not packaged: sys.executable is python.exe
    which_returns(monkeypatch, None)
    assert (app / "git" / "cmd" / GIT_EXE).is_file()
    assert installer.bundled_git() is None and installer.bundled_vcredist() is None and installer.find_git() is None


# ---- the environment of make_envs.ps1 / uv ----------------------------------------------------------------------------


def test_child_env_puts_the_bundled_git_on_path(o_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = fake_app(tmp_path, monkeypatch)
    monkeypatch.setattr(installer, "find_uv", lambda: None)
    which_returns(monkeypatch, None)
    path = installer._child_env()["PATH"].split(os.pathsep)
    assert path[0] == str(app / "git" / "cmd")
    assert path[1:] == os.environ["PATH"].split(os.pathsep)  # nothing else changed


def test_child_env_leaves_path_alone_when_the_user_has_git(o_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = fake_app(tmp_path, monkeypatch)
    monkeypatch.setattr(installer, "find_uv", lambda: None)
    which_returns(monkeypatch, tmp_path / "usr" / "bin" / GIT_EXE)
    env = installer._child_env()
    assert env["PATH"] == os.environ["PATH"] and str(app / "git" / "cmd") not in env["PATH"]


def test_child_env_in_a_dev_run_adds_no_git(o_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer, "find_uv", lambda: None)
    which_returns(monkeypatch, None)
    assert installer._child_env()["PATH"] == os.environ["PATH"]


def test_child_env_bundled_git_comes_before_the_uv_folder(o_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = fake_app(tmp_path, monkeypatch)
    uv = tmp_path / "bin" / "uv.exe"
    monkeypatch.setattr(installer, "find_uv", lambda: uv)
    which_returns(monkeypatch, None)
    path = installer._child_env({"X_EXTRA": "1"})["PATH"].split(os.pathsep)
    assert path[:2] == [str(app / "git" / "cmd"), str(uv.parent)] and installer._child_env({"X_EXTRA": "1"})["X_EXTRA"] == "1"


@pytest.mark.skipif(os.name != "nt", reason="the bundled MinGit is a part of the Windows app")
def test_a_plain_git_resolves_to_the_bundled_one_from_the_child_path(o_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """What make_envs.ps1 and uv do: look `git` up on the PATH they were started with."""
    real_which = shutil.which
    app = fake_app(tmp_path, monkeypatch)
    monkeypatch.setattr(installer, "find_uv", lambda: None)
    which_returns(monkeypatch, None)  # the user's PATH has no git
    env = installer._child_env()
    found = real_which("git", path=env["PATH"])
    assert found is not None and Path(found).resolve() == (app / "git" / "cmd" / GIT_EXE).resolve()


# ---- envs that clone repositories -------------------------------------------------------------------------------------


def test_envs_using_git_include_the_optional_extras() -> None:
    """rvc / seedvc clone in their own file; main (voxcpm, utmosv2 via git+https) and funasr (cosyvoice) in extras."""
    users = installer.envs_using_git()
    assert {"rvc", "seedvc", "main", "funasr"} <= set(users)
    assert "gguf" not in users and "chatterbox" not in users
    assert installer.env_needs_git("rvc") and installer.env_needs_git("seedvc") and not installer.env_needs_git("main")  # git_missing stays on the env's own file


def test_git_requirement_lines_count_as_git(tmp_path: Path) -> None:
    assert installer._text_needs_git("# @git Applio https://github.com/x/y 3.7.0\n")
    assert installer._text_needs_git("utmosv2 @ git+https://github.com/x/y.git@abc\ntimm==1.0.20\n")
    assert not installer._text_needs_git("# comment about git+https://example\nnumpy==2.2.6\n")
    assert not installer._text_needs_git("legit+https://x\n")


def test_env_install_runs_with_the_bundled_git(o_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Packaged app, no Git on PATH: install_envs does not stop with git_missing and hands make_envs a PATH that has git."""
    from vndub_app import jobs

    app = fake_app(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "_MEIPASS", str(Path(__file__).resolve().parents[2]), raising=False)  # the envs/*.txt of the repo stand in for the bundle
    which_returns(monkeypatch, None)
    monkeypatch.setattr(installer, "find_uv", lambda: Path("uv"))
    monkeypatch.setattr(installer, "ensure_python", lambda version, ctx: Path("python"))  # no `uv python install` here
    seen: dict[str, Any] = {}

    def fake_stream(cmd: list[str], ctx: Any, **kw: Any) -> tuple[int, list[str]]:
        seen["path"] = (kw.get("env") or {}).get("PATH", "")
        (installer.envs_dir() / "rvc").mkdir(parents=True, exist_ok=True)
        (installer.envs_dir() / "rvc" / "pyvenv.cfg").write_text("home = x\n", encoding="utf-8")
        scripts = installer.envs_dir() / "rvc" / ("Scripts" if os.name == "nt" else "bin")
        scripts.mkdir(parents=True, exist_ok=True)
        (scripts / ("python.exe" if os.name == "nt" else "python")).write_bytes(b"MZ")
        return 0, ["== env rvc"]

    monkeypatch.setattr(installer, "run_stream", fake_stream)
    job = jobs.runner().submit("envs_install", lambda ctx: installer.install_envs(["rvc"], "cpu", ctx), gpu=False)
    done = wait_job(job.id)
    assert done["state"] == "done", done.get("error")
    assert seen["path"].split(os.pathsep)[0] == str(app / "git" / "cmd")


# ---- the bundled redistributable ------------------------------------------------------------------------------------


def test_bundled_vcredist_is_found_next_to_the_exe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = fake_app(tmp_path, monkeypatch)
    if os.name == "nt":
        assert installer.bundled_vcredist() == app / "redist" / "vc_redist.x64.exe"
    (app / "redist" / "vc_redist.x64.exe").unlink()
    assert installer.bundled_vcredist() is None


def run_job(fake: Any) -> dict[str, Any]:
    from vndub_app import jobs

    job = jobs.runner().submit("vcredist_install", lambda ctx: installer.install_vcredist(ctx, run=fake), gpu=False)
    return wait_job(job.id)


@pytest.mark.skipif(os.name != "nt", reason="the redistributable is Windows only")
@pytest.mark.parametrize(("code", "reboot"), [(0, False), (1638, False), (3010, True)])
def test_vcredist_job_succeeds_on_0_1638_3010(o_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int, reboot: bool) -> None:
    app = fake_app(tmp_path, monkeypatch)
    calls: list[list[str]] = []

    def fake(cmd: list[str], ctx: Any) -> int:
        calls.append(cmd)
        return code

    done = run_job(fake)
    assert done["state"] == "done", done.get("error")
    assert calls == [[str(app / "redist" / "vc_redist.x64.exe"), "/install", "/passive", "/norestart"]]
    assert done["result"]["exit_code"] == code and done["result"]["reboot_required"] is reboot
    assert done["message_key"] == ("installer.msg.vcredist_reboot" if reboot else "installer.msg.vcredist")


@pytest.mark.skipif(os.name != "nt", reason="the redistributable is Windows only")
@pytest.mark.parametrize("code", [1223, 1602, 5, 1603])
def test_vcredist_job_fails_with_a_retry_when_declined_or_broken(o_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int) -> None:
    fake_app(tmp_path, monkeypatch)
    done = run_job(lambda cmd, ctx: code)
    assert done["state"] == "failed"
    assert done["error"]["code"] == "vcredist_failed" and done["error"]["params"] == {"exit_code": code} and done["error"]["action"] == "retry"


@pytest.mark.skipif(os.name != "nt", reason="the redistributable is Windows only")
def test_vcredist_job_without_the_file_is_file_not_found(o_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_app(tmp_path, monkeypatch, redist=False)
    done = run_job(lambda cmd, ctx: 0)
    assert done["state"] == "failed" and done["error"]["code"] == "file_not_found"


@pytest.mark.skipif(os.name != "nt", reason="installer exit codes above 255 (3010, 1223) exist on Windows only")
def test_run_waiting_returns_the_exit_code_of_a_real_process() -> None:
    class Ctx:
        cancelled = False

        def checkpoint(self) -> None:
            return None

        def watch_process(self, proc: Any) -> Any:
            return lambda: None

    for code in (0, 3010, 1223):
        assert installer._run_waiting([sys.executable, "-c", f"import sys; sys.exit({code})"], Ctx()) == code  # type: ignore[arg-type]


# ---- tools bundled next to the exe (manifest "bundled": bgi-extractor) ------------------------------------------------

BGI = Path("bin") / "bgi-extractor" / "bgi-extractor.exe"


def bgi_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    app = fake_app(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "_MEIPASS", str(Path(__file__).resolve().parents[2]), raising=False)  # the repo's app/tools_manifest.json stands in for the bundle
    (app / BGI).parent.mkdir(parents=True)
    (app / BGI).write_bytes(b"MZ")
    return app


def test_the_bundled_bgi_extractor_counts_as_installed(o_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = bgi_app(tmp_path, monkeypatch)
    e = installer.get_entry("bgi_extractor")
    assert installer.bundled_paths(e) == {"bgi_extractor": app / BGI}
    assert installer.tool_paths()["bgi_extractor"] == str(app / BGI)
    assert installer.tools_object().resolve("bgi_extractor") == app / BGI
    row = {t["name"]: t for t in installer.list_tools()}["bgi_extractor"]
    assert row["present"] is True and row["path"] == str(app / BGI)


def test_a_copy_in_the_tools_folder_wins_over_the_bundled_one(o_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bgi_app(tmp_path, monkeypatch)
    e = installer.get_entry("bgi_extractor")
    own = installer.target_dir(e) / "bgi-extractor.exe"
    own.parent.mkdir(parents=True, exist_ok=True)
    own.write_bytes(b"MZ built by me")
    assert installer.provided_paths(e) == {"bgi_extractor": own}
    assert installer.tools_object().resolve("bgi_extractor") == own


def test_no_bundled_tool_in_a_dev_run_or_when_the_file_is_gone(o_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = bgi_app(tmp_path, monkeypatch)
    e = installer.get_entry("bgi_extractor")
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    assert installer.bundled_paths(e) == {} and installer.provided_paths(e) == {}
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    (app / BGI).unlink()
    assert installer.bundled_paths(e) == {} and "bgi_extractor" not in installer.tool_paths()


def test_installing_a_bundled_tool_succeeds_without_writing_its_path(o_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The bundled copy is found anew at every start (the app folder may move): tools.local.yaml does not pin it."""
    from vndub_app import jobs

    bgi_app(tmp_path, monkeypatch)
    job = jobs.runner().submit("tools_install", lambda ctx: installer.install_tools(["bgi_extractor"], ctx), gpu=False)
    done = wait_job(job.id)
    assert done["state"] == "done", done.get("error")
    assert done["result"]["installed"] == ["bgi_extractor"] and all(f["skipped"] for f in done["result"]["files"])
    assert "bgi_extractor" not in installer.read_local_tools()


def test_project_configs_get_the_bundled_tool(o_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import yaml

    app = bgi_app(tmp_path, monkeypatch)
    game = tmp_path / "game"
    game.mkdir()
    h = appdirs.create_project(game)
    assert yaml.safe_load(h.config_file.read_text(encoding="utf-8"))["tools"]["bgi_extractor"] == str(app / BGI)


# ---- the Doctor fix actions -------------------------------------------------------------------------------------------


def test_open_vcredist_page_returns_the_microsoft_permalink(o_env: Path) -> None:
    c = client_for(doctor)
    r = c.post("/api/doctor/fix/open_vcredist_page").json()
    assert r == {"ok": True, "url": "https://aka.ms/vs/17/release/vc_redist.x64.exe"}


@pytest.mark.skipif(os.name != "nt", reason="the redistributable is Windows only")
def test_install_vcredist_fix_is_a_job_that_runs_the_bundled_exe(o_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = fake_app(tmp_path, monkeypatch)
    calls: list[list[str]] = []
    monkeypatch.setattr(installer, "_run_waiting", lambda cmd, ctx: calls.append(cmd) or 0)
    c = client_for(doctor)
    job = c.post("/api/doctor/fix/install_vcredist").json()
    assert job["kind"] == "vcredist_install" and job["gpu"] is False
    done = wait_job(job)
    assert done["state"] == "done" and calls[0][0] == str(app / "redist" / "vc_redist.x64.exe")
    # the row offers exactly this fix while the file is there
    row = doctor.row_vcredist("win32", doctor.VcRuntime(False, None, "none"), installer.bundled_vcredist() is not None)
    assert row["fix"]["action"] == "install_vcredist"


def test_install_vcredist_fix_without_the_file_is_a_404(o_env: Path) -> None:
    c = client_for(doctor)  # a dev run: nothing bundled
    r = c.post("/api/doctor/fix/install_vcredist")
    assert r.status_code == 404 and r.json()["error"]["code"] == "file_not_found"


def test_every_fix_action_is_documented_and_labelled() -> None:
    api = (Path(__file__).resolve().parents[2] / "vndub_app" / "API.md").read_text(encoding="utf-8")
    m = re.search(r"row ids: ((?:`\w+`,?\s*)+)", api)
    assert m and re.findall(r"`(\w+)`", m.group(1)) == list(doctor.ROW_IDS)
    for action in doctor.FIX_ACTIONS:
        assert action in api, action
    assert "vcredist_install" in api and "vcredist_failed" in api


def test_apply_fix_rejects_unknown_actions(o_env: Path) -> None:
    with pytest.raises(AppError):
        doctor.apply_fix("run_vc_redist_from_anywhere")
    assert appdirs.data_dir()  # the throw-away data folder, never the real one

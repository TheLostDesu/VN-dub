"""Where the app puts things the user can move: environments (+ uv's Python builds and cache), models; and the
processes the installer starts (killed when the job is cancelled).

* ``uv_dirs`` / ``_child_env``: uv's cache and CPython folder live below the envs folder (same volume as the venvs, so
  uv hardlinks wheels instead of storing torch once per env);
* ``ensure_python`` / ``concrete_python``: the interpreter handed to ``uv venv --python`` is a real folder, never uv's
  minor-version junction (os error 448 on some Windows machines);
* ``Settings.envs_dir`` / ``models_dir`` through ``PUT /api/settings`` and the ``paths`` of ``GET /api/settings``;
* ``run_stream`` / ``_run_waiting`` end their child process tree when the job is cancelled.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from o_helpers import client_for, drop_runner, fresh_runner, make_env, wait_job

from vndub_app import appdirs, installer, jobs, winproc
from vndub_app.routes import general

PATH_KEYS = {"data", "models", "envs", "tools", "python", "uv_cache"}
EXE_NAME = "python.exe" if os.name == "nt" else "bin/python3"


@pytest.fixture
def p_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Throw-away data folder; ``VNDUB_ENVS_DIR`` is set (as an outside variable) to ``<tmp>/envs`` like in the other app tests."""
    data = make_env(monkeypatch, tmp_path)
    fresh_runner()
    yield data
    installer.reset_environment_memory()
    drop_runner()


@pytest.fixture
def s_env(p_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Like ``p_env`` but with no ``VNDUB_ENVS_DIR`` at all and no dev checkout ``.venvs``: the app decides by itself."""
    monkeypatch.delenv("VNDUB_ENVS_DIR")
    monkeypatch.setattr(installer, "repo_envs_dir", lambda: tmp_path / "no-dev-venvs")
    installer.reset_environment_memory()
    yield p_env


def make_cpython(root: Path, name: str) -> Path:
    exe = root / name / EXE_NAME
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_bytes(b"")
    return exe


def make_link(link: Path, target: Path) -> bool:
    """A directory junction on Windows (what uv's minor-version link is), a symlink elsewhere. False when not possible here."""
    link.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True)
        return r.returncode == 0 and link.exists()
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        return False
    return True


def python_root() -> Path:
    return Path(installer.uv_dirs()["UV_PYTHON_INSTALL_DIR"])


def run_job(body: Callable[[Any], Any], timeout: float = 30.0) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run ``body(ctx)`` as a job; returns ``(final job info, {"value": result})``."""
    out: dict[str, Any] = {}

    def fn(ctx: Any) -> None:
        out["value"] = body(ctx)

    job = jobs.runner().submit("envs_install", fn, gpu=False)
    return wait_job(job.id, timeout=timeout), out


def pid_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(k32.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ---- uv's folders ------------------------------------------------------------------------------------------------------


def test_uv_dirs_are_below_the_envs_folder_and_reach_every_child(p_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    envs = p_env.parent / "envs"
    d = installer.uv_dirs()
    assert d == {"UV_CACHE_DIR": str(envs / "_uv" / "cache"), "UV_PYTHON_INSTALL_DIR": str(envs / "_uv" / "python")}
    assert installer.uv_dirs(Path("E:/somewhere")) == {"UV_CACHE_DIR": str(Path("E:/somewhere") / "_uv" / "cache"), "UV_PYTHON_INSTALL_DIR": str(Path("E:/somewhere") / "_uv" / "python")}
    monkeypatch.setenv("UV_CACHE_DIR", str(p_env.parent / "users-own-cache"))  # the app owns these two folders
    monkeypatch.setenv("UV_PYTHON_INSTALL_DIR", str(p_env.parent / "users-own-python"))
    env = installer._child_env({"X": "1"})
    assert env["UV_CACHE_DIR"] == d["UV_CACHE_DIR"] and env["UV_PYTHON_INSTALL_DIR"] == d["UV_PYTHON_INSTALL_DIR"]
    assert env["VNDUB_ENVS_DIR"] == str(envs) and env["X"] == "1"
    assert Path(d["UV_CACHE_DIR"]).is_dir() and Path(d["UV_PYTHON_INSTALL_DIR"]).is_dir()  # created
    # the same volume as the venvs: that is the whole point (hardlinks only work inside one volume)
    assert Path(d["UV_CACHE_DIR"]).parent.parent == envs and os.path.splitdrive(d["UV_CACHE_DIR"])[0] == os.path.splitdrive(str(envs))[0]
    assert os.environ["UV_CACHE_DIR"] != d["UV_CACHE_DIR"]  # only the children are changed, not this process


def test_the_uv_folder_is_not_an_environment(p_env: Path) -> None:
    installer._child_env()  # creates <envs>/_uv/...
    assert (p_env.parent / "envs" / "_uv").is_dir()
    assert "_uv" not in installer.env_names()
    assert "_uv" not in [r["name"] for r in installer.list_envs(with_size=False)]
    assert set(installer.env_names()) == {r["name"] for r in installer.list_envs()}


# ---- the interpreter for the venvs -------------------------------------------------------------------------------------


def test_concrete_python_takes_the_highest_patch_of_real_folders_only(p_env: Path) -> None:
    root = python_root()
    make_cpython(root, "cpython-3.11.4-windows-x86_64-none")
    best = make_cpython(root, "cpython-3.11.10-windows-x86_64-none")  # 10 > 9: numeric, not alphabetical
    make_cpython(root, "cpython-3.11.9-windows-x86_64-none")
    other = make_cpython(root, "cpython-3.12.1-windows-x86_64-none")
    (root / "cpython-3.11.50-windows-x86_64-none").mkdir()  # a folder without an interpreter
    (root / "unrelated").mkdir()
    assert installer.concrete_python("3.11") == best
    assert installer.concrete_python("3.12") == other
    assert installer.concrete_python("3.11.9") == root / "cpython-3.11.9-windows-x86_64-none" / EXE_NAME  # an exact patch
    assert installer.concrete_python("3.10") is None
    assert installer.concrete_python("3.11", p_env / "nowhere") is None
    # links never count, however high their patch number or however they are named
    elsewhere = make_cpython(p_env.parent, "elsewhere")
    if not make_link(root / "cpython-3.11.99-windows-x86_64-none", elsewhere.parent):
        pytest.skip("cannot create a junction / symlink here")
    assert make_link(root / "cpython-3.11-windows-x86_64-none", best.parent)
    assert installer._is_link(root / "cpython-3.11.99-windows-x86_64-none") and installer._is_link(root / "cpython-3.11-windows-x86_64-none")
    assert not installer._is_link(best.parent)
    assert installer.concrete_python("3.11") == best


def test_ensure_python_without_installing_when_the_interpreter_is_there(p_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The real folder is enough: uv is not started (offline machines, no needless work), and uv's minor-version link
    is deleted (its target stays), so a venv made now cannot end up depending on it."""
    root = python_root()
    exe = make_cpython(root, "cpython-3.11.9-windows-x86_64-none")
    link = root / "cpython-3.11-windows-x86_64-none"
    if not make_link(link, exe.parent):
        pytest.skip("cannot create a junction / symlink here")
    monkeypatch.setattr(installer, "find_uv", lambda: None)  # not even needed
    monkeypatch.setattr(installer, "run_stream", lambda *a, **k: pytest.fail("uv must not be started"))
    done, out = run_job(lambda ctx: installer.ensure_python("3.11", ctx))
    assert done["state"] == "done", done.get("error")
    assert out["value"] == exe
    assert not os.path.lexists(link)  # the link is gone ...
    assert exe.is_file()  # ... and what it pointed to is untouched


def test_ensure_python_installs_with_uv_then_drops_the_link_it_made(p_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = python_root()
    seen: list[list[str]] = []

    def fake_stream(cmd: list[str], ctx: Any, **kw: Any) -> tuple[int, list[str]]:
        seen.append(cmd)
        exe = make_cpython(root, "cpython-3.11.17-windows-x86_64-none")
        make_link(root / "cpython-3.11-windows-x86_64-none", exe.parent)  # what `uv python install` leaves behind
        return 0, ["Installed Python 3.11.17"]

    monkeypatch.setattr(installer, "find_uv", lambda: Path("uv"))
    monkeypatch.setattr(installer, "run_stream", fake_stream)
    done, out = run_job(lambda ctx: installer.ensure_python("3.11", ctx))
    assert done["state"] == "done", done.get("error")
    assert out["value"] == root / "cpython-3.11.17-windows-x86_64-none" / EXE_NAME
    assert seen == [["uv", "python", "install", "3.11", "--no-bin", "--no-registry"]]  # no ~/.local/bin entries, no registry keys
    assert not any(installer._MINOR_LINK.match(p.name) and installer._is_link(p) for p in root.iterdir())
    assert done["message_key"] == "installer.msg.python"
    again, out2 = run_job(lambda ctx: installer.ensure_python("3.11", ctx))  # second time: nothing to start
    assert again["state"] == "done" and out2["value"] == out["value"] and len(seen) == 1


def test_ensure_python_survives_an_install_that_fails_after_unpacking(p_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """uv exits non-zero when it cannot link the executable into ~/.local/bin, although the interpreter is in place."""
    root = python_root()

    def fake_stream(cmd: list[str], ctx: Any, **kw: Any) -> tuple[int, list[str]]:
        make_cpython(root, "cpython-3.11.17-windows-x86_64-none")
        return 1, ["error: Failed to install python3.11.exe", "  Caused by: Executable already exists but is not managed by uv"]

    monkeypatch.setattr(installer, "find_uv", lambda: Path("uv"))
    monkeypatch.setattr(installer, "run_stream", fake_stream)
    done, out = run_job(lambda ctx: installer.ensure_python("3.11", ctx))
    assert done["state"] == "done", done.get("error")
    assert out["value"] == root / "cpython-3.11.17-windows-x86_64-none" / EXE_NAME
    assert any("WARNING" in l and "exited with 1" in l for l in done["log_tail"])


def test_ensure_python_error_mapping(p_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer, "find_uv", lambda: Path("uv"))
    link = r"C:\Users\Ivan\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none"
    blocked = [
        f"error: Failed to inspect Python interpreter from managed installations at `{link}\\python.exe`",
        "  Caused by: Failed to query Python interpreter at `...`",
        "  Caused by: The path cannot be traversed because it contains an untrusted mount point. (os error 448)",
    ]
    monkeypatch.setattr(installer, "run_stream", lambda cmd, ctx, **kw: (2, blocked))
    done, _ = run_job(lambda ctx: installer.ensure_python("3.12", ctx))
    err = done["error"]
    assert done["state"] == "failed" and err["code"] == "python_junction_blocked" and err["action"] == "retry"
    assert err["params"] == {"path": link} and "os error 448" in err["details"]
    only_the_code = ["error: something (os error 448)"]  # no path in the message: the folder to look at is uv's Python folder
    monkeypatch.setattr(installer, "run_stream", lambda cmd, ctx, **kw: (2, only_the_code))
    done, _ = run_job(lambda ctx: installer.ensure_python("3.12", ctx))
    assert done["error"]["code"] == "python_junction_blocked" and done["error"]["params"] == {"path": str(python_root())}

    monkeypatch.setattr(installer, "run_stream", lambda cmd, ctx, **kw: (2, ["error: Failed to download", "dns error: no such host"]))
    done, _ = run_job(lambda ctx: installer.ensure_python("3.12", ctx))
    err = done["error"]
    assert err["code"] == "download_unreachable" and err["action"] == "show_manual"
    assert err["params"]["filename"] == "Python 3.12" and err["params"]["target_dir"] == str(python_root())

    monkeypatch.setattr(installer, "run_stream", lambda cmd, ctx, **kw: (1, ["error: disk exploded"]))
    done, _ = run_job(lambda ctx: installer.ensure_python("3.12", ctx))
    err = done["error"]
    assert err["code"] == "tool_install_failed" and err["params"] == {"tool": "python"} and "disk exploded" in err["details"]

    monkeypatch.setattr(installer, "find_uv", lambda: None)
    done, _ = run_job(lambda ctx: installer.ensure_python("3.12", ctx))
    assert done["error"]["code"] == "tool_missing" and done["error"]["params"]["tool"] == "uv"


def test_pytools_venv_is_made_from_the_concrete_interpreter(p_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = Path(r"C:\apps\envs\_uv\python\cpython-3.12.4-windows-x86_64-none\python.exe")
    seen: list[list[str]] = []

    def fake_stream(cmd: list[str], ctx: Any, **kw: Any) -> tuple[int, list[str]]:
        seen.append(cmd)
        venv = Path(cmd[2])
        py = installer._venv_python(venv)
        py.parent.mkdir(parents=True, exist_ok=True)
        py.write_bytes(b"MZ")
        (venv / "pyvenv.cfg").write_text("home = x\n", encoding="utf-8")
        return 0, []

    monkeypatch.setattr(installer, "find_uv", lambda: Path("uv"))
    monkeypatch.setattr(installer, "ensure_python", lambda version, ctx: base if version == installer.PYTOOLS_PYTHON else pytest.fail(version))
    monkeypatch.setattr(installer, "run_stream", fake_stream)
    monkeypatch.setattr(installer, "ensure_utf8_output", lambda py: None)
    done, out = run_job(lambda ctx: installer.ensure_pytools(ctx))
    assert done["state"] == "done", done.get("error")
    assert seen == [["uv", "venv", str(installer.pytools_dir()), "--python", str(base)]]


def test_a_broken_tools_environment_is_rebuilt_after_its_processes_are_terminated(p_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = p_env.parent / "py" / EXE_NAME
    venv = installer.pytools_dir()
    stale = installer._venv_python(venv)
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"MZ")  # a venv without pyvenv.cfg: not usable, rebuilt
    events: list[str] = []

    def kill_all(roots: Any, **kw: Any) -> list[winproc.ProcInfo]:
        events.append(f"kill {[str(r) for r in roots]} {kw['reason']} exists={venv.exists()}")
        return [winproc.ProcInfo(4321, 1, str(stale), "python.exe", None, "tool", False, True, 9)]

    def fake_stream(cmd: list[str], ctx: Any, **kw: Any) -> tuple[int, list[str]]:
        events.append("uv venv")
        py = installer._venv_python(Path(cmd[2]))
        py.parent.mkdir(parents=True, exist_ok=True)
        py.write_bytes(b"MZ")
        (Path(cmd[2]) / "pyvenv.cfg").write_text("home = x\n", encoding="utf-8")
        return 0, []

    monkeypatch.setattr(installer, "find_uv", lambda: Path("uv"))
    monkeypatch.setattr(installer, "ensure_python", lambda version, ctx: base)
    monkeypatch.setattr(installer, "run_stream", fake_stream)
    monkeypatch.setattr(installer, "ensure_utf8_output", lambda py: None)
    monkeypatch.setattr(winproc, "kill_all_under", kill_all)
    done, _ = run_job(lambda ctx: installer.ensure_pytools(ctx))
    assert done["state"] == "done", done.get("error")
    assert events == [f"kill {[str(venv)]} the tools environment is being rebuilt exists=True", "uv venv"]  # killed while the folder was still there
    assert f"terminated pid 4321 ({stale}): the tools environment is being rebuilt" in "\n".join(done["log_tail"])
    events.clear()
    done, _ = run_job(lambda ctx: installer.ensure_pytools(ctx))  # a working venv is left alone
    assert done["state"] == "done" and events == []


def test_process_roots_are_the_envs_and_the_tools_folder_and_in_the_packaged_app_the_bundled_helpers(p_env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert installer.process_roots() == [installer.envs_dir(), appdirs.tools_dir()]  # the envs folder first: it is the ledger's "env" root
    assert Path(installer.uv_dirs()["UV_PYTHON_INSTALL_DIR"]).is_relative_to(installer.process_roots()[0])  # uv's Pythons (the real interpreters behind the launchers) are covered
    monkeypatch.setattr(installer, "app_dir", lambda: tmp_path / "app")
    assert installer.process_roots() == [installer.envs_dir(), appdirs.tools_dir(), tmp_path / "app" / "bin", tmp_path / "app" / "git"]


def test_make_envs_command_gets_the_interpreter(p_env: Path) -> None:
    exe = Path("E:/envs/_uv/python/cpython-3.11.9-windows-x86_64-none/python.exe")
    cmd, extra = installer.build_make_envs_command(["main", "funasr"], "cpu", Path("E:/envs"), exe)
    plain, plain_extra = installer.build_make_envs_command(["main"], "cuda", Path("E:/envs"))
    if os.name == "nt":
        assert cmd[cmd.index("-Python") + 1] == str(exe) and cmd[cmd.index("-Env") + 1] == "main,funasr" and cmd[cmd.index("-EnvsDir") + 1] == str(Path("E:/envs")) and "-Cpu" in cmd
        assert "-Python" not in plain and plain_extra == {}
    else:
        assert extra == {"ENVS_DIR": str(Path("E:/envs")), "CPU": "1", "PYTHON": str(exe)} and cmd[-2:] == ["main", "funasr"]
        assert "PYTHON" not in plain_extra


def test_install_envs_hands_the_interpreter_to_make_envs(p_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    name = next(n for n in installer.env_names() if not installer.env_needs_git(n))
    base = p_env.parent / "py" / EXE_NAME
    seen: dict[str, Any] = {}

    def fake_build(envs: list[str], torch: str, envs_path: Path, python: Path | None = None) -> tuple[list[str], dict[str, str]]:
        seen["build"] = (envs, torch, envs_path, python)
        return ["fake-make-envs"], {}

    def fake_stream(cmd: list[str], ctx: Any, **kw: Any) -> tuple[int, list[str]]:
        seen["env"] = kw["env"]
        for rel in ("Scripts/python.exe", "bin/python"):
            f = installer.envs_dir() / name / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(b"")
        (installer.envs_dir() / name / "pyvenv.cfg").write_text("home = x", encoding="utf-8")
        return 0, [f"== env {name}"]

    monkeypatch.setattr(installer, "find_uv", lambda: Path("uv"))
    monkeypatch.setattr(installer, "ensure_python", lambda version, ctx: base if version == installer.ENVS_PYTHON else pytest.fail(version))
    monkeypatch.setattr(installer, "build_make_envs_command", fake_build)
    monkeypatch.setattr(installer, "run_stream", fake_stream)
    done, _ = run_job(lambda ctx: installer.install_envs([name], "cpu", ctx))
    assert done["state"] == "done", done.get("error")
    assert installer.ENVS_PYTHON == "3.11"
    assert seen["build"] == ([name], "cpu", p_env.parent / "envs", base)
    assert seen["env"]["UV_CACHE_DIR"] == installer.uv_dirs()["UV_CACHE_DIR"] and seen["env"]["UV_PYTHON_INSTALL_DIR"] == installer.uv_dirs()["UV_PYTHON_INSTALL_DIR"]


def test_a_blocked_junction_during_make_envs_is_reported_as_such(p_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    name = next(n for n in installer.env_names() if not installer.env_needs_git(n))
    monkeypatch.setattr(installer, "find_uv", lambda: Path("uv"))
    monkeypatch.setattr(installer, "ensure_python", lambda version, ctx: Path("py") / EXE_NAME)
    monkeypatch.setattr(installer, "run_stream", lambda cmd, ctx, **kw: (1, [r"error: Failed to inspect Python interpreter at C:\u\uv\python\cpython-3.11-windows-x86_64-none\python.exe (os error 448)"]))
    done, _ = run_job(lambda ctx: installer.install_envs([name], "cpu", ctx))
    assert done["error"]["code"] == "python_junction_blocked" and done["error"]["params"]["path"] == r"C:\u\uv\python\cpython-3.11-windows-x86_64-none"


# ---- Cancel kills the child process tree --------------------------------------------------------------------------------

SLEEPER = "import os, sys, time, pathlib; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); print('started', flush=True); time.sleep(60)"


def cancel_and_check(tmp_path: Path, body: Callable[[Any, list[str]], Any]) -> None:
    """Start ``body(ctx, argv)`` (it runs the sleeper) as a gpu=False job, cancel it, and check the speed and the aftermath."""
    pidfile = tmp_path / "child.pid"
    r = jobs.runner()
    job = r.submit("envs_install", lambda ctx: body(ctx, [sys.executable, "-c", SLEEPER, str(pidfile)]), gpu=False)
    t0 = time.monotonic()
    while not (pidfile.exists() and pidfile.read_text().strip()) and time.monotonic() - t0 < 30:
        time.sleep(0.05)
    assert pidfile.exists(), "the child never started"
    pid = int(pidfile.read_text())
    assert pid_alive(pid)
    t_cancel = time.monotonic()
    r.cancel(job.id)
    done = wait_job(job.id, timeout=15)
    took = time.monotonic() - t_cancel
    assert done["state"] == "cancelled" and done["error"] is None, done.get("error")  # not env_install_failed with the killed child's exit code
    assert took < 5.0, f"the cancel took {took:.1f}s"
    t1 = time.monotonic()
    while pid_alive(pid) and time.monotonic() - t1 < 5:
        time.sleep(0.05)
    assert not pid_alive(pid), "the child process is still running"


def test_run_stream_cancel_kills_the_child_at_once(p_env: Path, tmp_path: Path) -> None:
    cancel_and_check(tmp_path, lambda ctx, argv: installer.run_stream(argv, ctx))


def test_run_waiting_cancel_kills_the_child_at_once(p_env: Path, tmp_path: Path) -> None:
    cancel_and_check(tmp_path, lambda ctx, argv: installer._run_waiting(argv, ctx))


def test_run_stream_returns_the_exit_code_of_a_child_that_ends_by_itself(p_env: Path) -> None:
    done, out = run_job(lambda ctx: installer.run_stream([sys.executable, "-c", "import sys; print('hello'); sys.exit(3)"], ctx))
    assert done["state"] == "done"
    assert out["value"] == (3, ["hello"])


# ---- Settings: folders ---------------------------------------------------------------------------------------------------


def test_get_settings_has_the_effective_paths(s_env: Path) -> None:
    c = client_for(general)
    s = c.get("/api/settings").json()
    assert s["envs_dir"] == "" and s["models_dir"] == ""
    p = s["paths"]
    assert set(p) == PATH_KEYS
    envs = s_env / "envs"
    assert p == {
        "data": str(s_env), "models": str(s_env / "models"), "envs": str(envs), "tools": str(s_env / "tools"),
        "python": str(envs / "_uv" / "python"), "uv_cache": str(envs / "_uv" / "cache"),
    }
    assert all(Path(v).is_absolute() for v in p.values())
    assert Path(p["models"]).is_dir() and Path(p["envs"]).is_dir() and Path(p["tools"]).is_dir()  # "Open" works on a fresh install
    assert "VNDUB_ENVS_DIR" not in os.environ  # reading the settings decides nothing


def test_put_envs_dir_moves_the_environments_and_uv(s_env: Path, tmp_path: Path) -> None:
    c = client_for(general)
    new = tmp_path / "big drive" / "VNDub envs"
    r = c.put("/api/settings", json={"envs_dir": str(new)})
    assert r.status_code == 200, r.text
    s = r.json()
    assert s["envs_dir"] == str(new)
    assert s["paths"]["envs"] == str(new) and s["paths"]["python"] == str(new / "_uv" / "python") and s["paths"]["uv_cache"] == str(new / "_uv" / "cache")
    assert os.environ["VNDUB_ENVS_DIR"] == str(new) and installer.envs_dir() == new and new.is_dir()
    assert json.loads((s_env / "settings.json").read_text(encoding="utf-8"))["envs_dir"] == str(new)
    assert installer._child_env()["UV_CACHE_DIR"] == str(new / "_uv" / "cache")
    from vndub.backends.runner import envs_root

    assert envs_root() == new  # the pipeline in this process follows
    assert c.get("/api/settings").json()["paths"]["envs"] == str(new)
    # back to the default
    s = c.put("/api/settings", json={"envs_dir": ""}).json()
    assert s["envs_dir"] == "" and s["paths"]["envs"] == str(s_env / "envs") and os.environ["VNDUB_ENVS_DIR"] == str(s_env / "envs")
    assert s["paths"]["python"] == str(s_env / "envs" / "_uv" / "python")
    assert new.is_dir()  # nothing is deleted or moved


def test_put_envs_dir_validation(s_env: Path, tmp_path: Path) -> None:
    c = client_for(general)
    for bad in ({"envs_dir": "relative/folder"}, {"envs_dir": 5}, {"envs_dir": ["x"]}):
        r = c.put("/api/settings", json=bad)
        assert r.status_code == 422 and r.json()["error"]["code"] == "bad_request" and r.json()["error"]["params"]["field"] == "envs_dir"
    blocker = tmp_path / "a file"
    blocker.write_text("x", encoding="utf-8")
    r = c.put("/api/settings", json={"envs_dir": str(blocker / "envs")})  # cannot be created below a file
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "permission_denied" and err["action"] == "open_data_dir" and err["params"]["path"] == str(blocker / "envs")
    assert c.get("/api/settings").json()["envs_dir"] == ""  # nothing was saved
    assert c.put("/api/settings", json={"envs_dir": None}).json()["envs_dir"] == ""  # null = default, like ""


def test_put_envs_dir_is_refused_while_a_job_runs(s_env: Path, tmp_path: Path) -> None:
    c = client_for(general)
    gate = threading.Event()
    job = jobs.runner().submit("report", lambda ctx: gate.wait(20) and None, gpu=False)
    try:
        r = c.put("/api/settings", json={"envs_dir": str(tmp_path / "later")})
        assert r.status_code == 409
        err = r.json()["error"]
        assert err["code"] == "job_busy" and err["params"]["kind"] == "report" and err["params"]["job_id"] == job.id
        assert c.get("/api/settings").json()["envs_dir"] == "" and not (tmp_path / "later").exists()
        assert c.put("/api/settings", json={"ui_lang": "en"}).status_code == 200  # other settings stay changeable
    finally:
        gate.set()
        wait_job(job.id)
    assert c.put("/api/settings", json={"envs_dir": str(tmp_path / "later")}).status_code == 200


def test_an_outside_vndub_envs_dir_wins_over_the_setting(s_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c = client_for(general)
    outside = tmp_path / "from the shell"
    monkeypatch.setenv("VNDUB_ENVS_DIR", str(outside))
    chosen = tmp_path / "chosen in settings"
    s = c.put("/api/settings", json={"envs_dir": str(chosen)}).json()
    assert s["envs_dir"] == str(chosen)  # saved ...
    assert s["paths"]["envs"] == str(outside) and os.environ["VNDUB_ENVS_DIR"] == str(outside)  # ... but not what is used
    assert s["paths"]["python"] == str(outside / "_uv" / "python")
    assert installer.configure_environment(force=True) == outside
    assert installer.envs_dir() == outside


def test_configure_environment_decides_once_and_again_when_forced(s_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = installer.configure_environment()
    assert first == s_env / "envs" and os.environ["VNDUB_ENVS_DIR"] == str(first)
    s = appdirs.load_settings()
    s.envs_dir = str(tmp_path / "second")
    appdirs.save_settings(s)
    assert installer.configure_environment() == first  # idempotent: a running installation keeps its folder
    assert installer.configure_environment(force=True) == tmp_path / "second"
    assert os.environ["VNDUB_ENVS_DIR"] == str(tmp_path / "second")
    s.envs_dir = ""
    appdirs.save_settings(s)
    assert installer.configure_environment(force=True) == first
    # a dev checkout that already has environments keeps them, unless a folder was chosen in Settings
    legacy = tmp_path / "dev-venvs"
    (legacy / "main").mkdir(parents=True)
    (legacy / "main" / "pyvenv.cfg").write_text("home = x", encoding="utf-8")
    monkeypatch.setattr(installer, "repo_envs_dir", lambda: legacy)
    assert installer.configure_environment(force=True) == legacy
    s.envs_dir = str(tmp_path / "third")
    appdirs.save_settings(s)
    assert installer.configure_environment(force=True) == tmp_path / "third"


def test_effective_envs_dir_and_the_memory_of_outside_variables(p_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outside = p_env.parent / "envs"  # set by make_env, i.e. from outside
    installer.reset_environment_memory()
    assert appdirs.external_envs_dir() == outside
    assert appdirs.effective_envs_dir() == outside
    s = appdirs.Settings(envs_dir=str(tmp_path / "chosen"))
    assert appdirs.effective_envs_dir(s) == tmp_path / "chosen"
    monkeypatch.delenv("VNDUB_ENVS_DIR")
    assert appdirs.external_envs_dir() is None and appdirs.effective_envs_dir() == p_env / "envs"
    monkeypatch.setattr(installer, "repo_envs_dir", lambda: tmp_path / "none")
    exported = installer.configure_environment()  # exported by the app itself: no longer "outside"
    assert exported == p_env / "envs" and appdirs.external_envs_dir() is None
    installer.reset_environment_memory()  # forget: now the variable looks like it came from outside
    assert appdirs.external_envs_dir() == exported


def test_models_dir_setting_and_its_checks(s_env: Path, tmp_path: Path) -> None:
    c = client_for(general)
    new = tmp_path / "models on D"
    s = c.put("/api/settings", json={"models_dir": str(new)}).json()
    assert s["models_dir"] == str(new) and s["paths"]["models"] == str(new) and new.is_dir()
    assert appdirs.effective_models_dir() == new
    assert c.put("/api/settings", json={"models_dir": "relative"}).status_code == 422
    s = c.put("/api/settings", json={"models_dir": ""}).json()
    assert s["models_dir"] == "" and s["paths"]["models"] == str(s_env / "models")


def test_data_dir_switch_takes_the_default_envs_along(s_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c = client_for(general)
    assert c.put("/api/settings", json={"envs_dir": ""}).status_code == 200  # the app has decided and exported its default
    assert os.environ["VNDUB_ENVS_DIR"] == str(s_env / "envs")
    monkeypatch.delenv("VNDUB_DATA")  # the real switch is persisted in location.json (redirected into tmp_path)
    new = tmp_path / "elsewhere" / "VNDub"
    r = c.post("/api/settings/data-dir", json={"path": str(new)})
    try:
        assert r.status_code == 200, r.text
        assert r.json()["paths"]["envs"] == str(new.resolve() / "envs") and os.environ["VNDUB_ENVS_DIR"] == str(new.resolve() / "envs")
    finally:
        from vndub_app import server

        jobs.reset_runner(None)
        server.close_logging()

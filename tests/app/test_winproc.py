"""Processes the app must not leave behind: the ledger by image path, the reaper, the Job Object helpers, window attention, the HTTP route.

The ledger tests run real processes. ``python -m venv --without-pip <tmp>/envs/main`` makes an environment whose
``Scripts/python.exe`` is a launcher with an image path under the temp folder (like the launcher in front of every backend
adapter); the temp folder is the ledger's root, so nothing outside it can ever be listed or killed here.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from o_helpers import client_for, drop_runner, fresh_runner, make_env

from vndub_app import appdirs, session, winproc
from vndub_app.routes import processes as processes_route

REPO = Path(__file__).resolve().parents[2]
windows_only = pytest.mark.skipif(os.name != "nt", reason="the process ledger is Windows only")
SLEEP = "import time; time.sleep(120)"


def wait_for(pred: Callable[[], Any], timeout: float = 20.0, step: float = 0.1) -> Any:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        got = pred()
        if got:
            return got
        time.sleep(step)
    return pred()


def make_venv(folder: Path) -> Path:
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(folder)], check=True, capture_output=True, timeout=120)
    return folder / "Scripts" / "python.exe"


def find(roots: list[Path], pid: int) -> winproc.ProcInfo | None:
    return next((i for i in winproc.list_processes(roots) if i.pid == pid), None)


def gone(pid: int) -> bool:
    return not winproc.process_alive(pid)


def spawn_orphan(py: Path, *, pause: float = 0.0) -> int:
    """A sleeping process of ``py`` whose parent has already ended: started by an intermediate python that prints the pid and exits.

    The sleeper gets no pipes of the intermediate (it would hold them open and ``subprocess.run`` would wait for it).
    """
    code = (
        "import subprocess, sys, time\n"
        "null = subprocess.DEVNULL\n"
        f"p = subprocess.Popen([r'{py}', '-c', '{SLEEP}'], stdin=null, stdout=null, stderr=null)\n"
        f"print(p.pid, flush=True); time.sleep({pause})\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return int(out.stdout.split()[-1])


@pytest.fixture(scope="module")
def tree(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Path]]:
    """``envs/main`` and ``envs/other`` (two environments), ``tools``, and ``app`` (an environment whose launcher is also copied to ``VNDub.exe``)."""
    root = tmp_path_factory.mktemp("ledger")
    envs, tools = root / "envs", root / "tools"
    tools.mkdir()
    main_py, other_py = make_venv(envs / "main"), make_venv(envs / "other")
    app_py = make_venv(root / "app")
    vndub = app_py.with_name("VNDub.exe")
    shutil.copy(app_py, vndub)
    yield {"root": root, "envs": envs, "tools": tools, "main": main_py, "other": other_py, "vndub": vndub}
    winproc.kill_all_under([envs, tools], with_descendants=True, reason="test cleanup")  # whatever a failed test left running


@pytest.fixture
def procs() -> Iterator[list[subprocess.Popen[bytes]]]:
    started: list[subprocess.Popen[bytes]] = []
    yield started
    for p in started:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
        try:
            p.wait(10)
        except subprocess.TimeoutExpired:
            pass


def start(procs: list[subprocess.Popen[bytes]], py: Path) -> subprocess.Popen[bytes]:
    p = subprocess.Popen([str(py), "-c", SLEEP])
    procs.append(p)
    return p


# ---- the ledger: what it lists ---------------------------------------------------------------------------------------------


@windows_only
def test_the_ledger_describes_a_child_of_this_process(tree: dict[str, Path], procs: list[subprocess.Popen[bytes]]) -> None:
    p = start(procs, tree["main"])
    info = wait_for(lambda: find([tree["envs"], tree["tools"]], p.pid))
    assert info is not None
    assert info.pid == p.pid and info.ppid == os.getpid() and info.name == "python.exe"
    assert Path(info.exe) == tree["main"] and info.env == "main" and info.role == "adapter"
    assert info.ours is True and info.orphan is False and info.started > 0
    assert info.public() == {"pid": p.pid, "ppid": os.getpid(), "exe": info.exe, "name": "python.exe", "env": "main", "role": "adapter", "ours": True, "orphan": False}
    assert find([tree["envs"], tree["tools"]], os.getpid()) is None  # never this process, whatever its image


@windows_only
def test_roles_follow_the_folder_and_the_name(tree: dict[str, Path], procs: list[subprocess.Popen[bytes]]) -> None:
    """A python under the envs root is an adapter of that env; the same launcher found under a tools root is a tool."""
    p = start(procs, tree["other"])
    wait_for(lambda: find([tree["envs"]], p.pid))
    assert find([tree["envs"]], p.pid).env == "other"  # type: ignore[union-attr]
    as_tool = find([tree["tools"], tree["envs"]], p.pid)  # the roots' order decides which one is "the envs folder"
    assert as_tool is not None and as_tool.role == "tool" and as_tool.env is None


@windows_only
def test_only_processes_under_the_roots_are_listed(tree: dict[str, Path], procs: list[subprocess.Popen[bytes]]) -> None:
    p = start(procs, tree["main"])
    wait_for(lambda: find([tree["envs"]], p.pid))
    assert find([tree["envs"] / "other"], p.pid) is None  # a sibling folder
    assert find([tree["root"] / "env"], p.pid) is None  # a prefix of the name is not a parent folder
    assert winproc.list_processes([tree["root"] / "no" / "such" / "folder"]) == []
    assert winproc.list_processes([]) == []
    assert winproc.kill_orphans([]) == [] and winproc.kill_all_under([]) == []


@windows_only
def test_the_scan_is_fast(tree: dict[str, Path], procs: list[subprocess.Popen[bytes]]) -> None:
    start(procs, tree["main"])
    t0 = time.perf_counter()
    for _ in range(5):
        winproc.list_processes([tree["envs"], tree["tools"]])
    assert (time.perf_counter() - t0) / 5 < 0.5  # one snapshot, every candidate opened once: ~30 ms here; startup waits for it


@windows_only
def test_the_ledger_never_raises(tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def broken(*a: Any, **kw: Any) -> Any:
        raise OSError("snapshot refused")

    monkeypatch.setattr(winproc, "_Scan", broken)
    assert winproc.list_processes([tree["envs"]]) == []
    assert winproc.kill_orphans([tree["envs"]]) == []
    assert winproc.kill_all_under([tree["envs"]], with_descendants=True) == []
    assert "snapshot refused" in caplog.text
    assert winproc.kill_processes([]) == []


# ---- orphans ---------------------------------------------------------------------------------------------------------------------


@windows_only
def test_a_process_whose_parent_has_ended_is_an_orphan_and_kill_orphans_ends_it(tree: dict[str, Path], procs: list[subprocess.Popen[bytes]], caplog: pytest.LogCaptureFixture) -> None:
    orphan = spawn_orphan(tree["main"])
    mine = start(procs, tree["main"])
    roots = [tree["envs"], tree["tools"]]
    info = wait_for(lambda: find(roots, orphan))
    assert info is not None and info.orphan is True and info.ours is False and info.env == "main" and info.role == "adapter"
    assert wait_for(lambda: find(roots, mine.pid)).orphan is False  # a child of this process is neither
    caplog.set_level("INFO", logger="vndub_app.winproc")
    killed = winproc.kill_orphans(roots, reason="a test")
    assert [i.pid for i in killed] == [orphan]
    assert wait_for(lambda: gone(orphan), 15)
    assert mine.poll() is None  # the live adapter of the running job is not touched
    line = next(r.getMessage() for r in caplog.records if r.levelno == logging.INFO and f"pid {orphan}" in r.getMessage())
    assert str(tree["main"]) in line and "env=main" in line and "a test" in line  # pid, exe, env and the reason are in the log
    assert winproc.kill_orphans(roots) == []  # nothing left


@windows_only
def test_what_a_dead_intermediate_left_behind_is_found_through_the_chain(tree: dict[str, Path], procs: list[subprocess.Popen[bytes]]) -> None:
    """A launcher whose real interpreter's *parent launcher* is dead: the chain of parents breaks, the image path still finds it.

    ``taskkill /T`` walks the tree from the top, so a grandchild whose parent died first is out of its reach; this is that case.
    """
    code = (
        f"import subprocess, sys, time; g = subprocess.Popen([r'{tree['main']}', '-c', '{SLEEP}']); print(g.pid, flush=True); time.sleep(120)"
    )
    parent = subprocess.Popen([str(tree["other"]), "-c", code], stdout=subprocess.PIPE, text=True)
    procs.append(parent)
    assert parent.stdout is not None
    grand = int(parent.stdout.readline())
    roots = [tree["envs"], tree["tools"]]
    assert wait_for(lambda: find(roots, grand)).orphan is False  # its parents are alive, this process is at the top  # type: ignore[union-attr]
    assert wait_for(lambda: find(roots, grand)).ours is True  # type: ignore[union-attr]
    # kill the middle of the chain only: the real interpreter that sits between the two launchers
    mid_pid = find(roots, grand).ppid  # type: ignore[union-attr]
    subprocess.run(["taskkill", "/F", "/PID", str(mid_pid)], capture_output=True)
    assert wait_for(lambda: gone(mid_pid), 10)
    after = wait_for(lambda: (i := find(roots, grand)) is not None and i.orphan and i)
    assert after and after.orphan is True and after.ours is False  # the dead middle broke the chain: no longer "ours", an orphan
    assert [i.pid for i in winproc.kill_orphans(roots)] == [grand]
    assert wait_for(lambda: gone(grand), 15)


@windows_only
def test_a_process_owned_by_another_running_vndub_is_not_an_orphan(tree: dict[str, Path], procs: list[subprocess.Popen[bytes]]) -> None:
    """Two VNDub instances can share the folders: what a running ``VNDub.exe`` started is neither ours nor an orphan."""
    inner = (  # runs inside the other app: starts the env process, tells its pid, stays
        "import subprocess, sys, time\n"
        f"c = subprocess.Popen([r'{tree['main']}', '-c', '{SLEEP}'], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        f"open(r'{tree['root'] / 'child.pid'}', 'w').write(str(c.pid))\n"
        "time.sleep(120)\n"
    )
    launcher = (  # starts the other app and ends at once: the other app is no child of this process, like a second VNDub.exe
        "import subprocess, sys\n"
        "n = subprocess.DEVNULL\n"
        f"subprocess.Popen([r'{tree['vndub']}', '-c', {inner!r}], stdin=n, stdout=n, stderr=n)\n"
    )
    pid_file = tree["root"] / "child.pid"
    pid_file.unlink(missing_ok=True)
    roots = [tree["envs"], tree["tools"]]
    try:
        subprocess.run([sys.executable, "-c", launcher], check=True, timeout=60)
        child = int(wait_for(pid_file.exists, 30) and wait_for(lambda: pid_file.read_text() or None, 10))
        info = wait_for(lambda: find(roots, child))
        assert info is not None and info.ours is False and info.orphan is False  # its chain passes a running VNDub.exe, which nobody started here
        assert winproc.kill_orphans(roots) == [] and winproc.process_alive(child)
        killed = winproc.kill_all_under(roots, reason="the user pressed kill all")  # "all" does not ask who owns it
        assert child in [i.pid for i in killed]
    finally:
        winproc.kill_all_under([tree["vndub"].parent.parent], with_descendants=True, reason="test cleanup")  # the fake VNDub.exe and what is below it


@windows_only
def test_a_reused_parent_pid_does_not_hide_an_orphan(tree: dict[str, Path], procs: list[subprocess.Popen[bytes]]) -> None:
    """Windows hands out pids again: an orphan of an earlier run whose parent pid is now this process' must not count as ours."""
    p = start(procs, tree["main"])
    info = wait_for(lambda: find([tree["envs"]], p.pid))
    assert info is not None and info.ours is True
    scan = winproc._Scan([tree["envs"]])
    assert scan.judge(p.pid, info.started) == (True, False)
    # the same pid claims to have been created *before* this process: its recorded parent (this pid) cannot be its real parent
    assert scan.judge(p.pid, scan.me_meta.created - 10_000_000) == (False, True)  # type: ignore[union-attr]


# ---- killing -----------------------------------------------------------------------------------------------------------------------


@windows_only
def test_kill_processes_leaves_first_and_checks_the_identity(tree: dict[str, Path], procs: list[subprocess.Popen[bytes]]) -> None:
    p = start(procs, tree["main"])
    q = start(procs, tree["other"])
    roots = [tree["envs"], tree["tools"]]
    a, b = wait_for(lambda: find(roots, p.pid)), wait_for(lambda: find(roots, q.pid))
    assert a and b
    me = winproc.ProcInfo(os.getpid(), 0, sys.executable, "python.exe", None, "other", False, True, 1)
    stale = winproc.ProcInfo(q.pid, b.ppid, b.exe, b.name, b.env, b.role, b.ours, b.orphan, b.started + 12345)  # "the pid was reused since the listing"
    liar = winproc.ProcInfo(q.pid, b.ppid, str(tree["main"]) + "x", b.name, b.env, b.role, b.ours, b.orphan, b.started)
    assert winproc.kill_processes([me, stale, liar]) == []
    assert q.poll() is None
    assert sorted(winproc.kill_processes([a, b], reason="test")) == sorted([p.pid, q.pid])
    assert wait_for(lambda: p.poll() is not None and q.poll() is not None, 15)
    assert winproc.kill_processes([a, b]) == []  # already gone: nothing to do, nothing raised


@windows_only
def test_kill_all_under_with_descendants_takes_the_real_interpreter_behind_the_launcher(tree: dict[str, Path], procs: list[subprocess.Popen[bytes]]) -> None:
    """The launcher of an environment runs the real interpreter, which lives elsewhere (uv's Pythons) and holds the environment's DLLs."""
    p = start(procs, tree["main"])
    wait_for(lambda: find([tree["envs"]], p.pid))
    real = wait_for(lambda: [pid for pid, (pp, _n) in winproc._api().snapshot().items() if pp == p.pid])
    assert real, "the launcher started no interpreter"
    assert find([tree["envs"]], real[0]) is None  # outside the roots: a plain listing does not see it
    killed = winproc.kill_all_under([tree["envs"] / "main"], with_descendants=True, reason="the environment is being reinstalled")
    assert p.pid in [i.pid for i in killed] and real[0] in [i.pid for i in killed]
    assert wait_for(lambda: p.poll() is not None and gone(real[0]), 15)
    assert winproc.kill_all_under([tree["envs"] / "main"], with_descendants=True) == []


# ---- the reaper --------------------------------------------------------------------------------------------------------------------


@windows_only
def test_the_reaper_waits_for_its_parent_then_terminates_the_orphans_and_writes_one_line(tree: dict[str, Path], tmp_path: Path, procs: list[subprocess.Popen[bytes]]) -> None:
    orphan = spawn_orphan(tree["main"])
    assert wait_for(lambda: find([tree["envs"]], orphan))
    live_child = start(procs, tree["other"])  # ours: its parent (this pytest process) outlives the reaper
    parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3)"])
    procs.append(parent)
    log = tmp_path / "logs" / "reaper.log"
    cmd = winproc.reaper_command(parent.pid, [tree["envs"], tree["tools"]], log)
    assert "--reaper" in cmd and cmd[cmd.index("--parent") + 1] == str(parent.pid) and cmd[cmd.index("--roots") + 1] == f"{tree['envs']};{tree['tools']}"
    t0 = time.monotonic()
    reaper = subprocess.Popen(cmd, cwd=str(REPO), creationflags=winproc.CREATE_NO_WINDOW)
    time.sleep(1.0)
    assert reaper.poll() is None and winproc.process_alive(orphan)  # it waits as long as the parent lives
    assert not log.exists()
    parent.wait(30)
    assert reaper.wait(60) == 0
    assert time.monotonic() - t0 < 30
    assert wait_for(lambda: gone(orphan), 5), "the orphan survived the reaper"
    assert live_child.poll() is None  # not an orphan: the reaper left it alone
    line = log.read_text(encoding="utf-8").splitlines()[0]
    assert line[:4].isdigit() and line[4] == "-" and f"parent {parent.pid} ended; terminated 1: pid {orphan} {tree['main']}" in line
    assert len(log.read_text(encoding="utf-8").splitlines()) == 1  # one line, no noise


@windows_only
def test_the_reaper_exits_on_its_own_when_the_parent_cannot_be_opened(tmp_path: Path) -> None:
    log = tmp_path / "reaper.log"
    t0 = time.monotonic()
    assert winproc.reaper_main(["--reaper", "--parent", "4", "--roots", str(tmp_path), "--log", str(log)]) == 0  # pid 4 (System): access denied
    assert time.monotonic() - t0 < 5  # it did not wait for the parent
    assert "cannot open parent 4" in log.read_text(encoding="utf-8")


@windows_only
def test_the_reaper_sweeps_at_once_when_the_parent_is_already_gone(tree: dict[str, Path], tmp_path: Path) -> None:
    orphan = spawn_orphan(tree["main"])
    done = subprocess.Popen([sys.executable, "-c", "pass"])
    done.wait(30)
    log = tmp_path / "reaper.log"
    t0 = time.monotonic()
    assert winproc.reaper_main(["--reaper", "--parent", str(done.pid), "--roots", f"{tree['envs']};{tree['tools']}", "--log", str(log)]) == 0
    assert time.monotonic() - t0 < 15
    assert wait_for(lambda: gone(orphan), 5)
    assert f"parent {done.pid} ended; terminated 1: pid {orphan}" in log.read_text(encoding="utf-8")


@windows_only
def test_the_reaper_does_not_trust_a_reused_parent_pid(tree: dict[str, Path], tmp_path: Path, procs: list[subprocess.Popen[bytes]]) -> None:
    """``--parent-started`` is the parent's creation time: a pid that now belongs to a younger process is not the parent."""
    log = tmp_path / "reaper.log"
    stranger = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    procs.append(stranger)
    t0 = time.monotonic()
    assert winproc.reaper_main(["--reaper", "--parent", str(stranger.pid), "--parent-started", "1", "--roots", str(tree["tools"]), "--log", str(log)]) == 0
    assert time.monotonic() - t0 < 15 and stranger.poll() is None  # it did not wait 60 s for an unrelated process
    assert f"parent {stranger.pid} ended" in log.read_text(encoding="utf-8")


def test_main_hands_the_reaper_over_before_it_parses_or_locks_anything(monkeypatch: pytest.MonkeyPatch) -> None:
    from vndub_app import shell

    seen: list[list[str]] = []
    monkeypatch.setattr(winproc, "reaper_main", lambda argv: seen.append(argv) or 0)
    monkeypatch.setattr(shell, "InstanceLock", lambda *a: pytest.fail("the reaper must not touch the instance lock"))
    monkeypatch.setattr(shell, "start_server_async", lambda *a, **kw: pytest.fail("the reaper must not start the server"))
    assert shell.main(["--reaper", "--parent", "7", "--roots", "x;y", "--log", "z"]) == 0
    assert seen == [["--reaper", "--parent", "7", "--roots", "x;y", "--log", "z"]]


@windows_only
def test_the_module_entry_point_runs_the_reaper_without_importing_the_shell(tmp_path: Path) -> None:
    log = tmp_path / "reaper.log"
    code = (
        "import runpy, sys\n"
        f"sys.argv = ['vndub_app', '--reaper', '--parent', '4', '--roots', r'{tmp_path}', '--log', r'{log}']\n"
        "try:\n"
        "    runpy.run_module('vndub_app', run_name='__main__')\n"
        "except SystemExit as e:\n"
        "    print('exit', e.code, 'shell' , 'vndub_app.shell' in sys.modules, 'server', 'vndub_app.server' in sys.modules, 'fastapi', 'fastapi' in sys.modules)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO), timeout=60)
    assert out.stdout.strip() == "exit 0 shell False server False fastapi False", out.stderr
    assert "cannot open parent 4" in log.read_text(encoding="utf-8")


@windows_only
def test_start_reaper_leaves_the_job_object_and_keeps_it_closed_afterwards(tree: dict[str, Path], tmp_path: Path) -> None:
    """In a fresh process with the Job Object installed: the reaper is started outside the job (if the machine allows the
    breakaway; the result says), the job refuses breakaway again afterwards, and the reaper ends with its parent."""
    log = tmp_path / "reaper.log"
    code = (
        "import ctypes, json, os, sys, time\n"
        "from ctypes import wintypes\n"
        f"sys.path.insert(0, r'{REPO}')\n"
        "from vndub_app import winproc\n"
        "assert winproc.kill_children_on_exit()\n"
        f"res = winproc.start_reaper([r'{tree['envs']}'], r'{log}')\n"
        "k = ctypes.WinDLL('kernel32', use_last_error=True)\n"
        "k.OpenProcess.restype = wintypes.HANDLE\n"
        "k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]\n"
        "k.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]\n"
        "h = k.OpenProcess(0x1000, False, res['pid'])\n"
        "inside = wintypes.BOOL()\n"
        "k.IsProcessInJob(h, ctypes.c_void_p(winproc._JOB_HANDLE), ctypes.byref(inside))\n"
        "from ctypes import byref\n"
        "info = winproc._job_structs()[2]()\n"
        "got = wintypes.DWORD()\n"
        "k.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]\n"
        "k.QueryInformationJobObject(ctypes.c_void_p(winproc._JOB_HANDLE), 9, byref(info), ctypes.sizeof(info), byref(got))\n"
        "print(json.dumps([res, bool(inside.value), hex(info.BasicLimitInformation.LimitFlags)]), flush=True)\n"
        "time.sleep(1.0)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=90)
    assert out.returncode == 0, out.stderr
    res, in_job, flags = json.loads(out.stdout.strip().splitlines()[-1])
    assert res["pid"] and (res["error"] is None) == res["breakaway"]
    assert in_job is (not res["breakaway"])  # breakaway worked <=> the reaper is not a member of the app's job
    assert int(flags, 16) == winproc.JOB_LIMIT_KILL_ON_JOB_CLOSE  # breakaway is switched off again: children cannot escape
    assert wait_for(lambda: gone(res["pid"]), 30)  # the parent ended, so did the reaper
    if res["breakaway"]:
        assert wait_for(log.exists, 10) and "ended; terminated 0" in log.read_text(encoding="utf-8")  # it was outside the job: it lived to write its line


# ---- the end of this process and of its children ---------------------------------------------------------------------------------


@windows_only
def test_end_process_ends_the_process_with_its_code_even_with_a_thread_that_never_ends() -> None:
    code = (
        f"import sys, threading, time; sys.path.insert(0, r'{REPO}')\n"
        "from vndub_app import winproc\n"
        "threading.Thread(target=threading.Event().wait, daemon=False).start()\n"
        "winproc.end_process(7)\n"
    )
    t0 = time.monotonic()
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert p.returncode == 7 and time.monotonic() - t0 < 30, p.stderr


@windows_only
def test_kill_descendants_takes_the_tree_leaves_first_and_spares_the_reaper_and_webview2(tmp_path: Path) -> None:
    """In a throw-away process: sleeping children (the 'reaper', one named like WebView2) and a grandchild below another one."""
    wv = tmp_path / "msedgewebview2.exe"
    shutil.copy(sys.executable, wv)  # a launcher that runs the same python under another file name
    code = (
        f"import subprocess, sys, time; sys.path.insert(0, r'{REPO}')\n"
        "from vndub_app import winproc\n"
        f"sleep = [sys.executable, '-c', '{SLEEP}']\n"
        f"webview = subprocess.Popen([r'{wv}', '-c', '{SLEEP}'])\n"
        "reaper = subprocess.Popen(sleep)\n"
        f"child = subprocess.Popen([sys.executable, '-c', \"import subprocess, sys, time; g = subprocess.Popen([sys.executable, '-c', '{SLEEP}']); print(g.pid, flush=True); time.sleep(120)\"], stdout=subprocess.PIPE, text=True)\n"
        "grand = int(child.stdout.readline())\n"
        "time.sleep(1)\n"
        "winproc._REAPER_PID = reaper.pid\n"
        "n = winproc.kill_descendants()\n"
        "time.sleep(1)\n"
        "print('killed', n, 'reaper alive', reaper.poll() is None, 'child dead', child.poll() is not None, 'grand alive', winproc.process_alive(grand), 'webview alive', webview.poll() is None, flush=True)\n"
        "reaper.kill(); webview.kill()\n"
    )
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=90)
    assert p.returncode == 0, p.stderr
    line = p.stdout.strip().splitlines()[-1].split()
    assert line[0] == "killed" and int(line[1]) >= 3  # the child, its interpreter and the grandchild (+ their launchers)
    assert line[line.index("reaper") + 2] == "True" and line[line.index("child") + 2] == "True" and line[line.index("grand") + 2] == "False"
    assert line[line.index("webview") + 2] == "True"  # the window's own processes get to shut down by themselves


# ---- window attention --------------------------------------------------------------------------------------------------------------


def _offscreen_window(title: str, ex_style: int) -> int:
    u = ctypes.WinDLL("user32", use_last_error=True)
    u.CreateWindowExW.restype = ctypes.c_void_p
    u.CreateWindowExW.argtypes = [ctypes.c_ulong, ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 4
    hwnd = u.CreateWindowExW(ex_style, "STATIC", title, 0x80000000 | 0x10000000, -32000, -32000, 8, 8, None, None, None, None)  # WS_POPUP | WS_VISIBLE, far off screen
    assert hwnd, ctypes.get_last_error()
    return int(hwnd)


@windows_only
def test_window_helpers_work_on_a_real_window() -> None:
    u = ctypes.WinDLL("user32", use_last_error=True)
    u.GetForegroundWindow.restype = ctypes.c_void_p
    u.DestroyWindow.argtypes = [ctypes.c_void_p]
    stamp = f"{os.getpid()}-{time.monotonic_ns()}"
    quiet = _offscreen_window(f"VNDub-test-quiet-{stamp}", 0x80 | 0x08000000)  # WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE: never takes the focus
    normal = _offscreen_window(f"VNDub-test-normal-{stamp}", 0x80)  # may take the focus when it is created
    try:
        assert winproc.find_own_window(f"VNDub-test-quiet-{stamp}") == quiet and winproc.find_own_window(f"VNDub-test-normal-{stamp}") == normal
        assert winproc.is_foreground(quiet) is False  # a window nobody activated
        assert winproc.is_foreground(normal) == (int(u.GetForegroundWindow() or 0) == normal)  # the same answer as the OS gives
        assert winproc.flash_window(quiet) is True and winproc.flash_window(normal, count=2) is True
        assert winproc.flash_window(None) is False and winproc.flash_window(0) is False and winproc.is_foreground(None) is False
    finally:
        for h in (quiet, normal):
            u.DestroyWindow(h)
    assert winproc.find_own_window(f"VNDub-test-quiet-{stamp}") not in (quiet, normal)


# ---- before every model job ----------------------------------------------------------------------------------------------------------


@pytest.fixture
def s_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    data = make_env(monkeypatch, tmp_path)
    fresh_runner()
    yield data
    drop_runner()


def test_job_ctx_sweeps_the_orphans_before_it_builds_the_context(s_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vndub.cli

    events: list[str] = []
    monkeypatch.setattr(session, "sweep_orphans", lambda *a, **kw: events.append("sweep"))
    monkeypatch.setattr(appdirs, "write_project_config", lambda h: events.append("config"))
    ctx = type("Ctx", (), {"models": type("Models", (), {"close": staticmethod(lambda: events.append("close"))})()})()
    monkeypatch.setattr(vndub.cli, "make_ctx", lambda *a, **kw: events.append("make_ctx") or ctx)
    h = type("H", (), {"game_dir": s_env, "config_file": s_env / "pipeline.yaml"})()
    session.job_ctx(h)  # type: ignore[arg-type]
    assert events == ["sweep", "config", "make_ctx"]
    with session.short_ctx(h):  # type: ignore[arg-type]
        events.append("body")
    assert events[3:] == ["sweep", "config", "make_ctx", "body", "close", "sweep"]  # the short helpers (casting) sweep too, before and after


def test_sweep_orphans_uses_the_apps_roots_and_never_raises(s_env: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    from vndub_app import installer

    seen: list[Any] = []
    fake = winproc.ProcInfo(1234, 1, r"C:\x\envs\main\Scripts\python.exe", "python.exe", "main", "adapter", False, True)
    monkeypatch.setattr(winproc, "kill_orphans", lambda roots, reason="": seen.append(list(roots)) or [fake])
    caplog.set_level("INFO", logger="vndub_app.session")
    assert session.sweep_orphans() == 1
    assert seen == [installer.process_roots()] and seen[0][0] == Path(os.environ["VNDUB_ENVS_DIR"])  # the envs folder comes first
    assert "pid 1234" in caplog.text and "terminated 1 leftover process" in caplog.text

    def boom(*a: Any, **kw: Any) -> Any:
        raise RuntimeError("no way")

    monkeypatch.setattr(winproc, "kill_orphans", boom)
    assert session.sweep_orphans() == 0 and "orphan sweep failed: no way" in caplog.text


# ---- the HTTP route -----------------------------------------------------------------------------------------------------------------


@pytest.fixture
def api(s_env: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    infos = [
        winproc.ProcInfo(100, 50, r"C:\e\envs\main\Scripts\python.exe", "python.exe", "main", "adapter", True, False, 7),
        winproc.ProcInfo(200, 9, r"C:\e\tools\llama_server\b1\llama-server.exe", "llama-server.exe", None, "llama_server", False, True, 8),
    ]
    calls: list[tuple[str, list[Any], dict[str, Any]]] = []
    monkeypatch.setattr(winproc, "list_processes", lambda roots: infos)
    monkeypatch.setattr(winproc, "kill_orphans", lambda roots, **kw: calls.append(("orphans", list(roots), kw)) or [infos[1]])
    monkeypatch.setattr(winproc, "kill_all_under", lambda roots, **kw: calls.append(("all", list(roots), kw)) or list(infos))
    monkeypatch.setattr(winproc, "job_object_status", lambda: {"installed": False, "error": "AssignProcessToJobObject failed: Access is denied. (error 5)"})
    c = client_for(processes_route)
    c.calls = calls  # type: ignore[attr-defined]
    return c


def test_get_processes_has_the_contract_shape(api: Any) -> None:
    body = api.get("/api/system/processes").json()
    assert set(body) == {"job_object", "processes"}
    assert body["job_object"] == {"installed": False, "error": "AssignProcessToJobObject failed: Access is denied. (error 5)"}
    assert [set(p) for p in body["processes"]] == [{"pid", "ppid", "exe", "name", "env", "role", "ours", "orphan"}] * 2
    assert body["processes"][0] == {"pid": 100, "ppid": 50, "exe": r"C:\e\envs\main\Scripts\python.exe", "name": "python.exe", "env": "main", "role": "adapter", "ours": True, "orphan": False}
    assert body["processes"][1]["role"] == "llama_server" and body["processes"][1]["env"] is None and body["processes"][1]["orphan"] is True


def test_kill_defaults_to_the_orphans_and_all_kills_everything(api: Any) -> None:
    from vndub_app import installer

    assert api.post("/api/system/processes/kill").json() == {"killed": 1, "pids": [200]}  # no body at all
    assert api.post("/api/system/processes/kill", json={}).json() == {"killed": 1, "pids": [200]}
    assert api.post("/api/system/processes/kill", json={"scope": "orphans"}).json() == {"killed": 1, "pids": [200]}
    assert api.post("/api/system/processes/kill", json={"scope": "all"}).json() == {"killed": 2, "pids": [100, 200]}
    assert [c[0] for c in api.calls] == ["orphans", "orphans", "orphans", "all"]
    assert api.calls[0][1] == installer.process_roots()  # the app's own folders, nothing else
    assert api.post("/api/system/processes/kill", json={"scope": "everything"}).status_code == 422


def test_the_routes_are_registered_in_the_app() -> None:
    from vndub_app import routes

    assert "processes" in routes.MODULES and processes_route.router.prefix == "/api/system"


def test_describe_and_summarize_names() -> None:
    a = winproc.ProcInfo(1, 0, r"C:\a\python.exe", "python.exe", "main", "adapter", False, True)
    b = winproc.ProcInfo(2, 0, r"C:\b\python.exe", "python.exe", "hf", "adapter", False, True)
    c = winproc.ProcInfo(3, 0, r"C:\c\llama-server.exe", "llama-server.exe", None, "llama_server", False, True)
    assert winproc.summarize_names([a, b, c]) == "python.exe x2, llama-server.exe" and winproc.summarize_names([]) == ""
    assert winproc.describe([a, c]) == r"pid 1 C:\a\python.exe, pid 3 C:\c\llama-server.exe"
    assert winproc.describe([a] * 5, limit=2).endswith("... (+3)")


def test_the_functions_are_quiet_off_windows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    posix_os = types.ModuleType("os")  # a stand-in only for winproc: the real os.name stays what it is for pathlib, subprocess and pytest
    posix_os.__dict__.update(os.__dict__)
    posix_os.name = "posix"
    monkeypatch.setattr(winproc, "os", posix_os)
    assert winproc.list_processes([tmp_path]) == [] and winproc.kill_orphans([tmp_path]) == [] and winproc.kill_all_under([tmp_path]) == []
    assert winproc.kill_processes([winproc.ProcInfo(1, 0, "x", "x", None, "other", False, True)]) == []
    assert winproc.kill_descendants() == 0 and winproc.start_reaper([tmp_path], tmp_path / "r.log")["pid"] is None
    assert winproc.flash_window(5) is False and winproc.is_foreground(5) is False and winproc.find_own_window() is None


def test_the_reaper_log_stays_small(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log = tmp_path / "logs" / "reaper.log"
    winproc._reaper_line(str(log), "parent 1 ended; terminated 0")
    assert log.read_text(encoding="utf-8").endswith("parent 1 ended; terminated 0\n") and log.read_text(encoding="utf-8")[4] == "-"
    monkeypatch.setattr(winproc, "REAPER_LOG_MAX_BYTES", 2000)
    for i in range(400):
        winproc._reaper_line(str(log), f"parent {i} ended; terminated 0")
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) < 300 and lines[-1].endswith("parent 399 ended; terminated 0")  # the newest lines survive the trimming

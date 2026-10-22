"""Window shell: arguments, --selfcheck (in-process), activity tracking, single instance, job shutdown."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any, Iterator

import pytest
from o_helpers import drop_runner, fresh_runner, make_env

from vndub_app import appdirs, jobs, shell, winproc


@pytest.fixture
def o_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    data = make_env(monkeypatch, tmp_path, profile="apptest")
    fresh_runner()
    yield data
    try:
        from vndub_app import server

        server.close_logging()
    except Exception:
        pass
    drop_runner()


# ---- arguments ------------------------------------------------------------------------------------------------------------


def test_parser_has_the_documented_flags_and_hides_the_rest() -> None:
    p = shell.build_parser()
    a = p.parse_args(["--browser", "--port", "8123", "--data-dir", "D:/x", "--profile", "apptest"])
    assert a.browser and a.port == 8123 and a.data_dir == "D:/x" and a.profile == "apptest" and not a.selfcheck
    d = p.parse_args([])
    assert d.browser is False and d.port == 0 and d.data_dir == "" and d.profile == ""
    h = p.format_help()
    for visible in ("--browser", "--port", "--data-dir", "--profile"):
        assert visible in h
    for hidden in ("--selfcheck", "--selfcheck-out", "--idle-exit", "--debug"):
        assert hidden not in h
    with pytest.raises(SystemExit):
        p.parse_args(["--profile", "nonsense"])


def test_apply_args_sets_the_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("VNDUB_DATA", raising=False)
    monkeypatch.delenv("VNDUB_APP_PROFILE", raising=False)
    shell.apply_args(shell.build_parser().parse_args(["--data-dir", str(tmp_path / "d"), "--profile", "test"]))
    assert os.environ["VNDUB_DATA"] == str((tmp_path / "d").resolve()) and os.environ["VNDUB_APP_PROFILE"] == "test"
    monkeypatch.delenv("VNDUB_DATA")
    monkeypatch.delenv("VNDUB_APP_PROFILE")


def test_free_port_is_free() -> None:
    import socket

    port = shell.find_free_port()
    with socket.socket() as s:
        s.bind((shell.HOST, port))


def test_entry_points_exist() -> None:
    root = Path(__file__).resolve().parents[2]
    assert (root / "vndub_app" / "__main__.py").exists()
    assert 'vndub-app = "vndub_app.shell:main"' in (root / "pyproject.toml").read_text(encoding="utf-8")
    assert callable(shell.main)


# ---- selfcheck ---------------------------------------------------------------------------------------------------------------


def test_selfcheck_starts_the_server_and_reports_json(o_env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "selfcheck.json"
    before = {k: os.environ.get(k) for k in ("VNDUB_DATA", "VNDUB_APP_PROFILE", "VNDUB_ENVS_DIR")}
    code = shell.main(["--selfcheck", "--selfcheck-out", str(out)], hard_exit=False)
    res = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0 and res["ok"] is True and res["problems"] == [], res
    assert res["selfcheck"] is True and res["health"]["ok"] is True and isinstance(res["port"], int) and res["index_status"] == 200
    assert res["resources"]["missing"] == [] and res["imports"]["stages"] >= 12 and res["imports"]["backends"] > 40 and res["imports"]["engines"] == 12
    assert res["frozen"] is False and res["elapsed_s"] > 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1]) == res  # the same JSON line on stdout
    assert {k: os.environ.get(k) for k in before} == before  # the throw-away data folder / profile are undone
    assert not any(p.name.startswith("vndub-selfcheck-") for p in Path(os.environ.get("TEMP", "/tmp")).glob("vndub-selfcheck-*") if p.stat().st_mtime > time.time() - 120)


def test_selfcheck_reports_problems_and_exits_1(o_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vndub_app import server

    def broken() -> Any:
        raise RuntimeError("cannot build the app")

    monkeypatch.setattr(server, "create_app", broken)
    out = tmp_path / "selfcheck.json"
    code = shell.main(["--selfcheck", "--selfcheck-out", str(out)], hard_exit=False)
    res = json.loads(out.read_text(encoding="utf-8"))
    assert code == 1 and res["ok"] is False and any("server did not start" in p and "cannot build the app" in p for p in res["problems"])


# ---- inline pages ---------------------------------------------------------------------------------------------------------------


def test_inline_pages_take_their_text_from_the_i18n_tables(o_env: Path) -> None:
    ru, en = shell.loading_html("ru"), shell.loading_html("en")
    assert "Запускаю VNDub" in ru and "Starting VNDub" in en and "<title>VNDub</title>" in ru
    err = shell.error_html("<b>boom</b> & more", "ru")
    assert "&lt;b&gt;boom&lt;/b&gt; &amp; more" in err and "<b>boom</b>" not in err and "app.log" in err
    assert shell._t("no.such.key", "ru") == ""  # unknown keys give nothing, never a hard-coded text


# ---- activity tracking -------------------------------------------------------------------------------------------------------------


def test_activity_wrapper_counts_http_and_websockets() -> None:
    tracker = shell.ActivityTracker()
    log: list[str] = []

    async def inner(scope: dict[str, Any], receive: Any, send: Any) -> None:
        log.append(scope["type"])
        if scope["type"] == "websocket":
            assert tracker.open_ws == 1 and tracker.idle_for() == 0.0  # an open socket means "in use"

    wrapped = shell.ActivityASGI(inner, tracker)

    async def drive() -> None:
        await wrapped({"type": "lifespan"}, None, None)
        await wrapped({"type": "http"}, None, None)
        await wrapped({"type": "websocket"}, None, None)

    asyncio.run(drive())
    assert log == ["lifespan", "http", "websocket"] and tracker.open_ws == 0
    tracker.last -= 100
    assert tracker.idle_for() >= 99
    tracker.touch()
    assert tracker.idle_for() < 5

    async def failing(scope: dict[str, Any], receive: Any, send: Any) -> None:
        raise RuntimeError("handler crashed")

    t2 = shell.ActivityTracker()
    with pytest.raises(RuntimeError):
        asyncio.run(shell.ActivityASGI(failing, t2)({"type": "websocket"}, None, None))
    assert t2.open_ws == 0  # the counter never leaks


def tiny_app() -> Any:
    from fastapi import FastAPI

    app = FastAPI()

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True}

    return app


def test_browser_mode_exits_when_the_tab_is_gone(o_env: Path) -> None:
    state = shell.start_server_async(0, app_factory=tiny_app)
    try:
        t0 = time.monotonic()
        assert shell.wait_for_health(state, 30.0) == state.url
        code = shell.run_browser(state, idle_exit_s=1.5, open_browser=False)  # VNDUB_NO_BROWSER is set anyway
        assert code == 0 and 1.0 < time.monotonic() - t0 < 25
    finally:
        state.stop()


def test_browser_mode_stays_while_a_job_is_active(o_env: Path) -> None:
    state = shell.start_server_async(0, app_factory=tiny_app)
    r = jobs.runner()
    release = {"go": False}

    def long_job(ctx: Any) -> None:
        while not release["go"]:
            ctx.checkpoint()
            time.sleep(0.05)

    job = r.submit("report", long_job, gpu=False)
    result: dict[str, Any] = {}
    import threading

    th = threading.Thread(target=lambda: result.setdefault("code", shell.run_browser(state, idle_exit_s=0.5, open_browser=False)), daemon=True)
    try:
        assert shell.wait_for_health(state, 30.0)
        th.start()
        time.sleep(2.5)
        assert th.is_alive()  # idle for more than 0.5 s, but a job is running
        release["go"] = True
        r.wait(job.id, 10)
        th.join(timeout=20)
        assert not th.is_alive() and result["code"] == 0
    finally:
        release["go"] = True
        state.stop()


def test_server_start_failure_is_reported_not_raised(o_env: Path) -> None:
    def boom() -> Any:
        raise RuntimeError("no app today")

    state = shell.start_server_async(0, app_factory=boom)
    assert shell.wait_for_health(state, 10.0) is None
    assert isinstance(state.error, RuntimeError) and state.url is None
    assert shell.run_browser(state, idle_exit_s=1.0, open_browser=False) == 1


# ---- single instance ---------------------------------------------------------------------------------------------------------------


def test_second_instance_cannot_take_the_lock(tmp_path: Path) -> None:
    a, b = shell.InstanceLock(tmp_path), shell.InstanceLock(tmp_path)
    assert a.acquire() is True
    a.publish("http://127.0.0.1:5555")
    assert b.acquire() is False and b.running_url() == "http://127.0.0.1:5555"
    a.release()
    assert not (tmp_path / "app.instance.json").exists()
    assert b.acquire() is True
    b.release()
    assert shell.InstanceLock(tmp_path).running_url() is None


# ---- closing the window -------------------------------------------------------------------------------------------------------------


def test_stop_jobs_cancels_cleanly_and_marks_active_jobs_interrupted(o_env: Path) -> None:
    r = jobs.runner()
    cleaned: list[str] = []

    def worker(ctx: Any) -> None:
        try:
            while True:
                ctx.checkpoint()
                time.sleep(0.02)
        finally:
            cleaned.append("models closed")  # what a pipeline job does in its finally clause

    running = r.submit("full", worker, gpu=True)
    queued = r.submit("full", worker, gpu=True)
    quick = r.submit("report", lambda ctx: None, gpu=False)
    r.wait(quick.id, 10)
    t0 = time.monotonic()
    while r.get(running.id).state != "running" and time.monotonic() - t0 < 10:
        time.sleep(0.02)
    stopped = shell.stop_jobs(timeout=10)
    assert set(stopped) == {running.id, queued.id}
    assert r.get(running.id).state == "interrupted" and r.get(queued.id).state == "interrupted"  # next start offers "continue"
    assert r.get(quick.id).state == "done"  # finished jobs stay as they were
    assert cleaned == ["models closed"]
    assert [j.id for j in r.interrupted()] == [running.id, queued.id] or {j.id for j in r.interrupted()} == {running.id, queued.id}


# ---- no console windows --------------------------------------------------------------------------------------------------------------


@pytest.mark.skipif(os.name != "nt", reason="CREATE_NO_WINDOW is a Windows flag")
def test_child_consoles_are_hidden_and_the_patch_is_reversible() -> None:
    import subprocess
    import sys

    from vndub_app import winproc

    seen: list[int] = []
    real_init = subprocess.Popen.__init__

    def spy(self: Any, *a: Any, **kw: Any) -> None:
        seen.append(int(kw.get("creationflags", 0) or 0))
        real_init(self, *a, **kw)

    subprocess.Popen.__init__ = spy  # type: ignore[method-assign]
    undo = winproc.hide_child_consoles()
    try:
        assert winproc.hide_child_consoles() is not None  # idempotent: no double wrapping
        r = subprocess.run([sys.executable, "-c", "print('hi')"], capture_output=True, text=True)
        assert r.stdout.strip() == "hi" and seen[-1] & winproc.CREATE_NO_WINDOW
        subprocess.run([sys.executable, "-c", "pass"], creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        assert seen[-1] & winproc.CREATE_NO_WINDOW and seen[-1] & subprocess.CREATE_NEW_PROCESS_GROUP  # other flags are kept
        subprocess.run([sys.executable, "-c", "pass"], creationflags=winproc.CREATE_NEW_CONSOLE)
        assert not seen[-1] & winproc.CREATE_NO_WINDOW  # a call that wants its own console keeps it
    finally:
        undo()
        subprocess.Popen.__init__ = real_init  # type: ignore[method-assign]
    assert winproc._ORIGINAL is None


def test_preferred_port_is_stable_per_data_folder_and_free(tmp_path: Path) -> None:
    import socket

    a = shell.preferred_port(tmp_path / "A")
    assert a == shell.preferred_port(tmp_path / "A") and 49152 <= a < 61152
    with socket.socket() as s:  # a taken preferred port moves on to the next free one
        s.bind((shell.HOST, a))
        b = shell.preferred_port(tmp_path / "A")
        assert b != a and shell.port_is_free(b)
    assert shell.port_is_free(a)


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Objects")
def test_children_die_with_the_app(tmp_path: Path) -> None:
    """A throw-away parent process installs the job, starts a sleeping child and exits: the child must be gone."""
    import ctypes
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[2]
    code = (
        "import subprocess, sys, time\n"
        f"sys.path.insert(0, r'{repo}')\n"
        "from vndub_app import winproc\n"
        "assert winproc.kill_children_on_exit() is True\n"
        "assert winproc.kill_children_on_exit() is True\n"  # idempotent
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        "print(c.pid, flush=True)\n"
        "time.sleep(1.5)\n"
    )
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    pid = int(p.stdout.split()[-1])
    k32 = ctypes.windll.kernel32
    k32.OpenProcess.restype = ctypes.c_void_p
    deadline = time.monotonic() + 10
    alive = True
    while time.monotonic() < deadline:
        h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            alive = False
            break
        code_ = ctypes.c_ulong()
        k32.GetExitCodeProcess(ctypes.c_void_p(h), ctypes.byref(code_))
        k32.CloseHandle(ctypes.c_void_p(h))
        if code_.value != 259:  # STILL_ACTIVE
            alive = False
            break
        time.sleep(0.2)
    if alive:
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
    assert not alive, "the child survived its parent"


# ---- stop_jobs: one deadline for all jobs ---------------------------------------------------------------------------------------


def test_stop_jobs_waits_one_shared_deadline_not_one_per_job(o_env: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Two jobs that never look at the cancel flag: closing the window must cost one timeout, not two."""
    r = jobs.runner()
    release = threading.Event()
    started = threading.Barrier(3)

    def deaf(ctx: Any) -> None:  # a model call that does not return: no checkpoint, no progress
        started.wait(10)
        release.wait(30)

    a, b = r.submit("full", deaf, gpu=False), r.submit("full", deaf, gpu=False)
    try:
        started.wait(10)
        caplog.set_level(logging.INFO, logger="vndub_app.shell")
        t0 = time.monotonic()
        stopped = shell.stop_jobs(timeout=1.0)
        elapsed = time.monotonic() - t0
        assert set(stopped) == {a.id, b.id}
        assert 0.9 <= elapsed < 1.8, elapsed  # a per-job timeout would take 2 s
        warning = next(rec.getMessage() for rec in caplog.records if rec.levelno == logging.WARNING)
        assert "2 of 2 job" in warning and a.id in warning and b.id in warning and "running" in warning  # the log says who was stuck
    finally:
        release.set()
        r.wait(a.id, 10), r.wait(b.id, 10)


def test_stop_jobs_does_not_wait_for_the_deadline_when_the_jobs_stop(o_env: Path) -> None:
    """The deadline is shared, but jobs that stop in time are not waited for until it passes."""
    r = jobs.runner()

    def polite(ctx: Any) -> None:
        while True:
            ctx.checkpoint()
            time.sleep(0.02)

    for _ in range(3):
        r.submit("full", polite, gpu=False)
    time.sleep(0.2)
    t0 = time.monotonic()
    assert len(shell.stop_jobs(timeout=20.0)) == 3
    assert time.monotonic() - t0 < 5  # cancelled together, not one after the other
    assert all(j.state == "interrupted" for j in r.list())


# ---- ending the process ---------------------------------------------------------------------------------------------------------


class _Exited(Exception):
    """What the replaced ``_hard_exit`` raises in the main thread: ``os._exit`` does not return either."""

    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.code = code


class ExitRecorder:
    """Stands in for ``shell._hard_exit``: records (code, calling thread); only the main thread gets the exception."""

    def __init__(self, events: list[str]) -> None:
        self.calls: list[tuple[int, str]] = []
        self.fired = threading.Event()
        self.events = events
        self.startups = 0  # how often main() ran the startup housekeeping (it would start a real reaper)

    def __call__(self, code: int) -> None:
        name = threading.current_thread().name
        self.calls.append((code, name))
        self.events.append(f"exit:{name}")
        self.fired.set()
        if threading.current_thread() is threading.main_thread():
            raise _Exited(code)


def _watchdogs() -> list[Any]:
    return [t for t in threading.enumerate() if t.name == "vndub-exit-watchdog"]


@pytest.fixture
def main_env(o_env: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[ExitRecorder]:
    """``shell.main`` runs in the test process: nothing global is touched and no real exit can happen."""
    from vndub_app import webbuild

    events: list[str] = []
    rec = ExitRecorder(events)
    real_stop_jobs = shell.stop_jobs

    def spy_stop_jobs(*a: Any, **kw: Any) -> list[str]:
        events.append("stop_jobs")
        events.append(f"watchdog_alive_at_cleanup:{any(t.is_alive() for t in _watchdogs())}")
        return real_stop_jobs(*a, **kw)

    def spy_sweep() -> None:  # the real sweep would terminate the descendants of the pytest process itself
        events.append("final_sweep")

    def spy_startup() -> None:  # the real one terminates orphans and starts a reaper
        rec.startups += 1

    monkeypatch.setattr(shell, "_hard_exit", rec)
    monkeypatch.setattr(shell, "_final_sweep", spy_sweep)
    monkeypatch.setattr(shell, "_startup_housekeeping", spy_startup)
    monkeypatch.setattr(shell, "stop_jobs", spy_stop_jobs)

    def fake_logging_shutdown(handlers: Any = None) -> None:
        if handlers is None:  # uvicorn's dictConfig calls logging.shutdown(handler_list) itself: only main's own call counts
            events.append("logging.shutdown")

    monkeypatch.setattr(logging, "shutdown", fake_logging_shutdown)
    monkeypatch.setattr(winproc, "hide_child_consoles", lambda: (lambda: None))  # a global Popen patch / Job Object would outlive the test
    monkeypatch.setattr(winproc, "kill_children_on_exit", lambda: True)
    monkeypatch.setattr(shell, "_ROOTS_SEEN", [])
    monkeypatch.setattr(webbuild, "ensure_built", lambda *a, **kw: {})  # never `npm run build` from a test
    yield rec
    for wd in _watchdogs():  # a leftover one would end the whole pytest run when it fires
        wd.disarm()
        wd.join(5)
    assert not _watchdogs()


def test_main_ends_the_process_directly_after_the_cleanup(main_env: ExitRecorder, o_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The regression: the exit used to be a sleeping thread created after main() had returned, i.e. one that never ran."""
    from vndub_app import server

    monkeypatch.setattr(server, "create_app", tiny_app)
    with pytest.raises(_Exited) as e:
        shell.main(["--browser", "--idle-exit", "1"])  # real server, real browser-mode loop; leaves when the "tab" is gone
    assert e.value.code == 0
    assert main_env.calls == [(0, "MainThread")]  # called directly and synchronously, not from a helper thread
    assert main_env.events == ["stop_jobs", "watchdog_alive_at_cleanup:True", "final_sweep", "logging.shutdown", "exit:MainThread"]  # swept before app.log is closed
    assert main_env.startups == 1  # the startup housekeeping (orphans of an earlier run, the reaper) ran once, in the server thread
    assert not (o_env / "app.instance.json").exists()  # the instance lock was released before the exit
    (wd,) = _watchdogs()
    assert wd.daemon and wd.is_alive()  # armed (it is the safety net) and not fired


def test_main_without_hard_exit_returns_normally(main_env: ExitRecorder, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shell, "start_server_async", lambda *a, **kw: shell.ServerState())
    monkeypatch.setattr(shell, "run_browser", lambda state, idle_exit_s, **kw: 7)
    assert shell.main(["--browser"], hard_exit=False) == 7
    assert main_env.calls == [] and _watchdogs() == []  # tests and `--selfcheck` style runs never get a watchdog or an exit
    assert main_env.events[0] == "stop_jobs"


def test_hard_exit_sweeps_and_then_ends_the_process_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    """Job Object or not: the sweep comes first, the end is ``winproc.end_process`` (TerminateProcess), never a plain return."""
    seen: list[Any] = []
    monkeypatch.setattr(shell, "_final_sweep", lambda: seen.append("sweep"))
    monkeypatch.setattr(winproc, "end_process", seen.append)
    shell._hard_exit(3)
    assert seen == ["sweep", 3]
    monkeypatch.setattr(winproc, "job_object_status", lambda: {"installed": True, "error": None})  # the Job Object does not excuse the sweep
    seen.clear()
    shell._hard_exit(4)
    assert seen == ["sweep", 4]


def test_the_watchdog_forces_the_exit_when_the_cleanup_hangs(main_env: ExitRecorder, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shell, "start_server_async", lambda *a, **kw: shell.ServerState())
    monkeypatch.setattr(shell, "run_browser", lambda state, idle_exit_s, **kw: 0)
    monkeypatch.setattr(shell, "EXIT_WATCHDOG_S", 0.4)

    def stuck(*a: Any, **kw: Any) -> list[str]:  # a job that never stops: waits until something ends the process
        main_env.events.append("stop_jobs")
        main_env.fired.wait(15)
        return []

    monkeypatch.setattr(shell, "stop_jobs", stuck)
    t0 = time.monotonic()
    with pytest.raises(_Exited):
        shell.main(["--browser"])
    assert main_env.calls[0] == (0, "vndub-exit-watchdog")  # the watchdog ended it while the cleanup was still stuck ...
    assert 0.3 <= time.monotonic() - t0 < 10
    assert main_env.calls[-1] == (0, "MainThread")  # ... (here the stuck step then returns and main's own exit follows)
    assert main_env.events[:2] == ["stop_jobs", "exit:vndub-exit-watchdog"]


def test_the_watchdog_can_be_disarmed_and_never_fires_early(monkeypatch: pytest.MonkeyPatch) -> None:
    fired: list[int] = []
    monkeypatch.setattr(shell, "_hard_exit", fired.append)
    wd = shell._start_exit_watchdog(5, delay=30.0)
    try:
        assert wd.name == "vndub-exit-watchdog" and wd.daemon and wd.is_alive()
        time.sleep(0.2)
        assert fired == []
    finally:
        wd.disarm()
        wd.join(5)
    assert fired == [] and not wd.is_alive()
    short = shell._start_exit_watchdog(9, delay=0.1)
    short.join(10)
    assert fired == [9]


def test_a_crash_in_the_loop_still_cleans_up_and_exits_with_1(main_env: ExitRecorder, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shell, "start_server_async", lambda *a, **kw: shell.ServerState())

    def boom(state: Any, idle_exit_s: Any, **kw: Any) -> int:
        raise RuntimeError("the loop died")

    monkeypatch.setattr(shell, "run_browser", boom)
    with pytest.raises(_Exited) as e:
        shell.main(["--browser"])
    assert e.value.code == 1 and main_env.calls == [(1, "MainThread")] and main_env.events[0] == "stop_jobs"
    main_env.events.clear()
    with pytest.raises(RuntimeError):  # without hard_exit (tests) the error is not swallowed, the cleanup still ran
        shell.main(["--browser"], hard_exit=False)
    assert main_env.events[0] == "stop_jobs"


def test_a_window_that_cannot_start_falls_back_to_the_browser_and_exits_the_same_way(main_env: ExitRecorder, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shell, "start_server_async", lambda *a, **kw: shell.ServerState())

    def no_window(state: Any, debug: bool = False) -> int:
        raise RuntimeError("no WebView2")

    monkeypatch.setattr(shell, "run_window", no_window)
    monkeypatch.setattr(shell, "run_browser", lambda state, idle_exit_s, **kw: 0)
    with pytest.raises(_Exited) as e:
        shell.main([])
    assert e.value.code == 0 and main_env.events == ["stop_jobs", "watchdog_alive_at_cleanup:True", "final_sweep", "logging.shutdown", "exit:MainThread"]


def test_a_failing_cleanup_step_does_not_skip_the_next_ones(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    done: list[str] = []

    def bad_jobs(*a: Any, **kw: Any) -> list[str]:
        raise RuntimeError("jobs broke")

    class State:
        def stop(self) -> None:
            done.append("server")

    class Lock:
        def release(self) -> None:
            done.append("lock")

    monkeypatch.setattr(shell, "stop_jobs", bad_jobs)
    caplog.set_level(logging.WARNING, logger="vndub_app.shell")
    shell._cleanup_for_exit(State(), Lock())  # type: ignore[arg-type]
    assert done == ["server", "lock"] and "stopping jobs failed: jobs broke" in caplog.text


# ---- the Job Object safety net: why it is missing, in app.log --------------------------------------------------------------------


def test_job_object_status_has_the_documented_shape() -> None:
    st = winproc.job_object_status()
    assert set(st) == {"installed", "error"} and isinstance(st["installed"], bool)
    assert st["error"] is None or isinstance(st["error"], str)


def test_job_object_status_is_logged_as_info_or_warning(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="vndub_app.shell")
    denied = "AssignProcessToJobObject failed: Access is denied. (error 5)"
    for status, level, text in (
        ({"installed": True, "error": None}, logging.INFO, "Job Object installed"),
        ({"installed": False, "error": denied}, logging.WARNING, f"NOT installed ({denied})"),
    ):
        caplog.clear()
        monkeypatch.setattr(winproc, "job_object_status", lambda status=status: status)
        shell._log_job_object_status()
        assert [(r.levelno, text in r.getMessage()) for r in caplog.records] == [(level, True)]
    caplog.clear()
    monkeypatch.setattr(winproc, "job_object_status", lambda: {"installed": False, "error": None})  # never tried (selfcheck, not Windows)
    shell._log_job_object_status()
    assert caplog.records == []


def test_the_server_thread_logs_the_job_object_status_right_after_logging_is_set_up(o_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vndub_app import server

    events: list[str] = []
    real_setup = server.setup_logging

    def setup() -> Any:
        events.append("setup_logging")
        return real_setup()

    monkeypatch.setattr(server, "setup_logging", setup)
    monkeypatch.setattr(shell, "_log_job_object_status", lambda: events.append("job_object_status"))
    state = shell.start_server_async(0, app_factory=tiny_app)
    try:
        assert shell.wait_for_health(state, 30.0) == state.url
        assert events == ["setup_logging", "job_object_status"]
    finally:
        state.stop()


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    repo = Path(__file__).resolve().parents[2]
    return subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, r'{repo}')\n{code}"], capture_output=True, text=True, timeout=60)


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Objects")
def test_job_object_status_reports_installed_in_a_fresh_process() -> None:
    p = _run_python(
        "import json\n"
        "from vndub_app import winproc\n"
        "before = winproc.job_object_status()\n"
        "ok = winproc.kill_children_on_exit()\n"
        "print(json.dumps([before, ok, winproc.job_object_status(), winproc.JOB_OBJECT_ERROR]))\n"
    )
    assert p.returncode == 0, p.stderr
    before, ok, after, error = json.loads(p.stdout.strip().splitlines()[-1])
    assert before == {"installed": False, "error": None} and ok is True and after == {"installed": True, "error": None} and error is None


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Objects")
def test_a_failing_job_object_records_the_step_and_the_windows_error() -> None:
    """Make AssignProcessToJobObject fail with ERROR_ACCESS_DENIED (what a job that forbids nesting gives), then recover."""
    p = _run_python(
        "import ctypes, json\n"
        "from vndub_app import winproc\n"
        "real = ctypes.WinDLL\n"
        "class Fake:\n"
        "    def __init__(self, k32):\n"
        "        self._k32 = k32\n"
        "        def assign(*a):\n"
        "            ctypes.set_last_error(5)\n"
        "            return 0\n"
        "        self.AssignProcessToJobObject = assign\n"
        "    def __getattr__(self, name):\n"
        "        return getattr(self._k32, name)\n"
        "ctypes.WinDLL = lambda name, **kw: Fake(real(name, **kw))\n"
        "failed = winproc.kill_children_on_exit()\n"
        "status = winproc.job_object_status()\n"
        "ctypes.WinDLL = real\n"
        "recovered = winproc.kill_children_on_exit()\n"
        "print(json.dumps([failed, status, recovered, winproc.job_object_status()]))\n"
    )
    assert p.returncode == 0, p.stderr
    failed, status, recovered, final = json.loads(p.stdout.strip().splitlines()[-1])
    assert failed is False and status["installed"] is False
    assert "AssignProcessToJobObject" in status["error"] and "error 5" in status["error"]
    assert recovered is True and final == {"installed": True, "error": None}  # the error is cleared by a later success


def _pid_alive(pid: int) -> bool:
    import ctypes

    k32 = ctypes.windll.kernel32
    k32.OpenProcess.restype = ctypes.c_void_p
    h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    k32.GetExitCodeProcess(ctypes.c_void_p(h), ctypes.byref(code))
    k32.CloseHandle(ctypes.c_void_p(h))
    return code.value == 259  # STILL_ACTIVE


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Objects")
def test_the_real_app_process_ends_at_once_and_takes_its_children_with_it(tmp_path: Path) -> None:
    """End to end in a fresh process: a thread that never ends (a stand-in for pythonnet / WebView2) must not keep the app alive.

    The old exit was a daemon thread that slept 10 s before ``os._exit``, so the process lived at least 10 s longer
    than the ~3 s a clean run needs; the 10 s limit below tells the two apart.
    """
    repo = Path(__file__).resolve().parents[2]
    code = (
        "import subprocess, sys, threading, time\n"
        f"sys.path.insert(0, r'{repo}')\n"
        "from vndub_app import shell\n"
        "threading.Thread(target=threading.Event().wait, daemon=False).start()\n"  # the interpreter could never finish on its own
        "real = shell.run_browser\n"
        "def run_browser(state, idle_exit_s, **kw):\n"
        "    c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        "    print('child', c.pid, time.time(), flush=True)\n"
        "    return real(state, idle_exit_s, **kw)\n"
        "shell.run_browser = run_browser\n"
        "sys.exit(shell.main(['--browser', '--idle-exit', '1']))\n"
    )
    env = {
        **os.environ, "VNDUB_DATA": str(tmp_path / "data"), "VNDUB_APP_PROFILE": "apptest", "VNDUB_NO_BROWSER": "1",
        "VNDUB_NO_WEB_BUILD": "1", "VNDUB_NO_SHUTDOWN": "1", "VNDUB_ENVS_DIR": str(tmp_path / "envs"), "LOCALAPPDATA": str(tmp_path / "local"), "APPDATA": str(tmp_path / "roaming"),
    }
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    pid = 0
    try:
        out, err = p.communicate(timeout=60)
        ended = time.time()
        line = next(ln for ln in out.splitlines() if ln.startswith("child "))
        pid, started = int(line.split()[1]), float(line.split()[2])
        assert p.returncode == 0, err
        assert ended - started < 10, ended - started  # ~3 s: health check + 1 s idle + cleanup; no sleeping exit thread
        deadline = time.monotonic() + 10
        while _pid_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not _pid_alive(pid), "the child survived the end of the app"
    finally:
        if p.poll() is None:
            p.kill()
        if pid and _pid_alive(pid):
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)


# ---- nothing the app started may outlive it: the sweeps --------------------------------------------------------------------------


def _fake_proc(pid: int, exe: str = r"C:\d\envs\main\Scripts\python.exe") -> winproc.ProcInfo:
    return winproc.ProcInfo(pid, 1, exe, Path(exe).name, "main", "adapter", False, True, 3)


@pytest.mark.skipif(os.name != "nt", reason="the sweeps are Windows only")
def test_final_sweep_takes_the_children_then_the_orphans_and_logs_it(o_env: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Job Object or not: the exit sweeps. Twice in a row (main's, then _hard_exit's) it only runs once."""
    from vndub_app import installer

    monkeypatch.setattr(shell, "_SWEPT_AT", None)
    monkeypatch.setattr(shell, "_ROOTS_SEEN", [])
    calls: list[Any] = []
    monkeypatch.setattr(winproc, "kill_descendants", lambda *a: calls.append("descendants") or 2)
    monkeypatch.setattr(winproc, "kill_orphans", lambda roots, reason="": calls.append(("orphans", list(roots), reason)) or [_fake_proc(7)])
    monkeypatch.setattr(winproc, "job_object_status", lambda: {"installed": True, "error": None})  # the Job Object does not excuse it
    caplog.set_level(logging.INFO, logger="vndub_app.shell")
    shell._final_sweep()
    shell._final_sweep()
    assert calls == ["descendants", ("orphans", installer.process_roots(), "orphan at the end of the app")]
    msg = next(r.getMessage() for r in caplog.records if "exit sweep" in r.getMessage())
    assert "terminated 2 child process(es) and 1 orphan(s)" in msg and "pid 7" in msg
    monkeypatch.setattr(shell, "_SWEPT_AT", time.monotonic() - shell.SWEEP_ONCE_S - 1)  # later it runs again (a watchdog after a stuck cleanup)
    shell._final_sweep()
    assert len(calls) == 4


@pytest.mark.skipif(os.name != "nt", reason="the sweeps are Windows only")
def test_final_sweep_never_raises(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setattr(shell, "_SWEPT_AT", None)

    def boom(*a: Any) -> int:
        raise OSError("no snapshot")

    monkeypatch.setattr(winproc, "kill_descendants", boom)
    shell._final_sweep()
    assert "exit sweep failed: no snapshot" in caplog.text


@pytest.mark.skipif(os.name != "nt", reason="the sweeps are Windows only")
def test_startup_housekeeping_terminates_the_orphans_of_an_earlier_run_and_starts_the_reaper(o_env: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    from vndub_app import installer

    monkeypatch.setattr(shell, "_ROOTS_SEEN", [])
    seen: dict[str, Any] = {}
    monkeypatch.setattr(winproc, "kill_orphans", lambda roots, reason="": seen.setdefault("orphans", (list(roots), reason)) and [_fake_proc(11), _fake_proc(12, r"C:\d\tools\llama\llama-server.exe")])
    monkeypatch.setattr(winproc, "start_reaper", lambda roots, log_file: seen.setdefault("reaper", (list(roots), Path(log_file))) and {"pid": 4242, "breakaway": True, "error": None})
    caplog.set_level(logging.INFO, logger="vndub_app.shell")
    shell._startup_housekeeping()
    roots = installer.process_roots()
    assert seen["orphans"] == (roots, "orphan left by a previous run") and seen["reaper"] == (roots, appdirs.logs_dir() / "reaper.log")
    text = caplog.text
    assert "startup: terminated 2 orphan process(es) left by a previous run: pid 11" in text and "pid 12" in text and "llama-server.exe" in text
    assert "reaper started (pid 4242)" in text
    assert shell._current_roots() == roots  # the exit sweep knows the same folders


@pytest.mark.skipif(os.name != "nt", reason="the sweeps are Windows only")
def test_startup_housekeeping_is_quiet_without_orphans_and_says_when_the_reaper_is_inside_the_job(o_env: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setattr(shell, "_ROOTS_SEEN", [])
    monkeypatch.setattr(winproc, "kill_orphans", lambda roots, reason="": [])
    monkeypatch.setattr(winproc, "start_reaper", lambda roots, log_file: {"pid": 9, "breakaway": False, "error": "breakaway from the job refused (denied); the reaper runs inside the app's job"})
    caplog.set_level(logging.INFO, logger="vndub_app.shell")
    shell._startup_housekeeping()
    assert "orphan process" not in caplog.text  # nothing to report
    warning = next(r for r in caplog.records if r.levelno == logging.WARNING)
    assert "reaper started (pid 9)" in warning.getMessage() and "breakaway from the job refused" in warning.getMessage()
    monkeypatch.setattr(winproc, "start_reaper", lambda roots, log_file: {"pid": None, "breakaway": False, "error": "OSError: no exe"})
    caplog.clear()
    shell._startup_housekeeping()
    assert "reaper not started: OSError: no exe" in caplog.text


def test_startup_housekeeping_never_raises(o_env: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def boom(*a: Any, **kw: Any) -> Any:
        raise RuntimeError("process table unreadable")

    monkeypatch.setattr(winproc, "kill_orphans", boom)
    shell._startup_housekeeping()
    if os.name == "nt":
        assert "startup housekeeping failed: process table unreadable" in caplog.text


def test_the_startup_hook_runs_in_the_server_thread_after_logging_and_before_the_app_exists(o_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vndub_app import server

    events: list[str] = []
    real_setup = server.setup_logging

    def setup() -> Any:
        events.append("setup_logging")
        return real_setup()

    def factory() -> Any:
        events.append("create_app")
        return tiny_app()

    monkeypatch.setattr(server, "setup_logging", setup)
    monkeypatch.setattr(shell, "_log_job_object_status", lambda: events.append("job_object_status"))
    state = shell.start_server_async(0, app_factory=factory, startup=lambda: events.append("startup:" + threading.current_thread().name))
    try:
        assert shell.wait_for_health(state, 30.0) == state.url
        assert events == ["setup_logging", "job_object_status", "startup:vndub-server-start", "create_app"]  # no job can run before the sweep
    finally:
        state.stop()

    def failing_hook() -> None:
        raise RuntimeError("sweep failed")

    broken = shell.start_server_async(0, app_factory=tiny_app, startup=failing_hook)
    try:
        assert shell.wait_for_health(broken, 30.0) == broken.url  # a failing hook never stops the app from starting
    finally:
        broken.stop()


def test_main_runs_the_startup_housekeeping_only_for_a_real_run(main_env: ExitRecorder, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shell, "run_browser", lambda state, idle_exit_s, **kw: 0)
    started: list[Any] = []

    def fake_start(*a: Any, **kw: Any) -> shell.ServerState:
        started.append(kw.get("startup"))
        return shell.ServerState()

    monkeypatch.setattr(shell, "start_server_async", fake_start)
    shell.main(["--browser"], hard_exit=False)
    with pytest.raises(_Exited):
        shell.main(["--browser"])
    assert started[0] is None and started[1] is shell._startup_housekeeping  # `hard_exit=False` is the tests' / tools' way in: no reaper, no sweeping


# ---- a window that breaks after it was shown -----------------------------------------------------------------------------------


def test_a_window_that_breaks_after_it_was_shown_ends_the_app_instead_of_carrying_on_in_a_browser(main_env: ExitRecorder, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """pythonnet / WebView2 can raise while the window closes. The user sees the app gone, so it must not live on headless
    (jobs, models and VRAM for the idle-exit minutes) behind a browser tab nobody asked for."""

    def window(state: shell.ServerState, debug: bool = False) -> int:
        state.window_shown = True  # boot() loaded the interface
        raise RuntimeError("System.ObjectDisposedException while closing the form")

    monkeypatch.setattr(shell, "start_server_async", lambda *a, **kw: shell.ServerState())
    monkeypatch.setattr(shell, "run_window", window)
    monkeypatch.setattr(shell, "run_browser", lambda *a, **kw: pytest.fail("no browser fallback for a window that was up"))
    with pytest.raises(_Exited) as e:
        shell.main([])
    assert e.value.code == 1 and main_env.events[:1] == ["stop_jobs"]
    assert "the window stopped unexpectedly" in caplog.text and "ObjectDisposedException" in caplog.text


def test_boot_marks_the_window_as_shown_and_hooks_the_attention(monkeypatch: pytest.MonkeyPatch, o_env: Path) -> None:
    """``run_window``'s boot thread: the interface is loaded, then ``install_attention`` subscribes (a failing hook must not break the boot)."""
    shown: list[Any] = []

    class Closed:
        def __iadd__(self, fn: Any) -> "Closed":
            return self

    class Window:
        events = types.SimpleNamespace(closed=Closed())

        def load_url(self, url: str) -> None:
            shown.append(url)

        def load_html(self, html: str) -> None:
            shown.append("html")

    win = Window()
    boots: list[Any] = []
    fake_webview = types.SimpleNamespace(create_window=lambda *a, **kw: win, settings={}, start=lambda fn, **kw: boots.append(fn))
    monkeypatch.setitem(sys.modules, "webview", fake_webview)
    from vndub_app import system

    monkeypatch.setattr(system, "set_window", lambda w: None)
    monkeypatch.setattr(shell, "wait_for_health", lambda state, timeout=0: "http://127.0.0.1:1")
    hooked: list[Any] = []

    def failing_hook(w: Any, *a: Any, **kw: Any) -> None:
        hooked.append(w)
        raise RuntimeError("no runner")

    monkeypatch.setattr(shell, "install_attention", failing_hook)
    state = shell.ServerState()
    shell.run_window(state)
    assert len(boots) == 1
    boots[0]()
    assert shown == ["http://127.0.0.1:1"] and state.window_shown is True and hooked == [win]  # the failing hook did not stop anything


# ---- the taskbar flash -----------------------------------------------------------------------------------------------------------


@pytest.fixture
def flashes(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    flashed: list[int] = []
    monkeypatch.setattr(winproc, "flash_window", lambda hwnd, *a, **kw: flashed.append(hwnd) or True)
    return flashed


def _wait_state(job_id: str, states: tuple[str, ...]) -> None:
    t0 = time.monotonic()
    while jobs.runner().get(job_id).state not in states and time.monotonic() - t0 < 10:
        time.sleep(0.02)
    time.sleep(0.15)  # the event of the final state is emitted just after the state changes


def test_a_finished_job_flashes_the_taskbar_button_when_the_window_is_not_in_front(o_env: Path, monkeypatch: pytest.MonkeyPatch, flashes: list[int]) -> None:
    monkeypatch.setattr(winproc, "is_foreground", lambda hwnd: False)
    window = types.SimpleNamespace(native=types.SimpleNamespace(Handle=types.SimpleNamespace(ToInt64=lambda: 777)))  # a WinForms Form: Handle is a System.IntPtr
    unsubscribe = shell.install_attention(window)

    def failing(ctx: Any) -> None:
        raise RuntimeError("model failed")

    try:
        ok = jobs.runner().submit("report", lambda ctx: None, gpu=False)
        _wait_state(ok.id, ("done",))
        assert flashes == [777]  # once, however many events the job emitted
        bad = jobs.runner().submit("report", failing, gpu=False)
        _wait_state(bad.id, ("failed",))
        assert flashes == [777, 777]
    finally:
        unsubscribe()


def test_nothing_flashes_while_the_window_is_in_front_or_for_other_states(o_env: Path, monkeypatch: pytest.MonkeyPatch, flashes: list[int]) -> None:
    front = {"on": True}
    monkeypatch.setattr(winproc, "is_foreground", lambda hwnd: front["on"])
    window = types.SimpleNamespace(native=types.SimpleNamespace(Handle=555))
    unsubscribe = shell.install_attention(window)
    release = threading.Event()

    def spin(ctx: Any) -> None:
        while True:
            ctx.checkpoint()
            time.sleep(0.02)

    try:
        j = jobs.runner().submit("report", lambda ctx: None, gpu=False)
        _wait_state(j.id, ("done",))
        assert flashes == []  # the user is looking at the window
        front["on"] = False
        slow = jobs.runner().submit("report", lambda ctx: release.wait(10) and None, gpu=False)
        time.sleep(0.3)  # queued / running events: no flash
        assert flashes == []
        cancelled = jobs.runner().submit("report", spin, gpu=False)
        time.sleep(0.2)
        jobs.runner().cancel(cancelled.id)  # cancelled: not a result to look at
        _wait_state(cancelled.id, ("cancelled",))
        assert jobs.runner().get(cancelled.id).state == "cancelled" and flashes == []
        release.set()
        _wait_state(slow.id, ("done",))
        assert flashes == [555]  # only the job that really finished
    finally:
        release.set()
        unsubscribe()


def test_attention_finds_the_window_by_title_without_a_native_handle_and_survives_having_none(o_env: Path, monkeypatch: pytest.MonkeyPatch, flashes: list[int]) -> None:
    monkeypatch.setattr(winproc, "is_foreground", lambda hwnd: False)
    looked_for: list[str] = []
    monkeypatch.setattr(winproc, "find_own_window", lambda title="VNDub": looked_for.append(title) or 4711)
    unsubscribe = shell.install_attention(types.SimpleNamespace(native=None), title="VNDub")  # not the EdgeChromium backend
    try:
        _wait_state(jobs.runner().submit("report", lambda ctx: None, gpu=False).id, ("done",))
        assert flashes == [4711] and looked_for == ["VNDub"]
    finally:
        unsubscribe()
    monkeypatch.setattr(winproc, "find_own_window", lambda title="VNDub": None)  # no window at all: nothing happens, nothing raises
    flashes.clear()
    unsubscribe = shell.install_attention(object())
    try:
        _wait_state(jobs.runner().submit("report", lambda ctx: None, gpu=False).id, ("done",))
        assert flashes == []
    finally:
        unsubscribe()


def test_native_handle_accepts_an_intptr_or_an_int_and_nothing_else() -> None:
    assert shell._native_handle(types.SimpleNamespace(native=types.SimpleNamespace(Handle=types.SimpleNamespace(ToInt64=lambda: 9)))) == 9
    assert shell._native_handle(types.SimpleNamespace(native=types.SimpleNamespace(Handle=12))) == 12
    assert shell._native_handle(types.SimpleNamespace(native=types.SimpleNamespace(Handle=0))) is None
    assert shell._native_handle(types.SimpleNamespace()) is None and shell._native_handle(None) is None


def test_cleanup_sweeps_between_the_jobs_and_the_server_stop_when_asked_to(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Stopping the server closes app.log, so the sweep (and what it says) comes before that; a test run (sweep=False) never sweeps."""
    order: list[str] = []

    class State:
        def stop(self) -> None:
            order.append("server")

    class Lock:
        def release(self) -> None:
            order.append("lock")

    monkeypatch.setattr(shell, "stop_jobs", lambda *a, **kw: order.append("jobs") or [])
    monkeypatch.setattr(shell, "_final_sweep", lambda: order.append("sweep"))
    shell._cleanup_for_exit(State(), Lock())  # type: ignore[arg-type]
    assert order == ["jobs", "server", "lock"]
    order.clear()
    shell._cleanup_for_exit(State(), Lock(), sweep=True)  # type: ignore[arg-type]
    assert order == ["jobs", "sweep", "server", "lock"]

    def boom() -> None:
        raise OSError("sweep broke")

    monkeypatch.setattr(shell, "_final_sweep", boom)
    order.clear()
    shell._cleanup_for_exit(State(), Lock(), sweep=True)  # type: ignore[arg-type]
    assert order == ["jobs", "server", "lock"] and "terminating leftover processes failed: sweep broke" in caplog.text

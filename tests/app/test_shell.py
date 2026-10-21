"""Window shell: arguments, --selfcheck (in-process), activity tracking, single instance, job shutdown."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Iterator

import pytest
from o_helpers import drop_runner, fresh_runner, make_env

from vndub_app import appdirs, jobs, shell


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

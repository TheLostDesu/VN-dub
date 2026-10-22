"""Window shell: starts the local server and shows the SPA in a pywebview window (or the browser).

``uv run vndub-app`` (dev) and ``VNDub.exe`` (packaged, windowed: no console) both land in :func:`main`::

    vndub-app [--browser] [--port N] [--data-dir DIR] [--profile default|test|apptest]

* picks a free port on ``127.0.0.1``, the same one on every start for a data folder when it is free (``--port`` to force one) and serves ``vndub_app.server.create_app()``
  from a thread; the window opens at once with a tiny loading page and switches to the SPA when
  ``/api/health`` answers (the first screen is not blocked by the server import);
* ``vndub_app.system.set_window(window)`` gives the server the native file dialogs;
* closing the window stops the job runner (an active job is stopped cleanly - adapter processes, install
  scripts and ``llama-server`` are closed - and recorded as ``interrupted`` so the next start offers to continue)
  within one shared deadline (``JOBS_STOP_TIMEOUT_S``) however many jobs were active, then the process ends
  itself without finalization (:func:`_hard_exit`, ``TerminateProcess`` on itself, see :func:`main`): pythonnet / WebView2 threads
  can keep a normally returning process alive for good, and while ``VNDub.exe`` lives every child it started lives too. A watchdog
  thread (``EXIT_WATCHDOG_S``), started before the cleanup, forces the same exit if the cleanup itself hangs;
* if pywebview cannot start (no WebView2, no GUI) the app logs it and falls back to the default
  browser; in browser mode the process ends when the tab is closed (no open connection, no active job,
  idle for ``--idle-exit`` seconds) because there is no window to close;
* one instance per data folder: a second start opens the running instance in the browser and exits
  (``VNDUB_NO_BROWSER=1`` never opens a browser: tests, headless runs);
* console programs started by the pipeline (ffmpeg, nvidia-smi ...) get ``CREATE_NO_WINDOW``, and processes the app starts die with it
  (Windows Job Object, ``vndub_app.winproc``; whether it could be installed is written to ``app.log`` at start). The Job Object
  is only one of four layers that keep adapters and ``llama-server`` from outliving the app (they hold the GPU):
  at start the orphans of an earlier run are terminated (process ledger by image path, ``winproc.kill_orphans``) and a
  *reaper* is started (``VNDub.exe --reaper``: a ~20 MB helper (stdlib only: no server, no GUI) that waits for this process to end, however it ends, and
  sweeps once more; it logs to ``<data>/logs/reaper.log``); at the end of the app :func:`_final_sweep` terminates the
  descendants and the orphans before :func:`_hard_exit` ends the process; ``session.job_ctx`` sweeps again before every model job;
* when a job ends while the window is not in front, its taskbar button flashes (:func:`install_attention`);
* hidden ``--selfcheck [--selfcheck-out FILE]``: starts the server on a throw-away data folder, checks
  ``/api/health`` and the bundled resources/imports, prints one JSON line and exits 0 (ok) / 1 (verifies
  the packaged exe; a windowed exe has no console, so the line is also written to ``FILE``).

User-visible text of the loading/error page comes from the i18n tables (``shell.*`` keys); nothing is hard-coded.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
import webbrowser
import zlib
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("vndub_app.shell")

HOST = "127.0.0.1"
WINDOW_SIZE = (1280, 820)
WINDOW_MIN = (1000, 650)
HEALTH_TIMEOUT_S = 90.0
IDLE_EXIT_DEFAULT_S = 180.0
JOBS_STOP_TIMEOUT_S = 25.0  # one shared budget for all active jobs when the app closes
EXIT_WATCHDOG_S = JOBS_STOP_TIMEOUT_S + 20.0  # stop_jobs + server stop (10 s) + slack; then the exit is forced


# ----------------------------------------------------------------------------
# arguments, console, environment
# ----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vndub-app", description="VNDub desktop app: dub visual novels into Russian, locally.")
    p.add_argument("--browser", action="store_true", help="open the default browser instead of a window")
    p.add_argument("--port", type=int, default=0, help="port on 127.0.0.1 (default: a free one)")
    p.add_argument("--data-dir", default="", help="data folder for this run (default: the saved one, %%LOCALAPPDATA%%\\VNDub)")
    p.add_argument("--profile", choices=("default", "test", "apptest"), default="", help="pipeline profile (apptest = mocks, for UI work)")
    p.add_argument("--version", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--selfcheck", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--selfcheck-out", default="", help=argparse.SUPPRESS)
    p.add_argument("--idle-exit", type=float, default=IDLE_EXIT_DEFAULT_S, help=argparse.SUPPRESS)
    p.add_argument("--no-idle-exit", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--debug", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--reaper", action="store_true", help=argparse.SUPPRESS)  # the helper process: main() hands it to winproc.reaper_main before parsing
    return p


def _attach_parent_console() -> None:
    """A windowed exe started from a console: write to that console (``--selfcheck`` / ``--help`` output)."""
    if os.name != "nt" or not getattr(sys, "frozen", False):
        return
    try:
        import ctypes

        if ctypes.windll.kernel32.AttachConsole(-1):  # ATTACH_PARENT_PROCESS
            sys.stdout = open("CONOUT$", "w", encoding="utf-8")  # noqa: SIM115
            sys.stderr = open("CONOUT$", "w", encoding="utf-8")  # noqa: SIM115
    except Exception:
        pass


def _ensure_std_streams() -> None:
    for name in ("stdout", "stderr"):
        if getattr(sys, name, None) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))  # noqa: SIM115


def _emit(line: str) -> None:
    try:
        print(line, flush=True)
    except Exception:
        pass


def apply_args(args: argparse.Namespace) -> None:
    if args.data_dir:
        os.environ["VNDUB_DATA"] = str(Path(args.data_dir).expanduser().resolve())
    if args.profile:
        os.environ["VNDUB_APP_PROFILE"] = args.profile


def find_free_port(host: str = HOST) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def port_is_free(port: int, host: str = HOST) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except OSError:
            return False
        return True


def preferred_port(data_dir: Path, host: str = HOST) -> int:
    """A free port that is the same on every start for one data folder (else any free port).

    The SPA keeps small UI memories (e.g. "Diagnostics was shown") in ``localStorage``, which belongs to the
    origin ``http://127.0.0.1:<port>``: a new random port per start would forget them every time.
    """
    base = 49152 + zlib.crc32(str(data_dir).lower().encode("utf-8")) % 10000
    for i in range(8):
        if port_is_free(base + i, host):
            return base + i
    return find_free_port(host)


# ----------------------------------------------------------------------------
# i18n for the two inline pages
# ----------------------------------------------------------------------------


def _t(key: str, lang: str) -> str:
    """Text for ``key`` from the i18n tables (main table first, then the task parts); empty when unknown."""
    from vndub_app import appdirs

    base = appdirs.resource("app", "i18n")
    for f in (base / f"{lang}.json", base / "parts" / f"O.{lang}.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and isinstance(data.get(key), str):
            return data[key]
    return ""


def _lang() -> str:
    try:
        from vndub_app import appdirs

        return appdirs.load_settings().ui_lang
    except Exception:
        return "ru"


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_PAGE_CSS = (
    ":root{color-scheme:light dark;--bg:#f4f5f7;--fg:#1d2026;--mut:#6b7280;--ac:#3b6fe0}"
    "@media(prefers-color-scheme:dark){:root{--bg:#16181d;--fg:#e8eaee;--mut:#9aa1ad;--ac:#6d9cff}}"
    "html,body{height:100%;margin:0}body{background:var(--bg);color:var(--fg);font:16px system-ui,'Segoe UI',sans-serif;"
    "display:flex;align-items:center;justify-content:center;flex-direction:column;gap:18px;text-align:center;padding:24px;box-sizing:border-box}"
    ".brand{font-size:28px;font-weight:600;letter-spacing:.5px}.sub{color:var(--mut);max-width:34rem;line-height:1.5}"
    ".spin{width:34px;height:34px;border-radius:50%;border:3px solid var(--mut);border-top-color:var(--ac);animation:s 1s linear infinite}"
    "@keyframes s{to{transform:rotate(360deg)}}code{background:rgba(127,127,127,.2);padding:2px 6px;border-radius:4px;word-break:break-all}"
)


def loading_html(lang: str | None = None) -> str:
    lang = lang or _lang()
    text = _esc(_t("shell.loading", lang))
    return (
        f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><title>VNDub</title><style>{_PAGE_CSS}</style></head>'
        f'<body><div class="brand">VNDub</div><div class="spin"></div><div class="sub">{text}</div></body></html>'
    )


def error_html(detail: str, lang: str | None = None) -> str:
    lang = lang or _lang()
    return (
        f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><title>VNDub</title><style>{_PAGE_CSS}</style></head>'
        f'<body><div class="brand">VNDub</div><div class="sub"><b>{_esc(_t("shell.error.title", lang))}</b><br>'
        f'{_esc(_t("shell.error.hint", lang))}<br><br><code>{_esc(detail)}</code></div></body></html>'
    )


# ----------------------------------------------------------------------------
# activity tracking (browser mode has no window to close)
# ----------------------------------------------------------------------------


class ActivityTracker:
    def __init__(self) -> None:
        self.started = time.monotonic()
        self.last = self.started
        self.open_ws = 0
        self._lock = threading.Lock()

    def touch(self) -> None:
        self.last = time.monotonic()

    def ws_open(self) -> None:
        with self._lock:
            self.open_ws += 1
        self.touch()

    def ws_close(self) -> None:
        with self._lock:
            self.open_ws = max(0, self.open_ws - 1)
        self.touch()

    def idle_for(self) -> float:
        if self.open_ws > 0:
            return 0.0
        return time.monotonic() - self.last


class ActivityASGI:
    """Pure ASGI wrapper recording HTTP requests and open WebSockets; everything else passes through."""

    def __init__(self, app: Any, tracker: ActivityTracker) -> None:
        self.app = app
        self.tracker = tracker

    def __getattr__(self, name: str) -> Any:
        return getattr(self.app, name)

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        kind = scope.get("type")
        if kind == "http":
            self.tracker.touch()
            try:
                await self.app(scope, receive, send)
            finally:
                self.tracker.touch()
        elif kind == "websocket":
            self.tracker.ws_open()
            try:
                await self.app(scope, receive, send)
            finally:
                self.tracker.ws_close()
        else:
            await self.app(scope, receive, send)


# ----------------------------------------------------------------------------
# one instance per data folder
# ----------------------------------------------------------------------------


class InstanceLock:
    """Exclusive lock on ``<data>/app.lock`` (released by the OS when the process dies) + ``app.instance.json``."""

    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir / "app.lock"
        self.info = data_dir / "app.instance.json"
        self._fh: Any = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+b")  # noqa: SIM115
        try:
            if os.name == "nt":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        self._fh = fh
        return True

    def publish(self, url: str) -> None:
        try:
            self.info.write_text(json.dumps({"url": url, "pid": os.getpid(), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}), encoding="utf-8")
        except OSError:
            pass

    def running_url(self) -> str | None:
        try:
            return json.loads(self.info.read_text(encoding="utf-8")).get("url")
        except (OSError, ValueError):
            return None

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            self._fh.close()
        except OSError:
            pass
        self._fh = None
        try:
            self.info.unlink()
        except OSError:
            pass


# ----------------------------------------------------------------------------
# server
# ----------------------------------------------------------------------------


class ServerState:
    """The server thread's result: ``url`` once it accepts connections, ``error`` if it could not start."""

    def __init__(self) -> None:
        self.ready = threading.Event()
        self.server: Any = None
        self.url: str | None = None
        self.error: BaseException | None = None
        self.tracker = ActivityTracker()
        self.window_shown = False  # the SPA was loaded into the window: a later crash of the window is not "no GUI available"

    def stop(self) -> None:
        if self.server is not None:
            try:
                self.server.stop(10.0)
            except Exception as e:
                log.warning("server stop failed: %s", e)


def _log_job_object_status() -> None:
    """Say in ``app.log`` whether the "children die with the app" Job Object is in place (silent when it was never tried)."""
    from vndub_app import winproc

    st = winproc.job_object_status()
    if st["installed"]:
        log.info("Job Object installed: child processes end with the app")
    elif st["error"]:
        log.warning("Job Object NOT installed (%s): child processes may outlive a crashed or hung app", st["error"])


def start_server_async(
    port: int = 0,
    app_factory: Callable[[], Any] | None = None,
    tracker: ActivityTracker | None = None,
    on_started: Callable[[str], None] | None = None,
    startup: Callable[[], None] | None = None,
) -> ServerState:
    """Import and start the API server in a thread so the window is not blocked by the import.

    ``startup`` runs in that thread once logging exists and before the server accepts requests (``main`` passes
    :func:`_startup_housekeeping`): no job can start before the leftovers of an earlier run are gone.
    """
    state = ServerState()
    if tracker is not None:
        state.tracker = tracker

    def work() -> None:
        try:
            from vndub_app import appdirs, installer, server

            installer.configure_environment()
            server.setup_logging()
            _log_job_object_status()  # the Job Object is installed before there is a log to write to
            if startup is not None:
                try:
                    startup()
                except Exception as e:  # housekeeping is never a reason not to start
                    log.warning("startup housekeeping failed: %s", e)
            app = (app_factory or server.create_app)()
            srv = None
            for attempt in range(3):  # the chosen port may be taken between the check and the bind
                chosen = port or (preferred_port(appdirs.data_dir()) if attempt == 0 else find_free_port())
                try:
                    srv = server.BackgroundServer(HOST, chosen, app=ActivityASGI(app, state.tracker))
                    srv.start()
                    break
                except (OSError, RuntimeError, TimeoutError):
                    if port or attempt == 2:
                        raise
            assert srv is not None
            state.server, state.url = srv, srv.url
            if on_started is not None:
                on_started(srv.url)
        except BaseException as e:  # reported by the waiting side
            log.exception("server failed to start")
            state.error = e
        finally:
            state.ready.set()

    threading.Thread(target=work, name="vndub-server-start", daemon=True).start()
    return state


def http_get_json(url: str, timeout: float = 5.0) -> Any:
    import urllib.request  # not at the top: the reaper process starts through this module and has no use for http / ssl

    with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept": "application/json"}), timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def wait_for_health(state: ServerState, timeout: float = HEALTH_TIMEOUT_S) -> str | None:
    """Wait until the server answers ``/api/health``; returns the base URL or ``None`` (``state.error`` says why)."""
    import urllib.error

    deadline = time.monotonic() + timeout
    if not state.ready.wait(timeout):
        return None
    if state.error is not None or not state.url:
        return None
    while time.monotonic() < deadline:
        try:
            if http_get_json(state.url + "/api/health").get("ok"):
                return state.url
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(0.2)
    return None


def stop_jobs(timeout: float = JOBS_STOP_TIMEOUT_S) -> list[str]:
    """Stop the job runner for app exit; jobs that were active come back ``interrupted`` (offer to continue).

    ``timeout`` is one deadline for the whole call, not per job: every active job is cancelled at once (a cancel also
    kills the child processes the job watches), so the jobs wind down side by side and N stuck jobs must not cost
    N x ``timeout``. Jobs still active when the deadline passes are logged and left behind: the process is about to end.
    """
    from vndub_app import jobs

    r = jobs.runner()
    active = [j.id for j in r.list(active_only=True)]
    r.shutdown()
    deadline = time.monotonic() + timeout
    for jid in active:
        try:
            r.wait(jid, max(0.0, deadline - time.monotonic()))
        except Exception:
            pass
    stuck: list[str] = []
    for jid in active:
        try:
            rec = r.get(jid)
            if rec.state in jobs.ACTIVE:
                stuck.append(f"{rec.kind} {rec.id} ({rec.state})")
            elif rec.state == "cancelled":  # cancelled by the exit, not by the user
                rec.state = "interrupted"
        except Exception:
            pass
    if stuck:
        log.warning("%d of %d job(s) still active %.0f s after the app asked them to stop: %s", len(stuck), len(active), timeout, ", ".join(stuck))
    elif active:
        log.info("stopped %d active job(s)", len(active))
    r._save(force=True)  # noqa: SLF001
    return active


# ----------------------------------------------------------------------------
# processes that must not outlive the app
# ----------------------------------------------------------------------------

_ROOTS_SEEN: list[Path] = []  # the folders swept so far in this run: the envs folder may be changed in Settings meanwhile
_ROOTS_LOCK = threading.Lock()


def _remember_roots(roots: list[Path]) -> list[Path]:
    with _ROOTS_LOCK:
        for r in roots:
            if r not in _ROOTS_SEEN:
                _ROOTS_SEEN.append(r)
        return list(_ROOTS_SEEN)


def _current_roots() -> list[Path]:
    """Folders to sweep now: the installer's current ones (it is loaded once the server has started) plus the earlier ones."""
    mod = sys.modules.get("vndub_app.installer")
    if mod is not None:
        try:
            return _remember_roots(mod.process_roots())
        except Exception as e:
            log.debug("process_roots failed: %s", e)
    with _ROOTS_LOCK:
        return list(_ROOTS_SEEN)


def _startup_housekeeping() -> None:
    """Terminate what an earlier run left behind and start the reaper (server-start thread, after ``setup_logging``). Never raises.

    The orphans are found by image path (``winproc.kill_orphans``): a ``python.exe`` of an environment or a ``llama-server``
    whose parent is gone keeps its VRAM and its file locks, and the new run would load its models next to it.
    """
    if os.name != "nt":
        return
    try:
        from vndub_app import appdirs, installer, winproc

        roots = installer.process_roots()
        _remember_roots(roots)
        t0 = time.perf_counter()
        killed = winproc.kill_orphans(roots, reason="orphan left by a previous run")
        took = (time.perf_counter() - t0) * 1000
        if killed:
            log.info("startup: terminated %d orphan process(es) left by a previous run: %s", len(killed), winproc.describe(killed))
        else:
            log.debug("startup: no orphan processes under %s (%.0f ms)", ", ".join(str(r) for r in roots), took)
        r = winproc.start_reaper(roots, appdirs.logs_dir() / "reaper.log")
        if r["pid"] and r["breakaway"]:
            log.info("reaper started (pid %s): it outlives the app and sweeps once more when the app ends", r["pid"])
        elif r["pid"]:
            log.warning("reaper started (pid %s) but %s", r["pid"], r["error"] or "it runs inside the app's Job Object and ends together with the app")
        else:
            log.warning("reaper not started: %s", r["error"])
    except Exception as e:
        log.warning("startup housekeeping failed: %s", e)


_SWEPT_AT: float | None = None
SWEEP_ONCE_S = 5.0  # a second _final_sweep within this time (main's, then _hard_exit's) is skipped


def _log_nonblocking(level: int, msg: str, *args: Any) -> None:
    """Log from a helper thread and wait at most 2 s: at the very end the logging lock may be held by the stuck cleanup."""
    t = threading.Thread(target=log.log, args=(level, msg, *args), daemon=True)
    t.start()
    t.join(2.0)


def _final_sweep() -> None:
    """The last act before the process ends, Job Object or not (the Job Object stays the net for a crash): terminate the
    descendants of this process, then every orphan under the app's folders. Never raises; skipped when it just ran."""
    global _SWEPT_AT
    if os.name != "nt":
        return
    now = time.monotonic()
    if _SWEPT_AT is not None and now - _SWEPT_AT < SWEEP_ONCE_S:
        return
    _SWEPT_AT = now
    try:
        from vndub_app import winproc

        t0 = time.perf_counter()
        children = winproc.kill_descendants()
        orphans = winproc.kill_orphans(_current_roots(), reason="orphan at the end of the app")
        _log_nonblocking(
            logging.INFO, "exit sweep: terminated %d child process(es) and %d orphan(s) in %.0f ms%s",
            children, len(orphans), (time.perf_counter() - t0) * 1000, (": " + winproc.describe(orphans)) if orphans else "",
        )
    except Exception as e:
        _log_nonblocking(logging.WARNING, "exit sweep failed: %s", e)


# ----------------------------------------------------------------------------
# ending the process
# ----------------------------------------------------------------------------


def _hard_exit(code: int) -> None:
    """End the process now, without interpreter finalization (tests replace this).

    Returning from ``main`` and ``sys.exit`` run finalization, which waits for non-daemon threads and for pythonnet's
    CLR shutdown; a WebView2 / .NET thread that never ends keeps the process (and so its Job Object handle and every
    child) alive. ``winproc.end_process`` skips all of that, including the DLL detach step that ``os._exit`` still goes
    through. Before it, :func:`_final_sweep` makes sure nothing the app started is left, Job Object or not: the Job Object
    only covers a process that really ends and only when it could be installed, and it cannot reach an orphan whose
    parent died earlier.
    """
    from vndub_app import winproc

    _final_sweep()
    winproc.end_process(code)


class ExitWatchdog(threading.Thread):
    """Forces the exit if the cleanup in :func:`main` hangs (a daemon: it never keeps the app alive on its own).

    It is started at the *beginning* of the cleanup, while the interpreter is fully alive: a daemon thread that is only
    created at the very end can no longer run Python code once finalization has begun, which is how the old
    "sleep, then ``os._exit``" thread silently never fired. :meth:`disarm` exists for tests (the real app never
    outlives the watchdog: the direct exit (``winproc.end_process``) after the cleanup ends the process and the thread with it).
    """

    def __init__(self, code: int, delay: float) -> None:
        super().__init__(name="vndub-exit-watchdog", daemon=True)
        self.code = code
        self.delay = delay
        self._disarmed = threading.Event()

    def disarm(self) -> None:
        self._disarmed.set()

    def run(self) -> None:
        if self._disarmed.wait(self.delay):
            return
        # the log call may block on a lock the stuck cleanup holds: it gets a couple of seconds, the exit does not wait for it
        msg = threading.Thread(target=log.error, args=("exit cleanup still running after %.0f s: forcing the exit", self.delay), daemon=True)
        msg.start()
        msg.join(2.0)
        _hard_exit(self.code)


def _start_exit_watchdog(code: int, delay: float | None = None) -> ExitWatchdog:
    wd = ExitWatchdog(code, EXIT_WATCHDOG_S if delay is None else delay)
    wd.start()
    return wd


_CLOSE_WATCHDOG: ExitWatchdog | None = None  # armed by the window's closed event (run_window), disarmed by main()


def _arm_close_watchdog() -> None:
    """The window was closed: should ``webview.start()`` fail to return (a stuck WebView2 / .NET teardown), the process
    still ends after ``EXIT_WATCHDOG_S``, children included. ``main()`` disarms it when it takes over, so the two never race."""
    global _CLOSE_WATCHDOG
    if _CLOSE_WATCHDOG is None:
        _CLOSE_WATCHDOG = _start_exit_watchdog(0, EXIT_WATCHDOG_S)


def _cleanup_for_exit(state: ServerState, lock: InstanceLock, sweep: bool = False) -> None:
    """The ordered shutdown of :func:`main`; a failing step is logged and never skips the next one.

    ``sweep`` (a real exit, not a test) adds :func:`_final_sweep` right after the jobs were stopped: whatever they left running
    is terminated while ``app.log`` still exists (stopping the server closes it) and before the server's 10 s stop can keep it alive.
    """
    steps = [("stopping jobs", stop_jobs)]
    if sweep:
        steps.append(("terminating leftover processes", _final_sweep))
    steps += [("stopping the server", state.stop), ("releasing the instance lock", lock.release)]
    for what, step in steps:
        try:
            step()
        except Exception as e:
            log.warning("%s failed: %s", what, e)


# ----------------------------------------------------------------------------
# selfcheck
# ----------------------------------------------------------------------------

REQUIRED_RESOURCES = (
    ("configs", "pipeline.yaml"),
    ("configs", "tools.yaml"),
    ("configs", "interjections.yaml"),
    ("app", "tools_manifest.json"),
    ("scripts", "make_envs.ps1"),
    ("scripts", "make_envs.sh"),
    ("envs", "main.txt"),
)
UV_RESOURCES = (("bin", "uv.exe"), ("uv.exe",))


def _check_resources() -> dict[str, Any]:
    from vndub_app import appdirs

    from vndub.backends.runner import adapters_root

    missing = [str(Path(*p)) for p in REQUIRED_RESOURCES if not appdirs.resource(*p).exists()]
    rpc = adapters_root() / "adapters" / "common" / "rpc.py"  # <bundle>/backend/adapters when frozen
    if not rpc.is_file():
        missing.append(str(rpc.relative_to(appdirs.resource_root())) if rpc.is_relative_to(appdirs.resource_root()) else str(rpc))
    out: dict[str, Any] = {"missing": missing, "root": str(appdirs.resource_root())}
    if getattr(sys, "frozen", False) and os.name == "nt":
        out["uv"] = any(appdirs.resource(*p).is_file() for p in UV_RESOURCES)
        if not out["uv"]:
            missing.append("bin/uv.exe")
        from vndub_app import installer

        # tools that packaging/build_app.ps1 puts next to VNDub.exe (manifest "bundled")
        out["bundled_tools"] = {}
        for e in installer.entries():
            if e.get("bundled"):
                have = installer.bundled_paths(e)
                out["bundled_tools"][e["name"]] = len(have) == len(e["bundled"])
                missing += [rel for key, rel in e["bundled"].items() if key not in have]
    out["web_dist"] = (appdirs.web_dist() / "index.html").exists()
    if not getattr(sys, "frozen", False):
        from vndub_app import webbuild

        out["web_fresh"] = webbuild.status()["fresh"]
    out["docs"] = appdirs.resource("docs").is_dir()
    return out


def _check_imports() -> dict[str, Any]:
    """Everything the pipeline loads by name (importlib): backends, engines, stages."""
    out: dict[str, Any] = {}
    errors: list[str] = []
    try:
        from vndub.backends import registry

        out["backends"] = len(registry.all_specs())
    except Exception as e:
        errors.append(f"backends: {type(e).__name__}: {e}")
    try:
        from vndub.engines import all_engines

        out["engines"] = len(all_engines())
    except Exception as e:
        errors.append(f"engines: {type(e).__name__}: {e}")
    try:
        from vndub.stages.common import STAGES

        bad = []
        for st in STAGES:
            try:
                importlib.import_module(f"vndub.stages.{st}")
            except Exception as e:
                bad.append(f"{st}: {type(e).__name__}: {e}")
        out["stages"] = len(STAGES) - len(bad)
        errors += bad
    except Exception as e:
        errors.append(f"stages: {type(e).__name__}: {e}")
    for mod in ("numpy", "scipy", "soundfile", "librosa", "yaml", "pydantic", "orjson", "rapidfuzz", "jinja2", "tqdm", "python_multipart", "uvicorn", "fastapi"):
        try:
            importlib.import_module(mod)
        except Exception as e:
            errors.append(f"{mod}: {type(e).__name__}: {e}")
    try:  # the lazily imported submodules the model downloads use (httpx2 in huggingface_hub 2.x)
        from huggingface_hub import HfApi, hf_hub_download  # noqa: F401
    except Exception as e:
        errors.append(f"huggingface_hub: {type(e).__name__}: {e}")
    out["errors"] = errors
    return out


def selfcheck(args: argparse.Namespace) -> int:
    """Start the server on a throw-away data folder, check it and the bundle, print one JSON line."""
    t0 = time.monotonic()
    result: dict[str, Any] = {"selfcheck": True, "ok": False, "frozen": bool(getattr(sys, "frozen", False)), "python": sys.version.split()[0]}
    own_tmp = not args.data_dir
    tmp = tempfile.mkdtemp(prefix="vndub-selfcheck-") if own_tmp else ""
    saved = {k: os.environ.get(k) for k in ("VNDUB_DATA", "VNDUB_APP_PROFILE", "VNDUB_ENVS_DIR")}
    if own_tmp:
        os.environ["VNDUB_DATA"] = tmp
    os.environ.setdefault("VNDUB_APP_PROFILE", "apptest")
    state: ServerState | None = None
    problems: list[str] = []
    warnings: list[str] = []
    try:
        from vndub_app import __version__

        result["version"] = __version__
        result["resources"] = _check_resources()
        if result["resources"]["missing"]:
            problems.append("resources missing: " + ", ".join(result["resources"]["missing"]))
        imp = _check_imports()
        result["imports"] = {k: v for k, v in imp.items() if k != "errors"}
        problems += imp["errors"]
        state = start_server_async(args.port)
        url = wait_for_health(state, 60.0)
        if url is None:
            problems.append(f"server did not start: {state.error!r}" if state.error else "server did not answer /api/health")
        else:
            result["port"] = int(url.rsplit(":", 1)[1])
            result["health"] = http_get_json(url + "/api/health")
            if not result["health"].get("ok"):
                problems.append("health.ok is false")
            for path in ("/api/tools", "/api/envs", "/api/models"):
                try:
                    http_get_json(url + path, timeout=30.0)
                except Exception as e:
                    problems.append(f"GET {path} failed: {type(e).__name__}: {e}")
            try:
                import urllib.request

                with urllib.request.urlopen(url + "/", timeout=10) as r:
                    result["index_status"] = r.status
            except Exception as e:
                problems.append(f"GET / failed: {type(e).__name__}: {e}")
        if importlib.util.find_spec("websockets") is None and importlib.util.find_spec("wsproto") is None:
            warnings.append("no websocket library (websockets / wsproto): WS /api/progress cannot be served")
    except BaseException as e:
        problems.append(f"{type(e).__name__}: {e}")
    finally:
        if state is not None:
            state.stop()
        try:
            from vndub_app import jobs, server

            jobs.reset_runner(None)
            server.close_logging()
        except Exception:
            pass
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if own_tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    result["ok"] = not problems
    result["problems"] = problems
    result["warnings"] = warnings
    result["elapsed_s"] = round(time.monotonic() - t0, 2)
    line = json.dumps(result, ensure_ascii=False)
    _emit(line)
    if args.selfcheck_out:
        try:
            Path(args.selfcheck_out).write_text(line + "\n", encoding="utf-8")
        except OSError as e:
            _emit(f"cannot write {args.selfcheck_out}: {e}")
    return 0 if result["ok"] else 1


# ----------------------------------------------------------------------------
# window / browser
# ----------------------------------------------------------------------------


def run_browser(state: ServerState, idle_exit_s: float | None, open_browser: bool = True) -> int:
    """Open the default browser and serve until it goes away (or Ctrl+C)."""
    url = wait_for_health(state)
    if url is None:
        log.error("server did not start: %r", state.error)
        _emit(f"VNDub: server did not start: {state.error!r}")
        return 1
    if open_browser and not os.environ.get("VNDUB_NO_BROWSER"):
        try:
            webbrowser.open(url)
        except Exception as e:
            log.warning("cannot open the browser: %s", e)
    _emit(f"VNDub is running at {url}")
    from vndub_app import jobs

    try:
        while True:
            time.sleep(1.0)
            if idle_exit_s and state.tracker.idle_for() > idle_exit_s and jobs.runner().active() is None:
                log.info("browser closed (idle %.0f s): exiting", state.tracker.idle_for())
                break
    except KeyboardInterrupt:
        pass
    return 0


ATTENTION_STATES = ("done", "failed")  # job states that flash the taskbar button


def _native_handle(window: Any) -> int | None:
    """HWND of a pywebview window: with the EdgeChromium backend ``window.native`` is the WinForms Form and ``Handle`` its window handle."""
    try:
        h = window.native.Handle
        to_int = getattr(h, "ToInt64", None)  # System.IntPtr
        return (int(to_int()) if to_int is not None else int(h)) or None
    except Exception:
        return None


def install_attention(window: Any, runner: Any = None, title: str = "VNDub") -> Callable[[], None]:
    """Flash the window's taskbar button when a job ends (``done`` / ``failed``) while the window is not in front.

    Subscribes to the job runner (``{"type": "job", "job": {...}}`` events); each job flashes at most once. The window
    handle is looked up on the first event (``window.native.Handle``, else a visible window of this process titled
    ``title``) and without one nothing happens. Returns the unsubscribe function. Never raises into the runner.
    """
    from vndub_app import jobs, winproc

    flashed: set[str] = set()
    cached: list[int | None] = [None]

    def hwnd() -> int | None:
        if not cached[0]:
            cached[0] = _native_handle(window) or winproc.find_own_window(title)
        return cached[0]

    def on_event(ev: dict[str, Any]) -> None:
        try:
            if ev.get("type") != "job":
                return
            job = ev.get("job") or {}
            jid = str(job.get("id") or "")
            if not jid or job.get("state") not in ATTENTION_STATES or jid in flashed:
                return
            flashed.add(jid)
            h = hwnd()
            if h and not winproc.is_foreground(h):
                winproc.flash_window(h)
        except Exception as e:  # a window problem must never reach the job runner
            log.debug("attention failed: %s", e)

    return (runner or jobs.runner()).subscribe(on_event)


def run_window(state: ServerState, debug: bool = False) -> int:
    """pywebview window: loading page now, the SPA when the server answers. Raises if no GUI can start."""
    import webview  # type: ignore[import-not-found]

    from vndub_app import system

    lang = _lang()
    window = webview.create_window(
        "VNDub", html=loading_html(lang), width=WINDOW_SIZE[0], height=WINDOW_SIZE[1], min_size=WINDOW_MIN, text_select=True,
    )
    system.set_window(window)
    try:  # downloads (e.g. the problem report) and external links in the system browser
        webview.settings["ALLOW_DOWNLOADS"] = True
        webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True
    except Exception:
        pass

    def boot() -> None:
        url = wait_for_health(state)
        if url:
            window.load_url(url)
            state.window_shown = True
            try:
                install_attention(window)
            except Exception as e:
                log.debug("no taskbar attention: %s", e)
        else:
            detail = repr(state.error) if state.error else "server did not answer /api/health"
            log.error("window boot failed: %s", detail)
            window.load_html(error_html(detail, lang))

    try:
        window.events.closed += lambda *a: _arm_close_watchdog()  # see _arm_close_watchdog
    except Exception as e:  # an older pywebview without the event: the regular exit path still works
        log.debug("no window.events.closed (%s)", e)

    gui = "edgechromium" if sys.platform.startswith("win") else None  # never the old MSHTML fallback
    from vndub_app import appdirs

    try:  # keep the web profile (localStorage of the stable origin) between starts
        webview.start(boot, gui=gui, debug=debug, private_mode=False, storage_path=str(appdirs.sub("webview")))
    except TypeError:  # an older pywebview without these options
        webview.start(boot, gui=gui, debug=debug)
    return 0


def main(argv: list[str] | None = None, *, hard_exit: bool = True) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if "--reaper" in raw:  # the helper process (winproc.start_reaper): no parsing, no instance lock, no server, nothing heavy imported
        from vndub_app import winproc

        return winproc.reaper_main(raw)
    args = build_parser().parse_args(argv)
    if args.selfcheck or args.version:
        _attach_parent_console()
    _ensure_std_streams()
    if args.version:
        from vndub_app import __version__

        _emit(__version__)
        return 0
    apply_args(args)
    if args.selfcheck:
        return selfcheck(args)
    from vndub_app import winproc

    winproc.hide_child_consoles()  # a windowed exe must not make a console window flash per ffmpeg call
    winproc.kill_children_on_exit()  # adapters / llama-server must not outlive the app and keep the GPU

    from vndub_app import appdirs

    lock = InstanceLock(appdirs.data_dir())
    if not lock.acquire():
        url = lock.running_url()
        if url and not os.environ.get("VNDUB_NO_BROWSER"):
            webbrowser.open(url)
        return 0

    from vndub_app import webbuild

    webbuild.ensure_built()  # from source: rebuild an interface build older than its sources (needs Node.js)
    state = start_server_async(args.port, on_started=lock.publish, startup=_startup_housekeeping if hard_exit else None)
    code = 0
    try:
        if not args.browser:
            try:
                code = run_window(state, debug=args.debug)
            except Exception as e:
                if state.window_shown:  # the window was up and then broke (pythonnet / WebView2 teardown): the user sees an app that is
                    # gone, so it must be gone; carrying on headless in a browser would keep jobs, models and VRAM for minutes
                    log.exception("the window stopped unexpectedly (%s: %s): ending the app", type(e).__name__, e)
                    code = 1
                else:  # no WebView2 / no GUI toolkit: keep the user going in the browser
                    log.warning("window could not start (%s: %s): falling back to the browser", type(e).__name__, e)
                    args.browser = True
        if args.browser:
            code = run_browser(state, None if args.no_idle_exit else args.idle_exit)
    except BaseException as e:  # the window / browser loop died: still clean up, and end with a failing code
        code = 130 if isinstance(e, KeyboardInterrupt) else 1
        if not isinstance(e, KeyboardInterrupt):
            log.exception("the app stopped unexpectedly")
        if not hard_exit:
            raise
    finally:
        # Every way out of the window / browser loop lands here. With hard_exit the process must really end afterwards
        # (see _hard_exit), so the watchdog starts first, while the interpreter is alive, and covers all the cleanup.
        if hard_exit:
            _start_exit_watchdog(code)
        if _CLOSE_WATCHDOG is not None:  # main() is in charge from here on: its own watchdog covers the cleanup
            _CLOSE_WATCHDOG.disarm()
        try:
            _cleanup_for_exit(state, lock, sweep=hard_exit)
        finally:
            if hard_exit:  # pythonnet / GUI threads must not keep a closed app (and its children) alive
                log.info("cleanup done: ending the process now with code %s", code)
            try:
                from vndub_app import applog, server

                applog.log_shutdown(code)  # "exiting with code N after 1m05s": the last line of app.log
                logging.shutdown()  # flushes app.log, the "exiting" line included
                server.close_logging()
            except Exception:
                pass
            if hard_exit:
                for stream in (sys.stdout, sys.stderr):  # TerminateProcess does not flush them (a stuck flush is the watchdog's job)
                    try:
                        stream.flush()
                    except Exception:
                        pass
                _hard_exit(code)
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

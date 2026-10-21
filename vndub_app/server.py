"""The FastAPI application of the desktop app.

``create_app()`` builds the app (cheap: heavy pipeline modules are imported lazily, and warmed
up in a background thread after startup), ``run(host, port)`` serves it with uvicorn, and
``start_background(...)`` serves it from a thread (used by tests and the Playwright walk).

* every JSON error has the wire format of ``errors.py`` (``{"error": {code, params, action, details}}``);
* routers: ``vndub_app.routes.*`` (this task) and, via ``importlib``, the doctor / installer /
  models / patch-install modules of task O (a missing module is logged and skipped);
* the SPA from ``appdirs.web_dist()`` with an index.html fallback (a plain page when it is not built);
* ``WS /api/progress``: ``hello``, ``job`` / ``log`` events of the job runner, ``gpu`` every 2 s while a GPU job runs
  (``GET /api/progress/sse`` carries the same events as Server-Sent Events: the fallback when uvicorn has no
  WebSocket library);
* security: only loopback ``Host`` headers are served (DNS rebinding) and unsafe requests must not
  carry a foreign ``Origin``.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import logging
import os
import socket
import sys
import threading
import time
from contextlib import aclosing, asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, AsyncIterator
from urllib.parse import urlsplit

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from vndub_app import __version__, appdirs, errors, gpuinfo, jobs, session
from vndub_app.errors import AppError

log = logging.getLogger("vndub_app.server")

#: routers of task O (each module exposes ``router``)
EXTERNAL_MODULES = ("vndub_app.doctor", "vndub_app.installer", "vndub_app.models_api", "vndub_app.hf_custom", "vndub_app.patch_install")

ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1", "testserver"}
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
GPU_EVENT_INTERVAL_S = 2.0
_FILE_HANDLER_MARK = "_vndub_app_log_file"

PLAIN_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>VNDub</title>
<style>body{font-family:system-ui,sans-serif;max-width:40rem;margin:4rem auto;padding:0 1rem;color:#222}
code{background:#eee;padding:.1rem .3rem;border-radius:3px}</style></head><body>
<h1>VNDub</h1><p>The interface is not built yet (<code>vndub_app/web/dist</code> is missing).</p>
<p>The API is running: <a href="/api/health">/api/health</a>.</p>
<p>Build it with <code>npm install &amp;&amp; npm run build</code> in <code>vndub_app/web</code>.</p></body></html>"""


# ----------------------------------------------------------------------------
# logging
# ----------------------------------------------------------------------------


def ensure_std_streams() -> None:
    """``pythonw`` / a windowed exe has no stdout/stderr; logging and rich would crash on ``None``."""
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))  # noqa: SIM115


def setup_logging(level: int = logging.INFO) -> Path:
    """Rotating ``<data>/logs/app.log`` for the app, the pipeline loggers and uvicorn (idempotent)."""
    path = appdirs.logs_dir() / "app.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    for name in ("vndub_app", "vndub", "uvicorn.error"):
        lg = logging.getLogger(name)
        have = False
        for h in list(lg.handlers):
            if getattr(h, _FILE_HANDLER_MARK, False):
                if Path(getattr(h, "baseFilename", "")) == path:
                    have = True
                else:
                    lg.removeHandler(h)
                    h.close()
        if not have:
            fh = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
            fh.setFormatter(fmt)
            fh.setLevel(level)
            setattr(fh, _FILE_HANDLER_MARK, True)
            lg.addHandler(fh)
        if lg.level == logging.NOTSET or lg.level > level:
            lg.setLevel(level)
    return path


def close_logging() -> None:
    """Close the app.log handlers (tests on Windows must release the file before the folder is removed)."""
    for name in ("vndub_app", "vndub", "uvicorn.error"):
        lg = logging.getLogger(name)
        for h in list(lg.handlers):
            if getattr(h, _FILE_HANDLER_MARK, False):
                lg.removeHandler(h)
                h.close()


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def _host_ok(host_header: str | None) -> bool:
    if not host_header:
        return True
    try:
        host = urlsplit(f"//{host_header}").hostname
    except ValueError:
        return False
    return (host or "").lower() in ALLOWED_HOSTS


def _origin_ok(origin: str | None) -> bool:
    if not origin or origin == "null":
        return True  # same-origin fetches from file:// / pywebview carry none or "null"
    try:
        host = urlsplit(origin).hostname
    except ValueError:
        return False
    return (host or "").lower() in ALLOWED_HOSTS


def _forbidden() -> JSONResponse:
    err = AppError("permission_denied", params={"path": "origin"}, status=403)
    return JSONResponse(status_code=403, content={"error": err.to_dict()})


def find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _include_routers(app: FastAPI) -> list[str]:
    from vndub_app import routes

    loaded: list[str] = []
    for m in routes.MODULES:
        mod = importlib.import_module(f"vndub_app.routes.{m}")
        app.include_router(mod.router)
        loaded.append(mod.__name__)
    for name in EXTERNAL_MODULES:
        try:
            mod = importlib.import_module(name)
            app.include_router(mod.router)
            loaded.append(name)
        except ModuleNotFoundError as e:
            if e.name == name:
                log.warning("router module %s is not available (skipped)", name)
            else:
                log.warning("router module %s skipped: missing dependency %s", name, e.name)
        except Exception:  # a broken optional module must not take the app down
            log.exception("router module %s failed to import (skipped)", name)
    return loaded


def _spa_response(rel: str) -> Any:
    dist = appdirs.web_dist()
    index = dist / "index.html"
    if not index.exists():
        return HTMLResponse(PLAIN_PAGE)
    if rel:
        try:
            cand = (dist / rel).resolve()
            cand.relative_to(dist.resolve())
            if cand.is_file():
                immutable = "assets" in cand.relative_to(dist.resolve()).parts
                return FileResponse(cand, headers={"Cache-Control": "public, max-age=31536000, immutable" if immutable else "no-cache"})
        except (OSError, ValueError):
            pass
    return FileResponse(index, media_type="text/html", headers={"Cache-Control": "no-cache"})


# ----------------------------------------------------------------------------
# app
# ----------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):
    setup_logging()
    jobs.runner()  # loads jobs.json: jobs that were active at the last exit come back as "interrupted"
    threading.Thread(target=session.warm_up, name="vndub-warmup", daemon=True).start()
    if importlib.util.find_spec("websockets") is None and importlib.util.find_spec("wsproto") is None:
        log.warning("neither 'websockets' nor 'wsproto' is installed: uvicorn cannot serve WS /api/progress (install 'websockets')")
    try:
        yield
    finally:
        r = jobs._RUNNER  # noqa: SLF001  (persist the last state; active jobs reload as interrupted)
        if r is not None:
            r._save(force=True)  # noqa: SLF001
        close_logging()


def _sync_custom_models() -> None:
    """``VNDUB_CUSTOM_MODELS`` -> ``<data>/custom_models.json``: models added by link become backends the pipeline can use."""
    try:
        from vndub_app import hf_custom

        hf_custom.sync_env()
    except Exception:  # never keep the app from starting
        log.exception("cannot point the backend registry at custom_models.json")


def create_app() -> FastAPI:
    app = FastAPI(title="VNDub", version=__version__, docs_url=None, redoc_url=None, openapi_url="/api/openapi.json", lifespan=_lifespan)
    errors.install_handlers(app)
    _sync_custom_models()

    @app.middleware("http")
    async def _guard(request: Request, call_next):  # type: ignore[no-untyped-def]
        if not _host_ok(request.headers.get("host")):
            return _forbidden()
        if request.method not in SAFE_METHODS and not _origin_ok(request.headers.get("origin")):
            return _forbidden()
        return await call_next(request)

    loaded = _include_routers(app)
    app.state.routers = loaded

    @app.websocket("/api/progress")
    async def progress(ws: WebSocket) -> None:
        if not _origin_ok(ws.headers.get("origin")) or not _host_ok(ws.headers.get("host")):
            await ws.close(code=1008)
            return
        await _progress_loop(ws)

    @app.get("/api/progress/sse", include_in_schema=False)
    async def progress_sse(request: Request) -> StreamingResponse:
        return await _progress_sse(request)

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str) -> Any:
        if full_path == "api" or full_path.startswith("api/"):
            raise AppError("file_not_found", params={"path": "/" + full_path}, status=404)
        return _spa_response(full_path)

    return app


async def _drain(ws: WebSocket, stop: asyncio.Event) -> None:
    """Read (and ignore) client messages; sets ``stop`` when the client goes away."""
    try:
        while True:
            await ws.receive_text()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        stop.set()


async def _event_source(stop: asyncio.Event, keepalive_s: float | None = None) -> AsyncIterator[dict[str, Any] | None]:
    """The events of ``WS /api/progress`` (and its SSE twin): ``hello``, the job runner's ``job`` / ``log`` events
    and a ``gpu`` event every 2 s while a GPU job runs. Yields ``None`` as a keep-alive tick (SSE only)."""
    loop = asyncio.get_running_loop()
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=5000)

    def put(ev: dict[str, Any]) -> None:
        try:
            q.put_nowait(ev)
        except asyncio.QueueFull:
            pass  # a stalled client must not grow memory without bound

    def push(ev: dict[str, Any]) -> None:  # called from job worker threads
        try:
            loop.call_soon_threadsafe(put, ev)
        except RuntimeError:  # loop closed
            pass

    runner = jobs.runner()
    unsub = runner.subscribe(push)
    stopper = asyncio.ensure_future(stop.wait())
    try:
        yield {"type": "hello", "jobs": [j.info() for j in runner.list(active_only=True)]}
        next_gpu = time.monotonic() + GPU_EVENT_INTERVAL_S
        idle = 0.0
        while not stop.is_set():
            getter = asyncio.ensure_future(q.get())
            done, _ = await asyncio.wait({getter, stopper}, timeout=GPU_EVENT_INTERVAL_S, return_when=asyncio.FIRST_COMPLETED)
            if getter in done:
                idle = 0.0
                yield getter.result()
            else:
                getter.cancel()
            if stop.is_set():
                break
            now = time.monotonic()
            if now >= next_gpu:
                next_gpu = now + GPU_EVENT_INTERVAL_S
                if any(j.gpu and j.state == "running" for j in runner.list(active_only=True)):
                    g = await loop.run_in_executor(None, gpuinfo.query)
                    if g:
                        idle = 0.0
                        yield {"type": "gpu", "used_gb": g["used_gb"], "total_gb": g["total_gb"], "name": g["name"]}
            if keepalive_s is not None and getter not in done:
                idle += GPU_EVENT_INTERVAL_S
                if idle >= keepalive_s:
                    idle = 0.0
                    yield None
    finally:
        unsub()
        stopper.cancel()


async def _progress_loop(ws: WebSocket) -> None:
    await ws.accept()
    stop = asyncio.Event()
    recv = asyncio.create_task(_drain(ws, stop))
    try:
        async with aclosing(_event_source(stop)) as events:
            async for ev in events:
                if ev is not None:
                    await ws.send_json(ev)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        stop.set()
        recv.cancel()


def _sse(ev: dict[str, Any] | None) -> bytes:
    """One Server-Sent Events frame (``None`` = keep-alive comment)."""
    import json

    if ev is None:
        return b": keepalive" + bytes([10, 10])
    return b"data: " + json.dumps(ev, ensure_ascii=False).encode("utf-8") + bytes([10, 10])


async def _progress_sse(request: Request) -> StreamingResponse:
    """Server-Sent Events twin of ``WS /api/progress`` for environments where uvicorn has no WebSocket library."""
    stop = asyncio.Event()

    async def body() -> AsyncIterator[bytes]:
        async with aclosing(_event_source(stop, keepalive_s=15.0)) as events:  # runs the unsubscribe on disconnect
            async for ev in events:
                yield _sse(ev)

    return StreamingResponse(body(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ----------------------------------------------------------------------------
# serving
# ----------------------------------------------------------------------------


def run(host: str = "127.0.0.1", port: int = 0, *, log_level: str = "warning") -> None:
    """Serve the app (blocking). ``port=0`` picks a free port."""
    import uvicorn

    ensure_std_streams()
    port = port or find_free_port(host)
    uvicorn.run(create_app(), host=host, port=port, log_level=log_level, access_log=False)


class BackgroundServer:
    """uvicorn in a thread: ``start()`` returns once the socket is accepting; ``stop()`` shuts it down."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0, app: FastAPI | None = None, log_level: str = "warning") -> None:
        import uvicorn

        ensure_std_streams()
        self.host = host
        self.port = port or find_free_port(host)
        self.config = uvicorn.Config(app or create_app(), host=host, port=self.port, log_level=log_level, access_log=False)
        self.server = uvicorn.Server(self.config)
        self.thread = threading.Thread(target=self.server.run, name="vndub-uvicorn", daemon=True)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self, timeout: float = 20.0) -> "BackgroundServer":
        self.thread.start()
        t0 = time.monotonic()
        while not self.server.started:
            if not self.thread.is_alive():
                raise RuntimeError("uvicorn exited during startup")
            if time.monotonic() - t0 > timeout:
                raise TimeoutError("uvicorn did not start")
            time.sleep(0.05)
        return self

    def stop(self, timeout: float = 10.0) -> None:
        self.server.should_exit = True
        self.thread.join(timeout)

    def __enter__(self) -> "BackgroundServer":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


def start_background(host: str = "127.0.0.1", port: int = 0, **kw: Any) -> BackgroundServer:
    return BackgroundServer(host, port, **kw).start()


def serve_review(ctx: Any, host: str = "127.0.0.1", port: int = 0, open_browser: bool = False) -> None:
    """``vndub review --app``: open the app's review screen for a CLI project (blocking).

    The app project of the game folder is bound to the CLI's own work/out dirs
    (``project_meta.json: work_dir/out_dir``), made current, and the wizard state is set to the
    review step so the SPA opens on it.
    """
    import webbrowser

    ensure_std_streams()
    h = appdirs.create_project(Path(ctx.project_dir))
    meta = h.meta()
    meta.update(work_dir=str(ctx.paths.work), out_dir=str(ctx.paths.out))
    appdirs.write_json_atomic(h.meta_file, meta)
    st = appdirs.read_app_state(h)
    st["step"] = "review"
    appdirs.write_app_state(h, st)
    appdirs.write_project_config(h)
    session.set_current(h.slug)
    srv = BackgroundServer(host, port).start()
    print(f"VNDub review: {srv.url}  (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open_new_tab(srv.url)
    try:
        while srv.thread.is_alive():
            srv.thread.join(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        srv.stop()


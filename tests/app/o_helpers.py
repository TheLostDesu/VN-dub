"""Helpers of the task-O tests (doctor, installer, models, patch install, shell): no dependency on conftest.

``make_env`` points the app at a throw-away data folder, ``RangeServer`` is a tiny HTTP server with
``Range`` support (and switches to simulate a dropped connection / a server that ignores ``Range``),
``zip_bytes`` builds archives in memory, ``client_for`` mounts the routers under test.
"""

from __future__ import annotations

import hashlib
import io
import json
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import pytest

REPO = Path(__file__).resolve().parents[2]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_env(mp: pytest.MonkeyPatch, root: Path, profile: str = "default") -> Path:
    """Throw-away data folder; the real user folders and env vars are never touched."""
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    mp.setenv("VNDUB_DATA", str(data))
    mp.setenv("VNDUB_APP_PROFILE", profile)
    mp.setenv("VNDUB_NO_SHUTDOWN", "1")
    mp.setenv("VNDUB_NO_BROWSER", "1")
    mp.setenv("APPDATA", str(root / "appdata"))
    mp.setenv("LOCALAPPDATA", str(root / "localappdata"))
    mp.setenv("XDG_CONFIG_HOME", str(root / "xdg_config"))
    mp.setenv("XDG_DATA_HOME", str(root / "xdg_data"))
    mp.setenv("VNDUB_ENVS_DIR", str(root / "envs"))
    mp.delenv("VNDUB_MODELS_DIR", raising=False)
    return data


def fresh_runner() -> Any:
    from vndub_app import jobs

    r = jobs.JobRunner(None, None)
    jobs.reset_runner(r)
    return r


def drop_runner() -> None:
    from vndub_app import jobs

    jobs.reset_runner(None)


def client_for(*modules: Any) -> Any:
    """A TestClient over the routers of ``modules`` (+ the app's error handlers)."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vndub_app import errors

    app = FastAPI()
    errors.install_handlers(app)
    for m in modules:
        app.include_router(m.router)
    return TestClient(app, raise_server_exceptions=False)


def wait_job(job: dict[str, Any] | str, timeout: float = 60.0) -> dict[str, Any]:
    """Block until a job reaches a final state; returns ``JobRecord.info()``."""
    from vndub_app import jobs

    jid = job if isinstance(job, str) else job["id"]
    r = jobs.runner()
    t0 = time.monotonic()
    while True:
        rec = r.wait(jid, 0.2)
        if rec.state in ("done", "failed", "cancelled", "interrupted"):
            return rec.info(with_log=True)
        if time.monotonic() - t0 > timeout:
            raise TimeoutError(f"job {jid} still {rec.state}")


def zip_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


def write_manifest(path: Path, tools: list[dict[str, Any]]) -> Path:
    path.write_text(json.dumps({"schema": 1, "pinned": "test", "tools": tools}, indent=1), encoding="utf-8")
    return path


def tool_entry(name: str, url: str, filename: str, data: bytes, *, provides: dict[str, str], archive: str = "zip", **kw: Any) -> dict[str, Any]:
    """A manifest entry for ``data`` served at ``url`` (sha256 and size filled in)."""
    e: dict[str, Any] = {
        "name": name, "version": "1.0", "kind": "download", "platforms": ["win32", "linux", "darwin"], "engines": ["*"], "required": False,
        "url": url, "filename": filename, "archive": archive, "size": len(data), "sha256": sha(data), "verify": "sha256",
        "provides": provides, "extra": [],
    }
    e.update(kw)
    return e


class RangeServer:
    """``http://127.0.0.1:<port>/<name>`` for ``files`` with HTTP Range support.

    ``drop_first={"name": n}``  the first response for ``name`` is cut after ``n`` bytes (connection closed);
    ``ignore_range``            every response is a full 200 even when a Range header is sent;
    ``status={"name": 404}``    answer with that status.
    ``requests`` records ``(path, range header)`` of every GET.
    """

    def __init__(self, files: dict[str, bytes], *, drop_first: dict[str, int] | None = None, ignore_range: bool = False, status: dict[str, int] | None = None) -> None:
        self.files = files
        self.drop_first = dict(drop_first or {})
        self.ignore_range = ignore_range
        self.status = dict(status or {})
        self.requests: list[tuple[str, str | None]] = []
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a: Any) -> None:  # silence
                pass

            def do_HEAD(self) -> None:  # noqa: N802
                self._serve(head=True)

            def do_GET(self) -> None:  # noqa: N802
                self._serve(head=False)

            def _serve(self, head: bool) -> None:
                name = self.path.lstrip("/").split("?")[0]
                rng = self.headers.get("Range")
                if not head:
                    outer.requests.append((name, rng))
                code = outer.status.get(name)
                if code:
                    self.send_response(code)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                data = outer.files.get(name)
                if data is None:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                start = 0
                status = 200
                if rng and not outer.ignore_range and rng.startswith("bytes="):
                    start = int(rng[6:].split("-")[0] or 0)
                    if start >= len(data):
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{len(data)}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    status = 206
                body = data[start:]
                self.send_response(status)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                if status == 206:
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                self.end_headers()
                if head:
                    return
                cut = outer.drop_first.pop(name, None)
                if cut is not None:
                    self.wfile.write(body[:cut])
                    self.wfile.flush()
                    self.close_connection = True
                    return
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> "RangeServer":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def url(self, name: str) -> str:
        return f"http://127.0.0.1:{self.port}/{name}"


def free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def make_game_project(root: Path, *, engine: str = "renpy", encrypted: bool = False, game_files: dict[str, bytes] | None = None, patch_files: dict[str, bytes] | None = None, install_md: str = "# Install\n\nCopy it.\n") -> tuple[Any, Path, Path]:
    """A project in the data folder with ``work/project.yaml`` and a patch under ``out/<engine>/``.

    Returns ``(handle, game_dir, patch_dir)``. ``game_files`` / ``patch_files`` map relative paths to bytes.
    """
    from vndub.config import ProjectYaml, save_project_yaml
    from vndub.engines import get_engine
    from vndub_app import appdirs

    game = root / "game_copy"
    game.mkdir(parents=True, exist_ok=True)
    for rel, data in (game_files or {}).items():
        p = game / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    h = appdirs.create_project(game, name="Test Game")
    h.work.mkdir(parents=True, exist_ok=True)
    save_project_yaml(h.work / "project.yaml", ProjectYaml(game_dir=str(game), engine=engine, encrypted=encrypted))
    patch = h.out / get_engine(engine).out_name
    patch.mkdir(parents=True, exist_ok=True)
    for rel, data in (patch_files or {}).items():
        p = patch / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    if install_md is not None:
        (patch / "INSTALL.md").write_text(install_md, encoding="utf-8")
    (patch / "manifest_patch.json").write_text(
        json.dumps({"engine": engine, "game_dir": str(game), "files": [{"file": rel, "backend": "mock_tts", "route": "A"} for rel in (patch_files or {})]}), encoding="utf-8"
    )
    return h, game, patch


def tree_snapshot(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}

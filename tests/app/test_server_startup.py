"""Startup stays light, and the app serves through a real uvicorn (the way the shell and the Playwright walk use it)."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

REPO = Path(__file__).resolve().parents[2]


def test_create_app_does_not_import_heavy_pipeline_modules(tmp_path: Path) -> None:
    """The first screen must come up in < 3 s: no librosa / torch / gradio / typer at import or create_app time."""
    code = (
        "import json, sys, time\n"
        "t = time.perf_counter()\n"
        "from vndub_app.server import create_app\n"
        "app = create_app()\n"
        "dt = time.perf_counter() - t\n"
        "heavy = [m for m in ('librosa', 'numba', 'torch', 'gradio', 'typer', 'rich', 'scipy', 'soundfile', 'vndub.cli', 'vndub.stages.common', 'vndub.stages.score', 'vndub.stages.genutil') if m in sys.modules]\n"
        "print(json.dumps({'heavy': heavy, 'seconds': dt, 'routers': len(app.state.routers)}))\n"
    )
    env = {**__import__("os").environ, "VNDUB_DATA": str(tmp_path / "data"), "PYTHONPATH": str(REPO)}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=tmp_path, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res["heavy"] == [], f"imported at startup: {res['heavy']}"
    assert res["routers"] >= 7
    assert res["seconds"] < 10.0  # generous: a loaded CI box; the real target (< 3 s) is dominated by fastapi's own imports


def test_background_uvicorn_serves_api_and_spa(app_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vndub_app import appdirs, server

    monkeypatch.setattr(appdirs, "web_dist", lambda: tmp_path / "no-dist")
    with server.BackgroundServer() as srv:
        assert srv.port > 0 and srv.url.startswith("http://127.0.0.1:")
        r = httpx.get(f"{srv.url}/api/health", timeout=10)
        assert r.status_code == 200 and r.json()["ok"] is True
        assert "not built" in httpx.get(srv.url + "/", timeout=10).text
        err = httpx.get(f"{srv.url}/api/project", timeout=10)
        assert err.status_code == 404 and err.json()["error"]["code"] == "no_project"
        # a foreign Host header (DNS rebinding) is refused by the real server too
        assert httpx.get(f"{srv.url}/api/health", headers={"host": "evil.example.com"}, timeout=10).status_code == 403
    time.sleep(0.2)
    with pytest.raises(httpx.TransportError):
        httpx.get(f"{srv.url}/api/health", timeout=2)  # stopped


@pytest.mark.skipif(importlib.util.find_spec("websockets") is None and importlib.util.find_spec("wsproto") is None, reason="uvicorn has no WebSocket library installed (add 'websockets' to the dependencies)")
def test_background_uvicorn_serves_the_progress_websocket(app_env: Path) -> None:
    from websockets.sync.client import connect  # type: ignore[import-not-found]

    from vndub_app import jobs, server

    with server.BackgroundServer() as srv:
        with connect(f"ws://127.0.0.1:{srv.port}/api/progress", open_timeout=10) as ws:
            hello = json.loads(ws.recv(timeout=10))
            assert hello == {"type": "hello", "jobs": []}
            job = jobs.runner().submit("report", lambda ctx: {"x": 1}, gpu=False)
            seen = []
            while True:
                ev = json.loads(ws.recv(timeout=10))
                seen.append(ev["type"])
                if ev["type"] == "job" and ev["job"]["id"] == job.id and ev["job"]["state"] == "done":
                    break
            assert "job" in seen


def test_progress_sse_fallback_carries_the_same_events(app_env: Path) -> None:
    """GET /api/progress/sse: the same events as the WebSocket (for a uvicorn without a WebSocket library)."""
    from vndub_app import jobs, server

    with server.BackgroundServer() as srv:
        events: list[dict] = []
        with httpx.stream("GET", f"{srv.url}/api/progress/sse", timeout=httpx.Timeout(10.0, read=10.0)) as r:
            assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
            it = r.iter_lines()
            first = next(it)
            assert first.startswith("data: ")
            hello = json.loads(first[len("data: "):])
            assert hello == {"type": "hello", "jobs": []}
            job = jobs.runner().submit("report", lambda ctx: (ctx.log("sse line"), {"x": 1})[1], gpu=False)
            for line in it:
                if not line.startswith("data: "):
                    continue
                ev = json.loads(line[len("data: "):])
                events.append(ev)
                if ev["type"] == "job" and ev["job"]["id"] == job.id and ev["job"]["state"] == "done":
                    break
        got = [e for e in events if e["type"] == "log" and e["job_id"] == job.id]
        assert len(got) == 1 and re.fullmatch(r"\d\d:\d\d:\d\d sse line", got[0]["line"])
        assert events[-1]["job"]["result"] == {"x": 1}

"""Helpers shared by the app tests (import as ``from app_helpers import ...``)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "renpy_mini"


def setup_env(mp: pytest.MonkeyPatch, root: Path) -> Path:
    """Point the app at a throw-away data folder and the all-mock profile; returns the data dir."""
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    mp.setenv("VNDUB_DATA", str(data))
    mp.setenv("VNDUB_APP_PROFILE", "apptest")
    mp.setenv("VNDUB_NO_SHUTDOWN", "1")
    mp.setenv("APPDATA", str(root / "appdata"))
    mp.setenv("LOCALAPPDATA", str(root / "localappdata"))
    mp.setenv("XDG_CONFIG_HOME", str(root / "xdg_config"))
    mp.setenv("XDG_DATA_HOME", str(root / "xdg_data"))
    mp.delenv("VNDUB_MODELS_DIR", raising=False)
    # The app exports VNDUB_ENVS_DIR into os.environ on its own: without this the next test inherits the environments folder
    # of this one as if a shell had set it. set + del makes monkeypatch restore "absent" (or the outside value) afterwards.
    outside = os.environ.get("VNDUB_ENVS_DIR")
    mp.setenv("VNDUB_ENVS_DIR", outside or "-")
    if not outside:
        mp.delenv("VNDUB_ENVS_DIR")
    from vndub_app import installer

    installer.reset_environment_memory()
    return data


def reset_app_state() -> None:
    from vndub_app import jobs, session, viewdata

    jobs.reset_runner(jobs.JobRunner(None, None))
    session.reset()
    viewdata.clear_cache()


def teardown_app_state() -> None:
    from vndub_app import jobs, server, session, viewdata

    jobs.reset_runner(None)
    session.reset()
    viewdata.clear_cache()
    server.close_logging()


def wait_job(client: Any, job: dict[str, Any] | str, timeout: float = 300.0) -> dict[str, Any]:
    """Poll a job until it reaches a final state; returns the final JobInfo (with log tail)."""
    job_id = job if isinstance(job, str) else job["id"]
    t0 = time.monotonic()
    while True:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["state"] in ("done", "failed", "cancelled", "interrupted"):
            return j
        if time.monotonic() - t0 > timeout:
            raise TimeoutError(f"job {job_id} ({j['kind']}) still {j['state']} after {timeout}s: stage={j.get('stage')} {j.get('done')}/{j.get('total')}")
        time.sleep(0.1)


def run_job(client: Any, r: Any, timeout: float = 300.0) -> dict[str, Any]:
    """Assert that an endpoint returned a JobInfo, wait for it, assert it finished."""
    assert r.status_code == 200, r.text
    j = wait_job(client, r.json(), timeout)
    assert j["state"] == "done", f"{j['kind']} -> {j['state']}: {j.get('error')}\n" + "\n".join(j.get("log_tail", [])[-25:])
    return j

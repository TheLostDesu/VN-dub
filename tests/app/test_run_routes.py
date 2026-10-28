"""Step 4 / 5 routes: what they remember (the preview lines, "turn the computer off") is written only once the job is accepted."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Iterator

import pytest
from app_helpers import wait_job

from vndub_app import appdirs, jobs, pipeline

SLUG = "renpy-mini"


@pytest.fixture
def project(client: Any, game_dir: Path) -> dict[str, Any]:
    r = client.post("/api/project/open", json={"game_dir": str(game_dir)})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture
def stubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in ("preview", "full"):
        real = pipeline.FLOWS[k]
        monkeypatch.setitem(pipeline.FLOWS, k, pipeline.Flow(k, lambda slug, p: (lambda ctx: None), gpu=real.gpu, heavy=real.heavy))
    monkeypatch.setattr(pipeline, "first_scene_ids", lambda h: ["l1", "l2"])


@pytest.fixture
def busy() -> Iterator[jobs.JobRecord]:
    """A heavy job of the project that is running: every other heavy job is refused (409 job_busy)."""
    gate = threading.Event()
    job = jobs.runner().submit("full", lambda ctx: gate.wait(20) and None, project=SLUG)
    yield job
    gate.set()
    jobs.runner().wait(job.id, 10)


def test_a_refused_preview_leaves_the_remembered_lines_alone(client: Any, project: dict[str, Any], stubbed: None, busy: jobs.JobRecord) -> None:
    h = appdirs.project_handle(SLUG)
    before = appdirs.read_app_state(h)["preview"]
    r = client.post("/api/preview/run", json={"mode": "first_scene"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "job_busy"
    assert appdirs.read_app_state(h)["preview"] == before  # the lines of the preview that is on screen are still the ones it shows


def test_an_accepted_preview_remembers_its_lines(client: Any, project: dict[str, Any], stubbed: None) -> None:
    r = client.post("/api/preview/run", json={"mode": "first_scene"})
    assert r.status_code == 200, r.text
    assert wait_job(client, r.json())["state"] == "done"
    st = appdirs.read_app_state(appdirs.project_handle(SLUG))["preview"]
    assert st["line_ids"] == ["l1", "l2"] and st["mode"] == "first_scene"


def test_a_refused_full_run_does_not_change_turn_the_computer_off(client: Any, project: dict[str, Any], stubbed: None, busy: jobs.JobRecord) -> None:
    assert appdirs.load_settings().shutdown_when_done is False
    r = client.post("/api/run/full", json={"shutdown_when_done": True})
    assert r.status_code == 409 and r.json()["error"]["code"] == "job_busy"
    assert appdirs.load_settings().shutdown_when_done is False  # the running job did not ask for it


def test_an_accepted_full_run_stores_turn_the_computer_off(client: Any, project: dict[str, Any], stubbed: None) -> None:
    r = client.post("/api/run/full", json={"shutdown_when_done": True})
    assert r.status_code == 200, r.text
    assert r.json()["params"] == {"shutdown_when_done": True}
    assert appdirs.load_settings().shutdown_when_done is True
    wait_job(client, r.json())
    r = client.post("/api/run/full", json={"shutdown_when_done": False})
    assert r.status_code == 200 and appdirs.load_settings().shutdown_when_done is False

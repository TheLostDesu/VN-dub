"""Endpoints that need no pipeline run: health, settings, errors, audio safety, help, system, SPA, WebSocket, jobs."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from app_helpers import run_job, wait_job

ERROR_KEYS = {"code", "params", "action", "details"}


def assert_error(r: Any, code: str, status: int | None = None) -> dict[str, Any]:
    assert r.headers["content-type"].startswith("application/json"), r.text
    body = r.json()
    assert set(body) == {"error"}, body
    assert set(body["error"]) == ERROR_KEYS, body
    assert body["error"]["code"] == code, body
    if status is not None:
        assert r.status_code == status
    return body["error"]


def write_wav(path: Path, seconds: float = 0.5, sr: int = 16000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(sr * seconds)) / sr
    sf.write(str(path), (0.2 * np.sin(2 * np.pi * 440 * t)).astype("float32"), sr)
    return path


@pytest.fixture
def project(client: Any, game_dir: Path) -> dict[str, Any]:
    r = client.post("/api/project/open", json={"game_dir": str(game_dir)})
    assert r.status_code == 200, r.text
    return r.json()


# ----------------------------------------------------------------------------
# general
# ----------------------------------------------------------------------------


def test_health(client: Any) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    j = r.json()
    assert j["ok"] is True and j["profile"] == "apptest" and j["mock"] is True and j["version"]


def test_settings_roundtrip_and_validation(client: Any, app_env: Path) -> None:
    s = client.get("/api/settings").json()
    assert s["ui_lang"] == "ru" and s["data_dir"] == str(app_env) and s["gpu_budget_gb"] == 7.5
    r = client.put("/api/settings", json={"ui_lang": "en", "gpu_budget_gb": 6, "license_policy": "commercial", "player_volume": {"original": 0.4}, "shutdown_when_done": True})
    assert r.status_code == 200
    s = r.json()
    assert (s["ui_lang"], s["gpu_budget_gb"], s["license_policy"], s["shutdown_when_done"]) == ("en", 6.0, "commercial", True)
    assert s["player_volume"] == {"original": 0.4}
    s = client.put("/api/settings", json={"player_volume": {"dub": 1.5}}).json()  # merged and clamped
    assert s["player_volume"] == {"original": 0.4, "dub": 1.0}
    assert json.loads((app_env / "settings.json").read_text(encoding="utf-8"))["ui_lang"] == "en"
    for bad in ({"ui_lang": "de"}, {"gpu_budget_gb": 0}, {"gpu_budget_gb": "x"}, {"license_policy": "free"}, {"nope": 1}, {"profile": "x"}):
        assert_error(client.put("/api/settings", json=bad), "bad_request", 422)


def test_data_dir_switch(client: Any, app_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VNDUB_DATA")  # the real switch is persisted in location.json (redirected into tmp_path by the fixture)
    new = tmp_path / "elsewhere" / "VNDub"
    r = client.post("/api/settings/data-dir", json={"path": str(new)})
    assert r.status_code == 200, r.text
    assert Path(r.json()["data_dir"]) == new.resolve() and new.is_dir()
    assert client.get("/api/settings").json()["data_dir"] == str(new.resolve())
    assert_error(client.post("/api/settings/data-dir", json={"path": ""}), "bad_request", 422)


def test_data_dir_switch_refused_while_a_job_runs(client: Any, app_env: Path, tmp_path: Path) -> None:
    import threading

    from vndub_app import jobs

    gate = threading.Event()
    job = jobs.runner().submit("report", lambda ctx: gate.wait(10) and None, gpu=False)
    try:
        err = assert_error(client.post("/api/settings/data-dir", json={"path": str(tmp_path / "x")}), "job_busy", 409)
        assert err["params"]["kind"] == "report"
    finally:
        gate.set()
        wait_job(client, job.id)


def test_error_format_everywhere(client: Any) -> None:
    assert_error(client.get("/api/project"), "no_project", 404)
    assert_error(client.get("/api/jobs/doesnotexist"), "job_not_found", 404)
    assert_error(client.get("/api/nothing/here"), "file_not_found", 404)
    # request validation (pydantic) is mapped to bad_request too, never FastAPI's own {"detail": ...}
    e = assert_error(client.put("/api/settings", content=b"not json", headers={"content-type": "application/json"}), "bad_request", 422)
    assert e["action"] == "none"
    assert_error(client.post("/api/project/open", json={}), "bad_request", 422)
    err = assert_error(client.post("/api/project/open", json={"game_dir": "Z:/definitely/not/here"}), "game_dir_invalid", 400)
    assert err["action"] == "choose_game_dir" and err["params"]["path"]
    assert_error(client.post("/api/project/open", json={"slug": "unknown"}), "project_not_found", 404)


def test_foreign_host_and_origin_are_refused(client: Any) -> None:
    assert_error(client.get("/api/health", headers={"host": "evil.example.com"}), "permission_denied", 403)
    assert client.get("/api/health", headers={"host": "localhost:8123"}).status_code == 200
    assert client.get("/api/health", headers={"host": "127.0.0.1:1"}).status_code == 200
    assert_error(client.post("/api/system/open", json={"kind": "logs"}, headers={"origin": "http://evil.example.com"}), "permission_denied", 403)
    # reading with a foreign origin is not blocked by the server (the browser's CORS keeps the answer from the page)
    assert client.get("/api/health", headers={"origin": "http://evil.example.com"}).status_code == 200


# ----------------------------------------------------------------------------
# help, SPA
# ----------------------------------------------------------------------------


def test_help_docs(client: Any) -> None:
    idx = client.get("/api/help").json()
    ids = [d["id"] for d in idx]
    assert "troubleshooting" in ids and "readme" in ids
    assert all(d["title_key"] == f"help.{d['id']}.title" for d in idx)
    if "app" in ids:
        assert ids[0] == "app"  # the user manual comes first
    md = client.get("/api/help/troubleshooting").json()
    assert md["id"] == "troubleshooting" and "Troubleshooting" in md["markdown"]
    assert_error(client.get("/api/help/..%2f..%2fsecret"), "file_not_found", 404)
    assert_error(client.get("/api/help/unknown"), "file_not_found", 404)


def test_spa_fallback_and_plain_page(client: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vndub_app import appdirs

    monkeypatch.setattr(appdirs, "web_dist", lambda: tmp_path / "no-dist")
    r = client.get("/")
    assert r.status_code == 200 and "not built" in r.text and "/api/health" in r.text
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html><body>SPA</body></html>", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    monkeypatch.setattr(appdirs, "web_dist", lambda: dist)
    assert "SPA" in client.get("/").text
    assert "SPA" in client.get("/casting/deep/link").text  # client-side routes fall back to index.html
    js = client.get("/assets/app.js")
    assert js.status_code == 200 and "immutable" in js.headers["cache-control"]
    (tmp_path / "secret.txt").write_text("TOPSECRET", encoding="utf-8")
    for p in ("/%2e%2e/secret.txt", "/..%2fsecret.txt", "/assets/../../secret.txt"):
        assert "TOPSECRET" not in client.get(p).text  # no escape out of dist
    assert_error(client.get("/api/unknown"), "file_not_found", 404)  # API paths never fall back


# ----------------------------------------------------------------------------
# audio
# ----------------------------------------------------------------------------


def test_audio_serves_only_inside_work_and_out(client: Any, project: dict[str, Any], app_env: Path, tmp_path: Path) -> None:
    from vndub_app import appdirs

    h = appdirs.project_handle(project["slug"])
    write_wav(h.work / "audio" / "src" / "a.wav", 1.0)
    write_wav(h.out / "renpy" / "game" / "voice" / "b.wav")
    (h.work / "project.yaml").write_text("engine: renpy\n", encoding="utf-8")
    (h.work / "notes.txt").write_text("secret", encoding="utf-8")
    secret = tmp_path / "secret.wav"
    write_wav(secret)
    slug = project["slug"]

    r = client.get(f"/api/audio/{slug}/audio/src/a.wav")
    assert r.status_code == 200 and r.headers["content-type"].startswith("audio/") and len(r.content) > 1000
    assert client.get(f"/api/audio/{slug}/_out/renpy/game/voice/b.wav").status_code == 200

    # range requests (the browser's <audio> seeks with them)
    rr = client.get(f"/api/audio/{slug}/audio/src/a.wav", headers={"range": "bytes=0-99"})
    assert rr.status_code == 206 and len(rr.content) == 100 and rr.headers["content-range"].startswith("bytes 0-99/")

    for p in (
        "../../../secret.wav", "..%2f..%2f..%2fsecret.wav", "%2e%2e/%2e%2e/%2e%2e/secret.wav", "audio/../../app_state.json",
        "audio/src/../../../../settings.json", f"{secret.as_posix()}", f"/{secret.as_posix()}", "..\\..\\secret.wav", "audio\\..\\..\\secret.wav",
        "C:/Windows/win.ini", "project.yaml", "notes.txt", "audio/src/missing.wav", "audio/src", "_out", "_out/../work/audio/src/a.wav".replace("work", "x"),
    ):
        r = client.get(f"/api/audio/{slug}/{p}")
        # the client may normalise "../" away, in which case the request falls to the SPA page (HTML), never to a file
        served_a_file = r.status_code == 200 and not r.headers["content-type"].startswith("text/html")
        assert not served_a_file, (p, r.status_code)
        if r.status_code == 404:
            assert r.json()["error"]["code"] == "file_not_found"
    # another project / unknown project
    assert_error(client.get(f"/api/audio/nope/audio/src/a.wav"), "project_not_found", 404)
    # the data folder's own files are not reachable through any slug
    r = client.get(f"/api/audio/{slug}/..%2f..%2f..%2fsettings.json")
    assert r.status_code == 404 and r.json()["error"]["code"] == "file_not_found"


def test_audio_rejects_symlinks_leaving_the_project(client: Any, project: dict[str, Any], tmp_path: Path) -> None:
    from vndub_app import appdirs

    h = appdirs.project_handle(project["slug"])
    outside = write_wav(tmp_path / "outside" / "x.wav")
    h.work.mkdir(parents=True, exist_ok=True)
    try:
        (h.work / "link.wav").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not available")
    assert_error(client.get(f"/api/audio/{project['slug']}/link.wav"), "file_not_found", 404)


def test_voice_library_audio_slug(client: Any, app_env: Path) -> None:
    write_wav(app_env / "voices" / "v1.wav")
    assert client.get("/api/audio/_voices/v1.wav").status_code == 200
    assert client.get("/api/audio/_voices/..%2fsettings.json").status_code == 404
    assert client.get("/api/audio/_voices/v2.wav").status_code == 404


# ----------------------------------------------------------------------------
# system
# ----------------------------------------------------------------------------


def test_pick_dialogs_are_unsupported_without_a_window(client: Any) -> None:
    assert client.post("/api/system/pick-folder", json={"purpose": "game"}).json() == {"path": None, "unsupported": True}
    assert client.post("/api/system/pick-file", json={"purpose": "translation", "filters": ["*.json"]}).json() == {"path": None, "unsupported": True}
    assert client.post("/api/system/pick-folder").json()["unsupported"] is True


def test_pick_dialogs_use_the_window(client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from vndub_app import system

    calls: list[Any] = []

    class FakeWindow:
        def create_file_dialog(self, kind: Any, **kw: Any) -> list[str]:
            calls.append((kind, kw))
            return ["C:/games/Foo"]

    system.set_window(FakeWindow())
    try:
        assert client.post("/api/system/pick-folder", json={"purpose": "game"}).json() == {"path": "C:/games/Foo"}
        r = client.post("/api/system/pick-file", json={"purpose": "wav", "filters": ["*.wav", ".mp3"]}).json()
        assert r == {"path": "C:/games/Foo"}
        assert calls[1][1]["file_types"] == ("Files (*.wav;*.mp3)",) and calls[1][1]["allow_multiple"] is False
    finally:
        system.set_window(None)


def test_open_kinds(client: Any, project: dict[str, Any], app_env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from vndub_app import system

    opened: list[str] = []
    monkeypatch.setattr(system, "open_path", lambda p: opened.append(str(p)) or True)
    for kind in ("data_dir", "logs", "backups", "patch", "work"):
        assert client.post("/api/system/open", json={"kind": kind}).json() == {"ok": True}
    assert opened[0] == str(app_env) and opened[1].endswith("logs") and opened[2].endswith("backups")
    assert opened[3].endswith("out") and opened[4].endswith("work")
    assert client.post("/api/system/open", json={"kind": "path", "path": str(tmp_path)}).json() == {"ok": True}
    assert_error(client.post("/api/system/open", json={"kind": "path", "path": str(tmp_path / "missing")}), "file_not_found", 404)
    assert_error(client.post("/api/system/open", json={"kind": "bogus"}), "bad_request", 422)


def test_shutdown_pc_is_guarded_and_never_runs_in_tests(client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from vndub_app import system

    ran: list[Any] = []
    monkeypatch.setattr("subprocess.run", lambda *a, **k: ran.append(a))
    assert_error(client.post("/api/system/shutdown-pc"), "permission_denied", 403)  # only when the user ticked the box
    client.put("/api/settings", json={"shutdown_when_done": True})
    assert client.post("/api/system/shutdown-pc").json() == {"ok": False}  # apptest profile: skipped
    assert system.shutdown_pc() is False and ran == []


def test_problem_report_zip(client: Any, project: dict[str, Any], app_env: Path) -> None:
    from vndub_app import appdirs

    h = appdirs.project_handle(project["slug"])
    write_wav(h.work / "audio" / "src" / "a.wav")
    (h.work / "stage_status.json").write_text("{}", encoding="utf-8")
    (h.work / "project.yaml").write_text("engine: renpy\n", encoding="utf-8")
    (h.work / "logs").mkdir(exist_ok=True)
    (h.work / "logs" / "link.log").write_text("link log", encoding="utf-8")
    j = run_job(client, client.post("/api/system/report"))
    path = Path(j["result"]["path"])
    assert path.exists() and path.suffix == ".zip"
    names = zipfile.ZipFile(path).namelist()
    assert {"stage_status.json", "project.yaml", "app_state.json", "work_logs/link.log"} <= set(names)
    assert not [n for n in names if n.endswith((".wav", ".ogg", ".mp3"))]  # no audio
    assert j["kind"] == "report" and j["project"] is None or j["project"] == project["slug"]


# ----------------------------------------------------------------------------
# projects + state
# ----------------------------------------------------------------------------


def test_open_list_state(client: Any, game_dir: Path, project: dict[str, Any]) -> None:
    assert project["slug"] == "renpy-mini" and project["game_dir"] == str(game_dir.resolve())
    assert project["engine"] is None and project["project_yaml"] is None and project["stage_status"] == {}
    assert project["active_job"] is None and project["resume"] == {"available": False, "job": None, "next_stage": None}
    assert project["state"]["step"] == "game" and project["state"]["translation"]["tgt_lang"] == "ru"
    # opening the same folder again returns the same project
    again = client.post("/api/project/open", json={"game_dir": str(game_dir)}).json()
    assert again["slug"] == project["slug"]
    lst = client.get("/api/projects").json()
    assert [p["slug"] for p in lst] == ["renpy-mini"] and set(lst[0]) == {"slug", "name", "game_dir", "engine", "step", "updated"}
    assert client.get("/api/project").json()["slug"] == "renpy-mini"
    assert client.post("/api/project/open", json={"slug": "renpy-mini"}).status_code == 200

    # PUT state: deep merge, regenerates pipeline.yaml, protected keys are ignored
    st = client.put("/api/project/state", json={"step": "translate", "completed": ["game"], "translation": {"tgt_lang": "en"}, "config_overrides": {"x": 1}, "ui": {"tab": 2}}).json()
    assert st["step"] == "translate" and st["completed"] == ["game"] and st["translation"]["tgt_lang"] == "en" and st["translation"]["honorifics"] == "drop"
    assert "x" not in st["config_overrides"] and st["ui"] == {"tab": 2}
    st = client.put("/api/project/state", json={"review": {"touched": ["a", "b"], "last_apply": {"changed": 9}, "filter": "check"}, "preview": {"line_ids": ["z"], "n": 12}}).json()
    assert st["review"]["touched"] == [] and "last_apply" not in st["review"] and st["review"]["filter"] == "check"  # the backend owns these
    assert st["preview"]["line_ids"] == [] and st["preview"]["n"] == 12
    from vndub_app import appdirs

    h = appdirs.project_handle("renpy-mini")
    assert "tgt_lang: en" in h.config_file.read_text(encoding="utf-8")
    assert client.get("/api/project").json()["state"]["step"] == "translate"
    # the current project is remembered across a restart of the session
    from vndub_app import session

    session.reset()
    assert client.get("/api/project").json()["slug"] == "renpy-mini"


def test_endpoints_before_the_game_is_imported(client: Any, project: dict[str, Any]) -> None:
    """A freshly opened project answers every read endpoint with empty data, never with a crash."""
    s = client.get("/api/project/summary").json()
    assert s["engine"]["id"] is None and s["lines_total"] == 0 and s["characters"] == [] and s["voice_files"] == 0
    assert s["mapping"]["ok"] == 0 and s["no_text_lines"] == 0 and s["problem_files"] == [] and s["analysis"] == {"audio_passports": 0, "text_passports": 0}
    assert [t["tool"] for t in s["tools_missing"]] == [] or all({"tool", "url"} <= set(t) for t in s["tools_missing"])
    assert client.get("/api/characters").json() == []
    assert client.get("/api/project/glossary").json() == {"characters": [], "terms": []}
    r = client.get("/api/review/lines").json()
    assert r["items"] == [] and r["counts"] == {"ok": 0, "check": 0, "failed": 0} and r["total"] == 0
    assert client.get("/api/preview/items").json() == {"line_ids": [], "items": []}
    t = client.get("/api/project/translation").json()
    assert t["tgt_lang"] == "ru" and t["llm_available"] is True
    assert client.get("/api/project/delivery-cues").status_code == 200
    assert client.get("/api/voices").json() == []
    assert_error(client.post("/api/preview/run", json={"mode": "first_scene"}), "import_required", 400)
    assert_error(client.post("/api/review/apply"), "nothing_to_review", 400)
    assert_error(client.get("/api/review/lines/x"), "file_not_found", 404)
    assert_error(client.get("/api/casting/sakura/proposals"), "bad_request", 404)


def test_stage_run_validation(client: Any, project: dict[str, Any]) -> None:
    assert_error(client.post("/api/stages/nonsense/run", json={}), "bad_request", 422)
    assert_error(client.post("/api/stages/synth/run", json={"only": "not-a-list"}), "bad_request", 422)
    assert_error(client.post("/api/stages/synth/run", json={"sets": {"synth.nonexistent_key": 1}}), "bad_request", 422)
    assert_error(client.post("/api/stages/synth/run", json={"sets": {"synth.n_candidates": "many"}}), "bad_request", 422)


# ----------------------------------------------------------------------------
# jobs routes
# ----------------------------------------------------------------------------


def test_jobs_pause_resume_cancel_routes(client: Any) -> None:
    import threading
    import time

    from vndub_app import jobs

    steps: list[int] = []

    def fn(ctx: jobs.JobContext) -> None:
        for i in range(500):
            ctx.progress(i, 500)
            steps.append(i)
            time.sleep(0.01)

    job = jobs.runner().submit("stage", fn, project="p1")
    deadline = time.monotonic() + 10
    while len(steps) < 3 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert [j["id"] for j in client.get("/api/jobs", params={"active": 1}).json()] == [job.id]
    assert [j["id"] for j in client.get("/api/jobs", params={"project": "p1"}).json()] == [job.id]
    assert client.get("/api/jobs", params={"project": "other"}).json() == []
    assert client.post(f"/api/jobs/{job.id}/pause").json()["id"] == job.id
    while client.get(f"/api/jobs/{job.id}").json()["state"] != "paused" and time.monotonic() < deadline:
        time.sleep(0.02)
    d = client.get(f"/api/jobs/{job.id}").json()
    assert d["state"] == "paused" and "log_tail" in d
    n = len(steps)
    time.sleep(0.2)
    assert len(steps) <= n + 1
    client.post(f"/api/jobs/{job.id}/resume")
    while client.get(f"/api/jobs/{job.id}").json()["state"] != "running" and time.monotonic() < deadline:
        time.sleep(0.02)
    r = client.post(f"/api/jobs/{job.id}/cancel")
    assert r.status_code == 200
    assert wait_job(client, job.id)["state"] == "cancelled"
    assert "log_tail" not in client.get("/api/jobs").json()[0]  # the list stays light
    assert_error(client.post("/api/jobs/zzz/pause"), "job_not_found", 404)
    _ = threading


def test_resume_resubmits_the_interrupted_job(client: Any, project: dict[str, Any], app_env: Path) -> None:
    """A job that was active when the app died is offered for resume and re-submitted with its kind + params."""
    from vndub_app import jobs

    state = app_env / "jobs.json"
    old = jobs.JobRecord(id="oldjob", kind="report", title_key="jobs.report", params={"slug": "renpy-mini"}, project="renpy-mini", gpu=False, state="running",
                         created="2026-01-01T00:00:00+00:00", stages=["a", "b"], stage="b", stage_index=2, stage_count=2)
    state.write_text(json.dumps([old.info(with_log=True)]), encoding="utf-8")
    jobs.reset_runner(jobs.JobRunner(state, None))  # "restart": the persisted running job comes back as interrupted
    p = client.get("/api/project").json()
    assert p["resume"]["available"] is True and p["resume"]["job"]["id"] == "oldjob" and p["resume"]["next_stage"] == "b"
    r = client.post("/api/project/resume")
    assert r.status_code == 200
    new = r.json()
    assert new["kind"] == "report" and new["params"] == {"slug": "renpy-mini"} and new["id"] != "oldjob"
    assert wait_job(client, new)["state"] == "done"
    p = client.get("/api/project").json()
    assert p["resume"]["available"] is False  # the old record was forgotten
    assert_error(client.post("/api/project/resume"), "nothing_to_resume", 404)


def test_busy_and_duplicate_submission(client: Any, project: dict[str, Any]) -> None:
    import threading

    from vndub_app import jobs, pipeline

    gate = threading.Event()
    fl = pipeline.FLOWS["import"]
    pipeline.FLOWS["import"] = pipeline.Flow("import", lambda slug, p: (lambda ctx: gate.wait(10) and None), gpu=True, heavy=True)
    try:
        j1 = client.post("/api/project/import", json={})
        assert j1.status_code == 200
        err = assert_error(client.post("/api/project/import", json={}), "job_busy", 409)  # a second heavy job is refused
        assert err["params"]["job_id"] == j1.json()["id"]
        assert_error(client.post("/api/project/translate"), "job_busy", 409)
        assert client.get("/api/project").json()["active_job"]["id"] == j1.json()["id"]
    finally:
        gate.set()
        pipeline.FLOWS["import"] = fl
        wait_job(client, j1.json())
    _ = jobs


# ----------------------------------------------------------------------------
# WebSocket
# ----------------------------------------------------------------------------


def test_progress_websocket_hello_and_events(client: Any) -> None:
    import threading

    from vndub_app import jobs

    gate = threading.Event()
    running = jobs.runner().submit("report", lambda ctx: (ctx.log("first line"), gate.wait(10))[1] and None, gpu=False)
    with client.websocket_connect("/api/progress") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello" and [j["id"] for j in hello["jobs"]] == [running.id]
        gate.set()
        seen: set[str] = set()
        finished = False
        for _ in range(50):
            ev = ws.receive_json()
            seen.add(ev["type"])
            if ev["type"] == "job" and ev["job"]["id"] == running.id and ev["job"]["state"] == "done":
                finished = True
                break
        assert finished and "job" in seen
        # a job started after the socket opened is streamed too, with its log lines
        j2 = jobs.runner().submit("report", lambda ctx: (ctx.log("second line"), {"x": 1})[1], gpu=False)
        got_log = got_done = False
        for _ in range(50):
            ev = ws.receive_json()
            if ev["type"] == "log" and ev["job_id"] == j2.id:
                got_log = ev["line"] == "second line"
            if ev["type"] == "job" and ev["job"]["id"] == j2.id and ev["job"]["state"] == "done":
                got_done = True
                assert ev["job"]["result"] == {"x": 1}
                break
        assert got_log and got_done


def test_websocket_refuses_foreign_origin(client: Any) -> None:
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/progress", headers={"origin": "http://evil.example.com"}) as ws:
            ws.receive_json()


def test_gpu_endpoint_shape(client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from vndub_app import gpuinfo

    monkeypatch.setattr(gpuinfo, "query", lambda max_age=1.0: None)
    assert client.get("/api/gpu").json() == {"available": False}
    monkeypatch.setattr(gpuinfo, "query", lambda max_age=1.0: {"name": "RTX 4060", "total_gb": 8.0, "used_gb": 1.5})
    assert client.get("/api/gpu").json() == {"name": "RTX 4060", "total_gb": 8.0, "used_gb": 1.5}


def test_gpu_query_parses_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    from vndub_app import gpuinfo

    class R:
        returncode = 0
        stdout = "NVIDIA GeForce RTX 4060, 8188, 1024\n"

    monkeypatch.setattr(gpuinfo.shutil, "which", lambda n: "nvidia-smi")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: R())
    monkeypatch.setattr(gpuinfo, "_CACHE", (0.0, None))
    assert gpuinfo.query(max_age=0) == {"name": "NVIDIA GeForce RTX 4060", "total_gb": 8.0, "used_gb": 1.0}
    monkeypatch.setattr(gpuinfo.shutil, "which", lambda n: None)
    assert gpuinfo.query(max_age=0) is None


def test_routers_of_other_tasks_are_optional(app_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing optional router module is skipped with a warning; the app still starts."""
    import sys

    from fastapi.testclient import TestClient

    from vndub_app import server

    monkeypatch.setattr(server, "EXTERNAL_MODULES", ("vndub_app.does_not_exist", "vndub_app.errors"))  # errors has no `router`
    monkeypatch.delitem(sys.modules, "vndub_app.does_not_exist", raising=False)
    app = server.create_app()
    with TestClient(app) as c:
        assert c.get("/api/health").status_code == 200
    assert "vndub_app.does_not_exist" not in app.state.routers and "vndub_app.errors" not in app.state.routers

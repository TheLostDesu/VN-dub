"""Round 8: an honest resume (which job, with which settings, no surprise shutdown) and the warnings of a run.

The interrupted jobs come from a ``jobs.json`` that a "restarted" runner loads; the flows that would really run the
pipeline are replaced by stubs, because what is under test is the choice, the params and the record. The warnings tests
run a real ``pipeline_ctx`` (all-mock ``apptest`` profile) and push entries into the registry the way the stages do.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from app_helpers import wait_job

from vndub.gpu import Substitution
from vndub_app import appdirs, jobs, pipeline
from vndub_app.jobs import JobCancelled

SLUG = "renpy-mini"


def ago(hours: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(timespec="seconds")


@pytest.fixture
def project(client: Any, game_dir: Path) -> dict[str, Any]:
    r = client.post("/api/project/open", json={"game_dir": str(game_dir)})
    assert r.status_code == 200, r.text
    return r.json()


def record(job_id: str, kind: str, hours_ago: float, params: dict[str, Any] | None = None, settings: dict[str, Any] | None = None) -> dict[str, Any]:
    return jobs.JobRecord(
        id=job_id, kind=kind, title_key=f"jobs.{kind}", params=params or {}, project=SLUG, state="interrupted",
        created=ago(hours_ago), finished=ago(hours_ago - 0.1), settings=settings or {},
    ).info(with_log=True)


def restart(app_env: Path, records: list[dict[str, Any]]) -> jobs.JobRunner:
    """The app 'restarts': a fresh runner loads these records from jobs.json."""
    state = app_env / "jobs.json"
    state.write_text(json.dumps(records), encoding="utf-8")
    r = jobs.JobRunner(state, None)
    jobs.reset_runner(r)
    return r


def stub_flows(monkeypatch: pytest.MonkeyPatch, *kinds: str) -> None:
    """Replace the flows by ones that finish at once (their kind keeps its ``heavy`` / ``gpu`` flags)."""
    for k in kinds:
        real = pipeline.FLOWS[k]
        monkeypatch.setitem(pipeline.FLOWS, k, pipeline.Flow(k, lambda slug, p: (lambda ctx: None), gpu=real.gpu, heavy=real.heavy))


def assert_error(r: Any, code: str, status: int) -> None:
    assert r.status_code == status, r.text
    assert r.json()["error"]["code"] == code


# ----------------------------------------------------------------------------
# which job is resumed
# ----------------------------------------------------------------------------


def test_a_long_full_run_is_resumed_before_a_later_listen_and_the_listen_stays(
    client: Any, project: dict[str, Any], app_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_flows(monkeypatch, "full", "casting_listen")
    restart(app_env, [
        record("full1", "full", 30, {"shutdown_when_done": True}),
        record("listen1", "casting_listen", 1, {"speaker_id": "sakura", "n": 3}),
    ])
    p = client.get("/api/project").json()
    assert p["resume"]["available"] is True and p["resume"]["job"]["id"] == "full1"
    assert [j["id"] for j in p["resume"]["candidates"]] == ["listen1"]

    new = client.post("/api/project/resume").json()
    assert new["kind"] == "full" and new["id"] not in ("full1", "listen1")
    assert new["params"] == {}  # the shutdown of the interrupted run is not replayed
    assert wait_job(client, new)["state"] == "done"

    # the full run was forgotten, the listen stays interrupted and is offered now
    assert [j.id for j in jobs.runner().interrupted(SLUG)] == ["listen1"]
    p = client.get("/api/project").json()
    assert p["resume"]["job"]["id"] == "listen1" and p["resume"]["candidates"] == []
    again = client.post("/api/project/resume").json()
    assert again["kind"] == "casting_listen" and again["params"] == {"speaker_id": "sakura", "n": 3}
    assert wait_job(client, again)["state"] == "done"
    assert_error(client.post("/api/project/resume"), "nothing_to_resume", 404)


def test_resume_keeps_the_shutdown_only_when_asked(client: Any, project: dict[str, Any], app_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub_flows(monkeypatch, "full")
    for body, expected in (({"keep_shutdown": True}, {"shutdown_when_done": True}), ({"keep_shutdown": False}, {}), ({"keep_shutdown": "yes"}, {}), ({}, {})):
        restart(app_env, [record("full1", "full", 5, {"shutdown_when_done": True})])
        new = client.post("/api/project/resume", json=body).json()
        assert new["params"] == expected, body
        assert wait_job(client, new)["state"] == "done"


def test_only_jobs_of_the_same_kind_are_forgotten(client: Any, project: dict[str, Any], app_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub_flows(monkeypatch, "full", "tuning_listen")
    restart(app_env, [
        record("full_a", "full", 40), record("full_b", "full", 30, {"shutdown_when_done": False}),
        record("tune", "tuning_listen", 1, {"speaker_id": "sakura", "tuning": {}}),
    ])
    p = client.get("/api/project").json()
    assert p["resume"]["job"]["id"] == "full_b"  # the newest of the heavy kinds
    assert [j["id"] for j in p["resume"]["candidates"]] == ["tune"]  # the older full run is superseded by it, not offered
    new = client.post("/api/project/resume").json()
    assert wait_job(client, new)["state"] == "done"
    assert [j.id for j in jobs.runner().interrupted(SLUG)] == ["tune"]


def test_without_a_heavy_job_the_newest_of_the_rest_is_resumed(client: Any, project: dict[str, Any], app_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub_flows(monkeypatch, "casting_listen", "tuning_listen")
    restart(app_env, [record("listen", "casting_listen", 5, {"speaker_id": "a"}), record("tune", "tuning_listen", 2, {"speaker_id": "a"})])
    p = client.get("/api/project").json()
    assert p["resume"]["job"]["id"] == "tune" and [j["id"] for j in p["resume"]["candidates"]] == ["listen"]
    assert client.post("/api/project/resume").json()["kind"] == "tuning_listen"


def test_a_job_kind_this_version_does_not_have_is_not_offered(client: Any, project: dict[str, Any], app_env: Path) -> None:
    restart(app_env, [record("gone", "kind_from_the_future", 3)])
    p = client.get("/api/project").json()
    assert p["resume"] == {"available": False, "job": None, "next_stage": None, "candidates": []}
    assert_error(client.post("/api/project/resume"), "nothing_to_resume", 404)


# ----------------------------------------------------------------------------
# the settings snapshot
# ----------------------------------------------------------------------------


def test_submit_records_the_settings_of_the_moment(client: Any, project: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    stub_flows(monkeypatch, "full")
    job = pipeline.submit("full", SLUG, {})
    assert wait_job(client, job.id)["state"] == "done"
    s = job.settings
    assert set(s) == {"gpu_budget_gb", "translation_llm", "license_policy", "models_dir", "envs_dir", "profile", "src_lang", "tgt_lang"}
    assert (s["gpu_budget_gb"], s["license_policy"], s["profile"]) == (7.5, "research", "apptest")
    assert (s["translation_llm"], s["src_lang"], s["tgt_lang"]) == ("gemma4_12b", "auto", "ru")
    assert s["models_dir"] == str(appdirs.effective_models_dir()) and s["envs_dir"] == str(appdirs.effective_envs_dir())
    assert client.get(f"/api/jobs/{job.id}").json()["settings"] == s
    # a job that belongs to no project knows no languages
    rep = pipeline.submit("report", None, {})
    assert "src_lang" not in rep.settings and rep.settings["profile"] == "apptest"
    wait_job(client, rep.id)


def test_a_resume_with_changed_settings_carries_the_warning(client: Any, project: dict[str, Any], app_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub_flows(monkeypatch, "full")
    then = appdirs.settings_snapshot(SLUG)
    restart(app_env, [record("full1", "full", 3, settings=then)])
    assert client.put("/api/settings", json={"gpu_budget_gb": 6, "license_policy": "commercial"}).status_code == 200
    new = client.post("/api/project/resume").json()
    done = wait_job(client, new)
    assert done["state"] == "done"
    assert done["result"]["warnings"] == [{"kind": "settings_changed", "keys": "gpu_budget_gb, license_policy"}]
    assert done["settings"]["gpu_budget_gb"] == 6.0 and done["settings"]["license_policy"] == "commercial"  # it runs with the settings of now


def test_a_resume_with_the_same_settings_or_an_old_record_has_no_warning(
    client: Any, project: dict[str, Any], app_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_flows(monkeypatch, "full")
    restart(app_env, [record("full1", "full", 3, settings=appdirs.settings_snapshot(SLUG))])
    assert "warnings" not in wait_job(client, client.post("/api/project/resume").json())["result"]
    assert client.put("/api/settings", json={"gpu_budget_gb": 5}).status_code == 200
    restart(app_env, [record("full2", "full", 3, settings={})])  # written before the snapshot existed: nothing to compare
    assert "warnings" not in wait_job(client, client.post("/api/project/resume").json())["result"]


# ----------------------------------------------------------------------------
# warnings of a run: substitutions and skipped parts
# ----------------------------------------------------------------------------


def run_in_ctx(body: Any, *, kind: str = "stage") -> jobs.JobRecord:
    """Run ``body(job_ctx, pipeline_ctx)`` inside a real ``pipeline_ctx`` of the opened project."""
    h = appdirs.project_handle(SLUG)

    def fn(job_ctx: jobs.JobContext) -> None:
        with pipeline.pipeline_ctx(h, job_ctx) as ctx:
            body(job_ctx, ctx)

    r = jobs.runner()
    job = r.submit(kind, fn, project=SLUG)
    r.wait(job.id, 60)
    return job


def test_substitutions_and_skipped_parts_become_warnings(client: Any, project: dict[str, Any]) -> None:
    long_reason = "gemma4_12b: needs 9.8 GB, 4.1 GB free; " * 20

    def body(job_ctx: jobs.JobContext, ctx: Any) -> None:
        job_ctx.set_result(warnings=[{"kind": "settings_changed", "keys": "gpu_budget_gb"}])  # earlier warnings are kept
        ctx.models.substitutions.append(Substitution("gemma4_12b", "qwen35_9b", long_reason))
        assert jobs_message(job_ctx) == ("jobs.msg.substituted", {"requested": "gemma4_12b", "used": "qwen35_9b"})  # announced while active
        ctx.models.substitutions.append(Substitution("qwen35_9b", "mock_translator", "second"))
        ctx.models.record_unavailable("score", "cer", "no ASR backend: out of VRAM")
        ctx.models.record_unavailable("score", "cer", "no ASR backend: out of VRAM")  # the registry de-duplicates per stage + group
        ctx.models.record_unavailable("analyze", "events", "env missing")

    job = run_in_ctx(body)
    assert job.state == "done", job.error
    w = job.result["warnings"]
    assert w[0] == {"kind": "settings_changed", "keys": "gpu_budget_gb"}
    assert w[1]["kind"] == "substitution" and (w[1]["requested"], w[1]["used"]) == ("gemma4_12b", "qwen35_9b")
    assert len(w[1]["reason"]) <= 200 and w[1]["reason"].endswith("...")
    assert w[2] == {"kind": "substitution", "requested": "qwen35_9b", "used": "mock_translator", "reason": "second"}
    assert w[3:] == [
        {"kind": "unavailable", "stage": "score", "group": "cer", "reason": "no ASR backend: out of VRAM"},
        {"kind": "unavailable", "stage": "analyze", "group": "events", "reason": "env missing"},
    ]
    assert all(isinstance(v, str) for item in w for v in item.values())
    assert (job.message_key, job.message_params) == ("jobs.msg.substituted", {"requested": "gemma4_12b", "used": "qwen35_9b"})  # the first one only


def jobs_message(job_ctx: jobs.JobContext) -> tuple[str | None, dict[str, Any]]:
    return job_ctx.job.message_key, dict(job_ctx.job.message_params)


def test_warnings_survive_a_failed_run(client: Any, project: dict[str, Any]) -> None:
    def body(job_ctx: jobs.JobContext, ctx: Any) -> None:
        ctx.models.record_unavailable("synth", "route B", "RVC model missing")
        raise RuntimeError("boom")

    job = run_in_ctx(body)
    assert job.state == "failed"
    assert job.result["warnings"] == [{"kind": "unavailable", "stage": "synth", "group": "route B", "reason": "RVC model missing"}]


def test_a_clean_run_has_no_warnings_key(client: Any, project: dict[str, Any]) -> None:
    job = run_in_ctx(lambda job_ctx, ctx: None)
    assert job.state == "done" and "warnings" not in job.result


def test_the_stage_context_can_raise_the_cancel(client: Any, project: dict[str, Any]) -> None:
    seen: list[str] = []

    def body(job_ctx: jobs.JobContext, ctx: Any) -> None:
        assert ctx.extra["cancel_check"] == job_ctx.checkpoint
        ctx.check_cancel()  # nothing pending: returns
        job_ctx.runner.cancel(job_ctx.job.id)
        try:
            ctx.check_cancel()
        except JobCancelled:
            seen.append("raised")
            raise

    job = run_in_ctx(body)
    assert seen == ["raised"] and job.state == "cancelled"


def test_a_paused_job_waits_in_check_cancel(client: Any, project: dict[str, Any]) -> None:
    started, release = threading.Event(), threading.Event()
    order: list[str] = []

    def body(job_ctx: jobs.JobContext, ctx: Any) -> None:
        started.set()
        release.wait(10)
        ctx.check_cancel()  # paused by now: blocks until resumed
        order.append("after")

    h = appdirs.project_handle(SLUG)

    def fn(job_ctx: jobs.JobContext) -> None:
        with pipeline.pipeline_ctx(h, job_ctx) as ctx:
            body(job_ctx, ctx)

    r = jobs.runner()
    job = r.submit("stage", fn, project=SLUG)
    assert started.wait(30)
    r.pause(job.id)
    release.set()
    deadline = threading.Event()
    for _ in range(100):
        if job.state == "paused":
            break
        deadline.wait(0.05)
    assert job.state == "paused" and order == []
    r.resume(job.id)
    r.wait(job.id, 30)
    assert job.state == "done" and order == ["after"]


# ----------------------------------------------------------------------------
# warnings of a run: mixed translation models
# ----------------------------------------------------------------------------


def test_models_used_warning_names_the_models_with_their_lines() -> None:
    warn = pipeline.models_used_warning
    assert warn({"models_used": {"gemma4_12b": 812, "qwen35_9b": 140}}) == [{"kind": "mixed_models", "models": "gemma4_12b: 812, qwen35_9b: 140"}]
    assert warn({"models_used": {"qwen35_9b": 140, "gemma4_12b": 812, "same_lang": 3, "import": 9}})[0]["models"] == "gemma4_12b: 812, qwen35_9b: 140"
    assert warn({"models_used": {"gemma4_12b": 950, "same_lang": 5}}) == []  # one real model: nothing mixed
    assert warn({"models_used": {"import": 10, "same_lang": 5}}) == []
    assert warn({"models_used": {}}) == [] and warn({}) == [] and warn(None) == [] and warn({"models_used": "x"}) == []


def test_the_translate_flows_add_the_mixed_models_warning(client: Any, project: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    mixed = {"lines": 3, "failed": 0, "missing": 0, "backend": "x", "models_used": {"gemma4_12b": 2, "qwen35_9b": 1}}
    summaries: dict[str, Any] = {"translate": mixed, "text_emotion": {}}
    monkeypatch.setattr(pipeline, "run_stages", lambda ctx, job, names, regen=False: dict(summaries))
    r = jobs.runner()

    job = r.submit("translate", pipeline.FLOWS["translate"].build(SLUG, {}), project=SLUG)
    r.wait(job.id, 60)
    assert job.state == "done", job.error
    assert job.result["lines"] == 3
    assert job.result["warnings"] == [{"kind": "mixed_models", "models": "gemma4_12b: 2, qwen35_9b: 1"}]

    job = r.submit("stage", pipeline.FLOWS["stage"].build(SLUG, {"name": "translate"}), project=SLUG)  # the generic stage run, translate
    r.wait(job.id, 60)
    assert job.state == "done", job.error
    assert job.result["warnings"][0]["kind"] == "mixed_models"

    summaries["translate"] = {**mixed, "models_used": {"gemma4_12b": 3}}
    job = r.submit("translate", pipeline.FLOWS["translate"].build(SLUG, {}), project=SLUG)
    r.wait(job.id, 60)
    assert job.state == "done" and "warnings" not in job.result


# ----------------------------------------------------------------------------
# release 1.0: a resume keeps the caches, a hand-run job replaces the offer, one heavy job at a time, Cancel is a cancel
# ----------------------------------------------------------------------------


def test_a_resume_does_not_carry_force_over(client: Any, project: dict[str, Any], app_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``force`` clears the stage caches: Continue on a forced job must keep the work that was done, not redo everything."""
    stub_flows(monkeypatch, "import", "stage")
    restart(app_env, [record("imp", "import", 5, {"engine": "renpy", "force": True})])
    new = client.post("/api/project/resume").json()
    assert new["params"] == {"engine": "renpy"}
    assert wait_job(client, new)["state"] == "done"
    restart(app_env, [record("st", "stage", 5, {"name": "synth", "only": None, "limit": None, "force": True, "sets": None})])
    new = client.post("/api/project/resume").json()
    assert new["params"] == {"name": "synth", "only": None, "limit": None, "sets": None}
    assert wait_job(client, new)["state"] == "done"


def test_running_the_job_again_by_hand_withdraws_the_continue_offer(client: Any, project: dict[str, Any], app_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub_flows(monkeypatch, "full", "casting_listen")
    restart(app_env, [record("full1", "full", 30), record("listen1", "casting_listen", 2, {"speaker_id": "a"})])
    assert client.get("/api/project").json()["resume"]["job"]["id"] == "full1"
    job = pipeline.submit("full", SLUG, {"shutdown_when_done": False})  # RunScreen starts the run directly, not through Continue
    assert wait_job(client, job.id)["state"] == "done"
    resume = client.get("/api/project").json()["resume"]
    assert resume["job"]["id"] == "listen1"  # the old full run is no longer offered; the unrelated listen still is
    assert [j.id for j in jobs.runner().interrupted(SLUG)] == ["listen1"]


def test_two_clicks_on_a_heavy_flow_queue_one_job(client: Any, project: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """The busy check and the queueing are one step: a build that takes a moment lets the second click pass a check-then-submit."""
    gate = threading.Event()

    def slow_build(slug: str, p: dict[str, Any]) -> Any:
        import time

        time.sleep(0.15)
        return lambda ctx: gate.wait(20) and None

    monkeypatch.setitem(pipeline.FLOWS, "full", pipeline.Flow("full", slow_build, gpu=True, heavy=True))
    results: list[Any] = []
    barrier = threading.Barrier(2)

    def click() -> None:
        barrier.wait(5)
        try:
            results.append(pipeline.submit("full", SLUG, {"shutdown_when_done": False}))
        except Exception as e:  # noqa: BLE001
            results.append(e)

    ts = [threading.Thread(target=click) for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(15)
    gate.set()
    refused = [r for r in results if isinstance(r, Exception)]
    queued = [r for r in results if isinstance(r, jobs.JobRecord)]
    assert len(queued) == 1 and len(refused) == 1, results
    assert getattr(refused[0], "code", None) == "job_busy" and refused[0].status == 409  # type: ignore[attr-defined]
    assert len(jobs.runner().list(SLUG)) == 1


def test_a_light_duplicate_still_returns_the_running_job(client: Any, project: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    gate = threading.Event()
    monkeypatch.setitem(pipeline.FLOWS, "report", pipeline.Flow("report", lambda slug, p: (lambda ctx: gate.wait(20) and None), gpu=False))
    first = pipeline.submit("report", SLUG, {"slug": SLUG})
    again = pipeline.submit("report", SLUG, {"slug": SLUG})
    assert again is first and len(jobs.runner().list(SLUG)) == 1
    gate.set()


def _fake_review_pipeline(monkeypatch: pytest.MonkeyPatch, pack: Any) -> list[str]:
    """Replace everything review_apply touches except its own control flow; ``pack()`` is what the pack stage does."""
    from contextlib import contextmanager

    calls: list[str] = []

    @contextmanager
    def fake_ctx(h: Any, job: Any, **kw: Any) -> Any:
        yield object()

    def fake_stages(ctx: Any, job: Any, names: list[str], **kw: Any) -> dict[str, Any]:
        calls.append(",".join(names))
        if names == ["pack"]:
            pack()
        return {}

    state: dict[str, Any] = {"review": {"touched": ["a", "b", "z"]}}
    monkeypatch.setattr(pipeline, "_handle", lambda slug: types_ns(slug=slug))
    monkeypatch.setattr(pipeline, "pipeline_ctx", fake_ctx)
    monkeypatch.setattr(pipeline, "run_stages", fake_stages)
    monkeypatch.setattr(pipeline, "available_stages", lambda names: list(names))
    monkeypatch.setattr(pipeline, "preserve_status", lambda ctx, names: fake_ctx(None, None))
    monkeypatch.setattr(pipeline, "_final_state", lambda ctx, ids: {i: (None, None) for i in ids})
    monkeypatch.setattr(appdirs, "read_app_state", lambda h: state)
    monkeypatch.setattr(appdirs, "write_app_state", lambda h, st: calls.append("write_state"))
    monkeypatch.setitem(pipeline.FLOWS, "review_apply", pipeline.Flow("review_apply", pipeline._build_review_apply, gpu=True, heavy=True))  # noqa: SLF001
    return calls


def types_ns(**kw: Any) -> Any:
    import types

    return types.SimpleNamespace(**kw)


def test_cancel_during_the_pack_step_of_a_review_apply_is_a_cancel(client: Any, project: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    def cancelled_pack() -> None:
        raise JobCancelled("cancelled by user")

    calls = _fake_review_pipeline(monkeypatch, cancelled_pack)
    job = pipeline.submit("review_apply", SLUG, {"line_ids": ["a", "b"]})
    done = wait_job(client, job.id)
    assert done["state"] == "cancelled" and "pack_error" not in done["result"]
    assert "write_state" not in calls  # review.touched is not cleared: the lines still wait for an apply


def test_a_failing_pack_step_is_still_reported_as_pack_error_on_a_done_job(client: Any, project: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_pack() -> None:
        raise RuntimeError("zip failed")

    calls = _fake_review_pipeline(monkeypatch, broken_pack)
    job = pipeline.submit("review_apply", SLUG, {"line_ids": ["a", "b"]})
    done = wait_job(client, job.id)
    assert done["state"] == "done" and done["result"]["pack_error"]["code"] == "internal"
    assert "write_state" in calls

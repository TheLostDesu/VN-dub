"""ETA estimator, throughput store and the job runner (pause / cancel / interrupted on restart)."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from vndub_app import jobs
from vndub_app.errors import AppError
from vndub_app.jobs import EtaEstimator, JobCancelled, JobRunner, ThroughputStore


def wait_until(pred, timeout: float = 10.0, step: float = 0.02) -> None:
    t0 = time.monotonic()
    while not pred():
        if time.monotonic() - t0 > timeout:
            raise TimeoutError("condition not met")
        time.sleep(step)


# ----------------------------------------------------------------------------
# ETA
# ----------------------------------------------------------------------------


def test_eta_needs_measurements_then_converges() -> None:
    e = EtaEstimator(alpha=0.5, min_items=2)
    assert e.update(0, 10, now=0.0) is None
    assert e.update(1, 10, now=2.0) is None  # one item measured: not enough
    eta = e.update(2, 10, now=4.0)  # 2 s per item, 8 left
    assert eta == pytest.approx(16.0)
    assert e.rate_per_min == pytest.approx(30.0)
    # the EMA follows a change of speed
    e.update(4, 10, now=6.0)  # 1 s per item
    assert e.sec_per_item == pytest.approx(0.5 * 1.0 + 0.5 * 2.0)
    assert e.eta(10, 10) == 0.0


def test_eta_resets_when_progress_goes_backwards() -> None:
    e = EtaEstimator(min_items=1)
    e.update(0, 5, now=0.0)
    e.update(3, 5, now=3.0)
    assert e.eta(3, 5) == pytest.approx(2.0)
    e.update(0, 5, now=10.0)  # a new pass (regen round): the clock restarts, the average stays
    assert e._done_last == 0 and e._t_last == 10.0  # noqa: SLF001
    assert e.eta(0, 0) is None


def test_throughput_store_persists_and_estimates(tmp_path: Path) -> None:
    p = tmp_path / "throughput.json"
    ts = ThroughputStore(p)
    assert ts.estimate(["synth"], 10) is None
    ts.record("synth", 2.0)
    ts.record("score", 1.0)
    assert ts.estimate(["synth", "score"], 10) == pytest.approx(30.0)
    assert ts.estimate(["synth", "master"], 10) is None  # one stage was never measured
    ts.record("synth", 4.0)  # EMA 0.5
    assert ts.get("synth") == pytest.approx(3.0)
    again = ThroughputStore(p)
    assert again.get("synth") == pytest.approx(3.0) and again.get("score") == pytest.approx(1.0)


def test_job_progress_reports_eta_and_whole_job_estimate() -> None:
    r = JobRunner(None, None)
    r.throughput.record("master", 0.5)

    def fn(ctx: jobs.JobContext) -> dict:
        ctx.set_stages(["synth", "master"], n_items=10)
        ctx.begin_stage("synth")
        for i in range(1, 6):
            time.sleep(0.02)
            ctx.progress(i, 5, f"line{i}")
        return {"n": 5}

    job = r.submit("stage", fn)
    r.wait(job.id, 10)
    assert job.state == "done" and job.result == {"n": 5}
    assert job.stage == "synth" and job.stage_index == 1 and job.stage_count == 2
    assert job.done == 5 and job.total == 5 and job.item == "line5"
    assert job.eta_s == 0.0
    assert r.throughput.get("synth") is not None  # measured and stored for the next job's estimate
    r.shutdown()


# ----------------------------------------------------------------------------
# runner
# ----------------------------------------------------------------------------


def test_events_and_result() -> None:
    r = JobRunner(None, None)
    events: list[dict] = []
    r.subscribe(events.append)

    def fn(ctx: jobs.JobContext) -> dict:
        ctx.message("jobs.msg.report_zip", n=1)
        ctx.log("hello")
        ctx.set_result(a=1)
        return {"b": 2}

    job = r.submit("report", fn, gpu=False, project="p")
    r.wait(job.id, 10)
    assert job.state == "done" and job.result == {"a": 1, "b": 2} and job.project == "p"
    types = [e["type"] for e in events]
    assert "job" in types and {"type": "log", "job_id": job.id, "line": "hello"} in events
    assert events[-1]["job"]["state"] == "done"
    assert job.message_key == "jobs.msg.report_zip" and job.message_params == {"n": 1}
    assert r.get(job.id).info(with_log=True)["log_tail"] == ["hello"]
    r.shutdown()


def test_failure_becomes_an_app_error() -> None:
    r = JobRunner(None, None)

    def boom(ctx: jobs.JobContext) -> None:
        raise AppError("tool_missing", params={"tool": "ffmpeg"}, action="open_tools", details="x")

    def crash(ctx: jobs.JobContext) -> None:
        raise RuntimeError("something odd")

    a, b = r.submit("stage", boom), r.submit("stage", crash)
    r.wait(a.id, 10), r.wait(b.id, 10)
    assert a.state == "failed" and a.error == {"code": "tool_missing", "params": {"tool": "ffmpeg"}, "action": "open_tools", "details": "x"}
    assert b.state == "failed" and b.error["code"] == "internal" and b.error["action"] == "report_problem"
    assert "RuntimeError: something odd" in b.error["details"]
    r.shutdown()


def test_pause_blocks_progress_and_resume_continues() -> None:
    r = JobRunner(None, None)
    seen: list[int] = []

    def fn(ctx: jobs.JobContext) -> None:
        for i in range(1, 51):
            seen.append(i)
            ctx.progress(i, 50)
            time.sleep(0.01)

    job = r.submit("stage", fn)
    wait_until(lambda: len(seen) >= 3)
    r.pause(job.id)
    wait_until(lambda: job.state == "paused")
    n = len(seen)
    time.sleep(0.4)
    assert len(seen) <= n + 1  # the item in flight may finish, then it blocks in progress()
    frozen = len(seen)
    time.sleep(0.3)
    assert len(seen) == frozen and job.state == "paused"
    r.resume(job.id)
    r.wait(job.id, 10)
    assert job.state == "done" and len(seen) == 50


def test_cancel_running_and_queued() -> None:
    r = JobRunner(None, None)
    started = threading.Event()
    proceed = threading.Event()

    def long(ctx: jobs.JobContext) -> None:
        started.set()
        i = 0
        while True:
            i += 1
            ctx.progress(i, 10**6)
            time.sleep(0.01)

    def never(ctx: jobs.JobContext) -> None:
        proceed.set()

    j1 = r.submit("stage", long)
    j2 = r.submit("stage", never)  # queued behind j1 (one GPU job at a time)
    started.wait(5)
    assert j2.state == "queued"
    r.cancel(j2.id)
    assert j2.state == "cancelled"
    r.cancel(j1.id)
    r.wait(j1.id, 10)
    assert j1.state == "cancelled" and j1.finished
    time.sleep(0.2)
    assert not proceed.is_set()  # the cancelled queued job never ran
    r.shutdown()


def test_cancel_while_paused_raises_job_cancelled() -> None:
    r = JobRunner(None, None)
    got: list[str] = []

    def fn(ctx: jobs.JobContext) -> None:
        try:
            for i in range(1000):
                ctx.progress(i, 1000)
                time.sleep(0.01)
        except JobCancelled:
            got.append("cancelled")
            raise

    job = r.submit("stage", fn)
    wait_until(lambda: job.done > 2)
    r.pause(job.id)
    wait_until(lambda: job.state == "paused")
    r.cancel(job.id)
    r.wait(job.id, 10)
    assert job.state == "cancelled" and got == ["cancelled"]
    r.shutdown()


def test_gpu_jobs_run_one_at_a_time_light_jobs_in_parallel() -> None:
    r = JobRunner(None, None)
    lock = threading.Lock()
    running = 0
    peak = 0

    def gpu_job(ctx: jobs.JobContext) -> None:
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.15)
        with lock:
            running -= 1

    js = [r.submit("stage", gpu_job) for _ in range(3)]
    for j in js:
        r.wait(j.id, 10)
    assert peak == 1 and all(j.state == "done" for j in js)

    gate = threading.Event()
    light = [r.submit("report", lambda ctx: gate.wait(5) and None, gpu=False) for _ in range(2)]
    wait_until(lambda: all(j.state == "running" for j in light))  # both run at once
    gate.set()
    for j in light:
        r.wait(j.id, 10)
    r.shutdown()


def test_interrupted_on_restart_and_forget(tmp_path: Path) -> None:
    state = tmp_path / "jobs.json"
    r1 = JobRunner(state, None)
    block = threading.Event()
    started = threading.Event()

    def fn(ctx: jobs.JobContext) -> None:
        ctx.set_stages(["synth", "score"])
        ctx.begin_stage("synth")
        started.set()
        block.wait(10)

    running = r1.submit("preview", fn, params={"line_ids": ["a", "b"]}, project="demo")
    queued = r1.submit("full", lambda ctx: None, params={"shutdown_when_done": False}, project="demo")
    done = r1.submit("report", lambda ctx: None, gpu=False, project="demo")
    r1.wait(done.id, 10)
    started.wait(5)
    r1._save(force=True)  # noqa: SLF001  (what the server does at shutdown; the app is "killed" here)
    # a new process: the persisted active jobs come back as interrupted, finished ones stay as they were
    r2 = JobRunner(state, None)
    assert r2.get(running.id).state == "interrupted" and r2.get(queued.id).state == "interrupted"
    assert r2.get(done.id).state == "done"
    assert r2.get(running.id).params == {"line_ids": ["a", "b"]} and r2.get(running.id).stage == "synth"
    assert [j.id for j in r2.interrupted("demo")] == sorted([running.id, queued.id], key=lambda i: r2.get(i).created)
    assert r2.interrupted("other") == [] and r2.active("demo") is None
    r2.forget(running.id)
    assert [j.id for j in r2.interrupted("demo")] == [queued.id]
    assert JobRunner(state, None).list("demo")[0].id != running.id  # forgetting is persisted
    block.set()
    r1.shutdown()
    r2.shutdown()


def test_reset_runner_replaces_the_process_wide_runner() -> None:
    first = JobRunner(None, None)
    jobs.reset_runner(first)
    assert jobs.runner() is first
    second = JobRunner(None, None)
    jobs.reset_runner(second)
    assert jobs.runner() is second
    jobs.reset_runner(None)


def test_job_not_found_error() -> None:
    r = JobRunner(None, None)
    with pytest.raises(AppError) as e:
        r.get("nope")
    assert e.value.code == "job_not_found" and e.value.status == 404
    r.shutdown()

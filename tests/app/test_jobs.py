"""ETA estimator, throughput store and the job runner (pause / cancel / interrupted on restart)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from vndub_app import jobs, winproc
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


def test_eta_needs_measurements_then_averages() -> None:
    e = EtaEstimator(min_items=2)
    assert e.update(0, 10, now=0.0) is None
    assert e.update(1, 10, now=2.0) is None  # one item measured: not enough
    eta = e.update(2, 10, now=4.0)  # 2 s per item, 8 left
    assert eta == pytest.approx(16.0)
    assert e.rate_per_min == pytest.approx(30.0)
    e.update(4, 10, now=6.0)  # faster for a moment: the average since the start is 6 s / 4
    assert e.sec_per_item == pytest.approx(1.5)
    assert e.eta(10, 10) == 0.0


def test_eta_counts_the_wait_before_the_first_batch() -> None:
    e = EtaEstimator()
    e.reset(now=0.0)  # the stage began; nothing reported while the model loads and thinks
    e.update(16, 100, now=300.0)  # 5 minutes, then a batch of 16 lines
    assert e.rate_per_min == pytest.approx(3.2)
    assert e.eta(16, 100) == pytest.approx(84 * 300 / 16)


def test_eta_batches_landing_together_do_not_look_fast() -> None:
    e = EtaEstimator()
    e.reset(now=0.0)
    e.update(16, 64, now=300.0)  # two scenes in parallel finish their batches a second apart
    e.update(32, 64, now=301.0)
    assert e.rate_per_min == pytest.approx(32 / 301 * 60)


def test_eta_new_pass_keeps_the_clock() -> None:
    e = EtaEstimator(min_items=1)
    e.update(0, 5, now=0.0)
    e.update(3, 5, now=3.0)
    assert e.eta(3, 5) == pytest.approx(2.0)
    e.update(0, 5, now=10.0)  # a new pass (regen round): the count restarts, the 7 s between were work
    e.update(1, 5, now=11.0)
    assert e.sec_per_item == pytest.approx(11.0 / 4)
    assert e.eta(0, 0) is None


def test_eta_ignores_cached_scan_and_pauses() -> None:
    e = EtaEstimator(min_items=2)
    e.reset(now=0.0)
    e.update(60, 100, now=0.05)  # 60 cached results found at once: not throughput
    assert e.sec_per_item is None and e.eta(60, 100) is None
    e.update(61, 100, now=2.0)  # the scan's 0.05 s goes to this item
    e.update(62, 100, now=4.0)
    assert e.eta(62, 100) == pytest.approx(76.0)
    e.rebase(now=1000.0)  # paused for a long time
    e.update(63, 100, now=1002.0)
    assert e.sec_per_item == pytest.approx(2.0)


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


def test_job_rate_is_lines_not_candidates() -> None:
    r = JobRunner(None, None)
    seen: list[float | None] = []

    def fn(ctx: jobs.JobContext) -> None:
        ctx.set_stages(["synth"], n_items=5)  # 5 lines, 2 candidates each
        ctx.begin_stage("synth")
        ctx.eta.reset(now=time.monotonic() - 60.0)  # the stage has worked a minute
        ctx.progress(4, 10, "c4")
        seen.append(ctx.job.rate_per_min)

    job = r.submit("stage", fn)
    r.wait(job.id, 10)
    assert job.state == "done"
    assert seen[0] == pytest.approx(2.0, rel=0.05)  # 4 candidates a minute = 2 lines a minute
    r.shutdown()


def test_job_progress_reports_eta_and_whole_job_estimate() -> None:
    r = JobRunner(None, None)
    r.throughput.record("master", 0.5)

    def fn(ctx: jobs.JobContext) -> dict:
        ctx.set_stages(["synth", "master"], n_items=10)
        ctx.begin_stage("synth")
        for i in range(1, 6):
            time.sleep(0.07)  # slower than a cached result is found (EtaEstimator.INSTANT_S_PER_ITEM)
            ctx.progress(i, 5, f"line{i}")
        return {"n": 5}

    job = r.submit("stage", fn)
    r.wait(job.id, 10)
    assert job.state == "done" and job.result == {"n": 5}
    assert job.stage == "synth" and job.stage_index == 1 and job.stage_count == 2
    assert job.done == 5 and job.total == 5 and job.item == "line5"
    assert job.eta_s == 0.0
    # measured and stored per voiced line (5 progress items of 10 lines), for the next job's estimate
    assert r.throughput.get("synth") is not None
    assert r.throughput.get("synth") == pytest.approx(0.07 * 5 / 10, abs=0.03)
    r.shutdown()


def test_eta_cached_items_reported_one_by_one_are_not_work() -> None:
    """After a restart a stage reports every cached item on its own: 1000 of them in a second are a scan, not 1000 lines a second."""
    e = EtaEstimator(min_items=2)
    e.reset(now=0.0)
    for i in range(1, 1001):
        e.update(i, 2000, now=i * 0.001)  # 1 ms each, well under 50 ms
    assert e.measured == 0 and e.sec_per_item is None and e.rate_per_min is None and e.eta(1000, 2000) is None
    e.update(1001, 2000, now=4.0)  # the first real item: the scan's second goes to it
    e.update(1002, 2000, now=7.0)
    assert e.measured == 2 and e.sec_per_item == pytest.approx(3.5)
    assert e.rate_per_min == pytest.approx(60 / 3.5)


def test_eta_a_slow_cached_scan_is_still_cached() -> None:
    e = EtaEstimator(min_items=2)
    e.reset(now=0.0)
    e.update(2000, 5000, now=30.0)  # 30 s over 2000 cached items = 15 ms each
    assert e.measured == 0 and e.rate_per_min is None
    e.update(2001, 5000, now=33.0)  # a real item: 3 s
    e.update(2002, 5000, now=36.0)
    assert e.measured == 2 and e.sec_per_item == pytest.approx((33 + 3) / 2)  # the scan's time goes to the next real item
    # the boundary: under 50 ms per item is a cached result, over it is work
    w = EtaEstimator(min_items=1)
    w.reset(now=0.0)
    w.update(10, 100, now=0.49)  # 49 ms each
    assert w.measured == 0
    w.update(20, 100, now=1.5)  # 1.01 s over 10 items
    assert w.measured == 10


def test_throughput_store_clamps_a_new_value_to_a_third_up_or_down(tmp_path: Path) -> None:
    ts = ThroughputStore(tmp_path / "t.json")
    ts.record("synth", 1.0)  # nothing known yet: taken as it is
    assert ts.get("synth") == pytest.approx(1.0)
    ts.record("synth", 100.0)  # clamped to 3.0, then the EMA
    assert ts.get("synth") == pytest.approx(0.5 * 3.0 + 0.5 * 1.0)
    ts.record("score", 6.0)
    ts.record("score", 0.01)  # clamped to 2.0
    assert ts.get("score") == pytest.approx(0.5 * 2.0 + 0.5 * 6.0)


def _run_stage_job(r: JobRunner, n_hint: int, plan: list[tuple[str, int, float]]) -> jobs.JobRecord:
    """A job that enters the given stages in order: ``(name, items, seconds the stage has been working)``."""

    def fn(ctx: jobs.JobContext) -> None:
        ctx.set_stages(list(dict.fromkeys(n for n, _, _ in plan)), n_items=n_hint)
        for name, items, secs in plan:
            ctx.begin_stage(name)
            ctx.eta.reset(now=time.monotonic() - secs)  # the stage has worked this long
            ctx.progress(items, items, "x")

    job = r.submit("stage", fn)
    r.wait(job.id, 10)
    assert job.state == "done", job.error
    return job


def test_a_stage_entered_again_is_not_recorded() -> None:
    r = JobRunner(None, None)
    _run_stage_job(r, 30, [("synth", 30, 60.0), ("score", 30, 30.0)])
    assert r.throughput.get("synth") == pytest.approx(2.0, rel=0.05)  # 60 s / 30 items, 30 lines
    assert r.throughput.get("score") == pytest.approx(1.0, rel=0.05)
    # a regen round: synth and score again with a few items and a very different speed
    _run_stage_job(r, 30, [("synth", 30, 60.0), ("score", 30, 30.0), ("synth", 25, 150.0), ("score", 25, 100.0)])
    assert r.throughput.get("synth") == pytest.approx(2.0, rel=0.05)  # the second entry did not count
    assert r.throughput.get("score") == pytest.approx(1.0, rel=0.05)
    r.shutdown()


def test_a_small_job_or_mostly_cached_stage_is_not_recorded() -> None:
    r = JobRunner(None, None)
    _run_stage_job(r, 5, [("synth", 5, 60.0)])  # a 5-line listen job carries the model's load time
    assert r.throughput.get("synth") is None

    def mostly_cached(ctx: jobs.JobContext) -> None:
        ctx.set_stages(["score"], n_items=100)
        ctx.begin_stage("score")
        t0 = time.monotonic()
        ctx.eta.reset(now=t0)
        ctx.progress(190, 200)  # 190 cached results found at once
        ctx.eta.update(200, 200, now=t0 + 100.0)  # ... and 10 real ones in the next 100 s: under 20 items and under 20 % of 200

    job = r.submit("stage", mostly_cached)
    r.wait(job.id, 10)
    assert job.state == "done" and r.throughput.get("score") is None

    _run_stage_job(r, 100, [("score", 25, 100.0)])  # 25 real items of 25: recorded
    assert r.throughput.get("score") == pytest.approx(100 / 25 * 25 / 100, rel=0.05)
    r.shutdown()


def test_a_short_stage_with_all_its_items_measured_is_recorded() -> None:
    r = JobRunner(None, None)
    _run_stage_job(r, 12, [("pack", 3, 9.0)])  # 3 of 3 items: 100 % of its total although under 20 items
    assert r.throughput.get("pack") == pytest.approx(9.0 / 3 * 3 / 12, rel=0.05)
    r.shutdown()


def test_the_runner_keeps_throughput_in_a_v3_file(app_env: Path) -> None:
    (app_env / "throughput_v2.json").write_text(json.dumps({"synth": 0.001, "score": 0.002}), encoding="utf-8")  # poisoned by earlier resumes
    jobs.reset_runner(None)
    r = jobs.runner()
    try:
        assert r.throughput.get("synth") is None  # the v2 numbers are not read
        r.throughput.record("synth", 2.0)
        assert json.loads((app_env / "throughput_v3.json").read_text(encoding="utf-8")) == {"synth": 2.0}
        assert json.loads((app_env / "throughput_v2.json").read_text(encoding="utf-8")) == {"synth": 0.001, "score": 0.002}  # left alone
    finally:
        jobs.reset_runner(None)


def test_old_records_without_settings_load_and_new_ones_keep_them(tmp_path: Path) -> None:
    state = tmp_path / "jobs.json"
    old = jobs.JobRecord(id="old1", kind="full", title_key="jobs.full", project="demo", state="done", created="2026-10-01T10:00:00+00:00").info(with_log=True)
    del old["settings"]  # a record written before the field existed
    state.write_text(json.dumps([old]), encoding="utf-8")
    r1 = JobRunner(state, None)
    assert r1.get("old1").settings == {} and r1.get("old1").info()["settings"] == {}
    snap = {"gpu_budget_gb": 7.5, "profile": "apptest"}
    j = r1.submit("report", lambda ctx: None, gpu=False, project="demo", settings=snap)
    r1.wait(j.id, 10)
    r1._save(force=True)  # noqa: SLF001
    assert JobRunner(state, None).get(j.id).settings == snap and j.info()["settings"] == snap
    r1.shutdown()


def test_interrupted_records_older_than_two_weeks_are_forgotten_at_load(tmp_path: Path) -> None:
    from datetime import datetime, timedelta, timezone

    def ago(days: float) -> str:
        return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")

    def rec(i: str, state: str, created: str, finished: str | None) -> dict:
        return jobs.JobRecord(id=i, kind="full", title_key="jobs.full", project="demo", state=state, created=created, finished=finished).info(with_log=True)

    state = tmp_path / "jobs.json"
    state.write_text(json.dumps([
        rec("stale", "interrupted", ago(40), ago(15)),  # interrupted 15 days ago: gone
        rec("stale_no_finish", "interrupted", ago(20), None),  # no finish time: judged by created
        rec("fresh", "interrupted", ago(40), ago(13)),
        rec("old_done", "done", ago(60), ago(60)),  # only interrupted records expire
        rec("was_running", "running", ago(30), None),  # active when the app died: interrupted NOW, so it is fresh
    ]), encoding="utf-8")
    r = JobRunner(state, None)
    assert {j.id for j in r.list()} == {"fresh", "old_done", "was_running"}
    assert r.get("was_running").state == "interrupted"
    assert {j["id"] for j in json.loads(state.read_text(encoding="utf-8"))} == {"fresh", "old_done", "was_running"}  # forgetting is persisted
    r.shutdown()


def test_add_warnings_appends_without_duplicates() -> None:
    r = JobRunner(None, None)
    seen: list[int] = []
    w = {"kind": "unavailable", "stage": "score", "group": "cer", "reason": "r"}

    def fn(ctx: jobs.JobContext) -> dict:
        ctx.set_result(warnings=[{"kind": "settings_changed", "keys": "gpu_budget_gb"}])
        seen.append(ctx.add_warnings([w, w, {"kind": "settings_changed", "keys": "gpu_budget_gb"}]))
        seen.append(ctx.add_warnings([w]))
        return {"n": 1}

    job = r.submit("stage", fn)
    r.wait(job.id, 10)
    assert seen == [1, 0]
    assert job.result["warnings"] == [{"kind": "settings_changed", "keys": "gpu_budget_gb"}, w]
    assert job.result["n"] == 1  # the flow's own result went in beside them
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
    logs = [e for e in events if e["type"] == "log"]
    assert "job" in types and len(logs) == 1 and logs[0]["job_id"] == job.id and set(logs[0]) == {"type", "job_id", "line"}
    assert re.fullmatch(r"\d\d:\d\d:\d\d hello", logs[0]["line"])  # every line carries the local time
    assert events[-1]["job"]["state"] == "done"
    assert job.message_key == "jobs.msg.report_zip" and job.message_params == {"n": 1}
    assert r.get(job.id).info(with_log=True)["log_tail"] == [logs[0]["line"]]  # the tail is prefixed the same way
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


# ----------------------------------------------------------------------------
# JobContext.watch_process: cancelling a job ends the processes it started
# ----------------------------------------------------------------------------

SLEEPER = [sys.executable, "-c", "import time; time.sleep(60)"]


def pid_alive(pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    import ctypes

    k32 = ctypes.windll.kernel32
    k32.OpenProcess.restype = ctypes.c_void_p
    h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    k32.GetExitCodeProcess(ctypes.c_void_p(h), ctypes.byref(code))
    k32.CloseHandle(ctypes.c_void_p(h))
    return code.value == 259  # STILL_ACTIVE


def test_cancel_kills_a_watched_process_at_once() -> None:
    r = JobRunner(None, None)
    started = threading.Event()
    seen: dict = {}

    def fn(ctx: jobs.JobContext) -> None:
        proc = subprocess.Popen(SLEEPER)
        seen["proc"] = proc
        unwatch = ctx.watch_process(proc)
        started.set()
        try:
            proc.wait()  # what installer.run_stream does: block on the child (no output = no line to notice a cancel on)
        finally:
            unwatch()
        ctx.checkpoint()  # raises JobCancelled: the child did not end by itself

    job = r.submit("install", fn, gpu=False)
    try:
        assert started.wait(10)
        t0 = time.monotonic()
        r.cancel(job.id)
        r.wait(job.id, 10)
        assert job.state == "cancelled" and time.monotonic() - t0 < 5
        assert seen["proc"].poll() is not None  # the sleeper (60 s) is gone, not waiting for its next line of output
    finally:
        if "proc" in seen and seen["proc"].poll() is None:
            seen["proc"].kill()
        r.shutdown()


@pytest.mark.skipif(os.name != "nt", reason="the tree walk is taskkill /T; elsewhere only process groups are killed")
def test_cancel_kills_the_whole_process_tree() -> None:
    """PowerShell -> uv -> python: the grandchild must go too, or a torch download keeps running after Cancel."""
    r = JobRunner(None, None)
    started = threading.Event()
    seen: dict = {}
    parent_code = (
        "import subprocess, sys, time\n"
        "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "print(g.pid, flush=True)\n"
        "time.sleep(60)\n"
    )

    def fn(ctx: jobs.JobContext) -> None:
        proc = subprocess.Popen([sys.executable, "-c", parent_code], stdout=subprocess.PIPE, text=True)
        seen["proc"] = proc
        ctx.watch_process(proc)
        seen["grandchild"] = int(proc.stdout.readline())  # type: ignore[union-attr]
        started.set()
        proc.wait()
        ctx.checkpoint()

    job = r.submit("install", fn, gpu=False)
    try:
        assert started.wait(15)
        assert pid_alive(seen["grandchild"])
        r.cancel(job.id)
        r.wait(job.id, 10)
        assert job.state == "cancelled"
        wait_until(lambda: not pid_alive(seen["grandchild"]), timeout=5)
    finally:
        if "grandchild" in seen:
            winproc.kill_tree(seen["grandchild"])
        if "proc" in seen:
            winproc.kill_tree(seen["proc"])
            seen["proc"].stdout.close()
        r.shutdown()


def test_a_process_registered_after_the_cancel_is_killed_at_once() -> None:
    r = JobRunner(None, None)
    seen: dict = {}
    reached = threading.Event()

    def fn(ctx: jobs.JobContext) -> None:
        while not ctx.cancelled:  # an install step that only looks at the cancel flag between steps
            time.sleep(0.02)
        proc = subprocess.Popen(SLEEPER)
        seen["proc"] = proc
        t0 = time.monotonic()
        ctx.watch_process(proc)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        seen["waited"] = time.monotonic() - t0
        reached.set()

    job = r.submit("install", fn, gpu=False)
    try:
        wait_until(lambda: job.state == "running")
        r.cancel(job.id)
        assert reached.wait(15)
        assert seen["proc"].poll() is not None and seen["waited"] < 5  # killed by watch_process itself, not by the 10 s wait
        r.wait(job.id, 10)
    finally:
        if "proc" in seen and seen["proc"].poll() is None:
            seen["proc"].kill()
        r.shutdown()


def test_unwatch_removes_the_process_and_watching_twice_is_harmless() -> None:
    r = JobRunner(None, None)
    seen: dict = {}
    ready = threading.Event()

    def fn(ctx: jobs.JobContext) -> None:
        proc = subprocess.Popen(SLEEPER)
        seen["proc"] = proc
        unwatch = ctx.watch_process(proc)
        ctx.watch_process(proc)  # the same process again: still one entry
        seen["entries"] = len(ctx._procs)  # noqa: SLF001
        unwatch()
        unwatch()  # and unwatching twice is fine too
        seen["after"] = len(ctx._procs)  # noqa: SLF001
        ready.set()
        while True:
            ctx.checkpoint()
            time.sleep(0.02)

    job = r.submit("install", fn, gpu=False)
    try:
        assert ready.wait(10)
        assert seen["entries"] == 1 and seen["after"] == 0
        r.cancel(job.id)
        r.wait(job.id, 10)
        assert job.state == "cancelled"
        time.sleep(0.5)
        assert seen["proc"].poll() is None  # no longer watched: the cancel did not touch it
    finally:
        if "proc" in seen:
            winproc.kill_tree(seen["proc"])
            seen["proc"].wait(10)
        r.shutdown()


def test_cancel_runs_hooks_and_a_failure_after_it_is_a_cancel(tmp_path):
    r = JobRunner(tmp_path / "jobs.json", tmp_path / "t.json")
    hooked = threading.Event()
    started = threading.Event()

    def fn(ctx):
        ctx.on_cancel(hooked.set)
        started.set()
        hooked.wait(10)  # a blocking call that only the hook can end
        raise RuntimeError("backend process killed")  # what the broken call raises

    job = r.submit("stage", fn)
    assert started.wait(10)
    r.cancel(job.id)
    rec = r.wait(job.id, 10)
    assert hooked.is_set()
    assert rec.state == "cancelled" and rec.error is None
    assert rec.cancel_requested


# ----------------------------------------------------------------------------
# logging: time prefix, per-job log files, events in app.log
# ----------------------------------------------------------------------------

TIME = r"\d\d:\d\d:\d\d "


def _texts(lines: list[str]) -> list[str]:
    """Job log lines without their ``HH:MM:SS `` prefix."""
    assert all(re.match(TIME, l) for l in lines), lines
    return [l[9:] for l in lines]


def test_every_log_line_starts_with_the_local_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jobs, "_stamp", lambda: "12:34:56")
    r = JobRunner(None, None)
    events: list[dict] = []
    r.subscribe(events.append)

    def fn(ctx: jobs.JobContext) -> None:
        ctx.log("INFO vndub.link: a forwarded pipeline record")
        ctx.log("plain")
        raise RuntimeError("boom")

    job = r.submit("stage", fn)
    r.wait(job.id, 10)
    assert job.log_tail == [
        "12:34:56 INFO vndub.link: a forwarded pipeline record",
        "12:34:56 plain",
        "12:34:56 ERROR: RuntimeError: boom",  # the failure line goes through the same path
    ]
    streamed = [e for e in events if e["type"] == "log"]
    assert [e["line"] for e in streamed] == job.log_tail and all(set(e) == {"type", "job_id", "line"} and e["job_id"] == job.id for e in streamed)
    assert r.get(job.id).info(with_log=True)["log_tail"] == job.log_tail
    r.shutdown()


def test_the_real_clock_is_used_by_default() -> None:
    before = time.strftime("%H:%M:%S")
    got = jobs._stamp()  # noqa: SLF001
    after = time.strftime("%H:%M:%S")
    assert re.fullmatch(r"\d\d:\d\d:\d\d", got) and got[:2] in {before[:2], after[:2]}


def test_job_log_file_has_header_lines_events_and_footer(tmp_path: Path) -> None:
    logs = tmp_path / "logs" / "jobs"
    r = JobRunner(None, None, log_dir=logs)

    def fn(ctx: jobs.JobContext) -> None:
        ctx.set_stages(["synth", "score"])
        ctx.begin_stage("synth")
        ctx.log("first")
        ctx.progress(2, 2)
        ctx.begin_stage("score")
        ctx.log("second")

    job = r.submit("stage", fn, params={"name": "synth", "hf_token": "secret-value", "only": ["a"]}, project="demo")
    r.wait(job.id, 10)
    assert job.state == "done"
    files = list(logs.glob("*.log"))
    assert [str(f) for f in files] == [job.log_path]
    assert re.fullmatch(r"\d{8}-\d{6}-stage-" + job.id + r"\.log", files[0].name)
    raw = files[0].read_text(encoding="utf-8")
    assert "secret-value" not in raw
    lines = raw.splitlines()
    body = _texts(lines)
    assert body[0].startswith(f"=== job {job.id} kind=stage project=demo created={job.created} params=")
    assert json.loads(body[0].split(" params=", 1)[1]) == {"name": "synth", "hf_token": "***", "only": ["a"]}
    assert body[1:3] == ["--- stage synth begin (1/2)", "first"]
    assert re.fullmatch(r"--- stage synth done in \d+\.\ds \(2/2\)", body[3])
    assert body[4:6] == ["--- stage score begin (2/2)", "second"]
    assert re.fullmatch(r"--- stage score done in \d+\.\ds \(0/0\)", body[6])
    assert re.fullmatch(r"=== done after \d+\.\ds", body[7]) and len(body) == 8
    # the UI log shows what the job printed, not the events: the file is the fuller record
    assert _texts(job.log_tail) == ["first", "second"]
    r.shutdown()


def test_job_log_footer_names_the_state_and_the_error_code(tmp_path: Path) -> None:
    logs = tmp_path / "jobs"
    r = JobRunner(None, None, log_dir=logs)

    def boom(ctx: jobs.JobContext) -> None:
        ctx.set_stages(["synth"])
        ctx.begin_stage("synth")
        raise AppError("tool_missing", params={"tool": "ffmpeg"}, details="ffmpeg is gone\nsecond line")

    started = threading.Event()

    def long(ctx: jobs.JobContext) -> None:
        started.set()
        while True:
            ctx.progress(1, 10)
            time.sleep(0.01)

    bad = r.submit("stage", boom)
    r.wait(bad.id, 10)
    body = _texts(Path(bad.log_path).read_text(encoding="utf-8").splitlines())
    assert re.fullmatch(r"--- stage synth failed in \d+\.\ds \(0/0\)", body[-3]) and body[-2] == "ERROR: ffmpeg is gone"
    assert re.fullmatch(r"=== failed after \d+\.\ds error=tool_missing", body[-1])

    ok = r.submit("stage", long)
    started.wait(5)
    r.cancel(ok.id)
    r.wait(ok.id, 10)
    body = _texts(Path(ok.log_path).read_text(encoding="utf-8").splitlines())
    assert "=== cancel requested" in body and re.fullmatch(r"=== cancelled after \d+\.\ds", body[-1])
    r.shutdown()


def test_job_log_file_is_written_while_the_job_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jobs, "JOB_LOG_FLUSH_S", 0.05)
    r = JobRunner(None, None, log_dir=tmp_path / "jobs")
    gate, logged = threading.Event(), threading.Event()

    def fn(ctx: jobs.JobContext) -> None:
        ctx.log("early line")
        logged.set()
        gate.wait(10)

    job = r.submit("report", fn, gpu=False)
    try:
        assert logged.wait(5)
        # buffered, not written per line: the flusher thread puts it on disk within a moment, the job still running
        wait_until(lambda: "early line" in Path(job.log_path).read_text(encoding="utf-8"), timeout=5)
        assert job.state == "running"
    finally:
        gate.set()
        r.wait(job.id, 10)
        r.shutdown()


def test_many_lines_are_buffered_not_written_one_by_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    flushes: list[int] = []
    real = jobs._JobLogFile.flush  # noqa: SLF001
    monkeypatch.setattr(jobs._JobLogFile, "flush", lambda self: (flushes.append(1), real(self))[1])  # noqa: SLF001
    r = JobRunner(None, None, log_dir=tmp_path / "jobs")

    def fn(ctx: jobs.JobContext) -> None:
        for i in range(5000):  # what `uv pip install` prints
            ctx.log(f"line {i}")

    t0 = time.monotonic()
    job = r.submit("report", fn, gpu=False)
    r.wait(job.id, 30)
    assert job.state == "done" and time.monotonic() - t0 < 10
    assert len(flushes) < 50  # nowhere near one flush per line
    body = _texts(Path(job.log_path).read_text(encoding="utf-8").splitlines())
    assert body[1] == "line 0" and body[5000] == "line 4999" and len(body) == 5002  # header + 5000 lines + footer
    r.shutdown()


def test_old_job_logs_are_pruned_to_the_newest_thirty(tmp_path: Path) -> None:
    logs = tmp_path / "jobs"
    logs.mkdir()
    for i in range(35):
        (logs / f"20200101-0000{i:02d}-stage-old{i:02d}.log").write_text("x", encoding="utf-8")
    r = JobRunner(None, None, log_dir=logs)
    ids = []
    for _ in range(3):
        job = r.submit("stage", lambda ctx: None)
        r.wait(job.id, 10)
        ids.append(job.id)
    names = sorted(p.name for p in logs.glob("*.log"))
    assert len(names) == 30
    assert not any("old0" in n and int(n.split("old")[1][:2]) < 8 for n in names)  # the 8 oldest of the 38 are gone
    assert any("old34" in n for n in names) and all(any(i in n for n in names) for i in ids)
    r.shutdown()


def test_a_broken_log_folder_never_breaks_the_job(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    blocker = tmp_path / "jobs"
    blocker.write_text("a file where the folder should be", encoding="utf-8")
    r = JobRunner(None, None, log_dir=blocker / "sub")
    caplog.set_level("WARNING", logger="vndub_app.jobs")

    def fn(ctx: jobs.JobContext) -> dict:
        for i in range(3):
            ctx.log(f"line {i}")
        return {"ok": 1}

    job = r.submit("report", fn, gpu=False)
    r.wait(job.id, 10)
    assert job.state == "done" and job.result == {"ok": 1} and job.log_path is None
    assert _texts(job.log_tail) == ["line 0", "line 1", "line 2"]
    assert len([rec for rec in caplog.records if "cannot write its log file" in rec.getMessage()]) == 1
    assert r.read_log(job.id) == {"path": None, "lines": job.log_tail}
    r.shutdown()


def test_a_write_error_in_the_middle_warns_once_and_stops_writing(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("WARNING", logger="vndub_app.jobs")
    f = jobs._JobLogFile(tmp_path / "a.log", "j1")  # noqa: SLF001
    assert f.open()

    class Full:
        def write(self, _: str) -> None:
            raise OSError("disk full")

        def close(self) -> None:
            pass

    f._f.close()  # type: ignore[union-attr]  # noqa: SLF001
    f._f = Full()  # type: ignore[assignment]  # noqa: SLF001
    for _ in range(3):
        f.write("a")
    f.flush()
    f.close()
    assert len([r for r in caplog.records if "disk full" in r.getMessage()]) == 1


def test_read_log_returns_the_file_the_newest_part_of_a_big_one_or_the_tail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r = JobRunner(None, None, log_dir=tmp_path / "jobs")

    def fn(ctx: jobs.JobContext) -> None:
        for i in range(60):
            ctx.log(f"line {i:03d}")

    job = r.submit("report", fn, gpu=False)
    r.wait(job.id, 10)
    full = r.read_log(job.id)
    assert full["path"] == job.log_path and len(full["lines"]) == 62
    assert _texts(full["lines"])[1:-1] == _texts(job.log_tail)
    monkeypatch.setattr(jobs, "JOB_LOG_READ_MAX", 300)  # the cap: the newest lines win, the cut line is dropped
    cut = r.read_log(job.id)["lines"]
    assert 3 <= len(cut) < 62 and cut[-1] == full["lines"][-1] and all(re.match(TIME, l) for l in cut)
    assert cut == full["lines"][-len(cut):]
    Path(job.log_path).unlink()  # pruned or removed: what the record still has
    gone = r.read_log(job.id)
    assert gone == {"path": None, "lines": job.log_tail}
    with pytest.raises(AppError):
        r.read_log("nope")
    r.shutdown()


def test_job_events_go_to_the_app_log(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger="vndub_app.jobs")
    r = JobRunner(None, None)
    gate = threading.Event()

    def ok(ctx: jobs.JobContext) -> None:
        ctx.set_stages(["a", "b"])
        ctx.begin_stage("a")
        ctx.progress(3, 3)
        ctx.begin_stage("b")

    def paused(ctx: jobs.JobContext) -> None:
        gate.wait(10)
        ctx.checkpoint()
        raise AppError("tool_missing", details="ffmpeg is gone", action="open_tools")

    j1 = r.submit("stage", ok, project="demo", params={"name": "a", "only": None})
    r.wait(j1.id, 10)
    j2 = r.submit("stage", paused)
    wait_until(lambda: j2.state == "running")
    r.pause(j2.id)
    gate.set()
    wait_until(lambda: j2.state == "paused")
    r.resume(j2.id)
    r.wait(j2.id, 10)
    gate2 = threading.Event()
    block = r.submit("stage", lambda ctx: gate2.wait(10) and None)
    wait_until(lambda: block.state == "running")
    j3 = r.submit("stage", lambda ctx: None)  # queued behind the blocker: cancelled before it ever starts
    r.cancel(j3.id)
    r.wait(j3.id, 10)
    gate2.set()
    r.wait(block.id, 10)
    msgs = [rec.getMessage() for rec in caplog.records if rec.name == "vndub_app.jobs"]
    assert f"job {j1.id} submitted: kind=stage project=demo gpu=True params=['name', 'only']" in msgs
    assert any(re.fullmatch(rf"job {j1.id} started: kind=stage \(waited \d+\.\ds in the queue\)", m) for m in msgs)
    assert f"job {j1.id} stage a begin (1/2)" in msgs and f"job {j1.id} stage b begin (2/2)" in msgs
    assert any(re.fullmatch(rf"job {j1.id} stage a done in \d+\.\ds \(3/3\)", m) for m in msgs)
    assert any(re.fullmatch(rf"job {j1.id} stage b done in \d+\.\ds \(0/0\)", m) for m in msgs)  # the last stage ends with the job
    assert any(re.fullmatch(rf"job {j1.id} done: kind=stage duration=\d+\.\ds", m) for m in msgs)
    assert f"job {j2.id}: pause requested (running)" in msgs and f"job {j2.id}: resume requested" in msgs
    assert any(re.fullmatch(rf"job {j2.id} failed: kind=stage duration=\d+\.\ds error=tool_missing \(ffmpeg is gone\)", m) for m in msgs)
    assert f"job {j3.id}: cancel requested (queued)" in msgs
    assert f"job {j3.id} cancelled: kind=stage duration=0.0s" in msgs
    r.shutdown()


def test_job_records_from_before_log_path_still_load(tmp_path: Path) -> None:
    state = tmp_path / "jobs.json"
    old = jobs.JobRecord(id="old1", kind="report", title_key="jobs.report", state="done", created="2026-01-01T00:00:00+00:00").info(with_log=True)
    del old["log_path"]  # a jobs.json written before the field existed
    state.write_text(json.dumps([old]), encoding="utf-8")
    rec = JobRunner(state, None).get("old1")
    assert rec.log_path is None and rec.info()["log_path"] is None

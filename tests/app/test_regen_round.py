"""A regen round (score found lines without a passing candidate; synth and score run again for them) is reported as a
retry inside the quality step: the job's position never moves back to the voicing step, the record and the logs name the
round, and ``cli._regen_loop`` tells the job before each round."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from vndub.manifest.io import SelectionRecord, read_jsonl, write_jsonl
from vndub_app import jobs, pipeline
from vndub_app.jobs import JobRunner


def _texts(lines: list[str]) -> list[str]:
    return [line[9:] for line in lines]


def test_a_regen_round_keeps_the_job_at_the_quality_step(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger="vndub_app.jobs")
    r = JobRunner(None, None, log_dir=tmp_path / "jobs")
    seen: list[tuple[str | None, int, int, int, int]] = []

    def snap(ctx: jobs.JobContext) -> None:
        j = ctx.job
        seen.append((j.stage, j.stage_index, j.stage_round, j.round_total, j.round_lines))

    def fn(ctx: jobs.JobContext) -> None:
        ctx.set_stages(["voicebank", "synth", "score", "master"], n_items=20)
        for name in ("voicebank", "synth", "score"):
            ctx.begin_stage(name)
            ctx.progress(1, 1)
        snap(ctx)
        ctx.begin_round(1, 2, 7)
        ctx.begin_stage("synth")  # what the stage progress callback does when synth reports again
        ctx.progress(3, 28)
        snap(ctx)
        ctx.begin_stage("score")
        ctx.progress(1, 28)
        snap(ctx)
        ctx.begin_round(2, 2, 2)
        ctx.begin_stage("synth")
        snap(ctx)
        ctx.begin_stage("score")
        ctx.end_rounds()
        ctx.begin_stage("master")
        snap(ctx)

    job = r.submit("full", fn, project="demo")
    r.wait(job.id, 10)
    assert job.state == "done", job.error
    assert seen == [
        ("score", 3, 0, 0, 0),
        ("synth", 3, 1, 2, 7),  # synth again, but the job is still at step 3 of 4
        ("score", 3, 1, 2, 7),
        ("synth", 3, 2, 2, 2),
        ("master", 4, 0, 0, 0),  # the rounds are over: plain stages again
    ]
    assert job.log_path
    body = _texts(Path(job.log_path).read_text(encoding="utf-8").splitlines())
    note = "--- regen round 1/2: 7 line(s) without a passing candidate are voiced again"
    assert note in body
    assert "--- stage synth begin (3/4, regen round 1/2)" in body
    assert "--- stage score begin (3/4, regen round 2/2)" in body
    assert "--- stage master begin (4/4)" in body
    assert body.index(note) < body.index("--- stage synth begin (3/4, regen round 1/2)")
    msgs = [rec.getMessage() for rec in caplog.records]
    assert f"job {job.id} regen round 1/2: 7 line(s) without a passing candidate are voiced again" in msgs
    assert f"job {job.id} stage synth begin (3/4, regen round 1/2)" in msgs
    r.shutdown()


def test_the_rate_in_a_round_is_lines_of_the_round_not_of_the_script() -> None:
    r = JobRunner(None, None)
    rates: list[float | None] = []

    def fn(ctx: jobs.JobContext) -> None:
        ctx.set_stages(["synth", "score"], n_items=224)  # the script; the first pass made 4 takes a line
        ctx.begin_stage("synth")
        ctx.eta.reset(now=time.monotonic() - 60.0)
        ctx.progress(4, 896, "c4")
        rates.append(ctx.job.rate_per_min)
        ctx.begin_stage("score")
        ctx.begin_round(1, 2, 160)  # 160 lines again, 4 takes each
        ctx.begin_stage("synth")
        ctx.eta.reset(now=time.monotonic() - 60.0)
        ctx.progress(4, 640, "c104")
        rates.append(ctx.job.rate_per_min)

    job = r.submit("full", fn)
    r.wait(job.id, 10)
    assert job.state == "done", job.error
    assert rates[0] == pytest.approx(1.0, rel=0.05)  # 4 takes a minute over 224 lines / 896 takes = 1 line a minute
    assert rates[1] == pytest.approx(1.0, rel=0.05)  # the same pace over the round's 160 lines / 640 takes, not 224 / 640
    r.shutdown()


def test_end_rounds_without_a_round_changes_nothing() -> None:
    r = JobRunner(None, None)
    seen: list[int] = []

    def fn(ctx: jobs.JobContext) -> None:
        ctx.set_stages(["score", "master"])
        ctx.begin_stage("score")
        ctx.end_rounds()  # no regen round was needed
        ctx.begin_stage("master")
        seen.append(ctx.job.stage_index)

    job = r.submit("full", fn)
    r.wait(job.id, 10)
    assert job.state == "done" and seen == [2] and job.stage_round == 0
    r.shutdown()


def test_old_job_records_without_round_fields_load(tmp_path: Path) -> None:
    state = tmp_path / "jobs.json"
    old = {"id": "old1", "kind": "full", "title_key": "jobs.full", "state": "done", "created": "2026-01-01T00:00:00+00:00"}
    state.write_text(json.dumps([old]), encoding="utf-8")
    r = JobRunner(state, None)
    j = r.get("old1")
    assert j.stage_round == 0 and j.round_total == 0 and j.round_lines == 0
    assert j.info()["stage_round"] == 0
    r.shutdown()


# ----------------------------------------------------------------------------
# cli._regen_loop with the app's hook
# ----------------------------------------------------------------------------


def _ctx(tmp_path: Path, rounds: int) -> SimpleNamespace:
    return SimpleNamespace(
        cfg=SimpleNamespace(score=SimpleNamespace(regen_rounds=rounds)),
        paths=SimpleNamespace(selection=tmp_path / "selection.jsonl"),
        extra={},
        log=logging.getLogger("vndub.test_regen"),
    )


def _write_selection(path: Path, regen: list[str], auto: list[str]) -> None:
    recs = [SelectionRecord(line_id=lid, status="regen", reason="no candidate passes the gates") for lid in regen]
    recs += [SelectionRecord(line_id=lid, status="auto", reason="best of 2 passing") for lid in auto]
    write_jsonl(path, recs)


def test_regen_loop_tells_the_hook_before_each_round_and_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    from vndub import cli

    caplog.set_level("INFO", logger="vndub.test_regen")
    ctx = _ctx(tmp_path, rounds=2)
    _write_selection(ctx.paths.selection, regen=["a", "b", "c"], auto=["d"])
    calls: list[tuple[str, int, int, int]] = []
    ran: list[tuple[str, bool, int]] = []

    def fake_run_stage(name: str, c: SimpleNamespace) -> dict:
        ran.append((name, bool(c.extra.get("regen")), int(c.extra.get("regen_round", 0))))
        if name == "score":  # round 1 fixes two of the three lines, round 2 the last one
            left = ["c"] if c.extra["regen_round"] == 1 else []
            _write_selection(c.paths.selection, regen=left, auto=[x for x in "abcd" if x not in left])
        return {}

    monkeypatch.setattr(cli, "run_stage", fake_run_stage)
    cli._regen_loop(ctx, on_round=lambda rnd, total, n: calls.append(("round", rnd, total, n)))
    assert calls == [("round", 1, 2, 3), ("round", 2, 2, 1)]
    assert ran == [("synth", True, 1), ("score", True, 1), ("synth", True, 2), ("score", True, 2)]
    assert ctx.extra["regen"] is False
    msgs = [rec.getMessage() for rec in caplog.records]
    assert "regen round 1/2: 3 line(s) have no candidate passing the gates, new takes are synthesized for them" in msgs
    assert "regen round 2/2: 1 line(s) have no candidate passing the gates, new takes are synthesized for them" in msgs
    assert [s.status for s in read_jsonl(ctx.paths.selection, SelectionRecord)] == ["auto"] * 4


def test_regen_loop_stops_early_and_leaves_regen_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vndub import cli

    ctx = _ctx(tmp_path, rounds=3)
    _write_selection(ctx.paths.selection, regen=["a"], auto=["b"])
    calls: list[int] = []

    def fake_run_stage(name: str, c: SimpleNamespace) -> dict:
        if name == "score":
            _write_selection(c.paths.selection, regen=[], auto=["a", "b"])
        return {}

    monkeypatch.setattr(cli, "run_stage", fake_run_stage)
    cli._regen_loop(ctx, on_round=lambda rnd, total, n: calls.append(rnd))
    assert calls == [1]  # round 2 finds nothing to do
    assert ctx.extra["regen"] is False  # (before: the early return left the flag of the last round in place)

    # no round at all: the hook is never called and the loop works without one
    _write_selection(ctx.paths.selection, regen=[], auto=["a", "b"])
    calls.clear()
    cli._regen_loop(ctx, on_round=lambda rnd, total, n: calls.append(rnd))
    cli._regen_loop(ctx)
    assert calls == []


def test_run_stages_reports_the_rounds_through_the_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The app's stage loop wires ``_regen_loop`` to the job: the record carries the round while it runs and is clean after."""
    from vndub import cli
    from vndub.stages import common

    ctx = _ctx(tmp_path, rounds=2)
    _write_selection(ctx.paths.selection, regen=["a"], auto=["b"])
    seen: list[tuple[str | None, int, int, int]] = []

    def fake_run_stage(name: str, c: SimpleNamespace) -> dict:
        job: jobs.JobContext = c.extra["job"]
        job.stage_progress_cb()(name, 1, 1, "x")  # what a real stage does: report, which enters the stage
        j = job.job
        seen.append((j.stage, j.stage_index, j.stage_round, j.round_lines))
        if name == "score":
            fixed = bool(c.extra.get("regen"))
            _write_selection(c.paths.selection, regen=[] if fixed else ["a"], auto=["a", "b"] if fixed else ["b"])
        return {"status": {}}

    monkeypatch.setattr(cli, "run_stage", fake_run_stage)
    monkeypatch.setattr(common, "run_stage", fake_run_stage)
    r = JobRunner(None, None)

    def fn(job: jobs.JobContext) -> None:
        ctx.extra["job"] = job
        job.set_stages(["synth", "score", "master"], n_items=2)
        pipeline.run_stages(ctx, job, ["synth", "score", "master"], regen=True)

    job = r.submit("full", fn)
    r.wait(job.id, 10)
    assert job.state == "done", job.error
    assert seen == [("synth", 1, 0, 0), ("score", 2, 0, 0), ("synth", 2, 1, 1), ("score", 2, 1, 1), ("master", 3, 0, 0)]
    assert job.stage_round == 0 and job.round_total == 0 and job.round_lines == 0
    r.shutdown()

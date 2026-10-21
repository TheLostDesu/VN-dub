"""Job runner: long operations off the request thread, one GPU job at a time.

* :meth:`JobRunner.submit` queues a callable ``fn(job_ctx) -> dict | None``. Jobs with
  ``gpu=True`` (every pipeline stage) run strictly one after another on a single worker
  thread; ``gpu=False`` jobs (downloads, zips, installs) get a thread each.
* The callable reports through :class:`JobContext`: ``set_stages`` / ``begin_stage`` /
  ``progress`` / ``message`` / ``log`` / ``set_result``. ``progress`` and ``checkpoint``
  block while the job is paused and raise :class:`JobCancelled` once it is cancelled, so a
  pipeline stage that calls ``ctx.progress()`` (``StageContext.extra["progress_cb"]``,
  installed via :meth:`JobContext.stage_progress_cb`) is pausable and cancellable per line.
* ETA: :class:`EtaEstimator` (EMA of seconds per item inside the current stage) plus
  :class:`ThroughputStore` (measured seconds per line for every stage, persisted in
  ``<data>/throughput.json``) for the stages still ahead.
* Records are persisted to ``<data>/jobs.json``; jobs that were active when the app
  stopped come back as ``interrupted`` and the UI offers to continue (the pipeline resumes
  from ``work/stage_status.json`` + its caches).
* Every change is published to subscribers (the ``/api/progress`` WebSocket) as
  ``{"type": "job", "job": JobInfo}`` and ``{"type": "log", "job_id", "line"}``.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from vndub.util import now_iso

log = logging.getLogger("vndub_app.jobs")

STATES = ("queued", "running", "paused", "done", "failed", "cancelled", "interrupted")
ACTIVE = frozenset({"queued", "running", "paused"})
FINAL = frozenset({"done", "failed", "cancelled", "interrupted"})
LOG_TAIL = 300
EMIT_INTERVAL_S = 0.25


class JobCancelled(Exception):
    """Raised inside a job when the user pressed Cancel."""


# ----------------------------------------------------------------------------
# ETA
# ----------------------------------------------------------------------------


class EtaEstimator:
    """Remaining time of the current stage from measured per-item throughput.

    Keeps an exponential moving average of seconds per item over progress updates;
    returns ``None`` until ``min_items`` items were measured.
    """

    def __init__(self, alpha: float = 0.3, min_items: int = 2) -> None:
        self.alpha = alpha
        self.min_items = min_items
        self.reset()

    def reset(self) -> None:
        self._t_last: float | None = None
        self._done_last = 0
        self._measured = 0
        self.sec_per_item: float | None = None

    def update(self, done: int, total: int, now: float | None = None) -> float | None:
        now = time.monotonic() if now is None else now
        if self._t_last is None or done < self._done_last:
            self._t_last, self._done_last = now, done
            return self.eta(done, total)
        delta = done - self._done_last
        if delta > 0:
            spi = (now - self._t_last) / delta
            self.sec_per_item = spi if self.sec_per_item is None else self.alpha * spi + (1 - self.alpha) * self.sec_per_item
            self._measured += delta
            self._t_last, self._done_last = now, done
        return self.eta(done, total)

    def eta(self, done: int, total: int) -> float | None:
        if self.sec_per_item is None or self._measured < self.min_items or total <= 0:
            return None
        return max(0.0, (total - done) * self.sec_per_item)

    @property
    def rate_per_min(self) -> float | None:
        if not self.sec_per_item:
            return None
        return 60.0 / self.sec_per_item


class ThroughputStore:
    """Persisted seconds-per-line per stage (EMA across runs) for whole-job ETA."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.data: dict[str, float] = {}
        if path and path.exists():
            from vndub_app.appdirs import read_json

            self.data = {k: float(v) for k, v in (read_json(path, {}) or {}).items()}

    def get(self, stage: str) -> float | None:
        return self.data.get(stage)

    def record(self, stage: str, sec_per_item: float, alpha: float = 0.5) -> None:
        with self._lock:
            old = self.data.get(stage)
            self.data[stage] = sec_per_item if old is None else alpha * sec_per_item + (1 - alpha) * old
            if self.path:
                from vndub_app.appdirs import write_json_atomic

                write_json_atomic(self.path, self.data)

    def estimate(self, stages: list[str], n_items: int) -> float | None:
        vals = [self.data.get(s) for s in stages]
        if not stages or any(v is None for v in vals):
            return None
        return sum(vals) * n_items  # type: ignore[arg-type]


# ----------------------------------------------------------------------------
# records
# ----------------------------------------------------------------------------


@dataclass
class JobRecord:
    id: str
    kind: str
    title_key: str
    params: dict[str, Any] = field(default_factory=dict)
    project: str | None = None
    gpu: bool = True
    state: str = "queued"
    created: str = ""
    started: str | None = None
    finished: str | None = None
    stages: list[str] = field(default_factory=list)
    stage: str | None = None
    stage_index: int = 0
    stage_count: int = 0
    done: int = 0
    total: int = 0
    item: str | None = None
    eta_s: float | None = None
    stage_eta_s: float | None = None
    rate_per_min: float | None = None
    message_key: str | None = None
    message_params: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    result: dict[str, Any] = field(default_factory=dict)
    log_tail: list[str] = field(default_factory=list)

    def info(self, with_log: bool = False) -> dict[str, Any]:
        d = asdict(self)
        if not with_log:
            d.pop("log_tail", None)
        return d


class JobContext:
    """Handle a running job uses to report progress and honour pause / cancel."""

    def __init__(self, runner: "JobRunner", job: JobRecord) -> None:
        self.runner = runner
        self.job = job
        self._pause = threading.Event()
        self._pause.set()  # set = running
        self._cancel = threading.Event()
        self.eta = EtaEstimator()
        self._stage_t0: float | None = None
        self.n_items_hint: int = 0  # lines the job will process (for future-stage ETA)

    # -- control (called by the runner) -------------------------------------
    def _set_paused(self, paused: bool) -> None:
        if paused:
            self._pause.clear()
        else:
            self._pause.set()

    def _set_cancelled(self) -> None:
        self._cancel.set()
        self._pause.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def checkpoint(self) -> None:
        """Block while paused; raise :class:`JobCancelled` when cancelled."""
        if self._cancel.is_set():
            raise JobCancelled("cancelled by user")
        if not self._pause.is_set():
            self.runner._update(self.job, state="paused", force=True)
            while not self._pause.wait(0.2):
                pass
            if self._cancel.is_set():
                raise JobCancelled("cancelled by user")
            self.runner._update(self.job, state="running", force=True)

    # -- reporting -----------------------------------------------------------
    def set_stages(self, names: list[str], n_items: int = 0) -> None:
        self.n_items_hint = n_items
        self.runner._update(self.job, stages=list(names), stage_count=len(names), force=True)

    def begin_stage(self, name: str) -> None:
        self._finish_stage()
        stages = self.job.stages or [name]
        idx = stages.index(name) + 1 if name in stages else self.job.stage_index + 1
        self.eta.reset()
        self._stage_t0 = time.monotonic()
        self.runner._update(self.job, stage=name, stage_index=idx, done=0, total=0, item=None, stage_eta_s=None, force=True)
        self.checkpoint()

    def _finish_stage(self) -> None:
        if self.job.stage and self.eta.sec_per_item is not None:
            self.runner.throughput.record(self.job.stage, self.eta.sec_per_item)

    def progress(self, done: int, total: int, item: str | None = None) -> None:
        stage_eta = self.eta.update(done, total)
        eta = stage_eta
        if stage_eta is not None and self.job.stages and self.job.stage in self.job.stages:
            ahead = self.job.stages[self.job.stages.index(self.job.stage) + 1 :]
            rest = self.runner.throughput.estimate(ahead, self.n_items_hint or total)
            eta = stage_eta + rest if rest is not None else stage_eta
        self.runner._update(
            self.job, done=done, total=total, item=item, stage_eta_s=stage_eta, eta_s=eta, rate_per_min=self.eta.rate_per_min
        )
        self.checkpoint()

    def stage_progress_cb(self) -> Callable[[str, int, int, str | None], None]:
        """Callback for ``StageContext.extra["progress_cb"]`` (signature ``(stage, done, total, item)``)."""

        def cb(stage: str, done: int, total: int, item: str | None = None) -> None:
            if stage and stage != self.job.stage:
                self.begin_stage(stage)
            self.progress(done, total, item)

        return cb

    def message(self, key: str, **params: Any) -> None:
        self.runner._update(self.job, message_key=key, message_params=params, force=True)

    def log(self, line: str) -> None:
        self.runner._log(self.job, line)

    def set_result(self, **kv: Any) -> None:
        self.job.result.update(kv)
        self.runner._update(self.job, force=True)


# ----------------------------------------------------------------------------
# runner
# ----------------------------------------------------------------------------


class JobRunner:
    def __init__(self, state_file: Path | None = None, throughput_file: Path | None = None) -> None:
        self.state_file = state_file
        self.throughput = ThroughputStore(throughput_file)
        self._lock = threading.RLock()
        self._jobs: dict[str, JobRecord] = {}
        self._ctx: dict[str, JobContext] = {}
        self._fns: dict[str, Callable[[JobContext], Any]] = {}
        self._subs: list[Callable[[dict[str, Any]], None]] = []
        self._last_emit: dict[str, float] = {}
        self._last_save = 0.0
        self._gpu_q: "queue.Queue[str | None]" = queue.Queue()
        self._done_events: dict[str, threading.Event] = {}
        self._load()
        self._gpu_thread = threading.Thread(target=self._gpu_worker, name="vndub-gpu-jobs", daemon=True)
        self._gpu_thread.start()

    # -- persistence ---------------------------------------------------------
    def _load(self) -> None:
        if not self.state_file or not self.state_file.exists():
            return
        from vndub_app.appdirs import read_json

        for d in read_json(self.state_file, []) or []:
            try:
                rec = JobRecord(**{k: v for k, v in d.items() if k in JobRecord.__dataclass_fields__})
            except TypeError:
                continue
            if rec.state in ACTIVE:
                rec.state = "interrupted"
                rec.finished = rec.finished or now_iso()
            self._jobs[rec.id] = rec

    def _save(self, force: bool = False) -> None:
        if not self.state_file:
            return
        now = time.monotonic()
        if not force and now - self._last_save < 2.0:
            return
        self._last_save = now
        from vndub_app.appdirs import write_json_atomic

        with self._lock:
            recs = sorted(self._jobs.values(), key=lambda r: r.created)[-100:]
            data = [r.info(with_log=True) for r in recs]
        try:
            write_json_atomic(self.state_file, data)
        except OSError as e:  # never let bookkeeping kill a job
            log.warning("cannot save jobs: %s", e)

    # -- pub/sub -------------------------------------------------------------
    def subscribe(self, cb: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        """Register ``cb(event)`` (called from worker threads). Returns an unsubscribe function."""
        with self._lock:
            self._subs.append(cb)

        def unsubscribe() -> None:
            with self._lock:
                if cb in self._subs:
                    self._subs.remove(cb)

        return unsubscribe

    def _emit(self, ev: dict[str, Any]) -> None:
        with self._lock:
            subs = list(self._subs)
        for cb in subs:
            try:
                cb(ev)
            except Exception as e:  # a broken socket must not break the job
                log.debug("subscriber failed: %s", e)

    def _update(self, job: JobRecord, force: bool = False, **fields: Any) -> None:
        with self._lock:
            for k, v in fields.items():
                setattr(job, k, v)
        now = time.monotonic()
        if force or now - self._last_emit.get(job.id, 0.0) >= EMIT_INTERVAL_S:
            self._last_emit[job.id] = now
            self._emit({"type": "job", "job": job.info()})
            self._save(force=force and job.state in FINAL)

    def _log(self, job: JobRecord, line: str) -> None:
        with self._lock:
            job.log_tail.append(line)
            if len(job.log_tail) > LOG_TAIL:
                del job.log_tail[: len(job.log_tail) - LOG_TAIL]
        self._emit({"type": "log", "job_id": job.id, "line": line})

    # -- public API ----------------------------------------------------------
    def submit(
        self,
        kind: str,
        fn: Callable[[JobContext], dict[str, Any] | None],
        *,
        title_key: str | None = None,
        params: dict[str, Any] | None = None,
        project: str | None = None,
        gpu: bool = True,
    ) -> JobRecord:
        job = JobRecord(
            id=uuid.uuid4().hex[:12], kind=kind, title_key=title_key or f"jobs.{kind}", params=params or {},
            project=project, gpu=gpu, created=now_iso(),
        )
        with self._lock:
            self._jobs[job.id] = job
            self._ctx[job.id] = JobContext(self, job)
            self._fns[job.id] = fn
            self._done_events[job.id] = threading.Event()
        self._update(job, force=True)
        if gpu:
            self._gpu_q.put(job.id)
        else:
            threading.Thread(target=self._execute, args=(job.id,), name=f"vndub-job-{job.id}", daemon=True).start()
        return job

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            from vndub_app.errors import AppError

            raise AppError("job_not_found", params={"id": job_id}, status=404)
        return job

    def list(self, project: str | None = None, active_only: bool = False) -> list[JobRecord]:
        with self._lock:
            recs = list(self._jobs.values())
        if project is not None:
            recs = [r for r in recs if r.project == project]
        if active_only:
            recs = [r for r in recs if r.state in ACTIVE]
        return sorted(recs, key=lambda r: r.created)

    def active(self, project: str | None = None) -> JobRecord | None:
        act = self.list(project, active_only=True)
        return act[-1] if act else None

    def interrupted(self, project: str | None = None) -> list[JobRecord]:
        return [r for r in self.list(project) if r.state == "interrupted"]

    def pause(self, job_id: str) -> JobRecord:
        job = self.get(job_id)
        ctx = self._ctx.get(job_id)
        if ctx and job.state in ("running", "queued"):
            ctx._set_paused(True)
            if job.state == "queued":
                self._update(job, state="paused", force=True)
        return job

    def resume(self, job_id: str) -> JobRecord:
        job = self.get(job_id)
        ctx = self._ctx.get(job_id)
        if ctx and job.state == "paused":
            ctx._set_paused(False)
            if job.started is None:
                self._update(job, state="queued", force=True)
        return job

    def cancel(self, job_id: str) -> JobRecord:
        job = self.get(job_id)
        ctx = self._ctx.get(job_id)
        if ctx and job.state in ACTIVE:
            ctx._set_cancelled()
            if job.started is None:
                self._update(job, state="cancelled", finished=now_iso(), force=True)
                self._done_events[job_id].set()
        return job

    def wait(self, job_id: str, timeout: float | None = None) -> JobRecord:
        ev = self._done_events.get(job_id)
        if ev is not None:
            ev.wait(timeout)
        return self.get(job_id)

    def forget(self, job_id: str) -> None:
        """Drop a finished record (e.g. after the user resumed an interrupted job)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.state in FINAL:
                del self._jobs[job_id]
        self._save(force=True)

    def shutdown(self) -> None:
        for j in self.list(active_only=True):
            self.cancel(j.id)
        self._gpu_q.put(None)

    # -- execution -----------------------------------------------------------
    def _gpu_worker(self) -> None:
        while True:
            job_id = self._gpu_q.get()
            if job_id is None:
                return
            self._execute(job_id)

    def _execute(self, job_id: str) -> None:
        job = self.get(job_id)
        ctx = self._ctx[job_id]
        fn = self._fns.pop(job_id)
        try:
            if job.state == "cancelled":
                return
            # a job paused while queued waits here before it starts
            while not ctx._pause.is_set() and not ctx.cancelled:
                time.sleep(0.2)
            if ctx.cancelled:
                self._update(job, state="cancelled", finished=now_iso(), force=True)
                return
            self._update(job, state="running", started=now_iso(), force=True)
            res = fn(ctx)
            ctx._finish_stage()
            if res:
                job.result.update(res)
            self._update(job, state="done", finished=now_iso(), eta_s=0.0, stage_eta_s=0.0, force=True)
        except JobCancelled:
            self._update(job, state="cancelled", finished=now_iso(), force=True)
        except Exception as e:
            from vndub_app.errors import error_from_exception

            err = error_from_exception(e, stage=job.stage)
            log.exception("job %s (%s) failed", job.id, job.kind)
            self._log(job, f"ERROR: {err.details.splitlines()[0] if err.details else err.code}")
            self._update(job, state="failed", finished=now_iso(), error=err.to_dict(), force=True)
        finally:
            self._done_events[job_id].set()
            self._save(force=True)


_RUNNER: JobRunner | None = None
_RUNNER_LOCK = threading.Lock()


def runner() -> JobRunner:
    """The process-wide runner (persisting into the data folder)."""
    global _RUNNER
    with _RUNNER_LOCK:
        if _RUNNER is None:
            from vndub_app import appdirs

            d = appdirs.data_dir()
            _RUNNER = JobRunner(d / "jobs.json", d / "throughput.json")
        return _RUNNER


def reset_runner(r: JobRunner | None = None) -> None:
    """Tests: replace the process-wide runner."""
    global _RUNNER
    with _RUNNER_LOCK:
        if _RUNNER is not None and _RUNNER is not r:
            _RUNNER.shutdown()
        _RUNNER = r

"""Job runner: long operations off the request thread, one GPU job at a time.

* :meth:`JobRunner.submit` queues a callable ``fn(job_ctx) -> dict | None``. Jobs with
  ``gpu=True`` (every pipeline stage) run strictly one after another on a single worker
  thread; ``gpu=False`` jobs (downloads, zips, installs) get a thread each.
* The callable reports through :class:`JobContext`: ``set_stages`` / ``begin_stage`` /
  ``progress`` / ``message`` / ``log`` / ``set_result``. ``progress`` and ``checkpoint``
  block while the job is paused and raise :class:`JobCancelled` once it is cancelled, so a
  pipeline stage that calls ``ctx.progress()`` (``StageContext.extra["progress_cb"]``,
  installed via :meth:`JobContext.stage_progress_cb`) is pausable and cancellable per line.
* A job that runs a child process (``uv``, ``make_envs``, ``vc_redist``) registers it with
  :meth:`JobContext.watch_process`: cancelling the job (the Cancel button, or the app closing)
  kills the whole process tree at once instead of waiting for its next line of output.
* ETA: :class:`EtaEstimator` (average seconds per item since the current stage began) plus
  :class:`ThroughputStore` (measured seconds per line for every stage, persisted in
  ``<data>/throughput_v3.json``) for the stages still ahead. Only a stage that did real work is remembered: items
  that "finish" faster than ``INSTANT_S_PER_ITEM`` are cached results, a stage entered a second time (regen round)
  or a job of fewer than ``MIN_RECORD_ITEMS`` lines is not recorded, and a new value moves at most 3x per run.
* Records are persisted to ``<data>/jobs.json``; jobs that were active when the app
  stopped come back as ``interrupted`` and the UI offers to continue (the pipeline resumes
  from ``work/stage_status.json`` + its caches); an interrupted record nobody continued is forgotten after
  ``INTERRUPTED_KEEP_DAYS`` days. A record keeps the ``settings`` it was submitted with (``pipeline.submit``), so a
  resume can tell that the budget, the translation model or the folders changed in between.
* ``result["warnings"]``: what a run skipped or substituted (``JobContext.add_warnings``; shapes in API.md section 4).
* Every change is published to subscribers (the ``/api/progress`` WebSocket) as
  ``{"type": "job", "job": JobInfo}`` and ``{"type": "log", "job_id", "line"}``.
* Logging: every job log line carries the local time (``HH:MM:SS text``); with a ``log_dir`` the runner also keeps one
  buffered UTF-8 file per job (``<YYYYMMDD-HHMMSS>-<kind>-<id>.log``: header, every line, footer; the newest 30 are
  kept) and writes job / stage / pause / cancel events to ``app.log`` through the ``vndub_app.jobs`` logger.
"""

from __future__ import annotations

import json
import logging
import queue
import re
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TextIO

from vndub.util import now_iso

from vndub_app import winproc

log = logging.getLogger("vndub_app.jobs")

STATES = ("queued", "running", "paused", "done", "failed", "cancelled", "interrupted")
ACTIVE = frozenset({"queued", "running", "paused"})
FINAL = frozenset({"done", "failed", "cancelled", "interrupted"})
LOG_TAIL = 300
EMIT_INTERVAL_S = 0.25
JOB_LOG_KEEP = 30  # per-job log files kept in <data>/logs/jobs (the newest by name = by start time)
JOB_LOG_FLUSH_S = 2.0  # the flusher thread writes buffered job log lines to disk at least this often
JOB_LOG_READ_MAX = 5 * 1024 * 1024  # GET /api/jobs/{id}/log returns at most this much (newest lines)
INTERRUPTED_KEEP_DAYS = 14  # an interrupted record nobody continued is forgotten at the next start after this long
MIN_RECORD_ITEMS = 10  # a job that processes fewer lines (listen, tuning) does not feed the throughput store
MIN_RECORD_MEASURED = 20  # a stage is recorded when it measured this many items ...
MIN_RECORD_SHARE = 0.2  # ... or this share of its total with real work (the rest was cached)
THROUGHPUT_MAX_STEP = 3.0  # a recorded value moves at most this factor up or down per run
_PARAMS_LOG_MAX = 2000
_SECRET_KEY_RE = re.compile(r"token|secret|passw|api_?key", re.IGNORECASE)


def _stamp() -> str:
    """Local wall-clock time ``HH:MM:SS`` that prefixes every job log line."""
    return time.strftime("%H:%M:%S")


def _first_line(text: str | None, limit: int = 200) -> str:
    for line in (text or "").splitlines():
        if line.strip():
            line = line.strip()
            return line if len(line) <= limit else line[: limit - 3] + "..."
    return ""


def _age_days(stamp: str | None) -> float:
    """Days since an ISO timestamp (naive ones are local time); 0 for a missing or unreadable one, so nothing is dropped by mistake."""
    if not stamp:
        return 0.0
    try:
        t = datetime.fromisoformat(stamp)
        if t.tzinfo is None:
            t = t.astimezone()
        return (datetime.now(timezone.utc) - t).total_seconds() / 86400.0
    except (ValueError, OverflowError):
        return 0.0


def _params_json(params: dict[str, Any]) -> str:
    """Compact JSON of the job parameters for the log file header (secrets masked, long values cut)."""
    safe = {k: ("***" if _SECRET_KEY_RE.search(str(k)) else v) for k, v in params.items()}
    try:
        text = json.dumps(safe, ensure_ascii=False, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        text = repr(safe)
    return text if len(text) <= _PARAMS_LOG_MAX else text[: _PARAMS_LOG_MAX - 3] + "..."


class JobCancelled(Exception):
    """Raised inside a job when the user pressed Cancel."""


# ----------------------------------------------------------------------------
# ETA
# ----------------------------------------------------------------------------


class EtaEstimator:
    """Remaining time of the current stage from its measured throughput.

    The speed is the plain average since the stage began: all the time the stage has worked
    (model loading, the wait for the first batch, gaps between batches) divided by the items it
    finished. A stage that thinks 5 minutes and then reports a batch of 16 lines runs at
    3.2 lines a minute, not at the speed of the moment the batch arrived; batches that land
    close together (``llm.parallel`` > 1) don't make it look faster either.

    Not counted: the time the job was paused (:meth:`rebase`), and items that "finish" at once
    (cached results found by a scan): those are not throughput, and the time the scan took goes
    to the next real items.

    Cached or real is decided per *burst*: reports that follow each other closer than
    ``INSTANT_S_PER_ITEM`` belong to the burst the first of them opened, and the burst is work
    when its whole time (the gap before it plus its own spread) is at least ``INSTANT_S_PER_ITEM``
    per item. A batch of 16 lines that took 5 minutes and is then reported line by line within a
    few milliseconds is 16 lines of work; a thousand cached items reported one by one in a second
    (after a restart) are a scan. Judging each report alone would call the 15 lines after the
    first of a batch "cached" and leave the batch's whole time on one line. Returns ``None``
    until ``min_items`` items were measured.
    """

    INSTANT_S_PER_ITEM = 0.05

    def __init__(self, min_items: int = 2) -> None:
        self.min_items = min_items
        self.reset()

    def reset(self, now: float | None = None) -> None:
        """Start over; with ``now`` the clock starts at once (the stage began), else at the first update."""
        self._t_last: float | None = now  # where the committed work time ends (cached bursts do not move it)
        self._t_report = now  # the last report that changed the count
        self._done_last = 0
        self._measured = 0  # committed: bursts already closed
        self._work_s = 0.0
        self._b_ref = now  # the open burst: when the gap before it began, its items, when it last grew
        self._b_n = 0
        self._b_end = now

    def _burst_is_work(self) -> bool:
        if self._b_n <= 0 or self._b_ref is None or self._b_end is None:
            return False
        return (self._b_end - self._b_ref) / self._b_n >= self.INSTANT_S_PER_ITEM

    def _close_burst(self) -> None:
        if self._burst_is_work() and self._t_last is not None and self._b_end is not None:
            self._work_s += self._b_end - self._t_last
            self._measured += self._b_n
            self._t_last = self._b_end
        self._b_n = 0  # a cached burst leaves the clock where it was: its time goes to the next real items

    def rebase(self, now: float | None = None) -> None:
        """Forget the time since the last update (the job was paused): it was not work."""
        if self._t_last is not None:
            self._close_burst()
            now = time.monotonic() if now is None else now
            self._t_last = self._t_report = self._b_ref = self._b_end = now

    @property
    def measured(self) -> int:
        """Items that counted as work (cached results found at once are not among them)."""
        return self._measured + (self._b_n if self._burst_is_work() else 0)

    @property
    def sec_per_item(self) -> float | None:
        n = self.measured
        if n <= 0:
            return None
        work = self._work_s
        if self._burst_is_work() and self._b_end is not None and self._t_last is not None:
            work += self._b_end - self._t_last
        return work / n

    def update(self, done: int, total: int, now: float | None = None) -> float | None:
        now = time.monotonic() if now is None else now
        if self._t_last is None:
            self._t_last = self._t_report = self._b_ref = self._b_end = now
            self._done_last = done
            return self.eta(done, total)
        if done < self._done_last:  # a new pass (regen round): the clock keeps running, the count restarts
            self._done_last = done
            return self.eta(done, total)
        delta = done - self._done_last
        if delta > 0:
            prev = self._t_report if self._t_report is not None else self._t_last
            if now - prev >= self.INSTANT_S_PER_ITEM or self._b_n == 0:
                self._close_burst()
                self._b_ref = prev  # the burst's time starts where the gap before it began
            self._b_n += delta
            self._b_end = self._t_report = now
            self._done_last = done
        return self.eta(done, total)

    def eta(self, done: int, total: int) -> float | None:
        spi = self.sec_per_item
        if spi is None or self.measured < self.min_items or total <= 0:
            return None
        return max(0.0, (total - done) * spi)

    @property
    def rate_per_min(self) -> float | None:
        spi = self.sec_per_item
        if not spi or self.measured < self.min_items:
            return None
        return 60.0 / spi


class ThroughputStore:
    """Persisted seconds per voiced line for each stage (EMA across runs) for whole-job ETA.

    Per line, not per progress item: a stage counts its own units (candidates, parts, chunks), so only
    the per-line figure can be added up across stages.
    """

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
            if old is not None and old > 0:  # one odd run (a resume, a stalled disk) must not rewrite the history
                sec_per_item = min(max(sec_per_item, old / THROUGHPUT_MAX_STEP), old * THROUGHPUT_MAX_STEP)
            self.data[stage] = sec_per_item if old is None else alpha * sec_per_item + (1 - alpha) * old
            if self.path:
                from vndub_app.appdirs import write_json_atomic

                write_json_atomic(self.path, self.data)

    def estimate(self, stages: list[str], n_items: int) -> float | None:
        vals = [self.data.get(s) for s in stages]
        if not stages or n_items <= 0 or any(v is None for v in vals):
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
    cancel_requested: bool = False  # Cancel pressed, the job has not stopped yet
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
    log_path: str | None = None  # the per-job log file (set when the job starts running; None without a log folder)
    # what the app was set to when the job was submitted (gpu_budget_gb, translation_llm, license_policy, models_dir,
    # envs_dir, profile, src_lang, tgt_lang); a resume compares it with the settings of the moment. Old records: {}.
    settings: dict[str, Any] = field(default_factory=dict)
    log_tail: list[str] = field(default_factory=list)

    def info(self, with_log: bool = False) -> dict[str, Any]:
        d = asdict(self)
        if not with_log:
            d.pop("log_tail", None)
        return d


class _JobLogFile:
    """The buffered UTF-8 log file of one job. Writing never raises: the first failure is warned about once, then ignored.

    Lines go into the stdio buffer; :meth:`flush` (called by the runner's flusher thread every couple of seconds, and
    before the file is read) puts them on disk, so a job that prints thousands of lines costs no per-line disk write.
    """

    def __init__(self, path: Path, job_id: str) -> None:
        self.path = path
        self.job_id = job_id
        self._lock = threading.Lock()
        self._f: TextIO | None = None
        self._dirty = False
        self._warned = False

    def _failed(self, e: BaseException) -> None:
        if not self._warned:
            self._warned = True
            log.warning("job %s: cannot write its log file %s: %s (the job goes on without it)", self.job_id, self.path, e)
        f, self._f = self._f, None
        if f is not None:
            try:
                f.close()
            except (OSError, ValueError):
                pass

    def open(self) -> bool:
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._f = open(self.path, "w", encoding="utf-8", newline="\n")  # noqa: SIM115  (closed by close())
            except OSError as e:
                self._failed(e)
                return False
            return True

    def write(self, line: str) -> None:
        with self._lock:
            if self._f is None:
                return
            try:
                self._f.write(line + "\n")
                self._dirty = True
            except (OSError, ValueError) as e:
                self._failed(e)

    def flush(self) -> None:
        with self._lock:
            if self._f is None or not self._dirty:
                return
            try:
                self._f.flush()
                self._dirty = False
            except (OSError, ValueError) as e:
                self._failed(e)

    def close(self) -> None:
        with self._lock:
            f, self._f = self._f, None
            if f is None:
                return
            try:
                f.close()
            except (OSError, ValueError) as e:
                self._warned = True
                log.warning("job %s: cannot finish its log file %s: %s", self.job_id, self.path, e)


class JobContext:
    """Handle a running job uses to report progress and honour pause / cancel."""

    def __init__(self, runner: "JobRunner", job: JobRecord) -> None:
        self.runner = runner
        self.job = job
        self._pause = threading.Event()
        self._pause.set()  # set = running
        self._cancel = threading.Event()
        self.eta = EtaEstimator()
        self.t_queued = time.monotonic()
        self.t_started: float | None = None  # set by the runner when the job starts running
        self._stage_t0: float | None = None
        self._begun: set[str] = set()  # stages this job has started at least once (a second entry is a regen round)
        self._stage_recordable = True  # the current stage's first entry: its speed may go to the throughput store
        self.n_items_hint: int = 0  # lines the job will process (for future-stage ETA)
        self._procs: list["subprocess.Popen[Any]"] = []
        self._procs_lock = threading.Lock()
        self._on_cancel: list[Callable[[], None]] = []

    # -- control (called by the runner) -------------------------------------
    def _set_paused(self, paused: bool) -> None:
        if paused:
            self._pause.clear()
        else:
            self._pause.set()

    def _set_cancelled(self) -> None:
        self._cancel.set()
        self._pause.set()
        with self._procs_lock:
            procs = list(self._procs)
            hooks = list(self._on_cancel)
        for p in procs:  # outside the lock: taskkill takes a moment
            winproc.kill_tree(p)
        for h in hooks:
            try:
                h()
            except Exception:
                log.exception("job %s: cancel hook failed", self.job.id)

    def on_cancel(self, fn: Callable[[], None]) -> Callable[[], None]:
        """Run ``fn`` (from the thread that pressed Cancel) when this job is cancelled; runs at once if it already was.

        For work that blocks without reaching a checkpoint (a model process answering a long request): ``fn`` ends it.
        Returns a function that removes the hook.
        """
        with self._procs_lock:
            self._on_cancel.append(fn)
            cancelled = self._cancel.is_set()
        if cancelled:
            fn()

        def remove() -> None:
            with self._procs_lock:
                if fn in self._on_cancel:
                    self._on_cancel.remove(fn)

        return remove

    def watch_process(self, proc: "subprocess.Popen[Any]") -> Callable[[], None]:
        """Kill ``proc`` and everything it started the moment this job is cancelled.

        Returns a function that stops watching (call it once the process has ended). A process registered
        after the cancel is killed at once. Thread-safe; watching the same process twice is harmless.
        """
        with self._procs_lock:
            if proc not in self._procs:
                self._procs.append(proc)
            cancelled = self._cancel.is_set()
        if cancelled:
            winproc.kill_tree(proc)

        def unwatch() -> None:
            with self._procs_lock:
                if proc in self._procs:
                    self._procs.remove(proc)

        return unwatch

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
            self.eta.rebase()  # the pause is not part of the next item's time
            self.runner._update(self.job, state="running", force=True)

    # -- reporting -----------------------------------------------------------
    def set_stages(self, names: list[str], n_items: int = 0) -> None:
        self.n_items_hint = n_items
        self.runner._update(self.job, stages=list(names), stage_count=len(names), force=True)

    def begin_stage(self, name: str) -> None:
        self._finish_stage()
        self._stage_recordable = name not in self._begun  # a stage entered again (regen round) re-does a few items only
        self._begun.add(name)
        stages = self.job.stages or [name]
        idx = stages.index(name) + 1 if name in stages else self.job.stage_index + 1
        self._stage_t0 = time.monotonic()
        self.eta.reset(self._stage_t0)  # the time before the first report (loading a model) is the stage's too
        self.runner._update(self.job, stage=name, stage_index=idx, done=0, total=0, item=None, stage_eta_s=None, force=True)
        count = self.job.stage_count or len(stages)
        log.info("job %s stage %s begin (%d/%d)", self.job.id, name, idx, count)
        self.runner._file_note(self.job, f"--- stage {name} begin ({idx}/{count})")
        self.checkpoint()

    def _finish_stage(self) -> None:
        """The current stage ended normally: remember its throughput (when it measured enough real work) and log it."""
        if self._records_throughput():
            self.runner.throughput.record(self.job.stage or "", (self.eta.sec_per_item or 0.0) * self.job.total / self.n_items_hint)
        self._end_stage_log("done")

    def _records_throughput(self) -> bool:
        """Is what the current stage measured a fair speed? Not after a resume that found most of the work done, not
        for a stage entered again, not for a handful of lines (a listen job's 3 lines carry the model's load time)."""
        if not self.job.stage or not self._stage_recordable or self.n_items_hint < MIN_RECORD_ITEMS or self.job.total <= 0:
            return False
        n = self.eta.measured
        if self.eta.sec_per_item is None or n < self.eta.min_items:
            return False
        return n >= MIN_RECORD_MEASURED or n >= MIN_RECORD_SHARE * self.job.total

    def _end_stage_log(self, outcome: str) -> None:
        """Log the end of the current stage once (``done`` / ``failed`` / ``cancelled``), with its duration and item count."""
        t0, self._stage_t0 = self._stage_t0, None
        if t0 is None or not self.job.stage:
            return
        dt = time.monotonic() - t0
        log.info("job %s stage %s %s in %.1fs (%d/%d)", self.job.id, self.job.stage, outcome, dt, self.job.done, self.job.total)
        self.runner._file_note(self.job, f"--- stage {self.job.stage} {outcome} in {dt:.1f}s ({self.job.done}/{self.job.total})")

    def progress(self, done: int, total: int, item: str | None = None) -> None:
        stage_eta = self.eta.update(done, total)
        if stage_eta is None and total > 0 and self.n_items_hint > 0 and self.job.stage:
            known = self.runner.throughput.get(self.job.stage)  # not measured yet: what the last run took
            if known is not None:
                stage_eta = known * self.n_items_hint * max(0.0, 1 - done / total)
        eta = stage_eta
        if stage_eta is not None and self.job.stages and self.job.stage in self.job.stages:
            ahead = self.job.stages[self.job.stages.index(self.job.stage) + 1 :]
            rest = self.runner.throughput.estimate(ahead, self.n_items_hint)
            eta = stage_eta + rest if rest is not None else stage_eta
        rate = self.eta.rate_per_min
        if rate is not None and self.n_items_hint > 0 and total > self.n_items_hint:
            rate *= self.n_items_hint / total  # the stage counts candidates / parts: the UI shows lines a minute
        self.runner._update(self.job, done=done, total=total, item=item, stage_eta_s=stage_eta, eta_s=eta, rate_per_min=rate)
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

    def add_warnings(self, items: list[dict[str, Any]]) -> int:
        """Append ``result["warnings"]`` items the job does not have yet (what a run skipped or substituted); returns how many were new."""
        return self.runner.add_warnings(self.job, items)


# ----------------------------------------------------------------------------
# runner
# ----------------------------------------------------------------------------


class JobRunner:
    def __init__(self, state_file: Path | None = None, throughput_file: Path | None = None, log_dir: Path | None = None) -> None:
        self.state_file = state_file
        self.throughput = ThroughputStore(throughput_file)
        self.log_dir = log_dir  # per-job log files go here (None: no files, the tail in the record only)
        self._files: dict[str, _JobLogFile] = {}
        self._flush_stop = threading.Event()
        self._flusher: threading.Thread | None = None
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

        expired = 0
        for d in read_json(self.state_file, []) or []:
            try:
                rec = JobRecord(**{k: v for k, v in d.items() if k in JobRecord.__dataclass_fields__})
            except TypeError:
                continue
            if rec.state in ACTIVE:
                rec.state = "interrupted"
                rec.finished = rec.finished or now_iso()
            elif rec.state == "interrupted" and _age_days(rec.finished or rec.created) > INTERRUPTED_KEEP_DAYS:
                expired += 1  # offered for two weeks and never continued: it is not coming back
                log.info("job %s (%s): interrupted since %s, forgotten", rec.id, rec.kind, rec.finished or rec.created)
                continue
            self._jobs[rec.id] = rec
        if expired:
            self._save(force=True)

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
            old_state = job.state
            for k, v in fields.items():
                setattr(job, k, v)
            new_state = job.state
        if new_state != old_state:
            self._on_state(job, old_state, new_state)
        now = time.monotonic()
        if force or now - self._last_emit.get(job.id, 0.0) >= EMIT_INTERVAL_S:
            self._last_emit[job.id] = now
            self._emit({"type": "job", "job": job.info()})
            self._save(force=force and job.state in FINAL)

    def _log(self, job: JobRecord, line: str) -> None:
        """One job log line: ``HH:MM:SS text`` into the record's tail, the job's log file and the subscribers."""
        line = f"{_stamp()} {line}"
        with self._lock:
            job.log_tail.append(line)
            if len(job.log_tail) > LOG_TAIL:
                del job.log_tail[: len(job.log_tail) - LOG_TAIL]
        f = self._files.get(job.id)  # the disk write happens outside the runner lock (the file has its own)
        if f is not None:
            f.write(line)
        self._emit({"type": "log", "job_id": job.id, "line": line})

    # -- per-job log files -----------------------------------------------------
    def _file_note(self, job: JobRecord, text: str) -> None:
        """An event line (stage / pause / cancel) for the job's log file only: the UI log shows what the job itself printed."""
        f = self._files.get(job.id)
        if f is not None:
            f.write(f"{_stamp()} {text}")

    def _open_job_log(self, job: JobRecord) -> None:
        if self.log_dir is None:
            return
        safe_kind = re.sub(r"[^\w.-]+", "_", job.kind) or "job"
        path = self.log_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{safe_kind}-{job.id}.log"
        f = _JobLogFile(path, job.id)
        if not f.open():
            return
        with self._lock:
            self._files[job.id] = f
            job.log_path = str(path)
            if self._flusher is None:
                self._flusher = threading.Thread(target=self._flush_loop, name="vndub-job-log-flush", daemon=True)
                self._flusher.start()
        f.write(f"{_stamp()} === job {job.id} kind={job.kind} project={job.project or '-'} created={job.created} params={_params_json(job.params)}")
        self._prune_job_logs()

    def _prune_job_logs(self) -> None:
        """Keep the newest ``JOB_LOG_KEEP`` files (names start with the start time, so name order is time order)."""
        if self.log_dir is None:
            return
        try:
            files = sorted(self.log_dir.glob("*.log"), key=lambda p: p.name)
        except OSError:
            return
        with self._lock:
            busy = {f.path for f in self._files.values()}
        for p in files[: max(0, len(files) - JOB_LOG_KEEP)]:
            if p in busy:
                continue
            try:
                p.unlink()
            except OSError:  # open in an editor, or already gone
                pass

    def _flush_loop(self) -> None:
        while not self._flush_stop.wait(JOB_LOG_FLUSH_S):
            self.flush_logs()

    def flush_logs(self) -> None:
        """Write the buffered lines of every running job's log file to disk (before the file is read or zipped)."""
        with self._lock:
            files = list(self._files.values())
        for f in files:
            f.flush()

    def read_log(self, job_id: str) -> dict[str, Any]:
        """``{"path", "lines"}``: the job's whole log file (newest ``JOB_LOG_READ_MAX`` bytes), else the in-memory tail."""
        job = self.get(job_id)
        f = self._files.get(job_id)
        if f is not None:
            f.flush()
        if job.log_path:
            path = Path(job.log_path)
            try:
                size = path.stat().st_size
                with open(path, "rb") as fh:
                    if size > JOB_LOG_READ_MAX:
                        fh.seek(size - JOB_LOG_READ_MAX)
                        data = fh.read().split(b"\n", 1)[-1]  # the first line is cut in half
                    else:
                        data = fh.read()
                return {"path": str(path), "lines": data.decode("utf-8", "replace").splitlines()}
            except OSError:
                pass  # pruned, or the folder was cleaned: the tail is what is left
        with self._lock:
            return {"path": None, "lines": list(job.log_tail)}

    def _on_state(self, job: JobRecord, old: str, new: str) -> None:
        """A job changed state: open / close its log file and write the event to app.log."""
        ctx = self._ctx.get(job.id)
        if new == "running":
            if ctx is not None and ctx.t_started is None:  # the first time: it left the queue
                ctx.t_started = time.monotonic()
                self._open_job_log(job)
                log.info("job %s started: kind=%s (waited %.1fs in the queue)", job.id, job.kind, ctx.t_started - ctx.t_queued)
            else:
                self._file_note(job, "=== resumed")
        elif new == "paused":
            self._file_note(job, "=== paused")
        elif new in FINAL:
            self._on_final(job, ctx)

    def _on_final(self, job: JobRecord, ctx: JobContext | None) -> None:
        dur = time.monotonic() - ctx.t_started if ctx is not None and ctx.t_started is not None else 0.0
        code = (job.error or {}).get("code")
        why = _first_line((job.error or {}).get("details")) if job.error else ""
        log.info("job %s %s: kind=%s duration=%.1fs%s%s", job.id, job.state, job.kind, dur, f" error={code}" if code else "", f" ({why})" if why else "")
        with self._lock:
            f = self._files.pop(job.id, None)
        if f is not None:
            f.write(f"{_stamp()} === {job.state} after {dur:.1f}s" + (f" error={code}" if code else ""))
            f.close()

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
        settings: dict[str, Any] | None = None,
    ) -> JobRecord:
        job = JobRecord(
            id=uuid.uuid4().hex[:12], kind=kind, title_key=title_key or f"jobs.{kind}", params=params or {},
            project=project, gpu=gpu, created=now_iso(), settings=dict(settings or {}),
        )
        with self._lock:
            self._jobs[job.id] = job
            self._ctx[job.id] = JobContext(self, job)
            self._fns[job.id] = fn
            self._done_events[job.id] = threading.Event()
        log.info("job %s submitted: kind=%s project=%s gpu=%s params=%s", job.id, kind, project or "-", gpu, sorted(job.params))
        self._update(job, force=True)
        if gpu:
            self._gpu_q.put(job.id)
        else:
            threading.Thread(target=self._execute, args=(job.id,), name=f"vndub-job-{job.id}", daemon=True).start()
        return job

    def add_warnings(self, job: JobRecord, items: list[dict[str, Any]]) -> int:
        """Append the ``result["warnings"]`` items the job does not have yet (never overwrites); returns how many were new."""
        with self._lock:
            have = list(job.result.get("warnings") or [])
            added = 0
            for it in items:
                if it not in have:
                    have.append(dict(it))
                    added += 1
            if not added:
                return 0
            job.result["warnings"] = have
        self._update(job, force=True)
        return added

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
            log.info("job %s: pause requested (%s)", job.id, job.state)
            self._file_note(job, "=== pause requested")
            ctx._set_paused(True)
            if job.state == "queued":
                self._update(job, state="paused", force=True)
        return job

    def resume(self, job_id: str) -> JobRecord:
        job = self.get(job_id)
        ctx = self._ctx.get(job_id)
        if ctx and job.state == "paused":
            log.info("job %s: resume requested", job.id)
            self._file_note(job, "=== resume requested")
            ctx._set_paused(False)
            if job.started is None:
                self._update(job, state="queued", force=True)
        return job

    def cancel(self, job_id: str) -> JobRecord:
        job = self.get(job_id)
        ctx = self._ctx.get(job_id)
        if ctx and job.state in ACTIVE:
            log.info("job %s: cancel requested (%s)", job.id, job.state)
            self._file_note(job, "=== cancel requested")
            self._update(job, cancel_requested=True, force=True)
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
        self._flush_stop.set()  # the files of jobs still winding down are closed by their own final state
        self.flush_logs()

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
            ctx.eta.reset(time.monotonic())  # a job without stages: its clock starts now
            res = fn(ctx)
            ctx._finish_stage()
            if res:
                job.result.update(res)
            self._update(job, state="done", finished=now_iso(), eta_s=0.0, stage_eta_s=0.0, force=True)
        except JobCancelled:
            ctx._end_stage_log("cancelled")
            self._update(job, state="cancelled", finished=now_iso(), force=True)
        except Exception as e:
            if ctx.cancelled:  # Cancel killed what the job was waiting on: that error is the cancel, not a failure
                ctx._end_stage_log("cancelled")
                self._update(job, state="cancelled", finished=now_iso(), force=True)
                return
            ctx._end_stage_log("failed")
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
            _RUNNER = JobRunner(d / "jobs.json", d / "throughput_v3.json", d / "logs" / "jobs")  # v2 held numbers a resume had poisoned
        return _RUNNER


def reset_runner(r: JobRunner | None = None) -> None:
    """Tests: replace the process-wide runner."""
    global _RUNNER
    with _RUNNER_LOCK:
        if _RUNNER is not None and _RUNNER is not r:
            _RUNNER.shutdown()
        _RUNNER = r


def flush_logs() -> None:
    """Write the buffered per-job log lines of the process-wide runner to disk (the problem report calls this first)."""
    r = _RUNNER
    if r is not None:
        r.flush_logs()

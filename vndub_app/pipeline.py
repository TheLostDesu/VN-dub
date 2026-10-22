"""The job flows: how the app drives the pipeline through its Python API.

Every flow is a function ``fn(job_ctx) -> dict | None`` built by a *builder*
``build(slug, params)``; builders are registered per job *kind* (API.md section 4) so a job
that was interrupted by an app restart can be re-submitted from its stored ``kind`` and
``params`` (``POST /api/project/resume``) -- the pipeline's caches make it continue where it
stopped.

A flow follows API.md section 1: ``write_project_config`` -> ``make_ctx`` with
``extra["progress_cb"]`` -> ``begin_stage`` + ``run_stage`` per stage -> ``cli._regen_loop`` after
``score`` -> ``ctx.models.close()`` in ``finally``. ``vndub*`` log records are forwarded to
``job_ctx.log`` (see :class:`JobLogBridge`).

Flows never send display text: messages are i18n keys, errors are ``AppError`` codes.
"""

from __future__ import annotations

import logging
import random
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from vndub.util import now_iso
from vndub_app import appdirs, jobs, session, viewdata
from vndub_app.errors import AppError
from vndub_app.jobs import JobContext, JobRecord

log = logging.getLogger("vndub_app.pipeline")

FlowFn = Callable[[JobContext], "dict[str, Any] | None"]
Builder = Callable[[str, "dict[str, Any]"], FlowFn]


@dataclass(frozen=True)
class Flow:
    kind: str
    build: Builder
    gpu: bool = True
    heavy: bool = False  # at most one heavy job per project at a time
    project: bool = True


FLOWS: dict[str, Flow] = {}


def flow(kind: str, *, gpu: bool = True, heavy: bool = False, project: bool = True) -> Callable[[Builder], Builder]:
    def deco(build: Builder) -> Builder:
        FLOWS[kind] = Flow(kind, build, gpu=gpu, heavy=heavy, project=project)
        return build

    return deco


# ----------------------------------------------------------------------------
# submitting
# ----------------------------------------------------------------------------


def _same_params(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return {k: v for k, v in a.items() if k != "seed"} == {k: v for k, v in b.items() if k != "seed"}


def submit(kind: str, slug: str | None, params: dict[str, Any] | None = None) -> JobRecord:
    """Queue a flow. Heavy flows are refused while another heavy job of the project is active
    (``job_busy``); a light duplicate (same kind + params) returns the existing job."""
    fl = FLOWS.get(kind)
    if fl is None:
        raise AppError("bad_request", params={"field": "kind", "value": kind}, status=422)
    params = dict(params or {})
    runner = jobs.runner()
    if fl.project and slug:
        active = runner.list(slug, active_only=True)
        if fl.heavy:
            busy = next((j for j in active if FLOWS.get(j.kind) and FLOWS[j.kind].heavy), None)
            if busy is not None:
                raise AppError("job_busy", params={"job_id": busy.id, "kind": busy.kind}, action="none", status=409)
        else:
            dup = next((j for j in active if j.kind == kind and _same_params(j.params, params)), None)
            if dup is not None:
                return dup
    fn = fl.build(slug or "", params)
    return runner.submit(kind, fn, params=params, project=slug if fl.project else None, gpu=fl.gpu)


def resume_interrupted(slug: str) -> JobRecord:
    """Re-submit the latest interrupted job of the project (same kind + params)."""
    runner = jobs.runner()
    inter = runner.interrupted(slug)
    if not inter:
        raise AppError("nothing_to_resume", action="none", status=404)
    old = inter[-1]
    new = submit(old.kind, slug, old.params)
    for j in inter:
        runner.forget(j.id)
    return new


# ----------------------------------------------------------------------------
# job log forwarding
# ----------------------------------------------------------------------------


class JobLogBridge(logging.Handler):
    """Forward ``vndub*`` log records to ``job.log``.

    ``vndub.util.get_logger`` makes each stage logger (``vndub.<stage>``) non-propagating and
    removes its handlers every time a stage starts, so a handler on the ``vndub`` logger would
    miss stage output. Filters survive that reset, so the bridge attaches a filter to every
    stage logger and a handler to the parent (backends, registry ...); a record reaches exactly one.
    """

    def __init__(self, job: JobContext, level: int = logging.INFO) -> None:
        super().__init__(level)
        self.job = job
        self._filter = _ForwardFilter(self)
        self._stage_loggers: list[logging.Logger] = []

    def _forward(self, record: logging.LogRecord) -> None:
        if getattr(record, "_job_forwarded", False) or record.levelno < self.level:
            return
        record._job_forwarded = True  # type: ignore[attr-defined]
        try:
            self.job.log(f"{record.levelname} {record.name}: {record.getMessage()}")
        except Exception:  # logging must never break a stage
            pass

    def emit(self, record: logging.LogRecord) -> None:
        self._forward(record)

    def attach(self) -> "JobLogBridge":
        from vndub.stages.common import STAGES

        logging.getLogger("vndub").addHandler(self)
        for name in STAGES:
            lg = logging.getLogger(f"vndub.{name}")
            lg.addFilter(self._filter)
            self._stage_loggers.append(lg)
        return self

    def detach(self) -> None:
        logging.getLogger("vndub").removeHandler(self)
        for lg in self._stage_loggers:
            lg.removeFilter(self._filter)
        self._stage_loggers.clear()


class _ForwardFilter(logging.Filter):
    def __init__(self, bridge: JobLogBridge) -> None:
        super().__init__()
        self.bridge = bridge

    def filter(self, record: logging.LogRecord) -> bool:
        self.bridge._forward(record)
        return True


# ----------------------------------------------------------------------------
# context + stage helpers
# ----------------------------------------------------------------------------


@contextmanager
def pipeline_ctx(
    h: appdirs.ProjectHandle,
    job: JobContext,
    *,
    only: list[str] | None = None,
    limit: int | None = None,
    force: bool = False,
    sets: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """Pipeline context wired to the job (progress callback, log forwarding); models closed on exit."""
    ex = {"progress_cb": job.stage_progress_cb(), **(extra or {})}
    try:  # the project glossary is generated from characters.yaml + the Step 2 choices
        from vndub_app import translation

        translation.write_glossary(h)
    except Exception as e:  # never fatal: the translate stage works without a glossary
        log.warning("cannot write the project glossary: %s", e)
    ctx = session.job_ctx(h, only=only, limit=limit, force=force, sets=_with_unvoiced(sets), extra=ex)
    bridge = JobLogBridge(job).attach()
    unhook = job.on_cancel(ctx.models.abort_all)  # Cancel ends a model process mid-request, not after its answer
    try:
        yield ctx
    finally:
        unhook()
        bridge.detach()
        try:
            ctx.models.close()
        finally:
            viewdata.clear_cache(h.slug)
        session.sweep_orphans("leftover after a model job")  # a grandchild of an adapter outlives its graceful shutdown


def config_has(section: str, key: str) -> bool:
    """Does the pipeline config schema know ``section.key`` (other agents add keys while the app is built)."""
    try:
        from vndub.config import PipelineConfig

        sec = PipelineConfig.model_fields[section].annotation
        return key in getattr(sec, "model_fields", {})
    except Exception:
        return False


def _with_unvoiced(sets: dict[str, Any] | None) -> dict[str, Any]:
    """Every app job synthesizes unvoiced speaker lines (mode B) when the pipeline supports it."""
    out = dict(sets or {})
    if config_has("synth", "voice_unvoiced"):
        out.setdefault("synth.voice_unvoiced", True)
    return out


def available_stages(names: list[str]) -> list[str]:
    """Keep only stages this pipeline version has (``text_emotion`` arrives with another task)."""
    from vndub.stages.common import STAGES

    return [n for n in names if n in STAGES]


def run_stages(ctx: Any, job: JobContext, names: list[str], *, regen: bool = False) -> dict[str, Any]:
    from vndub.stages.common import run_stage

    out: dict[str, Any] = {}
    for n in names:
        job.begin_stage(n)
        out[n] = run_stage(n, ctx)
        if n == "score" and regen:
            from vndub.cli import _regen_loop

            _regen_loop(ctx)
    return out


def prepare_stage_ctx(ctx: Any, stage: str) -> None:
    """Set ``ctx.stage`` / ``ctx.log`` for code that calls stage helpers outside ``run_stage``."""
    from vndub.util import get_logger

    ctx.stage = stage
    ctx.log = get_logger(stage, ctx.paths.log(stage))


@contextmanager
def preserve_status(ctx: Any, names: list[str]) -> Iterator[None]:
    """Subset runs (preview, listen, apply) must not overwrite the stage summaries of the full run."""
    from vndub.manifest.io import write_json
    from vndub.stages.common import read_status

    before = read_status(ctx.paths)
    try:
        yield
    finally:
        cur = read_status(ctx.paths)
        for n in names:
            if n in before:
                cur[n] = before[n]
            else:
                cur.pop(n, None)
        write_json(ctx.paths.stage_status, cur)


def _handle(slug: str) -> appdirs.ProjectHandle:
    return appdirs.project_handle(slug)


def _n_hint(h: appdirs.ProjectHandle) -> int:
    try:
        return len(viewdata.snapshot(h).voiced)
    except Exception:
        return 0


# ----------------------------------------------------------------------------
# Step 1: import, recover text, ASR check
# ----------------------------------------------------------------------------


@flow("import", heavy=True)
def _build_import(slug: str, p: dict[str, Any]) -> FlowFn:
    def run(job: JobContext) -> dict[str, Any]:
        h = _handle(slug)
        names = available_stages(["detect", "extract", "parse", "link", "analyze", "text_emotion"])
        job.set_stages(names)
        extra = {"engine": p["engine"]} if p.get("engine") else None
        # the languages of an existing project file follow the Step 2 choices before detect keeps them
        # (a code that was never detected and no longer chosen becomes "auto": parse detects it over the lines)
        from vndub_app import translation

        tr = appdirs.read_app_state(h).get("translation", {})
        translation._sync_project_lang(h, tr.get("tgt_lang", "ru"), str(tr.get("src_lang") or "auto"))
        with pipeline_ctx(h, job, force=bool(p.get("force")), extra=extra) as ctx:
            run_stages(ctx, job, names)
        snap = viewdata.snapshot(h)
        return {"lines": len(snap.lines), "voiced": len(snap.voiced)}

    return run


@flow("recover_text", heavy=True)
def _build_recover(slug: str, p: dict[str, Any]) -> FlowFn:
    def run(job: JobContext) -> dict[str, Any]:
        h = _handle(slug)
        # parse first: for a voice-only game (no script) it writes one empty-text voiced line per voice file when
        # link.recover_text is set; for a normal game it is a cached no-op that keeps the user's character edits
        names = available_stages(["parse", "link", "analyze"])
        job.set_stages(names)
        with pipeline_ctx(h, job, sets={"link.recover_text": True}) as ctx:
            run_stages(ctx, job, names)
            recovered = int(viewdata.stage_summary(viewdata.stage_status(ctx), "link").get("recovered", 0) or 0)
        viewdata.clear_cache(slug)
        from vndub_app import summary

        return {"recovered": recovered, "no_text_lines": summary.no_text_count(h)}

    return run


@flow("asr_check", heavy=True)
def _build_asr_check(slug: str, p: dict[str, Any]) -> FlowFn:
    def run(job: JobContext) -> dict[str, Any]:
        h = _handle(slug)
        job.set_stages(["link"])
        with pipeline_ctx(h, job, sets={"link.mode": "asr"}) as ctx:
            res = run_stages(ctx, job, ["link"])
        s = res.get("link") or {}
        return {k: s[k] for k in ("ok", "review", "unmatched", "confidence") if k in s}

    return run


# ----------------------------------------------------------------------------
# Step 2: translate;  generic stage run
# ----------------------------------------------------------------------------


@flow("translate", heavy=True)
def _build_translate(slug: str, p: dict[str, Any]) -> FlowFn:
    def run(job: JobContext) -> dict[str, Any]:
        h = _handle(slug)
        names = available_stages(["translate", "text_emotion"])
        job.set_stages(names, _n_hint(h))
        with pipeline_ctx(h, job, force=bool(p.get("force"))) as ctx:
            res = run_stages(ctx, job, names)
        t = res.get("translate") or {}
        return {k: t[k] for k in ("lines", "failed", "missing", "backend") if k in t}

    return run


@flow("stage", heavy=True)
def _build_stage(slug: str, p: dict[str, Any]) -> FlowFn:
    def run(job: JobContext) -> dict[str, Any]:
        h = _handle(slug)
        name = str(p.get("name") or "")
        names = available_stages([name])
        if not names:
            raise AppError("bad_request", params={"field": "name", "value": name}, status=422)
        job.set_stages(names, _n_hint(h))
        with pipeline_ctx(h, job, only=p.get("only") or None, limit=p.get("limit"), force=bool(p.get("force")), sets=p.get("sets")) as ctx:
            res = run_stages(ctx, job, names)
        return {"summary": res.get(name) or {}}

    return run


# ----------------------------------------------------------------------------
# Step 4/5: preview and the full run
# ----------------------------------------------------------------------------

# translation (translate + text_emotion) is step 2 and is not repeated by the preview or the full run
PREVIEW_STAGES = ["voicebank", "synth", "score", "master"]
FULL_STAGES = ["voicebank", "synth", "score", "master", "pack", "report"]


@flow("preview", heavy=True)
def _build_preview(slug: str, p: dict[str, Any]) -> FlowFn:
    def run(job: JobContext) -> dict[str, Any]:
        h = _handle(slug)
        ids = list(p.get("line_ids") or [])
        if not ids:
            raise AppError("nothing_to_review", action="open_translation", status=400)
        names = available_stages(PREVIEW_STAGES)
        job.set_stages(names, len(ids))
        with pipeline_ctx(h, job, only=ids, extra={"preview": True}) as ctx:
            with preserve_status(ctx, names):
                run_stages(ctx, job, names, regen=True)
        return {"line_ids": ids}

    return run


@flow("full", heavy=True)
def _build_full(slug: str, p: dict[str, Any]) -> FlowFn:
    def run(job: JobContext) -> dict[str, Any]:
        h = _handle(slug)
        names = available_stages(FULL_STAGES)
        job.set_stages(names, _n_hint(h))
        with pipeline_ctx(h, job) as ctx:
            res = run_stages(ctx, job, names, regen=True)
        out: dict[str, Any] = {"score": (res.get("score") or {}).get("status", {}), "shutdown": False}
        if p.get("shutdown_when_done"):
            from vndub_app import system

            out["shutdown"] = system.shutdown_pc()
        return out

    return run


# ----------------------------------------------------------------------------
# Step 6: apply review changes
# ----------------------------------------------------------------------------


def _final_state(ctx: Any, ids: list[str]) -> dict[str, tuple[str | None, str | None]]:
    from vndub.util import hash_file

    sel = viewdata.selection(ctx)
    out: dict[str, tuple[str | None, str | None]] = {}
    for lid in ids:
        s = sel.get(lid)
        wav = ctx.paths.final_wav(lid)
        out[lid] = (s.winner if s else None, hash_file(wav) if wav.exists() else None)
    return out


@flow("review_apply", heavy=True)
def _build_review_apply(slug: str, p: dict[str, Any]) -> FlowFn:
    def run(job: JobContext) -> dict[str, Any]:
        h = _handle(slug)
        ids = list(p.get("line_ids") or [])
        if not ids:
            raise AppError("nothing_to_review", action="none", status=400)
        names = available_stages(["synth", "score", "master"])
        job.set_stages([*names, "pack"], len(ids))
        pack_error: dict[str, Any] | None = None
        with pipeline_ctx(h, job, only=ids) as ctx:
            before = _final_state(ctx, ids)
            with preserve_status(ctx, names):
                run_stages(ctx, job, names)
            after = _final_state(ctx, ids)
        changed = [lid for lid in ids if before.get(lid) != after.get(lid)]
        # the patch must contain the new files: re-pack everything (pack is cheap and idempotent)
        try:
            with pipeline_ctx(h, job) as ctx2:
                run_stages(ctx2, job, available_stages(["pack"]))
        except Exception as e:  # the review work is done; report that the patch is stale
            from vndub_app.errors import error_from_exception

            pack_error = error_from_exception(e, stage="pack").to_dict()
            log.warning("pack after review apply failed: %s", e)
        with session.STATE_LOCK:
            st = appdirs.read_app_state(h)
            st["review"]["touched"] = [x for x in st["review"].get("touched", []) if x not in set(ids)]
            st["review"]["last_apply"] = {"changed": len(changed), "lines": changed, "at": now_iso()}
            appdirs.write_app_state(h, st)
        res: dict[str, Any] = {"changed": len(changed), "lines": changed}
        if pack_error:
            res["pack_error"] = pack_error
        return res

    return run


# ----------------------------------------------------------------------------
# Step 3: casting
# ----------------------------------------------------------------------------


def _voicebank_fn(name: str) -> Callable[..., Any]:
    import importlib

    mod = importlib.import_module("vndub.stages.voicebank")
    fn = getattr(mod, name, None)
    if fn is None:
        raise AppError("feature_unavailable", params={"feature": f"voicebank.{name}"}, action="report_problem", details=f"vndub.stages.voicebank.{name} is not implemented in this build")
    return fn


@flow("casting_generate")
def _build_casting_generate(slug: str, p: dict[str, Any]) -> FlowFn:
    def run(job: JobContext) -> dict[str, Any]:
        from vndub_app import casting

        h = _handle(slug)
        sid = str(p["speaker_id"])
        fn = _voicebank_fn("propose_designs")
        job.set_stages(["voicebank"])
        job.begin_stage("voicebank")
        with pipeline_ctx(h, job) as ctx:
            prepare_stage_ctx(ctx, "voicebank")
            new = fn(ctx, sid, n=int(p.get("n") or 3), description=p.get("description"), sample_line_ids=None)
            designs = casting.designs_view(h, ctx, sid)
        viewdata.clear_cache(h.slug)
        return {"designs": designs, "new": [d["design_id"] for d in new]}

    return run


def maybe_train_rvc(h: appdirs.ProjectHandle, sid: str) -> None:
    """Route B needs the character's RVC model: train it in the background if it is not there yet."""
    chars = viewdata.load_characters(session.read_ctx(h))
    c = chars.get(sid)
    if c is None or c.route != "B":
        return
    model = (c.rvc or {}).get("model")
    if model and (h.work / model).exists():
        return
    submit("rvc_train", h.slug, {"speaker_id": sid})


@flow("casting_auto")
def _build_casting_auto(slug: str, p: dict[str, Any]) -> FlowFn:
    """"Автоматический кастинг": propose designs for every character that has none yet, then
    choose the top-ranked one for everybody (voicebank.auto_cast) and start route-B training."""

    def run(job: JobContext) -> dict[str, Any]:
        from vndub_app import casting

        h = _handle(slug)
        propose = _voicebank_fn("propose_designs")
        list_designs = _voicebank_fn("list_designs")
        auto = _voicebank_fn("auto_cast")
        sids = [c["speaker_id"] for c in casting.characters_view(h)]
        job.set_stages(["voicebank"], len(sids))
        job.begin_stage("voicebank")
        generated: list[str] = []
        with pipeline_ctx(h, job) as ctx:
            prepare_stage_ctx(ctx, "voicebank")
            n = int(p.get("n") or ctx.cfg.voicebank.performer.n_designs or 3)
            todo = [sid for sid in sids if not list_designs(ctx, sid)]
            for i, sid in enumerate(todo):
                job.progress(i, len(todo), sid)
                propose(ctx, sid, n=n, description=None, sample_line_ids=None)
                generated.append(sid)
            job.progress(len(todo), len(todo))
            chosen = dict(auto(ctx))
        viewdata.clear_cache(h.slug)
        for sid in chosen:
            maybe_train_rvc(h, sid)
        return {"chosen": chosen, "generated": generated}

    return run


def _listen_route(h: appdirs.ProjectHandle, sid: str) -> str:
    from vndub_app import casting

    return casting.effective_route(h, session.read_ctx(h), sid)


def render_lines(job: JobContext, h: appdirs.ProjectHandle, sid: str, line_ids: list[str], *, tuning: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Synthesize + master the given lines of a character with its chosen performer.

    ``synth.n_candidates=1`` and only the route of the chosen design; the winner is the single
    candidate (no scoring models are loaded: the target is a few seconds), then ``master``. The
    stage summaries of the full run are preserved. Returns ``[{line_id, url|None, error?}]``.
    """
    from vndub_app import casting

    names = available_stages(["synth", "master"])
    job.set_stages(names, len(line_ids))
    route = _listen_route(h, sid)
    if not line_ids:
        raise AppError("translation_missing", params={"speaker_id": sid}, action="open_translation", status=400)
    sets: dict[str, Any] = {"synth.n_candidates": 1, "synth.n_candidates_short": 1, **casting.route_sets(route)}
    out: list[dict[str, Any]] = []
    with casting.temporary_tuning(h, sid, tuning):
        with pipeline_ctx(h, job, only=line_ids, sets=sets, extra={"listen": True}) as ctx:
            with preserve_status(ctx, ["synth", "score", "master"]):
                job.begin_stage("synth")
                from vndub.stages.common import run_stage

                run_stage("synth", ctx)
                casting.select_listen_winners(ctx, line_ids, route)
                if "master" in names:
                    job.begin_stage("master")
                    run_stage("master", ctx)
            fin = viewdata.final_index(ctx)
            for lid in line_ids:
                rec = fin.get(lid)
                wav = ctx.paths.abs(rec.final_wav) if rec else None
                out.append({"line_id": lid, "url": session.work_url(h, wav) if wav and wav.exists() else None})
    return out


@flow("casting_listen")
def _build_casting_listen(slug: str, p: dict[str, Any]) -> FlowFn:
    def run(job: JobContext) -> dict[str, Any]:
        from vndub_app import casting

        h = _handle(slug)
        sid = str(p["speaker_id"])
        ids = casting.pick_lines(h, sid, int(p.get("n") or 3), mode="random", seed=p.get("seed"))
        rendered = {r["line_id"]: r for r in render_lines(job, h, sid, ids)}
        items = casting.listen_items(h, ids, rendered)
        return {"items": items}

    return run


@flow("tuning_listen")
def _build_tuning_listen(slug: str, p: dict[str, Any]) -> FlowFn:
    def run(job: JobContext) -> dict[str, Any]:
        from vndub_app import casting

        h = _handle(slug)
        sid = str(p["speaker_id"])
        tuning = casting.validate_tuning(p.get("tuning") or {})
        picks = casting.pick_tuning_lines(h, sid)
        ids = [lid for lid, _ in picks]
        rendered = {r["line_id"]: r for r in render_lines(job, h, sid, ids, tuning=tuning)}
        items = []
        for lid, kind in picks:
            r = rendered.get(lid, {})
            items.append({"line_id": lid, "kind": kind, "url": r.get("url"), "text_tgt": casting.text_tgt_of(h, lid)})
        return {"items": items, "tuning": tuning}

    return run


@flow("rvc_train")
def _build_rvc_train(slug: str, p: dict[str, Any]) -> FlowFn:
    def run(job: JobContext) -> dict[str, Any]:
        h = _handle(slug)
        sid = str(p["speaker_id"])
        fn = _voicebank_fn("train_rvc")
        job.set_stages(["voicebank"])
        job.begin_stage("voicebank")
        with pipeline_ctx(h, job) as ctx:
            prepare_stage_ctx(ctx, "voicebank")
            res = fn(ctx, sid)
        res = dict(res) if isinstance(res, dict) else {}
        if res.get("state") == "failed":  # the training itself failed (a skip for lack of audio is not a failure)
            raise AppError("stage_failed", params={"stage": "voicebank", "speaker_id": sid}, action="retry", details=str(res.get("reason") or res.get("reason_code") or ""))
        return {"speaker_id": sid, "state": res.get("state"), "reason_code": res.get("reason_code")}

    return run


# ----------------------------------------------------------------------------
# problem report
# ----------------------------------------------------------------------------

REPORT_MAX_FILE = 5 * 1024 * 1024
REPORT_JOB_LOGS = 5  # the newest per-job log files (logs/jobs/) that go into the report


def build_report_zip(slug: str | None, dest_dir: Path | None = None) -> Path:
    """Zip logs (``app.log*``, the newest 5 per-job logs) + ``stage_status.json`` + ``project.yaml`` + ``app_state.json`` (no audio, no game data)."""
    dest_dir = dest_dir or appdirs.sub("reports")
    dest_dir.mkdir(parents=True, exist_ok=True)
    ts = now_iso().replace(":", "").replace("-", "")[:15]
    out = dest_dir / f"vndub-report-{slug or 'app'}-{ts}.zip"

    def add(zf: zipfile.ZipFile, path: Path, arc: str) -> None:
        try:
            if path.is_file() and path.stat().st_size <= REPORT_MAX_FILE:
                zf.write(path, arc)
            elif path.is_file():  # keep the tail of a big log
                data = path.read_bytes()[-REPORT_MAX_FILE:]
                zf.writestr(arc, data)
        except OSError:
            pass

    jobs.flush_logs()  # the running jobs (this report job among them) have lines in their buffers
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(appdirs.logs_dir().glob("app.log*")):
            add(zf, f, f"logs/{f.name}")
        add(zf, appdirs.logs_dir() / "reaper.log", "logs/reaper.log")  # what the reaper ended after the app was gone
        job_logs = appdirs.logs_dir() / "jobs"
        for f in sorted(job_logs.glob("*.log"), key=lambda p: p.name)[-REPORT_JOB_LOGS:] if job_logs.exists() else []:
            add(zf, f, f"logs/jobs/{f.name}")
        add(zf, appdirs.settings_file(), "settings.json")
        if slug:
            h = appdirs.project_handle(slug)
            add(zf, h.work / "stage_status.json", "stage_status.json")
            add(zf, h.work / "project.yaml", "project.yaml")
            add(zf, h.state_file, "app_state.json")
            add(zf, h.config_file, "pipeline.yaml")
            add(zf, h.work / "review_queue.jsonl", "review_queue.jsonl")
            for f in sorted((h.work / "logs").glob("*.log")) if (h.work / "logs").exists() else []:
                add(zf, f, f"work_logs/{f.name}")
    return out


@flow("report", gpu=False, project=False)
def _build_report(slug: str, p: dict[str, Any]) -> FlowFn:
    def run(job: JobContext) -> dict[str, Any]:
        job.message("jobs.msg.report_zip")
        return {"path": str(build_report_zip(slug or p.get("slug") or None))}

    return run


# ----------------------------------------------------------------------------
# helpers used by routes
# ----------------------------------------------------------------------------


def first_scene_ids(h: appdirs.ProjectHandle, limit: int = 100) -> list[str]:
    snap = viewdata.snapshot(h)
    dub = snap.dubbable()
    if not dub:
        return []
    first = min(dub, key=lambda l: l.order).scene
    return [l.line_id for l in sorted(dub, key=lambda l: l.order) if l.scene == first][:limit]


def first_n_ids(h: appdirs.ProjectHandle, n: int) -> list[str]:
    snap = viewdata.snapshot(h)
    return [l.line_id for l in sorted(snap.dubbable(), key=lambda l: l.order)][: max(1, int(n))]


def random_sample(ids: list[str], n: int, seed: int | None = None) -> list[str]:
    rnd = random.Random(seed)
    return sorted(rnd.sample(ids, min(n, len(ids))))

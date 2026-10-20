"""``vndub`` command line.

Stages: detect extract parse link analyze translate voicebank synth score master pack report
Other:  run (all stages), review (Gradio UI), models (pull/status/selftest/list), tools (check), index (rebuild)

Every stage command takes ``--project DIR`` (the game dir, or a dir holding a
``project.yaml`` with ``game_dir``), ``--config``, ``--force``, ``--only id,..``,
``--limit N``, ``--work-dir``, ``--out-dir`` and ``--set key=value`` overrides.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import typer
import yaml
from rich.table import Table

from vndub import __version__
from vndub.backends.base import BackendUnavailable
from vndub.config import PipelineConfig, load_config, load_project_yaml
from vndub.paths import WorkPaths
from vndub.stages.common import STAGES, StageContext, StageError, console, run_stage
from vndub.tools import Tools, ToolMissing
from vndub.util import repo_root

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__, rich_markup_mode="rich")
models_app = typer.Typer(no_args_is_help=True, help="Model weights: pull, status, selftest, list.")
tools_app = typer.Typer(no_args_is_help=True, help="External CLI tools (tools.yaml).")
index_app = typer.Typer(no_args_is_help=True, help="SQLite index (rebuildable cache over work/).")
app.add_typer(models_app, name="models")
app.add_typer(tools_app, name="tools")
app.add_typer(index_app, name="index")


def _default_config() -> Path:
    local = Path("configs/pipeline.yaml")
    return local if local.exists() else repo_root() / "configs" / "pipeline.yaml"


# ----------------------------------------------------------------------------
# context construction
# ----------------------------------------------------------------------------


def _parse_sets(sets: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for s in sets or []:
        if "=" not in s:
            raise typer.BadParameter(f"--set expects key=value, got {s!r}")
        k, v = s.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def load_cfg(config: Path | None, sets: list[str] | None = None) -> PipelineConfig:
    path = config or _default_config()
    return load_config(path, _parse_sets(sets or []))


def make_ctx(
    project: Path,
    config: Path | None,
    *,
    force: bool = False,
    only: str | None = None,
    limit: int | None = None,
    work_dir: Path | None = None,
    out_dir: Path | None = None,
    sets: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> StageContext:
    from vndub import gpu

    cfg = load_cfg(config, sets)
    project = project.resolve()
    work = (work_dir or Path(cfg.project.work_dir)).resolve()
    out = (out_dir or Path(cfg.project.out_dir)).resolve()
    paths = WorkPaths.create(work, out)
    # project dir may carry a user-written project.yaml with game_dir
    game_dir = project
    user_yaml = project / "project.yaml"
    if user_yaml.exists():
        gd = load_project_yaml(user_yaml).game_dir
        game_dir = (project / gd).resolve() if gd and not Path(gd).is_absolute() else Path(gd or project)
    if paths.project_yaml.exists():
        existing = load_project_yaml(paths.project_yaml)
        if existing.game_dir and Path(existing.game_dir).resolve() != game_dir.resolve() and not (extra or {}).get("rebind_project"):
            raise StageError(
                f"work dir {work} belongs to game {existing.game_dir}, not {game_dir}. "
                "Use --work-dir/--out-dir for a separate project (or delete the work dir)."
            )
    tools = Tools.load(cfg.resolve(cfg.tools_file), cfg.tools)
    models = gpu.from_config(cfg, paths, tools)
    return StageContext(
        cfg=cfg,
        paths=paths,
        tools=tools,
        project_dir=game_dir,
        models=models,
        force=force,
        only=set(x.strip() for x in only.split(",") if x.strip()) if only else None,
        limit=limit,
        extra=extra or {},
    )


def _run(names: list[str], ctx: StageContext) -> None:
    try:
        for n in names:
            run_stage(n, ctx)
    except (StageError, BackendUnavailable, ToolMissing, FileNotFoundError) as e:
        console.print(f"[bold red]error:[/bold red] {e}")
        raise typer.Exit(2)
    finally:
        ctx.models.close()
        _print_substitutions(ctx)


def _print_substitutions(ctx: StageContext) -> None:
    subs = ctx.models.substitutions
    if subs:
        console.print("[yellow]backend substitutions (profile=test):[/yellow]")
        seen = set()
        for s in subs:
            if (s.requested, s.used) in seen:
                continue
            seen.add((s.requested, s.used))
            console.print(f"  {s.requested} -> {s.used}: {s.reason}")


# common option declarations
ProjectOpt = typer.Option(..., "--project", "-p", help="Game dir (or dir with project.yaml).", exists=True, file_okay=False)
ConfigOpt = typer.Option(None, "--config", "-c", help="Pipeline config (default configs/pipeline.yaml).")
ForceOpt = typer.Option(False, "--force", help="Ignore caches and recompute.")
OnlyOpt = typer.Option(None, "--only", help="Comma-separated line_ids to process.")
LimitOpt = typer.Option(None, "--limit", help="Process at most N lines.")
WorkOpt = typer.Option(None, "--work-dir", help="Override project.work_dir.")
OutOpt = typer.Option(None, "--out-dir", help="Override project.out_dir.")
SetOpt = typer.Option(None, "--set", help="Config override key=value (dotted keys), repeatable.")


def _stage(name: str, helptext: str) -> None:
    def cmd(
        project: Path = ProjectOpt,
        config: Optional[Path] = ConfigOpt,
        force: bool = ForceOpt,
        only: Optional[str] = OnlyOpt,
        limit: Optional[int] = LimitOpt,
        work_dir: Optional[Path] = WorkOpt,
        out_dir: Optional[Path] = OutOpt,
        set_: Optional[list[str]] = SetOpt,
    ) -> None:
        ctx = make_ctx(project, config, force=force, only=only, limit=limit, work_dir=work_dir, out_dir=out_dir, sets=set_)
        _run([name], ctx)

    cmd.__doc__ = helptext
    app.command(name, help=helptext)(cmd)


# Commands are registered in pipeline order (typer lists them in registration order).


@app.command("detect", help="Identify the engine and write work/project.yaml.")
def detect_cmd(
    project: Path = ProjectOpt,
    config: Optional[Path] = ConfigOpt,
    engine: Optional[str] = typer.Option(None, "--engine", help="Force the engine instead of auto-detect."),
    force: bool = ForceOpt,
    work_dir: Optional[Path] = WorkOpt,
    out_dir: Optional[Path] = OutOpt,
    set_: Optional[list[str]] = SetOpt,
) -> None:
    ctx = make_ctx(project, config, force=force, work_dir=work_dir, out_dir=out_dir, sets=set_, extra={"engine": engine, "rebind_project": force})
    _run(["detect"], ctx)


_stage("extract", "Extract archives/scripts/voices; decode voices to work/audio/{src,src16}.")


@app.command("parse", help="Scripts -> manifest/lines.jsonl + manifest/characters.yaml.")
def parse_cmd(
    project: Path = ProjectOpt,
    config: Optional[Path] = ConfigOpt,
    import_vntextpatch: Optional[Path] = typer.Option(None, "--import-vntextpatch", help="VNTextPatch extractlocal JSON dir (text + names)."),
    force: bool = ForceOpt,
    only: Optional[str] = OnlyOpt,
    limit: Optional[int] = LimitOpt,
    work_dir: Optional[Path] = WorkOpt,
    out_dir: Optional[Path] = OutOpt,
    set_: Optional[list[str]] = SetOpt,
) -> None:
    ctx = make_ctx(project, config, force=force, only=only, limit=limit, work_dir=work_dir, out_dir=out_dir, sets=set_, extra={"import_vntextpatch": str(import_vntextpatch) if import_vntextpatch else ""})
    _run(["parse"], ctx)


_stage("link", "Verify line<->voice mapping: ASR + fuzzy match + speaker clustering.")
_stage("analyze", "Expression passport per voiced line (emotion, A/D/V, events, prosody, SNR).")


@app.command("translate", help="Import a translation or translate with the local LLM; stress + normalization.")
def translate_cmd(
    project: Path = ProjectOpt,
    config: Optional[Path] = ConfigOpt,
    seed_glossary: bool = typer.Option(False, "--seed-glossary", help="Create/extend configs/glossary.yaml from characters.yaml and exit."),
    real: bool = typer.Option(False, "--real", help="Test profile: use the real LLM instead of MockTranslator."),
    force: bool = ForceOpt,
    only: Optional[str] = OnlyOpt,
    limit: Optional[int] = LimitOpt,
    work_dir: Optional[Path] = WorkOpt,
    out_dir: Optional[Path] = OutOpt,
    set_: Optional[list[str]] = SetOpt,
) -> None:
    ctx = make_ctx(project, config, force=force, only=only, limit=limit, work_dir=work_dir, out_dir=out_dir, sets=set_, extra={"seed_glossary": seed_glossary, "real": real})
    _run(["translate"], ctx)


_stage("voicebank", "Per character: reference clips, performer voice, RVC model, LoRA (route C).")


@app.command("synth", help="N candidates per voiced line per enabled backend and route.")
def synth_cmd(
    project: Path = ProjectOpt,
    config: Optional[Path] = ConfigOpt,
    regen: bool = typer.Option(False, "--regen", help="Only lines whose selection status is regen (new seeds/backends/short text)."),
    force: bool = ForceOpt,
    only: Optional[str] = OnlyOpt,
    limit: Optional[int] = LimitOpt,
    work_dir: Optional[Path] = WorkOpt,
    out_dir: Optional[Path] = OutOpt,
    set_: Optional[list[str]] = SetOpt,
) -> None:
    ctx = make_ctx(project, config, force=force, only=only, limit=limit, work_dir=work_dir, out_dir=out_dir, sets=set_, extra={"regen": regen})
    _run(["synth"], ctx)


@app.command("score", help="Score every candidate, apply gates, select winners -> selection.jsonl.")
def score_cmd(
    project: Path = ProjectOpt,
    config: Optional[Path] = ConfigOpt,
    reweight: bool = typer.Option(False, "--reweight", help="Recompute composites/selection from persisted metrics only."),
    force: bool = ForceOpt,
    only: Optional[str] = OnlyOpt,
    limit: Optional[int] = LimitOpt,
    work_dir: Optional[Path] = WorkOpt,
    out_dir: Optional[Path] = OutOpt,
    set_: Optional[list[str]] = SetOpt,
) -> None:
    ctx = make_ctx(project, config, force=force, only=only, limit=limit, work_dir=work_dir, out_dir=out_dir, sets=set_, extra={"reweight": reweight})
    _run(["score"], ctx)


_stage("master", "Trim, stretch, loudness-normalize and encode winners to the source codec.")
_stage("pack", "Pack final voice files into the engine's patch format + INSTALL.md.")
_stage("report", "HTML + Markdown report: coverage, emotions, scores, routes, review queue, VRAM.")


@app.command("run", help="Run all stages in order (score -> synth regen loop included).")
def run_cmd(
    project: Path = ProjectOpt,
    config: Optional[Path] = ConfigOpt,
    from_: Optional[str] = typer.Option(None, "--from", help="First stage to run."),
    to: Optional[str] = typer.Option(None, "--to", help="Last stage to run."),
    profile: Optional[str] = typer.Option(None, "--profile", help="'test' = configs/pipeline.test.yaml (CPU/tiny profile)."),
    force: bool = ForceOpt,
    only: Optional[str] = OnlyOpt,
    limit: Optional[int] = LimitOpt,
    work_dir: Optional[Path] = WorkOpt,
    out_dir: Optional[Path] = OutOpt,
    set_: Optional[list[str]] = SetOpt,
) -> None:
    if profile == "test":
        config = (config.parent if config else _default_config().parent) / "pipeline.test.yaml"
    elif profile not in (None, "default"):
        raise typer.BadParameter("--profile must be 'default' or 'test'")
    ctx = make_ctx(project, config, force=force, only=only, limit=limit, work_dir=work_dir, out_dir=out_dir, sets=set_, extra={"rebind_project": False})
    i0 = STAGES.index(from_) if from_ else 0
    i1 = STAGES.index(to) if to else len(STAGES) - 1
    names = STAGES[i0 : i1 + 1]
    try:
        for n in names:
            run_stage(n, ctx)
            if n == "score":
                _regen_loop(ctx)
    except (StageError, BackendUnavailable, ToolMissing, FileNotFoundError) as e:
        console.print(f"[bold red]error:[/bold red] {e}")
        raise typer.Exit(2)
    finally:
        ctx.models.close()
        _print_substitutions(ctx)
    console.print(f"[green]done.[/green] work: {ctx.paths.work}  out: {ctx.paths.out}")


def _regen_loop(ctx: StageContext) -> None:
    from vndub.manifest.io import SelectionRecord, read_jsonl

    for rnd in range(1, ctx.cfg.score.regen_rounds + 1):
        sel = read_jsonl(ctx.paths.selection, SelectionRecord)
        todo = [s.line_id for s in sel if s.status == "regen"]
        if not todo:
            return
        console.print(f"[cyan]regen round {rnd}: {len(todo)} line(s)[/cyan]")
        ctx.extra["regen"] = True
        ctx.extra["regen_round"] = rnd
        run_stage("synth", ctx)
        run_stage("score", ctx)
    ctx.extra["regen"] = False


@app.command("review", help="Gradio review UI: original vs candidates, overrides.jsonl.")
def review_cmd(
    project: Path = ProjectOpt,
    config: Optional[Path] = ConfigOpt,
    port: Optional[int] = typer.Option(None, "--port"),
    host: Optional[str] = typer.Option(None, "--host"),
    work_dir: Optional[Path] = WorkOpt,
    out_dir: Optional[Path] = OutOpt,
    no_browser: bool = typer.Option(True, "--no-browser/--browser"),
    set_: Optional[list[str]] = SetOpt,
) -> None:
    from vndub.review.app import launch

    ctx = make_ctx(project, config, work_dir=work_dir, out_dir=out_dir, sets=set_)
    launch(ctx, host=host or ctx.cfg.review.host, port=port or ctx.cfg.review.port, open_browser=not no_browser)


# ----------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------


@models_app.command("list", help="All registered backends with role, env, license and VRAM estimate.")
def models_list(
    role: Optional[str] = typer.Option(None, "--role"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Also show license and description."),
) -> None:
    from vndub.backends import registry

    t = Table()
    t.add_column("name", no_wrap=True)
    for c in ("role", "env", "VRAM GB", "commercial", "profiles"):
        t.add_column(c, no_wrap=True)
    if verbose:
        t.add_column("license", overflow="fold")
        t.add_column("description", overflow="fold")
    for s in sorted(registry.all_specs().values(), key=lambda s: (s.role, s.name)):
        if role and s.role != role:
            continue
        row = [s.name, s.role, s.env or "core", f"{s.vram_estimate_gb:.1f}", "yes" if s.commercial_ok else "no", ",".join(s.profiles)]
        if verbose:
            row += [s.license, s.description]
        t.add_row(*row)
    console.print(t)


@models_app.command("pull", help="Download weights (once; cached under models/). Names or --profile.")
def models_pull(
    names: Optional[list[str]] = typer.Argument(None),
    profile: Optional[str] = typer.Option(None, "--profile", help="default | test | optional | all"),
    config: Optional[Path] = ConfigOpt,
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    from vndub import models

    cfg = load_cfg(config)
    if not names and not profile:
        raise typer.BadParameter("give backend names or --profile")
    rows = models.pull(cfg, names or [], profile, dry_run=dry_run, console=console)
    _print_rows(rows, ["name", "weight", "status", "size_gb", "detail"])
    if any(r["status"] == "error" for r in rows):
        raise typer.Exit(1)


@models_app.command("status", help="Which weights are present, size on disk, VRAM estimate, usability.")
def models_status(
    profile: str = typer.Option("default", "--profile"),
    config: Optional[Path] = ConfigOpt,
) -> None:
    from vndub import models

    cfg = load_cfg(config)
    rows = models.status(cfg, profile)
    _print_rows(rows, ["name", "role", "env", "weights", "size_gb", "vram_gb", "license", "usable", "reason"])


@models_app.command("selftest", help="Load a backend, process a 2 s clip, report peak VRAM, unload.")
def models_selftest(
    names: list[str] = typer.Argument(...),
    config: Optional[Path] = ConfigOpt,
) -> None:
    from vndub import models

    cfg = load_cfg(config)
    failed = False
    for n in names:
        res = models.selftest(cfg, n)
        failed |= not res.get("ok")
        color = "green" if res.get("ok") else "red"
        console.print(f"[{color}]{n}[/{color}]: {res}")
    if failed:
        raise typer.Exit(1)


def _print_rows(rows: list[dict[str, Any]], cols: list[str]) -> None:
    t = Table(*cols)
    for r in rows:
        t.add_row(*[str(r.get(c, "")) for c in cols])
    console.print(t)


# ----------------------------------------------------------------------------
# tools / index
# ----------------------------------------------------------------------------


@tools_app.command("check", help="Verify external tools from tools.yaml; print download URLs.")
def tools_check(
    engine: Optional[str] = typer.Option(None, "--engine"),
    config: Optional[Path] = ConfigOpt,
) -> None:
    cfg = load_cfg(config)
    tools = Tools.load(cfg.resolve(cfg.tools_file), cfg.tools)
    rows = tools.check(engine)
    t = Table("tool", "present", "required", "engines", "path", "purpose / download")
    for r in rows:
        mark = "[green]yes[/green]" if r["present"] else ("[red]MISSING[/red]" if r["required"] else "[yellow]no[/yellow]")
        info = r["purpose"] + ("" if r["present"] else f"\n{r['url']}" + (f"\npip install {r['pip']}" if r["pip"] else ""))
        t.add_row(r["tool"], mark, "yes" if r["required"] else "", r["engines"], r["path"], info)
    console.print(t)
    if any(r["required"] and not r["present"] for r in rows):
        raise typer.Exit(1)


@index_app.command("rebuild", help="Rebuild work/index.sqlite from the JSONL manifests.")
def index_rebuild(
    project: Path = ProjectOpt,
    config: Optional[Path] = ConfigOpt,
    work_dir: Optional[Path] = WorkOpt,
) -> None:
    from vndub.manifest import index

    ctx = make_ctx(project, config, work_dir=work_dir)
    p = index.rebuild(ctx.paths)
    console.print(f"index rebuilt: {p}")


@app.command("version", help="Print the vndub version.")
def version_cmd() -> None:
    console.print(__version__)


def main() -> None:
    try:
        app()
    except StageError as e:
        console.print(f"[bold red]error:[/bold red] {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()

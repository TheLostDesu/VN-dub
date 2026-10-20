"""``vndub models pull | status | selftest``.

Weights are downloaded once (Hugging Face snapshots, direct URLs, or an
env-side ``pull`` hook for packages that fetch their own checkpoints) into
``models/``; afterwards the pipeline runs offline (``HF_HUB_OFFLINE=1`` is set
for adapters). Post-processing steps (e.g. Whisper -> CTranslate2) run inside
the backend's env.
"""

from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path
from typing import Any

from vndub.backends import registry
from vndub.backends.base import BackendSpec, BackendUnavailable, WeightSpec
from vndub.backends.runner import AdapterError, AdapterProcess, env_exists
from vndub.backends.weights import MARKER, dir_size_gb, is_present, local_dir, resolved
from vndub.config import PipelineConfig
from vndub.gpu import resolve_models_dir
from vndub.util import now_iso, repo_root


def _select(names: list[str], profile: str | None) -> list[BackendSpec]:
    specs: list[BackendSpec] = []
    if profile:
        specs += registry.for_profile(profile)
    for n in names:
        s = registry.get(n)
        if s not in specs:
            specs.append(s)
    return specs


def _mark(d: Path, w: WeightSpec) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / MARKER).write_text(f"{now_iso()} {w.kind} {w.repo or w.url or w.post}\n", encoding="utf-8")


def _env_call(spec: BackendSpec, models_dir: Path, method: str, params: dict[str, Any], timeout: float = 7200) -> Any:
    if spec.env is None:
        raise BackendUnavailable(spec.name, "post-processing requires a backend env")
    if not env_exists(spec.env):
        raise BackendUnavailable(spec.name, f"env '{spec.env}' not created", f"scripts/make_envs.ps1 -Env {spec.env}")
    proc = AdapterProcess(
        spec.env,
        spec.name,
        log_path=repo_root() / "models" / "pull.log",
        extra_env={"VNDUB_MODELS_DIR": str(models_dir), "HF_HOME": str(models_dir / "hf_home"), "TORCH_HOME": str(models_dir / "torch_home")},
    )
    proc.start()
    try:
        return proc.request(method, params, timeout=timeout)
    finally:
        proc.close()


def _pull_one(spec: BackendSpec, w: WeightSpec, models_dir: Path, console: Any = None) -> dict[str, Any]:
    d = local_dir(models_dir, w, spec.name)
    row = {"name": spec.name, "weight": w.repo or w.url or w.subdir or w.kind, "status": "present", "size_gb": "", "detail": str(d)}
    if is_present(models_dir, w, spec.name):
        row["size_gb"] = f"{dir_size_gb(d):.2f}"
        return row
    t0 = time.perf_counter()
    if w.kind in ("hf", "hf_file"):
        from huggingface_hub import snapshot_download

        if console:
            console.print(f"[cyan]downloading[/cyan] {w.repo}{'@' + w.revision if w.revision else ''} -> {d}")
        snapshot_download(repo_id=w.repo, revision=w.revision, local_dir=str(d), allow_patterns=list(w.allow) or None)
        if w.post:
            if console:
                console.print(f"[cyan]post-processing[/cyan] {w.post} in env {spec.env}")
            params = {"name": spec.name, "model": spec.adapter, "options": spec.options, "models_dir": str(models_dir), "weights": resolved(models_dir, spec), "post": w.post, "local_dir": str(d)}
            _env_call(spec, models_dir, "pull", params)
        _mark(d, w)
    elif w.kind == "url":
        d.mkdir(parents=True, exist_ok=True)
        dst = d / (w.filename or w.url.rsplit("/", 1)[-1])
        if console:
            console.print(f"[cyan]downloading[/cyan] {w.url}")
        urllib.request.urlretrieve(w.url, dst)
        _mark(d, w)
    elif w.kind == "env":
        if console:
            console.print(f"[cyan]env pull[/cyan] {spec.name} ({w.post}) in env {spec.env}")
        params = {"name": spec.name, "model": spec.adapter, "options": spec.options, "models_dir": str(models_dir), "weights": resolved(models_dir, spec), "post": w.post, "local_dir": str(d)}
        _env_call(spec, models_dir, "pull", params)
        _mark(d, w)
    elif w.kind == "builtin":
        return {**row, "status": "builtin"}
    else:
        raise ValueError(f"unknown weight kind {w.kind}")
    row.update(status="pulled", size_gb=f"{dir_size_gb(d):.2f}", detail=f"{time.perf_counter() - t0:.0f}s -> {d}")
    return row


def pull(cfg: PipelineConfig, names: list[str], profile: str | None, dry_run: bool = False, console: Any = None) -> list[dict[str, Any]]:
    models_dir = resolve_models_dir(cfg)
    models_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    done: set[str] = set()
    for spec in _select(names, profile):
        if not spec.weights:
            rows.append({"name": spec.name, "weight": "-", "status": "no weights", "size_gb": "", "detail": "core/local backend"})
            continue
        for w in spec.weights:
            key = str(local_dir(models_dir, w, spec.name))
            if key in done:
                continue
            done.add(key)
            if dry_run:
                rows.append({"name": spec.name, "weight": w.repo or w.url or w.subdir, "status": "present" if is_present(models_dir, w, spec.name) else "would pull", "size_gb": f"~{w.size_gb:.1f}", "detail": key})
                continue
            try:
                rows.append(_pull_one(spec, w, models_dir, console))
            except (BackendUnavailable, AdapterError, OSError, ValueError, RuntimeError) as e:
                rows.append({"name": spec.name, "weight": w.repo or w.url or w.subdir, "status": "error", "size_gb": "", "detail": str(e)[:300]})
    return rows


def status(cfg: PipelineConfig, profile: str) -> list[dict[str, Any]]:
    models_dir = resolve_models_dir(cfg)
    rows = []
    for spec in registry.for_profile(profile):
        present = sum(1 for w in spec.weights if is_present(models_dir, w, spec.name))
        size = sum(dir_size_gb(local_dir(models_dir, w, spec.name)) for w in spec.weights if w.kind != "builtin")
        u = registry.usable(spec, policy=cfg.license_policy, models_dir=models_dir)
        rows.append(
            {
                "name": spec.name,
                "role": spec.role,
                "env": spec.env or "core",
                "weights": f"{present}/{len(spec.weights)}" if spec.weights else "-",
                "size_gb": f"{size:.2f}",
                "vram_gb": f"{spec.vram_estimate_gb:.1f}",
                "license": spec.license,
                "usable": "yes" if u.ok else "no",
                "reason": (u.reason + (f" -> {u.hint}" if u.hint else "")) if not u.ok else "",
            }
        )
    return rows


def selftest(cfg: PipelineConfig, name: str) -> dict[str, Any]:
    """Load, process a 2 s clip, report peak VRAM, unload. Never raises for missing pieces."""
    try:
        spec = registry.get(name)
    except KeyError as e:
        return {"ok": False, "error": str(e)}
    models_dir = resolve_models_dir(cfg)
    u = registry.usable(spec, policy=cfg.license_policy, models_dir=models_dir)
    if not u.ok:
        msg = u.reason
        if "weights missing" in u.reason:
            msg = f"weights missing: run `vndub models pull {name}`"
        return {"ok": False, "error": msg, "hint": u.hint}
    from vndub.gpu import SystemCuda
    from vndub.tools import Tools

    device = cfg.gpu.device if (spec.heavy and SystemCuda().available()) else "cpu"
    tools = Tools.load(cfg.resolve(cfg.tools_file), cfg.tools)
    log_dir = repo_root() / "models"
    try:
        b = registry.instantiate(spec, device=device, models_dir=models_dir, log_dir=log_dir, context={"tools": tools})
        res = b.selftest() or {}
        res.setdefault("ok", True)
        res["device"] = device
        return res
    except BackendUnavailable as e:
        return {"ok": False, "error": str(e)}
    except AdapterError as e:
        return {"ok": False, "error": str(e), "data": getattr(e, "data", None)}
    except Exception as e:  # keep selftest output readable
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

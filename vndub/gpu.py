"""GPU ModelRegistry: the one-resident-heavy-model rule and the VRAM budget.

Stages call ``acquire(name)`` once per model and process all lines under it::

    asr = ctx.models.acquire(ctx.cfg.link.asr)
    results = asr.transcribe(items, lang="ja")
    # next acquire() of another heavy model unloads this one first

``acquire`` resolves the backend (license policy, platform, env, weights,
configured fallback chain, and -- only in the test profile -- the mock),
unloads any other resident heavy model (adapter ``unload`` -> process exit,
which frees all of its VRAM; the adapter calls ``torch.cuda.empty_cache``
first), checks ``vram_estimate_gb`` against the budget and the free memory
reported by CUDA, and records load/unload timings and peak VRAM for the report.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

from vndub.backends import registry
from vndub.backends.base import Backend, BackendSpec, BackendUnavailable, estimate_vram_gb
from vndub.manifest.io import append_jsonl
from vndub.util import get_logger, now_iso, repo_root

GB = 1024**3


class CudaProbe(Protocol):
    def available(self) -> bool: ...

    def mem_get_info(self) -> tuple[int, int]: ...  # (free, total) bytes

    def empty_cache(self) -> None: ...


def find_nvidia_smi() -> str | None:
    """``nvidia-smi`` on PATH, else where the driver installs it (an app started with a stripped PATH would
    otherwise decide "no CUDA" and run everything it may on the CPU; the Doctor looks in the same places)."""
    found = shutil.which("nvidia-smi")
    if found:
        return found
    if sys.platform.startswith("win"):
        candidates = (
            Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe",
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe",
        )
        for p in candidates:
            if p.exists():
                return str(p)
    return None


class SystemCuda:
    """CUDA info without torch in the core env: ``nvidia-smi`` query (cached briefly)."""

    def __init__(self, device_index: int = 0) -> None:
        self.index = device_index
        self._exe = find_nvidia_smi()
        self._cache: tuple[float, tuple[int, int]] | None = None

    def available(self) -> bool:
        if self._exe is None:
            return False
        try:
            self.mem_get_info()
            return True
        except Exception:
            return False

    def mem_get_info(self) -> tuple[int, int]:
        now = time.monotonic()
        if self._cache and now - self._cache[0] < 1.0:
            return self._cache[1]
        if self._exe is None:
            raise RuntimeError("nvidia-smi not found")
        r = subprocess.run(
            [self._exe, f"--id={self.index}", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode != 0 or not r.stdout.strip():
            raise RuntimeError(f"nvidia-smi failed: {r.stderr.strip()}")
        free_mb, total_mb = (int(x.strip()) for x in r.stdout.strip().splitlines()[0].split(","))
        val = (free_mb * 1024 * 1024, total_mb * 1024 * 1024)
        self._cache = (now, val)
        return val

    def empty_cache(self) -> None:  # VRAM is released when the adapter process exits
        self._cache = None


class NoCuda:
    def available(self) -> bool:
        return False

    def mem_get_info(self) -> tuple[int, int]:
        return (0, 0)

    def empty_cache(self) -> None:
        pass


class VramBudgetExceeded(BackendUnavailable):
    pass


@dataclass
class Resident:
    name: str
    spec: BackendSpec
    backend: Backend
    options_key: str
    loaded_at: float
    load_s: float
    substituted_for: str | None = None


@dataclass
class Substitution:
    requested: str
    used: str
    reason: str


BackendFactory = Callable[[BackendSpec, dict[str, Any], str], Backend]


@dataclass
class ModelRegistry:
    budget_gb: float = 7.5
    safety_margin_gb: float = 0.5
    device: str = "cuda:0"
    allow_cpu_fallback: bool = False
    license_policy: str = "research"
    fallback_to_mock: bool = False
    models_dir: Path = field(default_factory=lambda: repo_root() / "models")
    log_dir: Path | None = None
    timings_path: Path | None = None
    cuda: CudaProbe = field(default_factory=SystemCuda)
    factory: BackendFactory | None = None
    context: dict[str, Any] = field(default_factory=dict)
    check_usable: bool = True
    root: Path | None = None
    # tests: always use a backend's mock when it has one (deterministic, no envs)
    force_mock: bool = False

    def __post_init__(self) -> None:
        self.heavy: Resident | None = None
        self.light: dict[str, Resident] = {}
        self.substitutions: list[Substitution] = []
        self.events: list[dict[str, Any]] = []
        self.log = get_logger("gpu", (self.log_dir / "gpu.log") if self.log_dir else None)

    # -- resolution -----------------------------------------------------------
    def resolve(self, name: str) -> tuple[BackendSpec, Substitution | None]:
        """Pick the backend to run for ``name`` following fallbacks; raise with all reasons."""
        reasons: list[str] = []
        seen: set[str] = set()
        cur: str | None = name
        first = registry.get(name)
        if self.force_mock and first.mock:
            return registry.get(first.mock), Substitution(name, first.mock, "force_mock")
        while cur and cur not in seen:
            seen.add(cur)
            spec = registry.get(cur)
            if not self.check_usable:
                return spec, (Substitution(name, cur, "; ".join(reasons)) if cur != name else None)
            u = registry.usable(spec, policy=self.license_policy, models_dir=self.models_dir, root=self.root)
            if u.ok and not (spec.heavy and not self._cuda_ok() and not self._cpu_allowed(spec)):
                return spec, (Substitution(name, cur, "; ".join(reasons)) if cur != name else None)
            if u.ok:
                reasons.append(f"{cur}: CUDA not available" + ("" if spec.cpu_ok else " and the adapter needs a GPU"))
            else:
                reasons.append(f"{cur}: {u.reason}" + (f" ({u.hint})" if u.hint else ""))
            cur = spec.fallback
        if self.fallback_to_mock and first.mock:
            mock = registry.get(first.mock)
            sub = Substitution(name, mock.name, "; ".join(reasons))
            return mock, sub
        raise BackendUnavailable(name, "; ".join(reasons) or "unusable")

    def _cuda_ok(self) -> bool:
        try:
            return self.cuda.available()
        except Exception:
            return False

    def _cpu_allowed(self, spec: BackendSpec) -> bool:
        return self.allow_cpu_fallback and spec.cpu_ok

    # -- acquire / release ----------------------------------------------------
    def acquire(self, name: str, options: dict[str, Any] | None = None, *, substituted_for: str | None = None, reason: str = "") -> Backend:
        """Load ``name`` (or what its fallback chain resolves to) and make it resident.

        ``substituted_for`` / ``reason``: a stage that walked its own list of candidates (the translator's
        ``fallbacks``) names the backend it wanted first, so the substitution is logged and lands in the report
        like one the registry made itself.
        """
        spec, sub = self.resolve(name)
        options = dict(options or {})
        if substituted_for and substituted_for != spec.name:
            sub = Substitution(substituted_for, spec.name, reason or (sub.reason if sub else ""))
        key = repr(sorted(options.items()))
        for res in ([self.heavy] if self.heavy else []) + list(self.light.values()):
            if res.name == spec.name and res.options_key == key:
                return res.backend
        if sub is not None:
            self.substitutions.append(sub)
            self.log.warning("backend substitution: %s -> %s (%s)", sub.requested, sub.used, sub.reason)
        device = "cpu"
        need = estimate_vram_gb(spec, options)
        if spec.heavy:
            self.release_heavy()
            if self._cuda_ok():
                self._check_budget(spec, need)
                device = self.device
            elif not self._cpu_allowed(spec):
                raise BackendUnavailable(spec.name, "CUDA not available", "set gpu.allow_cpu_fallback: true to run on CPU (slow)")
        elif spec.name in self.light:
            self.release(spec.name)
        backend = self._make(spec, options, device)
        t0 = time.perf_counter()
        try:
            info = backend.load() or {}
        except BaseException:
            try:
                backend.unload()  # a failed load must not leave the adapter process (and its CUDA context) behind
            except Exception:
                pass
            raise
        load_s = time.perf_counter() - t0
        self._check_device(spec, backend, device, info)
        res = Resident(spec.name, spec, backend, key, time.time(), load_s, sub.requested if sub else None)
        if spec.heavy:
            self.heavy = res
        else:
            self.light[spec.name] = res
        self._event(
            "load", spec, load_s=round(load_s, 2), device=device, device_actual=info.get("device"), vram_estimate_gb=need,
            info=info, substituted_for=res.substituted_for,
        )
        return backend

    def _check_device(self, spec: BackendSpec, backend: Backend, device: str, info: dict[str, Any]) -> None:
        """A GPU was asked for but the adapter reports the CPU: a CPU-only torch in its env (or a CUDA that does not
        start there). Silently running a heavy model from system RAM is 10-100x slower than the GPU and was invisible
        before; it is an error unless the profile allows CPU runs."""
        actual = str(info.get("device") or "")
        if not (device.startswith("cuda") and actual.startswith("cpu")):
            return
        why = f"loaded on the CPU although {device} was requested: the torch of env '{spec.env or 'core'}' has no CUDA"
        if self._cpu_allowed(spec):
            self.log.warning("%s %s (gpu.allow_cpu_fallback is on: running slowly)", spec.name, why)
            return
        try:
            backend.unload()
        except Exception:
            pass
        raise BackendUnavailable(
            spec.name, why, "reinstall that environment with CUDA torch (Models -> environments), or set gpu.allow_cpu_fallback: true to run on CPU (slow)"
        )

    def _make(self, spec: BackendSpec, options: dict[str, Any], device: str) -> Backend:
        if self.factory is not None:
            return self.factory(spec, options, device)
        return registry.instantiate(spec, options=options, device=device, models_dir=self.models_dir, log_dir=self.log_dir, context=self.context, root=self.root)

    def _check_budget(self, spec: BackendSpec, need: float | None = None) -> None:
        """``need`` = what this load costs (``estimate_vram_gb`` with the options: an LLM grows with ctx x parallel)."""
        need = spec.vram_estimate_gb if need is None else need
        limit = self.budget_gb - self.safety_margin_gb
        if need > limit:
            raise VramBudgetExceeded(spec.name, f"needs ~{need:.1f} GB > budget {self.budget_gb:.1f} - margin {self.safety_margin_gb:.1f} GB", "raise gpu.vram_budget_gb only if the GPU really has the memory")
        free, total = self.cuda.mem_get_info()
        free_gb = free / GB
        if need > free_gb - self.safety_margin_gb:
            raise VramBudgetExceeded(
                spec.name,
                f"needs ~{need:.1f} GB but only {free_gb:.2f} GB free of {total / GB:.1f} GB (margin {self.safety_margin_gb:.1f} GB)",
                "close other GPU applications",
            )

    def _unload(self, res: Resident) -> None:
        t0 = time.perf_counter()
        peak = None
        try:
            peak = res.backend.peak_vram()
        except Exception:
            pass
        try:
            res.backend.unload()
        finally:
            try:
                self.cuda.empty_cache()
            except Exception:
                pass
        self._event("unload", res.spec, unload_s=round(time.perf_counter() - t0, 2), resident_s=round(time.time() - res.loaded_at, 1), peak_vram_gb=peak)

    def release_heavy(self) -> None:
        if self.heavy is not None:
            res, self.heavy = self.heavy, None
            self._unload(res)

    def release(self, name: str | None = None) -> None:
        if name is None:
            self.release_heavy()
            for n in list(self.light):
                self.release(n)
            return
        if self.heavy is not None and self.heavy.name == name:
            self.release_heavy()
        elif name in self.light:
            self._unload(self.light.pop(name))

    def close(self) -> None:
        self.release(None)

    def abort_all(self) -> None:
        """Kill the processes of every resident backend now (a job was cancelled); ``close()`` tidies up after."""
        residents = ([self.heavy] if self.heavy else []) + list(self.light.values())
        for res in residents:
            fn = getattr(res.backend, "abort", None)
            if fn is not None:
                try:
                    fn()
                except Exception as e:
                    self.log.warning("abort %s failed: %s", res.name, e)

    @contextmanager
    def use(self, name: str, options: dict[str, Any] | None = None) -> Iterator[Backend]:
        b = self.acquire(name, options)
        try:
            yield b
        finally:
            pass  # stays resident until another heavy model is acquired or close()

    @property
    def resident(self) -> list[str]:
        return ([self.heavy.name] if self.heavy else []) + list(self.light)

    def _event(self, kind: str, spec: BackendSpec, **data: Any) -> None:
        ev = {"ts": now_iso(), "event": kind, "model": spec.name, "role": spec.role, "heavy": spec.heavy, "vram_estimate_gb": spec.vram_estimate_gb, **data}
        self.events.append(ev)
        self.log.info("%s %s %s", kind, spec.name, {k: v for k, v in data.items() if k != "info"})
        if self.timings_path is not None:
            append_jsonl(self.timings_path, ev)


def from_config(cfg: Any, paths: Any, tools: Any, cuda: CudaProbe | None = None, factory: BackendFactory | None = None) -> ModelRegistry:
    models_dir = resolve_models_dir(cfg)
    return ModelRegistry(
        budget_gb=cfg.gpu.vram_budget_gb,
        safety_margin_gb=cfg.gpu.safety_margin_gb,
        device=cfg.gpu.device,
        allow_cpu_fallback=cfg.gpu.allow_cpu_fallback,
        license_policy=cfg.license_policy,
        fallback_to_mock=cfg.fallback_to_mock,
        models_dir=models_dir,
        log_dir=paths.logs,
        timings_path=paths.work / "gpu_events.jsonl",
        cuda=cuda or SystemCuda(_device_index(cfg.gpu.device)),
        factory=factory,
        context={"tools": tools},
    )


def resolve_models_dir(cfg: Any) -> Path:
    """``VNDUB_MODELS_DIR`` env var, else ``project.models_dir`` (relative to the repo root)."""
    import os

    env = os.environ.get("VNDUB_MODELS_DIR")
    if env:
        return Path(env)
    p = Path(cfg.project.models_dir)
    return p if p.is_absolute() else (cfg.base_dir() / p)


def _device_index(device: str) -> int:
    if ":" in device:
        try:
            return int(device.split(":", 1)[1])
        except ValueError:
            return 0
    return 0

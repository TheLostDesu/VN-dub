"""Backend registry: name -> :class:`BackendSpec` (role, env, license, VRAM, weights).

Specs are declared next to their client code in ``vndub/backends/<role>/*.py``
(``SPECS = [...]``) and collected here. :func:`usable` explains *why* a backend
cannot run (license policy, platform, env missing, weights missing) so callers
can fall back or mark lines for review with a precise reason.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vndub.backends.base import Backend, BackendSpec, BackendUnavailable, LocalBackend
from vndub.backends.weights import missing

MODULES = [
    "vndub.backends.asr.anime_whisper",
    "vndub.backends.asr.whisper",
    "vndub.backends.asr.gigaam",
    "vndub.backends.asr.qwen3_asr",
    "vndub.backends.align.qwen3_aligner",
    "vndub.backends.align.whisperx",
    "vndub.backends.ser.emotion2vec",
    "vndub.backends.ser.anime_ser",
    "vndub.backends.ser.audeering_adv",
    "vndub.backends.ser.sensevoice",
    "vndub.backends.caption.qwen2_audio",
    "vndub.backends.llm.llamacpp",
    "vndub.backends.tts.qwen3_tts",
    "vndub.backends.tts.voxcpm2",
    "vndub.backends.tts.cosyvoice3",
    "vndub.backends.tts.chatterbox",
    "vndub.backends.tts.fish_s2",
    "vndub.backends.tts.f5_ru",
    "vndub.backends.tts.moss_tts",
    "vndub.backends.tts.zonos2",
    "vndub.backends.tts.higgs_tts3",
    "vndub.backends.tts.sapi",
    "vndub.backends.vc.rvc",
    "vndub.backends.vc.seedvc",
    "vndub.backends.score.utmos",
    "vndub.backends.score.audiobox",
    "vndub.backends.score.speaker_sim",
    "vndub.backends.dsp",
    "vndub.backends.mock",
]

_SPECS: dict[str, BackendSpec] | None = None


def all_specs() -> dict[str, BackendSpec]:
    global _SPECS
    if _SPECS is None:
        specs: dict[str, BackendSpec] = {}
        for mod in MODULES:
            for s in getattr(importlib.import_module(mod), "SPECS", []):
                if s.name in specs:
                    raise RuntimeError(f"duplicate backend name {s.name!r} in {mod}")
                specs[s.name] = s
        _SPECS = specs
    return _SPECS


def get(name: str) -> BackendSpec:
    specs = all_specs()
    if name not in specs:
        close = [n for n in specs if name.split("_")[0] in n]
        raise KeyError(f"unknown backend {name!r}" + (f"; did you mean {close}?" if close else ""))
    return specs[name]


def by_role(role: str) -> list[BackendSpec]:
    return [s for s in all_specs().values() if s.role == role]


def for_profile(profile: str) -> list[BackendSpec]:
    if profile == "all":
        return list(all_specs().values())
    return [s for s in all_specs().values() if profile in s.profiles]


def license_allowed(spec: BackendSpec, policy: str) -> bool:
    return policy != "commercial" or spec.commercial_ok


def platform_ok(spec: BackendSpec) -> bool:
    plat = "win32" if sys.platform.startswith("win") else ("darwin" if sys.platform == "darwin" else "linux")
    return plat in spec.platforms


@dataclass
class Usability:
    ok: bool
    reason: str = ""
    hint: str = ""


def usable(spec: BackendSpec, *, policy: str, models_dir: Path, root: Path | None = None) -> Usability:
    from vndub.backends.runner import env_exists

    if not license_allowed(spec, policy):
        return Usability(False, f"license '{spec.license}' not allowed by license_policy=commercial")
    if not platform_ok(spec):
        return Usability(False, f"not supported on {sys.platform}")
    if spec.env is not None and not env_exists(spec.env, root):
        return Usability(False, f"env '{spec.env}' not created", f"scripts/make_envs.ps1 -Env {spec.env}")
    miss = missing(models_dir, spec)
    if miss:
        what = ", ".join(w.repo or w.subdir or w.kind for w in miss)
        return Usability(False, f"weights missing ({what})", f"run `vndub models pull {spec.name}`")
    return Usability(True)


def load_class(path: str) -> type:
    mod, _, cls = path.partition(":")
    return getattr(importlib.import_module(mod), cls)


def instantiate(
    spec: BackendSpec,
    *,
    options: dict[str, Any] | None = None,
    device: str = "cpu",
    models_dir: Path | None = None,
    log_dir: Path | None = None,
    context: dict[str, Any] | None = None,
    root: Path | None = None,
) -> Backend:
    if spec.env is None:
        if not spec.local_impl:
            raise BackendUnavailable(spec.name, "local backend without implementation")
        cls = load_class(spec.local_impl)
        ctx = {"models_dir": models_dir, "log_dir": log_dir, **(context or {})}
        inst: LocalBackend = cls(spec, options, ctx)
        return inst  # type: ignore[return-value]
    from vndub.backends.runner import RemoteBackend

    opts = inject_tools(spec, dict(options or {}), (context or {}).get("tools"))
    return RemoteBackend(spec, opts, device=device, models_dir=models_dir, log_dir=log_dir, root=root)  # type: ignore[return-value]


def inject_tools(spec: BackendSpec, options: dict[str, Any], tools: Any) -> dict[str, Any]:
    """Resolve external tool paths for adapters that drive binaries/checkouts.

    ``options.tool`` (e.g. ``audiocpp``) -> ``options.binary``;
    ``options.tool_dir`` (e.g. ``applio_dir``) -> ``options[<tool_dir>]``.
    Adapters run in another env and cannot read tools.yaml themselves.
    """
    if tools is None:
        return options
    merged = {**spec.options, **options}
    tool = merged.get("tool")
    if tool and not merged.get("binary"):
        p = tools.resolve(tool)
        if p is not None:
            options["binary"] = str(p)
    key = merged.get("tool_dir")
    if key and not merged.get(key):
        configured = (getattr(tools, "paths", {}) or {}).get(key, "")
        if configured and Path(configured).exists():
            options[key] = str(Path(configured))
    return options

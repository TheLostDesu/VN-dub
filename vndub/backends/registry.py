"""Backend registry: name -> :class:`BackendSpec` (role, env, license, VRAM, weights).

Specs are declared next to their client code in ``vndub/backends/<role>/*.py``
(``SPECS = [...]``) and collected here. :func:`usable` explains *why* a backend
cannot run (license policy, platform, env missing, weights missing) so callers
can fall back or mark lines for review with a precise reason.
"""

from __future__ import annotations

import glob
import importlib
import json
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vndub.backends.base import Backend, BackendSpec, BackendUnavailable, LocalBackend, WeightSpec
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

#: env var naming a JSON file with user-added models (the desktop app writes ``<data>/custom_models.json``
#: and sets the variable at startup). Unset -> the registry is exactly the built-in one (CLI behaviour).
CUSTOM_ENV = "VNDUB_CUSTOM_MODELS"
_CUSTOM_LOCK = threading.RLock()
_CUSTOM_SIG: tuple[str, int, int] | None = None
_CUSTOM_NAMES: set[str] = set()
# licence tags (Hugging Face ``license:`` values) the app treats as usable under ``license_policy: commercial``
_COMMERCIAL_LICENSES = frozenset({"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "cc-by-4.0", "cc0-1.0", "unlicense", "gemma"})


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
    _sync_custom()
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


# ----------------------------------------------------------------------------
# user-added models (``VNDUB_CUSTOM_MODELS``)
# ----------------------------------------------------------------------------


def custom_models_file() -> Path | None:
    raw = os.environ.get(CUSTOM_ENV, "").strip()
    return Path(raw) if raw else None


def read_custom_entries(path: Path) -> list[dict[str, Any]]:
    """Entries of a ``custom_models.json`` (a list, or ``{"models": [...]}``); a missing or broken file is empty."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    items = data.get("models") if isinstance(data, dict) else data
    return [e for e in items if isinstance(e, dict)] if isinstance(items, list) else []


def custom_spec(entry: dict[str, Any]) -> BackendSpec | None:
    """The :class:`BackendSpec` of one ``gguf_llm`` entry: a translation LLM for the llama.cpp adapter.

    The weights live in ``<models_dir>/<entry.dir>`` (``dir`` is relative to the models folder, e.g.
    ``custom/owner__repo``, so ``weights.local_dir`` resolves it through ``subdir``); ``is_present`` needs the
    ``.vndub_complete`` marker the app writes after the download, ``find_gguf`` the file names in ``allow``
    (matched against the base name; the leading ``*`` also lets the pattern match the repo path). Other kinds
    (RVC voices, plain files) are not backends and give ``None``.
    """
    if entry.get("kind") != "gguf_llm":
        return None
    name, repo, subdir = str(entry.get("id") or ""), str(entry.get("repo") or ""), str(entry.get("dir") or "")
    files = [str(f) for f in (entry.get("files") or []) if str(f).lower().endswith(".gguf")]
    if not (name and repo and subdir and files):
        return None
    size_gb = float(entry.get("size_gb") or 0.0)
    lic = str(entry.get("license") or "").strip()
    label = str(entry.get("label") or repo)
    return BackendSpec(
        name=name,
        role="llm",
        env=None,
        adapter="llamacpp",
        license=lic or "unknown",
        commercial_ok=lic.lower() in _COMMERCIAL_LICENSES,
        vram_estimate_gb=round(size_gb * 1.1 + 1.0, 1),
        heavy=True,
        weights=(
            WeightSpec(
                "hf_file",
                repo=repo,
                revision=entry.get("revision") or None,
                allow=tuple("*" + glob.escape(Path(f).name) for f in files),
                subdir=subdir,
                size_gb=size_gb,
            ),
        ),
        profiles=("optional",),
        options={"ctx": 8192, "chat_template_kwargs": {}},
        mock="mock_translator",
        description=f"{label} (added by the user)",
        homepage=f"https://huggingface.co/{repo}",
        local_impl="vndub.backends.llm.llamacpp:LlamaCppBackend",
    )


def _sync_custom() -> None:
    """Make ``_SPECS`` hold the built-in backends plus the entries of the current custom file (re-read on change).

    A change builds a *new* dict (copy on write): threads that are iterating the old one are not disturbed.
    """
    global _SPECS, _CUSTOM_SIG, _CUSTOM_NAMES
    path = custom_models_file()
    if path is None and _CUSTOM_SIG is None:
        return  # the common case: no custom models, nothing to do
    sig: tuple[str, int, int] | None = None
    if path is not None:
        try:
            st = path.stat()
            sig = (str(path), st.st_mtime_ns, st.st_size)
        except OSError:
            sig = (str(path), 0, -1)
    with _CUSTOM_LOCK:
        if sig == _CUSTOM_SIG or _SPECS is None:
            return
        specs = {n: s for n, s in _SPECS.items() if n not in _CUSTOM_NAMES}
        names: set[str] = set()
        if path is not None and sig is not None and sig[2] >= 0:
            for entry in read_custom_entries(path):
                spec = custom_spec(entry)
                if spec is not None and spec.name not in specs:
                    specs[spec.name] = spec
                    names.add(spec.name)
        _SPECS, _CUSTOM_NAMES, _CUSTOM_SIG = specs, names, sig


def custom_names() -> frozenset[str]:
    """Names of the backends that came from the custom file (not built in)."""
    all_specs()
    return frozenset(_CUSTOM_NAMES)


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

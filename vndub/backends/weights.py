"""Where downloaded weights live and how presence is detected.

Convention: every :class:`WeightSpec` downloads into
``<models_dir>/<subdir or repo basename or backend name>/``; a successful pull
writes ``.vndub_complete`` there. Repos shared between backends (e.g. the
Qwen3-TTS Base model) are therefore downloaded once. ``post: ct2_convert``
writes the CTranslate2 conversion into ``<local_dir>/ct2``.
"""

from __future__ import annotations

from pathlib import Path

from vndub.backends.base import BackendSpec, WeightSpec

MARKER = ".vndub_complete"


def local_dir(models_dir: Path, w: WeightSpec, backend: str) -> Path:
    if w.subdir:
        name = w.subdir
    elif w.repo:
        name = w.repo.split("/")[-1]
    else:
        name = backend
    return Path(models_dir) / name


def is_present(models_dir: Path, w: WeightSpec, backend: str) -> bool:
    if w.kind == "builtin":
        return True
    return (local_dir(models_dir, w, backend) / MARKER).exists()


def missing(models_dir: Path, spec: BackendSpec) -> list[WeightSpec]:
    return [w for w in spec.weights if not is_present(models_dir, w, spec.name)]


def dir_size_gb(p: Path) -> float:
    if not p.exists():
        return 0.0
    total = 0
    for f in p.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total / 1e9


def resolved(models_dir: Path | None, spec: BackendSpec) -> list[dict]:
    """Weight descriptors with ``local_dir`` filled in (sent to adapters)."""
    out = []
    for w in spec.weights:
        d = dict(w.__dict__)
        d["allow"] = list(w.allow)
        if models_dir is not None:
            d["local_dir"] = str(local_dir(models_dir, w, spec.name))
            d["present"] = is_present(models_dir, w, spec.name)
        out.append(d)
    return out

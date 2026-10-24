"""Every Hugging Face weight the registry offers resolves on the real hub and its ``allow`` patterns match files.

Skipped unless ``VNDUB_NETWORK_TESTS=1`` (``VNDUB_NETWORK_TESTS=1 pytest tests/test_registry_repos_network.py -rs``).
Nothing is downloaded: only repo metadata (``HfApi().model_info``). A repo that was renamed, never published or
made private shows up here instead of as a 404 page in the Models screen (``fishaudio/s2-pro-gguf`` and
``OpenMOSS-Team/MOSS-TTS-v1.5-GGUF`` were such guesses).

``allow`` is matched case-sensitively (``fnmatchcase``): that is how ``snapshot_download`` filters on Linux and macOS,
so a pattern that only matches on Windows (``*Q4_K*`` against ``q4_k_m``) fails here too.
"""

from __future__ import annotations

import fnmatch
import os
from typing import Any

import pytest

from vndub.backends import registry
from vndub.backends.base import BackendSpec, WeightSpec

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(os.environ.get("VNDUB_NETWORK_TESTS") != "1", reason="set VNDUB_NETWORK_TESTS=1 to talk to the real huggingface.co"),
]

HUB_KINDS = ("hf", "hf_file")


def _hub_weights() -> list[tuple[str, WeightSpec]]:
    """``(backend name, weight)`` for every hub weight of the built-in registry (a repo shared by several backends once per backend)."""
    out: list[tuple[str, WeightSpec]] = []
    for name, spec in registry.all_specs().items():
        if name in registry.custom_names():
            continue  # user-added links are validated when they are added
        out += [(name, w) for w in spec.weights if w.kind in HUB_KINDS]
    return out


def _id(item: tuple[str, WeightSpec]) -> str:
    name, w = item
    return f"{name}:{w.repo}"


_INFO: dict[tuple[str, str | None], Any] = {}


def _model_info(repo: str, revision: str | None) -> Any:
    from huggingface_hub import HfApi

    key = (repo, revision)
    if key not in _INFO:
        _INFO[key] = HfApi().model_info(repo, revision=revision, files_metadata=True)
    return _INFO[key]


def test_the_registry_has_hub_weights() -> None:
    weights = _hub_weights()
    assert len(weights) > 20 and all(w.repo and "/" in w.repo for _n, w in weights)


@pytest.mark.parametrize("item", _hub_weights(), ids=_id)
def test_hub_weight_resolves_and_allow_patterns_match(item: tuple[str, WeightSpec]) -> None:
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError

    name, w = item
    try:
        info = _model_info(w.repo, w.revision)
    except RepositoryNotFoundError:
        pytest.fail(f"{name}: https://huggingface.co/{w.repo} does not exist (or is private): the Models screen would 404")
    except RevisionNotFoundError:
        pytest.fail(f"{name}: https://huggingface.co/{w.repo} has no revision {w.revision!r}")
    except GatedRepoError:  # metadata of a gated repo is public; tolerate a hub that says otherwise
        pytest.skip(f"{name}: {w.repo} is gated")
    files = [(str(s.rfilename), int(s.size or 0)) for s in (info.siblings or [])]
    assert files, f"{name}: {w.repo} lists no files"
    mine = [(n, sz) for n, sz in files if any(fnmatch.fnmatchcase(n, pat) for pat in w.allow)] if w.allow else files
    assert mine, f"{name}: no file of {w.repo} matches allow={list(w.allow)} (files: {[n for n, _ in files][:12]})"
    if w.kind == "hf_file":
        assert w.allow, f"{name}: an hf_file weight needs allow patterns"
    # A stale size only mis-states the free-space check and the table; report it, do not fail on it.
    real_gb = sum(sz for _n, sz in mine) / 1e9
    if w.size_gb and real_gb and not (0.75 * real_gb <= w.size_gb <= 1.35 * real_gb + 0.2):
        print(f"\n[size drift] {name}: registry says {w.size_gb} GB, {w.repo} matches {real_gb:.2f} GB")


def test_no_two_weights_of_one_dir_overlap_with_different_repos() -> None:
    """A weight directory is shared by backends only when it is the same repo (otherwise one pull would mark the other done)."""
    from vndub.backends.weights import local_dir
    from pathlib import Path

    seen: dict[str, tuple[str, str]] = {}
    for name, w in _hub_weights():
        d = str(local_dir(Path("m"), w, name))
        prev = seen.setdefault(d, (name, w.repo))
        assert prev[1] == w.repo, f"{d}: {prev[0]} uses {prev[1]} but {name} uses {w.repo}"


def _specs_by_name() -> dict[str, BackendSpec]:
    return registry.all_specs()


def test_the_gguf_tts_backends_point_at_published_files() -> None:
    """The two guessed repos of the first versions are gone; the replacements are the ones the runtimes document."""
    specs = _specs_by_name()
    for name in ("fish_s2", "moss_tts"):
        (w,) = specs[name].weights
        info = _model_info(w.repo, w.revision)
        names = [str(s.rfilename) for s in info.siblings or []]
        hits = [n for n in names if any(fnmatch.fnmatchcase(n, p) for p in w.allow)]
        assert hits and all(n.endswith(".gguf") for n in hits), (name, w.repo, hits)

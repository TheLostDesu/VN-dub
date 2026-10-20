"""Shared pytest fixtures.

``make_ctx`` builds a real :class:`StageContext` on the test profile with work/out
under ``tmp_path`` and a ModelRegistry that always uses mocks (``force_mock``)
and never touches CUDA (``NoCuda``), so tests are deterministic and need no
backend envs or downloads.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "tests" / "fixtures"
TEST_CONFIG = REPO / "configs" / "pipeline.test.yaml"

sys.path.insert(0, str(REPO))  # scripts/ and adapters/ importable in tests


@pytest.fixture
def repo() -> Path:
    return REPO


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


def _make_ctx(tmp_path: Path, project_dir: Path, sets: dict[str, Any] | None = None, extra: dict[str, Any] | None = None, copy_project: bool = False):
    from vndub.config import load_config
    from vndub.gpu import ModelRegistry, NoCuda
    from vndub.paths import WorkPaths
    from vndub.stages.common import StageContext
    from vndub.tools import Tools

    if copy_project:
        dst = tmp_path / "game"
        shutil.copytree(project_dir, dst)
        project_dir = dst
    base = {
        "synth.backends.sapi_tts.enabled": False,  # keep unit tests fast and platform-independent
    }
    base.update(sets or {})
    cfg = load_config(TEST_CONFIG, base)
    paths = WorkPaths.create(tmp_path / "work", tmp_path / "out")
    tools = Tools.load(cfg.resolve(cfg.tools_file), cfg.tools)
    models = ModelRegistry(
        budget_gb=cfg.gpu.vram_budget_gb,
        safety_margin_gb=cfg.gpu.safety_margin_gb,
        allow_cpu_fallback=True,
        license_policy=cfg.license_policy,
        fallback_to_mock=True,
        models_dir=tmp_path / "models",
        log_dir=paths.logs,
        timings_path=paths.work / "gpu_events.jsonl",
        cuda=NoCuda(),
        context={"tools": tools},
        force_mock=True,
    )
    return StageContext(cfg=cfg, paths=paths, tools=tools, project_dir=Path(project_dir).resolve(), models=models, extra=extra or {})


@pytest.fixture
def make_ctx(tmp_path: Path) -> Callable[..., Any]:
    """make_ctx(project_dir, sets=None, extra=None, copy_project=False) -> StageContext"""

    def factory(project_dir: Path, sets: dict[str, Any] | None = None, extra: dict[str, Any] | None = None, copy_project: bool = False):
        return _make_ctx(tmp_path, project_dir, sets, extra, copy_project)

    return factory


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


requires_ffmpeg = pytest.mark.skipif(not has_ffmpeg(), reason="ffmpeg/ffprobe not on PATH")

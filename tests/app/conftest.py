"""Fixtures of the desktop-app tests.

Every test runs against a throw-away data folder (``VNDUB_DATA``), the all-mock ``apptest``
profile, a copy of ``tests/fixtures/renpy_mini`` and a fresh in-memory job runner. The user's
real ``%APPDATA%`` / ``%LOCALAPPDATA%`` are never touched (their env vars are redirected too).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Iterator

import pytest
from app_helpers import FIXTURE, reset_app_state, setup_env, teardown_app_state


@pytest.fixture
def app_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Data folder + env for one test; returns the data dir."""
    data = setup_env(monkeypatch, tmp_path)
    reset_app_state()
    yield data
    teardown_app_state()


@pytest.fixture
def game_dir(tmp_path: Path) -> Path:
    dst = tmp_path / "games" / "renpy_mini"
    shutil.copytree(FIXTURE, dst)
    return dst


@pytest.fixture
def client(app_env: Path) -> Iterator[Any]:
    from fastapi.testclient import TestClient

    from vndub_app.server import create_app

    with TestClient(create_app(), raise_server_exceptions=False) as c:
        yield c

"""``vndub review --app``: the CLI opens the app's review screen on the CLI project's own work/out dirs."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
from app_helpers import FIXTURE

REPO = Path(__file__).resolve().parents[2]


def test_review_app_flag_is_wired(app_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    from vndub import cli
    from vndub_app import server

    game = tmp_path / "game"
    shutil.copytree(FIXTURE, game)
    seen: dict[str, Any] = {}

    def fake_serve(ctx: Any, host: str = "127.0.0.1", port: int = 0, open_browser: bool = False) -> None:
        seen.update(work=ctx.paths.work, out=ctx.paths.out, host=host, port=port, open_browser=open_browser, project=ctx.project_dir)

    monkeypatch.setattr(server, "serve_review", fake_serve)
    res = CliRunner().invoke(cli.app, [
        "review", "--project", str(game), "--config", str(REPO / "configs" / "pipeline.apptest.yaml"),
        "--work-dir", str(tmp_path / "w"), "--out-dir", str(tmp_path / "o"), "--app", "--browser", "--port", "8123",
    ])
    assert res.exit_code == 0, res.output
    assert seen["work"] == (tmp_path / "w").resolve() and seen["out"] == (tmp_path / "o").resolve()
    assert seen["port"] == 8123 and seen["open_browser"] is True and seen["project"] == game.resolve()
    # the Gradio default keeps working when the flag is absent
    called: dict[str, Any] = {}
    import vndub.review.app as gradio_app

    monkeypatch.setattr(gradio_app, "launch", lambda ctx, host, port, open_browser: called.update(port=port))
    res = CliRunner().invoke(cli.app, ["review", "--project", str(game), "--config", str(REPO / "configs" / "pipeline.apptest.yaml"),
                                       "--work-dir", str(tmp_path / "w"), "--out-dir", str(tmp_path / "o"), "--port", "9999"])
    assert res.exit_code == 0, res.output
    assert called == {"port": 9999}


def test_serve_review_binds_the_project_to_the_cli_dirs(app_env: Path, tmp_path: Path, make_ctx: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from vndub_app import appdirs, server, session

    game = tmp_path / "gamecopy"
    shutil.copytree(FIXTURE, game)
    ctx = make_ctx(game)  # work/out under tmp_path (the CLI's own dirs)

    class FakeServer:
        url = "http://127.0.0.1:1"

        class thread:  # noqa: N801
            @staticmethod
            def is_alive() -> bool:
                return False

            @staticmethod
            def join(t: float = 0) -> None:
                return None

        def __init__(self, host: str, port: int) -> None:
            self.host, self.port = host, port

        def start(self) -> "FakeServer":
            return self

        def stop(self) -> None:
            return None

    opened: list[str] = []
    monkeypatch.setattr(server, "BackgroundServer", FakeServer)
    monkeypatch.setattr("webbrowser.open_new_tab", lambda u: opened.append(u))
    server.serve_review(ctx, open_browser=True)
    h = appdirs.find_project_for_game(game)
    assert h is not None and h.work == ctx.paths.work and h.out == ctx.paths.out
    assert appdirs.read_app_state(h)["step"] == "review" and session.current_slug() == h.slug
    cfg = h.config_file.read_text(encoding="utf-8")
    assert str(ctx.paths.work) in cfg and str(ctx.paths.out) in cfg
    assert opened == ["http://127.0.0.1:1"]

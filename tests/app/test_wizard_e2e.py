"""Playwright smoke test: walk the wizard Steps 1 -> 7 on the fixture in the all-mock profile.

A real uvicorn serves the built SPA (``vndub_app/web/dist``) against a throw-away data folder;
Chromium drives it through the ``data-testid`` hooks listed in ``vndub_app/web/TESTIDS.md``.
Screenshots of every screen land in ``docs/img/app/`` (used by ``docs/APP.md``).

Skipped when the SPA is not built or Playwright's Chromium is not installed
(``uv run playwright install chromium``). Run alone with ``uv run pytest -q -m e2e``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Iterator

import pytest
from app_helpers import REPO, reset_app_state, setup_env, teardown_app_state

pytestmark = [pytest.mark.e2e, pytest.mark.slow, pytest.mark.timeout(1500)]

sync_api = pytest.importorskip("playwright.sync_api")

DIST = REPO / "vndub_app" / "web" / "dist" / "index.html"
SHOTS = REPO / "docs" / "img" / "app"
VIEWPORT = {"width": 1366, "height": 900}
FAIL_SHOT = Path(os.environ.get("VNDUB_E2E_FAIL_SHOT", REPO / "tests" / "app" / "_e2e_failure.png"))

S = 1000  # ms


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    # setup_env redirects LOCALAPPDATA, where Playwright keeps its browsers on Windows
    if not os.environ.get("PLAYWRIGHT_BROWSERS_PATH") and os.environ.get("LOCALAPPDATA"):
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(Path(os.environ["LOCALAPPDATA"]) / "ms-playwright"))
    setup_env(monkeypatch, tmp_path)
    reset_app_state()
    from vndub_app.server import BackgroundServer

    srv = BackgroundServer().start()
    try:
        yield srv
    finally:
        srv.stop()
        teardown_app_state()


def test_wizard_walk(server: Any, game_dir: Path) -> None:
    if not DIST.exists():
        pytest.skip("SPA not built: run `npm run build` in vndub_app/web")
    expect = sync_api.expect
    SHOTS.mkdir(parents=True, exist_ok=True)
    voice = game_dir / "game" / "voice" / "sak_0001.ogg"
    orig_hash = _sha(voice)

    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as e:  # browser binaries missing
            pytest.skip(f"Playwright Chromium not available: {e}")
        page = browser.new_page(viewport=VIEWPORT, locale="ru-RU")
        page_errors: list[str] = []
        page.on("pageerror", lambda e: page_errors.append(str(e)))

        def tid(name: str) -> Any:
            return page.get_by_test_id(name)

        def shot(name: str, full: bool = False) -> None:
            page.wait_for_timeout(400)  # let animations / waveforms settle
            page.screenshot(path=str(SHOTS / f"{name}.png"), full_page=full)

        try:
            _walk(page, tid, shot, expect, server, game_dir)
        except Exception:
            page.screenshot(path=str(FAIL_SHOT), full_page=True)
            raise
        browser.close()

    assert not page_errors, f"uncaught errors in the page: {page_errors}"
    # the patch really replaced the original voice file, and a backup of it exists
    assert _sha(voice) != orig_hash
    backups = list((Path(server_data_dir()) / "backups").rglob("sak_0001.ogg"))
    assert backups and _sha(backups[0]) == orig_hash


def _walk(page: Any, tid: Any, shot: Any, expect: Any, server: Any, game_dir: Path) -> None:
        # Step 0 -- diagnostics (no NVIDIA GPU on the build machine: a real red row)
        page.goto(server.url + "/#/step/doctor")
        expect(tid("doctor-rows")).to_be_visible(timeout=60 * S)
        shot("00-doctor")

        # Step 1 -- game folder, import (detect .. analyze + text_emotion)
        page.goto(server.url + "/#/step/game")
        tid("game-pick-folder").click()
        tid("game-path-input").fill(str(game_dir))
        tid("game-path-open").click()
        expect(tid("game-summary")).to_be_visible(timeout=300 * S)
        expect(tid("stat-lines")).to_contain_text("41")
        expect(tid("game-characters-table")).to_be_visible()
        expect(tid("game-char-hero")).to_have_attribute("data-mode", "unvoiced")
        shot("01-game", full=True)
        tid("nav-next").click()

        # Step 2 -- translation (mock translator) + settings
        expect(tid("screen-translate")).to_be_visible(timeout=30 * S)
        shot("02-translate", full=True)
        tid("translate-run-button").click()
        expect(tid("translate-done-chip")).to_be_visible(timeout=300 * S)
        tid("nav-next").click()

        # Step 3 -- casting: automatic, then open one tuning panel
        expect(tid("screen-casting")).to_be_visible(timeout=30 * S)
        tid("casting-auto").click()
        page.wait_for_function(
            """() => {
                const cards = [...document.querySelectorAll('[data-testid^="char-"][data-chosen]')];
                return cards.length > 0 && cards.every(c => c.dataset.chosen === 'true');
            }""",
            timeout=300 * S,
        )
        expect(tid("char-hero")).to_have_attribute("data-mode", "unvoiced")
        shot("03-casting")
        if tid("char-toggle-sakura").count():
            tid("char-toggle-sakura").click()
            if tid("char-sakura-tuning-toggle").count():
                tid("char-sakura-tuning-toggle").click()
            tid("char-sakura").scroll_into_view_if_needed()
            shot("03b-casting-character", full=True)
        tid("nav-next").click()

        # Step 4 -- quick preview (first scene)
        expect(tid("screen-preview")).to_be_visible(timeout=30 * S)
        tid("preview-run").click()
        expect(tid("preview-job")).to_have_attribute("data-state", "done", timeout=600 * S)
        expect(tid("preview-items")).to_be_visible(timeout=60 * S)
        shot("04-preview", full=True)
        tid("preview-approve").click()

        # Step 5 -- full run (starts by itself after the preview approval)
        expect(tid("screen-run")).to_be_visible(timeout=30 * S)
        expect(tid("run-job")).to_be_visible(timeout=60 * S)
        page.wait_for_timeout(3 * S)
        shot("05-run")
        expect(tid("run-done")).to_be_visible(timeout=900 * S)
        shot("05-run-done")
        tid("run-to-review").click()

        # Step 6 -- review: accept the selected variant by keyboard, apply changes
        expect(tid("screen-review")).to_be_visible(timeout=30 * S)
        expect(tid("review-detail")).to_be_visible(timeout=60 * S)
        shot("06-review")
        first_line = tid("review-detail").get_attribute("data-line")
        tid("review-title").click()  # focus out of any control so the shortcuts apply
        page.keyboard.press("Enter")
        page.wait_for_function(
            "(l) => document.querySelector('[data-testid=\"review-detail\"]')?.dataset.line !== l",
            arg=first_line,
            timeout=30 * S,
        )
        tid("review-apply").click()
        expect(tid("review-apply-result")).to_be_visible(timeout=600 * S)
        shot("06-review-applied")
        tid("nav-next").click()

        # Step 7 -- install the patch into the (copied) game, with a backup
        expect(tid("screen-install")).to_be_visible(timeout=30 * S)
        expect(tid("install-plan")).to_be_visible(timeout=60 * S)
        expect(tid("install-backup")).to_be_checked()
        shot("07-install", full=True)
        tid("install-apply").click()
        expect(tid("install-done")).to_be_visible(timeout=300 * S)
        shot("07-install-done")

        # Always-available screens
        page.goto(server.url + "/#/models")
        expect(tid("models-table")).to_be_visible(timeout=60 * S)
        expect(tid("models-hf-card")).to_be_visible()  # "add a model by link" sits above the table (see test_models_hf_e2e.py)
        expect(tid("model-recommended-gemma4_12b")).to_be_visible()
        page.evaluate("document.querySelector('[data-testid=\"models-presets\"]').scrollIntoView(); window.scrollBy(0, -70)")  # the screenshot shows the table (below the sticky top bar)
        shot("08-models")
        page.goto(server.url + "/#/settings")
        expect(tid("settings-title")).to_be_visible(timeout=30 * S)
        shot("09-settings")
        page.goto(server.url + "/#/help")
        expect(tid("help-body")).to_be_visible(timeout=30 * S)
        shot("10-help")


def server_data_dir() -> str:
    from vndub_app import appdirs

    return str(appdirs.data_dir())

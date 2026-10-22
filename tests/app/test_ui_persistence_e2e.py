"""Playwright: leaving a screen must not lose its work (UI persistency while switching tabs and screens).

``/#/models`` and ``/#/tools`` are two routes, so switching the tab unmounts the whole tab component, and every wizard
step is a route of its own. The jobs keep running on the server (and in the SPA's global progress store); the screens
have to pick them up again, and the screen-local choices (ticked models, the torch radio) have to survive the round trip.

Scenario (all-mock profile, no network, no installer): start the import of the fixture game, leave for the Models and
Tools tabs at once, change the model selection and the torch choice there, come back to the game screen and expect its
job panel (running, or already finished) to be shown again and to reach ``done``; the finished panel must stay after one
more round trip. Same runner and skip rules as ``test_wizard_e2e.py`` (needs the built SPA and Chromium).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from test_wizard_e2e import DIST, FAIL_SHOT, VIEWPORT, S, server  # noqa: F401  (``server`` is the fixture)

pytestmark = [pytest.mark.e2e, pytest.mark.slow, pytest.mark.timeout(900)]

sync_api = pytest.importorskip("playwright.sync_api")

TICKED_JS = "els => els.filter(e => e.checked).map(e => e.dataset.testid).sort()"


def test_job_panels_and_choices_survive_route_changes(server: Any, game_dir: Path) -> None:  # noqa: F811
    if not DIST.exists():
        pytest.skip("SPA not built: run `npm run build` in vndub_app/web")
    expect = sync_api.expect

    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as e:  # browser binaries missing
            pytest.skip(f"Playwright Chromium not available: {e}")
        page = browser.new_page(viewport=VIEWPORT, locale="ru-RU")
        page_errors: list[str] = []
        page.on("pageerror", lambda e: page_errors.append(str(e)))
        tid = page.get_by_test_id

        def ticked_models() -> list[str]:
            return page.eval_on_selector_all('input[data-testid^="model-check-"]', TICKED_JS)

        try:
            # the import starts by itself when the folder is opened; leave as soon as its panel is there
            page.goto(server.url + "/#/step/game")
            tid("game-pick-folder").click()
            tid("game-path-input").fill(str(game_dir))
            tid("game-path-open").click()
            expect(tid("game-job")).to_be_visible(timeout=60 * S)
            expect(tid("game-job")).to_have_attribute("data-kind", "import")

            # Models tab: change the selection away from the pre-selected default, so that a reset would be noticed
            page.goto(server.url + "/#/models")
            expect(tid("models-table")).to_be_visible(timeout=60 * S)
            first = page.locator('input[data-testid^="model-check-"]').first
            expect(first).to_be_visible()
            was = first.is_checked()
            first.set_checked(not was)
            chosen = ticked_models()
            assert (first.get_attribute("data-testid") in chosen) is (not was)

            # Tools tab: the torch choice
            tid("tab-tools").click()
            expect(tid("envs-torch-cpu")).to_be_visible(timeout=60 * S)
            expect(tid("envs-torch-auto")).to_be_checked()
            tid("envs-torch-cpu").check()
            expect(tid("envs-torch-cpu")).to_be_checked()

            # and back: both choices are still there (the pre-selection must not run a second time)
            tid("tab-models").click()
            expect(tid("models-table")).to_be_visible(timeout=60 * S)
            assert ticked_models() == chosen
            tid("tab-tools").click()
            expect(tid("envs-torch-cpu")).to_be_checked(timeout=30 * S)

            # the game screen shows the import again: still running, or already finished (that is the point)
            page.goto(server.url + "/#/step/game")
            expect(tid("game-job")).to_be_visible(timeout=30 * S)
            expect(tid("game-job")).to_have_attribute("data-kind", "import")
            expect(tid("game-job")).to_have_attribute("data-state", "done", timeout=300 * S)
            expect(tid("game-summary")).to_be_visible(timeout=60 * S)

            # once it is over, the finished panel survives one more round trip
            page.goto(server.url + "/#/models")
            expect(tid("models-table")).to_be_visible(timeout=60 * S)
            page.goto(server.url + "/#/step/game")
            expect(tid("game-job")).to_have_attribute("data-state", "done", timeout=30 * S)
            expect(tid("game-summary")).to_be_visible(timeout=60 * S)
        except Exception:
            page.screenshot(path=str(FAIL_SHOT), full_page=True)
            raise
        finally:
            browser.close()

    assert not page_errors, f"uncaught errors in the page: {page_errors}"

"""Playwright: the "Голос за кадром" card on the casting screen (game with voiced lines shown without a name).

The project is imported and translated through the API of a real server, then Chromium opens Step 3:
the voice-over speaker has its own card with the "Без имени в игре" chip, and automatic casting gives
it a voice. Screenshot: ``docs/img/app/03c-casting-voice-over.png`` (used by ``docs/APP.md``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from app_helpers import run_job
from test_offscreen_voice import _game
from test_wizard_e2e import DIST, SHOTS, VIEWPORT, server  # noqa: F401  (server is a fixture)

pytestmark = [pytest.mark.e2e, pytest.mark.slow, pytest.mark.timeout(900)]

sync_api = pytest.importorskip("playwright.sync_api")

S = 1000  # ms


def test_casting_shows_voice_over(server: Any, tmp_path: Path) -> None:  # noqa: F811
    if not DIST.exists():
        pytest.skip("SPA not built: run `npm run build` in vndub_app/web")
    game = _game(tmp_path / "games" / "renpy_offscreen")
    with httpx.Client(base_url=server.url, timeout=60) as c:
        assert c.post("/api/project/open", json={"game_dir": str(game)}).status_code == 200
        run_job(c, c.post("/api/project/import", json={}), 400)
        run_job(c, c.post("/api/project/translate", json={}), 400)

    expect = sync_api.expect
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as e:
            pytest.skip(f"Playwright Chromium not available: {e}")
        page = browser.new_page(viewport=VIEWPORT, locale="ru-RU")
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(server.url + "/#/step/casting")
        card = page.get_by_test_id("char-voice_over")
        expect(card).to_be_visible(timeout=60 * S)
        expect(card).to_contain_text("Голос за кадром")
        expect(page.get_by_test_id("char-voice-over")).to_have_text("Без имени в игре")
        page.get_by_test_id("casting-auto").click()
        expect(page.get_by_test_id("char-chosen-voice_over")).to_be_visible(timeout=300 * S)
        if page.get_by_test_id("char-toggle-voice_over").get_attribute("aria-expanded") != "true":
            page.get_by_test_id("char-toggle-voice_over").click()
        card.evaluate("el => { el.scrollIntoView({block: 'start'}); window.scrollBy(0, -130); }")  # below the sticky header
        page.wait_for_timeout(600)
        SHOTS.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(SHOTS / "03c-casting-voice-over.png"), full_page=False)
        browser.close()
    assert not errors, errors

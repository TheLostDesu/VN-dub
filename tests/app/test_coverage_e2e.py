"""Playwright: the finished full run says what the patch holds, and the Review / Install / Game screens agree with it.

A real uvicorn serves the built SPA against a throw-away data folder in the all-mock profile. Steps 1-3 are done through the API
(``test_wizard_e2e.py`` walks their screens); the quality gates are made impossible to pass, so that every line's best take is
installed as provisional -- the situation the "Готово" screen has to be honest about -- and then the browser takes over:

* Step 5: start the run, wait for "Озвучка готова", read the coverage callout (warn tone, the numbers of the API), reload the page
  (an app opened on a finished run asks the server instead of a job) and follow the button to the review;
* Step 6: the rows and the detail carry the "предварительная" chip;
* Step 7: the three numbers beside the file count;
* a project imported by an older version gets the "разобрана прошлой версией" note on the Run and Game screens, and the import clears it.

Skipped when the SPA is not built or Chromium is missing (same as ``test_wizard_e2e.py``). Set ``VNDUB_COVERAGE_SHOTS`` to a folder
to keep a screenshot of each screen.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import httpx
import pytest
from app_helpers import wait_job
from test_wizard_e2e import DIST, FAIL_SHOT, VIEWPORT, S, server  # noqa: F401  (``server`` is the fixture)

pytestmark = [pytest.mark.e2e, pytest.mark.slow, pytest.mark.timeout(900)]

sync_api = pytest.importorskip("playwright.sync_api")


def job_done(api: httpx.Client, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    r = api.post(path, json=body or {})
    assert r.status_code == 200, r.text
    j = wait_job(api, r.json(), 600)
    assert j["state"] == "done", f"{path}: {j['state']} {j.get('error')}"
    return j


def test_done_screen_says_what_the_patch_holds(server: Any, game_dir: Path) -> None:  # noqa: F811
    if not DIST.exists():
        pytest.skip("SPA not built: run `npm run build` in vndub_app/web")
    from vndub_app import appdirs

    expect = sync_api.expect
    api = httpx.Client(base_url=server.url, timeout=60)
    slug = api.post("/api/project/open", json={"game_dir": str(game_dir)}).json()["slug"]
    job_done(api, "/api/project/import")
    job_done(api, "/api/project/translate")
    job_done(api, "/api/casting/auto")
    h = appdirs.project_handle(slug)
    st = appdirs.read_app_state(h)  # nothing else writes the state yet
    st["config_overrides"] = {**(st.get("config_overrides") or {}), "score.gates.utmos_min": 99.0}  # no take can pass: all are 'review'
    appdirs.write_app_state(h, st)
    appdirs.write_project_config(h)

    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as e:  # browser binaries missing
            pytest.skip(f"Playwright Chromium not available: {e}")
        page = browser.new_page(viewport=VIEWPORT, locale="ru-RU")
        page_errors: list[str] = []
        page.on("pageerror", lambda e: page_errors.append(str(e)))
        tid = page.get_by_test_id
        shots = os.environ.get("VNDUB_COVERAGE_SHOTS")

        def snap(name: str) -> None:
            if shots:
                Path(shots).mkdir(parents=True, exist_ok=True)
                page.wait_for_timeout(400)
                page.screenshot(path=str(Path(shots) / f"{name}.png"), full_page=True)

        try:
            # ---- Step 5: the run, then what it built -------------------------------------------------------
            page.goto(server.url + "/#/step/run")
            tid("run-start").click()
            expect(tid("run-done")).to_be_visible(timeout=600 * S)
            expect(tid("run-coverage")).to_be_visible(timeout=60 * S)
            cov = api.get("/api/patch/coverage").json()["coverage"]
            assert cov["provisional"] > 0 and cov["left_out"] == 0 and cov["installed"] == cov["dub_lines"]

            def check_callout() -> None:
                expect(tid("run-coverage")).to_have_class(re.compile(r"callout-warn"))  # something is provisional
                expect(tid("run-coverage-installed")).to_have_text(f"Озвучено реплик: {cov['installed']} из {cov['dub_lines']}.")
                expect(tid("run-coverage-provisional")).to_contain_text(f"Предварительно озвучено реплик: {cov['provisional']}.")
                expect(tid("run-coverage-provisional")).to_contain_text("на шаге «Проверка»")
                expect(tid("run-coverage-left-out")).to_have_count(0)  # nothing was left out: no line about it
                if cov["narration"]:
                    expect(tid("run-coverage-narration")).to_contain_text(f"Строк повествования без озвучки: {cov['narration']}.")
                else:
                    expect(tid("run-coverage-narration")).to_have_count(0)

            check_callout()
            snap("run-done")
            job = api.get("/api/jobs", params={"project": slug}).json()
            full = next(j for j in job if j["kind"] == "full")
            assert full["result"]["coverage"] == cov, "the run's own result carries what the screen shows"

            page.reload()  # the app opened on a finished run: no job to read the numbers from
            expect(tid("run-coverage")).to_be_visible(timeout=60 * S)
            check_callout()

            # ---- Step 6: the same lines are marked in the review -----------------------------------------------
            tid("run-coverage-review").click()
            expect(tid("screen-review")).to_be_visible(timeout=30 * S)
            expect(tid("review-detail")).to_be_visible(timeout=60 * S)
            chip = page.locator('[data-testid^="review-row-patch-"]').first
            expect(chip).to_have_text("предварительная")
            expect(tid("review-detail-patch")).to_have_text("в патче как предварительная")
            expect(tid("review-detail-status")).to_have_text("нужна проверка")
            snap("review")

            # ---- Step 7: the three numbers beside the file count --------------------------------------------------
            page.goto(server.url + "/#/step/install")
            expect(tid("install-coverage")).to_be_visible(timeout=60 * S)
            expect(tid("install-coverage-installed")).to_have_text(f"Озвучено реплик: {cov['installed']} из {cov['dub_lines']}.")
            expect(tid("install-coverage-provisional")).to_have_text(f"Предварительных: {cov['provisional']}")
            expect(tid("install-coverage-left-out")).to_have_text("Не попало в патч: 0")
            snap("install")

            # ---- a project imported by an older version -----------------------------------------------------------
            status_path = h.work / "stage_status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status["parse"]["version"] = "2"
            status_path.write_text(json.dumps(status), encoding="utf-8")
            page.goto(server.url + "/#/step/run")
            page.reload()
            expect(tid("run-stale-import")).to_contain_text("Игра разобрана прошлой версией программы")
            tid("run-stale-import-go").click()
            expect(tid("game-import-stale")).to_be_visible(timeout=30 * S)
            snap("game-stale")
            tid("game-import-stale-run").click()
            expect(tid("game-job")).to_have_attribute("data-state", "done", timeout=300 * S)
            expect(tid("game-import-stale")).to_have_count(0)  # the import ran: the project is current again
            assert api.get("/api/project").json()["stale"] == []
        except Exception:
            page.screenshot(path=str(FAIL_SHOT), full_page=True)
            raise
        finally:
            browser.close()
    assert not page_errors, f"uncaught errors in the page: {page_errors}"

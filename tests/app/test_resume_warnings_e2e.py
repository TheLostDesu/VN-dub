"""Playwright: the warnings callout of a job and the resume dialog (shutdown checkbox, the other interrupted jobs).

All-mock profile, the pipeline flows replaced by stubs: what is under test is what the SPA shows of ``result.warnings``
and of ``project.resume`` (``job-warnings``, ``resume-keep-shutdown``, ``resume-candidates``), and what a click sends
(``keep_shutdown``). Same runner and skip rules as ``test_wizard_e2e.py``. Screenshots go to ``$VNDUB_E2E_SHOTS`` when it
is set (else they are thrown away with the temporary folder); ``docs/img/app`` is not touched.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from test_wizard_e2e import AUDIT_JS, DIST, FAIL_SHOT, VIEWPORT, S, server  # noqa: F401  (``server`` is the fixture)

from vndub_app import jobs, pipeline

pytestmark = [pytest.mark.e2e, pytest.mark.slow, pytest.mark.timeout(600)]

sync_api = pytest.importorskip("playwright.sync_api")

WARNINGS = [
    {"kind": "substitution", "requested": "gemma4_12b", "used": "qwen35_9b", "reason": "gemma4_12b: needs 9.8 GB, 4.1 GB free"},
    {"kind": "unavailable", "stage": "score", "group": "cer", "reason": "no ASR backend: out of video memory"},
    {"kind": "mixed_models", "models": "gemma4_12b: 812, qwen35_9b: 140"},
    {"kind": "settings_changed", "keys": "gpu_budget_gb, translation_llm"},
]


def _ago(hours: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(timespec="seconds")


def _stub(kind: str, run: Any) -> None:
    real = pipeline.FLOWS[kind]
    pipeline.FLOWS[kind] = pipeline.Flow(kind, lambda slug, p: run, gpu=real.gpu, heavy=real.heavy)


def test_warnings_callout_and_resume_dialog(server: Any, game_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    if not DIST.exists():
        pytest.skip("SPA not built: run `npm run build` in vndub_app/web")
    expect = sync_api.expect
    shots = Path(os.environ.get("VNDUB_E2E_SHOTS") or tmp_path / "shots")
    shots.mkdir(parents=True, exist_ok=True)
    monkeypatch.setitem(pipeline.FLOWS, "import", pipeline.FLOWS["import"])  # restored after the test
    monkeypatch.setitem(pipeline.FLOWS, "full", pipeline.FLOWS["full"])

    def import_with_warnings(job: jobs.JobContext) -> dict[str, Any]:
        job.set_stages(["detect"])
        job.begin_stage("detect")
        job.add_warnings(WARNINGS)
        return {"lines": 0, "voiced": 0}

    _stub("import", import_with_warnings)
    _stub("full", lambda job: None)

    problems: list[str] = []
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as e:  # browser binaries missing
            pytest.skip(f"Playwright Chromium not available: {e}")
        page = browser.new_context(viewport=VIEWPORT, locale="ru-RU").new_page()
        page_errors: list[str] = []
        page.on("pageerror", lambda e: page_errors.append(str(e)))
        tid = page.get_by_test_id

        def shot(name: str) -> None:
            page.wait_for_timeout(400)
            page.screenshot(path=str(shots / f"{name}.png"), full_page=False)
            problems.extend(f"{name}: {x}" for x in page.evaluate(AUDIT_JS))

        try:
            # 1. a finished job with warnings: one warn callout, a line per warning, above the log toggle
            page.goto(server.url + "/#/step/game")
            tid("game-pick-folder").click()
            tid("game-path-input").fill(str(game_dir))
            tid("game-path-open").click()
            expect(tid("game-job")).to_have_attribute("data-state", "done", timeout=60 * S)
            box = tid("job-warnings")
            expect(box).to_be_visible()
            expect(box.locator("li")).to_have_count(4)
            expect(box.locator("li").nth(0)).to_contain_text("qwen35_9b")
            expect(box.locator("li").nth(1)).to_contain_text("no ASR backend")
            expect(box.locator("li").nth(3)).to_contain_text("модель перевода")  # the settings keys are named in words
            above_log = page.evaluate(
                "() => { const w = document.querySelector('[data-testid=job-warnings]'); const l = document.querySelector('[data-testid=job-log-toggle]');"
                " return !l || (w.compareDocumentPosition(l) & Node.DOCUMENT_POSITION_FOLLOWING) !== 0; }"
            )
            assert above_log
            shot("warnings-callout")

            # 2. a restart with two interrupted jobs: the long full run is offered, the listen is named, the shutdown is unchecked
            state = Path(os.environ["VNDUB_DATA"]) / "jobs.json"
            slug = page.evaluate("() => fetch('/api/project').then(r => r.json()).then(p => p.slug)")
            def rec(job_id: str, kind: str, hours: float, params: dict[str, Any]) -> dict[str, Any]:
                return jobs.JobRecord(
                    id=job_id, kind=kind, title_key=f"jobs.{kind}", params=params, project=slug, state="interrupted",
                    created=_ago(hours), finished=_ago(hours - 0.1),
                ).info(with_log=True)

            state.write_text(json.dumps([
                rec("full1", "full", 30, {"shutdown_when_done": True}),
                rec("listen1", "casting_listen", 1, {"speaker_id": "sakura", "n": 3}),
            ]), encoding="utf-8")
            jobs.reset_runner(jobs.JobRunner(state, None))
            page.goto(server.url + "/#/step/translate")
            page.reload()
            dialog = tid("resume-dialog")
            expect(dialog).to_be_visible(timeout=30 * S)
            expect(tid("resume-keep-shutdown")).not_to_be_checked()
            expect(tid("resume-candidates")).to_contain_text("Ещё прерваны")
            expect(tid("resume-candidates")).to_contain_text("Проба звучания")  # the casting listen, by its title
            shot("resume-dialog")

            # 3. Continue with the checkbox ticked sends keep_shutdown: the new full run keeps its shutdown
            tid("resume-keep-shutdown").check()
            tid("resume-continue").click()
            expect(dialog).to_be_hidden(timeout=30 * S)
            page.wait_for_timeout(500)
            new_full = [j for j in jobs.runner().list(slug) if j.kind == "full" and j.state != "interrupted"]
            assert new_full and new_full[-1].params == {"shutdown_when_done": True}
            assert [j.id for j in jobs.runner().interrupted(slug)] == ["listen1"]  # the listen stays interrupted

            # 4. a second restart: without the tick the shutdown is dropped; with no other job the line is not shown
            state.write_text(json.dumps([rec("full2", "full", 2, {"shutdown_when_done": True})]), encoding="utf-8")
            jobs.reset_runner(jobs.JobRunner(state, None))
            page.reload()
            expect(tid("resume-dialog")).to_be_visible(timeout=30 * S)
            expect(tid("resume-candidates")).to_have_count(0)
            tid("resume-continue").click()
            expect(tid("resume-dialog")).to_be_hidden(timeout=30 * S)
            page.wait_for_timeout(500)
            last = [j for j in jobs.runner().list(slug) if j.kind == "full" and j.state != "interrupted"][-1]
            assert last.params == {}

            # 5. an interrupted job without a shutdown: no checkbox at all
            state.write_text(json.dumps([rec("full3", "full", 2, {})]), encoding="utf-8")
            jobs.reset_runner(jobs.JobRunner(state, None))
            page.reload()
            expect(tid("resume-dialog")).to_be_visible(timeout=30 * S)
            expect(tid("resume-keep-shutdown")).to_have_count(0)
        except Exception:
            page.screenshot(path=str(FAIL_SHOT), full_page=True)
            raise
        browser.close()

    assert not page_errors, f"uncaught errors in the page: {page_errors}"
    assert not problems, "uneven layout:\n  " + "\n  ".join(problems)

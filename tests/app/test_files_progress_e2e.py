"""Playwright: the per-file progress of a model download is a box of its own with its own scrollbar.

A real uvicorn serves the built SPA against a throw-away data folder in the all-mock profile; the Hugging Face hub is replaced
with a fake that offers many files per repo and parks the download on chosen files, so the test can look at the Models screen
in the middle of a 28-file download:

* the ``files-progress`` list keeps a fixed height (it scrolls inside, the card and the page do not grow),
* the summary line above it counts the files and the megabytes,
* the file that is being downloaded stays inside the visible part of the box (the box scrolls, not the page),
* a reader who scrolled away is not dragged back when the next file starts.

Set ``VNDUB_FILES_SHOT`` to a path to keep the screenshot of the screen during the download.
Skipped when the SPA is not built or Chromium is missing (same as ``test_wizard_e2e.py``).
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import pytest
from test_wizard_e2e import DIST, FAIL_SHOT, VIEWPORT, S, server  # noqa: F401  (``server`` is the fixture)

from vndub_app import models_api

pytestmark = [pytest.mark.e2e, pytest.mark.slow, pytest.mark.timeout(600)]

sync_api = pytest.importorskip("playwright.sync_api")

WANTED = ["whisper_tiny", "gigaam_v3", "ecapa", "wavlm_sv"]  # four plain Hugging Face repos of the `main` environment
PER_REPO = 7
FILE_BYTES = 12_000_000  # what the listing says; the fake writes a few bytes only
# the downloads (0-based, in order) the fake hub waits on; the rows of a repo appear when its download starts, so from the 4th
# repo on (downloads 21..27) the list has all 28 rows
PARK = {22, 24, 26}


class ManyFilesHub:
    """``models_api.HubClient``: ``PER_REPO`` files for every repo; ``fetch`` number ``i`` in ``park`` waits for ``release(i)``."""

    def __init__(self, park: set[int] = PARK, fail_at: int | None = None) -> None:
        self.park = park
        self.fail_at = fail_at  # the download (0-based) that raises, with a long unbroken message
        self.count = 0
        self.reached = {i: threading.Event() for i in park}
        self.gates = {i: threading.Event() for i in park}
        self.lock = threading.Lock()

    def list_files(self, repo: str, revision: str | None, allow: tuple[str, ...]) -> list[tuple[str, int]]:
        return [(f"shard-{i:02d}.safetensors", FILE_BYTES + i * 100_000) for i in range(PER_REPO)]

    def fetch(self, repo: str, filename: str, revision: str | None, local: Path, progress: Any) -> None:
        with self.lock:
            n = self.count
            self.count += 1
        total = FILE_BYTES
        progress(total // 2, total)
        if n == self.fail_at:
            raise OSError("Connection reset: https://cdn.example.org/" + "x" * 240)
        if n in self.park:
            self.reached[n].set()
            if not self.gates[n].wait(120):
                raise TimeoutError(f"test did not release download {n}")
        out = local / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"x" * 16)
        progress(total, total)

    def release(self, i: int) -> None:
        self.gates[i].set()


@pytest.fixture
def hub() -> Any:
    h = ManyFilesHub()
    models_api.set_hub(h)
    yield h
    for g in h.gates.values():  # never leave the job thread parked
        g.set()
    models_api.set_hub(None)


@pytest.fixture
def failing_hub() -> Any:
    h = ManyFilesHub(park=set(), fail_at=25)
    models_api.set_hub(h)
    yield h
    models_api.set_hub(None)


SELECT_WANTED = """(wanted) => {
  for (const el of document.querySelectorAll('input[data-testid^="model-check-"]')) {
    const name = el.dataset.testid.slice('model-check-'.length);
    if (el.checked !== wanted.includes(name)) el.click();
  }
}"""


def test_files_progress_is_a_scrolling_box(server: Any, hub: ManyFilesHub, tmp_path: Path) -> None:  # noqa: F811
    if not DIST.exists():
        pytest.skip("SPA not built: run `npm run build` in vndub_app/web")
    expect = sync_api.expect
    shot = Path(os.environ.get("VNDUB_FILES_SHOT") or tmp_path / "models-progress.png")
    shot.parent.mkdir(parents=True, exist_ok=True)

    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as e:  # browser binaries missing
            pytest.skip(f"Playwright Chromium not available: {e}")
        page = browser.new_page(viewport=VIEWPORT, locale="ru-RU", color_scheme=os.environ.get("VNDUB_FILES_SCHEME", "light"))  # type: ignore[arg-type]
        page_errors: list[str] = []
        page.on("pageerror", lambda e: page_errors.append(str(e)))
        tid = page.get_by_test_id

        def box() -> dict[str, Any]:
            """Geometry of the list, of its active row and of the card, read in one go."""
            return page.evaluate(
                """() => {
                  const ul = document.querySelector('[data-testid="files-progress"]');
                  const act = ul.querySelector('li[data-state="downloading"]');
                  const b = ul.getBoundingClientRect();
                  const a = act ? act.getBoundingClientRect() : null;
                  const css = getComputedStyle(ul);
                  return {
                    client: ul.clientHeight, scroll: ul.scrollHeight, top: ul.scrollTop,
                    maxHeight: parseFloat(css.maxHeight), overflowY: css.overflowY,
                    rows: ul.children.length,
                    active: act ? act.dataset.fileId : null,
                    inView: a ? (a.top >= b.top - 1 && a.bottom <= b.bottom + 1) : null,
                    pageWidth: document.documentElement.scrollWidth, winWidth: window.innerWidth,
                  };
                }"""
            )

        try:
            page.goto(server.url + "/#/models")
            expect(tid("models-download")).to_be_visible(timeout=60 * S)
            # exactly the four models the fake hub serves
            page.evaluate(SELECT_WANTED, WANTED)
            for name in WANTED:
                expect(tid(f"model-check-{name}")).to_be_checked()
            expect(tid("models-download")).to_be_enabled()
            tid("models-download").click()

            # the 23rd file is being downloaded: 22 are done, 5 wait
            assert hub.reached[22].wait(60), "the download never reached file 22"
            expect(tid("files-summary")).to_contain_text("Файлов: 22 из 28", timeout=30 * S)
            expect(tid("files-progress").locator("li")).to_have_count(28)
            expect(tid("files-progress").locator('li[data-state="downloading"]')).to_have_count(1)
            page.wait_for_timeout(500)  # the follow-scroll runs after the render
            g = box()
            assert g["overflowY"] in ("auto", "scroll"), g
            assert g["maxHeight"] > 0, g  # a real limit, not "none"
            assert g["client"] <= g["maxHeight"] + 1, g
            assert g["scroll"] > g["client"] + 100, f"28 rows must not fit the box: {g}"
            assert g["top"] > 0, f"the box did not follow the active file: {g}"
            assert g["inView"] is True, f"the active file is outside the visible part of the box: {g}"
            assert g["pageWidth"] <= g["winWidth"], g
            # the summary tells the megabytes too: 22 done files of FILE_BYTES + i * 100 kB each
            text = tid("files-summary").inner_text()
            assert " из " in text and "MB" in text, text

            # the page itself did not scroll to follow the row (the box did)
            card_h = page.evaluate("document.querySelector('[data-testid=\"models-download-job\"]').getBoundingClientRect().height")
            assert card_h < 700, card_h
            # the screen as the user sees it: the download card right under the sticky bar
            page.evaluate(
                """() => {
                  const job = document.querySelector('[data-testid="models-download-job"]');
                  window.scrollTo(0, window.scrollY + job.getBoundingClientRect().top - 150);
                }"""
            )
            page.wait_for_timeout(300)
            page.screenshot(path=str(shot))

            # a reader who scrolled the list away is left alone when the next file starts
            page.evaluate("document.querySelector('[data-testid=\"files-progress\"]').scrollTop = 0")
            before_top = box()["top"]
            assert before_top == 0
            hub.release(22)
            assert hub.reached[24].wait(60), "the download never reached file 24"  # 23 went by, 24 is parked
            page.wait_for_timeout(500)
            g = box()
            assert g["top"] == 0, f"the box was dragged back to the active file: {g}"
            assert g["inView"] is False, g

            # back on the active row: from there the box follows again
            page.evaluate(
                """() => {
                  const ul = document.querySelector('[data-testid="files-progress"]');
                  const act = ul.querySelector('li[data-state="downloading"]');
                  ul.scrollTop = act.offsetTop - ul.clientHeight / 2;
                }"""
            )
            page.wait_for_timeout(300)
            assert box()["inView"] is True
            hub.release(24)
            assert hub.reached[26].wait(60), "the download never reached file 26"
            page.wait_for_timeout(500)
            g = box()
            assert g["inView"] is True, f"the box did not follow the active file again: {g}"
            assert g["client"] <= g["maxHeight"] + 1, g

            # finish: every file done, the summary says so, the box keeps its size
            hub.release(26)
            expect(tid("models-download-job")).to_have_attribute("data-state", "done", timeout=90 * S)
            expect(tid("files-summary")).to_contain_text("Файлов: 28 из 28", timeout=15 * S)
            expect(tid("files-summary-failed")).to_have_count(0)
            expect(tid("files-progress").locator('li[data-state="done"]')).to_have_count(28)
            g = page.evaluate(
                """() => { const ul = document.querySelector('[data-testid="files-progress"]'); return {c: ul.clientHeight, m: parseFloat(getComputedStyle(ul).maxHeight), s: ul.scrollHeight}; }"""
            )
            assert g["c"] <= g["m"] + 1 and g["s"] > g["c"], g
            # the log is a closed collapsible by default: it does not add its height to the card
            expect(tid("job-log-toggle")).to_have_attribute("aria-expanded", "false")
            expect(tid("job-log")).to_have_count(0)
        except Exception:
            page.screenshot(path=str(FAIL_SHOT), full_page=True)
            raise
        finally:
            browser.close()

    assert not page_errors, f"uncaught errors in the page: {page_errors}"


def test_failed_files_stay_visible_and_counted(server: Any, failing_hub: ManyFilesHub) -> None:  # noqa: F811
    """A download that breaks half way: the summary counts the failed files, the first of them is in the box, long messages wrap."""
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
        try:
            page.goto(server.url + "/#/models")
            expect(tid("models-download")).to_be_visible(timeout=60 * S)
            page.evaluate(SELECT_WANTED, WANTED)
            expect(tid("models-download")).to_be_enabled()
            tid("models-download").click()
            expect(tid("models-download-job")).to_have_attribute("data-state", "failed", timeout=90 * S)
            expect(tid("files-progress").locator("li")).to_have_count(28)
            expect(tid("files-summary")).to_contain_text("Файлов: 25 из 28", timeout=15 * S)
            expect(tid("files-summary-failed")).to_have_text("ошибок: 3")
            expect(tid("files-progress").locator('li[data-state="failed"]')).to_have_count(3)
            g = page.evaluate(
                """() => {
                  const ul = document.querySelector('[data-testid="files-progress"]');
                  const first = ul.querySelector('li[data-state="failed"]');
                  const b = ul.getBoundingClientRect();
                  const a = first.getBoundingClientRect();
                  const err = document.querySelector('.error-panel');
                  return {
                    client: ul.clientHeight, max: parseFloat(getComputedStyle(ul).maxHeight), scroll: ul.scrollHeight,
                    firstInView: a.top >= b.top - 1 && a.bottom <= b.bottom + 1,
                    errWide: err ? err.scrollWidth - err.clientWidth : null,
                    pageWide: document.documentElement.scrollWidth - window.innerWidth,
                  };
                }"""
            )
            assert g["client"] <= g["max"] + 1 and g["scroll"] > g["client"], g
            assert g["firstInView"] is True, g
            assert g["errWide"] is not None and g["errWide"] <= 1, g  # the long URL in the message wraps
            assert g["pageWide"] <= 0, g
        except Exception:
            page.screenshot(path=str(FAIL_SHOT), full_page=True)
            raise
        finally:
            browser.close()

    assert not page_errors, f"uncaught errors in the page: {page_errors}"

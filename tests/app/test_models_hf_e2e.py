"""Playwright: the "Add a model by link" card of the Models screen, the recommended chips and the Settings choice.

A real uvicorn serves the built SPA against a throw-away data folder; the Hugging Face clients are replaced with fakes
(no network): paste a link, check it, download the suggested file, see it in the list of added models, pick it as the
translation model in Settings, delete it. Screenshot: ``docs/img/app/08b-models-hf.png``.

Skipped when the SPA is not built or Chromium is missing (same as ``test_wizard_e2e.py``).
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from hf_helpers import FakeHub, FakeInfo, gguf, repo_info
from test_wizard_e2e import DIST, FAIL_SHOT, SHOTS, VIEWPORT, S, server  # noqa: F401  (``server`` is the fixture)

from vndub_app import hf_custom, models_api

pytestmark = [pytest.mark.e2e, pytest.mark.slow, pytest.mark.timeout(600)]

sync_api = pytest.importorskip("playwright.sync_api")

REPO = "acme/Cool-MT-7B-GGUF"
CID = "hf_acme_cool_mt_7b_gguf"
FILES = {"cool-mt-7b-Q8_0.gguf": gguf(400), "cool-mt-7b-Q4_K_M.gguf": gguf(200), "cool-mt-7b-IQ2_XS.gguf": gguf(100), "README.md": b"hello"}
SHOWN_SIZES = {"cool-mt-7b-Q8_0.gguf": 1_600_000_000, "cool-mt-7b-Q4_K_M.gguf": 900_000_000, "cool-mt-7b-IQ2_XS.gguf": 400_000_000, "README.md": 3_100}


@pytest.fixture
def fakes() -> Any:
    yield
    hf_custom.set_info_client(None)
    models_api.set_hub(None)


def test_add_a_model_by_link(server: Any, fakes: Any) -> None:  # noqa: F811
    if not DIST.exists():
        pytest.skip("SPA not built: run `npm run build` in vndub_app/web")
    expect = sync_api.expect
    SHOTS.mkdir(parents=True, exist_ok=True)
    infos = {
        REPO: repo_info(REPO, SHOWN_SIZES, license="apache-2.0"),
        "unsloth/gemma-4-12b-it-GGUF": repo_info("unsloth/gemma-4-12b-it-GGUF", {"gemma-4-12b-it-IQ4_XS.gguf": 6_400_000_000}, license="apache-2.0"),
        "someone/locked": repo_info("someone/locked", {"a.bin": 10}, gated="manual"),
    }
    hf_custom.set_info_client(FakeInfo(infos))
    hub = FakeHub({REPO: FILES, "someone/locked": {"a.bin": b"x" * 10}})
    models_api.set_hub(hub)

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
            expect(tid("models-hf-card")).to_be_visible(timeout=60 * S)
            expect(tid("models-hf-url")).to_have_attribute("placeholder", "https://huggingface.co/owner/model")

            # recommended models carry a chip in the table; recommended links fill the field
            expect(tid("model-recommended-gemma4_12b")).to_be_visible()
            expect(tid("model-recommended-fish_s2")).to_have_count(0)  # non-commercial, only in the "all" preset
            expect(tid("models-hf-recommended")).to_be_visible()
            n_reco = page.locator('[data-testid^="models-hf-reco-"]').count()
            assert 2 <= n_reco <= 4
            repo0 = tid("models-hf-reco-0").get_attribute("data-repo")
            tid("models-hf-reco-0").click()
            expect(tid("models-hf-url")).to_have_value(f"https://huggingface.co/{repo0}")

            # an address of a model that is already in the table selects its row
            tid("models-hf-url").fill("hf.co/unsloth/gemma-4-12b-it-GGUF")
            tid("models-hf-inspect").click()
            expect(tid("models-hf-result")).to_have_attribute("data-kind", "builtin", timeout=30 * S)
            expect(tid("model-check-gemma4_12b")).to_be_checked()
            expect(tid("models-hf-add")).to_have_count(0)

            # a gated repo explains the licence
            tid("models-hf-url").fill("someone/locked")
            tid("models-hf-inspect").click()
            expect(tid("models-hf-warn-gated")).to_be_visible(timeout=30 * S)

            # a bad address is an error with a hint, not a crash
            tid("models-hf-url").fill("https://example.com/acme/Cool-MT-7B-GGUF")
            tid("models-hf-inspect").click()
            expect(tid("error-panel").first).to_have_attribute("data-code", "hf_bad_url", timeout=30 * S)

            # the real flow: paste, check, pick the suggested quantization, download
            tid("models-hf-url").fill(f"https://huggingface.co/{REPO}/tree/main?download=true")
            tid("models-hf-inspect").click()
            expect(tid("models-hf-result")).to_have_attribute("data-kind", "gguf_llm", timeout=30 * S)
            names = FILES.keys()
            order = sorted(names)  # the listing is sorted by name
            suggested = order.index("cool-mt-7b-Q4_K_M.gguf")
            for i, n in enumerate(order):
                box = tid(f"models-hf-file-{i}")
                if i == suggested:
                    expect(box).to_be_checked()
                else:
                    expect(box).not_to_be_checked()
            expect(tid("models-hf-total")).to_contain_text("Выбрано файлов: 1")
            tid("models-hf-add").click()
            expect(tid("models-hf-job")).to_have_attribute("data-state", "done", timeout=60 * S)
            expect(tid("models-hf-done")).to_be_visible()
            expect(tid(f"models-custom-row-{CID}")).to_be_visible()
            expect(tid(f"models-custom-row-{CID}")).to_have_attribute("data-present", "true")
            # one tall viewport so the card is captured in one piece (no sticky top bar in the middle of it)
            page.set_viewport_size({"width": VIEWPORT["width"], "height": 1750})
            page.evaluate("window.scrollTo(0, 0)")
            page.wait_for_timeout(500)
            tid("models-hf-card").screenshot(path=str(SHOTS / "08b-models-hf.png"))
            page.set_viewport_size(VIEWPORT)

            # Settings: the new model can be picked as the translation model
            page.goto(server.url + "/#/settings")
            expect(tid("settings-translation-llm-select")).to_be_visible(timeout=30 * S)
            expect(tid("settings-translation-llm-select").locator("option")).to_have_count(2, timeout=15 * S)  # default + the new model
            options = tid("settings-translation-llm-select").locator("option").evaluate_all("els => els.map(e => e.value)")
            assert options[0] == "" and CID in options
            tid("settings-translation-llm-select").select_option(CID)
            page.wait_for_function(
                "async (id) => (await (await fetch('/api/settings')).json()).translation_llm === id", arg=CID, timeout=15 * S
            )

            # delete it again (confirm dialog)
            page.goto(server.url + "/#/models")
            expect(tid(f"models-custom-row-{CID}")).to_be_visible(timeout=30 * S)
            tid(f"models-custom-delete-{CID}").click()
            expect(tid("models-custom-delete-dialog")).to_be_visible()
            tid("models-custom-delete-confirm").click()
            expect(tid(f"models-custom-row-{CID}")).to_have_count(0, timeout=15 * S)
            expect(tid("models-custom-empty")).to_be_visible()
        except Exception:
            page.screenshot(path=str(FAIL_SHOT), full_page=True)
            raise
        finally:
            browser.close()

    assert not page_errors, f"uncaught errors in the page: {page_errors}"
    from vndub_app import appdirs

    saved = json.loads((appdirs.data_dir() / "custom_models.json").read_text(encoding="utf-8"))["models"]
    assert saved == []  # deleting removed the entry
    assert not (appdirs.effective_models_dir() / "custom" / "acme__Cool-MT-7B-GGUF").exists()
    assert hub.fetched == [(REPO, "cool-mt-7b-Q4_K_M.gguf")]  # only the suggested file was downloaded

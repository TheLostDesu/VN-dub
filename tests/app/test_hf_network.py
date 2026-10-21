"""Real Hugging Face, from the app's API: download a registry weight (cancel and resume), inspect and add a model by link.

Skipped unless ``VNDUB_NETWORK_TESTS=1`` (``uv run pytest tests/app/test_hf_network.py -m network -s`` with it set).
Everything goes into a throw-away data folder; the total download is about 90 MB (``speechbrain/spkrec-ecapa-voxceleb``,
the smallest default-profile weight) plus a 1 MB GGUF (``ggml-org/tiny-llamas``).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Iterator

import pytest
from o_helpers import client_for, drop_runner, fresh_runner, make_env

from vndub.backends import registry
from vndub.backends.llm.llamacpp import find_gguf
from vndub.backends.weights import MARKER
from vndub_app import appdirs, downloads, hf_custom, jobs, models_api
from vndub_app.routes import general

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(os.environ.get("VNDUB_NETWORK_TESTS") != "1", reason="set VNDUB_NETWORK_TESTS=1 to talk to the real huggingface.co"),
]

ECAPA_REPO = "speechbrain/spkrec-ecapa-voxceleb"
BIG_FILE = "embedding_model.ckpt"  # 83 MB: the one to interrupt


@pytest.fixture
def net(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    data = make_env(monkeypatch, tmp_path)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_home"))  # nothing is cached in the user's profile
    monkeypatch.setenv(registry.CUSTOM_ENV, "")
    hf_custom.sync_env()
    models_api.set_hub(None)
    hf_custom.set_info_client(None)
    fresh_runner()
    c = client_for(models_api, hf_custom, general)
    c.data = data  # type: ignore[attr-defined]
    yield c
    drop_runner()
    registry.all_specs()


def wait(job_id: str, timeout: float = 240.0) -> dict[str, Any]:
    t0 = time.monotonic()
    while True:
        rec = jobs.runner().get(job_id)
        if rec.state in ("done", "failed", "cancelled", "interrupted"):
            return rec.info()
        assert time.monotonic() - t0 < timeout, f"job {job_id} still {rec.state}"
        time.sleep(0.1)


def test_registry_weight_downloads_cancels_mid_file_and_resumes(net: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The built-in path of the Models screen against the real hub: per-file progress, prompt cancel, byte-level resume."""
    started: dict[str, Any] = {}
    real_fetch = downloads.fetch

    def spy(url: str, dest: Path, **kw: Any) -> Any:
        res = real_fetch(url, dest, **kw)
        started[dest.name] = res
        return res

    monkeypatch.setattr(downloads, "fetch", spy)
    row0 = next(r for r in net.get("/api/models").json()["rows"] if r["name"] == "ecapa")
    assert row0["present"] is False and row0["disk_gb"] == 0
    models = net.data / "models" / "spkrec-ecapa-voxceleb"

    # -- run 1: cancel while the big file is on its way
    t0 = time.monotonic()
    job = net.post("/api/models/download", json={"names": ["ecapa"]}).json()
    seen_states: set[str] = set()
    cancelled_at = 0
    while True:
        rec = jobs.runner().get(job["id"])
        files = {f["filename"]: f for f in rec.result.get("files", [])}
        big = files.get(BIG_FILE)
        seen_states |= {f["state"] for f in files.values()}
        if big and big["state"] == "downloading" and big["done"] >= 10_000_000 and not cancelled_at:
            cancelled_at = big["done"]
            t_cancel = time.monotonic()
            jobs.runner().cancel(job["id"])
        if rec.state in ("done", "failed", "cancelled"):
            break
        assert time.monotonic() - t0 < 240, "download too slow / stuck"
        time.sleep(0.05)
    first = wait(job["id"])
    cancel_lag = time.monotonic() - t_cancel
    assert first["state"] == "cancelled", first.get("error")
    assert cancel_lag < 5.0, f"cancel took {cancel_lag:.1f}s: it must not wait for the end of the file"
    assert {"pending", "downloading", "done"} <= seen_states
    part = models / f"{BIG_FILE}.part"
    assert part.exists() and 10_000_000 <= part.stat().st_size < 83_316_686 and not (models / BIG_FILE).exists()
    assert (models / "config.json").exists() and not (models / MARKER).exists()
    assert next(r for r in net.get("/api/models").json()["rows"] if r["name"] == "ecapa")["present"] is False
    left = part.stat().st_size

    # -- run 2: the rest only
    t1 = time.monotonic()
    job2 = net.post("/api/models/download", json={"names": ["ecapa"]}).json()
    second = wait(job2["id"])
    t_second = time.monotonic() - t1
    assert second["state"] == "done", second.get("error")
    big_res = started[BIG_FILE]
    assert big_res.resumed_from == left > 0 and big_res.verified  # continued at the byte it stopped at, sha256 checked
    assert (models / BIG_FILE).stat().st_size == 83_316_686 and not part.exists() and (models / MARKER).exists()
    files = second["result"]["files"]
    assert files and all(f["state"] == "done" and f["done"] == f["total"] for f in files)
    assert {f["filename"] for f in files if f.get("skipped")} >= {"config.json", "classifier.ckpt"}  # finished in run 1
    row = next(r for r in net.get("/api/models").json()["rows"] if r["name"] == "ecapa")
    assert row["present"] is True and row["reason"] != "weights_missing" and 0.08 < row["disk_gb"] < 0.12  # (env_missing is fine: no backend envs here)
    print(f"\n[network] ecapa: cancelled at {cancelled_at/1e6:.1f} MB (stopped {cancel_lag:.2f}s after the request, .part = {left/1e6:.1f} MB); "
          f"first run {t_cancel - t0:.1f}s, resume {t_second:.1f}s from byte {big_res.resumed_from}; {len(files)} files, {sum(f['total'] for f in files)/1e6:.1f} MB")


def test_inspect_and_add_a_real_tiny_gguf_then_use_it_as_the_translation_model(net: Any, tmp_path: Path) -> None:
    r = net.post("/api/models/hf/inspect", json={"url": "https://huggingface.co/ggml-org/tiny-llamas/tree/main?download=true"})
    assert r.status_code == 200, r.text
    ins = r.json()
    assert ins["repo"] == "ggml-org/tiny-llamas" and ins["kind"] == "gguf_llm" and ins["gated"] is False and ins["builtin"] is None
    names = {f["name"] for f in ins["files"]}
    assert {"stories260K.gguf", "stories15M-q4_0.gguf"} <= names
    assert ins["suggested"] == ["stories260K.gguf"]  # no Q4_K_M here: the smallest single file
    assert 1_000_000 < ins["suggested_size"] < 2_000_000 and ins["total_size"] > 100_000_000 and ins["free_gb"] > 0

    job = net.post("/api/models/hf/add", json={"repo": ins["repo"], "revision": ins["revision"], "files": ins["suggested"], "kind": "gguf_llm", "label": "Tiny stories"}).json()
    done = wait(job["id"], 120)
    assert done["state"] == "done", done.get("error")
    d = net.data / "models" / "custom" / "ggml-org__tiny-llamas"
    assert (d / "stories260K.gguf").read_bytes()[:4] == b"GGUF" and (d / MARKER).exists()
    rows = done["result"]["files"]
    assert [f["filename"] for f in rows] == ["stories260K.gguf"] and rows[0]["state"] == "done" and rows[0]["done"] == rows[0]["total"] > 1_000_000
    lst = net.get("/api/models/custom").json()
    assert [(e["id"], e["present"], e["usable"], e["label"]) for e in lst] == [("hf_ggml_org_tiny_llamas", True, True, "Tiny stories")]

    # the pipeline can find and use it
    spec = registry.get("hf_ggml_org_tiny_llamas")
    assert find_gguf(net.data / "models", spec) == d / "stories260K.gguf"
    game = tmp_path / "game"
    game.mkdir()
    h = appdirs.create_project(game, name="G")
    assert net.put("/api/settings", json={"translation_llm": "hf_ggml_org_tiny_llamas"}).status_code == 200
    from vndub.config import load_config

    appdirs.write_project_config(h)
    assert load_config(h.config_file).translate.llm.backend == "hf_ggml_org_tiny_llamas"
    assert net.delete("/api/models/custom/hf_ggml_org_tiny_llamas").json() == {"ok": True} and not d.exists()


def test_inspect_errors_against_the_real_hub(net: Any) -> None:
    r = net.post("/api/models/hf/inspect", json={"url": "vndub-no-such-owner-xyz/no-such-repo-123"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "hf_not_found"
    r = net.post("/api/models/hf/inspect", json={"url": "https://huggingface.co/ggml-org/tiny-llamas/tree/no-such-branch"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "hf_not_found"
    b = net.post("/api/models/hf/inspect", json={"url": "unsloth/gemma-4-12b-it-GGUF"}).json()
    assert b["kind"] == "builtin" and b["builtin"] == "gemma4_12b"


def test_gated_repo_is_flagged_and_its_download_explains_the_licence(net: Any) -> None:
    if os.environ.get("HF_TOKEN"):
        pytest.skip("HF_TOKEN is set: this machine may have access to the gated repo")
    b = net.post("/api/models/hf/inspect", json={"url": "meta-llama/Llama-3.2-1B"}).json()
    assert b["gated"] is True and any(w["code"] == "gated" for w in b["warnings"])
    job = net.post("/api/models/hf/add", json={"repo": "meta-llama/Llama-3.2-1B", "files": ["config.json"], "kind": "files"}).json()
    done = wait(job["id"], 60)
    if done["state"] == "done":
        pytest.skip("a cached Hugging Face login on this machine has access to the gated repo")
    assert done["state"] == "failed" and done["error"]["code"] == "hf_gated"
    assert done["error"]["params"]["page_url"] == "https://huggingface.co/meta-llama/Llama-3.2-1B"


def test_recommended_links_exist_and_list_the_file_the_app_would_pick(net: Any) -> None:
    items = net.get("/api/models/hf/recommended").json()["items"]
    assert len(items) >= 2
    for it in items:
        b = net.post("/api/models/hf/inspect", json={"url": it["url"]}).json()
        assert b["repo"] == it["repo"] and b["kind"] == "gguf_llm" and b["gated"] is False, it["repo"]
        assert b["suggested"] == [it["file"]], (it["repo"], b["suggested"])
        assert abs(b["suggested_size"] / 1e9 - it["size_gb"]) < 0.05 * it["size_gb"], (it["repo"], b["suggested_size"])
        assert (b["license"] or "").lower() == it["license"].lower(), (it["repo"], b["license"])
        print(f"\n[network] recommended {it['repo']}: {it['file']} {b['suggested_size']/1e9:.2f} GB, license {b['license']}")

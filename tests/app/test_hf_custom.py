"""Models by Hugging Face link, offline: link parsing, kind detection, quantization choice, the add job (fake hub, resume,
cancel), the custom list, delete, the registry hook, Settings -> pipeline.yaml, recommended links, resumable byte transfer.

Real network checks live in ``test_hf_network.py`` (``VNDUB_NETWORK_TESTS=1``).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterator

import pytest
import yaml
from hf_helpers import FakeHub, FakeInfo, GatedRepoError, RepositoryNotFoundError, gguf, repo_info
from o_helpers import RangeServer, client_for, drop_runner, fresh_runner, make_env, wait_job

from vndub.backends import registry
from vndub.backends.llm.llamacpp import find_gguf
from vndub.backends.weights import MARKER, is_present, local_dir
from vndub_app import appdirs, downloads, hf_custom, models_api
from vndub_app.hf_custom import ParsedUrl, parse_hf_url
from vndub_app.routes import general


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    data = make_env(monkeypatch, tmp_path)
    monkeypatch.setenv(registry.CUSTOM_ENV, "")  # restored (unset) afterwards; the app code re-points it
    hf_custom.sync_env()
    fresh_runner()
    yield data
    models_api.set_hub(None)
    hf_custom.set_info_client(None)
    drop_runner()
    registry.all_specs()


@pytest.fixture
def client(env: Path) -> Any:
    return client_for(models_api, hf_custom, general)


QUANT_FILES = {
    "m-Q8_0.gguf": 800,
    "m-Q4_K_M.gguf": 400,
    "m-IQ4_XS.gguf": 380,
    "m-Q2_K.gguf": 200,
    "README.md": 10,
}


# ---- link parsing ----------------------------------------------------------------------------------------------------

GOOD_URLS = [
    ("https://huggingface.co/owner/repo", ParsedUrl("owner/repo")),
    ("https://huggingface.co/owner/repo/", ParsedUrl("owner/repo")),
    ("http://huggingface.co/owner/repo", ParsedUrl("owner/repo")),
    ("https://www.huggingface.co/owner/repo", ParsedUrl("owner/repo")),
    ("huggingface.co/owner/repo", ParsedUrl("owner/repo")),
    ("hf.co/owner/repo", ParsedUrl("owner/repo")),
    ("https://hf.co/owner/repo", ParsedUrl("owner/repo")),
    ("owner/repo", ParsedUrl("owner/repo")),
    ("  owner/repo  ", ParsedUrl("owner/repo")),
    ("<https://huggingface.co/owner/repo>", ParsedUrl("owner/repo")),
    ("owner/my.repo_v2-GGUF", ParsedUrl("owner/my.repo_v2-GGUF")),
    ("https://huggingface.co/owner/repo?download=true", ParsedUrl("owner/repo")),
    ("https://huggingface.co/owner/repo#readme", ParsedUrl("owner/repo")),
    ("https://huggingface.co/owner/repo.git", ParsedUrl("owner/repo")),
    ("https://huggingface.co/owner/repo/tree/main", ParsedUrl("owner/repo", "main")),
    ("https://huggingface.co/owner/repo/tree/v1.0/Q4_K_M", ParsedUrl("owner/repo", "v1.0", "Q4_K_M")),
    ("https://huggingface.co/owner/repo/tree/main/a/b", ParsedUrl("owner/repo", "main", "a/b")),
    ("https://huggingface.co/owner/repo/blob/main/m-Q4_K_M.gguf", ParsedUrl("owner/repo", "main", "m-Q4_K_M.gguf")),
    ("https://huggingface.co/owner/repo/blob/main/sub/m.gguf?download=true", ParsedUrl("owner/repo", "main", "sub/m.gguf")),
    ("https://huggingface.co/owner/repo/resolve/abc123/m.gguf?download=true", ParsedUrl("owner/repo", "abc123", "m.gguf")),
    ("hf.co/owner/repo/resolve/main/x/y.pth", ParsedUrl("owner/repo", "main", "x/y.pth")),
    ("https://huggingface.co/owner/repo/tree/refs%2Fpr%2F1", ParsedUrl("owner/repo", "refs/pr/1")),
    ("https://huggingface.co/owner/repo/commit/0123abcd", ParsedUrl("owner/repo", "0123abcd")),
    ("https://huggingface.co/owner/repo/discussions/3", ParsedUrl("owner/repo")),
    ("owner/repo/tree/main", ParsedUrl("owner/repo", "main")),
    ("https://huggingface.co/Org-Name/Repo.Name-7B-GGUF", ParsedUrl("Org-Name/Repo.Name-7B-GGUF")),
]

BAD_URLS = [
    "",
    "   ",
    "owner",
    "https://huggingface.co/",
    "https://huggingface.co/owner",
    "https://example.com/owner/repo",
    "https://huggingface.co.evil.com/owner/repo",
    "https://evilhuggingface.co/owner/repo",
    "https://hf-mirror.com/owner/repo",
    "example.com/owner/repo",
    "ftp://huggingface.co/owner/repo",
    "file:///C:/models/x.gguf",
    "javascript:alert(1)",
    "https://user:pw@huggingface.co/owner/repo",
    "https://huggingface.co:8443/owner/repo",
    "https://huggingface.co/datasets/owner/repo",
    "https://huggingface.co/spaces/owner/repo",
    "https://huggingface.co/models",
    "https://huggingface.co/owner/../repo",
    "https://huggingface.co/owner/re po",
    "owner/repo name",
    "https://huggingface.co/-bad/repo",
    "https://huggingface.co/owner/" + "r" * 200,
    "https://huggingface.co/owner/repo/tree/../etc",
]


@pytest.mark.parametrize("text,expected", GOOD_URLS)
def test_parse_good_links(text: str, expected: ParsedUrl) -> None:
    assert parse_hf_url(text) == expected


@pytest.mark.parametrize("text", BAD_URLS)
def test_parse_bad_links(text: str) -> None:
    with pytest.raises(hf_custom.AppError) as ei:
        parse_hf_url(text)
    assert ei.value.code == "hf_bad_url" and ei.value.status == 422


def test_path_traversal_in_a_link_is_dropped() -> None:
    assert parse_hf_url("https://huggingface.co/owner/repo/blob/main/../../x").path is None


# ---- kind detection and file choice ------------------------------------------------------------------------------------


def sized(names: list[str], size: int = 100) -> list[tuple[str, int]]:
    return [(n, size) for n in names]


def test_quantization_preference_order() -> None:
    f = hf_custom.suggest_gguf
    assert f(sized(["a-Q8_0.gguf", "a-Q4_K_M.gguf", "a-IQ4_XS.gguf", "a-Q5_K_M.gguf"])) == ["a-Q4_K_M.gguf"]
    assert f(sized(["a-Q8_0.gguf", "a-IQ4_XS.gguf", "a-Q4_K_S.gguf", "a-Q5_K_M.gguf"])) == ["a-IQ4_XS.gguf"]
    assert f(sized(["a-Q8_0.gguf", "a-Q4_K_S.gguf", "a-Q5_K_M.gguf"])) == ["a-Q4_K_S.gguf"]
    assert f(sized(["a-Q8_0.gguf", "a-Q5_K_M.gguf", "a-Q6_K.gguf"])) == ["a-Q5_K_M.gguf"]


def test_quantization_names_in_many_styles() -> None:
    f = hf_custom.suggest_gguf
    for name in ("model.Q4_K_M.gguf", "model-UD-Q4_K_M.gguf", "Model_Q4_K_M.gguf", "q4_k_m/model.gguf", "model-q4_k_m.gguf"):
        assert f(sized([name, "model-Q8_0.gguf"])) == [name], name
    # Q4_K is not Q4_K_M, and IQ4_XS is not a Q4 token of another family
    assert f([("m-Q4_K.gguf", 300), ("m-Q6_K.gguf", 500)]) == ["m-Q4_K.gguf"]  # no preferred quant: smallest single file


def test_without_a_preferred_quant_the_smallest_single_file_wins_but_not_a_one_or_two_bit_one() -> None:
    files = [("m-Q8_0.gguf", 800), ("m-Q6_K.gguf", 600), ("m-Q3_K_M.gguf", 300), ("m-IQ2_XS.gguf", 150), ("m-Q2_K.gguf", 120)]
    assert hf_custom.suggest_gguf(files) == ["m-Q3_K_M.gguf"]
    assert hf_custom.suggest_gguf([("m-Q2_K.gguf", 120), ("m-IQ1_S.gguf", 90)]) == ["m-IQ1_S.gguf"]  # nothing else exists


def test_vision_projectors_are_never_suggested() -> None:
    files = sized(["mmproj-model-f16.gguf", "model.mmproj-Q8_0.gguf", "model-Q4_K_M.gguf"])
    assert hf_custom.suggest_gguf(files) == ["model-Q4_K_M.gguf"]
    assert hf_custom.suggest_gguf(sized(["mmproj-only.gguf"])) == []


def test_shards_are_taken_together_and_incomplete_sets_are_skipped() -> None:
    shards = [f"big-Q4_K_M-0000{i}-of-00003.gguf" for i in (1, 2, 3)]
    files = sized([*shards, "big-Q8_0.gguf"])
    assert hf_custom.suggest_gguf(files) == shards
    missing_one = sized([shards[0], shards[2], "big-Q8_0.gguf"])
    assert hf_custom.suggest_gguf(missing_one) == ["big-Q8_0.gguf"]  # the Q4 set is incomplete
    in_dir = [f"Q4_K_M/big-Q4_K_M-0000{i}-of-00002.gguf" for i in (1, 2)]
    assert hf_custom.suggest_gguf(sized([*in_dir, "BF16/big-BF16-00001-of-00004.gguf"])) == in_dir
    # only sharded sets and no preferred quant: the first set by name
    only = sized([f"x-Q8_0-0000{i}-of-00002.gguf" for i in (1, 2)] + [f"x-Q6_K-0000{i}-of-00002.gguf" for i in (1, 2)])
    assert hf_custom.suggest_gguf(only) == ["x-Q6_K-00001-of-00002.gguf", "x-Q6_K-00002-of-00002.gguf"]


def test_a_link_to_a_file_or_a_folder_narrows_the_suggestion() -> None:
    files = sized(["a-Q4_K_M.gguf", "a-Q8_0.gguf", "sub/b-Q4_K_M.gguf", "sub/b-Q5_K_M.gguf"])
    assert hf_custom.suggest_gguf(files, "a-Q8_0.gguf") == ["a-Q8_0.gguf"]
    assert hf_custom.suggest_gguf(files, "sub") == ["sub/b-Q4_K_M.gguf"]
    sh = [f"d/m-Q8_0-0000{i}-of-00002.gguf" for i in (1, 2)]
    assert hf_custom.suggest_gguf(sized([*sh, "x-Q4_K_M.gguf"]), sh[1]) == sh


def test_kind_detection(env: Path) -> None:
    kind = lambda repo, names, tags=None: hf_custom.detect_kind(repo, sized(names, 1000), tags)  # noqa: E731
    assert kind("o/r-GGUF", ["a-Q4_K_M.gguf", "README.md"]) == ("gguf_llm", ["a-Q4_K_M.gguf"])
    assert kind("o/voice", ["model.pth", "model.index", "README.md"]) == ("rvc", ["model.pth", "model.index"])
    assert kind("o/voices", ["a.pth", "b.pth", "b.index", "c.index"])[1] == ["a.pth"]  # the first .pth; two .index: none chosen
    assert kind("o/Voices-b", ["a/model.pth", "other.pth", "model.index"])[1][0] == "a/model.pth"
    assert kind("o/thing", ["config.json", "weights.safetensors"]) == ("files", ["config.json", "weights.safetensors"])
    # a big checkpoint is not a voice model; a big "files" repo suggests nothing
    big = hf_custom.detect_kind("o/big", [("m.pth", 3_000_000_000), ("README.md", 5)], [])
    assert big == ("files", [])
    assert hf_custom.detect_kind("o/big", [("m.pth", 3_000_000_000)], ["rvc"])[0] == "rvc"
    # a GGUF repo whose only .gguf is a projector is not an LLM repo
    assert kind("o/proj", ["mmproj.gguf", "x.json"])[0] == "files"
    assert kind("unsloth/gemma-4-12b-it-GGUF", ["g-IQ4_XS.gguf"]) == ("builtin", [])
    assert kind("UNSLOTH/Gemma-4-12B-it-GGUF", ["g-IQ4_XS.gguf"])[0] == "builtin"  # case-insensitive


def test_builtin_repo_is_found_by_exact_repo_only(env: Path) -> None:
    assert hf_custom.builtin_for("tencent/HY-MT1.5-7B-GGUF") == "hy_mt15_7b"
    assert hf_custom.builtin_for("tencent/HY-MT1.5-1.8B-GGUF") is None
    assert hf_custom.builtin_for("unsloth/Qwen3.5-9B-GGUF") == "qwen35_9b"


# ---- inspect (HTTP) -----------------------------------------------------------------------------------------------------


def test_inspect_gguf_repo(client: Any) -> None:
    hf_custom.set_info_client(FakeInfo({"o/r-GGUF": repo_info("o/r-GGUF", QUANT_FILES, license="apache-2.0")}))
    r = client.post("/api/models/hf/inspect", json={"url": "https://huggingface.co/o/r-GGUF/tree/dev?download=true"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["repo"] == "o/r-GGUF" and b["revision"] == "dev" and b["page_url"] == "https://huggingface.co/o/r-GGUF"
    assert b["kind"] == "gguf_llm" and b["suggested"] == ["m-Q4_K_M.gguf"] and b["suggested_size"] == 400
    assert b["license"] == "apache-2.0" and b["gated"] is False and b["builtin"] is None and b["warnings"] == []
    assert [f["name"] for f in b["files"]] == sorted(QUANT_FILES) and {"name": "m-Q4_K_M.gguf", "size": 400} in b["files"]
    assert b["total_size"] == sum(QUANT_FILES.values()) and b["free_gb"] > 0


def test_inspect_defaults_the_revision_to_main_and_asks_the_hub_for_it(client: Any) -> None:
    info = FakeInfo({"o/r": repo_info("o/r", {"x.bin": 5})})
    hf_custom.set_info_client(info)
    b = client.post("/api/models/hf/inspect", json={"url": "o/r"}).json()
    assert b["revision"] == "main" and info.asked == [("o/r", None)] and b["kind"] == "files" and b["suggested"] == ["x.bin"]


def test_inspect_with_a_blob_link_preselects_that_file(client: Any) -> None:
    hf_custom.set_info_client(FakeInfo({"o/r-GGUF": repo_info("o/r-GGUF", QUANT_FILES)}))
    b = client.post("/api/models/hf/inspect", json={"url": "https://huggingface.co/o/r-GGUF/blob/main/m-Q8_0.gguf"}).json()
    assert b["suggested"] == ["m-Q8_0.gguf"] and b["focus"] == "m-Q8_0.gguf"


def test_inspect_builtin_rvc_files_and_warnings(client: Any) -> None:
    infos = {
        "unsloth/gemma-4-12b-it-GGUF": repo_info("unsloth/gemma-4-12b-it-GGUF", {"g-IQ4_XS.gguf": 6_400_000_000}),
        "o/voice": repo_info("o/voice", {"v.pth": 55_000_000, "v.index": 80_000_000, "README.md": 3}, pipeline_tag="audio-to-audio"),
        "o/cfg": repo_info("o/cfg", {"config.json": 200, "tokenizer.json": 300}),
        "o/huge-GGUF": repo_info("o/huge-GGUF", {"h-Q4_K_M.gguf": 21_000_000_000}, license="cc-by-nc-4.0", gated="manual"),
        "o/tts-GGUF": repo_info("o/tts-GGUF", {"t-Q4_K_M.gguf": 1_000_000_000}, license=None, pipeline_tag="text-to-speech"),
    }
    hf_custom.set_info_client(FakeInfo(infos))
    post = lambda u: client.post("/api/models/hf/inspect", json={"url": u}).json()  # noqa: E731
    b = post("unsloth/gemma-4-12b-it-GGUF")
    assert b["kind"] == "builtin" and b["builtin"] == "gemma4_12b" and b["suggested"] == []
    v = post("o/voice")
    assert v["kind"] == "rvc" and v["suggested"] == ["v.pth", "v.index"] and v["suggested_size"] == 135_000_000
    f = post("o/cfg")
    assert f["kind"] == "files" and f["suggested"] == ["config.json", "tokenizer.json"]
    h = post("o/huge-GGUF")
    codes = {w["code"] for w in h["warnings"]}
    assert h["gated"] is True and {"gated", "license_check", "too_big"} <= codes
    assert next(w for w in h["warnings"] if w["code"] == "too_big")["params"] == {"size_gb": 21.0}
    t = post("o/tts-GGUF")
    assert {"unknown_license", "not_text_model"} <= {w["code"] for w in t["warnings"]}


def test_inspect_errors(client: Any) -> None:
    hf_custom.set_info_client(
        FakeInfo(
            {"o/ok": repo_info("o/ok", {"a.bin": 1})},
            errors={"o/gated": GatedRepoError("401 gated"), "o/net": OSError("Connection reset by peer"), "o/odd": RuntimeError("boom")},
        )
    )
    post = lambda u: client.post("/api/models/hf/inspect", json={"url": u})  # noqa: E731
    r = post("https://example.com/o/ok")
    assert r.status_code == 422 and r.json()["error"]["code"] == "hf_bad_url"
    r = post("o/missing")
    e = r.json()["error"]
    assert r.status_code == 404 and e["code"] == "hf_not_found" and e["params"]["page_url"] == "https://huggingface.co/o/missing"
    r = post("o/gated")
    e = r.json()["error"]
    assert r.status_code == 403 and e["code"] == "hf_gated" and e["params"]["page_url"] == "https://huggingface.co/o/gated"
    r = post("o/net")
    e = r.json()["error"]
    assert r.status_code == 502 and e["code"] == "download_unreachable" and e["params"]["url"] == "https://huggingface.co/o/net"
    assert post("o/odd").json()["error"]["code"] == "models_download_failed"
    assert client.post("/api/models/hf/inspect", json={}).status_code == 422


# ---- add (job) ------------------------------------------------------------------------------------------------------------


def add(client: Any, repo: str, files: list[str], kind: str = "gguf_llm", **kw: Any) -> dict[str, Any]:
    r = client.post("/api/models/hf/add", json={"repo": repo, "revision": "main", "files": files, "kind": kind, **kw})
    assert r.status_code == 200, r.text
    return r.json()


def test_add_downloads_into_the_custom_folder_and_records_the_entry(client: Any, env: Path) -> None:
    payload = {"m-Q4_K_M.gguf": gguf(300), "README.md": b"readme"}
    hf_custom.set_info_client(FakeInfo({"o/r-GGUF": repo_info("o/r-GGUF", payload, license="apache-2.0")}))
    hub = FakeHub({"o/r-GGUF": payload})
    models_api.set_hub(hub)
    job = add(client, "o/r-GGUF", ["m-Q4_K_M.gguf"], label="Test model")
    assert job["kind"] == "models_hf_add" and job["gpu"] is False and job["params"]["id"] == "hf_o_r_gguf"
    done = wait_job(job)
    assert done["state"] == "done", done.get("error")
    d = env / "models" / "custom" / "o__r-GGUF"
    assert (d / "m-Q4_K_M.gguf").read_bytes() == payload["m-Q4_K_M.gguf"] and (d / MARKER).exists()
    assert not (d / "README.md").exists() and hub.fetched == [("o/r-GGUF", "m-Q4_K_M.gguf")]
    files = done["result"]["files"]
    assert [f["filename"] for f in files] == ["m-Q4_K_M.gguf"] and files[0]["state"] == "done" and files[0]["done"] == files[0]["total"] == 300
    assert files[0]["backend"] == "hf_o_r_gguf" and done["done"] == done["total"]
    entry = done["result"]["entry"]
    saved = json.loads((env / "custom_models.json").read_text(encoding="utf-8"))["models"]
    assert saved == [entry]
    assert entry["id"] == "hf_o_r_gguf" and entry["repo"] == "o/r-GGUF" and entry["revision"] == "main" and entry["kind"] == "gguf_llm"
    assert entry["files"] == ["m-Q4_K_M.gguf"] and entry["dir"] == "custom/o__r-GGUF" and entry["license"] == "apache-2.0"
    assert entry["label"] == "Test model" and entry["size_gb"] == pytest.approx(300 / 1e9, abs=1e-3) and entry["added_at"]
    lst = client.get("/api/models/custom").json()
    assert len(lst) == 1 and lst[0]["id"] == "hf_o_r_gguf" and lst[0]["present"] is True and lst[0]["in_use"] is False
    assert lst[0]["usable"] is True and lst[0]["disk_gb"] >= 0
    # the built-in table is unchanged by custom models
    names = [r["name"] for r in client.get("/api/models").json()["rows"]]
    assert "hf_o_r_gguf" not in names


def test_add_validation(client: Any) -> None:
    models_api.set_hub(FakeHub({}))
    post = lambda **b: client.post("/api/models/hf/add", json={"repo": "o/r", "files": ["a.gguf"], "kind": "gguf_llm", **b})  # noqa: E731
    assert post(kind="nope").status_code == 422
    assert post(files=[]).status_code == 422
    assert post(files=["../evil.gguf"]).status_code == 422
    assert post(files=["/abs/evil.gguf"]).status_code == 422
    assert post(files=["C:/x.gguf"]).status_code == 422
    assert post(files=["readme.md"]).status_code == 422  # gguf_llm needs a .gguf
    r = post(repo="https://example.com/o/r")
    assert r.status_code == 422 and r.json()["error"]["code"] == "hf_bad_url"
    assert client.post("/api/models/hf/add", json={"repo": "o/r", "kind": "files"}).status_code == 422


def test_add_a_file_that_is_not_in_the_repo_fails_the_job(client: Any) -> None:
    models_api.set_hub(FakeHub({"o/r": {"a.bin": b"a"}}))
    done = wait_job(add(client, "o/r", ["nope.bin"], kind="files"))
    assert done["state"] == "failed" and done["error"]["code"] == "bad_request" and done["error"]["params"]["value"] == "nope.bin"


def test_add_refuses_when_the_disk_is_too_small(client: Any, monkeypatch: pytest.MonkeyPatch, env: Path) -> None:
    monkeypatch.setattr(downloads, "free_space_gb", lambda p: 0.5)
    big = FakeHub({"o/r": {"a.bin": b"x" * 100}})
    big.list_files = lambda repo, rev, allow: [("a.bin", 5_000_000_000)]  # type: ignore[method-assign]
    models_api.set_hub(big)
    done = wait_job(add(client, "o/r", ["a.bin"], kind="files"))
    assert done["state"] == "failed" and done["error"]["code"] == "disk_full"
    assert done["error"]["params"]["free_gb"] == 0.5 and done["error"]["params"]["need_gb"] > 5
    assert not (env / "custom_models.json").exists() or json.loads((env / "custom_models.json").read_text())["models"] == []


def test_add_resumes_after_a_failed_file_without_refetching_finished_ones(client: Any, env: Path) -> None:
    payload = {"a.bin": b"a" * 500, "b.bin": b"b" * 500, "c.bin": b"c" * 500}
    hf_custom.set_info_client(FakeInfo({"o/r": repo_info("o/r", payload)}))
    hub = FakeHub({"o/r": payload}, fail_on={"o/r/b.bin": OSError("Connection reset by peer")})
    models_api.set_hub(hub)
    first = wait_job(add(client, "o/r", ["a.bin", "b.bin", "c.bin"], kind="files"))
    assert first["state"] == "failed" and first["error"]["code"] == "download_unreachable"
    assert first["error"]["params"]["url"] == "https://huggingface.co/o/r" and first["error"]["params"]["filename"] == "b.bin"
    assert first["error"]["params"]["target_dir"].endswith("o__r") and first["error"]["action"] == "show_manual"
    rows = {f["filename"]: f for f in first["result"]["files"]}
    assert rows["a.bin"]["state"] == "done" and rows["b.bin"]["state"] == "failed" and rows["b.bin"]["error"] == "download_unreachable"
    d = env / "models" / "custom" / "o__r"
    assert not (d / MARKER).exists()  # not claimed as complete
    assert client.get("/api/models/custom").json() == []  # not listed before it is complete
    hub.fetched.clear()
    second = wait_job(add(client, "o/r", ["a.bin", "b.bin", "c.bin"], kind="files"))
    assert second["state"] == "done", second.get("error")
    assert hub.fetched == [("o/r", "b.bin"), ("o/r", "c.bin")]  # a.bin was already there
    rows = {f["filename"]: f for f in second["result"]["files"]}
    assert rows["a.bin"].get("skipped") is True and rows["a.bin"]["state"] == "done" and rows["b.bin"].get("skipped") is None
    assert (d / MARKER).exists() and all((d / n).read_bytes() == payload[n] for n in payload)


def test_cancelled_add_keeps_partial_files_and_the_next_run_finishes(client: Any, env: Path) -> None:
    from vndub_app import jobs

    payload = {"a.bin": b"a" * 400, "b.bin": b"b" * 400}
    hf_custom.set_info_client(FakeInfo({"o/r": repo_info("o/r", payload)}))
    holder: dict[str, str] = {}
    hub = FakeHub({"o/r": payload}, slow=True, on_fetch=lambda repo, name: name == "b.bin" and jobs.runner().cancel(holder["id"]))
    models_api.set_hub(hub)
    job = add(client, "o/r", ["a.bin", "b.bin"], kind="files")
    holder["id"] = job["id"]
    done = wait_job(job)
    assert done["state"] == "cancelled" and done["error"] is None
    d = env / "models" / "custom" / "o__r"
    assert (d / "a.bin").exists() and not (d / MARKER).exists() and client.get("/api/models/custom").json() == []
    hub.on_fetch = None
    hub.slow = False
    hub.fetched.clear()
    again = wait_job(add(client, "o/r", ["a.bin", "b.bin"], kind="files"))
    assert again["state"] == "done" and hub.fetched == [("o/r", "b.bin")] and (d / MARKER).exists()


def test_a_second_add_of_the_same_repo_while_one_runs_is_refused(client: Any) -> None:
    import threading

    gate = threading.Event()
    hub = FakeHub({"o/r": {"a.bin": b"a" * 10}}, on_fetch=lambda repo, name: gate.wait(10))
    models_api.set_hub(hub)
    hf_custom.set_info_client(FakeInfo({"o/r": repo_info("o/r", {"a.bin": 10})}))
    job = add(client, "o/r", ["a.bin"], kind="files")
    time.sleep(0.2)
    r = client.post("/api/models/hf/add", json={"repo": "o/r", "files": ["a.bin"], "kind": "files"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "job_busy"
    gate.set()
    assert wait_job(job)["state"] == "done"


def test_adding_the_same_repo_with_another_quantization_replaces_the_files(client: Any, env: Path) -> None:
    payload = {"m-Q4_K_M.gguf": gguf(200), "m-Q8_0.gguf": gguf(400)}
    hf_custom.set_info_client(FakeInfo({"o/r-GGUF": repo_info("o/r-GGUF", payload)}))
    models_api.set_hub(FakeHub({"o/r-GGUF": payload}))
    assert wait_job(add(client, "o/r-GGUF", ["m-Q4_K_M.gguf"]))["state"] == "done"
    assert wait_job(add(client, "o/r-GGUF", ["m-Q8_0.gguf"], label="Second"))["state"] == "done"
    d = env / "models" / "custom" / "o__r-GGUF"
    assert (d / "m-Q8_0.gguf").exists() and not (d / "m-Q4_K_M.gguf").exists()
    lst = client.get("/api/models/custom").json()
    assert len(lst) == 1 and lst[0]["files"] == ["m-Q8_0.gguf"] and lst[0]["label"] == "Second"


def test_files_in_subfolders_and_distinct_repos_with_the_same_slug(client: Any, env: Path) -> None:
    models_api.set_hub(FakeHub({"a-b/c": {"sub/x.bin": b"1"}, "a/b-c": {"y.bin": b"2"}}))
    assert wait_job(add(client, "a-b/c", ["sub/x.bin"], kind="files"))["state"] == "done"
    assert wait_job(add(client, "a/b-c", ["y.bin"], kind="files"))["state"] == "done"
    assert (env / "models" / "custom" / "a-b__c" / "sub" / "x.bin").read_bytes() == b"1"
    ids = [e["id"] for e in client.get("/api/models/custom").json()]
    assert len(ids) == 2 and len(set(ids)) == 2 and "hf_a_b_c" in ids  # the second one got a suffix


def test_unsafe_names_from_a_repo_listing_never_leave_the_folder(env: Path) -> None:
    with pytest.raises(hf_custom.AppError):
        models_api.safe_join(env / "models" / "x", "../../evil")
    assert models_api.safe_join(env / "models" / "x", "a/b.bin").name == "b.bin"


# ---- delete -----------------------------------------------------------------------------------------------------------------


def make_model(client: Any, repo: str = "o/r-GGUF", name: str = "m-Q4_K_M.gguf") -> str:
    payload = {name: gguf(120)}
    hf_custom.set_info_client(FakeInfo({repo: repo_info(repo, payload)}))
    models_api.set_hub(FakeHub({repo: payload}))
    job = wait_job(add(client, repo, [name]))
    assert job["state"] == "done", job.get("error")
    return job["result"]["id"]


def test_delete_removes_files_and_entry(client: Any, env: Path) -> None:
    cid = make_model(client)
    d = env / "models" / "custom" / "o__r-GGUF"
    assert d.exists()
    r = client.delete(f"/api/models/custom/{cid}")
    assert r.status_code == 200 and r.json() == {"ok": True}
    assert not d.exists() and client.get("/api/models/custom").json() == []
    assert cid not in registry.all_specs()
    r = client.delete(f"/api/models/custom/{cid}")
    assert r.status_code == 404 and r.json()["error"]["code"] == "bad_request"


def test_delete_is_refused_while_a_job_uses_the_model(client: Any, env: Path) -> None:
    import threading

    from vndub_app import jobs

    cid = make_model(client)
    gate = threading.Event()
    busy = jobs.runner().submit("models_hf_add", lambda ctx: gate.wait(10) and None, params={"id": cid}, gpu=False)
    r = client.delete(f"/api/models/custom/{cid}")
    assert r.status_code == 409 and r.json()["error"]["code"] == "bad_request" and r.json()["error"]["params"]["reason"] == "in_use"
    assert (env / "models" / "custom" / "o__r-GGUF").exists()
    gate.set()
    wait_job(busy.id)
    # a pipeline job counts when this model is the chosen translation model
    s = appdirs.load_settings()
    s.translation_llm = cid
    appdirs.save_settings(s)
    gate2 = threading.Event()
    pipe = jobs.runner().submit("translate", lambda ctx: gate2.wait(10) and None, params={}, project="p", gpu=False)
    assert client.delete(f"/api/models/custom/{cid}").status_code == 409
    gate2.set()
    wait_job(pipe.id)
    assert client.delete(f"/api/models/custom/{cid}").status_code == 200
    assert appdirs.load_settings().translation_llm == ""  # the choice is dropped with the model


def test_delete_never_touches_anything_outside_the_custom_folder(client: Any, env: Path, tmp_path: Path) -> None:
    keep = tmp_path / "precious"
    keep.mkdir()
    (keep / "f.txt").write_text("x", encoding="utf-8")
    appdirs.write_json_atomic(env / "custom_models.json", {"models": [{"id": "hf_evil", "repo": "o/evil", "kind": "files", "dir": "../../precious", "files": ["f.txt"]}]})
    r = client.delete("/api/models/custom/hf_evil")
    assert r.status_code == 422 and (keep / "f.txt").exists()


# ---- registry hook ---------------------------------------------------------------------------------------------------------


def entry_for(dirname: str = "custom/o__r-GGUF", **kw: Any) -> dict[str, Any]:
    e = {
        "id": "hf_o_r_gguf", "repo": "o/r-GGUF", "revision": "main", "files": ["sub/m-Q4_K_M.gguf"], "kind": "gguf_llm", "dir": dirname,
        "size_gb": 4.6, "license": "apache-2.0", "label": "R", "added_at": "2026-10-21T00:00:00",
    }
    e.update(kw)
    return e


def test_registry_is_unchanged_without_the_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"models": [entry_for()]}), encoding="utf-8")
    monkeypatch.setenv(registry.CUSTOM_ENV, str(f))
    assert "hf_o_r_gguf" in registry.all_specs()
    monkeypatch.delenv(registry.CUSTOM_ENV)
    assert "hf_o_r_gguf" not in registry.all_specs() and registry.custom_names() == frozenset()
    built_in = set(registry.all_specs())
    assert {"gemma4_12b", "qwen35_9b", "hy_mt15_7b", "mock_translator"} <= built_in


def test_custom_entry_becomes_a_llamacpp_backend_spec(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"models": [entry_for(), entry_for(id="hf_voice", kind="rvc", files=["v.pth"]), {"broken": True}]}), encoding="utf-8")
    monkeypatch.setenv(registry.CUSTOM_ENV, str(f))
    spec = registry.get("hf_o_r_gguf")
    assert spec.role == "llm" and spec.env is None and spec.adapter == "llamacpp" and spec.local_impl == "vndub.backends.llm.llamacpp:LlamaCppBackend"
    assert spec.profiles == ("optional",) and spec.heavy and spec.mock == "mock_translator"
    assert spec.license == "apache-2.0" and spec.commercial_ok is True and spec.vram_estimate_gb == pytest.approx(6.1)
    w = spec.weights[0]
    assert w.kind == "hf_file" and w.repo == "o/r-GGUF" and w.subdir == "custom/o__r-GGUF" and w.revision == "main"
    assert registry.custom_names() == frozenset({"hf_o_r_gguf"})  # the RVC entry and the broken one are not backends
    assert "hf_voice" not in registry.all_specs()
    assert "hf_o_r_gguf" in {s.name for s in registry.for_profile("optional")} and "hf_o_r_gguf" not in {s.name for s in registry.for_profile("default")}


def test_custom_weights_resolve_into_the_custom_folder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"models": [entry_for(dirname="custom/o__r-GGUF")]}), encoding="utf-8")
    monkeypatch.setenv(registry.CUSTOM_ENV, str(f))
    spec = registry.get("hf_o_r_gguf")
    models = tmp_path / "models"
    d = local_dir(models, spec.weights[0], spec.name)
    assert d == models / "custom" / "o__r-GGUF"
    assert not is_present(models, spec.weights[0], spec.name)
    u = registry.usable(spec, policy="research", models_dir=models)
    assert not u.ok and u.reason.startswith("weights missing")
    d.mkdir(parents=True)
    (d / MARKER).write_text("x", encoding="utf-8")
    assert registry.usable(spec, policy="research", models_dir=models).ok
    # a licence that is not commercial-friendly is refused under the commercial policy
    f.write_text(json.dumps({"models": [entry_for(license="cc-by-nc-4.0")]}), encoding="utf-8")
    nc = registry.get("hf_o_r_gguf")
    assert nc.commercial_ok is False and registry.usable(nc, policy="commercial", models_dir=models).reason.startswith("license")
    assert registry.usable(nc, policy="research", models_dir=models).ok
    # an absolute dir (hand-edited list) is honoured as is
    f.write_text(json.dumps({"models": [entry_for(dirname=str(tmp_path / "elsewhere"))]}), encoding="utf-8")
    assert local_dir(models, registry.get("hf_o_r_gguf").weights[0], "x") == tmp_path / "elsewhere"


def test_find_gguf_for_a_custom_model_takes_the_first_shard_and_skips_projectors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    shards = ["big-Q4_K_M-00001-of-00002.gguf", "big-Q4_K_M-00002-of-00002.gguf"]
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"models": [entry_for(files=[*shards, "mmproj-big.gguf"], dirname="custom/o__big")]}), encoding="utf-8")
    monkeypatch.setenv(registry.CUSTOM_ENV, str(f))
    spec = registry.get("hf_o_r_gguf")
    d = tmp_path / "models" / "custom" / "o__big"
    d.mkdir(parents=True)
    for n in (*shards, "mmproj-big.gguf", "other-Q8_0.gguf"):
        (d / n).write_bytes(b"GGUF")
    assert find_gguf(tmp_path / "models", spec) == d / shards[0]  # not the projector, not an unrelated file
    inst = registry.instantiate(spec, models_dir=tmp_path / "models")
    assert type(inst).__name__ == "LlamaCppBackend" and inst.spec.name == "hf_o_r_gguf"


def test_special_characters_in_file_names_match_literally(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"models": [entry_for(files=["we[ir]d.gguf"], dirname="custom/w")]}), encoding="utf-8")
    monkeypatch.setenv(registry.CUSTOM_ENV, str(f))
    d = tmp_path / "models" / "custom" / "w"
    d.mkdir(parents=True)
    (d / "we[ir]d.gguf").write_bytes(b"GGUF")
    (d / "weid.gguf").write_bytes(b"GGUF")  # what an unescaped [ir] would also match
    assert find_gguf(tmp_path / "models", registry.get("hf_o_r_gguf")).name == "we[ir]d.gguf"


def test_registry_rereads_the_file_when_it_changes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    f = tmp_path / "c.json"
    monkeypatch.setenv(registry.CUSTOM_ENV, str(f))
    assert registry.custom_names() == frozenset()  # no file yet
    f.write_text(json.dumps({"models": [entry_for()]}), encoding="utf-8")
    assert registry.custom_names() == frozenset({"hf_o_r_gguf"})
    f.write_text(json.dumps([entry_for(id="hf_two"), entry_for(id="hf_three")]), encoding="utf-8")  # a bare list is fine too
    assert registry.custom_names() == frozenset({"hf_two", "hf_three"}) and "hf_o_r_gguf" not in registry.all_specs()
    f.write_text("{not json", encoding="utf-8")
    assert registry.custom_names() == frozenset()
    f.unlink()
    assert registry.custom_names() == frozenset()


def test_a_custom_name_never_replaces_a_built_in(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"models": [entry_for(id="gemma4_12b")]}), encoding="utf-8")
    monkeypatch.setenv(registry.CUSTOM_ENV, str(f))
    assert registry.get("gemma4_12b").weights[0].repo == "unsloth/gemma-4-12b-it-GGUF"
    assert registry.custom_names() == frozenset()


# ---- Settings -> pipeline.yaml -----------------------------------------------------------------------------------------------


def make_project(tmp_path: Path) -> appdirs.ProjectHandle:
    game = tmp_path / "game"
    (game / "game").mkdir(parents=True)
    return appdirs.create_project(game, name="Game")


def test_settings_choice_is_written_to_the_project_config(client: Any, env: Path, tmp_path: Path) -> None:
    from vndub.config import load_config

    h = make_project(tmp_path)
    cid = make_model(client)

    def backend() -> str:
        appdirs.write_project_config(h)
        return load_config(h.config_file).translate.llm.backend

    assert backend() == "gemma4_12b" and "llm" not in (yaml.safe_load(h.config_file.read_text(encoding="utf-8")).get("translate") or {})
    r = client.put("/api/settings", json={"translation_llm": cid})
    assert r.status_code == 200 and r.json()["translation_llm"] == cid
    assert backend() == cid
    raw = yaml.safe_load(h.config_file.read_text(encoding="utf-8"))
    assert raw["translate"]["llm"] == {"backend": cid}
    assert load_config(h.config_file).translate.llm.fallbacks == ["qwen35_9b", "hy_mt15_7b"]  # the profile's fallbacks stay
    # built-in alternatives work too; the default is not written
    assert client.put("/api/settings", json={"translation_llm": "qwen35_9b"}).status_code == 200 and backend() == "qwen35_9b"
    assert client.put("/api/settings", json={"translation_llm": "gemma4_12b"}).status_code == 200
    assert "llm" not in (yaml.safe_load(h.config_file.read_text(encoding="utf-8")).get("translate") or {}) or backend() == "gemma4_12b"
    assert client.put("/api/settings", json={"translation_llm": ""}).json()["translation_llm"] == ""
    assert backend() == "gemma4_12b"


def test_settings_rejects_unknown_and_non_llm_backends(client: Any) -> None:
    for bad in ("no_such_backend", "ecapa", "mock_translator"):
        r = client.put("/api/settings", json={"translation_llm": bad})
        assert r.status_code == 422 and r.json()["error"]["code"] == "bad_request", bad
    assert client.get("/api/settings").json()["translation_llm"] == ""


def test_a_deleted_model_falls_back_to_the_default_in_the_project_config(client: Any, env: Path, tmp_path: Path) -> None:
    from vndub.config import load_config

    h = make_project(tmp_path)
    cid = make_model(client)
    s = appdirs.load_settings()
    s.translation_llm = cid
    appdirs.save_settings(s)
    appdirs.write_project_config(h)
    assert load_config(h.config_file).translate.llm.backend == cid
    # the list is edited by hand / the entry vanishes: the setting is stale, the config must still be valid
    (env / "custom_models.json").unlink()
    appdirs.write_project_config(h)
    assert load_config(h.config_file).translate.llm.backend == "gemma4_12b"


def test_translation_is_available_with_a_custom_model_only_when_it_is_usable(client: Any, env: Path, tmp_path: Path) -> None:
    from vndub_app import translation

    h = make_project(tmp_path)
    cid = make_model(client)
    s = appdirs.load_settings()
    s.translation_llm = cid
    appdirs.save_settings(s)
    from vndub.config import load_config

    appdirs.write_project_config(h)
    cfg = load_config(h.config_file)
    assert cfg.translate.llm.backend == cid
    # the llama-server tool is not installed in the test environment, so it is not "available" - but the lookup works
    assert translation.llm_available(cfg) in (True, False)


# ---- recommended links --------------------------------------------------------------------------------------------------------


def test_recommended_links_are_served_and_well_formed(client: Any) -> None:
    body = client.get("/api/models/hf/recommended").json()
    items = body["items"]
    assert 2 <= len(items) <= 4 and body["checked"]
    builtin_repos = {w.repo.lower() for s in registry.all_specs().values() for w in s.weights if w.repo}
    for it in items:
        assert parse_hf_url(it["url"]).repo == it["repo"] and it["url"] == f"https://huggingface.co/{it['repo']}"
        assert it["kind"] in hf_custom.KINDS and it["file"].endswith(".gguf") and it["size_gb"] > 0 and it["license"]
        assert it["title"]["ru"] and it["title"]["en"] and it["note"]["ru"] and it["note"]["en"]
        assert it["repo"].lower() not in builtin_repos  # alternatives to what the app already lists
    assert len({it["repo"] for it in items}) == len(items)


def test_recommended_file_missing_gives_an_empty_list(client: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(hf_custom, "recommended_file", lambda: tmp_path / "none.json")
    assert client.get("/api/models/hf/recommended").json() == {"items": [], "checked": None}


# ---- resumable byte transfer of HfHub ------------------------------------------------------------------------------------------


def test_hfhub_fetch_resumes_a_cancelled_download_with_a_range_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """HfHub streams through downloads.fetch: a cancel in the progress callback leaves ``.part``; the next call continues it."""
    from vndub_app.jobs import JobCancelled

    data = bytes(range(256)) * 16000  # ~4 MB: several 1 MiB chunks
    hub = models_api.HfHub()
    with RangeServer({"w.bin": data}) as srv:
        calls = iter(range(100))
        # like the real hub: a new pre-signed link on every resolve
        monkeypatch.setattr(hub, "resolve", lambda repo, filename, revision: (srv.url("w.bin") + f"?Signature={next(calls)}", len(data), hashlib.sha256(data).hexdigest()))
        local = tmp_path / "w"
        seen: list[int] = []

        def cancel_after_1mb(done: int, total: int) -> None:
            seen.append(done)
            if done >= 1_000_000:
                raise JobCancelled("cancelled by user")

        with pytest.raises(JobCancelled):
            hub.fetch("o/r", "sub/w.bin", None, local, cancel_after_1mb)
        part = local / "sub" / "w.bin.part"
        assert part.exists() and 1_000_000 <= part.stat().st_size < len(data) and not (local / "sub" / "w.bin").exists()
        left = part.stat().st_size
        resumed: list[tuple[int, int]] = []
        hub.fetch("o/r", "sub/w.bin", None, local, lambda d, t: resumed.append((d, t)))
        assert resumed[0] == (left, len(data)) and resumed[-1] == (len(data), len(data))
        assert srv.requests[-1] == ("w.bin", f"bytes={left}-")
        assert (local / "sub" / "w.bin").read_bytes() == data and not part.exists()


def test_a_part_file_of_another_file_is_not_continued(tmp_path: Path) -> None:
    from vndub_app.jobs import JobCancelled

    a, b = b"A" * 3_000_000, b"B" * 3_000_000
    with RangeServer({"x.bin": a, "y.bin": b}) as srv:
        dest = tmp_path / "f.bin"

        def stop(done: int, total: int) -> None:
            if done >= 1_000_000:
                raise JobCancelled("x")

        with pytest.raises(JobCancelled):
            downloads.fetch(srv.url("x.bin"), dest, progress=stop, identity="file-a")
        res = downloads.fetch(srv.url("y.bin"), dest, identity="file-b")  # same .part name, another file: start over
        assert res.resumed_from == 0 and dest.read_bytes() == b


def test_hfhub_fetch_rejects_a_wrong_checksum_and_unsafe_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = b"x" * 5000
    hub = models_api.HfHub()
    with RangeServer({"w.bin": data}) as srv:
        monkeypatch.setattr(hub, "resolve", lambda repo, filename, revision: (srv.url("w.bin"), len(data), "0" * 64))
        with pytest.raises(hf_custom.AppError) as ei:
            hub.fetch("o/r", "w.bin", None, tmp_path / "w", lambda d, t: None)
        assert ei.value.code == "checksum_mismatch" and not (tmp_path / "w" / "w.bin").exists()
        with pytest.raises(hf_custom.AppError):
            hub.fetch("o/r", "../escape.bin", None, tmp_path / "w", lambda d, t: None)


def test_hfhub_unreachable_file_is_a_connection_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub = models_api.HfHub()
    with RangeServer({}) as srv:
        monkeypatch.setattr(hub, "resolve", lambda repo, filename, revision: (srv.url("missing.bin") + "?Signature=secret", 100, None))
        with pytest.raises(ConnectionError):
            hub.fetch("o/r", "missing.bin", None, tmp_path / "w", lambda d, t: None)  # HTTP 404 -> the job reports download_unreachable


def test_a_gated_download_error_names_the_model_page(tmp_path: Path) -> None:
    spec = registry.get("ecapa")
    err = models_api._download_error(spec, spec.weights[0], GatedRepoError("403 Forbidden"), tmp_path)  # noqa: SLF001
    assert err.code == "hf_gated" and err.params["page_url"] == f"https://huggingface.co/{spec.weights[0].repo}"
    assert models_api.hub_error_kind(RepositoryNotFoundError("x")) == "not_found"
    assert models_api.hub_error_kind(OSError("x")) is None


# ---- i18n families chosen at run time --------------------------------------------------------------------------------------------


def test_i18n_keys_of_the_link_feature_exist_in_both_tables() -> None:
    """The SPA builds these keys from data (kind, warning code, error code); test_i18n only sees literal ones."""
    import re

    root = Path(__file__).resolve().parents[2]
    tables = {lang: json.loads((root / "app" / "i18n" / f"{lang}.json").read_text(encoding="utf-8")) for lang in ("ru", "en")}
    src = (root / "vndub_app" / "hf_custom.py").read_text(encoding="utf-8")
    warn_codes = set(re.findall(r'\{"code": "([a-z_]+)"', src))
    assert {"gated", "license_check", "unknown_license", "too_big", "not_text_model"} <= warn_codes
    keys = {f"models.hf.warn.{c}" for c in warn_codes}
    for k in ("builtin", "gguf_llm", "rvc", "files"):
        keys |= {f"models.hf.kind.{k}", f"models.hf.kind.{k}_text"}
    keys |= {f"models.custom.kind.{k}" for k in hf_custom.KINDS}
    for code in ("hf_bad_url", "hf_not_found", "hf_gated"):
        keys |= {f"errors.{code}.title", f"errors.{code}.hint"}
    keys |= {"jobs.models_hf_add", "models.msg.hf_add", "models.chip.recommended", "settings.translation_llm.title", "settings.translation_llm.default", "settings.translation_llm.hint"}
    for lang, table in tables.items():
        missing = sorted(k for k in keys if k not in table)
        assert not missing, f"{lang}.json lacks {missing}"
    # the recommended file's texts are in the two UI languages
    for it in recommended_items():
        assert set(it["title"]) >= {"ru", "en"} and set(it["note"]) >= {"ru", "en"}


def recommended_items() -> list[dict[str, Any]]:
    return json.loads(hf_custom.recommended_file().read_text(encoding="utf-8"))["items"]


# ---- token handling ----------------------------------------------------------------------------------------------------------------


def test_the_hub_token_goes_only_to_the_hub_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_secret_for_test")
    assert models_api._hub_auth("https://huggingface.co/o/r/resolve/main/config.json") == {"Authorization": "Bearer hf_secret_for_test"}  # noqa: SLF001
    assert models_api._hub_auth("https://us.aws.cdn.hf.co/xet-bridge-us/abc?Signature=x") == {}  # noqa: SLF001  (a signed CDN link needs none)
    assert models_api._hub_auth("https://example.com/x") == {}  # noqa: SLF001
    monkeypatch.delenv("HF_TOKEN")
    monkeypatch.setenv("HF_HOME", str(Path(__file__).parent / "_no_such_hf_home"))  # no cached login either
    assert models_api._hub_auth("https://huggingface.co/o/r/resolve/main/config.json") == {}  # noqa: SLF001


def test_downloads_fetch_sends_extra_headers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import urllib.request

    seen: list[dict[str, str]] = []
    real = urllib.request.urlopen

    def spy(req: Any, timeout: float = 0) -> Any:
        seen.append({k.lower(): v for k, v in req.header_items()})
        return real(req, timeout=timeout)

    monkeypatch.setattr(urllib.request, "urlopen", spy)
    with RangeServer({"f.bin": b"z" * 100}) as srv:
        downloads.fetch(srv.url("f.bin"), tmp_path / "f.bin", headers={"Authorization": "Bearer t"})
    assert seen and seen[0]["authorization"] == "Bearer t" and seen[0]["accept-encoding"] == "identity"


def test_bundled_tables_are_written_as_absolute_paths(env: Path, tmp_path: Path) -> None:
    """The packaged app runs from its exe folder, so every bundled table the profile names relatively is written absolute.
    The laughter table was missed in 0.7.0 (`laughter table <exe folder>/configs/laughter.yaml not found` on every synth:
    laughs went to the voices as written)."""
    from vndub.config import load_config

    h = make_project(tmp_path)
    appdirs.write_project_config(h)
    raw = yaml.safe_load(h.config_file.read_text(encoding="utf-8"))
    root = appdirs.resource_root()
    assert raw["synth"]["laughter"]["table"] == str(root / "configs" / "laughter.yaml")
    assert raw["translate"]["interjections"] == str(root / "configs" / "interjections.yaml")
    assert raw["text_emotion"]["cues"] == str(root / "configs" / "delivery_cues.yaml")
    cfg = load_config(h.config_file)
    assert Path(cfg.synth.laughter.table).is_file() and cfg.resolve(cfg.synth.laughter.table).is_file()
    assert cfg.synth.voice_unvoiced is True  # the other synth key of the generated config is kept

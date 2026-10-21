"""Model manager backend: table, presets, resumable per-file downloads, self-tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from o_helpers import RangeServer, client_for, drop_runner, fresh_runner, make_env, sha, wait_job

from vndub import models as pipeline_models
from vndub.backends import registry
from vndub.backends.weights import MARKER, is_present, local_dir
from vndub_app import appdirs, models_api


@pytest.fixture
def o_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    data = make_env(monkeypatch, tmp_path)
    fresh_runner()
    yield data
    models_api.set_hub(None)
    drop_runner()


class FakeHub:
    """A stand-in for huggingface_hub: ``repos`` maps a repo id to ``{filename: bytes}``."""

    def __init__(self, repos: dict[str, dict[str, bytes]], *, fail_on: dict[str, Exception] | None = None) -> None:
        self.repos = repos
        self.fail_on = dict(fail_on or {})
        self.fetched: list[tuple[str, str]] = []

    def list_files(self, repo: str, revision: str | None, allow: tuple[str, ...]) -> list[tuple[str, int]]:
        if repo not in self.repos:
            raise OSError(f"Connection refused: {repo}")
        import fnmatch

        return [(n, len(b)) for n, b in self.repos[repo].items() if not allow or any(fnmatch.fnmatch(n, p) for p in allow)]

    def fetch(self, repo: str, filename: str, revision: str | None, local: Path, progress: Callable[[int, int], None]) -> None:
        key = f"{repo}/{filename}"
        if key in self.fail_on:
            raise self.fail_on.pop(key)
        data = self.repos[repo][filename]
        out = local / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        progress(0, len(data))
        half = len(data) // 2
        out.write_bytes(data[:half])
        progress(half, len(data))
        out.write_bytes(data)
        progress(len(data), len(data))
        self.fetched.append((repo, filename))


def repo_of(name: str) -> str:
    return registry.get(name).weights[0].repo


# ---- presets and table ------------------------------------------------------------------------------------------


def test_presets_are_nested_and_made_of_real_backends(o_env: Path) -> None:
    p = models_api.preset_names()
    assert set(p) == {"min", "recommended", "all"}
    assert set(p["min"]) <= set(p["recommended"]) <= set(p["all"])
    assert p["recommended"] == [s.name for s in models_api.shown_specs() if "default" in s.profiles]  # = the default profile
    assert len(p["min"]) == len(models_api.PRESET_MIN) == len(set(p["min"]))
    roles = {registry.get(n).role for n in p["min"]}
    assert {"tts", "llm", "mos", "embedder", "asr", "ser"} <= roles  # one TTS family + translation + scoring + analysis
    tts = [n for n in p["min"] if registry.get(n).role == "tts"]
    assert set(tts) == {"qwen3_tts", "qwen3_tts_voicedesign"} and {registry.get(n).env for n in tts} == {"hf"}  # one family, one env
    assert [n for n in p["min"] if registry.get(n).role == "llm"] == ["hy_mt15_7b"]  # the smallest translation model
    assert not any(n.startswith("mock_") or n == "sapi_tts" for n in p["all"])  # test-only backends are not offered


def test_test_profile_also_lists_the_small_test_models(o_env: Path) -> None:
    default = {s.name for s in models_api.shown_specs("default")}
    test = {s.name for s in models_api.shown_specs("test")}
    assert "whisper_tiny" not in default and "whisper_tiny" in test and default <= test
    assert not any(n.startswith("mock_") for n in test)


def test_models_endpoint_shape(o_env: Path) -> None:
    c = client_for(models_api)
    body = c.get("/api/models").json()
    assert set(body) == {"rows", "presets", "preset_info", "disk"}
    assert body["disk"]["free_gb"] > 0 and body["disk"]["data_dir"] == str(o_env) and body["disk"]["models_dir"] == str(o_env / "models")
    row = next(r for r in body["rows"] if r["name"] == "qwen3_tts")
    assert row["role"] == "tts" and row["env"] == "hf" and row["size_gb"] == pytest.approx(8.6) and row["vram_gb"] == 4.5
    assert row["license"] == "Apache-2.0" and row["commercial_ok"] is True and row["present"] is False and row["usable"] is False
    assert row["reason"] in ("env_missing", "weights_missing") and row["selftest"] is None
    assert set(row["presets"]) == {"min", "recommended", "all"}
    nc = next(r for r in body["rows"] if r["name"] == "fish_s2")
    assert nc["commercial_ok"] is False and nc["presets"] == ["all"]
    for r in body["rows"]:
        assert r["reason"] == "" if r["usable"] else r["reason"] in ("license", "platform", "env_missing", "weights_missing", "unknown")
    info = body["preset_info"]
    assert info["min"]["count"] == 7 and 15 < info["min"]["size_gb"] < 25 and info["min"]["missing_gb"] == info["min"]["size_gb"]
    assert info["recommended"]["size_gb"] > info["min"]["size_gb"] and info["all"]["size_gb"] > info["recommended"]["size_gb"]


def test_table_agrees_with_vndub_models_status(o_env: Path) -> None:
    """The rows are derived from the same registry facts as `vndub models status`."""
    cfg = models_api.build_cfg()
    status = {r["name"]: r for r in pipeline_models.status(cfg, "default")}
    rows = {r["name"]: r for r in models_api.list_models()["rows"]}
    assert set(status) <= set(rows)
    for name, st in status.items():
        assert rows[name]["usable"] == (st["usable"] == "yes"), name
        assert rows[name]["env"] == st["env"] and rows[name]["license"] == st["license"], name
        assert rows[name]["vram_gb"] == pytest.approx(float(st["vram_gb"]), abs=0.05)
        n_pres, n_tot = (int(x) for x in st["weights"].split("/")) if "/" in st["weights"] else (0, 0)
        assert rows[name]["present"] == (n_pres == n_tot), name


def test_present_and_usable_follow_the_weights_and_envs(o_env: Path) -> None:
    spec = registry.get("utmos22")  # env 'score' + weights kind env
    d = local_dir(appdirs.effective_models_dir(), spec.weights[0], spec.name)
    d.mkdir(parents=True)
    (d / MARKER).write_text("x", encoding="utf-8")
    row = next(r for r in models_api.list_models()["rows"] if r["name"] == "utmos22")
    assert row["present"] is True and row["usable"] is False and row["reason"] == "env_missing" and row["reason_params"] == {"env": "score"}
    envs = o_env.parent / "envs" / "score"
    (envs / "Scripts").mkdir(parents=True)
    (envs / "Scripts" / "python.exe").write_bytes(b"")
    (envs / "bin").mkdir(parents=True)
    (envs / "bin" / "python").write_bytes(b"")
    (envs / "pyvenv.cfg").write_text("home=x", encoding="utf-8")
    row = next(r for r in models_api.list_models()["rows"] if r["name"] == "utmos22")
    assert row["usable"] is True and row["reason"] == ""


def test_license_policy_marks_noncommercial_models_unusable(o_env: Path) -> None:
    s = appdirs.load_settings()
    s.license_policy = "commercial"
    appdirs.save_settings(s)
    row = next(r for r in models_api.list_models()["rows"] if r["name"] == "fish_s2")
    assert row["usable"] is False and row["reason"] == "license"


def test_summary_for_doctor(o_env: Path) -> None:
    s = models_api.summary()
    assert s["min_total"] == 7 and s["min_present"] == 0 and s["default_total"] > 20 and s["default_missing_gb"] > 50
    spec = registry.get("ecapa")
    d = local_dir(appdirs.effective_models_dir(), spec.weights[0], spec.name)
    d.mkdir(parents=True)
    (d / MARKER).write_text("x", encoding="utf-8")
    assert models_api.summary()["min_present"] == 1


def test_custom_models_dir_setting_is_used_everywhere(o_env: Path, tmp_path: Path) -> None:
    s = appdirs.load_settings()
    s.models_dir = str(tmp_path / "elsewhere")
    appdirs.save_settings(s)
    assert models_api.list_models()["disk"]["models_dir"] == str(tmp_path / "elsewhere")
    assert str(models_api.build_cfg().project.models_dir) == str(tmp_path / "elsewhere")


# ---- downloads ------------------------------------------------------------------------------------------------------


def download(names: list[str]) -> dict[str, Any]:
    from vndub_app import jobs

    job = jobs.runner().submit("models_download", lambda ctx: models_api.download_models(names, ctx), gpu=False)
    return wait_job(job.id)


def test_download_writes_files_marker_and_per_file_progress(o_env: Path) -> None:
    repo = repo_of("ecapa")
    hub = FakeHub({repo: {"config.yaml": b"c" * 100, "embedding_model.ckpt": b"m" * 5000}})
    models_api.set_hub(hub)
    c = client_for(models_api)
    job = c.post("/api/models/download", json={"names": ["ecapa"]}).json()
    assert job["kind"] == "models_download" and job["gpu"] is False
    done = wait_job(job)
    assert done["state"] == "done", done.get("error")
    spec = registry.get("ecapa")
    d = local_dir(o_env / "models", spec.weights[0], "ecapa")
    assert (d / "embedding_model.ckpt").read_bytes() == b"m" * 5000 and (d / MARKER).exists() and is_present(o_env / "models", spec.weights[0], "ecapa")
    files = done["result"]["files"]
    assert [f["filename"] for f in files] == ["config.yaml", "embedding_model.ckpt"]
    assert all(f["state"] == "done" and f["done"] == f["total"] and f["backend"] == "ecapa" for f in files)
    assert [f["total"] for f in files] == [100, 5000] and files[0]["name"].endswith("/config.yaml")
    assert done["result"]["downloaded"] == ["ecapa"] and done["result"]["failed"] == []
    assert next(r for r in c.get("/api/models").json()["rows"] if r["name"] == "ecapa")["present"] is True
    # nothing is downloaded twice
    hub.fetched.clear()
    again = download(["ecapa"])
    assert again["state"] == "done" and hub.fetched == []


def test_download_resumes_after_a_failed_file(o_env: Path) -> None:
    repo = repo_of("ecapa")
    boom = OSError("Connection reset by peer")
    hub = FakeHub({repo: {"a.bin": b"a" * 1000, "b.bin": b"b" * 1000}}, fail_on={f"{repo}/b.bin": boom})
    models_api.set_hub(hub)
    first = download(["ecapa"])
    assert first["state"] == "failed" and first["error"]["code"] == "download_unreachable"
    assert first["error"]["action"] == "show_manual" and first["error"]["params"]["url"] == f"https://huggingface.co/{repo}"
    assert first["error"]["params"]["model"] == "ecapa" and first["error"]["params"]["target_dir"].endswith(repo.split("/")[-1])
    rows = {f["filename"]: f for f in first["result"]["files"]}
    assert rows["a.bin"]["state"] == "done" and rows["b.bin"]["state"] == "failed"
    spec = registry.get("ecapa")
    assert not is_present(o_env / "models", spec.weights[0], "ecapa")  # no marker: not claimed as present
    second = download(["ecapa"])
    assert second["state"] == "done" and is_present(o_env / "models", spec.weights[0], "ecapa")
    assert hub.fetched[-1] == (repo, "b.bin")


def test_one_failing_model_does_not_stop_the_others(o_env: Path) -> None:
    ok, bad = repo_of("ecapa"), repo_of("gigaam_v3")
    hub = FakeHub({ok: {"x.bin": b"x" * 10}}, fail_on={})  # gigaam's repo is unknown to the hub: unreachable
    models_api.set_hub(hub)
    done = download(["gigaam_v3", "ecapa"])
    assert done["state"] == "failed" and done["error"]["code"] == "download_unreachable"
    assert done["result"]["downloaded"] == ["ecapa"] and [f["name"] for f in done["result"]["failed"]] == ["gigaam_v3"]
    assert is_present(o_env / "models", registry.get("ecapa").weights[0], "ecapa")
    assert bad == registry.get("gigaam_v3").weights[0].repo


def test_download_respects_allow_patterns(o_env: Path) -> None:
    spec = registry.get("gemma4_12b")
    w = spec.weights[0]
    assert w.allow  # a GGUF repo: only the chosen quantization is wanted
    keep = next(p.replace("*", "q4") for p in w.allow)
    hub = FakeHub({w.repo: {keep: b"g" * 200, "other-q8.gguf": b"o" * 200, "README.md": b"r"}})
    models_api.set_hub(hub)
    done = download(["gemma4_12b"])
    assert done["state"] == "done", done.get("error")
    assert [f["filename"] for f in done["result"]["files"]] == [keep]


def test_download_of_an_env_weight_without_the_env_is_env_missing(o_env: Path) -> None:
    done = download(["rvc"])  # weights are pulled by the rvc env itself, which is not installed here
    assert done["state"] == "failed" and done["error"]["code"] == "env_missing" and done["error"]["params"]["env"] == "rvc"
    assert done["error"]["action"] == "open_tools"


def test_download_url_weights_resume_with_range(o_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """kind 'url' weights go through the shared resumable downloader."""
    from dataclasses import replace

    data = b"weights" * 20000
    spec = registry.get("ecapa")
    with RangeServer({"w.bin": data}, drop_first={"w.bin": 5000}) as srv:
        w = replace(spec.weights[0], kind="url", repo="", url=srv.url("w.bin"), filename="w.bin", subdir="urlmodel", size_gb=0.0001)
        monkeypatch.setitem(registry.all_specs(), "ecapa", replace(spec, weights=(w,)))
        done = download(["ecapa"])
        assert done["state"] == "done", done.get("error")
        assert len(srv.requests) == 2 and srv.requests[1][1] == "bytes=5000-"
    d = o_env / "models" / "urlmodel"
    assert (d / "w.bin").read_bytes() == data and (d / MARKER).exists()
    assert sha((d / "w.bin").read_bytes()) == sha(data)


def test_download_refuses_when_the_disk_is_too_small(o_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vndub_app import downloads

    monkeypatch.setattr(downloads, "free_space_gb", lambda p: 0.5)
    models_api.set_hub(FakeHub({}))
    done = download(["qwen3_tts"])
    assert done["error"]["code"] == "disk_full" and done["error"]["params"]["free_gb"] == 0.5 and done["error"]["params"]["need_gb"] > 8


def test_endpoint_validation(o_env: Path) -> None:
    c = client_for(models_api)
    for url in ("/api/models/download", "/api/models/check"):
        assert c.post(url, json={"names": []}).status_code == 422
        r = c.post(url, json={"names": ["no_such_backend"]})
        assert r.status_code == 422 and r.json()["error"]["code"] == "bad_request"


def test_progress_class_counts_bytes_even_when_tqdm_would_be_disabled() -> None:
    seen: list[tuple[int, int]] = []
    bar = models_api._progress_class(lambda d, t: seen.append((d, t)))(total=1000, initial=100, disable=None, unit="B")
    bar.update(400)
    bar.update(500)
    bar.close()
    assert seen[0] == (100, 1000) and seen[-1] == (1000, 1000)


# ---- self-tests -------------------------------------------------------------------------------------------------------


def test_check_job_stores_peak_vram_and_the_table_shows_it(o_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[str] = []

    def fake_selftest(cfg: Any, name: str) -> dict[str, Any]:
        asked.append(name)
        if name == "ecapa":
            return {"ok": True, "peak_vram_gb": 1.25, "device": "cuda:0", "op": "embed"}
        if name == "gigaam_v3":
            return {"ok": False, "error": "weights missing: run `vndub models pull gigaam_v3`", "hint": "run it"}
        return {"ok": False, "error": "env 'hf' not created", "hint": "make_envs"}

    monkeypatch.setattr(pipeline_models, "selftest", fake_selftest)
    c = client_for(models_api)
    job = c.post("/api/models/check", json={"names": ["ecapa", "gigaam_v3", "qwen3_tts"]}).json()
    assert job["kind"] == "models_check" and job["gpu"] is True  # self-tests load models on the GPU: one at a time
    done = wait_job(job)
    assert done["state"] == "done" and asked == ["ecapa", "gigaam_v3", "qwen3_tts"]
    rows = {r["name"]: r for r in done["result"]["rows"]}
    assert rows["ecapa"]["ok"] is True and rows["ecapa"]["peak_vram_gb"] == 1.25 and rows["ecapa"]["code"] is None
    assert rows["gigaam_v3"]["code"] == "weights_missing" and rows["qwen3_tts"]["code"] == "env_missing"
    table = {r["name"]: r for r in c.get("/api/models").json()["rows"]}
    assert table["ecapa"]["selftest"]["ok"] is True and table["ecapa"]["selftest"]["peak_vram_gb"] == 1.25 and table["ecapa"]["selftest"]["device"] == "cuda:0"
    assert table["gigaam_v3"]["selftest"]["ok"] is False and table["gigaam_v3"]["selftest"]["code"] == "weights_missing"
    assert table["ecapa"]["selftest"]["at"]
    # it survives a restart: stored in the data folder
    assert models_api.read_selftests()["ecapa"]["peak_vram_gb"] == 1.25 and (o_env / "model_selftests.json").exists()


def test_check_runs_with_the_apps_models_dir(o_env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import os

    s = appdirs.load_settings()
    s.models_dir = str(tmp_path / "elsewhere")
    appdirs.save_settings(s)
    seen: dict[str, str | None] = {}

    def fake_selftest(cfg: Any, name: str) -> dict[str, Any]:
        seen["env"] = os.environ.get("VNDUB_MODELS_DIR")
        seen["resolved"] = str(pipeline_models.resolve_models_dir(cfg))
        return {"ok": True}

    monkeypatch.setattr(pipeline_models, "selftest", fake_selftest)
    done = wait_job(client_for(models_api).post("/api/models/check", json={"names": ["ecapa"]}).json())
    assert done["state"] == "done" and seen["resolved"] == str(tmp_path / "elsewhere") and seen["env"] == str(tmp_path / "elsewhere")
    assert os.environ.get("VNDUB_MODELS_DIR") is None  # restored afterwards


def test_selftest_codes() -> None:
    f = models_api.selftest_row
    assert f("x", {"ok": True, "peak_vram_gb": None, "device": "cpu"})["code"] is None
    assert f("x", {"ok": False, "error": "license 'x' not allowed"})["code"] == "license"
    assert f("x", {"ok": False, "error": "CUDA OOM"})["code"] == "check_failed"
    assert "hint" in f("x", {"ok": False, "error": "boom", "hint": "hint"})["details"]

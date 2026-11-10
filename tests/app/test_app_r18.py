"""Round 18 (fix-app): settings / state races, full disk, download resume, HF mirror, the problem report's environment,
the environment check before a run, error reasons, translate warnings, the review selection reason, casting."""

from __future__ import annotations

import errno
import json
import os
import threading
from pathlib import Path
from typing import Any

import pytest


# ----------------------------------------------------------------------------
# 1. settings and project state
# ----------------------------------------------------------------------------


def test_volume_drag_from_three_threads_keeps_models_dir(app_env: Path) -> None:
    from vndub_app import appdirs
    from vndub_app.routes import general

    md = app_env / "mymodels"
    md.mkdir()
    appdirs.update_settings(lambda s: setattr(s, "models_dir", str(md)))
    lost: list[int] = []

    def drag(i: int) -> None:
        for n in range(150):
            general.put_settings({"player_volume": {f"k{i}": (n % 100) / 100}})
            if not appdirs.load_settings().models_dir:
                lost.append(n)

    ts = [threading.Thread(target=drag, args=(i,)) for i in range(3)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    s = appdirs.load_settings()
    assert s.models_dir == str(md) and not lost
    assert set(s.player_volume) == {"k0", "k1", "k2"}  # no PUT dropped another one's key


def test_read_json_retries_a_permission_error(app_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vndub_app import appdirs

    f = app_env / "x.json"
    f.write_text('{"a": 1}', encoding="utf-8")
    orig = Path.read_text
    calls = {"n": 0}

    def flaky(self: Path, *a: Any, **k: Any) -> str:
        if self == f and calls["n"] == 0:
            calls["n"] += 1
            raise PermissionError(13, "sharing violation")
        return orig(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", flaky)
    assert appdirs.read_json(f, {}) == {"a": 1}
    assert appdirs.read_json(app_env / "missing.json", {"d": 0}) == {"d": 0}


def test_corrupt_settings_are_kept_aside(app_env: Path) -> None:
    from vndub_app import appdirs

    appdirs.settings_file().write_text('{"models_dir": "D:/m", broken', encoding="utf-8")
    s = appdirs.load_settings()
    assert s.models_dir == ""
    kept = list(app_env.glob("settings.json.corrupt-*"))
    assert len(kept) == 1 and "broken" in kept[0].read_text(encoding="utf-8")


def test_update_app_state_merges_concurrent_changes(app_env: Path, game_dir: Path) -> None:
    from vndub_app import appdirs

    h = appdirs.create_project(game_dir)

    def add(i: int) -> None:
        for n in range(40):
            appdirs.update_app_state(h, lambda st: st.setdefault("review", {}).setdefault("touched", []).append(f"{i}-{n}"))

    ts = [threading.Thread(target=add, args=(i,)) for i in range(3)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(appdirs.read_app_state(h)["review"]["touched"]) == 120


def test_project_state_carries_review_touched(client: Any, game_dir: Path) -> None:
    r = client.post("/api/project/open", json={"game_dir": str(game_dir)})
    assert r.status_code == 200, r.text
    assert r.json()["state"]["review"]["touched"] == []


# ----------------------------------------------------------------------------
# 2. a full disk is a full disk; 3. a download that makes progress keeps its attempts
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("exc", [OSError(errno.ENOSPC, "No space left on device"), OSError(None, "disk full", None, 112)])
def test_full_disk_is_reported_at_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exc: OSError) -> None:
    from vndub_app import downloads
    from vndub_app.errors import AppError

    if exc.errno is None and getattr(exc, "winerror", None) != 112:
        exc.winerror = 112  # type: ignore[attr-defined]  # POSIX: OSError has no winerror slot filled from args
    src = tmp_path / "src.bin"
    src.write_bytes(b"x" * 1000)

    def full(*a: Any, **k: Any) -> Any:
        raise exc

    monkeypatch.setattr(downloads, "_pump", full)
    slept: list[float] = []
    with pytest.raises(AppError) as ei:
        downloads.fetch(src.as_uri(), tmp_path / "out" / "dst.bin", sleep=slept.append)
    assert ei.value.code == "disk_full" and slept == []
    assert set(ei.value.params) >= {"path", "free_gb"}


def test_hf_fetch_does_not_wrap_disk_full(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vndub_app import downloads, models_api
    from vndub_app.errors import AppError

    hub = models_api.HfHub()
    monkeypatch.setattr(hub, "resolve", lambda *a: ("https://cdn.example/x", 10, None))
    monkeypatch.setattr(downloads, "fetch", lambda *a, **k: (_ for _ in ()).throw(downloads.disk_full_error(tmp_path, OSError(28, "full"))))
    with pytest.raises(AppError) as ei:
        hub.fetch("o/r", "f.bin", None, tmp_path, lambda a, b: None)
    assert ei.value.code == "disk_full"


def test_attempts_reset_while_the_part_file_grows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vndub_app import downloads

    drops = {"n": 0}

    def stream(url: str, part: Path, offset: int, *a: Any) -> tuple[int, int]:
        with open(part, "ab") as f:
            f.write(b"y" * 100)
        if drops["n"] < 7:  # more drops than retries: each one made progress
            drops["n"] += 1
            raise ConnectionError("reset")
        return part.stat().st_size, offset

    monkeypatch.setattr(downloads, "_stream", stream)
    slept: list[float] = []
    r = downloads.fetch("https://example.invalid/big.bin", tmp_path / "big.bin", sleep=slept.append)
    assert r.size == 800 and drops["n"] == 7 and max(slept) <= downloads.BACKOFF_CAP_S

    def dead(url: str, part: Path, offset: int, *a: Any) -> tuple[int, int]:
        raise ConnectionError("down")

    monkeypatch.setattr(downloads, "_stream", dead)
    from vndub_app.errors import AppError

    with pytest.raises(AppError) as ei:
        downloads.fetch("https://example.invalid/other.bin", tmp_path / "other.bin", sleep=lambda s: None)
    assert ei.value.code == "download_unreachable"


# ----------------------------------------------------------------------------
# 4. environment.json; 5. the environment check before a run
# ----------------------------------------------------------------------------


def test_report_zip_holds_environment_json(app_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import zipfile

    from vndub_app import doctor, installer, pipeline

    monkeypatch.setattr(doctor, "run_doctor", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("probe broke")))
    monkeypatch.setattr(installer, "env_states", lambda: {"funasr": {"present": True, "problems": []}}, raising=False)
    (app_env / "model_selftests.json").write_text('{"cosyvoice3": {"ok": true}}', encoding="utf-8")
    z = pipeline.build_report_zip(None, app_env / "rep")
    with zipfile.ZipFile(z) as zf:
        env = json.loads(zf.read("environment.json"))
    assert env["app_version"] and "probe broke" in env["doctor"]["error"]
    assert env["envs"]["funasr"]["present"] is True and env["model_selftests"]["cosyvoice3"]["ok"] is True
    assert isinstance(env["tools"], list)


def test_run_is_refused_when_its_environment_is_broken(client: Any, game_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vndub_app import installer, pipeline

    assert client.post("/api/project/open", json={"game_dir": str(game_dir)}).status_code == 200
    monkeypatch.setattr(pipeline, "job_envs", lambda cfg, sections=None: ["funasr"])
    seen: list[Any] = []

    def problems(names: Any, level: str = "error") -> list[dict[str, Any]]:
        seen.append((list(names), level))
        return [{"level": "error", "code": "env_broken", "detail": "No module named 'pkg_resources'", "env": "funasr"}]

    monkeypatch.setattr(installer, "env_problems", problems, raising=False)
    r = client.post("/api/run/full", json={})
    assert r.status_code == 409, r.text
    err = r.json()["error"]
    assert err["code"] == "env_broken" and err["action"] == "open_doctor" and err["params"]["env"] == "funasr"
    assert "pkg_resources" in err["params"]["detail"] and seen == [(["funasr"], "error")]
    monkeypatch.setattr(installer, "env_problems", lambda names, level="error": [], raising=False)  # warnings only: runs
    assert client.post("/api/run/full", json={}).status_code == 200


def test_an_environment_that_was_never_installed_does_not_block_a_run(client: Any, game_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The packaged app's smoke test found it: on a fresh data folder every environment is `env_missing`, and the config
    names backends the user may never install (fallbacks). Only a built environment that does not work blocks a job."""
    from vndub_app import installer, pipeline

    assert client.post("/api/project/open", json={"game_dir": str(game_dir)}).status_code == 200
    monkeypatch.setattr(pipeline, "job_envs", lambda cfg, sections=None: ["main", "rvc"])
    missing = [{"level": "error", "code": "env_missing", "detail": f"{e} is not built", "env": e} for e in ("main", "rvc")]
    monkeypatch.setattr(installer, "env_problems", lambda names, level="error": list(missing), raising=False)
    assert client.post("/api/run/full", json={}).status_code == 200
    incomplete = [*missing, {"level": "error", "code": "env_incomplete", "detail": "the build did not finish", "env": "rvc"}]
    monkeypatch.setattr(installer, "env_problems", lambda names, level="error": list(incomplete), raising=False)
    r = client.post("/api/run/full", json={})
    assert r.status_code in (409, 200)  # 409 env_broken for rvc, unless the first run is still active (then job_busy is 409 too)
    if r.status_code == 409 and r.json()["error"]["code"] == "env_broken":
        assert r.json()["error"]["params"]["env"] == "rvc"


def test_job_envs_maps_config_backends_to_envs(app_env: Path, game_dir: Path) -> None:
    from vndub.backends import registry

    from vndub_app import appdirs, pipeline, session

    h = appdirs.create_project(game_dir)
    cfg = session.read_ctx(h).cfg
    envs = pipeline.job_envs(cfg)
    assert all(isinstance(e, str) and e for e in envs)
    llm = registry.get(cfg.translate.llm.backend)
    assert pipeline.job_envs(cfg, ("translate",)) == ([llm.env] if llm.env else [])


# ----------------------------------------------------------------------------
# 6. the real cause reaches the user (contract C6)
# ----------------------------------------------------------------------------


def test_error_reasons_and_codes() -> None:
    from vndub.backends.base import BackendUnavailable
    from vndub.backends.runner import AdapterError
    from vndub.stages.common import StageError

    from vndub_app.errors import error_from_exception

    e = error_from_exception(StageError("synth: 12 lines failed: voxcpm2 returned no audio\nTraceback ..."), stage="synth")
    assert e.code == "stage_failed" and e.params["reason"] == "synth: 12 lines failed: voxcpm2 returned no audio"
    assert error_from_exception(RuntimeError("x" * 900)).params["reason"].endswith("...")
    assert len(error_from_exception(RuntimeError("x" * 900)).params["reason"]) <= 300
    assert error_from_exception(AdapterError("voxcpm2", "RuntimeError: CUDA out of memory. Tried to allocate 2 GiB")).code == "cuda_oom"
    assert error_from_exception(OSError("[WinError 1455] The paging file is too small for this operation to complete")).code == "paging_file_small"
    for exc in (
        BackendUnavailable("cosyvoice3", "import failed in env 'funasr': No module named 'pkg_resources'", "recreate the env"),
        AdapterError("cosyvoice3", "ErrorDuringImport: problem in cosyvoice.flow - ModuleNotFoundError: No module named 'pkg_resources'"),
    ):
        err = error_from_exception(exc, stage="preview")
        assert err.code == "env_import_failed" and err.action == "open_doctor" and err.params["env"] == "funasr", err.to_dict()
    assert error_from_exception(BackendUnavailable("x", "does not fit the VRAM budget")).code == "backend_unavailable"


# ----------------------------------------------------------------------------
# 7. translate warnings (C3), summary gender (C10)
# ----------------------------------------------------------------------------


def test_translate_warnings_from_the_summary() -> None:
    from vndub_app import pipeline

    ws = pipeline.translate_warnings({"glossary_misses": 3, "glossary_miss_ids": ["a", "b"], "check_failed": 0, "stress": "none (ruaccent missing)"})
    kinds = {w["kind"]: w for w in ws}
    assert kinds["glossary_misses"]["count"] == 3 and kinds["glossary_misses"]["line_ids"] == ["a", "b"]
    assert "check_failed" not in kinds and kinds["stress_missing"]["detail"] == "ruaccent missing"
    assert pipeline.translate_warnings({"stress": "none (not russian)"}) == []
    assert pipeline.translate_warnings({"lines": 5}) == []  # a 1.1.0 summary


# ----------------------------------------------------------------------------
# 8. the selection reason in the review (contract C7)
# ----------------------------------------------------------------------------


def test_why_items_explain_the_selection() -> None:
    from vndub_app import verdicts

    th = verdicts.Thresholds()
    r = verdicts.why_items(metrics={"utmos": 3.8, "spk_sim": 0.5}, gates={"utmos": True, "spk_sim": True}, th=th, selection_status="review",
                           selection_reason="best of 3 passing, but not accepted automatically: gate not checked (CER: asr unavailable)")
    assert r == [{"gate": "selection", "key": "why.gate_unchecked", "params": {"detail": "gate not checked (CER: asr unavailable)"}}]
    r = verdicts.why_items(metrics={"cer": 0.6}, gates={}, th=th, selection_status="review", selection_reason="no gate could be evaluated: nothing to judge the takes by")
    assert r[0]["key"] == "why.gate_unchecked"
    r = verdicts.why_items(metrics={}, gates={}, th=th, selection_status="review",
                           selection_reason="no candidate passes the gates and a new take would not help: cer stuck (best fails: cer)")
    assert r[0]["key"] == "why.regen_hopeless"
    r = verdicts.why_items(metrics={}, gates={}, th=th, selection_status="review", selection_reason="something new")
    assert r[0] == {"gate": "selection", "key": "why.selection_reason", "params": {"detail": "something new"}}
    r = verdicts.why_items(metrics={"cer": 0.5}, gates={"cer": False}, th=th, selection_status="regen", selection_reason="cer")
    assert [x["key"] for x in r] == ["cer"]  # a failing gate already explains it
    assert verdicts.why_items(metrics={}, gates={}, th=th, selection_status="ok", selection_reason="best") == []


# ----------------------------------------------------------------------------
# 9. auto casting (contract C8); 10. casting details
# ----------------------------------------------------------------------------


def test_auto_cast_both_shapes() -> None:
    from vndub_app import pipeline

    assert pipeline._auto_cast_result({"a": "d1"}) == ({"a": "d1"}, [])
    assert pipeline._auto_cast_result({"chosen": {"a": "d1"}, "kept": ["b"]}) == ({"a": "d1"}, ["b"])
    calls: list[Any] = []
    pipeline._call_auto_cast(lambda ctx, overwrite=True: calls.append(overwrite), None)
    pipeline._call_auto_cast(lambda ctx: calls.append("old"), None)
    assert calls == [False, "old"]


def test_tuning_note_limit_and_similar_needs_a_gender(app_env: Path, game_dir: Path) -> None:
    from vndub.text.style_prompt import NOTE_MAX_CHARS

    from vndub_app import casting
    from vndub_app.errors import AppError

    assert casting.validate_tuning({"character_note": "x" * NOTE_MAX_CHARS})["character_note"] == "x" * NOTE_MAX_CHARS
    with pytest.raises(AppError) as ei:
        casting.validate_tuning({"character_note": "x" * (NOTE_MAX_CHARS + 1)})
    assert ei.value.params["max"] == NOTE_MAX_CHARS


def test_similar_tuning_and_gender_source(monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlib import contextmanager

    from vndub.manifest.io import Character

    from vndub_app import casting, translation

    chars = {s: Character(speaker_id=s) for s in ("a", "b", "c")}

    @contextmanager
    def edit(h: Any) -> Any:
        yield chars

    monkeypatch.setattr(casting, "edit_characters", edit)
    monkeypatch.setattr(translation, "write_glossary", lambda h: None)
    assert casting.apply_tuning_to_similar(None, "a") == []  # all unknown: nobody is "similar"
    casting.update_character(None, "a", gender="f")
    casting.update_character(None, "b", gender="f")
    assert chars["a"].model_extra.get("gender_source") == "user"
    assert casting.apply_tuning_to_similar(None, "a") == ["b"]


# ----------------------------------------------------------------------------
# 3. HF mirror (contract C5)
# ----------------------------------------------------------------------------


def test_hf_endpoint_setting_round_trip(client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from vndub_app import appdirs

    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.setattr(appdirs, "_EXPORTED_HF", None)
    assert client.get("/api/settings").json()["hf_endpoint"] == ""
    assert client.put("/api/settings", json={"hf_endpoint": "ftp://x"}).status_code == 422
    r = client.put("/api/settings", json={"hf_endpoint": "https://hf-mirror.com/"})
    assert r.status_code == 200 and r.json()["hf_endpoint"] == "https://hf-mirror.com"
    assert os.environ.get("HF_ENDPOINT") == "https://hf-mirror.com"
    from vndub_app import hf_custom

    assert hf_custom.parse_hf_url("https://hf-mirror.com/org/model").repo == "org/model"
    r = client.put("/api/settings", json={"hf_endpoint": ""})
    assert r.json()["hf_endpoint"] == "" and "HF_ENDPOINT" not in os.environ
    import sys

    c = sys.modules.get("huggingface_hub.constants")
    if c is not None:
        assert c.ENDPOINT == "https://huggingface.co"

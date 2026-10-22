"""The wizard end to end through the HTTP API (all-mock ``apptest`` profile on ``renpy_mini``).

One project is walked through in order (Step 1 import -> Step 2 translation -> Step 3 casting ->
Step 4 preview -> Step 5 full run -> Step 6 review) so the pipeline runs only once; the tests of a
step skip with a reason when an earlier step failed. Total time is about three minutes.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pytest
import soundfile as sf
import yaml
from app_helpers import FIXTURE, reset_app_state, run_job, setup_env, teardown_app_state, wait_job

ERROR_KEYS = {"code", "params", "action", "details"}
JOB_KEYS = {
    "id", "kind", "title_key", "params", "project", "gpu", "state", "created", "started", "finished", "stages", "stage", "stage_index",
    "stage_count", "done", "total", "item", "eta_s", "stage_eta_s", "rate_per_min", "message_key", "message_params", "error", "result",
}


@dataclass
class Flow:
    client: Any
    root: Path
    game: Path
    slug: str = ""
    done: set[str] = field(default_factory=set)
    store: dict[str, Any] = field(default_factory=dict)

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def work(self) -> Path:
        return self.data / "projects" / self.slug / "work"

    def needs(self, *names: str) -> None:
        for n in names:
            if n not in self.done:
                pytest.skip(f"requires the earlier step '{n}' to pass")

    def get(self, path: str, **kw: Any) -> Any:
        r = self.client.get(path, **kw)
        assert r.status_code == 200, f"GET {path}: {r.status_code} {r.text[:500]}"
        return r.json()

    def post(self, path: str, body: Any = None, status: int = 200) -> Any:
        r = self.client.post(path, json=body if body is not None else {})
        assert r.status_code == status, f"POST {path}: {r.status_code} {r.text[:500]}"
        return r.json()

    def put(self, path: str, body: Any, status: int = 200) -> Any:
        r = self.client.put(path, json=body)
        assert r.status_code == status, f"PUT {path}: {r.status_code} {r.text[:500]}"
        return r.json()

    def job(self, path: str, body: Any = None, timeout: float = 400.0) -> dict[str, Any]:
        r = self.client.post(path, json=body if body is not None else {})
        j = run_job(self.client, r, timeout)
        assert set(JOB_KEYS) <= set(j), set(JOB_KEYS) - set(j)
        return j

    def yaml_file(self, rel: str) -> Any:
        return yaml.safe_load((self.work / rel).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def flow(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Flow]:
    from fastapi.testclient import TestClient

    from vndub_app.server import create_app

    root = tmp_path_factory.mktemp("flow")
    mp = pytest.MonkeyPatch()
    setup_env(mp, root)
    game = root / "games" / "renpy_mini"
    shutil.copytree(FIXTURE, game)
    reset_app_state()
    try:
        with TestClient(create_app(), raise_server_exceptions=False) as c:  # a bug shows up as the 500 JSON, with its traceback in `details`
            yield Flow(c, root, game)
    finally:
        teardown_app_state()
        mp.undo()


def assert_error(r: Any, code: str, status: int) -> dict[str, Any]:
    assert r.status_code == status, r.text
    err = r.json()["error"]
    assert set(err) == ERROR_KEYS and err["code"] == code, err
    return err


def write_wav(path: Path, seconds: float, sr: int = 16000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(sr * seconds)) / sr
    sf.write(str(path), (0.2 * np.sin(2 * np.pi * 220 * t)).astype("float32"), sr)
    return path


# ============================================================================
# Step 1: game, import, summary
# ============================================================================


def test_01_open_project_and_import(flow: Flow) -> None:
    p = flow.post("/api/project/open", {"game_dir": str(flow.game)})
    flow.slug = p["slug"]
    assert p["engine"] is None and p["state"]["step"] == "game"
    j = flow.job("/api/project/import")
    assert j["kind"] == "import" and j["project"] == flow.slug and j["gpu"] is True
    assert j["stages"][:5] == ["detect", "extract", "parse", "link", "analyze"] and j["stage_count"] == len(j["stages"])
    assert j["stage"] == j["stages"][-1] and j["stage_index"] == j["stage_count"]
    assert j["result"]["lines"] > 0 and j["result"]["voiced"] > 0
    assert any("vndub.link" in line for line in j["log_tail"]), "stage log lines are forwarded to the job log"
    proj = flow.get("/api/project")
    assert proj["engine"] == "renpy" and proj["project_yaml"]["engine"] == "renpy"
    assert {"detect", "extract", "parse", "link", "analyze"} <= set(proj["stage_status"]) and proj["active_job"] is None
    assert proj["stage_status"]["link"]["summary"].get("mode") in ("no_asr", None)
    flow.done.add("import")


def test_02_summary(flow: Flow) -> None:
    flow.needs("import")
    s = flow.get("/api/project/summary")
    assert s["engine"] == {"id": "renpy", "label": "Ren'Py", "version": s["engine"]["version"], "encrypted": False, "drm": [], "warnings": []}
    assert s["languages"] == {"src": "ja", "src_detected": True, "audio": "ja", "tgt": "ru"}  # detected from the script text
    assert s["tools_missing"] == [] or all({"tool", "url"} <= set(t) for t in s["tools_missing"])
    assert s["voice_files"] == 20
    assert s["lines_total"] == s["voiced"] + s["unvoiced"] and s["voiced"] == 20
    chars = {c["speaker_id"]: c for c in s["characters"]}
    assert {"sakura", "haruto", "yui", "hero"} <= set(chars) and "narrator" not in chars
    assert [c["lines"] for c in s["characters"]] == sorted((c["lines"] for c in s["characters"]), reverse=True)
    for c in chars.values():
        assert c["lines"] == c["voiced_lines"] + c["unvoiced_lines"]
        assert c["mode"] == ("voiced" if c["unvoiced_lines"] == 0 else "unvoiced" if c["voiced_lines"] == 0 else "mixed")
    assert chars["hero"]["mode"] == "unvoiced" and chars["hero"]["minutes"] == 0 and chars["hero"]["voiced_lines"] == 0
    assert chars["sakura"]["mode"] == "voiced" and chars["sakura"]["minutes"] > 0
    assert sum(c["voiced_lines"] for c in chars.values()) == s["voiced"]
    m = s["mapping"]
    assert m["mode"] == "no_asr" and m["ok"] + m["review"] + m["unmatched"] == 20 and 0 <= m["confidence"] <= 1
    assert isinstance(m["suggest_asr_check"], bool)
    assert s["no_text_lines"] == 0 and s["no_text_sample"] == [] and s["recover_offer"] is False and isinstance(s["problem_files"], list)
    assert s["analysis"]["audio_passports"] == 20 and s["analysis"]["text_passports"] > 0, "unvoiced lines got text passports"


def test_03_recover_text_and_asr_check_jobs(flow: Flow) -> None:
    flow.needs("import")
    j = flow.job("/api/project/asr-check")
    assert j["kind"] == "asr_check" and j["stages"] == ["link"]
    assert set(j["result"]) <= {"ok", "review", "unmatched", "confidence"}
    assert flow.get("/api/project/summary")["mapping"]["mode"] == "asr"  # the last link run was the ASR check
    j = flow.job("/api/project/recover-text")
    assert j["kind"] == "recover_text" and j["stages"] == ["parse", "link", "analyze"] and j["result"]["no_text_lines"] == 0
    flow.job("/api/project/import")  # back to the cheap mode (everything cached): the summary reports no_asr again
    assert flow.get("/api/project/summary")["mapping"]["mode"] == "no_asr"


# ============================================================================
# Step 2: translation settings, glossary, cues
# ============================================================================


def test_04_translation_settings_roundtrip(flow: Flow) -> None:
    flow.needs("import")
    t = flow.get("/api/project/translation")
    assert (t["tgt_lang"], t["source"], t["honorifics"], t["address_default"], t["license_policy"]) == ("ru", "auto", "drop", "ты", "research")
    assert t["llm_available"] is True  # the mock translator
    # the source language: detected from the script text unless the user picks one
    assert t["src_lang"] == "auto" and t["src_lang_effective"] == "ja" and t["src_lang_detected"]["lang"] == "ja" and t["src_lang_detected"]["confidence"] > 0.8
    t = flow.put("/api/project/translation", {"src_lang": "en"})
    assert (t["src_lang"], t["src_lang_effective"], t["src_lang_detected"]) == ("en", "en", None)
    py = yaml.safe_load((flow.data / "projects" / flow.slug / "work" / "project.yaml").read_text(encoding="utf-8"))
    assert py["src_lang"] == "en" and not py.get("src_lang_detected")
    assert "src_lang: en" in (flow.data / "projects" / flow.slug / "pipeline.yaml").read_text(encoding="utf-8")
    t = flow.put("/api/project/translation", {"src_lang": "auto"})
    assert t["src_lang"] == "auto" and t["src_lang_effective"] == "ja" and t["src_lang_detected"]["lang"] == "ja"  # detected again
    py = yaml.safe_load((flow.data / "projects" / flow.slug / "work" / "project.yaml").read_text(encoding="utf-8"))
    assert py["src_lang"] == "ja" and py["src_lang_detected"]["lang"] == "ja"
    adv = t["advanced"]
    assert adv["n_candidates"] == 2 and adv["routes"] == {"A": True, "B": True, "C": False} and adv["asr_check"] is False
    assert {"name", "enabled", "vram_gb", "license", "commercial_ok", "role"} == set(adv["backends"][0])
    names = [b["name"] for b in adv["backends"]]
    assert "mock_tts" in names and next(b for b in adv["backends"] if b["name"] == "mock_tts")["enabled"] is True

    t = flow.put("/api/project/translation", {
        "honorifics": "adapt", "address_default": "вы", "license_policy": "commercial",
        "advanced": {"n_candidates": 3, "routes": {"B": False}, "backends": [{"name": "sapi_tts", "enabled": False}], "asr_check": True},
    })
    assert (t["honorifics"], t["address_default"], t["license_policy"]) == ("adapt", "вы", "commercial")
    assert t["advanced"]["n_candidates"] == 3 and t["advanced"]["routes"] == {"A": True, "B": False, "C": False} and t["advanced"]["asr_check"] is True
    # -> glossary.yaml and pipeline.yaml
    gl = yaml.safe_load((flow.data / "projects" / flow.slug / "glossary.yaml").read_text(encoding="utf-8"))
    assert gl["honorifics"] == "map" and gl["target_lang"] == "ru"
    sids = {sid for sid in gl["characters"]}
    assert {"sakura", "haruto", "yui", "hero"} <= sids and "narrator" not in sids
    assert all(gl["address_matrix"][sid]["default"] == "вы" for sid in sids)
    cfg = yaml.safe_load((flow.data / "projects" / flow.slug / "pipeline.yaml").read_text(encoding="utf-8"))
    assert cfg["license_policy"] == "commercial" and cfg["link"]["mode"] == "asr"
    assert cfg["synth"]["n_candidates"] == 3 and cfg["synth"]["n_candidates_short"] == 5 and cfg["synth"]["routes"]["B"] is False
    assert cfg["synth"]["backends"]["sapi_tts"]["enabled"] is False
    # the global setting is the single source of the license policy
    assert flow.get("/api/settings")["license_policy"] == "commercial"
    # back to the defaults the rest of the walk expects
    flow.put("/api/project/translation", {"honorifics": "keep", "address_default": "ты", "license_policy": "research",
                                         "advanced": {"n_candidates": 2, "routes": {"B": True}, "asr_check": False}})
    gl = yaml.safe_load((flow.data / "projects" / flow.slug / "glossary.yaml").read_text(encoding="utf-8"))
    assert gl["honorifics"] == "transliterate" and gl["address_matrix"]["sakura"]["default"] == "ты"
    t = flow.get("/api/project/translation")
    assert t["advanced"]["routes"] == {"A": True, "B": True, "C": False} and t["honorifics"] == "keep"
    flow.put("/api/project/translation", {"honorifics": "drop"})


def test_05_translation_settings_validation(flow: Flow) -> None:
    flow.needs("import")
    bad = [
        {"tgt_lang": "de"}, {"src_lang": "xx"}, {"source": "magic"}, {"honorifics": "x"}, {"address_default": "ну"}, {"license_policy": "free"},
        {"advanced": {"n_candidates": 0}}, {"advanced": {"n_candidates": 99}}, {"advanced": {"n_candidates": "many"}},
        {"advanced": {"routes": {"Z": True}}}, {"advanced": {"routes": {"A": False, "B": False, "C": False}}}, {"advanced": {"backends": [{"name": "nope", "enabled": True}]}},
    ]
    for body in bad:
        err = assert_error(flow.client.put("/api/project/translation", json=body), "bad_request", 422)
        assert err["params"]["field"]
    assert flow.get("/api/project/translation")["honorifics"] == "drop"  # nothing was half-applied


def test_06_glossary_roundtrip(flow: Flow) -> None:
    flow.needs("import")
    g = flow.get("/api/project/glossary")
    ids = [c["speaker_id"] for c in g["characters"]]
    assert {"sakura", "haruto", "yui", "hero"} <= set(ids) and g["terms"] == []
    sakura = next(c for c in g["characters"] if c["speaker_id"] == "sakura")
    assert sakura["src"] and sakura["tgt"] == "" and sakura["gender"] == "unknown"
    out = flow.put("/api/project/glossary", {
        "characters": [{"speaker_id": "sakura", "src": sakura["src"], "tgt": "Сакура", "gender": "f"}, {"speaker_id": "hero", "src": "蓮", "tgt": "Рэн", "gender": "m"}],
        "terms": [{"src": "放課後", "tgt": "после уроков", "note": "time of day"}, {"src": "", "tgt": "ignored"}],
    })
    assert next(c for c in out["characters"] if c["speaker_id"] == "sakura")["tgt"] == "Сакура"
    assert out["terms"] == [{"src": "放課後", "tgt": "после уроков", "note": "time of day"}]
    chars = flow.yaml_file("manifest/characters.yaml")
    assert chars["sakura"]["names"]["tgt"] == "Сакура" and chars["sakura"]["gender"] == "f" and chars["hero"]["gender"] == "m"
    gl = yaml.safe_load((flow.data / "projects" / flow.slug / "glossary.yaml").read_text(encoding="utf-8"))
    assert gl["characters"]["sakura"]["tgt"] == "Сакура" and gl["characters"]["sakura"]["gender"] == "f"
    assert gl["terms"] == [{"src": "放課後", "tgt": "после уроков", "notes": "time of day"}]  # the translate stage reads `notes`
    assert_error(flow.client.put("/api/project/glossary", json={"characters": [{"speaker_id": "ghost", "tgt": "x"}]}), "bad_request", 422)
    assert_error(flow.client.put("/api/project/glossary", json={"characters": [{"speaker_id": "sakura", "gender": "x"}]}), "bad_request", 422)


def test_07_delivery_cues_roundtrip(flow: Flow) -> None:
    flow.needs("import")
    cues = flow.get("/api/project/delivery-cues")  # {language: [{cue, emotion, intensity, ...}]}
    assert isinstance(cues, dict) and cues and all(isinstance(v, list) for v in cues.values())
    lang = next(iter(cues))
    edited = json.loads(json.dumps(cues))
    edited[lang].append({"cue": "app-test-cue", "emotion": "happy", "intensity": 0.7})
    flow.put("/api/project/delivery-cues", edited)
    own = flow.data / "projects" / flow.slug / "delivery_cues.yaml"
    assert own.exists()  # the project's own copy; the bundled table is never edited
    assert any(e["cue"] == "app-test-cue" for e in flow.get("/api/project/delivery-cues")[lang])
    cfg = yaml.safe_load((flow.data / "projects" / flow.slug / "pipeline.yaml").read_text(encoding="utf-8"))
    assert cfg["text_emotion"]["cues"] == str(own)
    flow.put("/api/project/delivery-cues", cues)  # restore
    assert not any(e["cue"] == "app-test-cue" for e in flow.get("/api/project/delivery-cues")[lang])
    bad = {**cues, "xx": {"not": "a list"}}
    assert_error(flow.client.put("/api/project/delivery-cues", json=bad), "bad_request", 422)
    assert_error(flow.client.put("/api/project/delivery-cues", json=[1, 2]), "bad_request", 422)


def test_08_preview_import_formats(flow: Flow, tmp_path: Path) -> None:
    flow.needs("import")
    from vndub.stages import translate

    if not hasattr(translate, "preview_import"):
        pytest.skip("translate.preview_import is not in this build (task P2)")
    lines = [json.loads(l) for l in (flow.work / "manifest" / "lines.jsonl").read_text(encoding="utf-8").splitlines()]
    voiced = [l for l in lines if l["has_voice"]]
    f = tmp_path / "mine.jsonl"
    f.write_text("\n".join(json.dumps({"line_id": l["line_id"], "text_tgt": f"Перевод {i}"}, ensure_ascii=False) for i, l in enumerate(voiced[:7])), encoding="utf-8")
    r = flow.post("/api/project/translation/preview-import", {"path": str(f)})
    assert r["matched"] == 7 and r["total"] >= 20 and len(r["preview"]) == 5 and r["format"]
    assert set(r["preview"][0]) == {"line_id", "text_src", "text_tgt"} and r["preview"][0]["text_tgt"] == "Перевод 0"
    assert_error(flow.client.post("/api/project/translation/preview-import", json={"path": str(tmp_path / "missing.json")}), "file_not_found", 404)
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert_error(flow.client.post("/api/project/translation/preview-import", json={"path": str(broken)}), "bad_request", 422)
    # nothing was written: the translation is still to be made
    assert not (flow.work / "translation").exists() or not list((flow.work / "translation").glob("*.jsonl"))


def test_09_translate_job(flow: Flow) -> None:
    flow.needs("import")
    j = flow.job("/api/project/translate")
    assert j["kind"] == "translate" and j["stages"][0] == "translate" and j["result"]["failed"] == 0 and j["result"]["lines"] >= 20
    assert (flow.work / "translation" / "lines.ru.jsonl").exists()
    flow.done.add("translate")


# ============================================================================
# Step 3: characters and casting
# ============================================================================


def test_10_characters(flow: Flow) -> None:
    flow.needs("translate")
    chars = flow.get("/api/characters")
    assert [c["lines"] for c in chars] == sorted((c["lines"] for c in chars), reverse=True)
    by = {c["speaker_id"]: c for c in chars}
    assert "narrator" not in by and {"sakura", "haruto", "yui", "hero"} <= set(by)
    keys = {"speaker_id", "name_src", "name_tgt", "gender", "age_hint", "description", "lines", "voiced_lines", "minutes", "mode", "emotion_hist", "clips", "chosen", "tuning", "rvc"}
    assert keys <= set(by["sakura"])
    s = by["sakura"]
    assert s["name_tgt"] == "Сакура" and s["gender"] == "f" and s["mode"] == "voiced" and s["minutes"] > 0
    assert 0.99 <= sum(s["emotion_hist"].values()) <= 1.01
    assert [c["kind"] for c in s["clips"]] == ["neutral", "emotional", "longest"]
    for c in s["clips"]:
        assert set(c) == {"kind", "line_id", "url", "text_src", "emotion"}
        r = flow.client.get(c["url"])
        assert r.status_code == 200 and r.headers["content-type"].startswith("audio/")
    assert len({c["line_id"] for c in s["clips"]}) == 3
    h = by["hero"]
    assert h["mode"] == "unvoiced" and h["clips"] == [] and h["voiced_lines"] == 0 and h["emotion_hist"], "text passports give the unvoiced hero a histogram"
    assert h["chosen"] is None and h["rvc"]["state"] == "none"
    assert h["tuning"] == {"pitch_st": 0.0, "speed": 1.0, "expressiveness": 0.5, "energy": 0.0, "timbre_preset": "", "character_note": ""}
    assert by["haruto"]["mode"] == "mixed"
    # the three clips: neutral = most neutral, emotional = most emotional (never the same clip twice)
    flow.store["characters"] = by


def test_11_update_character(flow: Flow) -> None:
    flow.needs("translate")
    c = flow.put("/api/characters/yui", {"gender": "f", "age_hint": "teen", "description": "shy, quiet voice"})
    assert (c["gender"], c["age_hint"], c["description"]) == ("f", "teen", "shy, quiet voice")
    c = flow.put("/api/characters/hero", {"description": "tired protagonist, low voice", "name_tgt": "Рэн", "gender": "m", "age_hint": "teen"})
    assert c["name_tgt"] == "Рэн" and c["mode"] == "unvoiced"
    y = flow.yaml_file("manifest/characters.yaml")
    assert y["yui"]["description"] == "shy, quiet voice" and y["hero"]["names"]["tgt"] == "Рэн"
    gl = yaml.safe_load((flow.data / "projects" / flow.slug / "glossary.yaml").read_text(encoding="utf-8"))
    assert gl["characters"]["hero"]["tgt"] == "Рэн", "the glossary follows the character sheet"
    assert_error(flow.client.put("/api/characters/yui", json={"gender": "robot"}), "bad_request", 422)
    assert_error(flow.client.put("/api/characters/ghost", json={"gender": "f"}), "bad_request", 404)


def _proposals_ok(flow: Flow, sid: str, designs: list[dict[str, Any]]) -> None:
    for d in designs:
        assert {"design_id", "kind", "backend", "prompt", "description", "samples", "spk_sim", "stars", "rank_score", "license", "commercial_ok", "created", "chosen"} <= set(d), set(d)
        assert d["kind"] in ("voice_design", "reference", "clone")
        assert d["samples"], d
        for smp in d["samples"]:
            assert {"line_id", "wav", "text", "url", "text_tgt"} <= set(smp)
            r = flow.client.get(smp["url"])
            assert r.status_code == 200 and r.headers["content-type"].startswith("audio/"), smp
        if d["stars"] is not None:
            assert 1 <= d["stars"] <= 5


def test_12_casting_generate_and_proposals(flow: Flow) -> None:
    flow.needs("translate")
    from vndub.stages import voicebank

    if not hasattr(voicebank, "propose_designs"):
        pytest.skip("voicebank.propose_designs is not in this build (task P3)")
    assert flow.get("/api/casting/sakura/proposals") == []
    j = flow.job("/api/casting/sakura/generate", {"n": 2})
    assert j["kind"] == "casting_generate" and j["params"] == {"speaker_id": "sakura", "n": 2}
    designs = j["result"]["designs"]
    assert len(j["result"]["new"]) >= 2 and len(designs) >= 2
    _proposals_ok(flow, "sakura", designs)
    assert any(d["kind"] == "clone" for d in designs), "a voiced character always gets the route-A clone design"
    listed = flow.get("/api/casting/sakura/proposals")
    assert {d["design_id"] for d in listed} == {d["design_id"] for d in designs}
    assert all(d["chosen"] is False for d in listed)
    clone = next(d for d in listed if d["kind"] == "clone")
    assert clone["stars"] is not None, "voiced characters get a 1-5 star similarity"
    j = flow.job("/api/casting/sakura/generate", {"n": 1})  # "+": three more -> adds to the list
    assert len(flow.get("/api/casting/sakura/proposals")) == len(designs) + 1

    # unvoiced hero: designs come from the character sheet; no original audio -> no stars
    j = flow.job("/api/casting/hero/generate", {"n": 2, "description": "calm young man, low voice"})
    hero = flow.get("/api/casting/hero/proposals")
    assert len(hero) >= 2 and all(d["stars"] is None and d["kind"] != "clone" for d in hero)
    _proposals_ok(flow, "hero", hero)
    assert flow.yaml_file("manifest/characters.yaml")["hero"]["description"] == "calm young man, low voice"
    for sid in ("haruto", "yui"):
        flow.job(f"/api/casting/{sid}/generate", {"n": 2})
    flow.done.add("designs")
    assert_error(flow.client.post("/api/casting/sakura/generate", json={"n": 0}), "bad_request", 422)
    assert_error(flow.client.get("/api/casting/ghost/proposals"), "bad_request", 404)


def test_13_choose_design_and_rvc_chip(flow: Flow) -> None:
    flow.needs("designs")
    props = flow.get("/api/casting/sakura/proposals")
    perf = next(d for d in props if d["kind"] != "clone")
    err = assert_error(flow.client.post("/api/casting/sakura/choose", json={"design_id": perf["design_id"], "route": "Z"}), "bad_request", 422)
    assert err["params"]["field"] == "route"
    assert_error(flow.client.post("/api/casting/sakura/choose", json={"design_id": "nope", "route": "B"}), "bad_request", 404)
    clone = next(d for d in props if d["kind"] == "clone")
    err = assert_error(flow.client.post("/api/casting/sakura/choose", json={"design_id": clone["design_id"], "route": "B"}), "bad_request", 422)
    assert err["params"]["reason"] == "route_mismatch"
    c = flow.post("/api/casting/sakura/choose", {"design_id": perf["design_id"], "route": "B"})
    assert c["chosen"] == {"design_id": perf["design_id"], "route": "B", "kind": perf["kind"]} and c["speaker_id"] == "sakura"
    assert flow.yaml_file("manifest/characters.yaml")["sakura"]["route"] == "B"
    assert next(d for d in flow.get("/api/casting/sakura/proposals") if d["design_id"] == perf["design_id"])["chosen"] is True
    # route B starts RVC training in the background: a job, and the chip follows it
    rvc_jobs = [j for j in flow.get("/api/jobs", params={"project": flow.slug}) if j["kind"] == "rvc_train"]
    assert len(rvc_jobs) == 1 and rvc_jobs[0]["params"] == {"speaker_id": "sakura"}
    j = wait_job(flow.client, rvc_jobs[0]["id"])
    assert j["state"] == "done", (j["error"], j["log_tail"][-10:])
    chip = next(c for c in flow.get("/api/characters") if c["speaker_id"] == "sakura")["rvc"]
    assert chip["state"] == "done" and chip["progress"] == 1.0
    # choosing again does not train twice
    flow.post("/api/casting/sakura/choose", {"design_id": perf["design_id"], "route": "B"})
    assert len([j for j in flow.get("/api/jobs", params={"project": flow.slug}) if j["kind"] == "rvc_train"]) == 1
    # route A (clone) needs no training
    flow.post("/api/casting/haruto/choose", {"design_id": next(d for d in flow.get("/api/casting/haruto/proposals") if d["kind"] == "clone")["design_id"], "route": "A"})
    assert not [j for j in flow.get("/api/jobs", params={"project": flow.slug}) if j["kind"] == "rvc_train" and j["params"]["speaker_id"] == "haruto"]
    # the unvoiced hero: performer only (route P)
    hero = flow.get("/api/casting/hero/proposals")[0]
    c = flow.post("/api/casting/hero/choose", {"design_id": hero["design_id"], "route": "P"})
    assert c["chosen"]["route"] == "P" and c["rvc"]["state"] == "none"


def test_14_auto_casting(flow: Flow) -> None:
    flow.needs("designs")
    res = flow.job("/api/casting/auto")["result"]  # a casting_auto job: designs for whoever lacks them, then choose
    chosen = res["chosen"]
    assert {"sakura", "haruto", "yui", "hero"} <= set(chosen)
    by = {c["speaker_id"]: c for c in flow.get("/api/characters")}
    for sid, did in chosen.items():
        assert by[sid]["chosen"]["design_id"] == did
    assert by["hero"]["chosen"]["route"] == "P" and by["yui"]["chosen"]["route"] in ("A", "B")
    wait_for_idle(flow)


def wait_for_idle(flow: Flow) -> None:
    for j in flow.get("/api/jobs", params={"project": flow.slug, "active": 1}):
        wait_job(flow.client, j["id"])


def test_15_casting_listen(flow: Flow) -> None:
    flow.needs("designs")
    j = flow.job("/api/casting/sakura/listen", {"n": 3})
    items = j["result"]["items"]
    assert j["kind"] == "casting_listen" and 1 <= len(items) <= 3
    for it in items:
        assert set(it) == {"line_id", "text_src", "text_tgt", "original_url", "result_url"}
        assert it["text_tgt"] and it["original_url"] and it["result_url"]
        for u in (it["original_url"], it["result_url"]):
            assert flow.client.get(u).status_code == 200
    j = flow.job("/api/casting/hero/listen", {"n": 2})  # unvoiced: nothing to compare with
    for it in j["result"]["items"]:
        assert it["original_url"] is None and it["result_url"] and flow.client.get(it["result_url"]).status_code == 200
    assert_error(flow.client.post("/api/casting/sakura/listen", json={"n": 0}), "bad_request", 422)
    # the listen run must not leave "synth/master finished" summaries of three lines behind
    st = flow.get("/api/project")["stage_status"]
    assert "master" not in st and "synth" not in st


def test_16_tuning(flow: Flow) -> None:
    flow.needs("designs")
    default = {"pitch_st": 0.0, "speed": 1.0, "expressiveness": 0.5, "energy": 0.0, "timbre_preset": "", "character_note": ""}
    assert flow.get("/api/characters/sakura/tuning") == default
    t = {"pitch_st": 2.0, "speed": 1.1, "expressiveness": 0.8, "energy": 0.3, "timbre_preset": "younger", "character_note": "cheeky"}
    assert flow.put("/api/characters/sakura/tuning", t) == t
    assert flow.get("/api/characters/sakura/tuning") == t
    y = flow.yaml_file("manifest/characters.yaml")["sakura"]["performer"]
    assert y["tuning"] == t and y.get("design_id"), "tuning is stored next to the chosen design"
    for bad in ({"pitch_st": 7}, {"pitch_st": "high"}, {"speed": 0.5}, {"speed": 1.5}, {"expressiveness": 1.1}, {"energy": -2}, {"timbre_preset": "huge"}, {"unknown": 1}):
        err = assert_error(flow.client.put("/api/characters/sakura/tuning", json=bad), "bad_request", 422)
        assert err["params"]["field"] in bad
    assert flow.get("/api/characters/sakura/tuning") == t
    assert_error(flow.client.get("/api/characters/ghost/tuning"), "bad_request", 404)
    # partial PUT fills the defaults
    assert flow.put("/api/characters/haruto/tuning", {"pitch_st": -3})["pitch_st"] == -3.0
    assert flow.get("/api/characters/haruto/tuning")["speed"] == 1.0
    # "apply to similar": same gender + age_hint
    flow.put("/api/characters/haruto", {"gender": "f", "age_hint": "teen"})
    flow.put("/api/characters/sakura", {"age_hint": "teen"})
    out = flow.post("/api/characters/sakura/tuning/apply-similar")
    assert set(out["updated"]) == {"haruto", "yui"}  # sakura, haruto and yui are female teens now; hero is not
    assert flow.get("/api/characters/yui/tuning") == t and flow.get("/api/characters/haruto/tuning") == t
    assert flow.get("/api/characters/hero/tuning") == default


def test_17_tuning_listen_does_not_persist(flow: Flow) -> None:
    flow.needs("designs")
    stored = flow.get("/api/characters/sakura/tuning")
    trial = {"pitch_st": -1.5, "speed": 0.9, "expressiveness": 0.2, "energy": -0.5, "timbre_preset": "softer", "character_note": "trial"}
    j = flow.job("/api/characters/sakura/tuning/listen", {"tuning": trial})
    items = j["result"]["items"]
    assert [i["kind"] for i in items] == ["neutral", "emotional"] and items[0]["line_id"] != items[1]["line_id"]
    for it in items:
        assert set(it) == {"line_id", "kind", "url", "text_tgt"} and it["text_tgt"] and it["url"]
        r = flow.client.get(it["url"])
        assert r.status_code == 200 and r.headers["content-type"].startswith("audio/")
    assert flow.get("/api/characters/sakura/tuning") == stored, "the trial tuning is rendered, not saved"
    assert flow.yaml_file("manifest/characters.yaml")["sakura"]["performer"]["tuning"] == stored
    # without a body the stored tuning is rendered
    j = flow.job("/api/characters/hero/tuning/listen", {})
    assert len(j["result"]["items"]) >= 1
    assert_error(flow.client.post("/api/characters/sakura/tuning/listen", json={"tuning": {"speed": 3}}), "bad_request", 422)


def test_18_upload_sample(flow: Flow, tmp_path: Path) -> None:
    flow.needs("designs")
    ok = write_wav(tmp_path / "mine.wav", 6.0)
    with open(ok, "rb") as f:
        r = flow.client.post("/api/casting/yui/upload", files={"file": ("mine.wav", f, "audio/wav")})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["kind"] == "reference" and d["samples"] and d["samples"][0]["url"]
    assert any(x["design_id"] == d["design_id"] for x in flow.get("/api/casting/yui/proposals"))
    assert flow.client.get(d["samples"][0]["url"]).status_code == 200
    assert list((flow.data / "projects" / flow.slug / "uploads" / "yui").glob("*mine.wav")), "uploads are kept in the project's uploads/ folder"
    short = write_wav(tmp_path / "short.wav", 1.0)
    with open(short, "rb") as f:
        assert_error(flow.client.post("/api/casting/yui/upload", files={"file": ("short.wav", f, "audio/wav")}), "sample_too_short", 400)
    long = write_wav(tmp_path / "long.wav", 25.0)
    with open(long, "rb") as f:
        assert_error(flow.client.post("/api/casting/yui/upload", files={"file": ("long.wav", f, "audio/wav")}), "sample_too_long", 400)
    notes = tmp_path / "notes.txt"
    notes.write_text("not audio", encoding="utf-8")
    with open(notes, "rb") as f:
        assert_error(flow.client.post("/api/casting/yui/upload", files={"file": ("notes.txt", f, "text/plain")}), "sample_bad_format", 400)
    garbage = tmp_path / "garbage.wav"
    garbage.write_bytes(b"RIFFxxxxnot really a wav file" * 20)
    with open(garbage, "rb") as f:
        assert_error(flow.client.post("/api/casting/yui/upload", files={"file": ("garbage.wav", f, "audio/wav")}), "sample_bad_format", 400)
    with open(ok, "rb") as f:
        assert_error(flow.client.post("/api/casting/ghost/upload", files={"file": ("mine.wav", f, "audio/wav")}), "bad_request", 404)
    # failed uploads leave no files behind
    left = [p.name for p in (flow.data / "projects" / flow.slug / "uploads" / "yui").iterdir()]
    assert len(left) == 1 and left[0].endswith("mine.wav")


def test_19_voice_library(flow: Flow) -> None:
    flow.needs("designs")
    assert flow.get("/api/voices") == []
    err = assert_error(flow.client.post("/api/voices", json={"speaker_id": "sakura", "name": ""}), "bad_request", 422)
    assert err["params"]["field"] == "name"
    v = flow.post("/api/voices", {"speaker_id": "sakura", "name": "Bright girl"})
    assert set(v) == {"id", "name", "gender", "description", "sample_url", "source_project"}
    assert v["name"] == "Bright girl" and v["gender"] == "f" and v["source_project"] == flow.slug and v["sample_url"] == f"/api/audio/_voices/{v['id']}.wav"
    assert flow.client.get(v["sample_url"]).status_code == 200
    assert (flow.data / "voices" / f"{v['id']}.yaml").exists() and (flow.data / "voices" / f"{v['id']}.wav").exists()
    assert [x["id"] for x in flow.get("/api/voices")] == [v["id"]]
    v2 = flow.post("/api/voices", {"speaker_id": "sakura", "name": "Bright girl"})  # same name: a distinct id
    assert v2["id"] != v["id"]
    d = flow.post("/api/casting/yui/from-library", {"voice_id": v["id"]})
    assert d["kind"] in ("voice_design", "reference") and d["samples"] and d["samples"][0]["url"]
    assert flow.client.get(d["samples"][0]["url"]).status_code == 200
    assert any(x["design_id"] == d["design_id"] for x in flow.get("/api/casting/yui/proposals"))
    assert_error(flow.client.post("/api/casting/yui/from-library", json={"voice_id": "nope"}), "file_not_found", 404)
    assert_error(flow.client.post("/api/casting/yui/from-library", json={"voice_id": "../settings"}), "file_not_found", 404)
    # a clone is not a reusable voice
    flow.post("/api/casting/haruto/choose", {"design_id": next(x for x in flow.get("/api/casting/haruto/proposals") if x["kind"] == "clone")["design_id"], "route": "A"})
    assert_error(flow.client.post("/api/voices", json={"speaker_id": "haruto", "name": "x"}), "bad_request", 422)
    # nothing chosen -> no_design_chosen
    flow.put("/api/characters/yui", {"description": "x"})
    wait_for_idle(flow)


# ============================================================================
# Step 4: preview
# ============================================================================


def test_20_preview_subset(flow: Flow) -> None:
    flow.needs("translate")
    from vndub_app import appdirs, pipeline

    h = appdirs.project_handle(flow.slug)
    scene_ids = pipeline.first_scene_ids(h)
    lines = [json.loads(l) for l in (flow.work / "manifest" / "lines.jsonl").read_text(encoding="utf-8").splitlines()]
    first_scene = min(lines, key=lambda l: l["order"])["scene"]
    assert scene_ids and all(next(l for l in lines if l["line_id"] == i)["scene"] == first_scene for i in scene_ids)
    assert not any(next(l for l in lines if l["line_id"] == i)["speaker_id"] == "narrator" for i in scene_ids), "narration is context, never dubbed"
    assert any(next(l for l in lines if l["line_id"] == i)["speaker_id"] == "hero" for i in scene_ids), "the unvoiced protagonist is part of the preview"

    assert_error(flow.client.post("/api/preview/run", json={"mode": "everything"}), "bad_request", 422)
    assert_error(flow.client.post("/api/preview/run", json={"mode": "first_n", "n": 0}), "bad_request", 422)
    j = flow.job("/api/preview/run", {"mode": "first_n", "n": 5})
    assert j["kind"] == "preview" and len(j["params"]["line_ids"]) == 5 and j["stages"][-3:] == ["synth", "score", "master"]
    assert j["result"]["line_ids"] == j["params"]["line_ids"]
    st = flow.get("/api/project")
    assert st["state"]["preview"]["line_ids"] == j["params"]["line_ids"] and st["state"]["preview"]["mode"] == "first_n"
    assert "master" not in st["stage_status"], "a subset run does not claim that the stage finished for the project"

    r = flow.get("/api/preview/items")
    assert r["line_ids"] == j["params"]["line_ids"] and len(r["items"]) == 5
    for it in r["items"]:
        assert set(it) == {"line_id", "speaker_id", "name", "text_src", "text_tgt", "original_url", "dub_url", "status"}
        assert it["text_tgt"] and it["status"] in ("ok", "check", "failed")
        assert it["dub_url"] and flow.client.get(it["dub_url"]).status_code == 200, it
        if it["speaker_id"] != "hero":
            assert it["original_url"] and flow.client.get(it["original_url"]).status_code == 200
    flow.done.add("preview")


def test_21_preview_first_scene_can_be_cancelled(flow: Flow) -> None:
    flow.needs("preview")
    r = flow.client.post("/api/preview/run", json={"mode": "first_scene"})
    assert r.status_code == 200
    j = r.json()
    assert j["kind"] == "preview" and len(j["params"]["line_ids"]) >= 5 and j["params"]["mode"] == "first_scene"
    assert flow.client.post(f"/api/jobs/{j['id']}/cancel").status_code == 200
    final = wait_job(flow.client, j["id"])
    assert final["state"] in ("cancelled", "done")  # on a very fast machine it may finish before the cancel lands
    assert flow.client.get("/api/project").json()["active_job"] is None


# ============================================================================
# Step 5: full run
# ============================================================================


def test_22_full_run(flow: Flow) -> None:
    flow.needs("preview")
    j = flow.job("/api/run/full", {"shutdown_when_done": False}, timeout=600)
    assert j["kind"] == "full" and j["stages"] == ["translate", "text_emotion", "voicebank", "synth", "score", "master", "pack", "report"]
    assert j["stage_index"] == 8 and j["result"]["shutdown"] is False and sum(j["result"]["score"].values()) >= 26
    assert j["stage"] == "report" and j["state"] == "done" and j["error"] is None
    st = flow.get("/api/project")["stage_status"]
    for stage in ("translate", "voicebank", "synth", "score", "master", "pack", "report"):
        assert st[stage]["finished"] and not st[stage]["error"], stage
    assert (flow.data / "projects" / flow.slug / "out" / "renpy" / "INSTALL.md").exists()
    assert flow.get("/api/settings")["shutdown_when_done"] is False
    flow.done.add("full")


def test_23_full_run_with_shutdown_flag_never_shuts_down(flow: Flow) -> None:
    flow.needs("full")
    j = flow.job("/api/run/full", {"shutdown_when_done": True})  # everything is cached: seconds
    assert j["result"]["shutdown"] is False  # the apptest profile / VNDUB_NO_SHUTDOWN: skipped
    assert flow.get("/api/settings")["shutdown_when_done"] is True
    flow.put("/api/settings", {"shutdown_when_done": False})


# ============================================================================
# Step 6: review
# ============================================================================


def test_24_review_list_filters(flow: Flow) -> None:
    flow.needs("full")
    r = flow.get("/api/review/lines")
    assert set(r) >= {"items", "facets", "counts", "total"} and r["total"] == len(r["items"]) >= 20
    assert r["counts"]["ok"] + r["counts"]["check"] + r["counts"]["failed"] == r["total"]
    it = r["items"][0]
    assert set(it) == {"line_id", "speaker_id", "name", "scene", "status", "composite", "text_src", "text_tgt", "winner_label", "flags", "delivery", "touched"}
    assert it["status"] in ("ok", "check", "failed") and it["touched"] is False and it["winner_label"] in ("A", "B", "C", "D")
    assert {s["speaker_id"] for s in r["facets"]["speakers"]} >= {"sakura", "hero"} and {s["scene"] for s in r["facets"]["scenes"]} >= {"start", "after_menu"}
    # voiced -> audio delivery; unvoiced -> the text passport's layer
    by = {i["line_id"]: i for i in r["items"]}
    voiced = next(i for i in r["items"] if i["speaker_id"] == "sakura")
    assert voiced["delivery"] == {"source": "audio", "layer": None}
    hero = next(i for i in r["items"] if i["speaker_id"] == "hero")
    assert hero["delivery"]["source"] == "text" and hero["delivery"]["layer"] in ("rule", "cue", "llm", "neutral")
    # filters
    only = flow.get("/api/review/lines", params={"speaker": "hero"})
    assert only["items"] and {i["speaker_id"] for i in only["items"]} == {"hero"} and only["total"] == len(only["items"])
    assert {i["scene"] for i in flow.get("/api/review/lines", params={"scene": "after_menu"})["items"]} == {"after_menu"}
    assert flow.get("/api/review/lines", params={"status": "failed"})["items"] == []
    q = voiced["text_src"][:4]
    assert voiced["line_id"] in {i["line_id"] for i in flow.get("/api/review/lines", params={"q": q})["items"]}
    assert voiced["line_id"] in {i["line_id"] for i in flow.get("/api/review/lines", params={"q": voiced["text_tgt"][:5].upper()})["items"]}, "search is case-insensitive over the translation too"
    assert flow.get("/api/review/lines", params={"q": "zzzz-no-such-text"})["items"] == []
    comp = sorted(i["composite"] for i in r["items"] if i["composite"] is not None)
    mid = comp[len(comp) // 2]
    hi = flow.get("/api/review/lines", params={"min": mid})["items"]
    assert hi and all(i["composite"] >= mid for i in hi) and len(hi) < len(r["items"])
    lo = flow.get("/api/review/lines", params={"max": mid})["items"]
    assert lo and all(i["composite"] <= mid for i in lo)
    page = flow.get("/api/review/lines", params={"limit": 3, "offset": 2})
    assert [i["line_id"] for i in page["items"]] == [i["line_id"] for i in r["items"][2:5]] and page["total"] == r["total"]
    assert_error(flow.client.get("/api/review/lines", params={"status": "weird"}), "bad_request", 422)
    assert by


def test_25_review_detail(flow: Flow) -> None:
    flow.needs("full")
    items = flow.get("/api/review/lines")["items"]
    voiced = next(i for i in items if i["speaker_id"] == "sakura")
    d = flow.get(f"/api/review/lines/{voiced['line_id']}")
    assert set(d) >= {"line", "original_url", "candidates", "why", "style_prompt", "text_tts", "refs", "tuning_override"}
    assert d["line"]["line_id"] == voiced["line_id"] and d["original_url"] and flow.client.get(d["original_url"]).status_code == 200
    cands = d["candidates"]
    assert 1 <= len(cands) <= 3 and [c["label"] for c in cands] == ["A", "B", "C"][: len(cands)]
    assert sum(c["is_winner"] for c in cands) == 1 and next(c for c in cands if c["is_winner"])["label"] == voiced["winner_label"]
    comps = [c["composite"] for c in cands]
    assert comps == sorted(comps, reverse=True), "labels follow the composite order"
    for c in cands:
        assert set(c) == {"id", "label", "url", "backend", "route", "composite", "passed", "is_winner", "verdicts"}
        assert flow.client.get(c["url"]).status_code == 200
        for v in c["verdicts"]:
            assert set(v) == {"metric", "key", "level", "value", "tooltip_params"} and v["level"] in ("good", "ok", "bad")
            assert v["key"] in {
                "voice": ("similar", "close", "different"), "emotion": ("close", "partly", "far"), "clarity": ("good", "ok", "poor"),
                "length": ("ok", "longish", "too_long", "shortish", "too_short"), "quality": ("good", "ok", "poor"),
            }[v["metric"]]
        assert {v["metric"] for v in c["verdicts"]} >= {"voice", "emotion", "quality"}
    assert d["why"] == [] and d["text_tts"] and d["refs"] and all(set(r) == {"path", "url", "emotion"} for r in d["refs"]) and d["tuning_override"] is None
    assert d["style_prompt"], "the winner's style prompt comes from its synth sidecar"
    # an unvoiced line: no original, no voice / emotion verdicts against it
    hero = next(i for i in items if i["speaker_id"] == "hero")
    dh = flow.get(f"/api/review/lines/{hero['line_id']}")
    assert dh["original_url"] is None and dh["line"]["delivery"]["source"] == "text"
    # no original to compare with: the voice verdict says it is the performer's consistency (the tooltip must not claim "like the original")
    assert {v["tooltip_params"]["reference"] for c in dh["candidates"] for v in c["verdicts"] if v["metric"] == "voice"} <= {"performer"}
    assert not any(v["metric"] == "emotion" for c in dh["candidates"] for v in c["verdicts"]), "no emotion reference without original audio"
    assert {v["tooltip_params"]["reference"] for c in cands for v in c["verdicts"] if v["metric"] == "voice"} == {"original"}
    assert_error(flow.client.get("/api/review/lines/nonexistent_line"), "file_not_found", 404)


def test_26_review_status_mapping_and_why(flow: Flow) -> None:
    """Hand-craft selection / scores for two lines: a failing gate (check) and a missing winner (failed)."""
    flow.needs("full")
    sel_path = flow.work / "selection.jsonl"
    original = sel_path.read_text(encoding="utf-8")
    items = [i for i in flow.get("/api/review/lines")["items"] if i["speaker_id"] == "yui"]
    a, b = items[0]["line_id"], items[1]["line_id"]
    rows = [json.loads(l) for l in original.splitlines()]
    scores_a = (flow.work / "scores" / f"{a}.jsonl")
    scores_orig = scores_a.read_text(encoding="utf-8")
    try:
        for r in rows:
            if r["line_id"] == a:
                r["status"], r["reason"] = "review", "no candidate passes the gates after 1 regen round(s)"
            if r["line_id"] == b:
                r.update(status="review", winner=None, backend=None, route=None, scores={}, reason="no candidates")
        sel_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        recs = [json.loads(l) for l in scores_orig.splitlines()]
        for rec in recs:  # candidate fails the cer + duration gates (the apptest profile has soft gates: cer <= 1.0, duration 0.4-2.5)
            rec["metrics"].update(cer=1.5, dur_ratio=4.0)
            rec["gates"].update(cer=False, dur=False)
            rec["passed"] = False
        scores_a.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n", encoding="utf-8")

        r = flow.get("/api/review/lines")
        by = {i["line_id"]: i for i in r["items"]}
        assert by[a]["status"] == "check" and by[b]["status"] == "failed"
        order = [i["status"] for i in r["items"]]
        assert order == sorted(order, key=lambda s: {"check": 0, "failed": 1, "ok": 2}[s]), "check lines come first, then failed, then ok"
        assert flow.get("/api/review/lines", params={"status": "check"})["items"][0]["line_id"] == a
        da = flow.get(f"/api/review/lines/{a}")
        keys = [w["key"] for w in da["why"]]
        assert "cer" in keys and "dur_ratio" in keys
        cer = next(w for w in da["why"] if w["key"] == "cer")
        assert cer["gate"] == "cer" and cer["params"]["value"] == 1.5 and cer["params"]["max"] == 1.0  # the limit comes from the project config
        dur = next(w for w in da["why"] if w["key"] == "dur_ratio")
        assert dur["params"] == {"value": 4.0, "min": 0.4, "max": 2.5}
        verdict = {v["metric"]: (v["key"], v["level"]) for v in da["candidates"][0]["verdicts"]}
        assert verdict["clarity"] == ("poor", "bad") and verdict["length"] == ("too_long", "bad")
        db = flow.get(f"/api/review/lines/{b}")
        assert all(not c["is_winner"] for c in db["candidates"])
        assert "no_candidates" in [w["key"] for w in db["why"]], "a line without a winner always says why"
        assert next(w for w in db["why"] if w["key"] == "no_candidates")["params"]["detail"] == "no candidates"
    finally:
        sel_path.write_text(original, encoding="utf-8")
        scores_a.write_text(scores_orig, encoding="utf-8")
    assert flow.get("/api/review/lines")["counts"]["check"] == 0


def test_27_review_actions_and_apply(flow: Flow) -> None:
    flow.needs("full")
    items = flow.get("/api/review/lines")["items"]
    sakura = [i for i in items if i["speaker_id"] == "sakura"]
    haruto = [i for i in items if i["speaker_id"] == "haruto"]
    hero = next(i for i in items if i["speaker_id"] == "hero")
    la, lb, lc, ld = sakura[0]["line_id"], sakura[1]["line_id"], haruto[0]["line_id"], hero["line_id"]

    def act(lid: str, action: str, payload: Any = None, status: int = 200) -> Any:
        return flow.post(f"/api/review/lines/{lid}/action", {"action": action, "payload": payload}, status=status)

    d = flow.get(f"/api/review/lines/{la}")
    other = next(c for c in d["candidates"] if not c["is_winner"])
    res = act(la, "accept", {"label": other["label"]})
    assert res == {"ok": True, "touched": [la], "touched_count": 1}
    ov = [json.loads(l) for l in (flow.work / "overrides.jsonl").read_text(encoding="utf-8").splitlines()]
    assert ov[-1]["action"] == "accept" and ov[-1]["payload"]["candidate"] == other["id"] and ov[-1]["line_id"] == la
    # Enter on a line without choosing accepts the current winner
    assert act(lb, "accept")["touched"] == [la, lb]
    assert [json.loads(l) for l in (flow.work / "overrides.jsonl").read_text(encoding="utf-8").splitlines()][-1]["payload"]["candidate"] == next(
        c for c in flow.get(f"/api/review/lines/{lb}")["candidates"] if c["is_winner"])["id"]
    assert flow.get("/api/review/lines", params={"q": la})["items"][0]["touched"] is True
    err = assert_error(flow.client.post(f"/api/review/lines/{la}/action", json={"action": "accept", "payload": {"candidate": "synth/nope.wav"}}), "bad_request", 422)
    assert err["params"]["field"] == "candidate"
    assert_error(flow.client.post(f"/api/review/lines/{la}/action", json={"action": "reject", "payload": {}}), "bad_request", 422)
    assert_error(flow.client.post(f"/api/review/lines/{la}/action", json={"action": "explode"}), "bad_request", 422)
    assert_error(flow.client.post("/api/review/lines/ghost/action", json={"action": "keep_original"}), "file_not_found", 404)

    act(lc, "set_text", {"text": "Да, давай так и сделаем.", "style_prompt": "calm"})
    act(lc, "regen", {"style_prompt": "warm"})
    assert_error(flow.client.post(f"/api/review/lines/{lc}/action", json={"action": "set_text", "payload": {"text": "  "}}), "bad_request", 422)
    ref = flow.get(f"/api/review/lines/{lc}")["refs"][0]
    act(lc, "set_ref", {"ref": ref["path"]})
    assert_error(flow.client.post(f"/api/review/lines/{lc}/action", json={"action": "set_ref", "payload": {"ref": "../../etc/passwd"}}), "bad_request", 422)
    assert_error(flow.client.post(f"/api/review/lines/{lc}/action", json={"action": "set_ref", "payload": {"ref": "voicebank/missing.wav"}}), "bad_request", 422)
    act(ld, "tuning", {"pitch_st": 1.0, "speed": 1.05, "expressiveness": 0.9})
    assert flow.get(f"/api/review/lines/{ld}")["tuning_override"]["pitch_st"] == 1.0
    assert_error(flow.client.post(f"/api/review/lines/{ld}/action", json={"action": "tuning", "payload": {"pitch_st": 50}}), "bad_request", 422)
    act(ld, "set_src_text", {"text": "……ゆい、本当に大丈夫か？"})
    reg = act(lc, "regen", {"text": "Ладно, договорились.", "style_prompt": "bright"})
    assert reg["ok"] is True
    written = [json.loads(l) for l in (flow.work / "overrides.jsonl").read_text(encoding="utf-8").splitlines()]
    tail = [(o["line_id"], o["action"]) for o in written if o["line_id"] == lc]
    assert ("set_text" in [a for _, a in tail]) and tail[-1] == (lc, "regen") and tail[-2] == (lc, "set_text"), "regen with text also records set_text first (as the Gradio tool does)"
    touched = flow.get("/api/review/lines", params={"q": ""})
    assert {i["line_id"] for i in touched["items"] if i["touched"]} == {la, lb, lc, ld}

    # "mark all as route X where it passed"
    pr = flow.post("/api/review/prefer-route", {"route": "A", "speaker_id": "yui"})
    assert pr["updated"] == len(pr["lines"]) and all(lid not in {la, lb, lc, ld} for lid in pr["lines"])
    assert_error(flow.client.post("/api/review/prefer-route", json={"route": "Q"}), "bad_request", 422)
    for lid in pr["lines"]:
        det = flow.get(f"/api/review/lines/{lid}")
        assert det["line"]["touched"] is True
    n_touched = len({la, lb, lc, ld} | set(pr["lines"]))

    j = flow.job("/api/review/apply")
    assert j["kind"] == "review_apply" and j["stages"] == ["synth", "score", "master", "pack"]
    assert len(j["params"]["line_ids"]) == n_touched
    assert j["result"]["changed"] == len(j["result"]["lines"]) and set(j["result"]["lines"]) <= set(j["params"]["line_ids"])
    assert la in j["result"]["lines"] or lb in j["result"]["lines"] or lc in j["result"]["lines"], "an accepted other candidate / regenerated line changed"
    assert "pack_error" not in j["result"]
    assert flow.get("/api/project")["state"]["review"]["touched"] == []
    assert flow.get("/api/project")["state"]["review"]["last_apply"]["changed"] == j["result"]["changed"]
    assert flow.get("/api/review/lines", params={"q": la})["items"][0]["touched"] is False
    st = flow.get("/api/project")["stage_status"]
    assert st["synth"]["summary"]["lines"] >= 20, "the apply run did not overwrite the full run's stage summary"
    # the accepted candidate is now the winner and the patch holds its audio
    assert next(c for c in flow.get(f"/api/review/lines/{la}")["candidates"] if c["is_winner"])["id"] == other["id"]
    assert (flow.data / "projects" / flow.slug / "out" / "renpy").exists()
    # nothing touched any more
    err = assert_error(flow.client.post("/api/review/apply"), "nothing_to_review", 400)
    assert err["action"] == "none"


def test_28_jobs_overview_and_project_status(flow: Flow) -> None:
    flow.needs("full")
    jobs_ = flow.get("/api/jobs", params={"project": flow.slug})
    kinds = {j["kind"] for j in jobs_}
    assert {"import", "translate", "preview", "full", "review_apply", "casting_generate", "casting_listen", "tuning_listen", "rvc_train", "asr_check", "recover_text"} <= kinds
    assert all(j["state"] in ("done", "cancelled", "failed") for j in jobs_)
    assert flow.get("/api/jobs", params={"project": flow.slug, "active": 1}) == []
    one = flow.get(f"/api/jobs/{jobs_[0]['id']}")
    assert "log_tail" in one and one["id"] == jobs_[0]["id"]
    assert flow.get("/api/project")["resume"] == {"available": False, "job": None, "next_stage": None}
    ps = flow.get("/api/projects")
    assert [p["slug"] for p in ps] == [flow.slug] and ps[0]["engine"] == "renpy"


def test_29_patch_audio_served_from_out(flow: Flow) -> None:
    flow.needs("full")
    from vndub_app import system

    mp = pytest.MonkeyPatch()
    opened: list[str] = []
    mp.setattr(system, "open_path", lambda p: opened.append(str(p)) or True)
    try:
        assert flow.post("/api/system/open", {"kind": "patch"}) == {"ok": True}
        assert flow.post("/api/system/open", {"kind": "work"}) == {"ok": True}
    finally:
        mp.undo()
    assert Path(opened[0]) == flow.data / "projects" / flow.slug / "out" / "renpy", "'open the patch folder' opens out/<engine>"
    assert Path(opened[1]) == flow.work
    out = flow.data / "projects" / flow.slug / "out"
    voice = next(out.rglob("*.ogg"), None)
    if voice is None:
        pytest.skip("pack wrote no .ogg files")
    rel = voice.relative_to(out).as_posix()
    r = flow.client.get(f"/api/audio/{flow.slug}/_out/{rel}")
    assert r.status_code == 200 and r.headers["content-type"].startswith("audio/ogg")

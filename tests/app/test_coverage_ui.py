"""What the patch covers, as the app shows it (API level): the review rows' ``patch`` state, the edited text's flags, the stale import.

One project is imported and translated once (all-mock ``apptest`` profile on ``renpy_mini``, seconds); the review state is
written by hand where a whole pipeline run would only slow the test down (the run itself is covered by ``test_flow.py`` and
the Playwright walk in ``test_coverage_e2e.py``).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pytest
from app_helpers import FIXTURE, reset_app_state, run_job, setup_env, teardown_app_state

from vndub_app import appdirs, pipeline, session, translation


@dataclass
class P:
    client: Any
    slug: str
    work: Path

    @property
    def paths(self) -> Any:
        """``WorkPaths`` of the project: ``lines``, ``selection``, ``final_index``, ``translation(lang)``."""
        return session.work_paths(appdirs.project_handle(self.slug))


@pytest.fixture(scope="module")
def proj(tmp_path_factory: pytest.TempPathFactory) -> Iterator[P]:
    from fastapi.testclient import TestClient

    from vndub_app.server import create_app

    root = tmp_path_factory.mktemp("coverage")
    mp = pytest.MonkeyPatch()
    setup_env(mp, root)
    game = root / "games" / "renpy_mini"
    shutil.copytree(FIXTURE, game)
    reset_app_state()
    try:
        with TestClient(create_app(), raise_server_exceptions=False) as c:
            slug = c.post("/api/project/open", json={"game_dir": str(game)}).json()["slug"]
            run_job(c, c.post("/api/project/import"), 300)
            run_job(c, c.post("/api/project/translate"), 300)
            yield P(c, slug, root / "data" / "projects" / slug / "work")
    finally:
        teardown_app_state()
        mp.undo()


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------
# flags follow the edited text
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flags", "text", "expected"),
    [
        (["honorific", "laughter", "interjection"], "Нет, так нельзя.", ["honorific"]),  # a laugh turned into words
        ([], "Ха-ха-ха!", ["laughter", "interjection"]),  # words turned into a laugh
        ([], "Ха-ха, ну надо же!", ["laughter"]),  # a laugh inside a sentence is not an interjection
        (["laughter", "interjection"], "Ха-ха, ну надо же!", ["laughter"]),  # the sentence is no longer laughter only
        (["laughter"], "Хе-хе.", ["laughter", "interjection"]),
        (["interjection"], "Ах!", ["interjection"]),  # an interjection of the source text (no laugh) is not ours to drop
        (["text_from_asr", "laughter"], "Haha!", ["text_from_asr", "laughter", "interjection"]),  # other flags stay, in their order
        (["laughter", "interjection", "x"], "", ["x"]),
    ],
)
def test_retag_flags(flags: list[str], text: str, expected: list[str]) -> None:
    assert translation.retag_flags(flags, text) == expected
    assert translation.retag_flags(translation.retag_flags(flags, text), text) == expected  # idempotent


def test_a_review_text_edit_updates_the_flags_of_the_line(proj: P) -> None:
    tr_path = proj.paths.translation("ru")
    rows = jsonl(tr_path)
    target = next(r for r in rows if r["text_tgt"].strip() and "translate_failed" not in r["flags"])
    lid = target["line_id"]
    before = list(target["flags"])
    target["flags"] = [*before, "custom_flag"]
    write_jsonl(tr_path, rows)

    def flags() -> list[str]:
        return next(r for r in jsonl(tr_path) if r["line_id"] == lid)["flags"]

    def act(action: str, payload: dict[str, Any]) -> None:
        r = proj.client.post(f"/api/review/lines/{lid}/action", json={"action": action, "payload": payload})
        assert r.status_code == 200, r.text

    act("set_text", {"text": "Ха-ха-ха!"})
    assert flags() == [*before, "custom_flag", "laughter", "interjection"]
    act("set_text", {"text": "Нет, так нельзя."})  # the laugh is gone
    assert flags() == [*before, "custom_flag"]
    act("regen", {"text": "Хе-хе.", "style_prompt": "soft"})  # regen with a text is a text edit too
    assert flags() == [*before, "custom_flag", "laughter", "interjection"]
    act("regen", {"style_prompt": "bright"})  # no new text: nothing to follow
    assert flags() == [*before, "custom_flag", "laughter", "interjection"]
    act("keep_original", {})  # other actions leave the flags alone
    assert flags() == [*before, "custom_flag", "laughter", "interjection"]
    # the other lines are untouched
    others_before = {r["line_id"]: r["flags"] for r in rows if r["line_id"] != lid}
    assert {r["line_id"]: r["flags"] for r in jsonl(tr_path) if r["line_id"] != lid} == others_before


def test_sync_text_flags_without_a_record_changes_nothing(proj: P) -> None:
    assert translation.sync_text_flags(appdirs.project_handle(proj.slug), "no-such-line", "Ха-ха!") is False


# ----------------------------------------------------------------------------
# the review rows know where the line stands in the patch
# ----------------------------------------------------------------------------


def test_review_rows_say_whether_the_line_is_in_the_patch(proj: P) -> None:
    ok, prov, absent, failed = [r["line_id"] for r in jsonl(proj.paths.lines) if r.get("has_voice")][:4]
    sel_path, fin_path = proj.paths.selection, proj.paths.final_index

    def sel(lid: str, status: str, winner: str | None = "synth/x.wav") -> dict[str, Any]:
        return {"line_id": lid, "winner": winner, "backend": "mock_tts", "route": "A", "scores": {"composite": 0.7}, "status": status, "reason": "", "round": 0, "n_candidates": 1}

    def fin(lid: str, status: str) -> dict[str, Any]:
        return {"line_id": lid, "voice_file": f"voice/{lid}.ogg", "final_wav": f"final/{lid}.wav", "encoded": f"encoded/{lid}.ogg", "source": "synth/x.wav", "status": status, "stretch": 1.0, "duration_s": 1.0}

    write_jsonl(sel_path, [sel(ok, "auto"), sel(prov, "review"), sel(absent, "review"), sel(failed, "review", winner=None)])

    def patch_of() -> dict[str, Any]:
        r = proj.client.get("/api/review/lines")
        assert r.status_code == 200, r.text
        return {i["line_id"]: i for i in r.json()["items"]}

    try:
        # nothing was mastered yet: no patch, no patch state (a "check" line is not claimed to be missing from anything)
        fin_path.unlink(missing_ok=True)
        rows = patch_of()
        assert {rows[x]["patch"] for x in (ok, prov, absent, failed)} == {None} and rows[prov]["status"] == "check"

        write_jsonl(fin_path, [fin(ok, "auto"), fin(prov, "provisional")])
        rows = patch_of()
        assert rows[ok]["patch"] == "ok" and rows[prov]["patch"] == "provisional"
        assert rows[prov]["status"] == "check" and rows[ok]["status"] == "ok"
        assert rows[absent]["patch"] == "none" and rows[failed]["patch"] == "none"  # dubbed lines without a file: silent in the game
        detail = proj.client.get(f"/api/review/lines/{prov}").json()
        assert detail["line"]["patch"] == "provisional"
        assert proj.client.get(f"/api/review/lines/{absent}").json()["line"]["patch"] == "none"

        # the user's apply masters and packs again: the provisional take is replaced by an accepted one, the chip follows
        write_jsonl(sel_path, [sel(ok, "auto"), sel(prov, "override"), sel(absent, "review"), sel(failed, "review", winner=None)])
        write_jsonl(fin_path, [fin(ok, "auto"), fin(prov, "override")])
        rows = patch_of()
        assert rows[prov]["patch"] == "ok" and rows[prov]["status"] == "ok"
        assert proj.client.get(f"/api/review/lines/{prov}").json()["line"]["patch"] == "ok"
        assert set(next(iter(rows.values()))) >= {"line_id", "status", "touched", "patch"}
    finally:
        fin_path.unlink(missing_ok=True)
        sel_path.unlink(missing_ok=True)


# ----------------------------------------------------------------------------
# a project imported by an older version is told to import again
# ----------------------------------------------------------------------------


def test_stale_import_is_reported_until_the_import_runs_again(proj: P) -> None:
    status_path = proj.work / "stage_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    now = status["parse"]["version"]
    assert now == "5" and proj.client.get("/api/project").json()["stale"] == []

    status["parse"]["version"] = "2"  # a project imported before the Ren'Py identifiers were computed as Ren'Py does
    status_path.write_text(json.dumps(status), encoding="utf-8")
    stale = proj.client.get("/api/project").json()["stale"]
    assert stale == [{"stage": "parse", "ran": "2", "now": "5"}]

    del status["parse"]["version"]  # no record of a version: nothing to compare
    status_path.write_text(json.dumps(status), encoding="utf-8")
    assert proj.client.get("/api/project").json()["stale"] == []
    status["parse"]["version"] = "2"
    status_path.write_text(json.dumps(status), encoding="utf-8")

    run_job(proj.client, proj.client.post("/api/project/import"), 300)  # the app re-runs the import
    assert proj.client.get("/api/project").json()["stale"] == []
    assert json.loads(status_path.read_text(encoding="utf-8"))["parse"]["version"] == "5"


def test_stale_stages_unit() -> None:
    assert pipeline.stale_stages({}) == []
    assert pipeline.stale_stages({"parse": {"version": "5"}}) == []
    assert pipeline.stale_stages({"parse": {}}) == []
    assert pipeline.stale_stages({"parse": {"version": "1"}, "synth": {"version": "0"}}) == [{"stage": "parse", "ran": "1", "now": "5"}]  # only the watched stages


# ----------------------------------------------------------------------------
# the coverage of a run
# ----------------------------------------------------------------------------


def test_run_coverage_reads_the_manifest_of_the_pack_summary(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest_patch.json"
    cov = {"dub_lines": 5, "installed": 4, "provisional": 2, "provisional_review": 1, "provisional_regen": 1, "left_out": 1,
           "left_out_reasons": {"no winner": 1}, "left_out_lines": [{"line_id": "l1", "status": "no winner", "reason": "x"}], "narration": 3, "include_review_lines": "provisional"}
    manifest.write_text(json.dumps({"engine": "renpy", "coverage": cov, "files": []}), encoding="utf-8")
    got = pipeline.run_coverage({"pack": {"manifest": str(manifest), "files": 4}})
    assert got == {"dub_lines": 5, "installed": 4, "provisional": 2, "provisional_review": 1, "provisional_regen": 1, "left_out": 1,
                   "left_out_reasons": {"no_winner": 1}, "narration": 3, "own_voice": 0, "include_review_lines": "provisional"}
    assert pipeline.run_coverage({}) is None  # pack did not run
    assert pipeline.run_coverage({"pack": {"manifest": str(tmp_path / "gone.json")}}) is None
    manifest.write_text(json.dumps({"engine": "renpy", "files": []}), encoding="utf-8")  # a manifest without the block
    assert pipeline.run_coverage({"pack": {"manifest": str(manifest)}}) is None

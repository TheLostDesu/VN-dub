"""Mode C in the app: voiced lines without script text, text recovered by ASR, the user's text edits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app_helpers import run_job


def _rewrite_jsonl(path: Path, fn: Any) -> None:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [fn(r) for r in rows]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def test_lines_without_text_recovered_text_and_edits(client: Any, game_dir: Path) -> None:
    from vndub_app import appdirs, viewdata

    p = client.post("/api/project/open", json={"game_dir": str(game_dir)}).json()
    h = appdirs.project_handle(p["slug"])
    run_job(client, client.post("/api/project/import", json={}))
    base = client.get("/api/project/summary").json()
    assert base["no_text_lines"] == 0 and base["text_from_asr_lines"] == 0
    voiced_ids = [l["line_id"] for l in (json.loads(x) for x in (h.work / "manifest" / "lines.jsonl").read_text(encoding="utf-8").splitlines()) if l["has_voice"] and "voice_sustain" not in l["flags"]]
    a, b = voiced_ids[:2]

    # 1. the script has no text for two voiced lines (an unsupported script): the offer "recognize the text from the voice"
    _rewrite_jsonl(h.work / "manifest" / "lines.jsonl", lambda r: {**r, "text_src": ""} if r["line_id"] in (a, b) else r)
    viewdata.clear_cache()
    s = client.get("/api/project/summary").json()
    assert s["no_text_lines"] == 2 and s["recover_offer"] is True and {x["line_id"] for x in s["no_text_sample"]} == {a, b} and all(x["voice_file"] for x in s["no_text_sample"])
    assert s["voiced"] == base["voiced"], "lines without text are still voiced lines"

    # 2. ASR recovered them (link.jsonl `text_recovered`, as the link stage stores it): they drop out of the list and carry the flag
    _rewrite_jsonl(h.work / "link.jsonl", lambda r: {**r, "text_recovered": "ねえ、聞こえる？"} if r["line_id"] == a else ({**r, "text_recovered": "うん、聞こえるよ。"} if r["line_id"] == b else r))
    viewdata.clear_cache()
    s = client.get("/api/project/summary").json()
    assert s["no_text_lines"] == 0 and s["text_from_asr_lines"] == 2 and s["recover_offer"] is False

    # 3. review: a recovered line is flagged and explained until the user edits the text
    sel = [{"line_id": x, "winner": f"synth/{x}/mock_tts_A_0.wav", "backend": "mock_tts", "route": "A", "scores": {"composite": 0.5}, "status": "review",
            "reason": "no candidate passes the gates", "round": 0, "n_candidates": 1} for x in (a, b)]
    (h.work / "selection.jsonl").write_text("\n".join(json.dumps(r) for r in sel) + "\n", encoding="utf-8")
    viewdata.clear_cache()
    items = {i["line_id"]: i for i in client.get("/api/review/lines").json()["items"]}
    assert items[a]["flags"] == ["text_from_asr"] and items[a]["text_src"] == "ねえ、聞こえる？" and items[a]["status"] == "check"
    assert "text_from_asr" in [w["key"] for w in client.get(f"/api/review/lines/{a}").json()["why"]]

    r = client.post(f"/api/review/lines/{a}/action", json={"action": "set_src_text", "payload": {"text": "ねえ、聞こえてる？"}})
    assert r.status_code == 200 and r.json()["touched"] == [a]
    items = {i["line_id"]: i for i in client.get("/api/review/lines").json()["items"]}
    assert items[a]["text_src"] == "ねえ、聞こえてる？" and set(items[a]["flags"]) == {"text_from_asr", "text_edited"}
    assert "text_from_asr" not in [w["key"] for w in client.get(f"/api/review/lines/{a}").json()["why"]], "no ASR warning once the user has corrected the text"
    assert "text_from_asr" in [w["key"] for w in client.get(f"/api/review/lines/{b}").json()["why"]]
    assert client.post(f"/api/review/lines/{a}/action", json={"action": "set_src_text", "payload": {"text": " "}}).status_code == 422

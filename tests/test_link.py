from __future__ import annotations

import json

import numpy as np

from analysis_helpers import add_orphan, build_project
from vndub.manifest.io import LineRecord, LinkRecord, read_jsonl
from vndub.stages import link
from vndub.stages.common import run_stage


def L(lid: str, text: str, spk: str = "a", voiced: bool = True, flags=None, **extra) -> LineRecord:
    return LineRecord(line_id=lid, scene="s", order=0, speaker_id=spk, text_src=text, has_voice=voiced, audio_id=f"au_{lid}" if voiced else None, flags=flags or [], **extra)


def test_similarity_and_expected_sustain_text():
    lines = [L("1", "ああ、そうしよう。", sustain_lines=["2"]), L("2", "雨が降りそうだしな。", voiced=False, flags=["voice_sustain"])]
    exp = link.expected_texts(lines)
    assert exp["1"] == "ああ、そうしよう。雨が降りそうだしな。"
    m = link.Matcher(lines, 5, 3)
    assert m.bound_score("1", "ああそうしよう雨が降りそうだしな") == 1.0
    assert link.similarity("", "x") == 0.0
    # continuation lines are never relink targets
    assert all(t.line_id != "2" for t in m.targets)


def test_judge_ok_review_unmatched_and_proposal():
    lines = [L("1", "おはようございます"), L("2", "今日はいい天気だね"), L("3", "明日は雨が降るらしい", voiced=False)]
    m = link.Matcher(lines, 5, 3)
    rec, prop = link.judge(lines[0], "おはようございます", m, 0.8, 0.6, True)
    assert rec.status == "ok" and prop is None
    rec, _ = link.judge(lines[0], "おはようござる", m, 0.8, 0.6, True)
    assert rec.status == "review" and "review band" in rec.reasons[0]
    rec, prop = link.judge(lines[1], "明日は雨が降るらしい", m, 0.8, 0.6, True)
    assert rec.status == "unmatched" and prop == ("3", 1.0)
    rec, prop = link.judge(lines[1], "明日は雨が降るらしい", m, 0.8, 0.6, False)
    assert prop is None
    rec, _ = link.judge(lines[0], "", m, 0.8, 0.6, True, "mock")
    assert rec.status == "review" and "empty" in rec.reasons[0]
    rec, _ = link.judge(lines[0], None, m, 0.8, 0.6, True)
    assert rec.status == "review" and "asr failed" in rec.reasons[0]


def test_apply_proposals_move_conflict_and_swap():
    recs = {k: LinkRecord(line_id=k, audio_id=f"au{k}", status="unmatched") for k in ("1", "2", "4", "5")}
    owner = {"1": "au1", "2": "au2", "3": None, "4": "au4", "5": "au5"}
    # 1 -> 3 (free), 2 -> 4 (owned, 4 does not move) = conflict, 4 <-> 5 swap
    props = [("1", "3", 0.95), ("2", "4", 0.9), ("4", "5", 0.92), ("5", "4", 0.91)]
    moved = link.apply_proposals(recs, props, owner)
    assert owner["3"] == "au1" and owner["1"] is None
    assert recs["3"].relinked_from == "1" and recs["3"].status == "ok"
    assert recs["1"].audio_id is None and "moved to 3" in recs["1"].reasons[-1]
    assert owner["2"] == "au2" and "already has a voice" in recs["2"].reasons[-1]
    assert owner["4"] == "au5" and owner["5"] == "au4"
    assert recs["4"].relinked_from == "5" and recs["5"].relinked_from == "4"
    assert moved == 3


def test_speaker_checks_outlier_and_cluster_majority():
    rng = np.random.default_rng(0)
    base_a, base_b = rng.normal(size=16), rng.normal(size=16)
    embs, owner, spk = {}, {}, {}
    for i in range(5):
        embs[f"a{i}"] = base_a + rng.normal(scale=0.05, size=16)
        owner[f"la{i}"], spk[f"la{i}"] = f"a{i}", "alice"
    for i in range(4):
        embs[f"b{i}"] = base_b + rng.normal(scale=0.05, size=16)
        owner[f"lb{i}"], spk[f"lb{i}"] = f"b{i}", "bob"
    # a line labelled alice whose clip sounds like bob
    embs["x"] = base_b + rng.normal(scale=0.05, size=16)
    owner["lx"], spk["lx"] = "x", "alice"
    recs = {k: LinkRecord(line_id=k, audio_id=v, status="ok") for k, v in owner.items()}
    out = link.speaker_checks(recs, owner, spk, embs, distance=0.3, outlier=0.5, judged=set(owner))
    assert out["clusters"] == 2
    assert recs["lx"].status == "review"
    assert any("clusters with speaker 'bob'" in r for r in recs["lx"].reasons)
    assert recs["la0"].status == "ok" and recs["la0"].spk_consistency > 0.9
    assert recs["la0"].spk_cluster != recs["lb0"].spk_cluster


def _sidecar(ctx, aid: str, text: str) -> None:
    (ctx.paths.audio_src16 / f"{aid}.json").write_text(json.dumps({"text_used": text}, ensure_ascii=False), encoding="utf-8")


ROWS = [
    dict(line_id="s_01", speaker_id="sakura", text="ねえ、ちょっと待って！", f0=260, seconds=1.0, audio_id="sak_01"),
    dict(line_id="s_02", speaker_id="haruto", text="どうしたの、さくら？", f0=120, seconds=1.0, audio_id="har_01"),
    dict(line_id="s_03", speaker_id="sakura", text="今日はいい天気だね。", f0=265, seconds=1.0, audio_id="sak_02"),
    dict(line_id="s_04", speaker_id="haruto", text="そうだな、散歩しようか。", f0=125, seconds=1.0, audio_id="har_02"),
    dict(line_id="s_05", speaker_id="sakura", text="テストで百点を取ったんだよ！", voiced=False),
    dict(line_id="s_06", speaker_id="haruto", text="また明日な。", voiced=False),
]


def test_link_stage_ok_review_relink_orphan(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path)
    build_project(ctx, ROWS)
    _sidecar(ctx, "sak_01", "ねえちょっと待って")  # exact -> ok
    _sidecar(ctx, "har_01", "どうしたのさくらちゃん")  # review band
    _sidecar(ctx, "sak_02", "テストで百点を取ったんだよ")  # belongs to unvoiced s_05 -> relink
    _sidecar(ctx, "har_02", "そうだな散歩しようか")
    add_orphan(ctx, "har_99")
    _sidecar(ctx, "har_99", "また明日な")  # unbound file -> s_06
    summary = run_stage("link", ctx)
    recs = {r.line_id: r for r in read_jsonl(ctx.paths.link, LinkRecord)}
    assert recs["s_01"].status == "ok" and recs["s_01"].match_score == 1.0
    assert recs["s_02"].status == "review"
    assert recs["s_05"].audio_id == "sak_02" and recs["s_05"].relinked_from == "s_03"
    assert recs["s_03"].audio_id is None and recs["s_03"].status == "unmatched"
    assert recs["s_06"].audio_id == "har_99" and recs["s_06"].relinked_from == "orphan:har_99"
    assert summary["relinked"] == 1 and summary["orphans_bound"] == 1
    assert all(r.spk_cluster is not None for r in recs.values() if r.audio_id)
    queue = read_jsonl(ctx.paths.review_queue)
    assert {q["line_id"] for q in queue} == {"s_02", "s_03"}
    # downstream stages see the repaired binding
    voiced = {l.line_id: l for l in ctx.voiced_lines()}
    assert "s_03" not in voiced and voiced["s_05"].audio_id == "sak_02" and "relinked" in voiced["s_05"].flags
    assert voiced["s_06"].audio_id == "har_99"
    ctx.models.close()


def test_link_stage_marks_review_when_asr_unavailable(make_ctx, tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)
    build_project(ctx, ROWS[:2])
    monkeypatch.setattr(link, "run_asr", lambda *a, **k: ({}, "", "env 'asr' not created"))
    summary = run_stage("link", ctx)
    recs = read_jsonl(ctx.paths.link, LinkRecord)
    assert len(recs) == 2 and all(r.status == "review" for r in recs)
    assert all("asr unavailable: env 'asr' not created" in r.reasons[0] for r in recs)
    assert summary["asr"].startswith("unavailable")
    ctx.models.close()


def test_link_asr_results_are_cached(make_ctx, tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)
    build_project(ctx, ROWS[:2])
    _sidecar(ctx, "sak_01", "ねえちょっと待って")
    run_stage("link", ctx)
    ctx.models.close()

    def boom(*a, **k):
        raise AssertionError("model acquired although every result is cached")

    monkeypatch.setattr(ctx.models, "acquire", boom)
    run_stage("link", ctx)
    assert read_jsonl(ctx.paths.link, LinkRecord)[0].status == "ok"

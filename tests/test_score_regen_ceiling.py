"""Regen rounds only where a new take can help: a line whose every take misreads the text goes to review at once
(``score.regen_cer_max``), and a take the ASR heard no words in is an unchecked gate, not a CER of 1.0."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.genproj import build
from vndub.backends import mock as mockb
from vndub.config import load_config
from vndub.manifest.io import CandidateScore, OverrideRecord, SelectionRecord, SynthSidecar, read_jsonl
from vndub.stages import genutil as gu
from vndub.stages import score, synth
from vndub.stages.score import apply_scores, regen_hopeless, select

CFG = load_config("configs/pipeline.yaml")


def _c(wav: str, **m) -> CandidateScore:
    return apply_scores(CandidateScore(line_id="l1", wav=wav, backend="qwen3_tts", route="P", metrics=m), CFG, 0.3, False)


def _sides(*wavs: str, rnd: int = 0) -> dict:
    return {w: SynthSidecar(line_id="l1", backend="qwen3_tts", route="P", k=0, round=rnd, created="2026-01-01T00:00:00") for w in wavs}


# ---- select() -------------------------------------------------------------------------------------


def test_every_take_misreading_the_text_goes_to_review_without_a_regen_round():
    far = _c("a.wav", utmos=4.5, cer=0.25, spk_sim=0.5)
    farther = _c("b.wav", utmos=4.0, cer=0.31, spk_sim=0.5)
    sides = _sides("a.wav", "b.wav")
    assert select("l1", [far, farther], sides, [], 2).status == "regen"  # no ceiling: the old behaviour
    s = select("l1", [far, farther], sides, [], 2, regen_cer_max=0.2)
    assert s.status == "review" and s.winner == "a.wav"
    assert "a new take would not help" in s.reason and "cer 0.25-0.31, more than 0.20" in s.reason
    assert "best fails: cer 0.25>0.1" in s.reason
    assert s.round == 0 and s.n_candidates == 2


def test_a_take_that_nearly_reads_the_text_keeps_the_regen_round():
    far = _c("a.wav", utmos=4.5, cer=0.25, spk_sim=0.5)
    near = _c("c.wav", utmos=2.5, cer=0.12, spk_sim=0.5)  # fails utmos, but its reading is close to the gate
    assert select("l1", [far, near], _sides("a.wav", "c.wav"), [], 2, regen_cer_max=0.2).status == "regen"
    at = _c("d.wav", utmos=4.5, cer=0.2, spk_sim=0.5)  # exactly the ceiling is not above it
    assert select("l1", [at], _sides("d.wav"), [], 2, regen_cer_max=0.2).status == "regen"


def test_a_best_take_failing_another_gate_only_is_never_hopeless():
    quiet = _c("d.wav", utmos=2.0, cer=0.05, spk_sim=0.5)
    assert regen_hopeless([quiet], quiet, 0.2) is None
    assert select("l1", [quiet], _sides("d.wav"), [], 2, regen_cer_max=0.2).status == "regen"
    nocer = _c("e.wav", utmos=2.0, spk_sim=0.5)  # no CER at all: nothing to judge the words by
    assert regen_hopeless([nocer], nocer, 0.2) is None
    assert regen_hopeless([quiet], quiet, None) is None


def test_the_rounds_used_up_review_as_before():
    far = _c("a.wav", utmos=4.5, cer=0.25, spk_sim=0.5)
    s = select("l1", [far], _sides("a.wav", rnd=2), [], 2, regen_cer_max=0.2)
    assert s.status == "review" and "after 2 regen round(s)" in s.reason and "would not help" not in s.reason


def test_a_persons_decision_still_wins():
    far = _c("a.wav", utmos=4.5, cer=0.25, spk_sim=0.5)
    acc = OverrideRecord(line_id="l1", action="accept", payload={"candidate": "a.wav"}, ts="2025-12-31T00:00:00")
    assert select("l1", [far], _sides("a.wav"), [acc], 2, regen_cer_max=0.2).status == "override"


def test_the_default_config_has_the_ceiling():
    assert CFG.score.regen_cer_max == pytest.approx(0.2)
    assert CFG.score.regen_rounds == 2


# ---- the stage with mocks ---------------------------------------------------------------------------


def _a_line_with_only_synth_takes(ctx) -> str:
    for line in ctx.dub_lines():
        files = gu.candidate_files(ctx, line.line_id)
        if files and all(s is not None and s.route != "orig" for _, s in files):
            return line.line_id
    raise AssertionError("no line with synth takes only")


def test_an_empty_transcript_is_an_unchecked_gate_not_a_cer_of_one(make_ctx, tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path / "game")
    build(ctx, n_per=2)
    synth.run(ctx)
    lid = _a_line_with_only_synth_takes(ctx)
    orig = mockb.MockASR.op_transcribe

    def deaf(self, item, lang="ja", **kw):
        res = orig(self, item, lang=lang, **kw)
        return {**res, "text": ""} if lid in Path(item["wav"]).parts else res

    monkeypatch.setattr(mockb.MockASR, "op_transcribe", deaf)
    summary = score.run(ctx)
    rows = read_jsonl(ctx.paths.scores(lid), CandidateScore)
    assert rows and all(r.metrics.get("cer") is None for r in rows)
    assert all("cer failed: the ASR heard no words in this take" in r.notes for r in rows)
    sel = {s.line_id: s for s in read_jsonl(ctx.paths.selection, SelectionRecord)}[lid]
    assert sel.status == "review" and "gate not checked" in sel.reason and "heard no words" in sel.reason
    assert summary["lines_with_unchecked_gates"] >= 1 and "review_without_regen" not in summary
    others = [s for s in read_jsonl(ctx.paths.selection, SelectionRecord) if s.line_id != lid]
    assert others and all("heard no words" not in s.reason for s in others)


def test_a_line_misread_by_every_take_skips_the_regen_round_in_the_stage(make_ctx, tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path / "game", sets={"score.gates.cer_max": 0.1})
    build(ctx, n_per=2)
    synth.run(ctx)
    lid = _a_line_with_only_synth_takes(ctx)
    orig = mockb.MockASR.op_transcribe

    def garbled(self, item, lang="ja", **kw):
        res = orig(self, item, lang=lang, **kw)
        return {**res, "text": "абракадабра бубликов " * 3} if lid in Path(item["wav"]).parts else res

    monkeypatch.setattr(mockb.MockASR, "op_transcribe", garbled)
    summary = score.run(ctx)
    sel = {s.line_id: s for s in read_jsonl(ctx.paths.selection, SelectionRecord)}
    assert sel[lid].status == "review" and "a new take would not help" in sel[lid].reason and sel[lid].winner
    assert summary["review_without_regen"] == 1
    assert all(s.status != "regen" for s in sel.values()), "the other lines read their text: nothing is left for a regen round"


def test_with_the_ceiling_off_the_same_line_asks_for_a_regen_round(make_ctx, tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path / "game", sets={"score.gates.cer_max": 0.1, "score.regen_cer_max": 0})
    build(ctx, n_per=2)
    synth.run(ctx)
    lid = _a_line_with_only_synth_takes(ctx)
    orig = mockb.MockASR.op_transcribe

    def garbled(self, item, lang="ja", **kw):
        res = orig(self, item, lang=lang, **kw)
        return {**res, "text": "абракадабра бубликов " * 3} if lid in Path(item["wav"]).parts else res

    monkeypatch.setattr(mockb.MockASR, "op_transcribe", garbled)
    summary = score.run(ctx)  # regen_rounds is 1 in the test profile: the line asks for its round
    sel = {s.line_id: s for s in read_jsonl(ctx.paths.selection, SelectionRecord)}
    assert sel[lid].status == "regen" and "review_without_regen" not in summary


# ---- laughter: no words, no CER, nothing to misread ----------------------------------------------------------------------------


def test_a_laughter_only_line_is_never_hopeless_by_cer():
    quiet = _c("a.wav", utmos=2.0, spk_sim=0.5)  # fails the MOS gate; a laugh has no CER
    quiet.notes = [score.LAUGHTER_NOTE]
    quiet.model_extra["na"] = ["cer"]
    assert regen_hopeless([quiet], quiet, 0.2) is None
    # even next to a stale CER of 1.0 on the same record (what an older run left): the line is not "misread"
    stale = _c("b.wav", utmos=2.0, cer=1.0, spk_sim=0.5)
    stale.notes = [score.LAUGHTER_NOTE]
    assert regen_hopeless([stale], stale, 0.2) is None
    s = select("l1", [quiet], _sides("a.wav"), [], 2, regen_cer_max=0.2)
    assert s.status == "regen" and "misread" not in s.reason and "would not help" not in s.reason
    assert "utmos 2.00<" in s.reason and "laughter only, no words to check" in s.reason
    # the same numbers on a line with words keep the old verdict
    words = _c("c.wav", utmos=4.5, cer=0.4, spk_sim=0.5)
    assert regen_hopeless([words], words, 0.2) is not None


def test_a_garbled_transcript_of_a_laugh_does_not_send_the_line_to_review(make_ctx, tmp_path, monkeypatch):
    from vndub.manifest.io import write_jsonl

    ctx = make_ctx(tmp_path / "game", sets={"score.gates.cer_max": 0.1})
    data = build(ctx, n_per=2)
    trs = data["translations"]
    only = trs[2]
    assert only.text_tgt == "Хе-хе."
    only.flags = ["interjection", "laughter"]
    write_jsonl(ctx.paths.translation("ru"), trs)
    synth.run(ctx)
    lid = only.line_id
    orig = mockb.MockASR.op_transcribe

    def garbled(self, item, lang="ja", **kw):
        res = orig(self, item, lang=lang, **kw)
        return {**res, "text": "абракадабра бубликов " * 3} if lid in Path(item["wav"]).parts else res

    monkeypatch.setattr(mockb.MockASR, "op_transcribe", garbled)
    summary = score.run(ctx)
    rows = [r for r in read_jsonl(ctx.paths.scores(lid), CandidateScore) if r.route != "orig"]
    assert rows and all(r.metrics.get("cer") is None and "cer" not in r.gates for r in rows)
    sel = {s.line_id: s for s in read_jsonl(ctx.paths.selection, SelectionRecord)}[lid]
    assert sel.status != "regen" and "misread" not in sel.reason and "would not help" not in sel.reason and "cer" not in sel.reason
    assert "review_without_regen" not in summary

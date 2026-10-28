"""score: composite terms, gates, selection logic, overrides, calibration, prosody metrics, stage run + reweight."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from tests.genproj import build
from vndub.backends.score import prosody as pros
from vndub.backends.score.speaker_sim import calibrate_eer
from vndub.config import load_config
from vndub.manifest.io import CandidateScore, OverrideRecord, SelectionRecord, SynthSidecar, TranslationRecord, read_jsonl
from vndub.stages import genutil as gu
from vndub.stages import score, synth
from vndub.stages.score import apply_scores, composite, cer_reference, evaluate_gates, select, terms, translation_problem

CFG = load_config("configs/pipeline.yaml")
W = CFG.score.weights


def _c(wav: str, route: str = "A", backend: str = "x", **m) -> CandidateScore:
    return CandidateScore(line_id="l1", wav=wav, backend=backend, route=route, metrics=m)


def test_terms_normalization():
    t = terms({"utmos": 5.0, "pq": 10, "spk_sim": 1.2, "emo_sim": 0.8, "adv_delta": 0.0, "cer": 0.0, "dur_ratio": 1.0, "prosody": 0.5})
    assert t["utmos"] == 1.0 and t["pq"] == 1.0 and t["spk_sim"] == 1.0 and t["cer"] == 1.0 and t["dur"] == 1.0
    assert t["emo_sim"] == pytest.approx(0.75 * 0.8 + 0.25)
    assert terms({"dur_ratio": 2.0})["dur"] == pytest.approx(0.0)
    assert terms({"dur_ratio": 1.25})["dur"] == pytest.approx(1 - math.log(1.25) / math.log(2))
    assert terms({"utmos": 1.0})["utmos"] == 0.0


def test_composite_renormalizes_missing():
    full, missing = composite({"utmos": 5, "pq": 10, "spk_sim": 1, "emo_sim": 1, "cer": 0, "dur_ratio": 1, "prosody": 1}, W)
    assert full == pytest.approx(1.0) and missing == []
    part, missing = composite({"utmos": 3.0, "cer": 0.0}, W)
    # only utmos (0.5) and cer (1.0) available: (0.25*0.5 + 0.15*1) / 0.40
    assert part == pytest.approx((0.25 * 0.5 + 0.15) / 0.40, abs=1e-4)
    assert set(missing) == {"pq", "spk_sim", "emo_sim", "dur", "prosody"}
    assert composite({}, W)[0] is None


def test_gates():
    g = CFG.score.gates
    ok = evaluate_gates({"utmos": 3.5, "cer": 0.05, "dur_ratio": 1.0, "spk_sim": 0.5}, g, 0.3)
    assert ok == {"utmos": True, "cer": True, "dur": True, "spk_sim": True}
    bad = evaluate_gates({"utmos": 2.0, "cer": 0.3, "dur_ratio": 1.4, "spk_sim": 0.1}, g, 0.3)
    assert not any(bad.values())
    # missing metrics are not evaluated
    assert evaluate_gates({"utmos": 3.5}, g, 0.3) == {"utmos": True}
    # original audio is exempt from CER/duration gates
    orig = evaluate_gates({"utmos": 3.5, "cer": 1.0, "dur_ratio": 3.0}, g, 0.3, route="orig")
    assert orig["cer"] and orig["dur"]
    # interjection CER gate
    assert evaluate_gates({"cer": 0.3}, g, 0.3, cer_max=0.5)["cer"]


def _sel(cands, overrides=(), rounds=0, regen_rounds=2, created="2026-01-01T00:00:00"):
    sides = {c.wav: SynthSidecar(line_id="l1", backend=c.backend, route=c.route, k=0, round=rounds, created=created) for c in cands}
    return select("l1", cands, sides, list(overrides), regen_rounds)


def test_selection_auto_regen_review():
    good = apply_scores(_c("a.wav", utmos=4.0, cer=0.02, dur_ratio=1.0, spk_sim=0.6), CFG, 0.3, False)
    better = apply_scores(_c("b.wav", utmos=4.5, cer=0.0, dur_ratio=1.02, spk_sim=0.7), CFG, 0.3, False)
    bad = apply_scores(_c("c.wav", utmos=4.9, cer=0.5, dur_ratio=1.0, spk_sim=0.9), CFG, 0.3, False)
    s = _sel([good, better, bad])
    assert s.status == "auto" and s.winner == "b.wav" and s.scores["composite"] == better.composite
    long = apply_scores(_c("d.wav", utmos=4.0, cer=0.02, dur_ratio=1.6, spk_sim=0.6), CFG, 0.3, False)
    s = _sel([bad, long], rounds=0)
    assert s.status == "regen" and "cer" in s.reason
    s = _sel([long], rounds=2)
    assert s.status == "review" and "text_tgt_short" in s.reason and s.winner == "d.wav"
    s = _sel([])
    assert s.status == "review" and s.winner is None


def test_overrides():
    a = apply_scores(_c("a.wav", utmos=4.0, cer=0.0, dur_ratio=1.0, spk_sim=0.6), CFG, 0.3, False)
    b = apply_scores(_c("b.wav", utmos=3.2, cer=0.05, dur_ratio=1.0, spk_sim=0.5), CFG, 0.3, False)
    o = apply_scores(_c("o.wav", route="orig", backend="orig", utmos=4.5, spk_sim=1.0), CFG, 0.3, True)
    ts_old, ts_new = "2025-12-31T00:00:00", "2026-02-01T00:00:00"
    acc = OverrideRecord(line_id="l1", action="accept", payload={"candidate": "b.wav"}, ts=ts_old)
    assert _sel([a, b], [acc]).winner == "b.wav" and _sel([a, b], [acc]).status == "override"
    rej = OverrideRecord(line_id="l1", action="reject", payload={"candidate": "a.wav"}, ts=ts_old)
    assert _sel([a, b], [rej]).winner == "b.wav"
    keep = OverrideRecord(line_id="l1", action="keep_original", payload={}, ts=ts_old)
    s = _sel([a, o], [keep])
    assert s.winner == "o.wav" and s.route == "orig" and s.status == "override"
    regen_new = OverrideRecord(line_id="l1", action="regen", payload={}, ts=ts_new)
    assert _sel([a], [regen_new]).status == "regen"
    regen_old = OverrideRecord(line_id="l1", action="regen", payload={}, ts=ts_old)
    assert _sel([a], [regen_old]).status == "auto"  # already consumed by a later synth


def test_translation_problem_and_cer_reference():
    assert translation_problem(None) == "translation missing"
    assert translation_problem(TranslationRecord(line_id="x", text_tgt="", flags=["translate_missing"])) == "translation missing"
    assert translation_problem(TranslationRecord(line_id="x", text_tgt="本当", text_tts="", flags=["translate_failed"])) == "translation failed"
    head = TranslationRecord(line_id="h", text_tgt="Ага, давай.", text_tts="Ага, дав+ай.", text_plain="Ага, давай.", text_plain_short="Ага.")
    cont = TranslationRecord(line_id="c", text_tgt="Пойдёт дождь.", text_tts="Пойдёт дождь.", text_plain="Пойдёт дождь.")
    assert cer_reference(None, head, [cont]) == "Ага, давай. Пойдёт дождь."
    side = SynthSidecar(line_id="h", backend="x", route="A", k=0, text_variant="short")
    assert cer_reference(side, head, []) == "Ага."


def test_calibrate_eer():
    same = [0.7, 0.75, 0.8, 0.85, 0.9]
    diff = [0.1, 0.2, 0.25, 0.3, 0.35]
    thr, eer = calibrate_eer(same, diff)
    assert 0.35 < thr <= 0.7 and eer == 0.0
    thr2, eer2 = calibrate_eer([0.5, 0.6, 0.4], [0.45, 0.3, 0.55])
    assert 0.0 < eer2 < 1.0


def test_prosody_metrics():
    t = np.linspace(0, 1, 200)
    f0 = 200 + 40 * np.sin(2 * np.pi * 2 * t)
    assert pros.f0_corr(f0, f0 * 1.5) == pytest.approx(1.0, abs=1e-6)  # shape, not level
    rise, fall = np.linspace(150, 250, 120), np.linspace(250, 150, 120)
    assert pros.f0_corr(rise, fall) < 0  # DTW cannot undo an inverted contour
    f0u = f0.copy()
    f0u[50:70] = 0  # unvoiced frames are ignored
    assert pros.f0_corr(f0u, f0) > 0.9
    assert pros.pause_match([], 2.0, [], 3.0) == 1.0
    assert pros.pause_match([{"t": 1.0, "d": 0.2}], 2.0, [{"t": 1.5, "d": 0.3}], 3.0) == 1.0
    assert pros.pause_match([{"t": 0.2, "d": 0.2}], 2.0, [{"t": 2.5, "d": 0.2}], 3.0) == 0.0
    assert pros.prosody_score(1.0, -1.0, 0.5) == pytest.approx(0.5)
    assert pros.prosody_score(None, None, None) is None


@pytest.fixture
def scored(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path / "game")
    build(ctx, n_per=2)
    synth.run(ctx)
    summary = score.run(ctx)
    return ctx, summary


def test_score_stage_with_mocks(scored):
    ctx, summary = scored
    sel = read_jsonl(ctx.paths.selection, SelectionRecord)
    assert len(sel) == len(ctx.voiced_lines())
    assert all(s.winner for s in sel)
    assert summary["candidates"] > 0 and sum(summary["status"].values()) == len(sel)
    lid = sel[0].line_id
    rows = read_jsonl(ctx.paths.scores(lid), CandidateScore)
    m = rows[0].metrics
    for k in ("utmos", "spk_sim", "emo_sim", "adv_delta", "cer", "dur_ratio"):
        assert m.get(k) is not None, k
    assert m["cer"] == 0.0  # mock ASR returns the synthesized text
    assert rows[0].composite is not None and rows[0].gates


def test_reweight_recomputes_without_models(scored):
    ctx, _ = scored
    lid = ctx.voiced_lines()[0].line_id
    before = read_jsonl(ctx.paths.scores(lid), CandidateScore)
    ctx.cfg.score.weights = {"utmos": 1.0}
    ctx.extra["reweight"] = True
    events_before = len(ctx.models.events)
    score.run(ctx)
    after = read_jsonl(ctx.paths.scores(lid), CandidateScore)
    assert len(ctx.models.events) == events_before  # no model loaded
    for b, a in zip(before, after):
        assert a.metrics == b.metrics
        assert a.composite == pytest.approx(round((a.metrics["utmos"] - 1) / 4, 4))


def test_translation_failed_line_goes_to_review(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path / "game")
    data = build(ctx, n_per=2, interjection=False)
    trs = data["translations"]
    trs[0].flags = ["translate_failed"]
    trs[0].text_tts = ""
    trs[0].text_tgt = "テストです0。"
    from vndub.manifest.io import write_jsonl

    write_jsonl(ctx.paths.translation("ru"), trs)
    s = synth.run(ctx)
    assert s["no_translation"] == 1
    score.run(ctx)
    sel = {r.line_id: r for r in read_jsonl(ctx.paths.selection, SelectionRecord)}
    assert sel[trs[0].line_id].status == "review" and sel[trs[0].line_id].reason == "translation failed"


# ---- round 8: a gate that could not run never lets a line through, and a killed run resumes ---------------------------------


def test_select_review_when_a_gate_group_was_unavailable():
    good = apply_scores(_c("a.wav", utmos=4.0, dur_ratio=1.0, spk_sim=0.6), CFG, 0.3, False)  # no cer: the ASR was not there
    assert good.passed  # it passes on the gates that could run
    s = _sel([good])  # nothing says why: the old behaviour (a bare candidate is trusted)
    assert s.status == "auto"
    s = select("l1", [good], {"a.wav": SynthSidecar(line_id="l1", backend="x", route="A", k=0, created="2026-01-01T00:00:00")}, [], 2, {"cer": "VRAM taken by an orphan"})
    assert s.status == "review" and s.winner == "a.wav"
    assert "gate not checked" in s.reason and "cer" in s.reason and "VRAM taken by an orphan" in s.reason
    # the reason also survives in the candidate's notes (reweight has no registry state)
    good.notes = ["cer unavailable: VRAM taken by an orphan"]
    assert _sel([good]).status == "review"
    # a non-gate group (emotion) does not block, a candidate that has the metric is not affected
    ok = apply_scores(_c("b.wav", utmos=4.0, cer=0.0, dur_ratio=1.0, spk_sim=0.6), CFG, 0.3, False)
    sides = {"b.wav": SynthSidecar(line_id="l1", backend="x", route="A", k=0, created="2026-01-01T00:00:00")}
    assert select("l1", [ok], sides, [], 2, {"cer": "VRAM"}).status == "auto"
    good.notes = []
    assert select("l1", [good], {"a.wav": sides["b.wav"]}, [], 2, {"emotion": "VRAM"}).status == "auto"
    # the original recording needs no ASR check
    orig = apply_scores(_c("o.wav", route="orig", backend="orig", utmos=4.5, spk_sim=1.0), CFG, 0.3, True)
    orig.model_extra["na"] = ["cer"]
    assert select("l1", [orig], {"o.wav": sides["b.wav"]}, [], 2, {"cer": "VRAM"}).status == "auto"
    # a person's decision stays a decision
    acc = OverrideRecord(line_id="l1", action="accept", payload={"candidate": "a.wav"}, ts="2025-12-31T00:00:00")
    assert select("l1", [good], {"a.wav": sides["b.wav"]}, [acc], 2, {"cer": "VRAM"}).status == "override"


@pytest.fixture
def synthed(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path / "game")
    build(ctx, n_per=2)
    synth.run(ctx)
    return ctx


def _lines_not_won_by_orig(ctx):
    return [s for s in read_jsonl(ctx.paths.selection, SelectionRecord) if s.route != "orig"]


def test_stage_with_the_asr_unavailable_never_auto_accepts_and_says_so(synthed):
    from tests.p3_helpers import watch_models
    from vndub.gpu import VramBudgetExceeded

    ctx = synthed
    counts = watch_models(ctx, unavailable={"whisper_tiny": VramBudgetExceeded("whisper_tiny", "needs 1.5 GB, 0.3 GB free (an orphan process holds the card)")})
    summary = score.run(ctx)
    assert "orphan process" in summary["metrics_unavailable"]["cer"]
    assert ("score", "cer") in [(u.stage, u.group) for u in ctx.models.unavailable]
    mine = _lines_not_won_by_orig(ctx)
    assert mine and all(s.status == "review" and "gate not checked" in s.reason and "cer" in s.reason for s in mine)
    assert summary["lines_with_unchecked_gates"] == len(mine)
    rows = read_jsonl(ctx.paths.scores(mine[0].line_id), CandidateScore)
    assert all(r.metrics.get("cer") is None and r.metrics.get("utmos") is not None for r in rows if r.route != "orig")
    assert any(n.startswith("cer unavailable") for r in rows for n in r.notes)
    # --reweight needs no models and keeps the verdict
    ctx.extra["reweight"] = True
    score.run(ctx)
    ctx.extra["reweight"] = False
    assert all(s.status == "review" for s in _lines_not_won_by_orig(ctx))
    # the ASR is back: only the CER group runs, and the lines are accepted
    counts.restore()
    again = watch_models(ctx)
    score.run(ctx)
    assert again["transcribe"] > 0 and again["mos"] == 0 and again["embed"] == 0 and again["emotion"] == 0
    assert all(s.status == "auto" for s in _lines_not_won_by_orig(ctx))
    rows = read_jsonl(ctx.paths.scores(mine[0].line_id), CandidateScore)
    assert not any("unavailable" in n for r in rows for n in r.notes)


def test_partial_metrics_survive_a_kill_between_groups(synthed):
    from tests.p3_helpers import watch_models

    ctx = synthed

    def kill_at_emotion(op, items):
        if op == "emotion":
            raise KeyboardInterrupt("killed")

    counts = watch_models(ctx, on_call=kill_at_emotion)
    with pytest.raises(KeyboardInterrupt):
        score.run(ctx)
    assert counts["mos"] > 0 and counts["embed"] > 0
    partial = ctx.paths.scores_dir / score.PARTIAL_FILE
    assert partial.exists() and not list(ctx.paths.scores_dir.glob("*.jsonl"))  # no final record yet, only the partial one
    from vndub.manifest.io import read_json

    items = read_json(partial)["items"]
    assert items and all(it["metrics"].get("utmos") is not None and it["metrics"].get("spk_sim") is not None for it in items.values())
    assert all(it["metrics"].get("cer") is None for it in items.values())
    counts.restore()
    again = watch_models(ctx)
    summary = score.run(ctx)
    assert again["mos"] == 0 and again["embed"] == 0, "a finished group was computed again"
    assert again["emotion"] > 0 and again["transcribe"] > 0
    assert summary["resumed_candidates"] == len(items) and not partial.exists()
    for s in read_jsonl(ctx.paths.selection, SelectionRecord):
        rows = read_jsonl(ctx.paths.scores(s.line_id), CandidateScore)
        assert all(r.metrics.get(k) is not None for r in rows for k in ("utmos", "spk_sim", "emo_sim", "cer", "dur_ratio") if not (k == "cer" and "cer" in (r.model_extra.get("na") or []))), s.line_id


def test_partial_metrics_are_written_per_chunk_inside_a_group(make_ctx, tmp_path, monkeypatch):
    from tests.p3_helpers import watch_models

    ctx = make_ctx(tmp_path / "game")
    build(ctx, n_per=6)
    synth.run(ctx)
    monkeypatch.setattr(score, "PARTIAL_EVERY_S", 0.0)  # every chunk is a checkpoint
    state = {"n": 0}

    def kill_second_asr_chunk(op, items):
        if op == "transcribe":
            state["n"] += 1
            if state["n"] == 2:
                raise KeyboardInterrupt("killed in the middle of the CER group")

    counts = watch_models(ctx, on_call=kill_second_asr_chunk)
    with pytest.raises(KeyboardInterrupt):
        score.run(ctx)
    n_asr_first_run = counts["transcribe"]
    assert n_asr_first_run == 16  # one chunk went through before the kill (the second never started)
    counts.restore()
    again = watch_models(ctx)
    score.run(ctx)
    asr_needed = sum(1 for s in read_jsonl(ctx.paths.selection, SelectionRecord) for r in read_jsonl(ctx.paths.scores(s.line_id), CandidateScore) if "cer" not in (r.model_extra.get("na") or []))
    assert asr_needed > 16 and again["transcribe"] == asr_needed - 16, "the chunk that was finished before the kill was transcribed again"
    assert again["mos"] == 0


def test_emotion_group_skips_candidates_with_an_embedding_file(synthed):
    from tests.p3_helpers import watch_models

    ctx = synthed
    score.run(ctx)
    assert list(ctx.paths.scores_dir.rglob("*.e2v.npy"))
    for p in ctx.paths.scores_dir.glob("*.jsonl"):  # no final records, no partial one: everything is to be computed again ...
        p.unlink()
    (ctx.paths.scores_dir / score.PARTIAL_FILE).unlink(missing_ok=True)
    counts = watch_models(ctx)
    score.run(ctx)
    assert counts["mos"] > 0 and counts["transcribe"] > 0
    assert counts["emotion"] == 0  # ... except what the embedding files already hold
    for s in read_jsonl(ctx.paths.selection, SelectionRecord):
        assert all(r.metrics.get("emo_sim") is not None for r in read_jsonl(ctx.paths.scores(s.line_id), CandidateScore))


def test_score_progress_total_is_every_unit_and_never_shrinks(synthed):
    ctx = synthed
    events: list[tuple[int, int]] = []
    ctx.extra["progress_cb"] = lambda stage, done, total, item: events.append((done, total))
    score.run(ctx)
    total = events[0][1]
    assert events[0][0] == 0 and total > 0 and events[-1] == (total, total)
    assert {t for _, t in events} == {total} and [d for d, _ in events] == sorted(d for d, _ in events)
    # a rerun that finds everything: n / n at once, the same total
    events.clear()
    score.run(ctx)
    assert events[0] == (total, total) and {t for _, t in events} == {total}
    # a run that resumes after a kill reports the finished units first and the full total
    from tests.p3_helpers import watch_models

    for p in ctx.paths.scores_dir.glob("*.jsonl"):
        p.unlink()
    (ctx.paths.scores_dir / score.PARTIAL_FILE).unlink(missing_ok=True)

    def kill(op, items):
        if op == "transcribe":
            raise KeyboardInterrupt()

    counts = watch_models(ctx, on_call=kill)
    events.clear()
    with pytest.raises(KeyboardInterrupt):
        score.run(ctx)
    counts.restore()
    events.clear()
    score.run(ctx)
    assert events[0][1] == total and 0 < events[0][0] < total  # most units were done before the kill
    assert {t for _, t in events} == {total} and events[-1] == (total, total)


def test_score_progress_class():
    seen: list[tuple[int, int]] = []

    class C:
        def progress(self, done, total, item=None):
            seen.append((done, total))

    p = score._Progress(C(), total=10, done=4)
    assert seen == [(4, 10)]  # the cached units are reported once, up front
    p.chunk([{"id": "orig|l1"}, {"id": "l1|a"}, {"id": "l1|b"}])  # reference items are not units
    assert seen[-1] == (6, 10)
    p.add(100)
    assert seen[-1] == (10, 10)  # never past the total
    seen.clear()
    score._Progress(C(), total=0, done=0)
    assert seen == []
    q = score._Progress(C(), total=5, done=9)
    assert q.done == 5


# ---- laughter: the CER is about the words ---------------------------------------------------------------------------------------


def test_cer_words_takes_the_laughter_out_of_both_sides():
    from vndub.backends.score.cer import cer

    assert score.cer_words("Haha, no way!", "no way") == ("no way!", "no way")
    assert score.cer_words("Ха-ха! Ты серьёзно?", "ха-ха ты серьёзно") == ("Ты серьёзно?", "ты серьёзно")
    assert score.cer_words("Хе-хе.", "хехе") == ("", "")
    assert score.cer_words("Ты что, (смеётся) издеваешься?", "ты что издеваешься")[0] == "Ты что, издеваешься?"
    assert score.cer_words("Привет, это тест.", "привет это тест") == ("Привет, это тест.", "привет это тест")  # no laughter: untouched
    assert score.cer_words("", "") == ("", "")
    # the laugh is what the ASR drops or garbles: counted as text it fails the gate, over the words it is a clean read
    assert cer("Ха-ха, ты серьёзно?", "ты серьёзно") > CFG.score.gates.cer_max
    assert cer(*score.cer_words("Ха-ха, ты серьёзно?", "ты серьёзно")) == 0.0
    assert cer(*score.cer_words("Haha, no way!", "haha no way")) == 0.0
    # a laugh the ASR heard as one stray syllable is not a recognizable laugh: the plain reading is the floor
    assert cer(*score.cer_words("Ха-ха, да ну!", "ха да ну")) > cer("Ха-ха, да ну!", "ха да ну")
    assert score.cer_over_words("Ха-ха, да ну!", "ха да ну") == pytest.approx(cer("Ха-ха, да ну!", "ха да ну"))
    assert score.cer_over_words("Ха-ха, да ну!", "да ну") == 0.0 and score.cer_over_words("Haha, no way!", "no way") == 0.0
    assert score.cer_over_words("Привет, это тест.", "привет это тест") == 0.0
    assert score.cer_over_words("Привет, это тест.", "совсем другое") == pytest.approx(cer("Привет, это тест.", "совсем другое"))


def test_a_laughter_line_has_no_cer_gate_and_can_be_accepted():
    # no cer metric, "cer" in na (the stage's marking): the gate does not exist, nothing is unchecked
    c = apply_scores(_c("a.wav", utmos=4.0, dur_ratio=1.0, spk_sim=0.6), CFG, 0.3, True)
    c.notes = [score.LAUGHTER_NOTE]
    c.model_extra["na"] = ["cer"]
    assert c.passed and "cer" not in c.gates and score.unchecked_gates(c) == {}
    assert score.unchecked_gates(c, {"cer": "ASR gone"}) == {}  # not even when the ASR is unavailable: it was never needed
    s = _sel([c])
    assert s.status == "auto" and "cer" not in s.reason
    # failing the speaker or the MOS gate it says so plainly, and says nothing about the words
    quiet = apply_scores(_c("b.wav", utmos=2.0, dur_ratio=1.0, spk_sim=0.1), CFG, 0.3, True)
    quiet.notes = [score.LAUGHTER_NOTE]
    quiet.model_extra["na"] = ["cer"]
    for rounds, status in ((0, "regen"), (2, "review")):
        s = _sel([quiet], rounds=rounds)
        assert s.status == status and "utmos 2.00<" in s.reason and "spk_sim 0.10<0.30" in s.reason
        assert "laughter only, no words to check" in s.reason and "misread" not in s.reason and "cer" not in s.reason
    # the same failures on a line with words do not get the laughter wording
    words = apply_scores(_c("w.wav", utmos=2.0, dur_ratio=1.0, spk_sim=0.1), CFG, 0.3, False)
    assert "laughter" not in _sel([words]).reason


def _laughter_ctx(make_ctx, tmp_path):
    """genproj's project with line 0 = laughter and words, and its own "Хе-хе." line (laughter only) flagged as translate does."""
    from vndub.manifest.io import write_jsonl

    ctx = make_ctx(tmp_path / "game")
    data = build(ctx, n_per=2)
    trs = data["translations"]
    words, only = trs[0], trs[2]
    assert only.text_tgt == "Хе-хе."
    words.text_tgt = words.text_tgt_short = words.text_tts = words.text_plain = "Ха-ха, ты серьёзно?"
    words.flags = ["laughter"]
    only.flags = ["interjection", "laughter"]
    write_jsonl(ctx.paths.translation("ru"), trs)
    synth.run(ctx)
    return ctx, words.line_id, only.line_id


def _tts_rows(ctx, lid):
    return [r for r in read_jsonl(ctx.paths.scores(lid), CandidateScore) if r.route != "orig"]


def test_stage_laughter_only_line_is_not_sent_to_the_asr_and_is_not_a_cer_failure(make_ctx, tmp_path, monkeypatch):
    from vndub.backends import mock as mockb

    ctx, words_id, only_id = _laughter_ctx(make_ctx, tmp_path)
    asked: list[str] = []
    orig = mockb.MockASR.op_transcribe

    def deaf_on_the_laugh(self, item, lang="ja", **kw):
        asked.append(item["wav"])
        res = orig(self, item, lang=lang, **kw)
        return {**res, "text": ""} if only_id in Path(item["wav"]).parts else res  # what an ASR hears in a laugh

    monkeypatch.setattr(mockb.MockASR, "op_transcribe", deaf_on_the_laugh)
    summary = score.run(ctx)
    rows = _tts_rows(ctx, only_id)
    assert rows
    for r in rows:
        assert r.metrics.get("cer") is None and "cer" in r.model_extra["na"] and score.LAUGHTER_NOTE in r.notes
        assert "cer" not in r.gates and not any(n.startswith("cer failed") for n in r.notes)
        assert r.metrics.get("utmos") is not None and r.metrics.get("spk_sim") is not None  # the other gates are measured
    assert asked and not any(only_id in Path(w).parts for w in asked)  # nothing to transcribe: the ASR is not asked
    sel = {s.line_id: s for s in read_jsonl(ctx.paths.selection, SelectionRecord)}[only_id]
    assert sel.status == "auto" and "gate not checked" not in sel.reason and "cer" not in sel.reason, sel
    assert "lines_with_unchecked_gates" not in summary and "review_without_regen" not in summary
    # the ordinary line next to it is still checked word by word
    assert all(r.metrics.get("cer") == 0.0 for r in _tts_rows(ctx, words_id))


def test_stage_cer_ignores_a_laugh_the_asr_dropped(make_ctx, tmp_path, monkeypatch):
    from vndub.backends import mock as mockb
    from vndub.backends.score.cer import cer

    ctx, words_id, _ = _laughter_ctx(make_ctx, tmp_path)
    orig = mockb.MockASR.op_transcribe

    def drops_the_laugh(self, item, lang="ja", **kw):
        res = orig(self, item, lang=lang, **kw)
        return {**res, "text": "Ты серьёзно?"} if words_id in Path(item["wav"]).parts else res

    monkeypatch.setattr(mockb.MockASR, "op_transcribe", drops_the_laugh)
    assert cer("Ха-ха, ты серьёзно?", "Ты серьёзно?") > CFG.score.gates.cer_max  # the old reading: a failed gate
    score.run(ctx)
    rows = _tts_rows(ctx, words_id)
    assert rows and all(r.metrics["cer"] == 0.0 and r.gates["cer"] for r in rows)
    assert all(r.model_extra["cer_ref"] == "ты серьёзно?" and r.model_extra["asr_text"] == "Ты серьёзно?" for r in rows)
    assert all(r.model_extra["cer_rule"] == score.CER_RULE and "cer" not in (r.model_extra.get("na") or []) for r in rows)
    sel = {s.line_id: s for s in read_jsonl(ctx.paths.selection, SelectionRecord)}[words_id]
    assert sel.status == "auto"


def test_stage_words_with_an_empty_or_laugh_only_transcript_stay_an_unchecked_gate(make_ctx, tmp_path, monkeypatch):
    from vndub.backends import mock as mockb

    ctx, words_id, _ = _laughter_ctx(make_ctx, tmp_path)
    orig = mockb.MockASR.op_transcribe
    heard = {"text": ""}

    def hears(self, item, lang="ja", **kw):
        res = orig(self, item, lang=lang, **kw)
        return {**res, "text": heard["text"]} if words_id in Path(item["wav"]).parts else res

    monkeypatch.setattr(mockb.MockASR, "op_transcribe", hears)
    for text in ("", "ха-ха-ха"):  # nothing at all / only the laugh: the words were not heard either way
        heard["text"] = text
        summary = score.run(ctx)
        rows = _tts_rows(ctx, words_id)
        assert rows and all(r.metrics.get("cer") is None for r in rows), text
        assert all("cer failed: the ASR heard no words in this take" in r.notes for r in rows), text
        sel = {s.line_id: s for s in read_jsonl(ctx.paths.selection, SelectionRecord)}[words_id]
        assert sel.status == "review" and "gate not checked" in sel.reason and "heard no words" in sel.reason, text
        assert summary["lines_with_unchecked_gates"] >= 1


def test_stage_recomputes_a_cer_stored_before_laughter_was_taken_out(make_ctx, tmp_path):
    from vndub.manifest.io import write_jsonl

    ctx, words_id, only_id = _laughter_ctx(make_ctx, tmp_path)
    score.run(ctx)
    # a project scored by the old rule: laughter counted as text (a CER on the laughter line, no cer_rule, no marking)
    for lid, cer_value in ((words_id, 0.6), (only_id, 0.7)):
        rows = read_jsonl(ctx.paths.scores(lid), CandidateScore)
        for r in rows:
            if r.route == "orig":
                continue
            r.metrics["cer"] = cer_value
            r.model_extra["asr_text"] = "стёртый"
            r.model_extra.pop("cer_rule", None)
            r.model_extra.pop("cer_ref", None)
            r.model_extra["na"] = []
            r.notes = [n for n in r.notes if n != score.LAUGHTER_NOTE]
        write_jsonl(ctx.paths.scores(lid), rows)
    score.run(ctx)
    for r in _tts_rows(ctx, words_id):
        assert r.metrics["cer"] == 0.0 and r.model_extra["cer_rule"] == score.CER_RULE and r.model_extra["asr_text"] != "стёртый"
    for r in _tts_rows(ctx, only_id):
        assert r.metrics.get("cer") is None and "cer" in r.model_extra["na"] and score.LAUGHTER_NOTE in r.notes
        assert "asr_text" not in r.model_extra and "cer" not in r.gates


# ---- release 1.0.0 review: ranking among passing takes, nothing to judge by, the speaker gate, --force ----------------------------------


def test_select_prefers_a_fully_checked_take_over_a_higher_composite_that_skipped_a_gate():
    """A take without a CER has its composite renormalised without that term and can outrank a take that passed everything."""
    blind = apply_scores(_c("a.wav", utmos=4.5, dur_ratio=1.0, spk_sim=0.95), CFG, 0.3, False)  # the ASR heard no words in it
    blind.notes = ["cer failed: the ASR heard no words in this take"]
    full = apply_scores(_c("b.wav", utmos=3.6, cer=0.05, dur_ratio=1.0, spk_sim=0.5), CFG, 0.3, False)
    assert blind.passed and full.passed and blind.composite > full.composite
    s = _sel([blind, full])
    assert s.status == "auto" and s.winner == "b.wav" and s.reason == "best of 2 passing"
    # the same through the run-wide reason (the group was unavailable for one take only is not a thing, but the order must not matter)
    assert _sel([full, blind]).winner == "b.wav"
    # every passing take has an unchecked gate: a person decides, on the best of them
    blind2 = apply_scores(_c("c.wav", utmos=4.0, dur_ratio=1.0, spk_sim=0.9), CFG, 0.3, False)
    blind2.notes = ["cer failed: the ASR heard no words in this take"]
    s = _sel([blind2, blind])
    assert s.status == "review" and s.winner == "a.wav" and "gate not checked" in s.reason and "cer" in s.reason
    # two fully checked takes: the composite decides, as before
    better = apply_scores(_c("d.wav", utmos=4.4, cer=0.0, dur_ratio=1.0, spk_sim=0.8), CFG, 0.3, False)
    assert _sel([full, better, blind]).winner == "d.wav"


def test_select_goes_to_review_when_no_gate_could_be_evaluated():
    only_emotion = apply_scores(_c("a.wav", emo_sim=0.5), CFG, 0.3, False)  # every gate group unavailable, no reference audio
    assert only_emotion.gates == {} and not only_emotion.passed
    s = _sel([only_emotion])  # (a regen round of takes that cannot be judged either would be wasted)
    assert s.status == "review" and s.winner == "a.wav" and "no gate could be evaluated" in s.reason
    s = select(
        "l1", [only_emotion], {"a.wav": SynthSidecar(line_id="l1", backend="x", route="A", k=0, created="2026-01-01T00:00:00")}, [], 2,
        {"mos": "no weights", "cer": "VRAM", "speaker": "no embedder"},
    )
    assert s.status == "review" and "no gate could be evaluated" in s.reason and "metrics unavailable" in s.reason and "VRAM" in s.reason
    # one gate that ran and failed keeps the regen round
    failing = apply_scores(_c("b.wav", utmos=2.0, emo_sim=0.5), CFG, 0.3, False)
    assert failing.gates == {"utmos": False}
    assert _sel([failing]).status == "regen"
    # a person's decision is still a decision
    acc = OverrideRecord(line_id="l1", action="accept", payload={"candidate": "a.wav"}, ts="2025-12-31T00:00:00")
    assert _sel([only_emotion], [acc]).status == "override"


def test_the_summary_counts_a_line_with_nothing_to_judge_by_as_unchecked(synthed):
    from tests.p3_helpers import watch_models
    from vndub.gpu import VramBudgetExceeded

    ctx = synthed
    gone = {n: VramBudgetExceeded(n, "an orphan process holds the card") for n in ("utmos22", "ecapa", "whisper_tiny")}
    ctx.cfg.score.aesthetics = ""
    watch_models(ctx, unavailable=gone)
    summary = score.run(ctx)
    sel = read_jsonl(ctx.paths.selection, SelectionRecord)
    assert sel and all(s.status == "review" for s in sel)
    assert summary["lines_with_unchecked_gates"] == len(sel)
    assert not any(s.status == "regen" for s in sel)  # not a single new take is synthesized for what cannot be judged


def test_a_take_with_no_reference_voice_to_compare_with_is_an_unchecked_speaker_gate(synthed, monkeypatch):
    ctx = synthed
    real = gu.embed_paths
    # every reference embedding (voicebank clips, performer clips, the original line) failed: only the takes were embedded
    monkeypatch.setattr(gu, "embed_paths", lambda c, b, jobs: {k: v for k, v in real(c, b, jobs).items() if k.startswith("cand|")})
    summary = score.run(ctx)
    sel = read_jsonl(ctx.paths.selection, SelectionRecord)
    assert sel and all(s.status == "review" and "gate not checked" in s.reason and "speaker similarity" in s.reason for s in sel)
    assert all("no reference voice to compare with" in s.reason for s in sel)
    assert summary["lines_with_unchecked_gates"] == len(sel)
    rows = read_jsonl(ctx.paths.scores(sel[0].line_id), CandidateScore)
    assert rows and all(r.metrics.get("spk_sim") is None and "speaker failed: no reference voice to compare with" in r.notes for r in rows)
    # the references are back: the same takes are measured and accepted (the stale note is not kept)
    monkeypatch.setattr(gu, "embed_paths", real)
    score.run(ctx)
    assert all(s.status != "review" or "gate not checked" not in s.reason for s in read_jsonl(ctx.paths.selection, SelectionRecord))
    rows = read_jsonl(ctx.paths.scores(sel[0].line_id), CandidateScore)
    assert all(r.metrics.get("spk_sim") is not None and not any(n.startswith("speaker failed") for n in r.notes) for r in rows)


def test_a_reference_clip_that_is_gone_does_not_end_the_stage(make_ctx, tmp_path):
    """``embed_paths`` stats the wav even when the cached .npy exists; a clip deleted after the first run must not raise."""
    from vndub.stages import voicebank

    ctx = make_ctx(tmp_path / "game")
    build(ctx, n_per=2)
    voicebank.run(ctx)
    synth.run(ctx)
    score.run(ctx)
    gone = 0
    for sid in {l.speaker_id for l in ctx.voiced_lines()}:
        for r in gu.load_refs(ctx, sid):
            ctx.paths.abs(r.path).unlink()
            gone += 1
    assert gone
    ctx.force = True  # every metric is computed again, the speaker group included
    score.run(ctx)
    rows = read_jsonl(ctx.paths.scores(ctx.voiced_lines()[0].line_id), CandidateScore)
    assert rows and all(r.metrics.get("spk_sim") is not None for r in rows)  # measured against the original line instead


def test_force_recomputes_every_metric_and_reweight_still_asks_no_model(scored):
    from tests.p3_helpers import watch_models

    ctx, _ = scored
    counts = watch_models(ctx)
    score.run(ctx)
    assert counts["mos"] == 0 and counts["transcribe"] == 0 and counts["embed"] == 0  # nothing changed: the stored records are reused
    ctx.force = True
    score.run(ctx)
    assert counts["mos"] > 0 and counts["transcribe"] > 0  # (the embedding files on disk are keyed by the audio's mtime: still reused)
    # --reweight never looks at a model, force or not
    before = dict(counts)
    ctx.extra["reweight"] = True
    score.run(ctx)
    assert dict(counts) == before

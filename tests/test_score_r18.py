"""Round 18: the CER gate failed good audio (157 of 160 lines on a real run, an independent check gave a mean CER of 0.014).

* the two sides of the CER are made comparable (numbers by value, laugh syllables, Latin words, short lines);
* a take the ASR heard no words in no longer passes next to a measured take;
* without a passing take, the provisional winner is the take closest to passing, not the best composite;
* a take of an older text (the translation was edited) is skipped by score.
"""

from __future__ import annotations

import pytest

from tests.genproj import build
from vndub.config import load_config
from vndub.manifest.io import CandidateScore, SelectionRecord, SynthSidecar, TranslationRecord, read_jsonl, write_jsonl
from vndub.stages import score, synth
from vndub.stages.score import apply_scores, cer_over_words, select
from vndub.text import ru

CFG = load_config("configs/pipeline.yaml")
THR = 0.25


def _plain(text: str) -> str:
    return ru.prepare_tts_text(text)[1]  # what translate stores as text_plain


# ---- 1. comparable sides ------------------------------------------------------------------------------------------------

# (reference, ASR transcript): the reviewer's table; before round 18 these scored 0.306 / 0.788 / 0.500 / 0.542 / 0.088 / 0.273
CER_TABLE = [
    ("Мне шестнадцать лет.", "Мне 16 лет."),
    (_plain("В 2024 году."), "В 2024 году."),
    (_plain("Встретимся в 10:30."), "Встретимся в 10:30."),
    (_plain("Это 100% правда."), "Это 100% правда."),
    (_plain("Я ждала тебя 3 часа, между прочим!"), "Я ждала тебя 3 часа, между прочим!"),
    ("Ха-ха, да ну!", "ха да ну"),
    # a declined reference against the ASR's digits (which expand to the nominative): compared by value
    ("С тремя друзьями.", "с 3 друзьями"),
    ("Это было в девяносто девятом году.", "это было в 99 году"),
    ("Мне нужно двести рублей.", "мне нужно 200 рублей"),
    ("Две тысячи двадцать четвёртый год.", "2024 год"),
]


@pytest.mark.parametrize("ref,hyp", CER_TABLE)
def test_cer_table_after_the_fix(ref, hyp):
    assert cer_over_words(ref, hyp) <= 0.01, (ref, hyp, score.cer_texts(ref, hyp))


def test_a_wrong_number_still_costs():
    assert cer_over_words("Мне шестнадцать лет.", "Мне 61 лет.") > 0
    assert cer_over_words("Мне шестнадцать лет.", "мне сорок лет") > 0


def test_numbers_to_digits():
    from vndub.backends.score.cer import numbers_to_digits

    assert numbers_to_digits("в две тысячи двадцать четвертом году") == "в 2024 году"
    assert numbers_to_digits("три два один") == "3 2 1"  # a countdown is three numbers
    assert numbers_to_digits("сто двадцать пять") == "125"
    assert numbers_to_digits("пять двадцать") == "5 20"
    assert numbers_to_digits("стол стоит") == "стол стоит"


def test_latin_words_transliterated_when_available(monkeypatch):
    fake = {"Patchouli": "Пачули", "Knowledge": "Нолидж"}
    monkeypatch.setattr(ru, "translit_latin", lambda t: " ".join(fake.get(w.strip(".,!"), w) for w in t.split()), raising=False)
    assert cer_over_words("Это Patchouli Knowledge.", "это пачули нолидж") == 0.0


def test_latin_words_dropped_without_translit(monkeypatch):
    monkeypatch.delattr(ru, "translit_latin", raising=False)
    assert score.comparable("Это Kevin, он друг.") == "это он друг"
    assert score.comparable("Kevin!") == "kevin"  # all Latin: kept, there would be nothing left
    assert cer_over_words("Это Kevin, он друг.", "это kevin он друг") == 0.0
    from vndub.backends.score.cer import cer

    assert cer("Это Patchouli Knowledge.", "это пачули нолидж") > 0.7
    assert cer_over_words("Это Patchouli Knowledge.", "это пачули нолидж") == 0.0  # the name's Cyrillic words leave too
    assert cer_over_words("Это Patchouli, правда.", "это пачули кривда") > 0  # but no more than the name's count


def test_english_target_is_not_russified():
    assert score.comparable("I am 16.", "en") == "i am 16"
    assert cer_over_words("No way!", "no way", lang="en") == 0.0


def _c(name: str, notes=(), ref: str | None = None, **m) -> CandidateScore:
    c = CandidateScore(line_id="L", wav=f"synth/L/{name}.wav", backend="qwen3_tts", route="P", metrics=dict(m), notes=list(notes))
    setattr(c, "na", [])
    if ref is not None:
        setattr(c, "cer_ref", ref)
    return apply_scores(c, CFG, THR, False)


def test_short_reference_tolerates_one_wrong_character():
    good = dict(utmos=3.8, pq=6.0, spk_sim=0.6, dur_ratio=1.0)
    assert _c("t0", ref="Да.", cer=0.5, **good).passed  # 2 letters, one wrong
    assert not _c("t0", ref="Да.", cer=1.0, **good).passed  # nothing right
    assert not _c("t0", ref="Это совсем длинная реплика.", cer=0.15, **good).passed  # the normal gate
    assert _c("t0", ref="Ну ладно.", cer=0.1, **good).passed
    assert score.cer_gate(0.1, "Да.", 8) == 0.5 and score.cer_gate(0.1, "Да.", 0) == 0.1 and score.cer_gate(0.1, "Хорошо, иду", 8) == 0.1


# ---- 2. the take nobody heard words in -----------------------------------------------------------------------------------

DEAF = [score.NO_WORDS_NOTE]


def _sides(cs, rnd):
    return {c.wav: SynthSidecar(line_id="L", backend="qwen3_tts", route="P", k=0, round=rnd, created="2026-11-01T00:00:00") for c in cs}


def _select(cs, rnd=0):
    return select("L", cs, _sides(cs, rnd), [], CFG.score.regen_rounds, None, None, 0, regen_cer_max=max(CFG.score.regen_cer_max, CFG.score.gates.cer_max))


def test_unheard_take_does_not_win_over_measured_ones():
    cs = [
        _c("t0", notes=DEAF, utmos=3.4, pq=6.0, spk_sim=0.6),
        _c("t1", utmos=3.6, pq=6.2, spk_sim=0.65, cer=0.12),
        _c("t2", utmos=3.5, pq=6.1, spk_sim=0.62, cer=0.15),
    ]
    assert cs[0].passed  # on its own gates
    s = _select(cs, rnd=0)
    assert s.status == "regen" and s.winner.endswith("t1.wav"), s
    score.demote_unheard(cs)  # what run() writes into the scores
    assert not cs[0].passed and "heard no words" in cs[0].model_extra["failures"]
    s = _select(cs, rnd=CFG.score.regen_rounds)
    assert s.status == "review" and s.winner.endswith("t1.wav"), s


def test_unheard_take_loses_to_a_passing_one_and_ranks_last():
    cs = [_c("t0", notes=DEAF, utmos=4.8, pq=8.0, spk_sim=0.9), _c("t1", utmos=3.6, pq=6.2, spk_sim=0.65, cer=0.05)]
    s = _select(cs)
    assert s.status == "auto" and s.winner.endswith("t1.wav")


def test_every_take_unheard_keeps_the_review():
    cs = [_c("t0", notes=DEAF, utmos=3.4, spk_sim=0.6), _c("t1", notes=DEAF, utmos=3.9, spk_sim=0.7)]
    s = _select(cs)
    assert s.status == "review" and "gate not checked" in s.reason and s.winner.endswith("t1.wav")


# ---- 3. the provisional winner -------------------------------------------------------------------------------------------


def test_no_pass_prefers_the_take_closest_to_passing():
    cs = [_c("t0", utmos=4.4, pq=7.0, spk_sim=0.72, cer=0.60), _c("t1", utmos=3.4, pq=6.0, spk_sim=0.60, cer=0.11)]
    assert cs[0].composite > cs[1].composite
    s = _select(cs, rnd=CFG.score.regen_rounds)
    assert s.status == "review" and s.winner.endswith("t1.wav"), s


def test_no_pass_fewer_failed_gates_first():
    one = _c("t0", utmos=2.0, pq=6.0, spk_sim=0.6, cer=0.02)  # fails utmos only, by a lot
    two = _c("t1", utmos=3.4, pq=6.0, spk_sim=0.2, cer=0.11)  # fails speaker and cer, by a little
    s = _select([two, one], rnd=CFG.score.regen_rounds)
    assert s.winner.endswith("t0.wav")


# ---- 4. stale takes ------------------------------------------------------------------------------------------------------


def _tr(lid="L", text="Привет, как дела?", short="Как дела?"):
    return TranslationRecord(line_id=lid, text_tgt=text, text_tts=text, text_plain=text, text_tgt_short=short, text_tts_short=short, text_plain_short=short)


def _side(plain, variant="full", backend="qwen3_tts", route="P", **ex):
    return SynthSidecar(line_id="L", backend=backend, route=route, k=0, text_variant=variant, text_plain=plain, **ex)


def test_stale_take_shapes():
    tr, cont = _tr(), TranslationRecord(line_id="C", text_tgt="Пойдём.", text_tts="Пойдём.", text_plain="Пойдём.")
    st = score.stale_take
    assert not st(_side("Привет, как дела?"), tr, [], None)
    assert not st(_side("Прив+ет, как дел+а"), tr, [], None)  # stress / punctuation do not matter
    assert not st(_side("Как дела?", "short"), tr, [], None)
    assert not st(_side("Привет, как дела? Пойдём.", "full"), tr, [cont], None)  # sustain-joined
    assert not st(_side("Как дела? Пойдём.", "short"), tr, [cont], None)
    assert not st(_side("Привет, как дела?", "interjection"), tr, [], None)
    assert not st(_side("Свой текст.", "override"), tr, [], "Свой текст.")  # the current set_text
    assert not st(_side("", "orig", backend="orig", route="orig"), tr, [], None)
    assert not st(_side("", "sound", backend="sample"), tr, [], None)
    assert not st(_side("Хр-р-р", "full", sound={"kind": "snore"}), tr, [], None)  # a sound stand-in: rendered reference
    assert not st(_side("Совсем старый текст."), None, [], None)  # nothing to compare with
    assert not st(SynthSidecar(line_id="L", backend="x", route="P", k=0), tr, [], None)  # an old sidecar without text_plain
    # an older text
    assert st(_side("Совсем старый текст."), tr, [], None)
    assert st(_side("Привет, как дела?", "override"), tr, [], "Новый текст.") is False  # still the translation: allowed
    assert st(_side("Старый свой текст.", "override"), tr, [], "Новый текст.")


def test_stage_skips_takes_of_an_older_text(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path / "game")
    data = build(ctx, n_per=2)
    synth.run(ctx)
    trs = data["translations"]
    lid = trs[0].line_id
    trs[0].text_tgt = trs[0].text_tts = trs[0].text_plain = trs[0].text_tgt_short = "Совсем другой текст реплики."
    trs[0].text_tts_short = trs[0].text_tgt_short
    if trs[0].model_extra:
        trs[0].model_extra.pop("text_plain_short", None)
    write_jsonl(ctx.paths.translation("ru"), trs)
    summary = score.run(ctx)
    assert summary.get("stale_candidates_skipped", 0) >= 1
    sel = {s.line_id: s for s in read_jsonl(ctx.paths.selection, SelectionRecord)}[lid]
    rows = read_jsonl(ctx.paths.scores(lid), CandidateScore)
    assert all(r.route == "orig" for r in rows), rows  # no take of the old text was scored
    assert sel.winner is None or "orig" in sel.winner
    # the other lines keep all their takes
    other = trs[1].line_id
    assert any(r.route != "orig" for r in read_jsonl(ctx.paths.scores(other), CandidateScore))


def test_stage_keeps_set_text_takes(make_ctx, tmp_path):
    from vndub.review.app import write_override

    ctx = make_ctx(tmp_path / "game")
    data = build(ctx, n_per=2)
    lid = data["translations"][1].line_id
    write_override(ctx.paths, lid, "set_text", {"text": "Исправленный пользователем текст."})
    synth.run(ctx)
    summary = score.run(ctx)
    assert "stale_candidates_skipped" not in summary
    assert any(r.route != "orig" for r in read_jsonl(ctx.paths.scores(lid), CandidateScore))

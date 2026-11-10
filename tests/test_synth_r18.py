"""Round 18 synth fixes: superseded takes are purged, candidates are keyed without the synth section, a backend that fails
every item is restarted once and then stops the stage, load errors are "unavailable", review texts get translate's
preparation, and a NaN in an adapter reply is parsed."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.genproj import build, speechlike
from tests.p3_helpers import add_unvoiced
from vndub import cli
from vndub.backends import mock
from vndub.backends.base import BackendUnavailable
from vndub.backends.runner import AdapterAborted, AdapterError, AdapterTimeout, _lenient_reply
from vndub.config import load_config
from vndub.manifest.io import TranslationRecord, read_json, read_jsonl, write_jsonl
from vndub.review.app import write_override
from vndub.stages import genutil as gu
from vndub.stages import synth
from vndub.stages.common import StageError, run_stage
from tests.conftest import TEST_CONFIG


def _takes(ctx, lid):
    return {wav.name: side for wav, side in gu.candidate_files(ctx, lid)}


def _regen_project(make_ctx, tmp_path, **sets):
    """A project after round 0 and one regen round (every line fails the utmos gate), so k=100.. takes exist."""
    ctx = make_ctx(tmp_path / "game", sets={"score.regen_rounds": 1, "score.gates.utmos_min": 4.99, **sets})
    build(ctx, n_per=2)
    return ctx


def _round(ctx):
    run_stage("synth", ctx)
    run_stage("score", ctx)
    cli._regen_loop(ctx)


# ---- 1: takes that no longer describe the line ---------------------------------------------------------------------------


def test_a_translation_edit_purges_the_regen_round_takes_of_the_old_text(make_ctx, tmp_path):
    ctx = _regen_project(make_ctx, tmp_path)
    _round(ctx)
    lid = ctx.voiced_lines()[0].line_id
    assert any(n.endswith("_100.wav") for n in _takes(ctx, lid))
    p = ctx.paths.translation("ru")
    trs = read_jsonl(p, TranslationRecord)
    for t in trs:
        if t.line_id == lid:
            t.text_tgt = t.text_tts = t.text_plain = t.text_tgt_short = "Совсем другой текст реплики."
    write_jsonl(p, trs)
    run_stage("synth", ctx)
    takes = _takes(ctx, lid)
    assert takes and all(s.model_extra["text_plain"] == "Совсем другой текст реплики." for s in takes.values())
    assert not any(n.endswith("_100.wav") for n in takes)
    other = ctx.voiced_lines()[1].line_id
    assert any(n.endswith("_100.wav") for n in _takes(ctx, other)), "an unchanged line keeps its regen-round takes"


def test_set_text_purges_the_takes_of_the_translation(make_ctx, tmp_path):
    ctx = _regen_project(make_ctx, tmp_path)
    _round(ctx)
    lid = ctx.voiced_lines()[1].line_id
    write_override(ctx.paths, lid, "set_text", {"text": "Исправленный пользователем текст."})
    run_stage("synth", ctx)
    takes = _takes(ctx, lid)
    assert takes and {s.model_extra["text_plain"] for s in takes.values()} == {"Исправленный пользователем текст."}
    assert all(s.round == 2 for s in takes.values())


def test_a_new_performer_design_purges_the_old_voice_takes(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path / "game", sets={"synth.voice_unvoiced": True, "score.regen_rounds": 1, "score.gates.utmos_min": 4.99})
    build(ctx, n_per=2)
    ids = add_unvoiced(ctx)
    run_stage("voicebank", ctx)
    _round(ctx)
    lid = ids["hero"][1]
    takes = _takes(ctx, lid)
    assert any(n.endswith("_100.wav") for n in takes) and all(s.model_extra.get("ref_hashes") for s in takes.values())
    perf = gu.load_performer(ctx, "hero")
    for r in perf["refs"]:  # a new design: same file names, another voice
        speechlike(ctx.paths.abs(r["path"]), 2.0, 90.0, seed=77)
    run_stage("synth", ctx)
    takes = _takes(ctx, lid)
    assert takes and not any(n.endswith("_100.wav") for n in takes)


# ---- 2: the cache key ------------------------------------------------------------------------------------------------------


def _reconfig(ctx, **sets):
    ctx.cfg = load_config(TEST_CONFIG, {"synth.backends.sapi_tts.enabled": False, **sets})


def test_more_takes_or_another_backend_keep_the_existing_takes(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path / "game")
    build(ctx, n_per=3)
    first = run_stage("synth", ctx)
    _reconfig(ctx, **{"synth.backends.voxcpm2.enabled": True})  # resolves to the same mock: nothing new to make
    s = run_stage("synth", ctx)
    assert s["generated"] == 0 and s["cached"] == first["candidates"]
    _reconfig(ctx, **{"synth.n_candidates": 3, "synth.n_candidates_short": 4})
    s = run_stage("synth", ctx)
    assert s["cached"] == first["candidates"]
    assert s["generated"] == s["candidates"] - first["candidates"] > 0


def test_a_cache_written_by_1_1_0_is_still_honoured(make_ctx, tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path / "game")
    build(ctx, n_per=2)
    monkeypatch.setattr(synth, "_candidate_hash", synth._synth_hash)  # 1.1.0 keyed every take with the synth section
    first = run_stage("synth", ctx)
    monkeypatch.undo()
    s = run_stage("synth", ctx)
    assert s["generated"] == 0 and s["cached"] == first["candidates"]


# ---- 7: a backend that fails every item ----------------------------------------------------------------------------------


def test_a_backend_failing_every_item_is_restarted_once_then_stops_the_stage(make_ctx, tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path / "game")
    build(ctx, n_per=2)
    loads = []
    orig_load = mock.MockTTS.load

    def load(self):
        loads.append(1)
        return orig_load(self)

    def boom(self, item, **_):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(mock.MockTTS, "load", load)
    monkeypatch.setattr(mock.MockTTS, "op_synthesize", boom)
    with pytest.raises(StageError, match="mock_tts"):
        run_stage("synth", ctx)
    assert len(loads) == 2  # the first load and one restart
    assert not any(read_json(p).get("error") for p in ctx.paths.synth_dir(ctx.voiced_lines()[0].line_id).glob("*.json")) if ctx.paths.synth_dir(ctx.voiced_lines()[0].line_id).exists() else True


def test_a_restart_that_heals_the_backend_finishes_the_stage(make_ctx, tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path / "game")
    build(ctx, n_per=2)
    state = {"loads": 0}
    orig_load, orig_op = mock.MockTTS.load, mock.MockTTS.op_synthesize

    def load(self):
        state["loads"] += 1
        return orig_load(self)

    def flaky(self, item, **kw):
        if state["loads"] == 1:
            raise RuntimeError("CUDA error: device-side assert triggered")
        return orig_op(self, item, **kw)

    monkeypatch.setattr(mock.MockTTS, "load", load)
    monkeypatch.setattr(mock.MockTTS, "op_synthesize", flaky)
    s = run_stage("synth", ctx)
    assert state["loads"] == 2 and s["failed"] == 0 and s["generated"] == s["candidates"] > 0


def test_one_line_the_model_cannot_say_is_an_item_failure(make_ctx, tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path / "game")
    build(ctx, n_per=2)
    bad = ctx.voiced_lines()[0].line_id
    orig_op = mock.MockTTS.op_synthesize

    def picky(self, item, **kw):
        if item["id"].startswith(bad + "|"):
            raise ValueError("text too odd")
        return orig_op(self, item, **kw)

    monkeypatch.setattr(mock.MockTTS, "op_synthesize", picky)
    s = run_stage("synth", ctx)
    assert s["failed"] >= 1 and s["candidates"] > 0


def test_machine_errors():
    assert gu.is_machine_error("RuntimeError: CUDA error: an illegal memory access was encountered")
    assert gu.is_machine_error("CUBLAS_STATUS_EXECUTION_FAILED when calling cublasGemmEx")
    assert gu.is_machine_error("CUDA out of memory")
    assert not gu.is_machine_error("ValueError: text too odd")


# ---- 8: a load that fails inside the adapter -----------------------------------------------------------------------------


def _ctx_raising(exc):
    def acquire(name, options=None):
        raise exc

    return SimpleNamespace(models=SimpleNamespace(acquire=acquire, record_unavailable=lambda *a: None), log=SimpleNamespace(warning=lambda *a, **k: None), stage="score")


@pytest.mark.parametrize("exc", [AdapterError("utmosv2", "RuntimeError: CUDA out of memory: Tried to allocate 20.00 MiB"), AdapterTimeout("audeering_adv", "'load' timed out after 1800s")])
def test_a_load_error_is_unavailable(exc):
    ctx = _ctx_raising(exc)
    with pytest.raises(BackendUnavailable) as e:
        gu.acquire(ctx, "x")
    assert "loading failed" in e.value.reason
    b, used, reason = gu.acquire_first(ctx, ["x"], group="mos")
    assert b is None and "loading failed" in reason


def test_a_vram_load_error_names_the_gpu():
    with pytest.raises(BackendUnavailable) as e:
        gu.acquire(_ctx_raising(AdapterError("utmosv2", "CUDA out of memory")), "x")
    assert "GPU" in e.value.hint


def test_a_cancel_during_a_load_still_propagates():
    with pytest.raises(AdapterAborted):
        gu.acquire_first(_ctx_raising(AdapterAborted("x", "aborted")), ["x"])


def test_a_tts_load_error_blocks_the_backend(make_ctx, tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path / "game")
    build(ctx, n_per=2)

    def bad_load(self):
        raise AdapterError("mock_tts", "OSError: Unable to load weights from model.safetensors")

    monkeypatch.setattr(mock.MockTTS, "load", bad_load)
    with pytest.raises(StageError, match="loading failed"):
        run_stage("synth", ctx)


# ---- review texts ----------------------------------------------------------------------------------------------------------


def test_a_review_text_is_prepared_like_a_translation(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path / "game")
    build(ctx, n_per=2)
    lid = ctx.voiced_lines()[0].line_id
    write_override(ctx.paths, lid, "set_text", {"text": "Мне 16 лет."})
    run_stage("synth", ctx)
    sides = list(_takes(ctx, lid).values())
    assert sides and all("шестнадцать" in s.text_used and "16" not in s.text_used for s in sides)
    assert {s.model_extra["text_plain"] for s in sides} == {"Мне шестнадцать лет."}


def test_user_stress_marks_are_kept(monkeypatch):
    import vndub.stages.translate as tr

    called = []
    monkeypatch.setattr(tr, "make_stresser", lambda ctx, texts: (called.append(texts) or (lambda t: t.replace("о", "+о", 1)), "mock", ""))
    ctx = SimpleNamespace(cfg=SimpleNamespace(translate=SimpleNamespace(stress=SimpleNamespace(yo="safe"))), log=SimpleNamespace(warning=lambda *a, **k: None))
    out = synth._prepare_user_texts(ctx, ["Зам+ок, 3 штуки.", "Хорошо, 3 раза."], "ru")
    plain_tts, plus, plain, marked = out["Зам+ок, 3 штуки."]
    assert marked and plus == "Зам+ок, три штуки." and plain == plain_tts == "Замок, три штуки."
    plain_tts, plus, plain, marked = out["Хорошо, 3 раза."]
    assert not marked and "+о" in plus and "три" in plain and "+" not in plain_tts
    assert called == [["Хорошо, три раза."]]  # the stresser sees only the unmarked text, normalized


# ---- 6: NaN in a reply ---------------------------------------------------------------------------------------------------


def test_a_nan_reply_is_parsed_with_none():
    msg = _lenient_reply(b'{"jsonrpc": "2.0", "id": 3, "result": [{"id": "a", "mos": NaN, "x": -Infinity}]}')
    assert msg == {"jsonrpc": "2.0", "id": 3, "result": [{"id": "a", "mos": None, "x": None}]}
    assert _lenient_reply(b"Loading checkpoint NaN%") is None
    assert _lenient_reply(b'{"jsonrpc": broken') is None

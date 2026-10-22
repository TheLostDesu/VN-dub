from __future__ import annotations

import json

import yaml

from analysis_helpers import build_project
from vndub.manifest.io import TranslationRecord, read_jsonl
from vndub.stages import translate
from vndub.stages.common import run_stage

ROWS = [
    dict(line_id="s01_01", speaker_id="sakura", speaker_raw="さくら", text="テストで百点を取ったんだよ！", seconds=0.5),
    dict(line_id="s01_02", speaker_id="haruto", speaker_raw="陽翔", text="どうしたの、さくら？", seconds=0.5),
    dict(line_id="s01_03", speaker_id="yui", speaker_raw="ゆい", text="うん！", seconds=0.5),
    dict(line_id="s01_04", speaker_id="narrator", text="三人は並んで校門を出た。", voiced=False),
    dict(line_id="s01_05", speaker_id="haruto", speaker_raw="陽翔", text="ああ、そうしよう。", seconds=0.5, extra={"sustain_lines": ["s01_06"]}, flags=["sustain_head"]),
    dict(line_id="s01_06", speaker_id="haruto", speaker_raw="陽翔", text="雨が降りそうだしな。", voiced=False, flags=["voice_sustain"]),
    dict(line_id="s02_01", speaker_id="sakura", speaker_raw="さくら", text="じゃあね！", seconds=0.5, scene="s02"),
]

DICT = {
    "テストで百点を取ったんだよ！": "Я получила 100 баллов на тесте!",
    "どうしたの、さくら？": "Что случилось, Сакура?",
    "ああ、そうしよう。": "Ага, давай.",
    "雨が降りそうだしな。": "Похоже, скоро пойдёт дождь.",
    "じゃあね！": "Пока!",
}


def _ctx(make_ctx, tmp_path, **sets):
    d = tmp_path / "dict.yaml"
    d.write_text(yaml.safe_dump(DICT, allow_unicode=True), encoding="utf-8")
    base = {"translate.mock_dictionary": str(d), "translate.glossary": str(tmp_path / "glossary.yaml")}
    base.update(sets)
    ctx = make_ctx(tmp_path, sets=base)
    build_project(ctx, ROWS)
    return ctx


def _records(ctx) -> dict[str, TranslationRecord]:
    return {r.line_id: r for r in read_jsonl(ctx.paths.translation("ru"), TranslationRecord)}


def test_mock_translator_end_to_end(make_ctx, tmp_path):
    ctx = _ctx(make_ctx, tmp_path)
    summary = run_stage("translate", ctx)
    recs = _records(ctx)
    # voiced lines + the sustain continuation; the unvoiced narrator line is skipped
    assert set(recs) == {"s01_01", "s01_02", "s01_03", "s01_05", "s01_06", "s02_01"}
    r = recs["s01_01"]
    assert r.text_tgt == "Я получила 100 баллов на тесте!"
    assert r.text_tts == "Я получила сто баллов на тесте!"
    assert r.text_plain == "Я получила сто баллов на тесте!"
    assert r.source == "llm:mock_translator" and r.syllables > 5
    assert recs["s01_03"].text_tgt == "угу" and "interjection" in recs["s01_03"].flags and recs["s01_03"].source == "interjection"
    assert recs["s01_06"].text_tgt.startswith("Похоже")
    assert recs["s01_02"].glossary_hits == ["さくら"]
    assert summary["interjections"] == 1 and summary["failed"] == 0 and summary["sources"]["llm"] == 5
    assert "no_stress" not in r.flags  # mock_stress is a stresser (identity)
    ctx.models.close()


def test_translate_batch_cache(make_ctx, tmp_path, monkeypatch):
    ctx = _ctx(make_ctx, tmp_path)
    run_stage("translate", ctx)
    ctx.models.close()

    def boom(*a, **k):
        raise AssertionError("llm acquired although every batch is cached")

    monkeypatch.setattr(translate, "_acquire_llm", boom)
    monkeypatch.setattr(ctx.models, "acquire", boom)  # stress results are cached too
    run_stage("translate", ctx)
    assert _records(ctx)["s01_01"].text_tgt.startswith("Я получила")


class FlakyLLM:
    """Bad JSON, then a partial answer, then answers only single-line batches correctly."""

    def __init__(self, fail_ids=()):
        self.calls = []
        self.fail_ids = set(fail_ids)

    def chat(self, messages, temperature=0.3, **kw):
        batch = json.loads(messages[-1]["content"].split("\n", 1)[1])
        self.calls.append((len(batch), temperature))
        n = len(self.calls)
        if n == 1:
            return "this is not json"
        if n == 2 and len(batch) > 1:
            return json.dumps({"lines": [{"id": batch[0]["id"], "ru": "первый", "ru_short": "1", "notes": "x"}]})
        out = [{"id": b["id"], "ru": f"перевод {b['id']}", "ru_short": "", "notes": "calm and slow and very very long note"} for b in batch if b["id"] not in self.fail_ids and len(batch) == 1]
        return "```json\n" + json.dumps({"lines": out}, ensure_ascii=False) + "\n```"


def _translator(ctx, llm):
    return translate.LlmTranslator(ctx, llm, "fake", "ru", {}, {})


def _batch(ids):
    return [{"id": i, "speaker": "A", "speaker_id": "a", "gender": "f", "emotion": "", "text": f"テキスト{i}"} for i in ids]


def test_retry_temperature_zero_then_chunk_down(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path)
    llm = FlakyLLM()
    got = _translator(ctx, llm).translate(_batch(["a", "b", "c", "d"]), [], "")
    assert set(got) == {"a", "b", "c", "d"}
    assert got["a"]["text_tgt"] == "первый"
    assert llm.calls[0] == (4, 0.3) and llm.calls[1] == (4, 0.0)  # invalid JSON -> same batch again at T=0
    assert any(size == 1 for size, _ in llm.calls)  # chunked down to single lines
    assert got["b"]["text_tgt_short"] == got["b"]["text_tgt"]  # empty short -> full
    assert len(got["b"]["notes"].split()) <= 6


def test_lines_that_keep_failing_are_flagged(make_ctx, tmp_path, monkeypatch):
    ctx = _ctx(make_ctx, tmp_path, **{"translate.batch_lines": 2})
    llm = FlakyLLM(fail_ids={"s01_02"})
    monkeypatch.setattr(translate, "_acquire_llm", lambda c: (llm, "fake_llm"))
    summary = run_stage("translate", ctx)
    recs = _records(ctx)
    bad = recs["s01_02"]
    assert "translate_failed" in bad.flags and bad.text_tgt == "どうしたの、さくら？" and bad.text_tts == ""
    assert recs["s01_01"].text_tgt == "первый"  # from the partial answer
    assert summary["failed"] == 1
    ctx.models.close()


def test_parse_response_validation():
    ids = ["1", "2"]
    assert translate.parse_response('{"lines": [{"id": "1", "ru": "да"}, {"id": "9", "ru": "x"}, {"id": "2", "ru": ""}]}', ids, "ru") == {"1": {"text_tgt": "да", "text_tgt_short": "да", "notes": ""}}
    assert translate.parse_response('[{"id": "2", "ru": "нет", "ru_short": "н"}]', ids, "ru")["2"]["text_tgt_short"] == "н"
    assert translate.parse_response("Sure! {\"lines\": [{\"id\": \"1\", \"ru\": \"ок\"}]} hope this helps", ids, "ru")["1"]["text_tgt"] == "ок"
    assert translate.parse_response("nope", ids, "ru") == {}
    assert translate.parse_response("", ids, "ru") == {}


def test_messages_shape_for_mock_protocol(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path)
    tr = _translator(ctx, None)
    msgs = tr.messages(_batch(["a"]), [{"speaker": "A", "text": "前", "ru": "до"}], "summary")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    last = msgs[-1]["content"]
    assert "[" not in last.split("\n", 1)[0]  # the batch is the only JSON array in the last message
    assert json.loads(last.split("\n", 1)[1])[0]["id"] == "a"
    assert "speaker_id" not in last
    assert '"ru_short"' in msgs[0]["content"]


def test_import_jsonl_and_csv(make_ctx, tmp_path):
    imp = tmp_path / "import.jsonl"
    imp.write_text("\n".join(json.dumps({"line_id": k, "text_tgt": v}, ensure_ascii=False) for k, v in [("s01_01", "Сто баллов!"), ("s01_02", "Что такое?")]), encoding="utf-8")
    ctx = _ctx(make_ctx, tmp_path, **{"translate.import": str(imp)})
    summary = run_stage("translate", ctx)
    recs = _records(ctx)
    assert recs["s01_01"].text_tgt == "Сто баллов!" and recs["s01_01"].source == "import"
    assert recs["s01_03"].source == "interjection"
    assert "translate_missing" in recs["s01_05"].flags and recs["s01_05"].text_tts == ""
    assert summary["backend"] == "import" and summary["missing"] == 3

    csv_path = tmp_path / "t.tsv"
    csv_path.write_text("line_id\tru\ns01_05\tАга.\n", encoding="utf-8")
    assert translate.load_import(csv_path, [], "ru", ctx.log) == {"s01_05": "Ага."}
    ctx.models.close()


def test_import_vntextpatch_positional_and_by_source(make_ctx, tmp_path):
    ctx = _ctx(make_ctx, tmp_path)
    lines = ctx.lines()
    for l in lines:
        l.source_file = f"scenario/{l.scene}.ks"
    d = tmp_path / "vtp"
    d.mkdir()
    s01 = [l for l in lines if l.scene == "s01"]
    (d / "s01.json").write_text(json.dumps([{"name": "x", "message": f"RU {l.line_id}"} for l in s01], ensure_ascii=False), encoding="utf-8")
    (d / "s02.json").write_text(json.dumps([{"original": "じゃあね！", "translation": "Пока-пока!"}], ensure_ascii=False), encoding="utf-8")
    got = translate.load_import(d, lines, "ru", ctx.log)
    assert got["s01_02"] == "RU s01_02" and got["s01_06"] == "RU s01_06"
    assert got["s02_01"] == "Пока-пока!"


def test_import_xlsx(make_ctx, tmp_path):
    import openpyxl

    ctx = _ctx(make_ctx, tmp_path)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Original", "Translation"])
    ws.append(["どうしたの、さくら？", "В чём дело?"])
    p = tmp_path / "t.xlsx"
    wb.save(p)
    assert translate.load_import(p, ctx.lines(), "ru", ctx.log) == {"s01_02": "В чём дело?"}


def test_seed_glossary(make_ctx, tmp_path):
    ctx = _ctx(make_ctx, tmp_path)
    ctx.extra["seed_glossary"] = True
    out = run_stage("translate", ctx)
    g = yaml.safe_load((tmp_path / "glossary.yaml").read_text(encoding="utf-8"))
    assert set(g["characters"]) == {"sakura", "haruto", "yui"}
    assert g["characters"]["haruto"]["src"] == "陽翔" and g["honorifics"] == "drop"
    assert out["characters_added"] == 3
    # extending keeps user edits
    g["characters"]["haruto"]["tgt"] = "Харуто"
    (tmp_path / "glossary.yaml").write_text(yaml.safe_dump(g, allow_unicode=True), encoding="utf-8")
    out = run_stage("translate", ctx)
    assert out["characters_added"] == 0
    assert yaml.safe_load((tmp_path / "glossary.yaml").read_text(encoding="utf-8"))["characters"]["haruto"]["tgt"] == "Харуто"
    assert not ctx.paths.translation("ru").exists()


def test_system_prompt_glossary_filtering():
    gloss = {
        "honorifics": "map",
        "honorific_map": {"先輩": "семпай", "先生": "сэнсэй"},
        "address_matrix": {"yui": {"default": "ты", "teacher": "вы"}},
        "characters": {"yui": {"src": "ゆい", "tgt": "Юи", "gender": "f", "notes": "shy"}, "boss": {"src": "社長", "tgt": "Директор", "gender": "m"}},
        "terms": [{"src": "放課後", "tgt": "после уроков"}, {"src": "魔法", "tgt": "магия"}],
    }
    batch = [{"id": "1", "speaker_id": "yui", "text": "先輩、放課後に会おう。"}]
    sp = translate.system_prompt("ru", batch, ["yui"], gloss, {}, 8)
    assert "先輩 -> семпай" in sp and "先生" not in sp
    assert "放課後 -> после уроков" in sp and "魔法" not in sp
    assert "ゆい -> Юи" in sp and "社長" not in sp and "gender f" in sp
    assert "to teacher 'вы'" in sp and "Russian" in sp


def test_plus_variant_when_backend_needs_it(make_ctx, tmp_path):
    ctx = _ctx(make_ctx, tmp_path, **{"synth.backends.f5_ru.enabled": True})
    summary = run_stage("translate", ctx)
    r = _records(ctx)["s01_01"]
    assert summary["plus_variant"] is True
    assert getattr(r, "text_tts_plus") == "Я получила сто баллов на тесте!"  # mock stresser adds no marks
    ctx.models.close()


# ----------------------------------------------------------------------------
# round 8: the model that runs is the model the cache is keyed by, and is recorded
# ----------------------------------------------------------------------------

import urllib.error  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402

from vndub.backends import registry  # noqa: E402
from vndub.backends.base import BackendSpec  # noqa: E402
from vndub.gpu import GB, ModelRegistry  # noqa: E402
from vndub.stages.common import StageError  # noqa: E402


class _Cuda:
    def __init__(self, total_gb: float = 8.0, other_gb: float = 0.3) -> None:
        self.total, self.used = int(total_gb * GB), int(other_gb * GB)

    def available(self) -> bool:
        return True

    def mem_get_info(self) -> tuple[int, int]:
        return self.total - self.used, self.total

    def empty_cache(self) -> None:
        pass


class _Cancelled(Exception):
    """Stands for the desktop app's JobCancelled (an Exception subclass)."""


def _answer(backend, kind, messages):
    if kind == "summary":
        return json.dumps({"summary": f"summary by {backend.spec.name}"})
    batch = json.loads(messages[-1]["content"].split("\n", 1)[1])
    rows = [{"id": b["id"], "ru": f"{backend.spec.name} {b['id']}", "ru_short": "", "notes": ""} for b in batch]
    return json.dumps({"lines": rows}, ensure_ascii=False)


class _ChatBackend:
    """An LLM of the fake registry: answers the translate protocol and writes its own name into every line."""

    def __init__(self, spec, script=_answer):
        self.spec, self.script, self.loaded = spec, script, False
        self.requests: list[str] = []

    def load(self):
        self.loaded = True
        return {"device": "cuda:0"}

    def unload(self):
        self.loaded = False

    def peak_vram(self):
        return None

    def chat(self, messages, temperature=0.3, **kw):
        kind = "summary" if messages[0]["content"].startswith("Summarize") else "translate"
        self.requests.append(kind)
        return self.script(self, kind, messages)


def _gpu_ctx(make_ctx, tmp_path, monkeypatch, cuda=None, big=4.5, small=2.0, script=_answer, **sets):
    """A project whose LLM is llm_big with llm_small as the fallback, on a fake 8 GB card."""
    specs = {
        n: BackendSpec(name=n, role="llm", env="fake", adapter=n, license="MIT", commercial_ok=True, vram_estimate_gb=v, heavy=True)
        for n, v in (("llm_big", big), ("llm_small", small))
    }
    monkeypatch.setattr(registry, "_SPECS", {**registry.all_specs(), **specs})
    base = {"translate.llm.backend": "llm_big", "translate.llm.fallbacks": ["llm_small"], "translate.stress.tool": "none"}
    ctx = _ctx(make_ctx, tmp_path, **{**base, **sets})
    cuda = cuda or _Cuda()
    made: list[_ChatBackend] = []

    def factory(spec, options, device):
        b = _ChatBackend(spec, script)
        made.append(b)
        return b

    ctx.models = ModelRegistry(
        budget_gb=7.5, safety_margin_gb=0.5, check_usable=False, factory=factory, cuda=cuda, models_dir=tmp_path / "models",
        log_dir=ctx.paths.logs, timings_path=ctx.paths.work / "gpu_events.jsonl", vram_retry_s=0.0,
    )
    return ctx, cuda, made


def _cache_values(ctx) -> dict:
    p = ctx.paths.cache / "results" / "translate_llm.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def test_cache_key_follows_the_model_that_fits(make_ctx, tmp_path, monkeypatch):
    # both fit: the primary
    ctx, cuda, _ = _gpu_ctx(make_ctx, tmp_path, monkeypatch)
    assert translate._llm_key_backend(ctx) == "llm_big"
    # the primary is too big for the budget: the fallback, and that is the model _acquire_llm loads
    ctx, cuda, _ = _gpu_ctx(make_ctx, tmp_path, monkeypatch, big=7.9)
    assert translate._llm_key_backend(ctx) == "llm_small"
    backend, name = translate._acquire_llm(ctx)
    assert name == "llm_small" and backend.loaded
    assert [(s.requested, s.used) for s in ctx.models.substitutions] == [("llm_big", "llm_small")]
    assert "budget" in ctx.models.substitutions[0].reason
    ctx.models.close()
    # the primary fits the estimate but another program holds the card (live free memory)
    ctx, cuda, _ = _gpu_ctx(make_ctx, tmp_path, monkeypatch, cuda=_Cuda(other_gb=5.0))
    assert translate._llm_key_backend(ctx) == "llm_small"
    assert translate._acquire_llm(ctx)[1] == "llm_small"
    ctx.models.close()
    # nothing fits: the primary (the stage raises when it needs a model)
    ctx, cuda, _ = _gpu_ctx(make_ctx, tmp_path, monkeypatch, big=7.9, small=7.9)
    assert translate._llm_key_backend(ctx) == "llm_big"
    with pytest.raises(StageError, match="no translation backend usable"):
        translate._acquire_llm(ctx)
    assert ctx.models.resident == []  # the walk loaded nothing


def test_key_walk_loads_and_logs_nothing(make_ctx, tmp_path, monkeypatch):
    ctx, cuda, made = _gpu_ctx(make_ctx, tmp_path, monkeypatch, big=7.9)
    translate._llm_key_backend(ctx)
    assert made == [] and ctx.models.resident == [] and ctx.models.substitutions == []


def test_models_used_one_model_and_entries_name_it(make_ctx, tmp_path, monkeypatch):
    ctx, cuda, _ = _gpu_ctx(make_ctx, tmp_path, monkeypatch)
    summary = run_stage("translate", ctx)
    # five translated lines; the interjection (a lexicon, no model) is left out
    assert summary["models_used"] == {"llm_big": 5} and summary["backend"] == "llm_big"
    assert {r.source for r in _records(ctx).values() if r.source.startswith("llm")} == {"llm:llm_big"}
    batches = [v for k, v in _cache_values(ctx).items() if k.startswith("batch:")]
    assert batches and all(e["source"] == "llm:llm_big" for b in batches for e in b.values())  # stored with the entry
    ctx.models.close()


def test_models_used_mock_and_import(make_ctx, tmp_path):
    ctx = _ctx(make_ctx, tmp_path)
    assert run_stage("translate", ctx)["models_used"] == {"mock_translator": 5}
    ctx.models.close()
    imp = tmp_path / "import.jsonl"
    imp.write_text("\n".join(json.dumps({"line_id": k, "text_tgt": v}, ensure_ascii=False) for k, v in [("s01_01", "Сто баллов!"), ("s01_02", "Что такое?")]), encoding="utf-8")
    ctx = _ctx(make_ctx, tmp_path, **{"translate.import": str(imp)})
    assert run_stage("translate", ctx)["models_used"] == {"import": 2}  # the missing lines have no translation to count
    ctx.models.close()


def test_resume_with_another_model_reads_the_cache_and_names_both(make_ctx, tmp_path, monkeypatch):
    # run 1: only the last scene, by the preferred model
    ctx, cuda, _ = _gpu_ctx(make_ctx, tmp_path, monkeypatch)
    ctx.only = {"s02_01"}
    assert run_stage("translate", ctx)["models_used"] == {"llm_big": 1}
    ctx.models.close()
    # run 2 after a restart: another program holds the card, llm_big no longer fits, llm_small does
    ctx, cuda, made = _gpu_ctx(make_ctx, tmp_path, monkeypatch, cuda=_Cuda(other_gb=5.0))
    summary = run_stage("translate", ctx)
    assert summary["models_used"] == {"llm_small": 4, "llm_big": 1}  # the cached scene keeps its author's name
    recs = _records(ctx)
    assert recs["s02_01"].source == "llm:llm_big" and recs["s01_01"].source == "llm:llm_small"
    assert recs["s02_01"].text_tgt == "llm_big s02_01" and recs["s01_01"].text_tgt == "llm_small s01_01"
    assert [(s.requested, s.used) for s in ctx.models.substitutions] == [("llm_big", "llm_small")]
    assert [b.spec.name for b in made] == ["llm_small"]  # the cached scene cost no model call
    ctx.models.close()
    # run 3: the card is free again; what llm_small wrote is translated again by the preferred model
    ctx, cuda, made = _gpu_ctx(make_ctx, tmp_path, monkeypatch)
    summary = run_stage("translate", ctx)
    assert summary["models_used"] == {"llm_big": 5} and ctx.models.substitutions == []
    ctx.models.close()


def test_a_fully_cached_project_is_not_retranslated_when_the_preferred_model_does_not_fit(make_ctx, tmp_path, monkeypatch):
    ctx, cuda, _ = _gpu_ctx(make_ctx, tmp_path, monkeypatch)
    run_stage("translate", ctx)
    ctx.models.close()
    ctx, cuda, made = _gpu_ctx(make_ctx, tmp_path, monkeypatch, cuda=_Cuda(other_gb=5.0))
    summary = run_stage("translate", ctx)
    assert summary["models_used"] == {"llm_big": 5} and made == []  # no load, nothing translated by the smaller model
    ctx.models.close()


class _Truncating:
    def __init__(self):
        self.calls = 0

    def chat(self, messages, temperature=0.3, **kw):
        batch = json.loads(messages[-1]["content"].split("\n", 1)[1])
        self.calls += 1
        text = json.dumps({"lines": [{"id": b["id"], "ru": f"x {b['id']}", "ru_short": "", "notes": ""} for b in batch]})
        return text[: len(text) // 2] if self.calls == 1 else text


def test_entries_carry_the_model_through_retries_and_halving(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path)
    got = _translator(ctx, FlakyLLM()).translate(_batch(["a", "b", "c", "d"]), [], "")
    assert set(got) == {"a", "b", "c", "d"} and {e["source"] for e in got.values()} == {"llm:fake"}
    salvaged = translate.LlmTranslator(ctx, _Truncating(), "fake", "ru", {}, {}).translate(_batch([f"l{i}" for i in range(8)]), [], "")
    assert len(salvaged) == 8 and {e["source"] for e in salvaged.values()} == {"llm:fake"}


# -- cancel is never an empty answer -------------------------------------------------------------------------


def _install_cancel(ctx):
    state = {"cancelled": False}

    def check():
        if state["cancelled"]:
            raise _Cancelled("job cancelled")

    ctx.extra["cancel_check"] = check
    return state


def test_cancel_during_a_translation_call_propagates_within_one_attempt(make_ctx, tmp_path, monkeypatch):
    holder: dict = {}

    def script(backend, kind, messages):
        if kind == "summary":
            return _answer(backend, kind, messages)
        holder["state"]["cancelled"] = True  # Cancel killed llama-server: the call that was in flight fails
        raise urllib.error.URLError(ConnectionRefusedError(10061, "refused"))

    ctx, cuda, made = _gpu_ctx(make_ctx, tmp_path, monkeypatch, script=script)
    holder["state"] = _install_cancel(ctx)
    with pytest.raises(_Cancelled):
        run_stage("translate", ctx)
    assert made[0].requests.count("translate") == 1  # no retry at T=0, no halving
    assert not [k for k in _cache_values(ctx) if k.startswith("batch:")]
    ctx.models.close()


def test_cancel_during_the_scene_summary_caches_no_summary(make_ctx, tmp_path, monkeypatch):
    holder: dict = {}

    def script(backend, kind, messages):
        if kind == "summary":
            holder["state"]["cancelled"] = True
            raise ConnectionResetError("server killed")
        return _answer(backend, kind, messages)

    ctx, cuda, made = _gpu_ctx(make_ctx, tmp_path, monkeypatch, script=script)
    holder["state"] = _install_cancel(ctx)
    with pytest.raises(_Cancelled):
        run_stage("translate", ctx)
    assert made[0].requests == ["summary"]  # nothing was translated after the cancel
    assert _cache_values(ctx) == {}  # in particular no derived summary for the scene
    ctx.models.close()


def test_cancel_during_a_delivery_call_propagates(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path)
    state = _install_cancel(ctx)

    class Dying:
        def chat(self, messages, **kw):
            state["cancelled"] = True
            raise ConnectionResetError("killed")

    tr = _translator(ctx, Dying())
    with pytest.raises(_Cancelled):
        tr.deliver([{**b, "want_delivery": True} for b in _batch(["a"])])
    assert tr.calls == 1
    with pytest.raises(_Cancelled):
        _translator(ctx, Dying()).summarize("s01", [{"speaker": "A", "text": "x"}])


def test_derived_summary_is_used_for_the_batch_but_never_stored(make_ctx, tmp_path, monkeypatch):
    def script(backend, kind, messages):
        return "not json at all" if kind == "summary" else _answer(backend, kind, messages)

    ctx, cuda, made = _gpu_ctx(make_ctx, tmp_path, monkeypatch, script=script)
    summary = run_stage("translate", ctx)
    assert summary["failed"] == 0 and summary["models_used"] == {"llm_big": 5}  # the batches were translated anyway
    values = _cache_values(ctx)
    assert any(k.startswith("batch:") for k in values) and not [k for k in values if k.startswith("summary:")]
    ctx.models.close()


def test_a_summary_the_model_wrote_is_stored(make_ctx, tmp_path, monkeypatch):
    ctx, cuda, made = _gpu_ctx(make_ctx, tmp_path, monkeypatch)
    run_stage("translate", ctx)
    assert {k for k in _cache_values(ctx) if k.startswith("summary:")} == {"summary:s01", "summary:s02"}
    ctx.models.close()


# -- a dead server stops the halving --------------------------------------------------------------------------


class _Failing:
    def __init__(self, exc, proc=None):
        self.exc, self.calls = exc, 0
        if proc is not None:
            self.proc = proc

    def chat(self, messages, **kw):
        self.calls += 1
        raise self.exc


def test_dead_server_raises_instead_of_halving(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path)
    refused = _Failing(urllib.error.URLError(ConnectionRefusedError(10061, "refused")))
    with pytest.raises(StageError, match="stopped answering"):
        _translator(ctx, refused).translate(_batch(["a", "b", "c", "d"]), [], "")
    assert refused.calls == 1
    exited = _Failing(RuntimeError("boom"), proc=SimpleNamespace(poll=lambda: 3))
    with pytest.raises(StageError, match="exited with code 3"):
        _translator(ctx, exited).translate(_batch(["a", "b"]), [], "")
    assert exited.calls == 1
    reset = _Failing(ConnectionResetError("reset"))
    with pytest.raises(StageError):
        _translator(ctx, reset).translate(_batch(["a", "b"]), [], "")
    assert reset.calls == 1


def test_a_living_server_that_answers_badly_keeps_the_retry_path(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path)
    err = urllib.error.HTTPError("http://127.0.0.1/v1", 500, "slot busy", {}, None)
    http = _Failing(err, proc=SimpleNamespace(poll=lambda: None))
    got = _translator(ctx, http).translate(_batch(["a", "b", "c", "d"]), [], "")
    assert got == {} and http.calls > 2  # the retry at T=0 and the halves ran
    timeout = _Failing(TimeoutError("timed out"), proc=SimpleNamespace(poll=lambda: None))
    assert _translator(ctx, timeout).translate(_batch(["a", "b"]), [], "") == {} and timeout.calls > 1

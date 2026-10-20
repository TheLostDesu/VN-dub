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

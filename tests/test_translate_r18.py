"""Round 18: batch keys by glossary slice, partial answers cached, one stream in script order, gender, checks, names hygiene."""

from __future__ import annotations

import json

import yaml

from analysis_helpers import build_project
from vndub.manifest.io import TranslationRecord, load_characters, read_jsonl
from vndub.stages import translate
from vndub.stages.common import run_stage

ROWS = [
    dict(line_id="s01_01", speaker_id="alice", speaker_raw="Alice", text="I finally finished the report!", seconds=1.0),
    dict(line_id="s01_02", speaker_id="bob", speaker_raw="Bob", text="Welcome back! Did you meet Patchouli there?", seconds=1.0),
    dict(line_id="s01_03", speaker_id="alice", speaker_raw="Alice", text="Patchouli said no.", seconds=1.0),
    dict(line_id="s02_01", speaker_id="bob", speaker_raw="Bob", text="The Mansion is dark tonight.", seconds=1.0, scene="s02"),
]

GLOSS = {
    "player": {"name": "", "gender": "f", "notes": ""},
    "characters": {"alice": {"src": "Alice", "tgt": "Алиса", "gender": "f"}, "bob": {"src": "Bob", "tgt": "Боб", "gender": "m"}},
    "terms": [],
}


def _long(b):
    """An answer as long as the source (a much shorter one is "omitted" and re-asked)."""
    return "Перевод " + b["id"] + " " + " ".join(["слово"] * len(b["text"].split()))


class FakeLLM:
    """Records every request; answers every line unless its id is in ``fail`` (then the line is left out of the answer)."""

    def __init__(self, fail=(), answer=None):
        self.calls: list[list[dict]] = []
        self.fail = set(fail)
        self.answer = answer or _long
        self.last = {}

    def chat(self, messages, temperature=0.3, **kw):
        sys_txt = messages[0]["content"]
        if "Summarize" in sys_txt:
            return json.dumps({"summary": "A scene."})
        if "proper noun" in sys_txt:
            self.calls.append([{"names": messages[-1]["content"]}])
            return json.dumps({"names": []})
        if "gender" in sys_txt and "genders" in sys_txt:
            self.calls.append([{"genders": messages[-1]["content"]}])
            return json.dumps({"genders": []})
        batch = json.loads(messages[-1]["content"].split("\n", 1)[1])
        self.calls.append(batch)
        return json.dumps({"lines": [{"id": b["id"], "ru": self.answer(b), "ru_short": "", "notes": "calm"} for b in batch if b["id"] not in self.fail]}, ensure_ascii=False)

    def batches(self):
        return [c for c in self.calls if c and "id" in c[0]]


def _ctx(make_ctx, tmp_path, rows=ROWS, gloss=GLOSS, **sets):
    gp = tmp_path / "glossary.yaml"
    if not gp.exists():
        gp.write_text(yaml.safe_dump(gloss, allow_unicode=True), encoding="utf-8")
    base = {"translate.glossary": str(gp), "translate.batch_lines": 2}
    base.update(sets)
    ctx = make_ctx(tmp_path, sets=base)
    if not ctx.paths.lines.exists():
        build_project(ctx, rows)
    return ctx


def _run(ctx, monkeypatch, llm):
    monkeypatch.setattr(translate, "_acquire_llm", lambda c: (llm, "fake_llm"))
    monkeypatch.setattr(translate, "_llm_key_backend", lambda c: "fake_llm")
    summary = run_stage("translate", ctx)
    ctx.models.close()
    return summary


def _records(ctx):
    return {r.line_id: r for r in read_jsonl(ctx.paths.translation("ru"), TranslationRecord)}


def _edit_glossary(tmp_path, fn):
    gp = tmp_path / "glossary.yaml"
    g = yaml.safe_load(gp.read_text(encoding="utf-8"))
    fn(g)
    gp.write_text(yaml.safe_dump(g, allow_unicode=True), encoding="utf-8")


# ---------------------------------------------------------------- 1: glossary slice keys, partial answers


def test_an_unrelated_glossary_edit_retranslates_nothing_and_a_term_only_its_batch(make_ctx, tmp_path, monkeypatch):
    ctx = _ctx(make_ctx, tmp_path)
    _run(ctx, monkeypatch, FakeLLM())
    llm = FakeLLM()
    _edit_glossary(tmp_path, lambda g: g["characters"]["alice"].update(age_hint="teen"))  # not in any translation prompt
    _edit_glossary(tmp_path, lambda g: g["terms"].append({"src": "Gensokyo", "tgt": "Генсокё"}))  # no line uses it
    _run(ctx, monkeypatch, llm)
    assert llm.batches() == []
    _edit_glossary(tmp_path, lambda g: g["terms"].append({"src": "Mansion", "tgt": "Особняк"}))  # only s02 uses it
    llm = FakeLLM()
    _run(ctx, monkeypatch, llm)
    # its batch, then the glossary-miss recheck (the fake does not write "Особняк"); nothing else
    assert [[b["id"] for b in c] for c in llm.batches()] == [["s02_01"], ["s02_01"]] and "previous_answer" in llm.batches()[1][0]


def test_a_partial_answer_is_cached_and_a_rerun_asks_only_the_missing_line(make_ctx, tmp_path, monkeypatch):
    ctx = _ctx(make_ctx, tmp_path)
    summary = _run(ctx, monkeypatch, FakeLLM(fail={"s01_02"}))
    assert summary["failed"] == 1
    llm = FakeLLM()
    _run(ctx, monkeypatch, llm)
    # only the missing line of its batch is asked; the next batch of the scene follows (its rolling context now holds that line),
    # the other scene is untouched
    assert [[b["id"] for b in c] for c in llm.batches()] == [["s01_02"], ["s01_03"]]
    assert _records(ctx)["s01_02"].text_tgt.startswith("Перевод s01_02")


# ---------------------------------------------------------------- 7: one stream in script order


def test_a_never_voiced_project_is_translated_in_script_order_with_rolling_context(make_ctx, tmp_path, monkeypatch):
    rows = [{**r, "voiced": False} for r in ROWS] + [dict(line_id="s01_04", speaker_id="narrator", text="She left.", voiced=False)]
    rows = rows[:3] + [rows[4], rows[3]]
    ctx = _ctx(make_ctx, tmp_path, rows=rows, **{"synth.voice_unvoiced": True, "translate.include_narration": "on"})
    llm = FakeLLM()
    _run(ctx, monkeypatch, llm)
    order = [b["id"] for c in llm.batches() for b in c]
    assert order == ["s01_01", "s01_02", "s01_03", "s01_04", "s02_01"]
    first = llm.batches()[0]
    assert all(b.get("want_delivery") for b in first if b["id"] != "s01_04")
    assert all(not k.endswith(":delivery") for k in json.loads((ctx.paths.cache / "results" / "translate_llm.json").read_text(encoding="utf-8")))
    assert set(_records(ctx)) == {"s01_01", "s01_02", "s01_03", "s01_04", "s02_01"}


def test_the_second_batch_sees_the_first_batch_translations(make_ctx, tmp_path, monkeypatch):
    rows = [{**r, "voiced": False} for r in ROWS]
    ctx = _ctx(make_ctx, tmp_path, rows=rows, **{"synth.voice_unvoiced": True})
    seen: list[str] = []

    class Spy(FakeLLM):
        def chat(self, messages, temperature=0.3, **kw):
            seen.append("\n".join(m["content"] for m in messages[1:-1]))
            return super().chat(messages, temperature, **kw)

    _run(ctx, monkeypatch, Spy())
    assert any("Перевод s01_01" in s and "Перевод s01_02" in s for s in seen)


# ---------------------------------------------------------------- 2: player gender for narration / protagonist, unknown rule, recheck rows


def test_narration_and_the_protagonist_take_the_player_gender(make_ctx, tmp_path, monkeypatch):
    rows = [
        dict(line_id="s01_01", speaker_id="narrator", text="I was so tired that day.", voiced=False),
        dict(line_id="s01_02", speaker_id="mc", speaker_raw="[player_name]", text="I came back late.", voiced=False),
        dict(line_id="s01_03", speaker_id="stranger", speaker_raw="???", text="I was waiting.", seconds=1.0),
    ]
    ctx = _ctx(make_ctx, tmp_path, rows=rows, **{"synth.voice_unvoiced": True, "translate.include_narration": "on", "translate.batch_lines": 5, "translate.gender_pass": False})
    sys_prompts: list[str] = []

    class Spy(FakeLLM):
        def chat(self, messages, temperature=0.3, **kw):
            sys_prompts.append(messages[0]["content"])
            return super().chat(messages, temperature, **kw)

    llm = Spy()
    _run(ctx, monkeypatch, llm)
    (batch,) = llm.batches()
    g = {b["id"]: b["gender"] for b in batch}
    assert g == {"s01_01": "f", "s01_02": "f", "s01_03": "unknown"}
    assert next(b for b in batch if b["id"] == "s01_02").get("player") is True
    system = next(s for s in sys_prompts if "localizer" in s)
    assert "first person" in system and "gender unknown: decide the gender from the scene" in system


def test_a_rechecked_line_keeps_its_speaker_name_and_gender(make_ctx, tmp_path, monkeypatch):
    ctx = _ctx(make_ctx, tmp_path)
    llm = FakeLLM(answer=lambda b: "Я закончила, Kevin" if b["id"] == "s01_01" and "previous_answer" not in b else _long(b))
    _run(ctx, monkeypatch, llm)
    redo = [c for c in llm.batches() if any("previous_answer" in b for b in c)]
    assert redo and redo[0][0]["speaker"] == "Алиса" and redo[0][0]["gender"] == "f"
    assert _records(ctx)["s01_01"].text_tgt.startswith("Перевод s01_01")


# ---------------------------------------------------------------- gender pass (C3)


class GenderLLM(FakeLLM):
    def __init__(self, genders):
        super().__init__()
        self.genders = genders
        self.gender_asks: list[list[dict]] = []

    def chat(self, messages, temperature=0.3, **kw):
        if "genders" in messages[0]["content"]:
            items = json.loads(messages[-1]["content"])
            self.gender_asks.append(items)
            return json.dumps({"genders": [{"id": i["id"], "gender": self.genders.get(i["id"], "unknown")} for i in items]})
        return super().chat(messages, temperature, **kw)


def test_the_gender_pass_fills_unknown_genders_and_never_a_users(make_ctx, tmp_path, monkeypatch):
    rows = [
        dict(line_id="s01_01", speaker_id="mira", speaker_raw="Mira", text="I am ready.", seconds=1.0),
        dict(line_id="s01_02", speaker_id="kai", speaker_raw="Kai", text="Mira, she is ready, I said.", seconds=1.0),
        dict(line_id="s01_03", speaker_id="ren", speaker_raw="Ren", text="Fine.", seconds=1.0),
    ]
    ctx = _ctx(make_ctx, tmp_path, rows=rows, gloss={"player": {"gender": "unknown"}, "characters": {}, "terms": []}, **{"translate.batch_lines": 5})
    from vndub.manifest.io import save_characters

    chars = load_characters(ctx.paths.characters)
    chars["ren"].gender = "m"
    setattr(chars["ren"], "gender_source", "user")
    save_characters(ctx.paths.characters, chars)
    llm = GenderLLM({"mira": "f", "kai": "m", "ren": "f"})
    summary = _run(ctx, monkeypatch, llm)
    (asked,) = llm.gender_asks
    assert [i["id"] for i in asked] == ["mira", "kai"]  # ren has the user's gender
    assert next(i for i in asked if i["id"] == "mira")["mentions"] == ["Mira, she is ready, I said."]
    assert summary["gender_guessed"] == {"mira": "f", "kai": "m"}
    chars = load_characters(ctx.paths.characters)
    assert chars["mira"].gender == "f" and chars["mira"].model_extra["gender_source"] == "guess" and chars["ren"].gender == "m"
    (batch,) = llm.batches()
    assert {b["id"]: b["gender"] for b in batch} == {"s01_01": "f", "s01_02": "m", "s01_03": "m"}  # the same run's prompt
    llm2 = GenderLLM({})
    assert _run(ctx, monkeypatch, llm2)["gender_guessed"] == {"mira": "f", "kai": "m"} and not llm2.gender_asks and not llm2.batches()


# ---------------------------------------------------------------- 3: names pass hygiene, word-boundary terms


def test_name_candidates_leave_out_address_words_interjections_and_words_of_terms():
    script = [
        "I went to the Mansion yesterday.", "Professor Oak is here.", "Did you see Professor Oak?", "Mom said no.", "We speak English here.",
        "Okay, Kevin.", "Thanks, Kevin.", "Let's go to Tokyo!", "Nope.", "Nope, Kevin.", "Oops.", "Oops, sorry.", "Darling, welcome back.",
        "Darling!", "Yay!", "Yay, we won!", "Honey, I'm home.", "Shit.", "Shit, I forgot.", "The Scarlet Devil Mansion is red.", "Whoa.", "Whoa, Kevin!",
    ]
    cands = translate.collect_name_candidates(script, ["Scarlet Devil Mansion"])
    assert cands == ["Kevin", "Oak", "Tokyo"]


def test_terms_match_whole_words_and_short_names_decline():
    terms = [{"src": "Ann", "tgt": "Энн"}, {"src": "Oak", "tgt": "Оук"}, {"src": "Mia", "tgt": "Мия"}, {"src": "さくら", "tgt": "Сакура"}]
    assert translate.glossary_missing("We are planning to soak in the bath, Annoying.", "Мы примем ванну.", terms) == []
    assert translate.glossary_missing("I met Mia.", "Я встретил Мию.", terms) == [] and translate.glossary_missing("With Mia.", "С Мией.", terms) == []
    assert translate.glossary_missing("I met Mia.", "Я встретил её.", terms) == ["Mia"]
    assert translate.glossary_missing("さくらちゃん！", "Сакуре!", terms) == []  # Japanese: a substring match
    prompt = translate.system_prompt("ru", [{"text": "Annoying! The Oaks are tall. ann was here."}], [], {"terms": terms}, {}, 8, src="en")
    assert "Энн" in prompt and "Оук" not in prompt


# ---------------------------------------------------------------- 8 / 4: checks, summary ids, stress unavailable


def test_new_checks():
    chk = translate.check_translation
    assert chk("Hello there, how are you doing today, my old friend?", "Привет.", "ru", "en") == "omitted"
    assert chk("It's 5 o'clock and I have 3 cats.", "Сейчас пять часов, у меня кошки.", "ru", "en") == "lost_number"
    assert chk("It's 5 o'clock and I have 3 cats.", "Сейчас пять часов, у меня три кошки.", "ru", "en") == ""
    assert chk("Hello, [player_name]!", "Привет, [игрок]!", "ru", "en") == "lost_token"
    assert chk("Hello, [player_name]!", "Привет, [player_name]!", "ru", "en") == ""
    assert chk("Hi, Kevin!", "Привет, Kevin!", "ru", "en") == "latin_run"
    assert chk("Wait!", "Wait!", "ru", "en") == "untranslated"
    assert chk("OK.", "OK.", "ru", "en") == ""  # an abbreviation is spelled by the TTS normalizer
    assert chk("Hi, Kevin!", "Привет, Kevin!", "ru", "en", [{"src": "Kevin", "tgt": "Kevin"}]) == ""  # the user's own spelling
    assert chk("テストで１００点！", "Сто баллов!", "ru", "ja") == ""  # a number in words; a Japanese source has no omission check


def test_summary_names_glossary_misses_and_check_failures(make_ctx, tmp_path, monkeypatch):
    ctx = _ctx(make_ctx, tmp_path, gloss={**GLOSS, "terms": [{"src": "Patchouli", "tgt": "Пачули"}]})
    llm = FakeLLM(answer=lambda b: "Привет, Kevin " + _long(b) if b["id"] == "s01_01" else _long(b))
    s = _run(ctx, monkeypatch, llm)
    assert s["check_failed"] == 1 and s["check_failed_ids"] == ["s01_01"]
    assert s["glossary_misses"] == 2 and s["glossary_miss_ids"] == ["s01_02", "s01_03"]
    recs = _records(ctx)
    assert "glossary_miss" in recs["s01_02"].flags and "check_failed" in recs["s01_01"].flags
    fix = [b["fix"] for c in llm.batches() for b in c if b["id"] == "s01_02" and "fix" in b]
    assert fix and "Patchouli -> Пачули" in fix[0]


def test_missing_stress_is_recorded_as_unavailable(make_ctx, tmp_path, monkeypatch):
    ctx = _ctx(make_ctx, tmp_path)
    monkeypatch.setattr(translate, "make_stresser", lambda c, texts: (None, "ruaccent", "ruaccent env missing"))
    seen = []
    monkeypatch.setattr(translate.gu, "record_unavailable", lambda c, stage, group, reason: seen.append((stage, group, reason)))
    _run(ctx, monkeypatch, FakeLLM())
    assert seen == [("translate", "stress", "ruaccent env missing")]


# ---------------------------------------------------------------- 9 / 10: prompt rules and diet


def _prompt(batch, gloss=None, src="en", chars=None):
    return translate.system_prompt("ru", batch, sorted({b.get("speaker_id", "") for b in batch}), gloss or {}, chars or {}, 8, src=src)


def test_prompt_diet():
    speech = [{"id": "a", "speaker_id": "alice", "text": "I finally finished the long report for the club."}]
    p = _prompt(speech, {"honorifics": "drop", "style_rules": [*translate.DEFAULT_RULES, "Keep character names exactly as listed in the glossary."],
                         "address_matrix": {"alice": {"default": "ты", "bob": "вы"}}, "characters": {"alice": {"tgt": "Алиса"}, "bob": {"tgt": "Боб"}}})
    assert "non-speech sound" not in p  # no sound candidate in the batch
    assert "Keep character names exactly as listed in the glossary" not in p  # the duplicate of the craft rule
    assert "Honorifics" not in p  # an English line without -san
    assert "Алиса says 'ты' to everyone, 'вы' to Боб" in p and "alice:" not in p
    assert "'ru_short' only when the translation has more than 8 words" in p
    assert "Write numbers in words, declined to fit the sentence" in p
    assert "non-speech sound" in _prompt([{"id": "b", "speaker_id": "alice", "text": "ZZZ..."}])
    assert "Honorifics" in _prompt([{"id": "c", "speaker_id": "alice", "text": "Thanks, Sakura-chan."}], {"honorifics": "drop"})
    assert "Honorifics" in _prompt([{"id": "d", "speaker_id": "alice", "text": "ありがとう"}], {"honorifics": "drop"}, src="ja")


def test_rows_have_no_empty_emotion_and_a_one_batch_scene_has_no_summary_call(make_ctx, tmp_path, monkeypatch):
    ctx = _ctx(make_ctx, tmp_path, **{"translate.batch_lines": 5})
    seen: list[str] = []

    class Spy(FakeLLM):
        def chat(self, messages, temperature=0.3, **kw):
            seen.append(messages[0]["content"][:20])
            return super().chat(messages, temperature, **kw)

    llm = Spy()
    _run(ctx, monkeypatch, llm)
    assert not any(s.startswith("Summarize") for s in seen)
    assert all("emotion" not in b for c in llm.batches() for b in c)


# ---------------------------------------------------------------- 6 (translate half): flags follow the user's text after a re-run


def test_a_translate_run_keeps_the_laughter_flags_of_the_users_text(make_ctx, tmp_path, monkeypatch):
    from vndub.review.app import write_override

    ctx = _ctx(make_ctx, tmp_path)
    write_override(ctx.paths, "s01_01", "set_text", {"text": "Ха-ха-ха!"})
    _run(ctx, monkeypatch, FakeLLM())
    assert {"laughter", "interjection"} <= set(_records(ctx)["s01_01"].flags)
    assert "laughter" not in _records(ctx)["s01_02"].flags

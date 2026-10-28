"""Source-language detection: the detector itself, how the stages pick the language up, and the parts of
translate / link / analyze that stop assuming a Japanese script."""

from __future__ import annotations

import json

import yaml

from analysis_helpers import build_project
from vndub.config import ProjectYaml, load_project_yaml, save_project_yaml
from vndub.manifest.io import TranslationRecord, read_jsonl
from vndub.stages import analyze, link, translate
from vndub.stages.common import resolve_src_lang, run_stage
from vndub.text import ja
from vndub.text.lang import count_syllables, detect_language, language_name, speech_units

JA = ["テストで百点を取ったんだよ！", "どうしたの、さくら？", "うん！", "三人は並んで校門を出た。", "ああ、そうしよう。"]
EN = ["I got a perfect score on the test!", "What's wrong, Sakura?", "Yeah!", "The three of them left the school gate together.", "You know it's not like that."]
RU = ["Я получила сто баллов на тесте!", "Что случилось, Сакура?", "Да!", "Они вместе вышли из школы."]
ZH = ["我在考试中得了满分！", "怎么了，樱？", "嗯！", "三个人一起走出了校门。"]
KO = ["시험에서 만점을 받았어!", "무슨 일이야, 사쿠라?", "응!", "세 사람은 나란히 교문을 나섰다."]
DE = ["Ich habe die Prüfung mit voller Punktzahl bestanden!", "Was ist los, Sakura?", "Ja!", "Die drei verließen gemeinsam das Schultor und sie sind nicht müde."]
UK = ["Я отримала сто балів на тесті!", "Що сталося, Сакуро?", "Так!", "Вони разом вийшли зі школи, і їм було весело."]


# ----------------------------------------------------------------------------
# detector
# ----------------------------------------------------------------------------


def test_detects_the_scripts_of_visual_novels():
    assert detect_language(JA).lang == "ja"
    assert detect_language(EN).lang == "en"
    assert detect_language(RU).lang == "ru"
    assert detect_language(ZH).lang == "zh"
    assert detect_language(KO).lang == "ko"
    assert detect_language(DE).lang == "de"
    assert detect_language(UK).lang == "uk"
    g = detect_language(EN * 10)
    assert g.confidence > 0.95 and g.letters > 400 and g.lines == 50 and g.scripts["latin"] == g.letters


def test_markup_names_and_short_samples():
    # Ren'Py / KAG markup and ruby are not letters; a Japanese name inside an English script does not flip it
    assert detect_language(["[ruby text=\"さくら\"]Sakura[/ruby] {i}said{/i} nothing, and that was all she said."]).lang == "en"
    assert detect_language(EN + JA[:1]).lang == "en"
    assert detect_language(JA + EN[:1]).lang == "ja"
    # punctuation only / empty: no guess at all
    g = detect_language(["……", "!!", ""])
    assert g.lang == "" and g.confidence == 0.0
    # a very short sample is a weak guess
    assert detect_language(["Yes."]).confidence < 0.2
    assert detect_language([]).lang == ""


AR = ["لقد حصلت على الدرجة الكاملة في الاختبار!", "ما الخطب يا ساكورا؟", "نعم!", "خرج الثلاثة من بوابة المدرسة معا."]
TH = ["ฉันได้คะแนนเต็มในการสอบ!", "เกิดอะไรขึ้นหรือซากุระ?", "ใช่!", "ทั้งสามคนเดินออกจากประตูโรงเรียนด้วยกัน"]
EL = ["Πήρα άριστα στο διαγώνισμα!", "Τι συμβαίνει, Σακούρα;", "Ναι!", "Οι τρεις βγήκαν μαζί από την πύλη του σχολείου."]
HI = ["मुझे परीक्षा में पूरे अंक मिले!", "क्या हुआ, सकुरा?", "हाँ!", "तीनों साथ में स्कूल के गेट से बाहर निकले।"]


def test_unlisted_scripts_are_unknown_not_russian():
    # zero Latin and zero Cyrillic letters used to tie-break to "ru" with confidence 0.0
    for texts in (AR, TH, EL, HI):
        g = detect_language(texts * 3)
        assert g.lang == "", (texts[0], g)
    assert detect_language(AR + EN[:1]).lang == ""  # one Latin line among Arabic does not make it English
    assert detect_language(EN * 3 + AR[:1]).lang == "en"  # and one Arabic line does not unmake English


def test_units_and_names():
    assert speech_units(JA[0], "ja") == ja.count_morae(JA[0])
    assert speech_units(JA[0], "auto") == ja.count_morae(JA[0])
    syl = count_syllables(EN[0], "en")
    assert 9 <= syl <= 11  # I got a per-fect score on the test (silent e counts as a group: an estimate)
    assert speech_units(EN[0], "en") == round(syl * 1.5)
    assert count_syllables(RU[0], "ru") == 11
    assert count_syllables("Sakura-san, 3 o'clock.", "en") == 8
    assert count_syllables("", "en") == 0 and speech_units("...", "en") == 0
    assert language_name("en") == "English" and language_name("xx") == "xx"


# ----------------------------------------------------------------------------
# the stages
# ----------------------------------------------------------------------------


def rows(texts: list[str], speaker: str = "sakura") -> list[dict]:
    return [dict(line_id=f"s01_{i:02d}", speaker_id=speaker, speaker_raw="Sakura", text=t, seconds=0.5) for i, t in enumerate(texts, 1)]


def test_context_detects_and_persists_the_language(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path)
    save_project_yaml(ctx.paths.project_yaml, ProjectYaml(game_dir=str(tmp_path), engine="renpy"))
    assert load_project_yaml(ctx.paths.project_yaml).src_lang == "auto"
    assert ctx.src_lang == "ja"  # nothing parsed yet: the legacy default, nothing written
    assert load_project_yaml(ctx.paths.project_yaml).src_lang == "auto"
    build_project(ctx, rows(EN))
    assert ctx.src_lang == "en" and ctx.audio_lang == "en"
    proj = load_project_yaml(ctx.paths.project_yaml)
    assert proj.src_lang == "en" and proj.src_lang_detected["lang"] == "en" and proj.src_lang_detected["confidence"] > 0.9
    # the detected value is stable on later reads and re-detected on request (parse does that over new lines)
    build_project(ctx, rows(JA))
    assert ctx.src_lang == "en"
    assert resolve_src_lang(ctx, redetect=True) == "ja"
    assert load_project_yaml(ctx.paths.project_yaml).src_lang == "ja"
    # a code the user set is never replaced
    proj = load_project_yaml(ctx.paths.project_yaml)
    proj.src_lang, proj.src_lang_detected = "en", None
    save_project_yaml(ctx.paths.project_yaml, proj)
    assert resolve_src_lang(ctx, redetect=True) == "en" and ctx.src_lang == "en"
    # the voices may be in another language than the text
    proj.audio_lang = "ja"
    save_project_yaml(ctx.paths.project_yaml, proj)
    assert ctx.audio_lang == "ja" and ctx.src_lang == "en"


def test_unknown_script_keeps_the_configured_language(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path, sets={"project.src_lang": "auto"})
    save_project_yaml(ctx.paths.project_yaml, ProjectYaml(game_dir=str(tmp_path), engine="renpy"))
    build_project(ctx, rows(AR * 3))
    # nothing detected: the legacy default comes back and project.yaml is not rewritten with a wrong code
    assert ctx.src_lang == "ja"
    proj = load_project_yaml(ctx.paths.project_yaml)
    assert proj.src_lang == "auto" and not proj.src_lang_detected
    # an earlier detection survives a re-parse of a script the detector cannot read
    build_project(ctx, rows(EN))
    assert ctx.src_lang == "en"
    build_project(ctx, rows(TH * 3))
    assert resolve_src_lang(ctx, redetect=True) == "en"
    assert load_project_yaml(ctx.paths.project_yaml).src_lang == "en"


def test_translate_english_script(make_ctx, tmp_path):
    d = tmp_path / "dict.yaml"
    d.write_text(yaml.safe_dump({EN[0]: "Я получила сто баллов на тесте!", EN[2]: "Ага!"}, allow_unicode=True), encoding="utf-8")
    ctx = make_ctx(tmp_path, sets={"translate.mock_dictionary": str(d), "translate.glossary": str(tmp_path / "g.yaml")})
    build_project(ctx, rows(EN[:3]))
    summary = run_stage("translate", ctx)
    recs = {r.line_id: r for r in read_jsonl(ctx.paths.translation("ru"), TranslationRecord)}
    assert summary["src_lang"] == "en" and summary["tgt_lang"] == "ru" and summary["interjections"] == 0
    assert recs["s01_01"].text_tgt.startswith("Я получила") and recs["s01_01"].source == "llm:mock_translator"
    assert recs["s01_03"].text_tgt == "Ага!" and "interjection" not in recs["s01_03"].flags  # "Yeah!" is not a kana line
    ctx.models.close()


def test_translate_same_language_keeps_the_text(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path, sets={"project.tgt_lang": "en", "translate.glossary": str(tmp_path / "g.yaml")})
    build_project(ctx, rows(EN[:2]))
    summary = run_stage("translate", ctx)
    recs = {r.line_id: r for r in read_jsonl(ctx.paths.translation("en"), TranslationRecord)}
    assert summary["backend"] == "same_lang" and summary["sources"] == {"same_lang": 2} and summary["failed"] == 0
    assert recs["s01_01"].text_tgt == EN[0] and recs["s01_01"].text_tts == EN[0]
    ctx.models.close()


def test_prompts_name_the_source_language(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path)
    tr = translate.LlmTranslator(ctx, None, "fake", "ru", {}, {}, src="en")
    batch = [{"id": "a", "speaker": "A", "speaker_id": "a", "gender": "f", "emotion": "", "text": "Hello"}]
    system = tr.messages(batch, [], "")[0]["content"]
    assert "from English into Russian" in system and "Japanese visual novel" not in system
    assert "from Japanese into Russian" in translate.system_prompt("ru", batch, ["a"], {}, {}, 8)
    assert "English line" in translate.delivery_system_prompt(batch, ["a"], {}, {}, src="en")
    assert translate._src_keys("en")[-2:] == ("en", "english") and "ja" in translate._src_keys("ja")


def test_truncated_answers_keep_their_complete_lines():
    cut = '{"lines": [{"id": "1", "ru": "да", "ru_short": "да", "notes": ""}, {"id": "2", "ru": "нет"}, {"id": "3", "ru": "обре'
    got = translate.parse_response(cut, ["1", "2", "3"], "ru")
    assert set(got) == {"1", "2"} and got["2"]["text_tgt"] == "нет"
    assert translate.parse_response("```json\n" + cut, ["1", "2", "3"], "ru").keys() == {"1", "2"}
    assert translate._salvage_rows('[{"id": "1", "ru": "да"}, {"id": "2", "ru": "н') == [{"id": "1", "ru": "да"}]
    assert translate._salvage_rows("nothing here") == [] and translate._salvage_rows('{"lines": "x"}') == []
    assert translate._json_object('Sure: {"summary": "two sentences."} done') == {"summary": "two sentences."}
    assert translate._json_object("") is None and translate._json_object("{broken") is None


class _Truncating:
    """Answers the first batch with a cut-off JSON and records what the translator logged about it."""

    def __init__(self) -> None:
        self.calls = 0
        self.last: dict = {}

    def chat(self, messages, temperature=0.3, **kw):
        batch = json.loads(messages[-1]["content"].split("\n", 1)[1])
        self.calls += 1
        rows = [{"id": b["id"], "ru": f"перевод {b['id']}", "ru_short": "", "notes": ""} for b in batch]
        text = json.dumps({"lines": rows}, ensure_ascii=False)
        if self.calls == 1:
            self.last = {"finish_reason": "length", "reasoning_chars": 0, "usage": {}}
            return text[: len(text) // 2]
        self.last = {"finish_reason": "stop", "reasoning_chars": 0, "usage": {}}
        return text


def test_truncated_batch_is_completed_and_logged(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path)
    llm = _Truncating()
    tr = translate.LlmTranslator(ctx, llm, "fake", "ru", {}, {})
    ids = [f"l{i}" for i in range(8)]
    got = tr.translate([{"id": i, "speaker": "A", "speaker_id": "a", "gender": "f", "emotion": "", "text": "テキスト"} for i in ids], [], "")
    assert set(got) == set(ids) and llm.calls == 2 and tr.truncated == 1 and tr.failures == 1
    log = (ctx.paths.logs / translate.ANSWER_LOG).read_text(encoding="utf-8").splitlines()
    row = json.loads(log[-1])
    assert row["kind"] == "translate" and row["finish_reason"] == "length" and row["covered"] < 8 and row["content"].startswith('{"lines"')


def test_link_duration_check_in_english_and_backend_gating():
    text = "I got a perfect score on the test!"
    ratio, units, bad = link.duration_check(text, 2.0, 3.0, 13.0, 1.0, "en")
    assert units == speech_units(text, "en") and not bad and 0.5 < ratio < 2.0
    assert link.duration_check(text, 0.3, 3.0, 13.0, 1.0, "en")[2] is True
    assert link.duration_check("今日はいい天気ですね", 2.0, 3.0, 13.0, 1.0)[1] == ja.count_morae("今日はいい天気ですね")
    ok, notes = analyze.alignment_backends(["whisperx_ja", "qwen3_aligner"], "en")
    assert ok == ["qwen3_aligner"] and "ja aligner" in notes[0]
    assert analyze.alignment_backends(["whisperx_ja", "qwen3_aligner"], "ja")[0] == ["whisperx_ja", "qwen3_aligner"]


def test_link_with_english_text_uses_english_units(make_ctx, tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path, sets={"link.mode": "no_asr"})
    save_project_yaml(ctx.paths.project_yaml, ProjectYaml(game_dir=str(tmp_path), engine="renpy"))
    build_project(ctx, rows(EN[:3]))
    summary = run_stage("link", ctx)
    assert summary["src_lang"] == "en" and summary["audio_lang"] == "en" and "note" not in summary
    recs = {r["line_id"]: r for r in read_jsonl(ctx.paths.link)}
    assert recs["s01_01"]["signals"]["morae"] == speech_units(EN[0], "en")
    ctx.models.close()


def test_link_voices_in_another_language_than_the_text(make_ctx, tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path, sets={"link.mode": "asr"})
    save_project_yaml(ctx.paths.project_yaml, ProjectYaml(game_dir=str(tmp_path), engine="renpy", audio_lang="ja"))
    build_project(ctx, rows(EN[:3]))

    def boom(*a, **k):
        raise AssertionError("ASR started although its text cannot be compared with the script")

    monkeypatch.setattr(link, "run_asr", boom)
    summary = run_stage("link", ctx)
    assert summary["mode"] == "no_asr" and summary["audio_lang"] == "ja" and "voices are ja" in summary["note"]
    ctx.models.close()


def test_analyze_skips_the_aligner_when_the_voices_speak_another_language(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path)
    save_project_yaml(ctx.paths.project_yaml, ProjectYaml(game_dir=str(tmp_path), engine="renpy", audio_lang="ja"))
    build_project(ctx, rows(EN[:2]))
    summary = run_stage("analyze", ctx)
    assert summary["lines"] == 2 and "nothing to align" in summary["unavailable"]["align"] and summary["backends"]["align"] == "unavailable"
    ctx.models.close()

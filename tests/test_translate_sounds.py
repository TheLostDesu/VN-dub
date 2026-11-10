"""Translate stage: non-speech sounds (snoring, kisses ...) named by the LLM or found by the rules."""

from __future__ import annotations

import json

import yaml

from analysis_helpers import build_project
from vndub.manifest.io import TranslationRecord, read_jsonl
from vndub.stages import translate
from vndub.stages.common import run_stage
from vndub.text.delivery import merge_layers, rule_pass

LINES = [
    ("l1", "I'm so sleepy."),
    ("l2", "ZZZ..."),
    ("l3", "Muah!"),
    ("l4", "Brrr, it's cold."),
    ("l5", "*yawns* Good morning."),
    ("l6", "Haha! ZZZ..."),
]
DICT = {
    "I'm so sleepy.": "Я так хочу спать.",
    "ZZZ...": "З-З-З...",
    "Muah!": "Му-ах!",
    "Brrr, it's cold.": "Бр-р, как холодно.",
    "*yawns* Good morning.": "*зевает* Доброе утро.",
    "Haha! ZZZ...": "Ха-ха! Хр-р-р...",
}


def _run(make_ctx, tmp_path):
    d = tmp_path / "dict.yaml"
    d.write_text(yaml.safe_dump(DICT, allow_unicode=True), encoding="utf-8")
    ctx = make_ctx(tmp_path, sets={"translate.mock_dictionary": str(d), "translate.glossary": str(tmp_path / "g.yaml")})
    build_project(ctx, [dict(line_id=lid, speaker_id="sakura", speaker_raw="Sakura", text=t, seconds=0.5) for lid, t in LINES])
    summary = run_stage("translate", ctx)
    recs = {r.line_id: r for r in read_jsonl(ctx.paths.translation("ru"), TranslationRecord)}
    return ctx, summary, recs


def _prompt(tgt="ru"):
    batch = [{"id": "a", "text": "ZZZ...", "speaker_id": "sakura"}]
    return translate.system_prompt(tgt, batch, ["sakura"], {}, {}, 8, False, "en")


def test_prompt_has_the_sound_rule():
    p = _prompt()
    for needle in ("non-speech sound", "'sound'", '"only": true|false', "snore (ZZZ", "kiss (muah", "yawn", "gasp", "sneeze (achoo", "cough (ahem",
                   "hum (mmm", "hmph", "shiver (brr", "eat (nom nom", "stage direction", "Laughter is not a sound"):
        assert needle in p, needle
    assert "standard Russian spelling" in p
    assert translate.TRANSLATE_PROTOCOL == "4"


def test_delivery_prompt_does_not_offer_sound_events():
    # the delivery object keeps its five speech events; the sound goes in its own field
    doc = translate._delivery_object_doc()
    assert "laugh|sigh|cry|breath|whisper" in doc and "snore" not in doc


def test_parse_response_keeps_a_valid_sound_and_drops_junk():
    content = json.dumps(
        {
            "lines": [
                {"id": "a", "ru": "Хр-р-р...", "sound": {"kind": "snore", "only": True}},
                {"id": "b", "ru": "Бр-р, холодно.", "sound": {"kind": "Shiver", "only": False}},
                {"id": "c", "ru": "Привет.", "sound": {"kind": "teleport", "only": True}},
                {"id": "d", "ru": "Пока.", "sound": "nonsense"},
                {"id": "e", "ru": "Чмок!", "sound": "kiss"},
                {"id": "f", "ru": "Да.", "sound": 7},
                {"id": "g", "ru": "Нет."},
            ]
        },
        ensure_ascii=False,
    )
    got = translate.parse_response(content, list("abcdefg"), "ru")
    assert got["a"]["sound"] == {"kind": "snore", "only": True}
    assert got["b"]["sound"] == {"kind": "shiver", "only": False}
    assert got["e"]["sound"] == {"kind": "kiss", "only": False}
    for k in "cdfg":
        assert "sound" not in got[k] and got[k]["text_tgt"]  # junk never invalidates the translation


def test_sound_record_prefers_the_model_then_the_rules():
    e = {"text_tgt": "Что-то.", "sound": {"kind": "kiss", "only": True}}
    assert translate.sound_record("Muah!", e, "en", "ru") == {"kind": "kiss", "only": True, "stage": False, "source": "llm"}
    # only:true on real speech is not believed
    liar = {"text_tgt": "Я иду домой.", "sound": {"kind": "snore", "only": True}}
    assert translate.sound_record("I am going home now, see you.", liar, "en", "ru")["only"] is False
    # rules: the source, then the translation
    assert translate.sound_record("ZZZ...", {"text_tgt": "Спит."}, "en", "ru") == {"kind": "snore", "only": True, "stage": False, "source": "rule"}
    assert translate.sound_record("Sleeping.", {"text_tgt": "Хр-р-р..."}, "en", "ru")["kind"] == "snore"
    assert translate.sound_record("Hello.", {"text_tgt": "Привет."}, "en", "ru") is None


def test_spell_sound_line_keeps_the_lines_punctuation():
    table = {"langs": {"ru": {"snore": "Хр-р-р... хр-р-р...", "kiss": "Чмок", "yawn": "А-а-ах..."}}}
    assert translate.spell_sound_line(table, "ru", "snore", "x") == "Хр-р-р... хр-р-р..."
    assert translate.spell_sound_line(table, "ru", "kiss", "Му-ах!", "Muah!") == "Чмок!"
    assert translate.spell_sound_line(table, "ru", "kiss", "Му-ах", "Muah?") == "Чмок?"
    assert translate.spell_sound_line(table, "ru", "kiss", "Му-ах", "Muah") == "Чмок"
    assert translate.spell_sound_line(table, "ru", "eat", "x") is None


def test_stage_run_flags_spells_and_summarises(make_ctx, tmp_path):
    ctx, summary, recs = _run(make_ctx, tmp_path)
    assert summary["failed"] == 0
    # speech: nothing
    assert recs["l1"].sound is None and recs["l1"].flags == []
    # ZZZ... -> snoring, spelled by the table, stress bypassed, interjection
    z = recs["l2"]
    assert z.sound == {"kind": "snore", "only": True, "stage": False, "source": "rule"}
    assert z.flags == ["sound", "interjection", "sound_spelled"]
    # round 18: the screen text (text_tgt, the Ren'Py text patch) keeps the model's translation; only the TTS fields are spelled
    assert z.text_tgt == z.text_tgt_short == "З-З-З..." and z.model_extra["sound_spelling"] == "Хр-р-р... хр-р-р..."
    assert z.text_tts == z.text_tts_short == z.text_plain == "Хр-р-р... хр-р-р..."
    assert "́" not in z.text_tts and "+" not in z.text_tts
    # Muah! -> kiss, the line's "!" survives
    k = recs["l3"]
    assert k.sound["kind"] == "kiss" and k.sound["only"] is True
    assert k.text_tgt == "Му-ах!" and k.text_tts == "Чмок!" and "sound_spelled" in k.flags and "interjection" in k.flags
    # a sound inside speech: flagged, the model's translation is left alone
    b = recs["l4"]
    assert b.sound["kind"] == "shiver" and b.sound["only"] is False
    assert b.flags == ["sound"] and b.text_tgt == "Бр-р, как холодно."
    # a stage direction in front of speech
    y = recs["l5"]
    assert y.sound["kind"] == "yawn" and y.sound["stage"] is True and y.sound["only"] is False
    assert y.flags == ["sound"] and y.text_tgt == "*зевает* Доброе утро."
    # a laugh and a sound in one line: both flags, not only
    m = recs["l6"]
    assert m.sound["kind"] == "snore" and m.sound["only"] is False
    assert "laughter" in m.flags and "sound" in m.flags and "interjection" not in m.flags
    assert summary["sounds"] == 5 and summary["sounds_spelled"] == 2
    # the delivery of the snoring line, end to end
    d = merge_layers([rule_pass("ZZZ...", z.text_tgt, None, "en")])
    assert d.sound["kind"] == "snore" and d.sound["only"] is True and "snore" in d.events
    ctx.models.close()


def test_sounds_table_is_read_defensively(tmp_path):
    import logging
    from types import SimpleNamespace

    def fake(table):
        cfg = SimpleNamespace(synth=SimpleNamespace(sounds=SimpleNamespace(table=table)) if table is not None else SimpleNamespace(),
                              resolve=lambda p: tmp_path / p if not str(p).startswith("configs") else translate.sounds.default_table_path())
        return SimpleNamespace(cfg=cfg, log=logging.getLogger("t"))

    assert translate.sounds_table(fake(None))["langs"]["ru"]["snore"]  # no synth.sounds section yet: the repo table
    assert translate.sounds_table(fake("nope.yaml")) is None  # a missing file: no spelling, no crash
    (tmp_path / "bad.yaml").write_text("ru: {nonsense: x}", encoding="utf-8")
    assert translate.sounds_table(fake("bad.yaml")) is None

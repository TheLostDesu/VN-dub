"""vndub.text.laughter: the shared laughter core (detection, kinds, laughter-only lines, stripping, respelling)."""

from __future__ import annotations

import pytest

from vndub.text import laughter as lf


@pytest.mark.parametrize(
    "text,kind,units",
    [
        ("Haha!", "hearty", 2),
        ("Hahaha, you really did it.", "hearty", 3),
        ("Ha ha ha!", "hearty", 3),
        ("Ha-ha. Very funny.", "hearty", 2),
        ("Ahahaha!", "hearty", 4),
        ("Hehe.", "chuckle", 2),
        ("Heheh.", "chuckle", 2),
        ("Heh.", "chuckle", 1),
        ("Hihihi~", "giggle", 3),
        ("Ufufu~", "giggle", 2),
        ("Fufufu.", "giggle", 3),
        ("Teehee!", "giggle", 2),
        ("Hee hee hee.", "giggle", 3),
        ("Hohoho!", "hearty", 3),
        ("Mwahahaha!", "evil", 3),
        ("Bwahaha!", "evil", 2),
        ("Kukuku...", "evil", 3),
        ("Ehehe...", "nervous", 3),
        ("Ahaha... right.", "nervous", 3),
        ("Pfft.", "snort", 1),
        ("lol", "hearty", 1),
    ],
)
def test_latin_laughs(text: str, kind: str, units: int) -> None:
    ls = lf.find_laughs(text)
    assert len(ls) == 1, ls
    assert ls[0].kind == kind
    assert ls[0].units == units
    assert ls[0].lang == "en" and not ls[0].stage


@pytest.mark.parametrize("text", ["Hannah has a hat.", "Aha, I see.", "The chaos is here.", "Mahalo!", "He hesitated.", "Hahn", "Ha!", "Hey!", "Heh-llo? no", "uh-huh", "huh"])
def test_latin_no_false_positives(text: str) -> None:
    assert lf.find_laughs(text) == []


@pytest.mark.parametrize(
    "text,kind",
    [
        ("*laughs*", "hearty"),
        ("(laughs softly)", "hearty"),
        ("(giggles)", "giggle"),
        ("(chuckles)", "chuckle"),
        ("(snickers)", "chuckle"),
        ("(cackles)", "evil"),
        ("[snorts]", "snort"),
    ],
)
def test_latin_stage_directions(text: str, kind: str) -> None:
    ls = lf.find_laughs("Oh. " + text + " Right.")
    assert len(ls) == 1 and ls[0].stage and ls[0].kind == kind and ls[0].units == 0


@pytest.mark.parametrize(
    "text,kind,units",
    [
        ("Ха-ха!", "hearty", 2),
        ("Ха-ха-ха, ну ты даёшь.", "hearty", 3),
        ("Ахаха!", "hearty", 3),
        ("Ахахаха!", "hearty", 4),
        ("Хахаха.", "hearty", 3),
        ("Хе-хе.", "chuckle", 2),
        ("Хех.", "chuckle", 1),
        ("Хи-хи-хи~", "giggle", 3),
        ("Хихи.", "giggle", 2),
        ("Хо-хо-хо!", "hearty", 3),
        ("Муахаха!", "evil", 2),
        ("Бугага!", "hearty", 2),
        ("Гы-гы.", "chuckle", 2),
        ("Пфф.", "snort", 1),
        ("Ха-ха... ну да.", "nervous", 2),
    ],
)
def test_ru_laughs(text: str, kind: str, units: int) -> None:
    ls = lf.find_laughs(text)
    assert len(ls) == 1, ls
    assert ls[0].kind == kind and ls[0].units == units and ls[0].lang == "ru"


@pytest.mark.parametrize("text", ["Кхе-кхе.", "Ха!", "Хам.", "Хе.", "Ахинея какая-то.", "Пахать и пахать.", "Хихикать не надо.", "Ухо."])
def test_ru_no_false_positives(text: str) -> None:
    assert lf.find_laughs(text) == []


@pytest.mark.parametrize("text,kind", [("(смеётся)", "hearty"), ("(хихикает)", "giggle"), ("(усмехается)", "chuckle"), ("(фыркает)", "snort"), ("*хохочет*", "hearty"), ("(со смехом)", "hearty")])
def test_ru_stage_directions(text: str, kind: str) -> None:
    ls = lf.find_laughs("Да. " + text + " Конечно.")
    assert len(ls) == 1 and ls[0].stage and ls[0].kind == kind


@pytest.mark.parametrize(
    "text,kind",
    [
        ("あはは、ごめんね", "hearty"),
        ("ははは！", "hearty"),
        ("アハハハ！", "hearty"),
        ("ふふっ", "giggle"),
        ("うふふ", "giggle"),
        ("くすくす", "giggle"),
        ("えへへ", "nervous"),
        ("くくく…", "evil"),
        ("ぷっ", "snort"),
        ("にしし", "giggle"),
        ("ぎゃははは！", "hearty"),
        ("げらげら", "hearty"),
    ],
)
def test_ja_laughs(text: str, kind: str) -> None:
    ls = lf.find_laughs(text)
    assert ls, text
    assert ls[0].kind == kind and ls[0].lang == "ja"


def test_ja_katakana_span_maps_back_to_original_text() -> None:
    ls = lf.find_laughs("えっ、アハハ！そうなの")
    assert len(ls) == 1 and ls[0].text == "アハハ"


@pytest.mark.parametrize("text,kind", [("（笑）", "hearty"), ("(苦笑)", "nervous"), ("（くすっ）", "giggle")])
def test_ja_stage_directions(text: str, kind: str) -> None:
    ls = lf.find_laughs("そうね" + text)
    assert len(ls) == 1 and ls[0].stage and ls[0].kind == kind


def test_lang_filter_and_auto() -> None:
    text = "Haha, ха-ха, あはは"
    assert [l.lang for l in lf.find_laughs(text)] == ["en", "ru", "ja"]
    assert [l.lang for l in lf.find_laughs(text, "ru")] == ["ru"]
    assert [l.lang for l in lf.find_laughs(text, "en")] == ["en"]
    assert [l.lang for l in lf.find_laughs(text, "ja")] == ["ja"]
    assert lf.find_laughs("", "auto") == []


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Haha!", True),
        ("Ha ha ha...", True),
        ("Hehe~", True),
        ("Ха-ха-ха!", True),
        ("«Хи-хи.»", True),
        ("ふふっ……", True),
        ("(laughs)", True),
        ("Haha, no way!", False),
        ("Ха-ха, ну да.", False),
        ("Hello.", False),
        ("", False),
        ("...", False),
    ],
)
def test_is_laughter_only(text: str, expected: bool) -> None:
    assert lf.is_laughter_only(text) is expected


def test_strip_laughter() -> None:
    assert lf.strip_laughter("Haha, no way!") == "no way!"
    assert lf.strip_laughter("No way, haha!") == "No way!"
    assert lf.strip_laughter("Ха-ха-ха! Ты серьёзно?") == "Ты серьёзно?"
    assert lf.strip_laughter("Хе-хе.") == ""
    assert lf.strip_laughter("Well... *laughs* fine.") == "Well... fine."
    assert lf.strip_laughter("Nothing funny here.") == "Nothing funny here."


def test_dominant_kind_prefers_spoken_and_longest() -> None:
    ls = lf.find_laughs("(chuckles) Hihi... ahahaha!")
    assert lf.dominant_kind(ls) == "hearty"
    assert lf.dominant_kind(lf.find_laughs("(giggles) yes")) == "giggle"
    assert lf.dominant_kind([]) is None
    assert lf.laughter_kind("Хе-хе, ну ладно.") == "chuckle"


def test_spell_laughter_with_table() -> None:
    table = {
        "hearty": {"short": "Ха-ха!", "long": "Ахахаха!"},
        "giggle": {"any": "Хихихи"},
        "chuckle": "Хе-хе",
    }
    render = lf.table_renderer(table)
    assert lf.spell_laughter("Ха-ха, ну ты даёшь.", render) == "Ха-ха, ну ты даёшь."
    assert lf.spell_laughter("ха-ха-ха! Ты серьёзно?", render) == "Ахахаха! Ты серьёзно?"
    assert lf.spell_laughter("Хи-хи-хи~ милый.", render) == "Хихихи~ милый."
    assert lf.spell_laughter("хех, ладно", render) == "Хе-хе, ладно"
    # a kind without an entry falls back to "hearty"; unknown with nothing at all keeps the spelling
    assert lf.spell_laughter("Пфф.", render) == "Ха-ха."
    assert lf.spell_laughter("Пфф", render) == "Ха-ха!"
    assert lf.spell_laughter("Муахаха!", lf.table_renderer({"giggle": "x"})) == "Муахаха!"
    # stage directions are removed by default and kept with stage=None
    assert lf.spell_laughter("Да. (смеётся) Конечно.", render) == "Да. Конечно."
    assert lf.spell_laughter("Да. (смеётся) Конечно.", lf.table_renderer(table, stage=None)) == "Да. (смеётся) Конечно."
    assert lf.spell_laughter("Да. (смеётся) Конечно.", lf.table_renderer(table, stage="[laughter]")) == "Да. [laughter] Конечно."


def test_spell_laughter_inline_token_renderer() -> None:
    render = lambda l: "[laughter]"  # noqa: E731 - a backend's inline event token for every laugh
    assert lf.spell_laughter("Haha, no way! Hehe.", render) == "[laughter], no way! [laughter]."
    assert lf.spell_laughter("Nothing funny.", render) == "Nothing funny."
    assert lf.spell_laughter("Ха-ха-ха", lambda l: None) == "Ха-ха-ха"


# ----------------------------------------------------------------------------
# additions for the pipeline stages: hyphenated lead-in, "heh heh", summarize, the spelling table, apply_laughter
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,laugh,kind,units",
    [
        ("Муа-ха-ха-ха!", "Муа-ха-ха-ха", "evil", 3),
        ("Mwa-ha-ha-ha!", "Mwa-ha-ha-ha", "evil", 3),
        ("А-ха-ха...", "А-ха-ха", "nervous", 3),  # trails into "...": nervous
        ("Ну а-ха-ха", "а-ха-ха", "hearty", 3),  # the lead-in syllable counts
        ("Пф-ф-ф.", "Пф-ф-ф", "snort", 1),
        ("Heh heh heh.", "Heh heh heh", "chuckle", 3),  # one laugh, not three
        ("Heh heh, fine.", "Heh heh", "chuckle", 2),
        ("heh-heh!", "heh-heh", "chuckle", 2),
        ("Hehehe", "Hehehe", "chuckle", 3),
    ],
)
def test_hyphenated_lead_in_and_repeated_heh_are_one_laugh(text: str, laugh: str, kind: str, units: int) -> None:
    ls = lf.find_laughs(text)
    assert [(l.text, l.kind, l.units) for l in ls] == [(laugh, kind, units)]


def test_summarize() -> None:
    assert lf.summarize("Nothing funny.") is None
    s = lf.summarize("Haha, no way!")
    assert s == {"kind": "hearty", "units": 2, "n": 1, "stage": False, "only": False}
    assert lf.summarize("Ха-ха-ха!")["only"] is True and lf.summarize("Ха-ха-ха!")["units"] == 3
    assert lf.summarize("*laughs*") == {"kind": "hearty", "units": 0, "n": 1, "stage": True, "only": True}
    mixed = lf.summarize("Хи-хи. А-ха-ха-ха! Пф-ф.")
    assert mixed["n"] == 3 and mixed["kind"] == "hearty" and mixed["units"] == 4  # the longest spoken laugh sets the kind
    assert lf.summarize("(giggles) Hello there.")["stage"] is True
    assert set(lf.KIND_NOTE) == set(lf.KINDS) and lf.LAUGH_NOTES == set(lf.KIND_NOTE.values())


# ---- the table -----------------------------------------------------------------

TABLE = {
    "ru": {"hearty": {"short": "Ха-ха!", "long": "Ахахаха!"}, "chuckle": "Хе-хе", "giggle": {"short": "Хи-хи"}, "nervous": {"short": "А-ха..."}},
    "en": {"hearty": {"short": "Ha ha!", "long": "Hahaha!"}, "any": "Ha"},
    "default": {"hearty": "Ha"},
    "backends": {
        "tagged": {"spelling": "en", "en": {"hearty": {"short": "Haha", "long": "Hahaha"}}},
        "plain": {"mode": "keep"},
        "cosy": {"only": "tag", "tokens": {"giggle": "[giggle]"}},
    },
}


def test_load_default_table_and_validation(tmp_path) -> None:
    t = lf.load_table(lf.default_table_path())
    assert {"ru", "en", "ja", "default", "backends"} <= set(t) and set(t["ru"]) == set(lf.KINDS)
    for lang in ("ru", "en", "ja"):
        for kind in lf.KINDS:
            entry = t[lang][kind]
            assert entry["short"] and entry["long"], (lang, kind)
    assert t["stage"] == "remove" and "qwen3_tts" in t["backends"]
    with pytest.raises(FileNotFoundError):
        lf.load_table(tmp_path / "nope.yaml")
    bad = {
        "unknown kind": {"ru": {"cackle": "Ха"}},
        "not a language": {"Russian": {"hearty": "Ха"}},
        "empty spelling": {"ru": {"hearty": ""}},
        "bad entry field": {"ru": {"hearty": {"medium": "Ха"}}},
        "bad stage": {"stage": "keep"},
        "bad mode": {"backends": {"x": {"mode": "loud"}}},
        "bad only": {"backends": {"x": {"only": "keep"}}},
        "unknown backend key": {"backends": {"x": {"colour": "red"}}},
        "bad tokens": {"backends": {"x": {"tokens": {"cackle": "[x]"}}}},
        "backends not a mapping": {"backends": ["x"]},
        "not a mapping": ["ru"],
    }
    for name, data in bad.items():
        with pytest.raises(ValueError):
            lf.validate_table(data)
        p = tmp_path / "t.yaml"
        import yaml

        p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        with pytest.raises(ValueError, match="t.yaml"):
            lf.load_table(p)
    (tmp_path / "broken.yaml").write_text("ru: [unclosed", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid YAML"):
        lf.load_table(tmp_path / "broken.yaml")
    assert lf.validate_table({}) == {}


def test_spelling_table_lookup_order() -> None:
    assert lf.spelling_table(TABLE, "ru")["chuckle"] == "Хе-хе"
    assert lf.spelling_table(TABLE, "de") == {"hearty": "Ha"}  # no block of its own: default
    assert lf.spelling_table(TABLE, "en-US")["hearty"]["short"] == "Ha ha!"  # region suffix
    assert lf.spelling_table({"ru": {"hearty": "x"}}, "ja") is None
    # a backend block overrides per kind; the rest of the language block stays
    over = lf.spelling_table(TABLE, "en", "tagged")
    assert over["hearty"] == {"short": "Haha", "long": "Hahaha"} and over["any"] == "Ha"


def test_parse_decision() -> None:
    assert lf.parse_decision("inline:[laughter]") == ("inline", "[laughter]")
    assert lf.parse_decision("tag:<|sfx:laughter|>") == ("tag", "<|sfx:laughter|>")  # only the first colon splits
    assert lf.parse_decision("respell") == ("respell", "") and lf.parse_decision("") == ("respell", "")
    assert lf.parse_decision("loudly:x") == ("respell", "x")


# ---- apply_laughter: what the TTS text becomes -----------------------------------


def test_apply_respell_short_long_and_punctuation() -> None:
    out, rec = lf.apply_laughter("Ха-ха-ха, ну ты даёшь!", "ru", "respell", TABLE, "qwen3_tts")
    assert out == "Ахахаха, ну ты даёшь!"  # 3 syllables: the long spelling; the comma of the line replaces the mark of the spelling
    assert rec == {"kind": "hearty", "n": 1, "mode": "respell", "only": False, "stage": False}
    assert lf.apply_laughter("Ха-ха, ладно.", "ru", "respell", TABLE)[0] == "Ха-ха, ладно."
    assert lf.apply_laughter("Хе-хе.", "ru", "respell", TABLE)[0] == "Хе-хе."
    assert lf.apply_laughter("Ну... хе-хе... ладно", "ru", "respell", TABLE)[0] == "Ну... А-ха... ладно"  # a chuckle trailing into "..." is nervous
    assert lf.apply_laughter("Ну... хи-хи... ладно", "ru", "respell", TABLE)[0] == "Ну... Хи-хи... ладно"  # a giggle is not
    out, rec = lf.apply_laughter("Hahahaha!", "en", "respell", TABLE)
    assert out == "Hahaha!" and rec["only"] is True
    # the laugh is respelled in the language of the TARGET whatever script the translator wrote
    assert lf.apply_laughter("Haha, ладно", "ru", "respell", TABLE)[0] == "Ха-ха, ладно"
    assert lf.apply_laughter("あはは、ありがとう", "ru", "respell", TABLE)[0] == "Ха-ха、ありがとう"  # the punctuation of the line stays


def test_apply_nothing_to_do() -> None:
    assert lf.apply_laughter("Привет, как дела?", "ru", "respell", TABLE) == ("Привет, как дела?", None)
    assert lf.apply_laughter("", "ru", "respell", TABLE) == ("", None)
    # a language with no block and no default keeps the spelling (nothing to put in its place)
    out, rec = lf.apply_laughter("Ха-ха!", "ko", "respell", {"ru": TABLE["ru"]})
    assert out == "Ха-ха!" and rec["mode"] == "respell"


def test_apply_stage_directions() -> None:
    assert lf.apply_laughter("*laughs* Нет уж.", "ru", "respell", TABLE)[0] == "Нет уж."  # removed, never spoken as words
    assert lf.apply_laughter("Нет уж (смеётся).", "ru", "respell", TABLE)[0] == "Нет уж."
    # next to a spoken laugh a stage direction is always dropped
    assert lf.apply_laughter("(laughs) Ха-ха, нет.", "ru", "respell", TABLE)[0] == "Ха-ха, нет."
    # stage: respell speaks it as a laugh of its kind
    t2 = {**TABLE, "stage": "respell"}
    assert lf.apply_laughter("(giggles) Нет уж.", "ru", "respell", t2)[0] == "Хи-хи Нет уж."
    # a line that is only a stage direction is never sent empty
    out, rec = lf.apply_laughter("*laughs*", "ru", "respell", TABLE)
    assert out == "Ха-ха!" and rec["only"] is True and rec["stage"] is True
    out, rec = lf.apply_laughter("(giggles)", "ru", "keep", TABLE)  # even where the mode would drop it
    assert out == "Хи-хи"
    assert lf.apply_laughter("(смеётся)", "ko", "respell", {"ru": {}}) == ("(смеётся)", {"kind": "hearty", "n": 1, "mode": "keep", "only": True, "stage": True})


def test_apply_inline_token_replaces_the_laugh() -> None:
    cosy = {"any": "[laughter]"}
    out, rec = lf.apply_laughter("Ха-ха-ха, ну ты даёшь!", "ru", "inline:[laughter]", TABLE, "cosyvoice3", cosy)
    assert out == "[laughter] ну ты даёшь!" and rec["mode"] == "inline:[laughter]"  # the comma after an event token is dropped
    assert lf.apply_laughter("Ну, ха-ха, ладно", "ru", "inline:[laughter]", TABLE, "x", cosy)[0] == "Ну, [laughter] ладно"
    assert lf.apply_laughter("Хи-хи. Ладно", "ru", "inline:[laughter]", TABLE, "x", cosy)[0] == "[laughter]. Ладно"
    assert lf.apply_laughter("*laughs* Ладно", "ru", "inline:[laughter]", TABLE, "x", cosy)[0] == "[laughter] Ладно"  # a stage direction is a laugh here
    # a line that is only a laugh: the backend section can ask for the token plus a spelling
    out, rec = lf.apply_laughter("Ха-ха!", "ru", "inline:[laughter]", TABLE, "cosy", cosy)
    assert out == "[laughter] Ха-ха!" and rec["mode"] == "tag:[laughter]" and rec["only"] is True
    # per-kind tokens of the backend section
    assert lf.apply_laughter("Хи-хи, ну что ты", "ru", "inline:[laughter]", TABLE, "cosy", cosy)[0] == "[giggle] ну что ты"


def test_apply_tag_mode_pairs_the_tag_with_the_spelling() -> None:
    fish = {"any": "[laughing]", "chuckle": "[chuckling]", "nervous": "[laughing nervously]"}
    out, rec = lf.apply_laughter("Ха-ха-ха, ну ты даёшь!", "ru", "tag:[laughing]", TABLE, "fish_s2", fish)
    assert out == "[laughing] Ахахаха, ну ты даёшь!" and rec["mode"] == "tag:[laughing]"
    assert lf.apply_laughter("Хе-хе.", "ru", "tag:[laughing]", TABLE, "fish_s2", fish)[0] == "[chuckling] Хе-хе."
    assert lf.apply_laughter("*laughs* Ладно", "ru", "tag:[laughing]", TABLE, "fish_s2", fish)[0] == "[laughing] Ха-ха! Ладно"
    # Higgs: the <|sfx:laughter|> tag is written right against the onomatopoeia, in the language the backend section names
    higgs = {"any": "<|sfx:laughter|>"}
    out, rec = lf.apply_laughter("Ха-ха, ну ты даёшь!", "ru", "tag:<|sfx:laughter|>", TABLE, "tagged", higgs)
    assert out == "<|sfx:laughter|>Haha, ну ты даёшь!" and rec["mode"] == "tag:<|sfx:laughter|>"


def test_apply_backend_section_overrides_the_decision() -> None:
    out, rec = lf.apply_laughter("Ха-ха, ладно (смеётся)", "ru", "inline:[laughter]", TABLE, "plain", {"any": "[laughter]"})
    assert out == "Ха-ха, ладно" and rec["mode"] == "keep"  # spoken laughs stay as written, stage directions still go
    assert lf.laughter_renderer("respell", {"ru": TABLE["ru"], "backends": {"b": {"mode": "inline", "token": "<laugh>"}}}, "ru", "b")[1] == "inline:<laugh>"
    # inline / tag with no token known anywhere falls back to the spelling
    assert lf.laughter_renderer("inline", TABLE, "ru", "nobody")[1] == "respell"


def test_apply_several_laughs_and_capitals() -> None:
    out, rec = lf.apply_laughter("Хе-хе... ха-ха! Пф-ф.", "ru", "respell", {"ru": {"hearty": "Ха", "chuckle": "Хе", "nervous": "А-ха", "snort": "Пф"}})
    assert rec["n"] == 3 and out.count("Ха") == 1
    assert lf.apply_laughter("ну ха-ха-ха", "ru", "respell", TABLE)[0] == "ну Ахахаха!"
    assert lf.apply_laughter("Ха-ха", "ru", "respell", {"ru": {"hearty": "хаха"}})[0] == "Хаха"  # the capital of the original is kept

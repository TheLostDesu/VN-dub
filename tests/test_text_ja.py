from __future__ import annotations

from vndub.text import ja

LEX = {"うん": "угу", "え？": "э?", "え": "э", "ん？": "м?", "えへへ": "хе-хе", "うーん": "хм", "はい": "да"}


def test_normalize_strips_markup_ruby_punct_and_unifies_kana():
    a = ja.normalize_for_match("「ねえ、陽翔《はると》くん……ちょっと待って！」[r][p]")
    assert a == "ねえ陽翔くんちょっと待って"
    assert ja.normalize_for_match("|陽翔《はると》") == "陽翔"
    assert ja.normalize_for_match("{rb}陽翔{/rb}{rt}はると{/rt}") == "陽翔"
    assert ja.normalize_for_match("ネエ　ハルト") == "ねえはると"
    assert ja.normalize_for_match("ＡＢＣ　１２３！？") == "abc123"
    assert ja.normalize_for_match("ふふ～♪") == "ふふ"


def test_normalize_keeps_long_vowel():
    assert ja.normalize_for_match("すごーい！") == "すごーい"


def test_count_morae():
    assert ja.count_morae("ちょっと") == 3  # cho t to
    assert ja.count_morae("きゃー") == 2
    assert ja.count_morae("ん") == 1
    assert ja.count_morae("本当に？") == 5  # 2 kanji x 2 + に
    assert ja.count_morae("OK") == 1
    assert ja.count_morae("【さくら】[r]") == 3
    assert ja.count_morae("") == 0


def test_kata_to_hira():
    assert ja.kata_to_hira("カタカナ・ヴ") == "かたかな・ゔ"


def test_match_interjection_exact_question_and_plain():
    assert ja.match_interjection("え？", LEX) == "э?"
    assert ja.match_interjection("え", LEX) == "э"
    assert ja.match_interjection("うん！", LEX) == "угу"
    assert ja.match_interjection("「えへへ。」", LEX) == "хе-хе"
    assert ja.match_interjection("うーん……", LEX) == "хм"
    # question mark carried over when the lexicon entry has none
    assert ja.match_interjection("うん？", LEX) == "угу?"


def test_match_interjection_repetition_and_misses():
    assert ja.match_interjection("うんうん", LEX) == "угу-угу"
    assert ja.match_interjection("あはは、さくらちゃんったら！", LEX) is None
    assert ja.match_interjection("", LEX) is None
    assert ja.match_interjection("うん", {}) is None


def test_is_interjection():
    assert ja.is_interjection("はい", LEX)
    assert ja.is_interjection("ふぁ", LEX, max_morae=2)  # short kana-only line
    assert not ja.is_interjection("ふぁあぁあん", LEX, max_morae=2)
    assert not ja.is_interjection("本当", LEX, max_morae=2)  # kanji -> not an interjection


def test_load_interjections(tmp_path):
    p = tmp_path / "i.yaml"
    p.write_text("うん: угу\nええ: да\n", encoding="utf-8")
    assert ja.load_interjections(p) == {"うん": "угу", "ええ": "да"}
    assert ja.load_interjections(tmp_path / "missing.yaml") == {}
    assert ja.load_interjections(None) == {}


def test_repo_lexicon_loads(repo):
    lex = ja.load_interjections(repo / "configs" / "interjections.yaml")
    assert lex["うん"] == "угу"
    assert ja.match_interjection("ん？", lex) == "м?"

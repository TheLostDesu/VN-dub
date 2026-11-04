"""text_emotion: rule pass, narration cues, LLM layer and merge, the stage, caching, translate's delivery field,
preview_import and the cue-table helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from analysis_helpers import build_project
from vndub.cli import app
from vndub.manifest.io import Character, LineRecord, Passport, TranslationRecord, read_json, read_jsonl, write_json, write_jsonl
from vndub.stages import text_emotion, translate
from vndub.stages.common import STAGES, run_stage
from vndub.text import delivery as dl
from vndub.text import ja
from vndub.text.style_prompt import StyleSpec, render

# self-contained on purpose: `from conftest import ...` resolves to tests/app/conftest.py in a full-suite run
REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "tests" / "fixtures"
TEST_CONFIG = REPO / "configs" / "pipeline.test.yaml"
requires_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg/ffprobe not on PATH")

GOLDEN = yaml.safe_load((FIXTURES / "text_emotion_golden.yaml").read_text(encoding="utf-8"))
CUES_PATH = REPO / "configs" / "delivery_cues.yaml"
LEXICON = ja.load_interjections(REPO / "configs" / "interjections.yaml")
DEFAULT_CUES = dl.compile_cues(dl.load_cues(CUES_PATH))


# ----------------------------------------------------------------------------
# golden set (rule / cue / merge, pure functions)
# ----------------------------------------------------------------------------


def _scene(case: dict[str, Any]):
    chars = {sid: Character(names={"src": v["src"], "tgt": v["tgt"]}) for sid, v in GOLDEN["characters"].items()}
    lines: list[LineRecord] = []
    tgt: dict[str, str] = {}
    voiced: set[str] = set()
    unvoiced: list[str] = []
    for i, r in enumerate(case["lines"]):
        lid = f"l{i:02d}"
        if r.get("n"):
            sid, raw = "narrator", ""
        else:
            sid, raw = r["sp"], chars[r["sp"]].names["src"]
        lines.append(LineRecord(line_id=lid, scene=f"s{r.get('scene', 1)}", order=i, speaker_raw=raw, speaker_id=sid, text_src=r["text"], has_voice=bool(r.get("voiced"))))
        if r.get("tgt"):
            tgt[lid] = r["tgt"]
        if r.get("voiced"):
            voiced.add(lid)
        elif not r.get("n"):
            unvoiced.append(lid)
            if r.get("target"):
                unvoiced = [lid]
    assert len(unvoiced) >= 1, case["id"]
    return lines, chars, tgt, unvoiced[-1] if len(unvoiced) == 1 else unvoiced[0], voiced


def _check(d: dl.Delivery, expect: dict[str, Any], where: str) -> None:
    for key in ("layer", "emotion", "tempo", "loudness", "arc"):
        if key in expect:
            assert getattr(d, key) == expect[key], f"{where}: {key} {getattr(d, key)!r} != {expect[key]!r}  ({d.to_dict()})"
    if "events" in expect:
        assert d.events == expect["events"], f"{where}: events {d.events} != {expect['events']}"
    for key, attr in (("intensity", d.intensity), ("confidence", d.confidence)):
        if key in expect:
            lo, hi = expect[key]
            assert lo - 1e-6 <= attr <= hi + 1e-6, f"{where}: {key} {attr} not in {expect[key]}  ({d.to_dict()})"
    for c in expect.get("cues", []):
        assert c in d.cues, f"{where}: cue {c!r} missing from {d.cues}"
    for n in expect.get("notes", []):
        assert n in d.notes, f"{where}: note {n!r} missing from {d.notes!r}"


def test_golden_set_is_large_enough():
    assert len(GOLDEN["cases"]) >= 30
    ids = [c["id"] for c in GOLDEN["cases"]]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: c["id"])
def test_golden(case):
    lines, chars, tgt_texts, tid, voiced = _scene(case)
    cfg = case.get("cfg", {})
    hits = text_emotion.attribute_cues(lines, chars, DEFAULT_CUES, "ja", "ru", tgt_texts, cfg.get("narration_window", 3))
    line = next(l for l in lines if l.line_id == tid)
    d = text_emotion.build_delivery(line, hits.get(tid, []), tgt_texts.get(tid, ""), case.get("llm"), LEXICON, cfg.get("min_confidence", 0.6))
    _check(d, case["expect"], case["id"])
    # whatever the evidence, the Delivery renders through StyleSpec for every renderer
    spec = dl.to_style(d)
    assert spec.emotion == d.emotion and 0.0 <= spec.intensity <= 1.0
    render(spec, {"instruct": "natural", "tags": "fish", "knobs": "chatterbox"})


def test_single_period_is_no_evidence():
    r = dl.rule_pass("そうですか。")
    assert r.emotion == "neutral" and r.confidence < 0.6 and r.cues == []
    assert dl.merge_layers([r]).layer == "neutral"


def test_rule_pass_is_deterministic_and_bounded():
    for text in ["！！！！！", "うううううう", "ー" * 40, "…" * 30, "NO NO NO NO！！！", "♪♪♪♪", "ははははははは！？"]:
        a, b = dl.rule_pass(text), dl.rule_pass(text)
        assert a == b
        assert 0.0 <= a.confidence <= 0.95 and 0.1 <= (a.intensity or 0.3) <= 0.9


def test_rule_pass_uses_lexicon_equivalent_when_there_is_no_translation():
    # ふふ -> хи-хи (interjections.yaml): the Russian side contributes laughter
    r = dl.rule_pass("ふふ。", "", LEXICON)
    assert "laugh" in r.events


# ----------------------------------------------------------------------------
# merge, style, validation
# ----------------------------------------------------------------------------


def test_merge_later_layer_needs_confidence():
    rule = dl.Layer("rule", 0.7, emotion="other", intensity=0.55, loudness="loud")
    cue = dl.Layer("cue", 0.5, emotion="sad", intensity=0.2)
    llm = dl.Layer("llm", 0.6, emotion="happy", intensity=0.8, tempo="fast", loudness="normal")
    d = dl.merge_layers([rule, cue], 0.6)
    assert (d.layer, d.emotion, d.loudness) == ("rule", "other", "loud") and d.layers["cue"]["accepted"] is False
    d = dl.merge_layers([rule, cue, llm], 0.6)  # exactly at the threshold counts
    assert (d.layer, d.emotion, d.tempo, d.loudness, d.intensity) == ("llm", "happy", "fast", "normal", 0.8)
    assert dl.merge_layers([rule, cue, llm], 0.61).layer == "rule"
    d = dl.merge_layers([dl.Layer("rule", 0.4, emotion="angry")], 0.6)
    assert (d.layer, d.emotion, d.intensity, d.tempo, d.loudness, d.events) == ("neutral", "neutral", 0.3, "normal", "normal", [])


def test_to_style_uses_the_stylespec_vocabulary():
    d = dl.Delivery(emotion="happy", intensity=0.7, tempo="fast", loudness="loud", events=["laugh", "whisper"], confidence=0.8, layer="rule", arc="rising_end", notes="playful")
    spec = dl.to_style(d, "teasing")
    assert spec.events == ["laughter", "whisper"] and spec.notes == "playful, teasing" and spec.arc == "rising_end"
    assert StyleSpec.from_dict(spec.to_dict()) == spec
    out = render(spec, {"instruct": "natural", "tags": "fish", "knobs": "chatterbox"})
    assert "laugh" in out["instruct"] and "[laughing]" in out["text_prefix"] and out["knobs"]["exaggeration"] > 0.7


def test_validate_delivery_is_strict():
    ok = {"emotion": "Sad", "intensity": 0.5, "tempo": "slow", "loudness": "soft", "events": ["laughter", "cry"], "confidence": 0.9}
    assert dl.validate_delivery(ok) == {"emotion": "sad", "intensity": 0.5, "tempo": "slow", "loudness": "soft", "events": ["laugh", "cry"], "confidence": 0.9}
    assert dl.validate_delivery({**ok, "id": "x"}, "x") is not None and dl.validate_delivery({**ok, "id": "y"}, "x") is None
    bad: list[dict[str, Any]] = [
        {**ok, "emotion": "furious"}, {**ok, "intensity": 1.2}, {**ok, "intensity": -0.1}, {**ok, "intensity": "0.5"}, {**ok, "intensity": True},
        {**ok, "confidence": 2}, {**ok, "tempo": "slowish"}, {**ok, "loudness": "quiet"}, {**ok, "events": ["shout"]}, {**ok, "events": "cry"},
    ]
    for b in bad:
        assert dl.validate_delivery(b) is None, b
    assert dl.validate_delivery(None) is None and dl.validate_delivery("sad") is None
    no_events = {k: v for k, v in ok.items() if k != "events"}
    assert dl.validate_delivery(no_events)["events"] == []
    assert dl.validate_delivery({k: v for k, v in ok.items() if k != "tempo"}) is None


# ----------------------------------------------------------------------------
# cue table: content, load_cues / save_cues
# ----------------------------------------------------------------------------


def test_default_cue_table_content():
    table = dl.load_cues(CUES_PATH)
    assert len(table["ja"]) >= 40 and len(table["ru"]) >= 40
    ja_cues = {e["cue"] for e in table["ja"]}
    ru_cues = {e["cue"] for e in table["ru"]}
    assert {"叫んだ", "ささやいた", "怒鳴る", "泣きながら", "笑いながら"} <= ja_cues
    assert {"воскликнул", "прошептал", "сердито", "сквозь слёзы", "усмехнулся"} <= ru_cues
    for rows in table.values():
        for e in rows:
            assert e["emotion"] is None or e["emotion"] in dl.EMOTIONS
    assert dl.save_cues  # public helper


def test_cue_stems_match_gender_and_tense_forms():
    for text in ("Он воскликнул.", "Она воскликнула.", "Они воскликнули хором."):
        assert [h.cue for h in dl.scan_cues(text, "ru", DEFAULT_CUES)] == ["воскликнул"], text
    assert [h.cue for h in dl.scan_cues("Она усмехнулась.", "ru", DEFAULT_CUES)] == ["усмехнулся"]
    assert [h.cue for h in dl.scan_cues("— шепнула Юи.", "ru", DEFAULT_CUES)] == ["шепнул"]
    assert dl.scan_cues("Он вышел из комнаты.", "ru", DEFAULT_CUES) == []
    # ja: katakana == hiragana, a longer cue shadows the shorter one it contains
    assert [h.cue for h in dl.scan_cues("彼女はササヤイタ。", "ja", DEFAULT_CUES)] == ["ささやいた"]
    assert [h.cue for h in dl.scan_cues("彼は泣きながら叫んだ。", "ja", DEFAULT_CUES)] == ["叫んだ", "泣きながら"] or {h.cue for h in dl.scan_cues("彼は泣きながら叫んだ。", "ja", DEFAULT_CUES)} == {"叫んだ", "泣きながら"}


def test_save_and_load_cues_roundtrip(tmp_path):
    table = dl.load_cues(CUES_PATH)
    out = tmp_path / "cues.yaml"
    dl.save_cues(out, table)
    assert dl.load_cues(out) == table
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# Delivery cues") and "叫んだ" in text and "\\u" not in text
    # an edit made by the app: one more row, one removed
    table["ja"].append({"cue": "どなった", "emotion": "angry", "intensity": 0.8, "loudness": "loud"})
    table["ja"] = [e for e in table["ja"] if e["cue"] != "叫ぶ"]
    dl.save_cues(out, table)
    again = dl.load_cues(out)
    assert any(e["cue"] == "どなった" and e["tempo"] is None and e["confidence"] == 0.75 for e in again["ja"])
    assert not any(e["cue"] == "叫ぶ" for e in again["ja"])


@pytest.mark.parametrize(
    "entry,msg",
    [
        ({"cue": "x", "emotion": "furious", "intensity": 0.5}, "unknown emotion"),
        ({"cue": "x", "emotion": "sad", "intensity": 1.5}, "intensity"),
        ({"cue": "x", "emotion": "sad", "intensity": "high"}, "intensity"),
        ({"cue": "x", "emotion": "sad", "tempo": "fastish"}, "tempo"),
        ({"cue": "x", "emotion": "sad", "loudness": "quiet"}, "loudness"),
        ({"cue": "x", "emotion": "sad", "events": ["shout"]}, "events"),
        ({"cue": "x", "emotion": "sad", "events": "cry"}, "events"),
        ({"cue": "x", "emotion": "sad", "confidence": 1.2}, "confidence"),
        ({"cue": "x", "emotion": "sad", "position": "middle"}, "position"),
        ({"cue": "", "emotion": "sad"}, "non-empty"),
        ({"emotion": "sad"}, "non-empty"),
        ({"cue": "x"}, "sets nothing"),
        ({"cue": "x", "emotion": "sad", "colour": "red"}, "unknown field"),
    ],
)
def test_save_cues_validation(tmp_path, entry, msg):
    out = tmp_path / "cues.yaml"
    with pytest.raises(ValueError, match=msg):
        dl.save_cues(out, {"ja": [entry]})
    assert not out.exists()  # nothing written for an invalid table


def test_save_cues_rejects_duplicates_and_bad_shapes(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        dl.validate_cues({"ru": [{"cue": "Воскликнул", "emotion": "other"}, {"cue": "воскликнул", "emotion": "other"}]})
    with pytest.raises(ValueError, match="list of entries"):
        dl.validate_cues({"ja": {"cue": "x"}})
    with pytest.raises(ValueError, match="mapping"):
        dl.validate_cues([])
    bad = tmp_path / "bad.yaml"
    bad.write_text("ja:\n  - {cue: x, emotion: nope}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bad.yaml.*unknown emotion"):
        dl.load_cues(bad)
    with pytest.raises(FileNotFoundError):
        dl.load_cues(tmp_path / "missing.yaml")
    keep = tmp_path / "keep.yaml"
    dl.save_cues(keep, {"ja": [{"cue": "叫んだ", "emotion": "other"}]})
    before = keep.read_bytes()
    with pytest.raises(ValueError):
        dl.save_cues(keep, {"ja": [{"cue": "叫んだ", "emotion": "nope"}]})
    assert keep.read_bytes() == before  # a failed save leaves the old file


# ----------------------------------------------------------------------------
# the stage on a project
# ----------------------------------------------------------------------------

ROWS = [
    dict(line_id="s01_01", speaker_id="narrator", text="放課後の教室。窓の外は夕焼けだった。", voiced=False),
    dict(line_id="s01_02", speaker_id="sakura", speaker_raw="さくら", text="ねえ、勇太くん！", seconds=0.5),
    dict(line_id="s01_03", speaker_id="hero", speaker_raw="勇太", text="なんだよ、急に。", voiced=False),
    dict(line_id="s01_04", speaker_id="narrator", text="勇太は叫んだ。", voiced=False),
    dict(line_id="s01_05", speaker_id="hero", speaker_raw="勇太", text="うそ……本当に？", voiced=False),
    dict(line_id="s01_06", speaker_id="yui", speaker_raw="ゆい", text="あはは、さくらちゃんったら！", seconds=0.5),
    dict(line_id="s01_07", speaker_id="hero", speaker_raw="勇太", text="雨が降りそうだな。", voiced=False, flags=["voice_sustain"]),
    dict(line_id="s01_08", speaker_id="hero", speaker_raw="勇太", text="そうだね。", voiced=False),
]
UNVOICED = ["s01_03", "s01_05", "s01_08"]  # s01_07 is a voice-sustain continuation: covered by a head line's audio
DICT = {
    "ねえ、勇太くん！": "Эй, Юта!",
    "なんだよ、急に。": "Что такое? Вдруг.",
    "うそ……本当に？": "Не может быть... Правда?",
    "あはは、さくらちゃんったら！": "Ахаха, ну ты даёшь, Сакура!",
    "雨が降りそうだな。": "Похоже, будет дождь.",
    "そうだね。": "Точно.",
}


def _project(make_ctx, tmp_path, rows=ROWS, **sets):
    d = tmp_path / "dict.yaml"
    d.write_text(yaml.safe_dump(DICT, allow_unicode=True), encoding="utf-8")
    base = {"translate.mock_dictionary": str(d), "translate.glossary": str(tmp_path / "glossary.yaml"), "text_emotion.cues": str(CUES_PATH)}
    base.update(sets)
    ctx = make_ctx(tmp_path, sets=base)
    build_project(ctx, rows)
    return ctx


def _audio_passport(ctx, line_id: str) -> bytes:
    p = Passport(line_id=line_id, duration_s=0.5, emotion={"e2v": {"label": "happy", "probs": {"happy": 1.0}}}, style={"emotion": "happy", "intensity": 0.8})
    write_json(ctx.paths.passport(line_id), p)
    return ctx.paths.passport(line_id).read_bytes()


def test_stages_order_and_cli_registration():
    assert STAGES.index("text_emotion") == STAGES.index("translate") + 1 == STAGES.index("voicebank") - 1
    r = CliRunner().invoke(app, ["--help"])
    assert r.exit_code == 0 and "text-emotion" in r.output
    r = CliRunner().invoke(app, ["text-emotion", "--help"])
    assert r.exit_code == 0 and "--only" in r.output and "--force" in r.output and "--project" in r.output


def test_stage_writes_text_passports_and_leaves_audio_passports(make_ctx, tmp_path):
    ctx = _project(make_ctx, tmp_path)
    before = {lid: _audio_passport(ctx, lid) for lid in ("s01_02", "s01_06")}
    summary = run_stage("text_emotion", ctx)
    assert summary["lines"] == 3 and summary["written"] == 3 and summary["cached"] == 0
    for lid, raw in before.items():
        assert ctx.paths.passport(lid).read_bytes() == raw  # byte-identical: voiced lines are never touched
    # narration, voiced lines and the sustain continuation get no text passport
    assert not ctx.paths.passport("s01_01").exists() and not ctx.paths.passport("s01_07").exists()

    for lid in UNVOICED:
        raw = read_json(ctx.paths.passport(lid))
        p = Passport.model_validate(raw)
        assert raw["source"] == "text" and p.duration_s == 0.0 and p.missing["audio"]
        d = raw["delivery"]
        assert set(d) >= {"emotion", "intensity", "tempo", "loudness", "events", "confidence", "layer", "cues"}
        assert d["layer"] in ("rule", "cue", "llm", "neutral") and d["emotion"] in dl.EMOTIONS
        spec = StyleSpec.from_dict(raw["style"])  # synth renders it unchanged
        assert spec.emotion == d["emotion"] and spec.intensity == d["intensity"]
        render(spec, {"instruct": "natural", "knobs": "chatterbox"})

    # s01_05: "うそ……本当に？" (rule: ellipsis + question) with the narration "勇太は叫んだ。" before it (cue, after -> s01_03)
    d3 = read_json(ctx.paths.passport("s01_03"))["delivery"]
    assert d3["layer"] == "cue" and d3["loudness"] == "loud" and "叫んだ" in d3["cues"]
    d5 = read_json(ctx.paths.passport("s01_05"))["delivery"]
    assert d5["layer"] == "rule" and d5["arc"] == "rising_end" and d5["tempo"] == "slow"
    d8 = read_json(ctx.paths.passport("s01_08"))["delivery"]
    assert d8["layer"] == "neutral" and d8["emotion"] == "neutral" and d8["intensity"] == 0.3
    assert summary["layers"] == {"cue": 1, "rule": 1, "neutral": 1}
    ctx.models.close()


def test_stage_never_overwrites_an_audio_passport_even_for_an_unvoiced_line(make_ctx, tmp_path):
    ctx = _project(make_ctx, tmp_path)
    raw = _audio_passport(ctx, "s01_05")  # e.g. the voice was moved away by link; the old audio passport stays
    summary = run_stage("text_emotion", ctx)
    assert ctx.paths.passport("s01_05").read_bytes() == raw
    assert summary["kept_audio_passport"] == 1 and summary["written"] == 2
    ctx.models.close()


def test_stage_runs_without_translation_and_with_only_limit(make_ctx, tmp_path):
    ctx = _project(make_ctx, tmp_path)
    assert not ctx.paths.translation("ru").exists()
    ctx.only = {"s01_05"}
    summary = run_stage("text_emotion", ctx)
    assert summary["lines"] == 1 and ctx.paths.passport("s01_05").exists() and not ctx.paths.passport("s01_03").exists()
    ctx.models.close()


def test_stage_progress_callback(make_ctx, tmp_path):
    seen = []
    ctx = _project(make_ctx, tmp_path)
    ctx.extra["progress_cb"] = lambda stage, done, total, item: seen.append((stage, done, total, item))
    run_stage("text_emotion", ctx)
    assert [s[1] for s in seen[:3]] == [0, 1, 2] and seen[-1][1:3] == (3, 3) and seen[0][0] == "text_emotion" and seen[0][3] == "s01_03"
    ctx.models.close()


def test_stage_disabled_and_no_unvoiced_lines_are_noops(make_ctx, tmp_path):
    ctx = _project(make_ctx, tmp_path, **{"text_emotion.enabled": False})
    assert run_stage("text_emotion", ctx)["lines"] == 0 and not ctx.paths.passport("s01_03").exists()
    ctx.models.close()
    voiced_only = [r for r in ROWS if r.get("voiced", True)]
    sub = tmp_path / "v"
    sub.mkdir()
    ctx = _project(make_ctx, sub, rows=voiced_only)
    raws = {r["line_id"]: _audio_passport(ctx, r["line_id"]) for r in voiced_only}
    s = run_stage("text_emotion", ctx)
    assert s["lines"] == 0 and s["written"] == 0
    assert all(ctx.paths.passport(k).read_bytes() == v for k, v in raws.items())
    ctx.models.close()


def test_stage_caching(make_ctx, tmp_path):
    ctx = _project(make_ctx, tmp_path)
    assert run_stage("text_emotion", ctx)["written"] == 3
    again = run_stage("text_emotion", ctx)
    assert again["written"] == 0 and again["cached"] == 3
    # a changed neighbouring narration line invalidates only the lines it can describe
    lines = ctx.lines()
    for l in lines:
        if l.line_id == "s01_04":
            l.text_src = "勇太はささやいた。"
    write_jsonl(ctx.paths.lines, lines)
    third = run_stage("text_emotion", ctx)
    assert third["written"] >= 1 and third["cached"] <= 2
    assert read_json(ctx.paths.passport("s01_03"))["delivery"]["loudness"] == "soft"
    assert run_stage("text_emotion", ctx)["written"] == 0
    # editing the cue table invalidates everything (cue table hash is part of the key)
    cues = tmp_path / "my_cues.yaml"
    table = dl.load_cues(CUES_PATH)
    table["ja"].append({"cue": "どなった", "emotion": "angry"})
    dl.save_cues(cues, table)
    ctx.cfg.text_emotion.cues = str(cues)
    assert run_stage("text_emotion", ctx)["written"] == 3
    assert run_stage("text_emotion", ctx)["cached"] == 3
    # --force recomputes
    ctx.force = True
    assert run_stage("text_emotion", ctx)["written"] == 3
    ctx.models.close()


def test_invalid_cue_table_fails_with_a_readable_message(make_ctx, tmp_path):
    from vndub.stages.common import StageError

    bad = tmp_path / "bad_cues.yaml"
    bad.write_text("ja:\n  - {cue: 叫んだ, emotion: furious}\n", encoding="utf-8")
    ctx = _project(make_ctx, tmp_path, **{"text_emotion.cues": str(bad)})
    with pytest.raises(StageError, match="unknown emotion 'furious'"):
        run_stage("text_emotion", ctx)
    ctx.models.close()


def test_missing_cue_file_falls_back_to_the_bundled_table(make_ctx, tmp_path):
    ctx = _project(make_ctx, tmp_path, **{"text_emotion.cues": str(tmp_path / "nowhere.yaml")})
    assert text_emotion.cues_path(ctx.cfg) == dl.default_cues_path()
    run_stage("text_emotion", ctx)
    assert read_json(ctx.paths.passport("s01_03"))["delivery"]["layer"] == "cue"
    ctx.models.close()


def test_stage_cache_misses_when_a_passport_file_is_missing(make_ctx, tmp_path):
    ctx = _project(make_ctx, tmp_path)
    run_stage("text_emotion", ctx)
    ctx.paths.passport("s01_05").unlink()
    s = run_stage("text_emotion", ctx)
    assert s["written"] == 1 and s["cached"] == 2 and ctx.paths.passport("s01_05").exists()
    ctx.models.close()


def test_min_confidence_comes_from_config(make_ctx, tmp_path):
    ctx = _project(make_ctx, tmp_path, **{"text_emotion.min_confidence": 0.99})
    run_stage("text_emotion", ctx)
    for lid in UNVOICED:
        assert read_json(ctx.paths.passport(lid))["delivery"]["layer"] == "neutral"
    ctx.models.close()


def test_narration_window_comes_from_config(make_ctx, tmp_path):
    ctx = _project(make_ctx, tmp_path, **{"text_emotion.narration_window": 0})
    run_stage("text_emotion", ctx)
    assert read_json(ctx.paths.passport("s01_03"))["delivery"]["layer"] == "neutral"  # the cue is one line away
    ctx.models.close()


def test_stage_only_unvoiced_false_adds_voiced_lines_without_a_passport(make_ctx, tmp_path):
    ctx = _project(make_ctx, tmp_path, **{"text_emotion.only_unvoiced": False})
    raw = _audio_passport(ctx, "s01_02")
    s = run_stage("text_emotion", ctx)
    assert ctx.paths.passport("s01_02").read_bytes() == raw  # still never overwritten
    assert read_json(ctx.paths.passport("s01_06"))["source"] == "text"  # voiced, but analyze produced nothing for it
    assert s["written"] == 4
    ctx.models.close()


# ----------------------------------------------------------------------------
# LLM layer: translate's delivery field
# ----------------------------------------------------------------------------


def _records(ctx) -> dict[str, TranslationRecord]:
    return {r.line_id: r for r in read_jsonl(ctx.paths.translation("ru"), TranslationRecord)}


def _delivery_of(rec: TranslationRecord) -> dict[str, Any] | None:
    return (rec.model_extra or {}).get("delivery")


def test_translate_includes_unvoiced_speaker_lines_when_dubbed(make_ctx, tmp_path):
    ctx = _project(make_ctx, tmp_path, **{"synth.voice_unvoiced": True})  # include_unvoiced auto follows it (the app)
    run_stage("translate", ctx)
    # voiced lines, the sustain continuation and the unvoiced speaker lines; narration (s01_01, s01_04) is context only
    assert set(_records(ctx)) == {"s01_02", "s01_03", "s01_05", "s01_06", "s01_07", "s01_08"}
    assert {lid for lid, r in _records(ctx).items() if _delivery_of(r)} == set(UNVOICED)
    ctx.models.close()


def test_translate_include_unvoiced_off_and_auto(make_ctx, tmp_path):
    ctx = _project(make_ctx, tmp_path, **{"translate.include_unvoiced": "off"})
    run_stage("translate", ctx)
    assert set(_records(ctx)) == {"s01_02", "s01_06", "s01_07"}  # the old behaviour: voiced + sustain only
    assert all(_delivery_of(r) is None for r in _records(ctx).values())
    ctx.models.close()
    sub = tmp_path / "auto"
    sub.mkdir()
    ctx = _project(make_ctx, sub, **{"translate.include_unvoiced": "auto"})
    assert {l.line_id for l in translate.lines_to_translate(ctx)} == {"s01_02", "s01_06", "s01_07"}  # synth.voice_unvoiced is off
    ctx.cfg.synth.voice_unvoiced = True
    assert {l.line_id for l in translate.lines_to_translate(ctx)} == {"s01_02", "s01_03", "s01_05", "s01_06", "s01_07", "s01_08"}
    ctx.models.close()


def test_llm_pass_with_mock_translator(make_ctx, tmp_path):
    ctx = _project(make_ctx, tmp_path, **{"translate.include_unvoiced": "on"})
    summary = run_stage("translate", ctx)
    recs = _records(ctx)
    assert set(recs) == {"s01_02", "s01_03", "s01_05", "s01_06", "s01_07", "s01_08"}
    assert summary["deliveries"] == 3
    # deliveries only for the unvoiced speaker lines (not for the voiced ones, not for the sustain continuation)
    assert {lid for lid, r in recs.items() if _delivery_of(r)} == set(UNVOICED)
    for lid in UNVOICED:
        assert dl.validate_delivery(_delivery_of(recs[lid])) == _delivery_of(recs[lid])
    # the mock reads the narration around the line: "勇太は叫んだ。" describes s01_03 -> a confident shout
    d3 = _delivery_of(recs["s01_03"])
    assert d3["loudness"] == "loud" and d3["confidence"] >= 0.6
    # a line with nothing to go on gets the prompt's neutral / 0.3 / low-confidence answer
    d8 = _delivery_of(recs["s01_08"])
    assert (d8["emotion"], d8["intensity"]) == ("neutral", 0.3) and d8["confidence"] < 0.6
    ctx.models.close()

    ctx = _project(make_ctx, tmp_path, **{"translate.include_unvoiced": "on"})
    run_stage("text_emotion", ctx)
    p3 = read_json(ctx.paths.passport("s01_03"))["delivery"]
    assert p3["layer"] == "llm" and p3["loudness"] == "loud"  # llm (0.9) overrides the cue layer (0.9 is not lower)
    p5 = read_json(ctx.paths.passport("s01_05"))
    assert p5["delivery"]["layer"] in ("llm", "rule")
    p8 = read_json(ctx.paths.passport("s01_08"))["delivery"]
    assert p8["layer"] == "neutral"  # llm said neutral/0.3/0.3 confidence: below min_confidence, nothing else indicates anything
    ctx.models.close()


def test_llm_off_ignores_stored_deliveries_and_does_not_ask(make_ctx, tmp_path, monkeypatch):
    ctx = _project(make_ctx, tmp_path, **{"translate.include_unvoiced": "on", "text_emotion.llm": "off"})
    run_stage("translate", ctx)
    assert all(_delivery_of(r) is None for r in _records(ctx).values())  # not requested
    ctx.models.close()
    # a record that does carry a delivery is ignored by the stage when llm is off
    recs = read_jsonl(ctx.paths.translation("ru"), TranslationRecord)
    for r in recs:
        if r.line_id == "s01_08":
            r.delivery = {"emotion": "sad", "intensity": 0.7, "tempo": "slow", "loudness": "soft", "events": [], "confidence": 0.95}
    write_jsonl(ctx.paths.translation("ru"), recs)
    ctx = _project(make_ctx, tmp_path, **{"text_emotion.llm": "off"})
    run_stage("text_emotion", ctx)
    assert read_json(ctx.paths.passport("s01_08"))["delivery"]["layer"] == "neutral"
    ctx.cfg.text_emotion.llm = "auto"
    run_stage("text_emotion", ctx)
    assert read_json(ctx.paths.passport("s01_08"))["delivery"]["layer"] == "llm"
    ctx.models.close()


class FakeLLM:
    """Answers the translate protocol; the delivery objects are whatever the test hands in."""

    def __init__(self, deliveries: dict[str, Any] | None = None):
        self.deliveries = deliveries or {}
        self.requests: list[list[dict[str, Any]]] = []
        self.system: list[str] = []

    def chat(self, messages, temperature=0.3, **kw):
        self.system.append(messages[0]["content"])
        batch = json.loads(messages[-1]["content"].split("\n", 1)[1])
        self.requests.append(batch)
        lines = []
        for b in batch:
            e = {"id": b["id"], "ru": f"перевод {b['id']}", "ru_short": "", "notes": "x"}
            if b["id"] in self.deliveries:
                e["delivery"] = self.deliveries[b["id"]]
            lines.append(e)
        return json.dumps({"lines": lines}, ensure_ascii=False)


GOOD = {"emotion": "sad", "intensity": 0.6, "tempo": "slow", "loudness": "soft", "events": ["sigh"], "confidence": 0.85}


def test_invalid_llm_delivery_is_dropped_and_the_stage_falls_back(make_ctx, tmp_path, monkeypatch):
    ctx = _project(make_ctx, tmp_path, **{"translate.include_unvoiced": "on"})
    fake = FakeLLM(
        {
            "s01_03": {**GOOD, "intensity": 7},          # out of range
            "s01_05": {**GOOD, "emotion": "furious"},    # unknown enum
            "s01_08": GOOD,                              # valid
            "s01_02": GOOD,                              # a voiced line: never asked for, ignored
        }
    )
    monkeypatch.setattr(translate, "_acquire_llm", lambda c: (fake, "fake_llm"))
    summary = run_stage("translate", ctx)
    recs = _records(ctx)
    assert summary["failed"] == 0 and summary["deliveries"] == 1  # the invalid ones never fail the batch
    assert _delivery_of(recs["s01_03"]) is None and _delivery_of(recs["s01_05"]) is None and _delivery_of(recs["s01_02"]) is None
    assert _delivery_of(recs["s01_08"]) == dl.validate_delivery(GOOD)
    assert recs["s01_03"].text_tgt == "перевод s01_03"  # the translation itself is intact
    # stage: lines without a valid delivery fall back to rule/cue; the valid one is merged
    run_stage("text_emotion", ctx)
    assert read_json(ctx.paths.passport("s01_03"))["delivery"]["layer"] == "cue"
    assert read_json(ctx.paths.passport("s01_05"))["delivery"]["layer"] == "rule"
    p8 = read_json(ctx.paths.passport("s01_08"))
    assert p8["delivery"]["layer"] == "llm" and p8["delivery"]["events"] == ["sigh"] and p8["style"]["events"] == ["sigh"]
    ctx.models.close()


def test_garbage_delivery_never_triggers_a_retry(make_ctx, tmp_path, monkeypatch):
    ctx = _project(make_ctx, tmp_path, **{"translate.include_unvoiced": "on"})
    fake = FakeLLM({"s01_03": "loud!", "s01_05": [1, 2], "s01_08": None})
    monkeypatch.setattr(translate, "_acquire_llm", lambda c: (fake, "fake_llm"))
    run_stage("translate", ctx)
    target_calls = [r for r in fake.requests if any(b.get("want_delivery") for b in r)]
    assert len(target_calls) == 1  # one request for the three target lines, no retries because of the delivery field
    ctx.models.close()


def test_delivery_request_shape(make_ctx, tmp_path, monkeypatch):
    ctx = _project(make_ctx, tmp_path, **{"translate.include_unvoiced": "on", "text_emotion.llm_context_lines": 2})
    msgs: list[Any] = []

    class Spy(FakeLLM):
        def chat(self, messages, **kw):
            msgs.append(messages)
            return super().chat(messages, **kw)

    spy = Spy()
    monkeypatch.setattr(translate, "_acquire_llm", lambda c: (spy, "fake_llm"))
    run_stage("translate", ctx)
    voiced_batch = next(r for r in spy.requests if not any(b.get("want_delivery") for b in r))
    target_batch = next(r for r in spy.requests if all(b.get("want_delivery") for b in r))
    assert {b["id"] for b in voiced_batch} == {"s01_02", "s01_06", "s01_07"} and all("want_delivery" not in b for b in voiced_batch)
    assert [b["id"] for b in target_batch] == UNVOICED and all(b["emotion"] == "" for b in target_batch)
    target_msgs = next(m for m in msgs if '"want_delivery"' in m[-1]["content"])
    system = target_msgs[0]["content"]
    assert "Do not invent drama" in system and "delivery" in system and "neutral" in system and '"intensity": 0.3' in system
    ctx_msg = next(m["content"] for m in target_msgs if "scene_lines:" in m["content"])
    scene_lines = json.loads(ctx_msg.split("scene_lines:", 1)[1].split("\n", 1)[1])
    assert any(r["speaker"] == "narrator" and r["text"] == "勇太は叫んだ。" for r in scene_lines)  # narration is part of the context
    assert [r["id"] for r in scene_lines if r.get("deliver")] == UNVOICED
    assert all(r["id"] in {l.line_id for l in ctx.lines()} for r in scene_lines)
    # the voiced batch's system prompt knows nothing about delivery
    voiced_msgs = next(m for m in msgs if '"want_delivery"' not in m[-1]["content"])
    assert "delivery" not in voiced_msgs[0]["content"] and not any("scene_lines" in m["content"] for m in voiced_msgs)
    ctx.models.close()


def test_translate_cache_of_voiced_lines_survives_enabling_unvoiced(make_ctx, tmp_path, monkeypatch):
    ctx = _project(make_ctx, tmp_path, **{"translate.include_unvoiced": "off"})  # voiced lines only: the old behaviour
    run_stage("translate", ctx)
    ctx.models.close()

    ctx = _project(make_ctx, tmp_path, **{"translate.include_unvoiced": "on"})
    fake = FakeLLM()
    monkeypatch.setattr(translate, "_acquire_llm", lambda c: (fake, "fake_llm"))
    # the cache key contains the backend name: keep it the same as in the first run
    monkeypatch.setattr(translate, "_llm_key_backend", lambda c: "mock_translator")
    run_stage("translate", ctx)
    asked = {b["id"] for r in fake.requests for b in r}
    assert asked == set(UNVOICED)  # voiced batches came from the cache: the model only saw the new lines
    recs = _records(ctx)
    assert recs["s01_02"].text_tgt == "Эй, Юта!"  # still the original (mock dictionary) translation
    ctx.models.close()


def test_translate_target_batches_are_cached(make_ctx, tmp_path, monkeypatch):
    ctx = _project(make_ctx, tmp_path, **{"translate.include_unvoiced": "on"})
    run_stage("translate", ctx)
    ctx.models.close()

    def boom(*a, **k):
        raise AssertionError("llm acquired although every batch is cached")

    monkeypatch.setattr(translate, "_acquire_llm", boom)
    monkeypatch.setattr(ctx.models, "acquire", boom)
    s = run_stage("translate", ctx)
    assert s["deliveries"] == 3 and all(_delivery_of(_records(ctx)[l]) for l in UNVOICED)


def test_import_loads_no_llm_for_delivery_unless_on(make_ctx, tmp_path, monkeypatch):
    imp = tmp_path / "import.jsonl"
    imp.write_text("\n".join(json.dumps({"line_id": k, "text_tgt": f"ru {k}"}, ensure_ascii=False) for k in ["s01_02", "s01_03", "s01_05", "s01_08"]), encoding="utf-8")

    def boom(*a, **k):
        raise AssertionError("llm acquired for an imported translation")

    ctx = _project(make_ctx, tmp_path, **{"translate.import": str(imp), "translate.include_unvoiced": "on"})
    monkeypatch.setattr(translate, "_acquire_llm", boom)
    s = run_stage("translate", ctx)  # llm: auto -> no delivery request, no model load
    assert s["backend"] == "import" and s["deliveries"] == 0
    assert _records(ctx)["s01_03"].text_tgt == "ru s01_03"
    ctx.models.close()
    monkeypatch.undo()

    ctx = _project(make_ctx, tmp_path, **{"translate.import": str(imp), "translate.include_unvoiced": "on", "text_emotion.llm": "on"})
    s = run_stage("translate", ctx)  # delivery-only batch through the same client code (mock translator here)
    recs = _records(ctx)
    assert s["backend"] == "import" and s["deliveries"] == 3
    assert recs["s01_03"].text_tgt == "ru s01_03" and recs["s01_03"].source == "import"  # the imported text is untouched
    assert _delivery_of(recs["s01_03"])["loudness"] == "loud"
    assert _delivery_of(recs["s01_02"]) is None  # voiced line
    ctx.models.close()


def test_import_delivery_only_failure_is_not_fatal(make_ctx, tmp_path, monkeypatch):
    from vndub.backends.base import BackendUnavailable

    imp = tmp_path / "import.jsonl"
    imp.write_text(json.dumps({"line_id": "s01_03", "text_tgt": "ru"}), encoding="utf-8")
    ctx = _project(make_ctx, tmp_path, **{"translate.import": str(imp), "translate.include_unvoiced": "on", "text_emotion.llm": "on"})

    def unavailable(c):
        raise BackendUnavailable("gemma4_12b", "no model")

    monkeypatch.setattr(translate, "_acquire_llm", unavailable)
    s = run_stage("translate", ctx)
    assert s["deliveries"] == 0 and _records(ctx)["s01_03"].text_tgt == "ru"
    ctx.models.close()


def test_translate_config_sections_include_text_emotion():
    assert "text_emotion" in translate.CONFIG_SECTIONS and text_emotion.CONFIG_SECTIONS == ("text_emotion",)


# ----------------------------------------------------------------------------
# preview_import
# ----------------------------------------------------------------------------


def _tree(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def test_preview_import_jsonl_and_no_writes(make_ctx, tmp_path):
    ctx = _project(make_ctx, tmp_path, **{"translate.include_unvoiced": "on"})
    imp = tmp_path / "imp" / "import.jsonl"
    imp.parent.mkdir()
    imp.write_text("\n".join(json.dumps({"line_id": k, "text_tgt": v}, ensure_ascii=False) for k, v in [("s01_06", "Ахаха!"), ("s01_02", "Эй!"), ("s01_03", "Что?"), ("zzz", "x"), ("s01_05", "  ")]), encoding="utf-8")
    snapshot = _tree(tmp_path)
    out = translate.preview_import(ctx, str(imp), n=2)
    assert _tree(tmp_path) == snapshot  # no writes
    assert out["format"] == "jsonl" and out["matched"] == 3 and out["in_file"] == 5
    assert out["total"] == 6  # voiced + sustain + the three unvoiced speaker lines (include_unvoiced on)
    assert out["preview"] == [
        {"line_id": "s01_02", "text_src": "ねえ、勇太くん！", "text_tgt": "Эй!"},
        {"line_id": "s01_03", "text_src": "なんだよ、急に。", "text_tgt": "Что?"},
    ]
    assert translate.preview_import(ctx, str(imp), n=0)["preview"] == []
    assert len(translate.preview_import(ctx, str(imp), n=50)["preview"]) == 3
    ctx.models.close()


def test_preview_import_total_follows_include_unvoiced(make_ctx, tmp_path):
    imp = tmp_path / "t.tsv"
    imp.write_text("line_id\tru\ns01_02\tЭй!\ns01_03\tЧто?\n", encoding="utf-8")
    ctx = _project(make_ctx, tmp_path, **{"translate.include_unvoiced": "off"})
    out = translate.preview_import(ctx, str(imp))
    assert out["format"] == "tsv" and out["total"] == 3 and out["matched"] == 1  # s01_03 is not translated with `off`
    ctx.models.close()
    sub = tmp_path / "on"
    sub.mkdir()
    ctx = _project(make_ctx, sub, **{"synth.voice_unvoiced": True})  # the app: the unvoiced protagonist's lines count
    out = translate.preview_import(ctx, str(imp))
    assert out["total"] == 6 and out["matched"] == 2 and [p["line_id"] for p in out["preview"]] == ["s01_02", "s01_03"]
    ctx.models.close()


def test_preview_import_formats_and_folder(make_ctx, tmp_path):
    import openpyxl

    ctx = _project(make_ctx, tmp_path, **{"synth.voice_unvoiced": True})
    xlsx = tmp_path / "t.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Original", "Translation"])
    ws.append(["ねえ、勇太くん！", "Эй, Юта!"])
    wb.save(xlsx)
    out = translate.preview_import(ctx, str(xlsx))
    assert out["format"] == "xlsx" and out["matched"] == 1 and out["preview"][0]["line_id"] == "s01_02"

    folder = tmp_path / "vtp"
    folder.mkdir()
    lines = ctx.lines()
    (folder / "s01.json").write_text(json.dumps([{"name": "x", "message": f"RU {l.line_id}"} for l in lines if l.scene == "s01"], ensure_ascii=False), encoding="utf-8")
    for l in lines:
        l.source_file = "scenario/s01.ks"
    write_jsonl(ctx.paths.lines, lines)
    out = translate.preview_import(ctx, str(folder))
    assert out["format"] == "folder:json" and out["files"] == 1 and out["matched"] == 6  # positional VNTextPatch mapping
    assert out["preview"][0] == {"line_id": "s01_02", "text_src": "ねえ、勇太くん！", "text_tgt": "RU s01_02"}

    mapping = tmp_path / "m.json"
    mapping.write_text(json.dumps({"s01_02": "Эй!"}), encoding="utf-8")
    assert translate.preview_import(ctx, str(mapping))["format"] == "json_map"
    pairs = tmp_path / "p.json"
    pairs.write_text(json.dumps([{"original": "そうだね。", "translation": "Точно."}]), encoding="utf-8")
    assert translate.preview_import(ctx, str(pairs))["format"] == "json_pairs"
    ctx.models.close()


def test_preview_import_errors_are_stage_errors(make_ctx, tmp_path):
    from vndub.stages.common import StageError

    ctx = _project(make_ctx, tmp_path)
    with pytest.raises(StageError, match="not found"):
        translate.preview_import(ctx, str(tmp_path / "nope.jsonl"))
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(StageError, match="cannot read"):
        translate.preview_import(ctx, str(broken))
    ctx.models.close()


# ----------------------------------------------------------------------------
# the mixed fixture (voiced heroines + the unvoiced protagonist and his narration)
# ----------------------------------------------------------------------------


@requires_ffmpeg
def test_mixed_renpy_mini_fixture(make_ctx):
    ctx = make_ctx(FIXTURES / "renpy_mini", sets={"synth.voice_unvoiced": True, "translate.include_narration": "off"}, copy_project=True)  # app config: unvoiced speaker lines are translated  # narration off: Ren'Py would translate it as text-only lines by default (translate.include_narration: auto)
    for st in ("detect", "extract", "parse"):
        run_stage(st, ctx)
    lines = ctx.lines()
    hero = {l.text_src: l for l in lines if l.speaker_id == "hero"}
    assert len(hero) >= 6 and not any(l.has_voice for l in hero.values())
    summary = run_stage("text_emotion", ctx)
    assert summary["lines"] == len(hero) and summary["written"] == len(hero)
    assert not any(ctx.paths.passport(l.line_id).exists() for l in lines if l.has_voice)  # analyze did not run: no audio passports to touch

    def delivery(prefix: str) -> dict[str, Any]:
        l = next(l for t, l in hero.items() if t.startswith(prefix))
        p = read_json(ctx.paths.passport(l.line_id))
        assert p["source"] == "text"
        return p["delivery"]

    d = delivery("……ゆい、大丈夫か？")  # "...蓮はゆいにだけ聞こえるよう、小さな声でささやいた。" follows it (name 蓮, not ゆい)
    assert d["layer"] == "cue" and d["loudness"] == "soft" and "whisper" in d["events"] and "ささやいた" in d["cues"]
    d = delivery("はいはい、百点は")  # "...蓮は笑いながら言った。" (next line is nearer than the previous one)
    assert d["layer"] == "cue" and d["emotion"] == "happy" and "laugh" in d["events"]
    d = delivery("でも、ゆいだって")  # "...蓮は悔しさをこらえきれず、泣きながら言った。"
    assert d["layer"] == "cue" and d["emotion"] == "sad" and "cry" in d["events"]
    d = delivery("みんな、帰ったのか")  # nothing says how: generic
    assert d["layer"] == "neutral" and d["emotion"] == "neutral" and d["intensity"] == 0.3
    d = delivery("おい、さくら、ノートを")  # "...待ってくれと叫んだ。" after it
    assert d["layer"] == "cue" and d["loudness"] == "loud"
    d = delivery("もういい、明日渡す")  # "...腹立ちまぎれに怒鳴った。" before it
    assert d["layer"] == "cue" and d["emotion"] == "angry" and d["loudness"] == "loud"

    # the LLM layer (mock translator): translate adds deliveries for exactly these lines, the stage merges them
    run_stage("translate", ctx)
    recs = {r.line_id: r for r in read_jsonl(ctx.paths.translation("ru"), TranslationRecord)}
    assert {lid for lid, r in recs.items() if _delivery_of(r)} == {l.line_id for l in hero.values()}
    assert all(l.line_id in recs for l in hero.values()) and not any(l.line_id in recs for l in lines if l.speaker_id == "narrator" and not ({"menu", "menu_caption"} & set(l.flags)))  # menus are game text, translated for the text patch
    s2 = run_stage("text_emotion", ctx)
    assert s2["written"] >= 1 and s2["layers"].get("llm", 0) >= 1
    assert run_stage("text_emotion", ctx)["written"] == 0
    ctx.models.close()


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def test_cli_text_emotion_command_runs_the_stage(make_ctx, tmp_path):
    ctx = _project(make_ctx, tmp_path)
    ctx.models.close()
    r = CliRunner().invoke(
        app,
        ["text-emotion", "--project", str(tmp_path), "--config", str(TEST_CONFIG), "--work-dir", str(tmp_path / "work"), "--out-dir", str(tmp_path / "out"), "--set", f"text_emotion.cues={CUES_PATH}"],
    )
    assert r.exit_code == 0, r.output
    assert read_json(tmp_path / "work" / "analysis" / "s01_03.json")["source"] == "text"
    status = read_json(tmp_path / "work" / "stage_status.json")
    assert status["text_emotion"]["finished"] and not status["text_emotion"]["error"]


# ----------------------------------------------------------------------------
# laughter: every source language, kinds, stage directions, laughter-only lines (vndub.text.laughter feeds the rule pass)
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,kind,emotion,note",
    [
        ("Haha, no way!", "hearty", "happy", "a hearty laugh"),
        ("Hehe, you wish.", "chuckle", "happy", "chuckling"),
        ("Hihi!", "giggle", "happy", "giggling"),
        ("Mwahaha! Bow before me!", "evil", "other", "a sinister laugh"),
        ("Ehehe... sorry about that.", "nervous", "embarrassed", "a nervous laugh"),
        ("Pfft, seriously?", "snort", "other", "a snort"),
    ],
)
def test_english_laughter_gives_event_kind_emotion_and_note(text, kind, emotion, note):
    r = dl.rule_pass(text, src_lang="en")
    assert r.events == ["laugh"] and r.emotion == emotion and note in r.notes
    assert r.laughter["kind"] == kind and r.laughter["stage"] is False
    d = dl.merge_layers([r])
    assert d.layer == "rule" and d.laughter["kind"] == kind and note in d.notes
    spec = dl.to_style(d)
    assert spec.events == ["laughter"] and spec.laughter["kind"] == kind
    assert StyleSpec.from_dict(spec.to_dict()) == spec


@pytest.mark.parametrize(
    "text,tgt,lang,kind",
    [
        ("あはは、おかしい！", "", "ja", "hearty"),
        ("えへへ……", "", "ja", "nervous"),
        ("くすくす。", "", "ja", "giggle"),
        ("そうだね。", "Хи-хи-хи.", "ja", "giggle"),
        ("Ну ты даёшь, ха-ха-ха!", "", "ru", "hearty"),
        ("Муа-ха-ха-ха!", "", "ru", "evil"),
        ("Mwa-ha-ha-ha!", "", "en", "evil"),
    ],
)
def test_laughter_kind_in_every_script(text, tgt, lang, kind):
    r = dl.rule_pass(text, tgt, src_lang=lang)
    assert "laugh" in r.events and r.laughter["kind"] == kind, (text, r)


@pytest.mark.parametrize(
    "text,kind,only",
    [("[laughs]", "hearty", True), ("[giggles] Oh, come on.", "giggle", False), ("{i}[chuckles]{/i} Fine.", "chuckle", False)],
)
def test_bracketed_stage_direction_in_a_latin_source_is_found(text, kind, only):
    # `[...]` used to be stripped as a KAG tag before the laugh finder ran (the translation was never stripped)
    r = dl.rule_pass(text, src_lang="en")
    assert r.events == ["laugh"] and r.laughter["kind"] == kind and r.laughter["stage"] is True and r.laughter["only"] is only
    assert dl.rule_pass(text, "Ха-ха.", src_lang="en").events == ["laugh"]
    # a real KAG / engine tag is still stripped and is not a laugh
    assert dl.rule_pass("[l][r]Hello[wait time=3]", src_lang="en").events == []


def test_source_language_limits_what_counts_as_the_source_laugh():
    assert dl.rule_pass("Haha!", src_lang="ja").events == []  # a Japanese script has no Latin laughter
    assert dl.rule_pass("あはは", src_lang="en").events == []
    assert "laugh" in dl.rule_pass("Haha!", src_lang="auto").events and "laugh" in dl.rule_pass("Haha!").events
    # the translation is read in every script whatever the source language
    assert "laugh" in dl.rule_pass("Haha!", "Ха-ха!", src_lang="ja").events


def test_a_long_laugh_raises_the_intensity_more():
    short, long_ = dl.rule_pass("Haha.", src_lang="en"), dl.rule_pass("Hahahaha.", src_lang="en")
    assert short.laughter["units"] == 2 and long_.laughter["units"] == 4
    assert long_.intensity > short.intensity > 0.3
    assert dl.rule_pass("Heh heh heh.", src_lang="en").intensity > dl.rule_pass("Heh heh.", src_lang="en").intensity


@pytest.mark.parametrize("text,lang", [("*laughs*", "en"), ("(giggles)", "en"), ("(смеётся)", "ru"), ("（笑）", "ja")])
def test_stage_direction_is_weaker_evidence_than_a_spoken_laugh(text, lang):
    r = dl.rule_pass(text, src_lang=lang)
    assert r.events == ["laugh"] and r.laughter["stage"] is True and r.laughter["only"] is True
    spoken = dl.rule_pass({"en": "Haha.", "ru": "Ха-ха.", "ja": "あはは。"}[lang], src_lang=lang)
    assert spoken.laughter["stage"] is False
    assert 0.6 <= r.confidence < spoken.confidence  # still accepted at min_confidence 0.6, but weaker
    assert r.intensity < spoken.intensity
    assert dl.merge_layers([r]).layer == "rule"


def test_laughter_only_flag_looks_at_what_the_dub_speaks():
    assert dl.rule_pass("Hahaha!", src_lang="en").laughter["only"] is True
    assert dl.rule_pass("Haha, no way!", src_lang="en").laughter["only"] is False
    assert dl.rule_pass("あはは", "Ну ты даёшь!", src_lang="ja").laughter["only"] is False  # the translation has words
    assert dl.rule_pass("あはは、ひどい", "Ха-ха-ха!", src_lang="ja").laughter["only"] is True
    d = dl.merge_layers([dl.rule_pass("*laughs* Hi.", src_lang="en")])
    assert d.laughter["only"] is False and d.laughter["stage"] is True


def test_evil_laugh_does_not_outvote_a_named_emotion():
    # "other" names no direction: an angry lead-in wins over a sinister laugh, the laugh alone gives "other"
    r = dl.rule_pass("ちくしょう、くくく", src_lang="ja")
    assert r.laughter["kind"] == "evil" and r.emotion == "angry"
    assert dl.rule_pass("Mwahaha!", src_lang="en").emotion == "other"


def test_delivery_laughter_round_trips_and_old_passports_load():
    d = dl.merge_layers([dl.rule_pass("Hihi!", src_lang="en")])
    assert dl.Delivery.from_dict(d.to_dict()) == d and d.to_dict()["laughter"]["kind"] == "giggle"
    plain = dl.merge_layers([dl.rule_pass("待って！")])
    assert "laughter" not in plain.to_dict()  # absent without a laugh: stored passports of earlier versions read the same
    old = {k: v for k, v in d.to_dict().items() if k != "laughter"}
    assert dl.Delivery.from_dict(old).laughter == {} and dl.to_style(dl.Delivery.from_dict(old)).laughter is None
    # a laugh whose layer was rejected leaves nothing behind
    weak = dl.Layer("rule", 0.4, emotion="happy", events=["laugh"], laughter={"kind": "hearty"})
    assert dl.merge_layers([weak]).laughter == {}


def test_llm_laugh_kind_is_validated_and_only_fills_a_gap():
    ok = {"emotion": "happy", "intensity": 0.6, "tempo": "normal", "loudness": "normal", "confidence": 0.8}
    v = dl.validate_delivery({**ok, "events": ["giggling"]})
    assert v["events"] == ["laugh"] and v["laugh_kind"] == "giggle"
    v = dl.validate_delivery({**ok, "events": [], "laugh_kind": "Evil laugh"})
    assert v["laugh_kind"] == "evil" and v["events"] == ["laugh"]  # a kind implies the event
    assert dl.validate_delivery({**ok, "laughter": "chuckling"})["laugh_kind"] == "chuckle"
    v = dl.validate_delivery({**ok, "events": ["laughter"], "laugh_kind": "wibble"})
    assert v is not None and "laugh_kind" not in v and v["events"] == ["laugh"]  # an unknown word is ignored, it does not invalidate
    assert "laugh_kind" not in dl.validate_delivery({**ok, "events": ["cry"]})
    llm = dl.llm_layer({**ok, "events": ["laugh"], "laugh_kind": "giggle"})
    assert llm.laughter == {"kind": "giggle"}
    # the rule pass read the actual text: the LLM kind only fills a gap
    rule = dl.rule_pass("Haha!", src_lang="en")
    assert dl.merge_layers([rule, llm]).laughter["kind"] == "hearty"
    bare = dl.Layer("rule", 0.7, emotion="other", intensity=0.5, events=["laugh"])
    assert dl.merge_layers([bare, llm]).laughter == {"kind": "giggle"}


def test_mock_delivery_reports_laughter():
    d = dl.mock_delivery("Hihi!", src_lang="en")
    assert "laugh" in d["events"] and d["laugh_kind"] == "giggle" and dl.validate_delivery(d) is not None
    j = dl.mock_delivery("あはは！")
    assert "laugh" in j["events"] and j["laugh_kind"] == "hearty"
    assert "laugh_kind" not in dl.mock_delivery("そうですか。")


EN_ROWS = [
    dict(line_id="e01_01", speaker_id="hero", speaker_raw="Hero", text="Haha, no way!", voiced=False),
    dict(line_id="e01_02", speaker_id="hero", speaker_raw="Hero", text="Hehe...", voiced=False),
    dict(line_id="e01_03", speaker_id="hero", speaker_raw="Hero", text="*laughs*", voiced=False),
    dict(line_id="e01_04", speaker_id="hero", speaker_raw="Hero", text="Fine.", voiced=False),
]


def test_stage_reads_english_laughter_from_the_project_language(make_ctx, tmp_path):
    ctx = _project(make_ctx, tmp_path, rows=EN_ROWS, **{"project.src_lang": "en"})
    assert ctx.src_lang == "en"
    summary = run_stage("text_emotion", ctx)
    assert summary["lines"] == 4 and summary["written"] == 4
    d1 = read_json(ctx.paths.passport("e01_01"))["delivery"]
    assert d1["events"] == ["laugh"] and d1["laughter"] == {"kind": "hearty", "units": 2, "stage": False, "only": False}
    assert "a hearty laugh" in d1["notes"] and d1["emotion"] == "happy"
    p1 = Passport.model_validate(read_json(ctx.paths.passport("e01_01")))
    spec = StyleSpec.from_dict(p1.style)
    assert spec.events == ["laughter"] and spec.laughter["kind"] == "hearty"
    d2 = read_json(ctx.paths.passport("e01_02"))["delivery"]
    assert d2["laughter"]["kind"] == "nervous" and d2["laughter"]["only"] is True and d2["emotion"] == "embarrassed"
    d3 = read_json(ctx.paths.passport("e01_03"))["delivery"]
    assert d3["laughter"]["stage"] is True and d3["laughter"]["only"] is True and d3["layer"] == "rule"
    d4 = read_json(ctx.paths.passport("e01_04"))["delivery"]
    assert d4["layer"] == "neutral" and "laughter" not in d4
    again = run_stage("text_emotion", ctx)
    assert again["written"] == 0 and again["cached"] == 4
    ctx.models.close()
    # the same text read as a Japanese script has no laughter: the project language is part of the cache key and of the reading
    (tmp_path / "ja").mkdir()
    ctx2 = _project(make_ctx, tmp_path / "ja", rows=EN_ROWS, **{"project.src_lang": "ja"})
    run_stage("text_emotion", ctx2)
    assert read_json(ctx2.paths.passport("e01_01"))["delivery"]["events"] == []
    ctx2.models.close()

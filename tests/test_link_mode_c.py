"""Mode C: voiced lines without script text are transcribed (only those), the text is applied by
StageContext.lines() with flag text_from_asr; user edits arrive as set_src_text overrides."""

from __future__ import annotations

import json

import pytest
import yaml

from analysis_helpers import build_project
from conftest import _make_ctx, requires_ffmpeg
from scripts_helpers import tone_bytes
from vndub.manifest.io import Character, LinkRecord, OverrideRecord, append_jsonl, load_lines, read_jsonl, write_jsonl
from vndub.stages import link, parse
from vndub.stages.common import StageError, apply_text_patches, run_stage

HEARD_1 = "ここは小さな声で言った"
HEARD_2 = "また明日会いましょう"

ROWS = [
    dict(line_id="m_01", speaker_id="sakura", text="ねえ、ちょっと待って！", seconds=1.9, f0=260, audio_id="sak_0001"),
    dict(line_id="m_02", speaker_id="sakura", text="", seconds=2.2, f0=260, audio_id="sak_0002"),  # voiced, script text missing
    dict(line_id="m_03", speaker_id="haruto", text="どうしたの、さくら？", seconds=1.9, f0=120, audio_id="har_0001"),
    dict(line_id="m_04", speaker_id="haruto", text="", seconds=1.9, f0=120, audio_id="har_0002"),  # voiced, script text missing
    dict(line_id="m_05", speaker_id="hero", text="僕は主人公だ。", voiced=False),  # unvoiced with text: never transcribed
    dict(line_id="m_06", speaker_id="narrator", text="", voiced=False),  # unvoiced without text: nothing to recover
]


def sidecar(ctx, aid: str, text: str) -> None:
    (ctx.paths.audio_src16 / f"{aid}.json").write_text(json.dumps({"text_used": text}, ensure_ascii=False), encoding="utf-8")


def project(tmp_path, sets=None):
    ctx = _make_ctx(tmp_path, tmp_path, sets={"link.mode": "no_asr", "link.recover_text": True, **(sets or {})})
    build_project(ctx, ROWS)
    sidecar(ctx, "sak_0002", HEARD_1)
    sidecar(ctx, "har_0002", HEARD_2)
    return ctx


def spy_asr(monkeypatch):
    calls: list[dict] = []
    real = link.run_asr

    def spy(ctx, items, lang, names=None, on_items=None):
        calls.append({"audio": sorted(aid for aid, _ in items), "names": names})
        return real(ctx, items, lang, names=names, on_items=on_items)

    monkeypatch.setattr(link, "run_asr", spy)
    return calls


def effective(ctx):
    return {l.line_id: l for l in ctx.lines()}


# ----------------------------------------------------------------------------
# recovery
# ----------------------------------------------------------------------------


def test_only_voiced_lines_without_text_are_transcribed(tmp_path, monkeypatch):
    ctx = project(tmp_path)
    raw_before = ctx.paths.lines.read_bytes()
    assert link.no_text_lines(ctx) == ["m_02", "m_04"]
    calls = spy_asr(monkeypatch)
    summary = run_stage("link", ctx)
    ctx.models.close()

    # one ASR pass, over exactly the two clips whose line has no text, with the recovery backend (+ fallback)
    assert calls == [{"audio": ["har_0002", "sak_0002"], "names": [ctx.cfg.link.recover_asr, ctx.cfg.link.asr_fallback]}]
    loads = [e for e in ctx.models.events if e["event"] == "load"]
    assert [e["model"] for e in loads if e["role"] == "asr"] == ["mock_asr"]
    assert summary["mode"] == "no_asr" and summary["recover"] == {"asr": "mock_asr", "candidates": 2, "recovered": 2, "no_text_left": 0}

    recs = {r.line_id: r for r in read_jsonl(ctx.paths.link, LinkRecord)}
    assert recs["m_02"].model_extra["text_recovered"] == HEARD_1 and recs["m_04"].model_extra["text_recovered"] == HEARD_2
    assert "text_recovered" not in (recs["m_01"].model_extra or {})

    lines = effective(ctx)
    assert lines["m_02"].text_src == HEARD_1 and "text_from_asr" in lines["m_02"].flags
    assert lines["m_04"].text_src == HEARD_2 and "text_from_asr" in lines["m_04"].flags
    assert lines["m_01"].text_src == "ねえ、ちょっと待って！" and "text_from_asr" not in lines["m_01"].flags
    assert lines["m_05"].text_src == "僕は主人公だ。" and lines["m_06"].text_src == ""
    voiced = {l.line_id: l for l in ctx.voiced_lines()}
    assert voiced["m_02"].text_src == HEARD_1 and "text_from_asr" in voiced["m_02"].flags  # every later stage sees it
    assert link.no_text_lines(ctx) == [] and link.text_recovered_lines(ctx) == ["m_02", "m_04"]
    # lines.jsonl stays owned by parse
    assert ctx.paths.lines.read_bytes() == raw_before
    assert [l.text_src for l in ctx.lines(recovered=False) if l.line_id == "m_02"] == [""]
    # the cheap signals now see the recovered text: the clip lengths fit it
    assert recs["m_02"].status == "ok" and recs["m_02"].signals["morae"] > 5 and "no_text" not in recs["m_02"].signals


def test_recovery_runs_in_asr_mode_too_and_is_not_compared_with_itself(tmp_path, monkeypatch):
    ctx = project(tmp_path, sets={"link.mode": "asr"})
    sidecar(ctx, "sak_0001", "ねえちょっと待って")
    sidecar(ctx, "har_0001", "どうしたのさくら")
    calls = spy_asr(monkeypatch)
    summary = run_stage("link", ctx)
    ctx.models.close()
    assert [c["audio"] for c in calls] == [["har_0002", "sak_0002"], ["har_0001", "sak_0001"]]  # recovery pass, then the verification pass
    recs = {r.line_id: r for r in read_jsonl(ctx.paths.link, LinkRecord)}
    assert recs["m_01"].status == "ok" and recs["m_01"].match_score == 1.0
    assert recs["m_02"].status == "ok" and recs["m_02"].asr_text == HEARD_1 and "recovered" in recs["m_02"].reasons[0]
    assert summary["mode"] == "asr" and summary["recover"]["recovered"] == 2
    assert effective(ctx)["m_02"].text_src == HEARD_1


def test_without_recover_text_nothing_is_transcribed(tmp_path, monkeypatch):
    ctx = project(tmp_path, sets={"link.recover_text": False})
    monkeypatch.setattr(link, "run_asr", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ASR must not run")))
    summary = run_stage("link", ctx)
    assert "recover" not in summary
    recs = {r.line_id: r for r in read_jsonl(ctx.paths.link, LinkRecord)}
    assert recs["m_02"].signals["no_text"] is True and recs["m_02"].status == "ok"  # nothing to check the clip against
    assert link.no_text_lines(ctx) == ["m_02", "m_04"] and effective(ctx)["m_02"].text_src == ""


def test_text_with_script_is_never_overwritten_by_a_stale_recovery(tmp_path):
    ctx = project(tmp_path)
    run_stage("link", ctx)
    # a later parse found the script text of m_02 (lines.jsonl is rewritten by parse only)
    lines = load_lines(ctx.paths.lines)
    next(l for l in lines if l.line_id == "m_02").text_src = "台本にあった本当のセリフ"
    write_jsonl(ctx.paths.lines, lines)
    l2 = effective(ctx)["m_02"]
    assert l2.text_src == "台本にあった本当のセリフ" and "text_from_asr" not in l2.flags
    assert effective(ctx)["m_04"].text_src == HEARD_2  # the other one still comes from ASR
    run_stage("link", ctx)  # the stale recovery is dropped from the record of m_02
    recs = {r.line_id: r for r in read_jsonl(ctx.paths.link, LinkRecord)}
    assert "text_recovered" not in (recs["m_02"].model_extra or {}) and recs["m_04"].model_extra["text_recovered"] == HEARD_2


def test_recovered_text_survives_reruns_and_force_transcribes_again(tmp_path, monkeypatch):
    ctx = project(tmp_path)
    run_stage("link", ctx)
    # same project without recovery: the stored text stays attached to its records, no ASR
    ctx2 = _make_ctx(tmp_path, tmp_path, sets={"link.mode": "no_asr"})
    monkeypatch.setattr(link, "run_asr", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ASR must not run")))
    run_stage("link", ctx2)
    assert effective(ctx2)["m_02"].text_src == HEARD_1 and link.no_text_lines(ctx2) == []
    monkeypatch.undo()
    # --force with recover_text transcribes again (previous recoveries are not applied while recomputing)
    sidecar(ctx, "sak_0002", "書き直した認識結果")
    ctx.force = True
    calls = spy_asr(monkeypatch)
    summary = run_stage("link", ctx)
    ctx.models.close()
    assert calls[0]["audio"] == ["har_0002", "sak_0002"] and summary["recover"]["recovered"] == 2
    assert effective(ctx)["m_02"].text_src == "書き直した認識結果"


def test_recovery_survives_a_missing_asr_backend(tmp_path, monkeypatch):
    ctx = project(tmp_path)
    monkeypatch.setattr(link, "run_asr", lambda *a, **k: ({}, "", "env 'asr' not created"))
    summary = run_stage("link", ctx)  # the stage does not fail
    assert summary["recover"] == {"asr": "unavailable: env 'asr' not created", "candidates": 2, "recovered": 0, "no_text_left": 2}
    assert link.no_text_lines(ctx) == ["m_02", "m_04"] and effective(ctx)["m_02"].text_src == ""


def test_empty_asr_result_leaves_the_line_without_text(tmp_path):
    ctx = project(tmp_path)
    sidecar(ctx, "har_0002", "")  # the recognizer heard nothing
    summary = run_stage("link", ctx)
    assert summary["recover"]["recovered"] == 1 and summary["recover"]["no_text_left"] == 1
    assert link.no_text_lines(ctx) == ["m_04"]


# ----------------------------------------------------------------------------
# set_src_text overrides
# ----------------------------------------------------------------------------


def test_set_src_text_override_latest_wins_and_flags(tmp_path):
    ctx = project(tmp_path)
    run_stage("link", ctx)
    ov = ctx.paths.overrides
    append_jsonl(ov, OverrideRecord(line_id="m_02", action="set_src_text", payload={"text": "最初の修正"}))
    append_jsonl(ov, OverrideRecord(line_id="m_01", action="set_text", payload={"text": "Russian override: not a source edit"}))
    append_jsonl(ov, OverrideRecord(line_id="m_02", action="set_src_text", payload={"text": "最後の修正"}))
    append_jsonl(ov, OverrideRecord(line_id="m_01", action="set_src_text", payload={"text": "ねえ、ちょっと待ってよ！"}))
    append_jsonl(ov, OverrideRecord(line_id="nope", action="set_src_text", payload={"text": "x"}))  # unknown ids are ignored
    lines = effective(ctx)
    assert lines["m_02"].text_src == "最後の修正"
    assert "text_from_asr" in lines["m_02"].flags and "text_edited" in lines["m_02"].flags  # edited ASR text keeps both flags
    assert lines["m_01"].text_src == "ねえ、ちょっと待ってよ！" and lines["m_01"].flags == ["text_edited"]  # a script line can be corrected too
    assert lines["m_04"].flags == ["text_from_asr"]
    assert ctx.lines(edits=False)[1].text_src == HEARD_1  # layer switches
    assert [l.text_src for l in ctx.voiced_lines() if l.line_id == "m_02"] == ["最後の修正"]
    # clearing the text of a line makes it a no-text line again
    append_jsonl(ov, OverrideRecord(line_id="m_04", action="set_src_text", payload={"text": ""}))
    assert "m_04" in link.no_text_lines(ctx)


def test_apply_text_patches_ignores_malformed_override_payloads(tmp_path):
    ctx = project(tmp_path, sets={"link.recover_text": False})
    ov = ctx.paths.overrides
    ov.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in [
            {"line_id": "m_01", "action": "set_src_text", "payload": {}},
            {"line_id": "m_01", "action": "set_src_text", "payload": {"text": 5}},
            {"line_id": "m_01", "action": "set_src_text"},
        ]) + "\n",
        encoding="utf-8",
    )
    assert apply_text_patches(ctx.paths, load_lines(ctx.paths.lines))[0].text_src == "ねえ、ちょっと待って！"


def test_translate_follows_the_effective_text(tmp_path):
    """The translation batch key contains the effective source text: an edit is translated again."""
    from vndub.stages import translate

    d = tmp_path / "dict.yaml"
    d.write_text(yaml.safe_dump({"ねえ、ちょっと待って！": "Эй, подожди!"}, allow_unicode=True), encoding="utf-8")
    ctx = project(tmp_path, sets={"translate.mock_dictionary": str(d), "translate.glossary": str(tmp_path / "glossary.yaml")})
    run_stage("link", ctx)
    run_stage("translate", ctx)

    def tgt():
        return {r["line_id"]: r["text_tgt"] for r in read_jsonl(ctx.paths.translation("ru"))}

    assert tgt()["m_02"] == HEARD_1  # recovered text, the mock dictionary does not know it: passthrough
    assert tgt()["m_01"] == "Эй, подожди!"
    append_jsonl(ctx.paths.overrides, OverrideRecord(line_id="m_02", action="set_src_text", payload={"text": "ここは大きな声で言った"}))
    run_stage("translate", ctx)
    assert tgt()["m_02"] == "ここは大きな声で言った" and tgt()["m_01"] == "Эй, подожди!"
    assert translate.lines_to_translate(ctx)  # the stage reads the patched lines


# ----------------------------------------------------------------------------
# parse: voice files without any script text (generic engine / unparsed script)
# ----------------------------------------------------------------------------


def _voice_only_game(tmp_path):
    game = tmp_path / "game"
    (game / "voice").mkdir(parents=True)
    for name, f0 in (("sak_0001", 260), ("sak_0002", 270), ("har_0001", 120), ("har_0002", 125)):
        (game / "voice" / f"{name}.wav").write_bytes(tone_bytes(f0, seconds=0.6))
    return game


@requires_ffmpeg
def test_generic_engine_without_script_yields_voiced_lines_with_empty_text(tmp_path):
    game = _voice_only_game(tmp_path)
    # option off: the engine's own error, exactly as before
    ctx = _make_ctx(tmp_path / "off", game)
    for st in ("detect", "extract"):
        run_stage(st, ctx)
    assert ctx.require_project().engine == "generic"
    with pytest.raises(StageError, match="VNTextPatch"):
        run_stage("parse", ctx)
    # option on (mode C): one voiced line with empty text per file; speakers from the file prefixes
    ctx = _make_ctx(tmp_path / "on", game, sets={"link.recover_text": True, "link.mode": "no_asr"})
    for st in ("detect", "extract"):
        run_stage(st, ctx)
    summary = run_stage("parse", ctx)
    assert summary["lines"] == 4 and summary["voiced"] == 4 and summary["voice_coverage"] == 1.0
    lines = load_lines(ctx.paths.lines)
    assert [l.voice_file for l in lines] == ["voice/har_0001.wav", "voice/har_0002.wav", "voice/sak_0001.wav", "voice/sak_0002.wav"]
    assert all(l.has_voice and l.text_src == "" and "voice_only" in l.flags and l.scene == "voice" for l in lines)
    assert [l.speaker_id for l in lines] == ["har", "har", "sak", "sak"]
    chars = ctx.characters()
    assert set(chars) == {"har", "sak"} and chars["sak"].voice_prefix == "sak"
    # link recovers the text from the audio (mock ASR reads the sidecar next to the 16 kHz clip)
    assert len(link.no_text_lines(ctx)) == 4
    for aid, text in (("sak_0001", "おはよう"), ("sak_0002", "こんにちは"), ("har_0001", "やあ"), ("har_0002", "またね")):
        sidecar(ctx, aid, text)
    s = run_stage("link", ctx)
    assert s["recover"]["recovered"] == 4 and link.no_text_lines(ctx) == []
    assert [l.text_src for l in ctx.lines()] == ["やあ", "またね", "おはよう", "こんにちは"]
    assert all("text_from_asr" in l.flags and "voice_only" in l.flags for l in ctx.lines())
    ctx.models.close()


@requires_ffmpeg
def test_script_with_no_lines_falls_back_to_voice_only_lines(tmp_path):
    game = _voice_only_game(tmp_path)
    empty = tmp_path / "vntp"
    empty.mkdir()
    (empty / "a.json").write_text("[]", encoding="utf-8")  # a text export without any message
    ctx = _make_ctx(tmp_path / "w", game, sets={"link.recover_text": True}, extra={"import_vntextpatch": str(empty)})
    for st in ("detect", "extract"):
        run_stage(st, ctx)
    summary = run_stage("parse", ctx)
    assert summary["voiced"] == 4 and summary["source"].endswith("+voice_only")
    # without the option the empty result stays empty (unchanged behaviour)
    ctx2 = _make_ctx(tmp_path / "w2", game, extra={"import_vntextpatch": str(empty)})
    for st in ("detect", "extract"):
        run_stage(st, ctx2)
    assert run_stage("parse", ctx2)["lines"] == 0


def test_parse_keeps_the_character_description(tmp_path):
    new = {"a": Character(names={"src": "あ", "tgt": ""})}
    old = {"a": Character(names={"src": "あ", "tgt": "А"}, description="строгая учительница, низкий голос", gender="f")}
    merged = parse.merge_characters(new, old)
    assert merged["a"].description == "строгая учительница, низкий голос" and merged["a"].names["tgt"] == "А" and merged["a"].gender == "f"

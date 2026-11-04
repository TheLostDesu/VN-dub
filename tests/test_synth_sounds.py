"""synth / score / pack: non-speech sounds (snoring, kisses, yawns ...): stand-in spelling, the three ``only`` modes, CER, style."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from tests.genproj import build
from vndub.manifest.io import CandidateScore, SelectionRecord, read_json, read_jsonl, write_jsonl
from vndub.stages import genutil as gu
from vndub.stages import pack as pack_stage
from vndub.stages import score, synth
from vndub.text import sounds as snd
from vndub.text.style_prompt import StyleSpec, render, to_fish_tags, to_higgs_tags, to_instruction


@pytest.fixture
def proj(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path / "game")
    build(ctx)
    return ctx


def _cands(ctx, line_id):
    d = ctx.paths.synth_dir(line_id)
    return sorted(p for p in d.glob("*.wav") if not p.name.endswith(".prevc.wav"))


def _sides(ctx, line_id):
    return [read_json(p.with_suffix(".json")) for p in _cands(ctx, line_id) if not p.name.startswith("orig")]


def _say(ctx, line_id, text, plain=None, **extra):
    trs = gu.load_translations(ctx)
    r = trs[line_id]
    r.text_tgt = r.text_tgt_short = r.text_tts = text
    r.text_plain = text if plain is None else plain
    for k, v in extra.items():
        setattr(r, k, v)
    write_jsonl(ctx.paths.translation(ctx.cfg.project.tgt_lang), list(trs.values()))


def _clip(path, sr=16000, secs=0.6):
    t = np.arange(int(sr * secs)) / sr
    sf.write(str(path), (0.2 * np.sin(2 * np.pi * 120 * t)).astype("float32"), sr)
    return path


def test_old_project_snore_gets_the_target_stand_in(proj):
    ctx = proj
    line = ctx.voiced_lines()[0]
    _say(ctx, line.line_id, "З-З-З...")  # what a project translated before the sounds table carries; no `sound` field
    s = synth.run(ctx)
    assert s["failed"] == 0 and s["sounds"]["lines"] == 1 and s["sounds"]["modes"]["respell"] >= 1 and s["sounds"]["silent"] == 0
    sides = _sides(ctx, line.line_id)
    assert sides
    for side in sides:
        assert side["text_used"] == "Хр-р-р... хр-р-р..."
        assert side["text_plain"] == "Хр-р-р... хр-р-р..."
        assert side["sound"] == {"kind": "snore", "n": 1, "mode": "respell", "only": True, "stage": False}
        assert side["style"]["sound"]["kind"] == "snore" and "snore" in side["style"]["events"]
        assert "snoring" in side["style_prompt"] and "no words" in side["style_prompt"]
        assert "+" not in side["text_used"] and "́" not in side["text_used"]
    # a line without a sound is untouched: no sound field in the sidecar nor in the stored style
    for side in _sides(ctx, ctx.voiced_lines()[1].line_id):
        assert "sound" not in side and "sound" not in side["style"]


def test_new_project_record_names_the_kind(proj):
    ctx = proj
    line = ctx.voiced_lines()[0]
    # translate stored the spelling and the LLM's kind; the regexes would find "Хр-р-р" too
    _say(ctx, line.line_id, "Хр-р-р... хр-р-р...", sound={"kind": "snore", "only": True, "stage": False, "source": "llm"})
    synth.run(ctx)
    for side in _sides(ctx, line.line_id):
        assert side["text_used"] == "Хр-р-р... хр-р-р..." and side["sound"]["kind"] == "snore" and side["sound"]["only"] is True
    # the regexes find nothing in a word, but the record says the line is only a snore: the whole text is the stand-in
    _say(ctx, line.line_id, "Он спит", sound={"kind": "snore", "only": True, "stage": False, "source": "llm"})
    synth.run(ctx)
    for side in _sides(ctx, line.line_id):
        assert side["text_used"] == "Хр-р-р... хр-р-р..." and side["sound"]["mode"] == "respell"


def test_stage_direction_is_removed_and_the_event_reaches_the_style(proj):
    ctx = proj
    line = ctx.voiced_lines()[0]
    _say(ctx, line.line_id, "*yawns* Доброе утро.")
    synth.run(ctx)
    for side in _sides(ctx, line.line_id):
        assert side["text_used"] == "Доброе утро."
        assert side["text_plain"] == "Доброе утро."
        assert side["sound"]["kind"] == "yawn" and side["sound"]["stage"] is True and side["sound"]["only"] is False
        assert "yawn" in side["style"]["events"] and "yawning" in side["style_prompt"]
        assert not side["style_prompt"].startswith("Only ")  # speech with a yawn, not a sound-only line


def test_sound_inside_speech_is_spelled_for_the_target(proj):
    ctx = proj
    line = ctx.voiced_lines()[0]
    _say(ctx, line.line_id, "Achoo! Sorry.")
    synth.run(ctx)
    for side in _sides(ctx, line.line_id):
        assert side["text_used"] == "Апчхи! Sorry." and side["sound"]["kind"] == "sneeze" and side["sound"]["only"] is False


def test_only_sample_replaces_the_tts_with_the_clip(proj, tmp_path):
    ctx = proj
    line = ctx.voiced_lines()[0]
    _say(ctx, line.line_id, "З-З-З...")
    clip = _clip(tmp_path / "snore.wav")
    ctx.cfg.synth.sounds.only = "sample"
    ctx.cfg.synth.sounds.samples = {"snore": str(clip)}
    s = synth.run(ctx)
    assert s["failed"] == 0 and s["sounds"]["modes"] == {f"sample:{clip}": 1}
    names = [p.name for p in _cands(ctx, line.line_id) if not p.name.startswith("orig")]
    assert names == ["sample_P_0.wav"]  # one candidate, no TTS take
    side = read_json(ctx.paths.synth_dir(line.line_id) / "sample_P_0.json")
    assert side["backend"] == "sample" and side["sound"]["mode"] == f"sample:{clip}" and side["sound"]["only"] is True
    y, sr = sf.read(str(ctx.paths.synth_dir(line.line_id) / "sample_P_0.wav"))
    assert sr == 16000 and y.ndim == 1 and len(y) == 9600
    # second run: cached
    assert synth.run(ctx)["generated"] == 0
    # score passes it unchecked and selects it automatically
    score.run(ctx)
    rows = read_jsonl(ctx.paths.scores(line.line_id), CandidateScore)
    samples = [r for r in rows if r.backend == "sample"]
    assert len(samples) == 1 and samples[0].passed and samples[0].metrics.get("cer") is None and score.SAMPLE_NOTE in samples[0].notes
    sel = {x.line_id: x for x in read_jsonl(ctx.paths.selection, SelectionRecord)}
    assert sel[line.line_id].status == "auto" and sel[line.line_id].backend == "sample"


def test_sample_without_a_clip_falls_back_to_the_stand_in_and_warns_once(proj, caplog):
    ctx = proj
    ids = [l.line_id for l in ctx.voiced_lines()[:2]]
    for lid in ids:
        _say(ctx, lid, "Хр-р-р...")
    ctx.cfg.synth.sounds.only = "sample"
    ctx.cfg.synth.sounds.samples = {"kiss": "nowhere.wav"}
    with caplog.at_level("WARNING"):
        synth.run(ctx)
    for lid in ids:
        sides = _sides(ctx, lid)
        assert sides and all(s["backend"] == "mock_tts" and s["sound"]["mode"] == "respell" for s in sides)
    assert sum("synth.sounds.only is sample" in r.getMessage() for r in caplog.records) == 1


def test_only_silent_leaves_the_line_out_and_pack_lists_it(proj):
    ctx = proj
    line = ctx.voiced_lines()[0]
    _say(ctx, line.line_id, "З-З-З...")
    synth.run(ctx)
    assert _sides(ctx, line.line_id)  # spelled in the first run
    ctx.cfg.synth.sounds.only = "silent"
    s = synth.run(ctx)
    assert s["sounds"]["silent"] == 1
    assert not [p for p in _cands(ctx, line.line_id) if not p.name.startswith("orig")]  # stale takes are gone
    assert [l for l in ctx.voiced_lines() if l.line_id != line.line_id and _cands(ctx, l.line_id)]  # the other lines are voiced
    sc = score.run(ctx)
    sel = {x.line_id: x for x in read_jsonl(ctx.paths.selection, SelectionRecord)}
    assert line.line_id not in sel  # neither a failure nor a review item
    assert sc["status"].get("review", 0) == 0
    cov = pack_stage.patch_coverage(ctx, [], sel)
    assert cov["left_out_reasons"]["sound_silent"] == 1
    row = next(x for x in cov["left_out_lines"] if x["line_id"] == line.line_id)
    assert row["status"] == "sound_silent"
    notes = "\n".join(pack_stage._coverage_notes(cov))
    assert "only a sound" in notes and "synth.sounds.only" in notes


def test_sounds_can_be_switched_off_and_do_not_move_the_cache_hash(proj):
    ctx = proj
    line = ctx.voiced_lines()[0]
    _say(ctx, line.line_id, "З-З-З...")
    h = synth._synth_hash(ctx.cfg)
    ctx.cfg.synth.sounds.only = "silent"
    ctx.cfg.synth.sounds.table = "configs/other.yaml"
    assert synth._synth_hash(ctx.cfg) == h
    ctx.cfg.synth.sounds.only = "spell"
    ctx.cfg.synth.sounds.table = "configs/sounds.yaml"
    ctx.cfg.synth.sounds.enabled = False
    s = synth.run(ctx)
    assert "sounds" not in s
    for side in _sides(ctx, line.line_id):
        assert side["text_used"] == "З-З-З..." and "sound" not in side


def test_a_sound_line_only_regenerates_its_own_candidates(proj):
    ctx = proj
    synth.run(ctx)
    line = ctx.voiced_lines()[0]
    other = ctx.voiced_lines()[1]
    before = {p.name: p.stat().st_mtime_ns for p in _cands(ctx, other.line_id)}
    _say(ctx, line.line_id, "З-З-З...")
    s = synth.run(ctx)
    assert s["generated"] == len(_sides(ctx, line.line_id))
    assert before == {p.name: p.stat().st_mtime_ns for p in _cands(ctx, other.line_id)}


# ---- score: the CER reference ------------------------------------------------------------------------------------------


def test_cer_reference_and_words_take_sounds_out_of_both_sides():
    ref = score.cer_reference(None, _Tr("Апчхи! Sorry."), [])
    assert "Sorry" in ref and "Апчхи" not in ref
    ref_w, hyp_w = score.cer_words("Хр-р-р... хр-р-р... Ладно.", "хр-р-р ладно")
    assert "Хр" not in ref_w and "Ладно" in ref_w and "р-р" not in hyp_w
    assert score.cer_words("*yawns* Good morning.", "Good morning.")[0].strip(" .!") == "Good morning"
    assert score.cer_over_words("Апчхи! Sorry.", "Sorry.") == 0.0


class _Tr:
    flags: list = []
    model_extra: dict = {}

    def __init__(self, text):
        self.text_plain = self.text_tgt = text


def test_sound_only_candidate_has_no_cer_gate(proj):
    ctx = proj
    line = ctx.voiced_lines()[0]
    _say(ctx, line.line_id, "З-З-З...")
    synth.run(ctx)
    score.run(ctx)
    rows = [r for r in read_jsonl(ctx.paths.scores(line.line_id), CandidateScore) if r.route != "orig"]
    assert rows
    for r in rows:
        assert score.SOUND_NOTE in r.notes and r.metrics.get("cer") is None and "cer" in (r.model_extra or {}).get("na", [])
    sel = {x.line_id: x for x in read_jsonl(ctx.paths.selection, SelectionRecord)}
    assert sel[line.line_id].winner


# ---- style ------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,phrase",
    [("snore", "snoring"), ("kiss", "kiss"), ("yawn", "yawning"), ("gasp", "gasping"), ("sneeze", "sneezing"), ("cough", "coughing"),
     ("hum", "humming"), ("hmph", "huffing in annoyance"), ("shiver", "shivering"), ("eat", "chewing")],
)
def test_style_phrase_per_kind(kind, phrase):
    spec = StyleSpec(events=[kind], sound={"kind": kind, "n": 1, "only": False, "stage": False})
    assert phrase in to_instruction(spec)
    only = StyleSpec(events=[kind], sound={"kind": kind, "n": 1, "only": True, "stage": False})
    text = to_instruction(only)
    assert text.startswith("Only ") and "no words" in text and "Speak in a" not in text


def test_spec_without_a_sound_keeps_its_stored_form():
    assert "sound" not in StyleSpec().to_dict()
    assert StyleSpec.from_dict(StyleSpec(sound={"kind": "yawn", "n": 1, "only": False, "stage": True}).to_dict()).sound["kind"] == "yawn"


def test_no_second_tag_for_a_cough_the_text_already_tags():
    spec = StyleSpec(events=["cough"], sound={"kind": "cough", "n": 1, "only": False, "stage": False})
    assert "[clearing throat]" in to_fish_tags(spec)
    assert "clearing throat" not in to_fish_tags(spec, inline_sound=True)
    assert "<|sfx:cough|>" in to_higgs_tags(spec)
    assert "<|sfx:cough|>" not in to_higgs_tags(spec, inline_sound=True)
    caps = {"instruct": "none", "tags": "fish", "knobs": "none"}
    assert "clearing throat" not in render(spec, caps, sound_inline=True)["text_prefix"]


def test_tag_backend_text_carries_the_table_tag_once(proj, monkeypatch):
    from vndub.backends import registry

    ctx = proj
    line = ctx.voiced_lines()[0]
    _say(ctx, line.line_id, "Кхе-кхе. Ладно.")
    monkeypatch.setitem(registry.get("mock_tts").caps, "tags", "higgs")
    monkeypatch.setitem(registry.get("mock_tts").caps, "instruct", "none")
    table = snd.load_table(ctx.cfg.resolve("configs/sounds.yaml"))
    table["backends"]["mock_tts"] = dict(table["backends"]["higgs_tts3"])
    monkeypatch.setattr(synth, "_sounds_table", lambda c: table)
    synth.run(ctx)
    for side in _sides(ctx, line.line_id):
        assert side["text_used"].count("<|sfx:cough|>") == 1
        assert side["sound"]["mode"] == "tag:<|sfx:cough|>"

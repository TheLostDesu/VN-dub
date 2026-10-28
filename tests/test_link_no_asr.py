"""link.mode: no_asr -- the mapping is verified with cheap signals only, no ASR backend is ever loaded.

Signals: speaker-embedding consistency, duration vs text length, voice-file prefix / index order.
Also: analyze never recognises speech (the aligner only aligns known text).
"""

from __future__ import annotations

import json

import numpy as np

from analysis_helpers import add_orphan, build_project
from conftest import FIXTURES, _make_ctx, requires_ffmpeg
from vndub.manifest.io import Character, LineRecord, LinkRecord, iter_jsonl, read_json, read_jsonl
from vndub.stages import analyze, link
from vndub.stages.common import run_stage
from vndub.text import ja

NO_ASR = {"link.mode": "no_asr"}
TEXT = "今日はいい天気ですね、散歩に行きましょう。"  # ~20 morae
MORAE = ja.count_morae(TEXT)
GOOD_S = round(MORAE / 6.2 + 0.4, 2)  # a typical speaking rate plus lead/trail silence


def row(i: int, spk: str, aid: str | None = None, text: str = TEXT, seconds: float | None = None, f0: float | None = None, **kw):
    r = dict(line_id=f"s_{i:02d}", speaker_id=spk, text=text, seconds=GOOD_S if seconds is None else seconds, f0=f0 or (260 if spk == "sakura" else 120))
    if aid:
        r["audio_id"] = aid
    r.update(kw)
    return r


def two_speakers(n: int = 5) -> list[dict]:
    rows = []
    for k in range(1, n + 1):
        rows.append(row(2 * k - 1, "sakura", f"sak_{k:04d}"))
        rows.append(row(2 * k, "haruto", f"har_{k:04d}"))
    return rows


def new_ctx(tmp_path, sets=None):
    """A StageContext owning ``tmp_path`` (work/out inside it); several can live in one test."""
    return _make_ctx(tmp_path, tmp_path, sets={**NO_ASR, **(sets or {})})


def forbid_asr(monkeypatch) -> None:
    def boom(*a, **k):
        raise AssertionError("an ASR pass was started in link.mode=no_asr")

    monkeypatch.setattr(link, "run_asr", boom)


def run_link(tmp_path, rows, sets=None, monkeypatch=None):
    ctx = new_ctx(tmp_path, sets)
    build_project(ctx, rows)
    if monkeypatch is not None:
        forbid_asr(monkeypatch)
    summary = run_stage("link", ctx)
    recs = {r.line_id: r for r in read_jsonl(ctx.paths.link, LinkRecord)}
    return ctx, summary, recs


def reasons(rec: LinkRecord) -> list[str]:
    return [r.split(":", 1)[0] for r in rec.reasons]


# ----------------------------------------------------------------------------
# no ASR backend is acquired (GPU registry log)
# ----------------------------------------------------------------------------


def test_no_asr_mode_never_loads_an_asr_backend(tmp_path, monkeypatch):
    ctx, summary, recs = run_link(tmp_path, two_speakers(), monkeypatch=monkeypatch)
    assert summary["mode"] == "no_asr" and summary["asr"].startswith("skipped")
    assert summary["ok"] == len(recs) == 10 and summary["review"] == 0 and summary["unmatched"] == 0
    ctx.models.close()
    # the registry log (in memory and the work/gpu_events.jsonl the report reads): the speaker embedder loaded, nothing with role asr
    events = list(iter_jsonl(ctx.paths.work / "gpu_events.jsonl"))
    loads = [e for e in events if e["event"] == "load"]
    assert loads and {e["role"] for e in loads} == {"embedder"}
    assert not [e for e in ctx.models.events if e["role"] == "asr"]
    assert all(r.asr_text == "" and r.match_score == 0.0 for r in recs.values())


def test_asr_mode_still_loads_the_asr_backend(make_ctx, tmp_path):
    """The CLI default (link.mode: asr) is unchanged: the ASR backend is acquired and the summary carries the new keys."""
    ctx = make_ctx(tmp_path)  # default mode
    build_project(ctx, two_speakers(2))
    (ctx.paths.audio_src16 / "sak_0001.json").write_text(json.dumps({"text_used": TEXT}, ensure_ascii=False), encoding="utf-8")
    summary = run_stage("link", ctx)
    ctx.models.close()
    assert ctx.cfg.link.mode == "asr" and summary["mode"] == "asr"
    assert any(e["event"] == "load" and e["role"] == "asr" for e in ctx.models.events)
    assert summary["confidence"] == summary["coverage_ok"] and summary["outliers"] == summary["review"] + summary["unmatched"]
    assert summary["suggest_asr_check"] is False and summary["asr"].startswith("mock_asr")


# ----------------------------------------------------------------------------
# duration vs text length
# ----------------------------------------------------------------------------


def test_duration_check_math():
    # typical rate = sqrt(3 * 13) = 6.24 mora/s -> ratio 1.0
    ratio, morae, bad = link.duration_check(TEXT, MORAE / 6.24, 3.0, 13.0, 1.0)
    assert morae == MORAE and abs(ratio - 1.0) < 0.02 and not bad
    ratio, _, bad = link.duration_check(TEXT, 0.4, 3.0, 13.0, 1.0)  # far too short for the text
    assert bad and ratio < 0.2
    ratio, _, bad = link.duration_check(TEXT, MORAE / 3.0 + 1.5, 3.0, 13.0, 1.0)  # slower than the slowest plausible speech + silence
    assert bad and ratio > 2.0
    _, _, bad = link.duration_check(TEXT, MORAE / 3.0 + 0.8, 3.0, 13.0, 1.0)  # within the silence tolerance
    assert not bad
    # short texts get at least 1.5 s (+ tolerance): a laugh may last a while
    assert link.duration_check("えへへ。", 2.4, 3.0, 13.0, 1.0)[2] is False
    assert link.duration_check("えへへ。", 9.0, 3.0, 13.0, 1.0)[2] is True
    # nothing to compare: no morae / no audio
    assert link.duration_check("……", 2.0, 3.0, 13.0, 1.0) == (None, 0, False)
    assert link.duration_check(TEXT, 0.0, 3.0, 13.0, 1.0)[0] is None


def test_implausible_durations_are_flagged_with_the_ratio(tmp_path, monkeypatch):
    rows = two_speakers(4)
    rows[0] = row(1, "sakura", "sak_0001", seconds=0.3)  # 20 morae in 0.3 s
    rows[2] = row(3, "sakura", "sak_0002", text="はい。", seconds=9.0)  # 2 morae in 9 s
    ctx, summary, recs = run_link(tmp_path, rows, monkeypatch=monkeypatch)
    assert recs["s_01"].status == "review" and recs["s_01"].reasons[0].startswith("duration_implausible:")
    assert float(recs["s_01"].reasons[0].split(":")[1]) < 0.3
    assert recs["s_03"].status == "review" and float(recs["s_03"].reasons[0].split(":")[1]) > 2.0
    assert all(recs[k].status == "ok" for k in recs if k not in ("s_01", "s_03"))
    assert recs["s_01"].signals["dur_ratio"] < 0.3 and recs["s_01"].signals["morae"] == MORAE
    assert summary["review"] == 2 and summary["outliers"] == 2 and summary["signals"] == {"duration_implausible": 2}
    # flagged lines are in the review queue with their reasons and signals
    queue = {q["line_id"]: q for q in read_jsonl(ctx.paths.review_queue)}
    assert set(queue) == {"s_01", "s_03"} and queue["s_01"]["status"] == "review"
    assert queue["s_01"]["reasons"] == recs["s_01"].reasons and queue["s_01"]["signals"]["morae"] == MORAE


def test_sustain_text_counts_for_the_head_line(tmp_path, monkeypatch):
    rows = [
        row(1, "haruto", "har_0001", text="ああ、そうしよう。", seconds=1.0, extra={"sustain_lines": ["s_02"]}, flags=["sustain_head"]),
        row(2, "haruto", text="雨が降りそうだしな。", voiced=False, flags=["voice_sustain"]),
    ]
    rows += [row(3, "sakura", "sak_0001"), row(4, "sakura", "sak_0002"), row(5, "haruto", "har_0002"), row(6, "haruto", "har_0003")]
    # 1.0 s fits "ああ、そうしよう。" alone (7 morae) but the file also says the continuation: 16 morae in 1 s is too fast
    ctx, _, recs = run_link(tmp_path, rows, monkeypatch=monkeypatch)
    assert recs["s_01"].status == "review" and recs["s_01"].signals["morae"] == ja.count_morae("ああ、そうしよう。雨が降りそうだしな。")


# ----------------------------------------------------------------------------
# prefix / index order
# ----------------------------------------------------------------------------


def test_voice_name_parts_and_order_gaps():
    assert link.voice_name_parts("voice/SAK_0012.ogg") == ("sak", 12)
    assert link.voice_name_parts("voice\\har0003.wav") == ("har", 3)
    assert link.voice_name_parts("voice/start_636ae3f5.ogg") is None  # hashed auto-voice names carry no order
    assert link.voice_name_parts("voice/s01_sak_0001.ogg") is None and link.voice_name_parts(None) is None

    def L(lid, scene, vf):
        return LineRecord(line_id=lid, scene=scene, order=0, speaker_id="a", text_src="x", has_voice=True, audio_id=lid, voice_file=vf)

    lines = [L("1", "s1", "v/sak_0001.ogg"), L("2", "s1", "v/sak_0002.ogg"), L("3", "s1", "v/sak_0009.ogg"), L("4", "s1", "v/sak_0003.ogg"),
             L("5", "s1", "v/sak_0004.ogg"), L("6", "s2", "v/sak_0001.ogg"), L("7", "s1", "v/har_0001.ogg"), L("8", "s1", "v/sak_0004.ogg")]
    # 9 -> 3 inverts (both flagged); 3 -> 4 is fine; the same index again (a reused voice) is not an inversion; other scene/prefix are separate
    assert link.order_gaps(lines) == {"3": "sak", "4": "sak"}


def test_order_gap_is_flagged_on_both_lines_of_an_inversion(tmp_path, monkeypatch):
    rows = [row(1, "sakura", "sak_0001"), row(2, "sakura", "sak_0002"), row(3, "sakura", "sak_0004"), row(4, "sakura", "sak_0003"), row(5, "sakura", "sak_0005")]
    rows += [row(6 + i, "haruto", f"har_{i + 1:04d}") for i in range(4)]
    ctx, summary, recs = run_link(tmp_path, rows, monkeypatch=monkeypatch)
    assert recs["s_03"].reasons == ["order_gap"] and recs["s_04"].reasons == ["order_gap"]
    assert recs["s_03"].status == recs["s_04"].status == "review"
    assert summary["signals"] == {"order_gap": 2} and summary["ok"] == 7


def test_prefix_mismatch_uses_the_declared_voice_prefix(tmp_path, monkeypatch):
    rows = two_speakers(5)
    rows.append(row(11, "sakura", "har_0006", f0=260))  # a sakura line carrying a haruto voice file
    ctx = new_ctx(tmp_path)
    lines = build_project(ctx, rows)
    chars = ctx.characters()
    assert chars["sakura"].voice_prefix == "sak"  # inferred by parse/build_characters
    forbid_asr(monkeypatch)
    summary = run_stage("link", ctx)
    recs = {r.line_id: r for r in read_jsonl(ctx.paths.link, LinkRecord)}
    assert recs["s_11"].reasons == ["prefix_mismatch:har"] and recs["s_11"].status == "review"
    assert summary["review"] == 1 and len(lines) == 11
    # characters.yaml can declare several accepted prefixes
    from vndub.manifest.io import save_characters

    chars["sakura"].voice_prefix = "sak, har"
    save_characters(ctx.paths.characters, chars)
    run_stage("link", ctx)
    assert {r.line_id: r for r in read_jsonl(ctx.paths.link, LinkRecord)}["s_11"].status == "ok"


def test_prefix_check_needs_prefixes_that_identify_speakers():
    def L(lid, spk, vf):
        return LineRecord(line_id=lid, scene="s", order=0, speaker_id=spk, text_src="x", has_voice=True, audio_id=lid, voice_file=vf)

    # chapter-style prefixes (every speaker uses ch01 / ch02): a "wrong" prefix means nothing
    chapters = [L(str(i), "a" if i % 2 else "b", f"v/ch0{1 + i // 4}_{i:04d}.ogg") for i in range(8)]
    assert link.prefix_expectations(chapters, {}) == {}
    # one shared prefix for everybody
    shared = [L(str(i), "a" if i % 2 else "b", f"v/v_{i:04d}.ogg") for i in range(8)]
    assert link.prefix_expectations(shared, {}) == {}
    # speaker-specific prefixes: majority per speaker, narrator never checked, declared prefix wins
    mixed = [L(str(i), "a", f"v/aaa_{i:04d}.ogg") for i in range(4)] + [L(str(10 + i), "b", f"v/bbb_{i:04d}.ogg") for i in range(4)] + [L("20", "narrator", "v/nnn_0001.ogg")]
    assert link.prefix_expectations(mixed, {}) == {"a": {"aaa"}, "b": {"bbb"}}
    chars = {"a": Character(names={"src": "a"}, voice_prefix="AAA,x")}
    assert link.prefix_expectations(mixed, chars)["a"] == {"aaa", "x"}
    assert link.prefix_expectations(mixed[:3], {}) == {}  # too few lines to say anything


# ----------------------------------------------------------------------------
# speaker consistency (embedder)
# ----------------------------------------------------------------------------


def fake_embeddings(monkeypatch, vectors: dict[str, np.ndarray]) -> None:
    monkeypatch.setattr(link, "run_embed", lambda ctx, items, on_items=None: ({aid: vectors[aid] for aid, _ in items}, "fake_embedder", ""))


def test_speaker_checks_machine_reasons():
    rng = np.random.default_rng(0)
    base_a, base_b = rng.normal(size=16), rng.normal(size=16)
    embs, owner, spk = {}, {}, {}
    for i in range(5):
        embs[f"a{i}"] = base_a + rng.normal(scale=0.05, size=16)
        owner[f"la{i}"], spk[f"la{i}"] = f"a{i}", "alice"
    for i in range(4):
        embs[f"b{i}"] = base_b + rng.normal(scale=0.05, size=16)
        owner[f"lb{i}"], spk[f"lb{i}"] = f"b{i}", "bob"
    embs["x"] = base_b + rng.normal(scale=0.05, size=16)
    owner["lx"], spk["lx"] = "x", "alice"
    recs = {k: LinkRecord(line_id=k, audio_id=v, status="ok") for k, v in owner.items()}
    link.speaker_checks(recs, owner, spk, embs, distance=0.3, outlier=0.5, judged=set(owner), machine=True)
    assert recs["lx"].status == "review"
    assert any(r.startswith("spk_outlier:") and float(r.split(":")[1]) < 0.5 for r in recs["lx"].reasons)
    assert "spk_cluster:bob" in recs["lx"].reasons
    assert recs["la0"].status == "ok" and recs["la0"].reasons == []


def test_speaker_outlier_in_no_asr_mode(tmp_path, monkeypatch):
    rows = two_speakers(5)
    rng = np.random.default_rng(1)
    base = {"sakura": rng.normal(size=16), "haruto": rng.normal(size=16)}
    vec = {}
    for r in rows:
        spk = r["speaker_id"]
        vec[r["audio_id"]] = base[spk] + rng.normal(scale=0.05, size=16)
    vec["sak_0003"] = base["haruto"] + rng.normal(scale=0.05, size=16)  # the third sakura clip sounds like haruto
    fake_embeddings(monkeypatch, vec)
    ctx, summary, recs = run_link(tmp_path, rows, monkeypatch=monkeypatch)
    bad = next(r for r in recs.values() if r.audio_id == "sak_0003")
    assert bad.status == "review" and any(r.startswith("spk_outlier:") for r in bad.reasons)
    assert sum(1 for r in recs.values() if r.status == "ok") == 9
    assert summary["embedder"] == "fake_embedder" and summary["signals"]["spk_outlier"] == 1
    assert bad.spk_consistency is not None and bad.spk_consistency < ctx.cfg.link.outlier_consistency


# ----------------------------------------------------------------------------
# audio missing, no repairs, summary
# ----------------------------------------------------------------------------


def test_missing_audio_is_unmatched(tmp_path, monkeypatch):
    rows = two_speakers(3) + [row(7, "sakura", voiced=False, flags=["voice_missing"], extra={"voice_file": "voice/gone_0001.ogg"})]
    ctx = new_ctx(tmp_path)
    build_project(ctx, rows)
    (ctx.paths.audio_src16 / "har_0002.wav").unlink()  # the decoded clip of s_04 is gone
    forbid_asr(monkeypatch)
    summary = run_stage("link", ctx)
    recs = {r.line_id: r for r in read_jsonl(ctx.paths.link, LinkRecord)}
    assert recs["s_04"].status == "unmatched" and recs["s_04"].reasons == ["audio_missing:voice/har_0002.ogg"]
    assert recs["s_07"].status == "unmatched" and recs["s_07"].reasons == ["audio_missing:voice/gone_0001.ogg"]
    assert summary["unmatched"] == 2 and summary["ok"] == 5 and summary["lines_judged"] == 7
    # the unmatched line owns no usable voice but the binding itself is not rewritten
    voiced = {l.line_id for l in ctx.voiced_lines()}
    assert "s_04" in voiced and "s_07" not in voiced


def test_no_auto_repair_and_orphans_stay_unbound(tmp_path, monkeypatch):
    rows = two_speakers(3)
    ctx = new_ctx(tmp_path)
    build_project(ctx, rows)
    add_orphan(ctx, "har_99")
    forbid_asr(monkeypatch)
    before = {l.line_id: l.audio_id for l in ctx.voiced_lines()}
    summary = run_stage("link", ctx)
    assert summary["relinked"] == 0 and summary["orphans"] == 1 and summary["orphans_bound"] == 0
    assert {l.line_id: l.audio_id for l in ctx.voiced_lines()} == before
    assert all(r.relinked_from is None for r in read_jsonl(ctx.paths.link, LinkRecord))


def test_confidence_outliers_and_suggest_asr_check(tmp_path, monkeypatch):
    def with_bad(n_bad: int, n: int = 10):
        rows = [row(i + 1, "sakura" if i % 2 == 0 else "haruto", f"{'sak' if i % 2 == 0 else 'har'}_{i + 1:04d}", seconds=(0.3 if i < n_bad else None)) for i in range(n)]
        return rows

    # 3 of 10 lines flagged: confidence 0.7 and the share (0.3) is above link.suggest_asr_share (0.2)
    ctx, s, _ = run_link(tmp_path / "a", with_bad(3), monkeypatch=monkeypatch)
    assert s["confidence"] == s["coverage_ok"] == 0.7 and s["outliers"] == 3 and s["suggest_asr_check"] is True
    # 1 of 10 flagged: a healthy game has a few outliers (interjections, whispers) -> no suggestion
    ctx, s, _ = run_link(tmp_path / "b", with_bad(1), monkeypatch=monkeypatch)
    assert s["confidence"] == 0.9 and s["outliers"] == 1 and s["suggest_asr_check"] is False
    # all of 4 flagged, but fewer than 5 judged lines say nothing
    ctx, s, _ = run_link(tmp_path / "c", with_bad(4, n=4), monkeypatch=monkeypatch)
    assert s["outliers"] == 4 and s["suggest_asr_check"] is False
    # the threshold is configurable
    ctx, s, _ = run_link(tmp_path / "d", with_bad(1), sets={"link.suggest_asr_share": 0.1}, monkeypatch=monkeypatch)
    assert s["suggest_asr_check"] is True


def test_no_asr_rerun_is_stable_and_merges_unjudged_lines(tmp_path, monkeypatch):
    ctx, s1, recs1 = run_link(tmp_path, two_speakers(4), monkeypatch=monkeypatch)
    ctx.only = {"s_01"}
    s2 = run_stage("link", ctx)
    assert s2["lines_judged"] == 1
    assert {r.line_id for r in read_jsonl(ctx.paths.link, LinkRecord)} == set(recs1)  # records of the other lines are kept
    ctx.only = None
    s3 = run_stage("link", ctx)
    assert (s3["ok"], s3["review"]) == (s1["ok"], s1["review"])


# ----------------------------------------------------------------------------
# the mixed fixture: voiced lines are linked and analysed without any ASR
# ----------------------------------------------------------------------------


@requires_ffmpeg
def test_fixture_link_and_analyze_run_without_asr(make_ctx):
    ctx = make_ctx(FIXTURES / "renpy_mini", sets=NO_ASR)
    for st in ("detect", "extract", "parse", "link", "analyze"):
        summary = run_stage(st, ctx)
        if st == "link":
            assert summary["mode"] == "no_asr" and summary["lines_judged"] == 20
            assert summary["ok"] == 20 and summary["confidence"] == 1.0 and summary["suggest_asr_check"] is False
        if st == "analyze":
            assert summary["lines"] == 20 and summary["characters"] == 3  # the unvoiced hero has no passport and no stats
    ctx.models.close()
    roles = {e["role"] for e in ctx.models.events if e["event"] == "load"}
    assert "asr" not in roles and {"embedder", "align", "ser"} <= roles
    # every passport belongs to a voiced line
    passports = sorted(p.stem for p in ctx.paths.analysis.glob("*.json"))
    voiced = sorted(l.line_id for l in ctx.voiced_lines())
    assert passports == voiced and len(voiced) == 20
    assert read_json(ctx.paths.passport(voiced[0]))["backends"]["align"] == "mock_aligner"


# ----------------------------------------------------------------------------
# analyze: alignment of known text, never recognition
# ----------------------------------------------------------------------------


def test_alignment_backends_refuse_asr_models():
    ok, notes = analyze.alignment_backends(["qwen3_aligner", "whisperx_ja", "mock_aligner", ""])
    assert ok == ["qwen3_aligner", "whisperx_ja", "mock_aligner"] and notes == []
    ok, notes = analyze.alignment_backends(["anime_whisper", "whisper_large_v3_turbo", "mock_aligner"])
    assert ok == ["mock_aligner"] and len(notes) == 2 and "would transcribe" in notes[0]
    assert analyze.alignment_backends(["not_a_backend"]) == (["not_a_backend"], [])  # resolved (and reported) by the registry


def test_event_tags_drop_the_recognised_text():
    assert analyze.event_tags("<|ja|><|HAPPY|><|Laughter|><|woitn|>こんにちは") == "<|ja|><|HAPPY|><|Laughter|><|woitn|>"
    assert analyze.event_tags("") == analyze.event_tags(None) == ""


def test_analyze_falls_back_to_vad_timing_when_the_aligner_would_transcribe(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path, sets={"analyze.aligner": "anime_whisper", "analyze.aligner_fallback": "whisper_large_v3_turbo"})
    build_project(ctx, two_speakers(2))
    summary = run_stage("analyze", ctx)
    ctx.models.close()
    assert not [e for e in ctx.models.events if e["role"] == "asr"]
    assert "align" in summary["unavailable"] and "would transcribe" in summary["unavailable"]["align"]
    p = read_json(ctx.paths.passport("s_01"))
    assert "would transcribe" in p["missing"]["align"] and "VAD" in p["missing"]["align"]
    assert p["backends"]["timing"] == "vad_only" and "align" not in p["backends"]
    assert p["prosody"]["rate_mora_per_s"] and p["align"] == {}  # the mora rate still exists: morae / VAD speech time


def test_analyze_with_a_real_aligner_keeps_forced_alignment(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path)
    build_project(ctx, two_speakers(2))
    run_stage("analyze", ctx)
    ctx.models.close()
    p = read_json(ctx.paths.passport("s_01"))
    assert p["backends"]["align"] == "mock_aligner" and "timing" not in p["backends"] and p["missing"] == {}
    assert p["align"]["chars"]
    roles = {e["role"] for e in ctx.models.events if e["event"] == "load"}
    assert "asr" not in roles


def test_analyze_aligns_the_effective_text(make_ctx, tmp_path):
    """Recovered / edited text (StageContext.lines) is what the aligner and the mora rate see."""
    from vndub.manifest.io import OverrideRecord, append_jsonl

    ctx = make_ctx(tmp_path)
    build_project(ctx, two_speakers(2))
    append_jsonl(ctx.paths.overrides, OverrideRecord(line_id="s_01", action="set_src_text", payload={"text": "あ"}))
    run_stage("analyze", ctx)
    ctx.models.close()
    p = read_json(ctx.paths.passport("s_01"))
    assert p["prosody"]["morae"] == 1 and [c["c"] for c in p["align"]["chars"]] == ["あ"]


# ----------------------------------------------------------------------------
# release 1.0 review fixes
# ----------------------------------------------------------------------------


def test_an_empty_clip_is_flagged_audio_empty_not_accepted(tmp_path, monkeypatch):
    import soundfile as sf

    ctx = new_ctx(tmp_path)
    build_project(ctx, two_speakers(4))
    forbid_asr(monkeypatch)
    # sak_0002 is a zero-length file whose index record says 0.0 s
    for p in (ctx.paths.audio_src / "sak_0002.wav", ctx.paths.audio_src16 / "sak_0002.wav"):
        sf.write(str(p), np.zeros(0, dtype=np.float32), 16000, subtype="PCM_16")
    idx = read_jsonl(ctx.paths.audio_index)
    for r in idx:
        if r["id"] == "sak_0002":
            r["duration"] = 0.0
    from vndub.manifest.io import write_jsonl

    write_jsonl(ctx.paths.audio_index, idx)
    summary = run_stage("link", ctx)
    recs = {r.line_id: r for r in read_jsonl(ctx.paths.link, LinkRecord)}
    empty = recs["s_03"]  # sakura's second line
    assert empty.audio_id == "sak_0002" and empty.status == "review" and "audio_empty" in empty.reasons
    assert empty.signals["audio_empty"] is True
    assert summary["signals"].get("audio_empty") == 1
    assert all(recs[k].status == "ok" for k in recs if k != "s_03")
    # an index that merely lacks the length is not an empty clip: the file is asked
    idx = read_jsonl(ctx.paths.audio_index)
    for r in idx:
        r["duration"] = 0.0
    write_jsonl(ctx.paths.audio_index, idx)
    run_stage("link", ctx)
    recs = {r.line_id: r for r in read_jsonl(ctx.paths.link, LinkRecord)}
    assert [k for k, r in recs.items() if "audio_empty" in r.reasons] == ["s_03"]


def test_link_records_of_lines_that_no_longer_exist_are_dropped(tmp_path, monkeypatch):
    from vndub.manifest.io import load_lines, write_jsonl

    ctx = new_ctx(tmp_path)
    rows = two_speakers(4)
    rows[0] = row(1, "sakura", "sak_0001", seconds=0.3)  # a review record that would sit in the queue
    build_project(ctx, rows)
    forbid_asr(monkeypatch)
    run_stage("link", ctx)
    assert "s_01" in {r.line_id for r in read_jsonl(ctx.paths.link, LinkRecord)}
    assert "s_01" in {q["line_id"] for q in read_jsonl(ctx.paths.review_queue)}
    # a re-parse removed the line
    write_jsonl(ctx.paths.lines, [l for l in load_lines(ctx.paths.lines) if l.line_id != "s_01"])
    summary = run_stage("link", ctx)
    recs = {r.line_id: r for r in read_jsonl(ctx.paths.link, LinkRecord)}
    assert "s_01" not in recs and len(recs) == 7
    assert "s_01" not in {q["line_id"] for q in read_jsonl(ctx.paths.review_queue)}
    assert summary["review_queue"] == 0


def test_an_unknown_backend_name_is_unavailable_not_a_keyerror(tmp_path):
    ctx = new_ctx(tmp_path, {"link.speaker_embedder": "ecapa_typo"})
    build_project(ctx, two_speakers(2))
    summary = run_stage("link", ctx)  # used to end the stage with KeyError: unknown backend 'ecapa_typo'
    assert summary["embedder"].startswith("unavailable") and "ecapa_typo" in summary["embedder"]
    # the same for the ASR names: every line goes to review with the reason
    ctx2 = _make_ctx(tmp_path / "b", tmp_path / "b", sets={"link.mode": "asr", "link.asr": "whisper_typo", "link.asr_fallback": ""})
    build_project(ctx2, two_speakers(2))
    s2 = run_stage("link", ctx2)
    assert s2["review"] == 4 and "whisper_typo" in s2["asr"]

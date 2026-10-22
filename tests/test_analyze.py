from __future__ import annotations

import numpy as np
import soundfile as sf

from analysis_helpers import build_project
from vndub.backends.base import BackendUnavailable
from vndub.manifest.io import Passport, load_characters, read_json, save_characters
from vndub.stages import analyze
from vndub.stages.common import run_stage
from vndub.text.style_prompt import ALL_EMOTIONS

ROWS = [
    dict(line_id="s_01", speaker_id="sakura", text="テストで百点を取ったんだよ！", f0=260, seconds=1.2, audio_id="sak_01", amp=0.5),
    dict(line_id="s_02", speaker_id="sakura", text="ねえ、ちょっと待って！", f0=280, seconds=1.2, audio_id="sak_02"),
    dict(line_id="s_03", speaker_id="haruto", text="また明日な。", f0=120, seconds=0.8, audio_id="har_01", amp=0.2),
]


def _two_bursts(path, sr):
    t = np.arange(int(1.2 * sr)) / sr
    y = 0.2 * np.sin(2 * np.pi * 200 * t)
    y[(t < 0.2) | ((t > 0.55) & (t < 0.95)) | (t > 1.1)] = 0.0
    sf.write(str(path), y.astype(np.float32), sr, subtype="PCM_16")


def test_analyze_passports_stats_style(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path)
    build_project(ctx, ROWS)
    # a user edit in characters.yaml must survive
    chars = load_characters(ctx.paths.characters)
    chars["sakura"].gender = "f"
    chars["sakura"].names["tgt"] = "Сакура"
    save_characters(ctx.paths.characters, chars)
    # a clip with an inner pause
    _two_bursts(ctx.paths.audio_src16 / "sak_02.wav", 16000)
    _two_bursts(ctx.paths.audio_src / "sak_02.wav", 24000)

    summary = run_stage("analyze", ctx)
    assert summary["lines"] == 3 and summary["characters"] == 2
    assert summary["backends"]["e2v"] == "mock_ser" and summary["backends"]["f0"] == "librosa_f0"

    p = Passport.model_validate(read_json(ctx.paths.passport("s_01")))
    assert p.duration_s == 1.2
    assert set(p.emotion) == {"e2v", "anime", "adv"}
    assert ctx.paths.abs(p.emotion["e2v"]["embedding_path"]).exists()
    assert sum(p.emotion["e2v"]["probs"].values()) > 0.99
    pros = p.prosody
    assert pros["morae"] == 16 and pros["rate_mora_per_s"] > 0
    assert 230 < pros["f0_median_hz"] < 300
    assert ctx.paths.abs(pros["f0_contour_path"]).exists() and ctx.paths.abs(pros["energy_contour_path"]).exists()
    assert pros["lufs"] is not None and pros["arc"] in ("flat", "rising_end", "falling_end")
    assert p.snr_db is not None and p.bgm is False
    assert p.align["chars"] and p.align["chars"][0]["c"] == "テ"
    assert p.style["emotion"] in ALL_EMOTIONS and p.style["tempo"] in ("slow", "normal", "fast")
    assert p.missing == {}

    p2 = read_json(ctx.paths.passport("s_02"))
    assert len(p2["prosody"]["pauses"]) == 1 and abs(p2["prosody"]["pauses"][0]["d"] - 0.4) < 0.08
    assert abs(p2["prosody"]["leading_silence_s"] - 0.2) < 0.05

    chars = load_characters(ctx.paths.characters)
    st = chars["sakura"].stats
    assert st.n_lines == 2 and st.total_sec == 2.4
    assert st.f0_median_hz and st.rate_median and st.lufs_median and st.energy_db_median
    assert abs(sum(st.emotion_hist.values()) - 1.0) < 0.01
    assert chars["sakura"].gender == "f" and chars["sakura"].names["tgt"] == "Сакура"
    assert chars["sakura"].ref_candidates and all(r.startswith("audio/src/") for r in chars["sakura"].ref_candidates)
    assert chars["haruto"].stats.f0_median_hz < chars["sakura"].stats.f0_median_hz
    ctx.models.close()


def test_analyze_is_cached(make_ctx, tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)
    build_project(ctx, ROWS[2:])
    run_stage("analyze", ctx)
    ctx.models.close()
    before = read_json(ctx.paths.passport("s_03"))

    def boom(*a, **k):
        raise AssertionError("model acquired although every result is cached")

    monkeypatch.setattr(ctx.models, "acquire", boom)
    run_stage("analyze", ctx)
    assert read_json(ctx.paths.passport("s_03")) == before


def test_analyze_records_missing_parts_and_disables_caption(make_ctx, tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path, sets={"analyze.caption": True, "analyze.caption_backend": "qwen2_audio", "analyze.anime_ser": "", "analyze.events": ""})
    build_project(ctx, ROWS[2:])
    real_resolve = ctx.models.resolve

    def resolve(name):
        if name in ("qwen2_audio", "mock_adv"):
            raise BackendUnavailable(name, "env 'hf' not created")
        return real_resolve(name)

    monkeypatch.setattr(ctx.models, "resolve", resolve)
    summary = run_stage("analyze", ctx)
    assert "captions disabled" in summary["caption"]
    p = read_json(ctx.paths.passport("s_03"))
    assert "caption" in p["missing"] and "adv" in p["missing"]
    assert "adv" not in p["emotion"]
    assert p["style"]["intensity"] >= 0.0  # style still built without A/D/V
    ctx.models.close()


def test_aggregate_and_ref_candidates_rules():
    from vndub.manifest.io import LineRecord

    ps = [
        {"duration_s": 4.0, "snr_db": 30, "bgm": False, "prosody": {"f0_median_hz": 200, "energy_db_mean": -20, "lufs": -23, "rate_mora_per_s": 7}, "emotion": {"e2v": {"probs": {"happy": 1.0}}}},
        {"duration_s": 5.0, "snr_db": 35, "bgm": False, "prosody": {"f0_median_hz": 220, "energy_db_mean": -22, "lufs": -21, "rate_mora_per_s": 8}, "emotion": {"e2v": {"probs": {"sad": 1.0}}}},
        {"duration_s": 5.0, "snr_db": 40, "bgm": True, "prosody": {"f0_median_hz": 210}, "emotion": {"e2v": {"probs": {"happy": 1.0}}}},
        {"duration_s": 1.0, "snr_db": 40, "bgm": False, "prosody": {"f0_median_hz": 210}, "emotion": {"e2v": {"probs": {"happy": 1.0}}}},
    ]
    st = analyze.aggregate(ps, 0.6, 0.4)
    assert st["n_lines"] == 4 and st["f0_median_hz"] == 210.0 and st["emotion_hist"]["happy"] == 0.75
    items = [(LineRecord(line_id=str(i), scene="s", order=i, src_wav=f"audio/src/{i}.wav"), p) for i, p in enumerate(ps)]
    refs = analyze.ref_candidates(items, 25, 3, 10, 2, 0.6, 0.4)
    # bgm and too-short clips excluded; one happy + one sad
    assert refs == ["audio/src/0.wav", "audio/src/1.wav"] or refs == ["audio/src/1.wav", "audio/src/0.wav"]


# ---- round 8: a skipped part is told to the registry (the job's warnings) -----------------------------------------------------


def test_skipped_parts_are_recorded_as_unavailable(make_ctx, tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path, sets={"analyze.caption": True, "analyze.caption_backend": "qwen2_audio", "analyze.anime_ser": "", "analyze.events": ""})
    build_project(ctx, ROWS[2:])
    real_resolve = ctx.models.resolve

    def resolve(name):
        if name in ("qwen2_audio", "mock_adv"):
            raise BackendUnavailable(name, "env 'hf' not created")
        return real_resolve(name)

    monkeypatch.setattr(ctx.models, "resolve", resolve)
    run_stage("analyze", ctx)
    got = {(u.stage, u.group): u.reason for u in ctx.models.unavailable}
    assert "env 'hf' not created" in got[("analyze", "adv")]
    assert "env 'hf' not created" in got[("analyze", "caption")]
    assert ("analyze", "e2v") not in got and ("analyze", "f0") not in got  # what ran is not reported
    ctx.models.close()


def test_part_that_fell_back_and_a_part_out_of_vram_are_recorded(make_ctx, tmp_path):
    from tests.p3_helpers import watch_models
    from vndub.gpu import VramBudgetExceeded

    ctx = make_ctx(tmp_path, sets={"analyze.vad": "silero", "analyze.anime_ser": "", "analyze.events": ""})
    build_project(ctx, ROWS[2:])
    watch_models(ctx, unavailable={
        "silero": VramBudgetExceeded("silero", "an orphan process holds the VRAM"),
        "emotion2vec_plus_seed": VramBudgetExceeded("emotion2vec_plus_seed", "needs 4 GB, 1 GB free"),
    })
    summary = run_stage("analyze", ctx)
    got = {(u.stage, u.group): u.reason for u in ctx.models.unavailable}
    assert "orphan process" in got[("analyze", "vad")] and "energy_vad was used instead" in got[("analyze", "vad")]  # a degraded part
    assert "needs 4 GB" in got[("analyze", "e2v")]  # a missing part
    assert summary["backends"]["vad"] == "energy_vad" and "e2v" in summary["unavailable"]
    assert "e2v" in read_json(ctx.paths.passport("s_03"))["missing"]
    ctx.models.close()

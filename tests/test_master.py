"""master: loudness target, stretch limits, source codec/sr/channels, original passthrough."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import requires_ffmpeg
from tests.genproj import build, speechlike
from vndub.audio import features, loudness, stretch
from vndub.manifest.io import FinalIndexRecord, SelectionRecord, read_jsonl, write_jsonl
from vndub.stages import master


def _probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(path)], capture_output=True, text=True, check=True).stdout
    return [s for s in json.loads(out)["streams"] if s["codec_type"] == "audio"][0]


def test_stretch_plan_limits():
    assert not stretch.plan(1.05, 1.0).apply  # inside window
    p = stretch.plan(1.2, 1.0, (0.9, 1.1), 0.15)
    assert p.apply and p.factor == pytest.approx(1.1 / 1.2)
    assert not stretch.plan(1.4, 1.0, (0.9, 1.1), 0.15).apply  # would need > 15 %
    p = stretch.plan(0.85, 1.0, (0.9, 1.1), 0.15)
    assert p.apply and p.factor == pytest.approx(0.9 / 0.85)


def test_trim_to_profile():
    sr = 16000
    y = np.zeros(sr * 2, dtype=np.float32)
    y[int(0.5 * sr) : int(1.5 * sr)] = 0.3 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)
    out = master.trim_to_profile(y, sr, 0.05, 0.1)
    assert len(out) / sr == pytest.approx(1.0 + 0.15, abs=0.05)


@requires_ffmpeg
def test_master_codec_loudness_stretch(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path / "game")
    data = build(ctx, n_per=2, codec="vorbis", stereo_for=("haruto",), interjection=False, src_sr=44100)
    sel = []
    plans = {}
    for i, line in enumerate(data["lines"]):
        src_d = data["audio"][i].duration
        ratio = [1.0, 1.2, 1.4, 1.05][i % 4]
        cand = ctx.paths.synth_dir(line.line_id) / "mock_tts_A_0.wav"
        # a candidate with different loudness and duration (incl. similar silence margins)
        speechlike(cand, src_d * ratio, 200.0, sr=24000, seed=100 + i, amp=0.05 + 0.2 * i)
        plans[line.line_id] = ratio
        sel.append(SelectionRecord(line_id=line.line_id, winner=ctx.paths.rel(cand), backend="mock_tts", route="A", status="auto", scores={"composite": 0.8}))
    write_jsonl(ctx.paths.selection, sel)
    s = master.run(ctx)
    assert s["encoded"] == len(data["lines"]) and "errors" not in s
    finals = {r.line_id: r for r in read_jsonl(ctx.paths.final_index, FinalIndexRecord)}
    chars = data["chars"]
    for line, a in zip(data["lines"], data["audio"]):
        r = finals[line.line_id]
        enc = ctx.paths.abs(r.encoded)
        assert enc.exists() and r.voice_file == line.voice_file and enc.name == Path(line.voice_file).name
        pr = _probe(enc)
        assert pr["codec_name"] == "vorbis" and int(pr["sample_rate"]) == a.sr and int(pr["channels"]) == a.ch
        y, sr = features.load(ctx.paths.abs(r.final_wav), mono=True)  # mono downmix
        target = chars[line.speaker_id].stats.lufs_median
        assert loudness.lufs(y, sr) == pytest.approx(target, abs=1.0)
        assert loudness.true_peak_db(y, sr) <= -1.0 + 0.3
        assert 1 - ctx.cfg.master.stretch.max_stretch - 1e-6 <= r.stretch <= 1 + ctx.cfg.master.stretch.max_stretch + 1e-6
    # cached rerun
    s2 = master.run(ctx)
    assert s2["cached"] == s2["encoded"]


@requires_ffmpeg
def test_master_orig_passthrough_and_review_skip(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path / "game", sets={"master.include_review_lines": "skip"})
    data = build(ctx, n_per=2, codec="vorbis", interjection=False)
    l0, l1 = data["lines"][0], data["lines"][1]
    write_jsonl(
        ctx.paths.selection,
        [
            SelectionRecord(line_id=l0.line_id, winner=l0.src_wav, backend="orig", route="orig", status="override"),
            SelectionRecord(line_id=l1.line_id, winner=l1.src_wav, backend="x", route="A", status="review"),
        ],
    )
    s = master.run(ctx)
    finals = {r.line_id: r for r in read_jsonl(ctx.paths.final_index, FinalIndexRecord)}
    assert l1.line_id not in finals and s["skipped"]["review"] == 1
    enc = ctx.paths.abs(finals[l0.line_id].encoded)
    assert enc.read_bytes() == Path(data["audio"][0].src_path).read_bytes()
    assert finals[l0.line_id].source == "orig"

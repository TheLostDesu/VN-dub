"""ctx.progress(done, total, item) calls of detect / extract / parse / link / analyze / report.

The desktop app's job runner installs extra["progress_cb"]; it may block (pause) or raise (cancel),
so a stage must call it only between items, never with a half-written output on disk.
"""

from __future__ import annotations

import pytest

from analysis_helpers import build_project
from conftest import FIXTURES, requires_ffmpeg
from vndub.manifest.io import read_jsonl
from vndub.stages.common import run_stage


class Cancelled(Exception):
    pass


class Recorder:
    def __init__(self, cancel_stage: str | None = None, cancel_at: int = 1) -> None:
        self.calls: list[tuple[str, int, int, str | None]] = []
        self.cancel_stage = cancel_stage
        self.cancel_at = cancel_at  # raise from the Nth call of cancel_stage on

    def __call__(self, stage: str, done: int, total: int, item: str | None) -> None:
        self.calls.append((stage, done, total, item))
        if stage == self.cancel_stage and len(self.of(stage)) >= self.cancel_at:
            raise Cancelled()

    def of(self, stage: str) -> list[tuple[str, int, int, str | None]]:
        return [c for c in self.calls if c[0] == stage]


def check_progress(rec: Recorder, stage: str, ends_complete: bool = True) -> list[tuple[str, int, int, str | None]]:
    calls = rec.of(stage)
    assert calls, f"{stage} reported no progress"
    assert all(0 <= done <= total and total >= 1 for _, done, total, _ in calls)
    assert [c[1] for c in calls] == sorted(c[1] for c in calls)  # never goes backwards
    if ends_complete:
        assert calls[-1][1] == calls[-1][2]
    return calls


@requires_ffmpeg
def test_stages_report_progress_on_the_fixture(make_ctx):
    rec = Recorder()
    ctx = make_ctx(FIXTURES / "renpy_mini", sets={"link.mode": "no_asr"}, extra={"progress_cb": rec})
    for st in ("detect", "extract", "parse", "link", "analyze", "report"):
        run_stage(st, ctx)
    ctx.models.close()
    # detect: one item per engine plugin
    detect = check_progress(rec, "detect")
    assert detect[-1][2] == detect[-1][1] >= 5 and all(c[3] for c in detect)
    # extract: one item per voice file (finished decodes carry the audio id)
    extract = check_progress(rec, "extract")
    assert extract[-1][1] == 20 and {c[3] for c in extract if c[3]} >= {"sak_0001", "har_0007"}
    check_progress(rec, "parse")
    # link: clips embedded + lines judged; analyze: model passes x lines + passports
    link = check_progress(rec, "link")
    assert {c[3] for c in link if c[3]} >= {"start_000002"}
    analyze = check_progress(rec, "analyze")
    assert analyze[-1][1] == analyze[-1][2] and analyze[-1][2] >= 20 * 5
    assert {c[3] for c in analyze if c[3]} >= {"start_000002", "after_menu_000009"}
    check_progress(rec, "report")
    assert {c[0] for c in rec.calls} == {"detect", "extract", "parse", "link", "analyze", "report"}  # every call names its stage


@requires_ffmpeg
def test_cached_extract_still_reports_completion(make_ctx):
    ctx = make_ctx(FIXTURES / "renpy_mini")
    run_stage("detect", ctx)
    run_stage("extract", ctx)
    rec = Recorder()
    ctx.extra["progress_cb"] = rec
    run_stage("extract", ctx)  # every file is cached: nothing to decode, but the app still gets (20, 20)
    assert rec.of("extract")[-1][1:3] == (20, 20)


def test_no_progress_callback_is_a_noop(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path)
    build_project(ctx, [dict(line_id="s_01", speaker_id="a", text="あ", seconds=0.5)])
    run_stage("link", ctx)  # no extra["progress_cb"]: nothing raised
    assert "progress_cb" not in ctx.extra


@requires_ffmpeg
def test_cancel_during_extract_leaves_no_half_written_index(make_ctx):
    rec = Recorder("extract", cancel_at=3)
    ctx = make_ctx(FIXTURES / "renpy_mini", extra={"progress_cb": rec})
    run_stage("detect", ctx)
    with pytest.raises(Cancelled):
        run_stage("extract", ctx)
    assert not ctx.paths.audio_index.exists()  # written atomically at the end, never partial
    assert not list(ctx.paths.work.glob(".audio_index*"))
    # the stage status records the failure; a rerun (no callback) finishes the work
    ctx.extra.pop("progress_cb")
    s = run_stage("extract", ctx)
    assert s["decoded"] == 20 and len(read_jsonl(ctx.paths.audio_index)) == 20


def test_cancel_during_link_keeps_link_jsonl_whole(make_ctx, tmp_path):
    rows = [dict(line_id=f"s_{i:02d}", speaker_id="a", text="あいうえお", seconds=1.0, audio_id=f"aaa_{i:04d}") for i in range(1, 6)]
    rec = Recorder("link", cancel_at=2)
    ctx = make_ctx(tmp_path, sets={"link.mode": "no_asr"}, extra={"progress_cb": rec})
    build_project(ctx, rows)
    with pytest.raises(Cancelled):
        run_stage("link", ctx)
    # cancelled before the records were written: no file at all; never a partial one
    assert not ctx.paths.link.exists() or len(read_jsonl(ctx.paths.link)) == 5
    ctx.extra.pop("progress_cb")
    run_stage("link", ctx)
    assert len(read_jsonl(ctx.paths.link)) == 5

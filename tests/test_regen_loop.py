"""``cli._regen_loop``: --force is for the first pass only, and the rounds stay inside --only / --limit."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from tests.genproj import build
from vndub import cli
from vndub.manifest.io import SelectionRecord, read_json, write_jsonl
from vndub.stages.common import run_stage


def _write_selection(ctx: Any, regen: list[str], other: list[str] = ()) -> None:
    recs = [SelectionRecord(line_id=lid, status="regen", reason="no candidate passes the gates") for lid in regen]
    recs += [SelectionRecord(line_id=lid, status="auto", reason="best of 2 passing") for lid in other]
    write_jsonl(ctx.paths.selection, recs)


def _fake_ctx(tmp_path, rounds: int = 2, **kw: Any) -> SimpleNamespace:
    return SimpleNamespace(
        cfg=SimpleNamespace(score=SimpleNamespace(regen_rounds=rounds)), paths=SimpleNamespace(selection=tmp_path / "selection.jsonl"),
        extra={}, log=logging.getLogger("vndub.test_regen_loop"), **kw,
    )


# ---- --force is for the first pass only ----------------------------------------------------------------------------------


def test_force_is_off_during_the_rounds_and_back_afterwards(tmp_path, monkeypatch):
    ctx = _fake_ctx(tmp_path, force=True, only=None, limit=None)
    _write_selection(ctx, regen=["a"], other=["b"])
    seen: list[tuple[str, bool]] = []

    def fake_run_stage(name: str, c: Any) -> dict:
        seen.append((name, c.force))
        if name == "score" and c.extra["regen_round"] == 2:
            _write_selection(c, regen=[], other=["a", "b"])
        return {}

    monkeypatch.setattr(cli, "run_stage", fake_run_stage)
    cli._regen_loop(ctx)
    assert seen == [("synth", False), ("score", False), ("synth", False), ("score", False)]
    assert ctx.force is True  # the stages after the loop (master, pack, report) still run with it


def test_force_is_restored_when_a_round_fails(tmp_path, monkeypatch):
    ctx = _fake_ctx(tmp_path, force=True, only=None, limit=None)
    _write_selection(ctx, regen=["a"])

    def boom(name: str, c: Any) -> dict:
        raise RuntimeError("cancelled")

    monkeypatch.setattr(cli, "run_stage", boom)
    with pytest.raises(RuntimeError):
        cli._regen_loop(ctx)
    assert ctx.force is True


def test_a_forced_run_keeps_the_synth_cache_of_the_lines_the_rounds_did_not_touch(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path / "game")
    build(ctx, n_per=2)
    run_stage("synth", ctx)
    run_stage("score", ctx)
    cache_file = ctx.paths.cache / "synth.json"
    before = set(read_json(cache_file) or {})
    assert len(before) > 3
    ids = [l.line_id for l in ctx.voiced_lines()]
    _write_selection(ctx, regen=ids[:1], other=ids[1:])
    ctx.force = True  # `vndub run --force`: the first pass has just done its work
    cli._regen_loop(ctx)
    assert ctx.force is True
    after = set(read_json(cache_file) or {})
    assert before <= after, "the next run must still find every take of the other lines in the cache"
    assert len(after) > len(before)  # and the round's new takes


# ---- the rounds stay inside --only / --limit -----------------------------------------------------------------------------


def test_rounds_only_count_and_run_the_lines_in_scope(make_ctx, tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path / "game")
    build(ctx, n_per=2)
    ids = [l.line_id for l in ctx.voiced_lines()]
    assert len(ids) >= 4
    ctx.cfg.score.regen_rounds = 2
    _write_selection(ctx, regen=ids[:3], other=ids[3:])
    calls: list[tuple[int, int]] = []
    ran: list[str] = []
    monkeypatch.setattr(cli, "run_stage", lambda name, c: ran.append(name) or {})

    ctx.only = {ids[1], ids[2], ids[4]}  # the preview: a few lines; ids[4] is not in regen at all
    cli._regen_loop(ctx, on_round=lambda rnd, total, n: calls.append((rnd, n)))
    assert calls == [(1, 2), (2, 2)]  # not 3: ids[0] is a regen line of the project, but not of this run
    assert ran == ["synth", "score", "synth", "score"]

    calls.clear()
    ran.clear()
    ctx.only = {ids[4]}  # nothing to redo inside the scope: no round at all
    cli._regen_loop(ctx, on_round=lambda rnd, total, n: calls.append((rnd, n)))
    assert calls == [] and ran == []

    ctx.only = None
    ctx.limit = 2  # the first two lines of the run
    cli._regen_loop(ctx, on_round=lambda rnd, total, n: calls.append((rnd, n)))
    assert calls == [(1, 2), (2, 2)]

    calls.clear()
    ctx.limit = None  # no scope: every regen line, as before
    cli._regen_loop(ctx, on_round=lambda rnd, total, n: calls.append((rnd, n)))
    assert calls == [(1, 3), (2, 3)]

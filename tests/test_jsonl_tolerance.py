"""JSONL files that a hard kill cut in the middle of a record.

``iter_jsonl`` used to raise on the half-written last line, and report / overrides / gpu_events stayed broken until someone
edited the file by hand. A cut last line is skipped (with a warning); damage with records after it is still an error.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from vndub.manifest.io import append_jsonl, iter_jsonl, read_jsonl


def test_a_cut_last_line_is_skipped_with_a_warning(tmp_path: Path, caplog) -> None:
    p = tmp_path / "gpu_events.jsonl"
    p.write_bytes(b'{"a": 1}\n{"b": 2}\n{"event": "loa')
    with caplog.at_level(logging.WARNING, logger="vndub.manifest.io"):
        assert list(iter_jsonl(p)) == [{"a": 1}, {"b": 2}]
        assert read_jsonl(p) == [{"a": 1}, {"b": 2}]
    warned = [r.getMessage() for r in caplog.records if r.name == "vndub.manifest.io"]
    assert warned and "incomplete last line" in warned[0] and f"{p}:3:" in warned[0]


def test_blank_lines_after_the_cut_one_do_not_make_it_a_middle_line(tmp_path: Path) -> None:
    p = tmp_path / "e.jsonl"
    p.write_bytes(b'{"a": 1}\n{"b": \n\n  \n')
    assert list(iter_jsonl(p)) == [{"a": 1}]


def test_a_bad_line_with_records_after_it_is_still_an_error(tmp_path: Path) -> None:
    p = tmp_path / "e.jsonl"
    p.write_bytes(b'{"a": 1}\n{"b": \n{"c": 3}\n')
    with pytest.raises(ValueError, match=r"e\.jsonl:2: invalid JSON"):
        list(iter_jsonl(p))


def test_appending_to_a_cut_file_drops_the_fragment_instead_of_gluing_to_it(tmp_path: Path, caplog) -> None:
    p = tmp_path / "overrides.jsonl"
    p.write_bytes(b'{"a": 1}\n{"b": ')
    with caplog.at_level(logging.WARNING, logger="vndub.manifest.io"):
        append_jsonl(p, {"c": 3})
    assert any("dropped an incomplete last line" in r.getMessage() for r in caplog.records)
    assert p.read_bytes() == b'{"a": 1}\n{"c":3}\n'  # the fragment is gone, the record is whole
    assert list(iter_jsonl(p)) == [{"a": 1}, {"c": 3}]  # and the file reads without a warning-worthy line
    append_jsonl(p, {"d": 4})
    assert list(iter_jsonl(p)) == [{"a": 1}, {"c": 3}, {"d": 4}]


def test_appending_after_a_whole_record_that_lacks_its_newline_keeps_both(tmp_path: Path) -> None:
    p = tmp_path / "e.jsonl"
    p.write_bytes(b'{"a": 1}')
    append_jsonl(p, {"c": 3})
    assert list(iter_jsonl(p)) == [{"a": 1}, {"c": 3}]


def test_append_is_on_disk_when_it_returns(tmp_path: Path) -> None:
    p = tmp_path / "sub" / "events.jsonl"
    append_jsonl(p, {"n": 1})
    with open(p, "rb") as other:  # a second handle sees the whole line: nothing waits in a buffer
        assert other.read() == b'{"n":1}\n'
    append_jsonl(p, {"n": 2})
    assert [r["n"] for r in iter_jsonl(p)] == [1, 2]


def test_atomic_write_retries_a_refused_replace(tmp_path, monkeypatch):
    """Windows refuses the rename while another process reads the target (PermissionError 5): a few short retries
    instead of a failed stage."""
    import os

    from vndub.manifest import io as mio

    real = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError(13, "The process cannot access the file because it is being used by another process")
        return real(src, dst)

    monkeypatch.setattr(mio.os, "replace", flaky)
    monkeypatch.setattr(mio.time, "sleep", lambda s: None)
    target = tmp_path / "x.json"
    mio._atomic_write_bytes(target, b"{}")
    assert target.read_bytes() == b"{}" and calls["n"] == 3
    assert [p for p in tmp_path.iterdir()] == [target], "no tmp file left behind"

    def always(src, dst):
        raise PermissionError(13, "busy")

    monkeypatch.setattr(mio.os, "replace", always)
    import pytest

    with pytest.raises(PermissionError):
        mio._atomic_write_bytes(tmp_path / "y.json", b"{}")
    assert not list(tmp_path.glob(".y.json.*")), "the tmp file is removed when every try failed"

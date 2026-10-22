"""Manifest IO (atomic writes, schemas, merge rules), stage cache invalidation, SQLite index."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from vndub.manifest import index
from vndub.manifest.cache import FileHasher, StageCache
from vndub.manifest.io import (
    Character,
    LineRecord,
    append_jsonl,
    load_characters,
    merge_by_key,
    read_jsonl,
    read_yaml,
    save_characters,
    write_jsonl,
    write_yaml,
)
from vndub.paths import WorkPaths


def test_jsonl_roundtrip_typed(tmp_path: Path) -> None:
    p = tmp_path / "m" / "lines.jsonl"
    recs = [
        LineRecord(line_id="s_000001", scene="s", order=1, speaker_raw="寧々", speaker_id="nene", text_src="……本当に？", voice_file="voice/nen_0123.ogg", has_voice=True),
        LineRecord(line_id="s_000002", scene="s", order=2, text_src="地の文"),
    ]
    assert write_jsonl(p, recs) == 2
    back = read_jsonl(p, LineRecord)
    assert back[0].text_src == "……本当に？" and back[0].has_voice
    assert back[1].speaker_id == "narrator" and back[1].voice_file is None
    # extra fields survive (additive schemas)
    write_jsonl(p, [{**recs[0].model_dump(), "sustain_lines": ["s_000002"]}])
    assert read_jsonl(p, LineRecord)[0].model_extra == {"sustain_lines": ["s_000002"]}


def test_atomic_write_leaves_no_temp_files(tmp_path: Path) -> None:
    p = tmp_path / "a.jsonl"
    write_jsonl(p, [{"line_id": "x"}])
    write_jsonl(p, [{"line_id": "y"}])
    assert [r["line_id"] for r in read_jsonl(p)] == ["y"]
    assert [f.name for f in tmp_path.iterdir()] == ["a.jsonl"]


def test_append_and_bad_json(tmp_path: Path) -> None:
    p = tmp_path / "o.jsonl"
    append_jsonl(p, {"line_id": "a", "action": "accept"})
    append_jsonl(p, {"line_id": "b", "action": "reject"})
    assert len(read_jsonl(p)) == 2
    p.write_text('{"ok": 1}\n{broken\n', encoding="utf-8")
    with pytest.raises(ValueError, match=":2:"):
        read_jsonl(p)


def test_merge_by_key_is_shallow_and_appends() -> None:
    base = [{"line_id": "a", "x": 1}, {"line_id": "b", "x": 2}]
    out = merge_by_key(base, [{"line_id": "b", "y": 3}, {"line_id": "c", "x": 9}])
    assert out == [{"line_id": "a", "x": 1}, {"line_id": "b", "x": 2, "y": 3}, {"line_id": "c", "x": 9}]
    assert base[1] == {"line_id": "b", "x": 2}  # input untouched


def test_characters_yaml_roundtrip_unicode(tmp_path: Path) -> None:
    p = tmp_path / "characters.yaml"
    chars = {"nene": Character(names={"src": "寧々", "tgt": "Нэнэ"}, gender="f", voice_prefix="nen")}
    chars["nene"].stats.f0_median_hz = 265.0
    save_characters(p, chars)
    assert "寧々" in p.read_text(encoding="utf-8")  # allow_unicode
    back = load_characters(p)
    assert back["nene"].names["tgt"] == "Нэнэ" and back["nene"].stats.f0_median_hz == 265.0
    write_yaml(tmp_path / "x.yaml", {"a": [1, 2]})
    assert read_yaml(tmp_path / "x.yaml") == {"a": [1, 2]}


def test_stage_cache_invalidation(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    f = tmp_path / "in.wav"
    f.write_bytes(b"RIFF....data")
    out = tmp_path / "out.json"

    c = StageCache(cache_dir, "analyze", "1", "cfgA")
    k = c.key("line1", {"model": "e2v"}, files=[f])
    assert not c.fresh("line1", k, [out])
    out.write_text("{}")
    c.put("line1", k)
    c.save()

    # same inputs -> fresh after reload (resumable)
    c2 = StageCache(cache_dir, "analyze", "1", "cfgA")
    assert c2.fresh("line1", c2.key("line1", {"model": "e2v"}, files=[f]), [out])
    # missing output -> recompute
    out.unlink()
    assert not c2.fresh("line1", c2.key("line1", {"model": "e2v"}, files=[f]), [out])
    out.write_text("{}")
    # content change -> new key
    f.write_bytes(b"RIFF....data2")
    os.utime(f, ns=(1, 1))
    assert not c2.fresh("line1", c2.key("line1", {"model": "e2v"}, files=[f]), [out])
    # config change / stage version change -> new key
    c3 = StageCache(cache_dir, "analyze", "1", "cfgB")
    assert c3.key("line1", {"model": "e2v"}) != c2.key("line1", {"model": "e2v"})
    c4 = StageCache(cache_dir, "analyze", "2", "cfgA")
    assert c4.key("line1", {"model": "e2v"}) != c2.key("line1", {"model": "e2v"})


def test_file_hasher_memo(tmp_path: Path) -> None:
    f = tmp_path / "a.bin"
    f.write_bytes(b"x" * 1000)
    h = FileHasher(tmp_path / "memo.json")
    first = h(f)
    assert h(f) == first
    h.save()
    assert FileHasher(tmp_path / "memo.json")(f) == first
    assert h(tmp_path / "missing") == "missing"


def test_index_rebuild_and_query(tmp_path: Path) -> None:
    paths = WorkPaths.create(tmp_path / "work", tmp_path / "out")
    write_jsonl(paths.lines, [LineRecord(line_id="s_1", scene="s", order=1, speaker_id="a", has_voice=True, flags=["x"])])
    write_jsonl(paths.link, [{"line_id": "s_1", "match_score": 0.91, "status": "ok"}])
    write_jsonl(paths.selection, [{"line_id": "s_1", "status": "auto", "scores": {"composite": 0.8}}])
    index.rebuild(paths)
    rows = index.query(paths, "SELECT l.line_id, k.match_score, s.scores_composite FROM lines l JOIN link k USING(line_id) JOIN selection s USING(line_id)")
    assert rows == [{"line_id": "s_1", "match_score": 0.91, "scores_composite": 0.8}]
    assert index.table_exists(paths, "lines") and not index.table_exists(paths, "translation_ru")


class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_put_checkpoints_by_time_and_save_is_cheap_when_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vndub.manifest import cache as cache_mod

    clock = _Clock()
    c = StageCache(tmp_path / "cache", "synth", "1", "cfg")
    c._clock = clock
    c._last_save = clock()
    c.put("a", "k1")
    assert not c.path.exists()  # fewer than 50 entries and the last save is recent: nothing written yet
    clock.t += 5
    c.put("b", "k2")
    assert not c.path.exists()
    clock.t += 6  # 11 s since the last save: the next put is a checkpoint, a kill now loses nothing
    c.put("c", "k3")
    assert StageCache(tmp_path / "cache", "synth", "1", "cfg")._items == {"a": "k1", "b": "k2", "c": "k3"}
    # the clock restarts at every save
    clock.t += 3
    c.put("d", "k4")
    assert StageCache(tmp_path / "cache", "synth", "1", "cfg")._items == {"a": "k1", "b": "k2", "c": "k3"}
    # the count threshold still works on its own
    for i in range(cache_mod.CHECKPOINT_ITEMS):
        c.put(f"n{i}", "k")
    assert "n0" in StageCache(tmp_path / "cache", "synth", "1", "cfg")._items
    # save() with nothing dirty writes nothing
    c.save()
    writes: list[Path] = []
    real = cache_mod.write_json
    monkeypatch.setattr(cache_mod, "write_json", lambda p, *a, **k: (writes.append(Path(p)), real(p, *a, **k))[1])
    c.save()
    c.save()
    assert writes == []
    c.drop("a")
    c.save()
    assert writes == [c.path]


def test_rekey_accepts_and_migrates_legacy_entries(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    out = tmp_path / "out.wav"
    out.write_bytes(b"x")
    old = StageCache(cache_dir, "synth", "1", "cfg-with-dirs")
    old_key = old.key("cand", {"text": "t"})
    old.put("l1|a", old_key)
    old.save()

    new = StageCache(cache_dir, "synth", "1", "cfg-with-dirs")
    new.rekey("cfg-without-dirs")
    key = new.key("cand", {"text": "t"})
    assert key != old_key
    assert new.fresh("l1|a", key, [out])  # the entry of the old key still counts ...
    assert new._items["l1|a"] == key  # ... and is migrated to the new one
    assert not new.fresh("l1|a", new.key("cand", {"text": "changed"}), [out])  # other inputs are not fresh
    new.save()
    # after the migration the new hash alone is enough
    again = StageCache(cache_dir, "synth", "1", "cfg-without-dirs")
    assert again.fresh("l1|a", again.key("cand", {"text": "t"}), [out])
    # a missing output is still stale, and nothing is migrated for it
    old2 = StageCache(tmp_path / "c2", "synth", "1", "A")
    old2.put("x", old2.key("i"))
    old2.save()
    new2 = StageCache(tmp_path / "c2", "synth", "1", "A")
    new2.rekey("B")
    assert not new2.fresh("x", new2.key("i"), [tmp_path / "missing.wav"])
    assert new2._items["x"] == old2.key("i")
    new2.rekey("B")  # idempotent
    assert new2.legacy_config_hashes == ["A"]

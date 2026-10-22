"""Sub-batching helper of the adapter protocol (adapters/common/rpc.py: run_in_batches) and the batch methods
that use it. No model is loaded: fake ModelImpl subclasses and stub model objects stand in for the libraries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from adapters.common import rpc
from adapters.common.rpc import ModelImpl, UnavailableError, WeightsMissingError, make_test_clip, run_in_batches


def _items(n: int) -> list[dict[str, Any]]:
    return [{"id": f"i{k}", "v": k} for k in range(n)]


# ---------------------------------------------------------------------------
# run_in_batches
# ---------------------------------------------------------------------------


def test_batches_are_cut_by_size_and_keep_item_order() -> None:
    calls: list[list[str]] = []

    def batch_fn(sub):
        calls.append([it["id"] for it in sub])
        return [{"sq": it["v"] ** 2} for it in sub]

    res = run_in_batches(_items(7), 3, batch_fn, lambda it: {"sq": -1})
    assert calls == [["i0", "i1", "i2"], ["i3", "i4", "i5"]]  # the lone last item goes to item_fn
    assert [r["id"] for r in res] == [f"i{k}" for k in range(7)]
    assert [r["sq"] for r in res] == [0, 1, 4, 9, 16, 25, -1]


def test_sort_key_orders_the_batches_but_not_the_result() -> None:
    seen: list[list[int]] = []

    def batch_fn(sub):
        seen.append([it["v"] for it in sub])
        return [{"v": it["v"]} for it in sub]

    items = [{"id": "a", "v": 5}, {"id": "b", "v": 1}, {"id": "c", "v": 4}, {"id": "d", "v": 2}]
    res = run_in_batches(items, 2, batch_fn, lambda it: {"v": it["v"]}, sort_key=lambda it: it["v"])
    assert seen == [[1, 2], [4, 5]]
    assert [(r["id"], r["v"]) for r in res] == [("a", 5), ("b", 1), ("c", 4), ("d", 2)]


def test_failing_sub_batch_is_redone_item_by_item() -> None:
    def batch_fn(sub):
        if any(it["v"] == 4 for it in sub):
            raise RuntimeError("boom in batch")
        return [{"ok": it["v"]} for it in sub]

    def item_fn(it):
        if it["v"] == 4:
            raise ValueError("bad clip")
        return {"ok": it["v"], "path": "item"}

    res = run_in_batches(_items(6), 3, batch_fn, item_fn)
    assert [r["id"] for r in res] == [f"i{k}" for k in range(6)]
    assert res[:3] == [{"id": f"i{k}", "ok": k} for k in range(3)]  # first sub-batch stayed batched
    assert res[3] == {"id": "i3", "ok": 3, "path": "item"}  # second one fell back
    assert res[4]["error"] == "ValueError: bad clip" and "ok" not in res[4]
    assert res[5] == {"id": "i5", "ok": 5, "path": "item"}


def test_exception_entries_from_batch_fn_become_error_rows_without_fallback() -> None:
    calls = {"batch": 0, "item": 0}

    def batch_fn(sub):
        calls["batch"] += 1
        return [ValueError("unreadable") if it["v"] == 1 else {"v": it["v"]} for it in sub]

    def item_fn(it):
        calls["item"] += 1
        return {}

    res = run_in_batches(_items(3), 8, batch_fn, item_fn)
    assert calls == {"batch": 1, "item": 0}
    assert res[0] == {"id": "i0", "v": 0}
    assert res[1] == {"id": "i1", "error": "ValueError: unreadable"}
    assert res[2] == {"id": "i2", "v": 2}


def test_wrong_result_count_falls_back() -> None:
    res = run_in_batches(_items(4), 4, lambda sub: [{"x": 1}], lambda it: {"x": it["v"]})
    assert [r["x"] for r in res] == [0, 1, 2, 3]


def test_oom_in_batch_frees_cuda_and_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    freed: list[int] = []
    monkeypatch.setattr(rpc, "free_cuda", lambda: freed.append(1))

    def batch_fn(sub):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    res = run_in_batches(_items(4), 4, batch_fn, lambda it: {"v": it["v"]})
    assert [r["v"] for r in res] == [0, 1, 2, 3]
    assert freed == [1]


def test_single_item_sub_batch_uses_the_item_path() -> None:
    def batch_fn(sub):
        raise AssertionError("batch_fn must not run for one item")

    assert run_in_batches(_items(1), 8, batch_fn, lambda it: {"v": 7}) == [{"id": "i0", "v": 7}]
    # batch_size 1 (the "no padded batching" setting of some adapters) is the plain per-item loop
    assert [r["v"] for r in run_in_batches(_items(3), 1, batch_fn, lambda it: {"v": it["v"]})] == [0, 1, 2]


def test_empty_input_and_degenerate_batch_size() -> None:
    assert run_in_batches([], 8, lambda s: [], lambda it: {}) == []
    assert [r["v"] for r in run_in_batches(_items(2), 0, lambda s: [{"v": 9}] * len(s), lambda it: {"v": it["v"]})] == [0, 1]


@pytest.mark.parametrize("exc", [WeightsMissingError("m"), UnavailableError("gone")])
def test_weights_and_availability_errors_propagate(exc: Exception) -> None:
    def batch_fn(sub):
        raise exc

    with pytest.raises(type(exc)):
        run_in_batches(_items(3), 3, batch_fn, lambda it: {})

    def item_fn(it):
        raise exc

    with pytest.raises(type(exc)):
        run_in_batches(_items(1), 3, batch_fn, item_fn)


def test_batch_fn_leaving_an_item_unfilled_is_an_error_not_an_empty_success() -> None:
    res = run_in_batches(_items(3), 3, lambda sub: [{"v": 1}, None, {"v": 3}], lambda it: {})
    assert res[0] == {"id": "i0", "v": 1} and res[2] == {"id": "i2", "v": 3}
    assert res[1]["error"].startswith("RuntimeError") and "v" not in res[1]


def test_sort_key_is_not_evaluated_without_batching() -> None:
    def key(it):
        raise AssertionError("no need to read durations for batch_size 1")

    assert len(run_in_batches(_items(3), 1, lambda s: [], lambda it: {}, sort_key=key)) == 3


def test_item_fn_none_payload_and_error_payload() -> None:
    res = run_in_batches(_items(2), 1, lambda s: [], lambda it: None if it["v"] == 0 else {"error": "model said no"})
    assert res == [{"id": "i0"}, {"id": "i1", "error": "model said no"}]


# ---------------------------------------------------------------------------
# ModelImpl.run with a fake batch_<op>
# ---------------------------------------------------------------------------


class FakeBatchModel(ModelImpl):
    """``batch_x`` raises for the sub-batch that holds a "poison" item, ``op_x`` fails only for the item "bad"."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params or {"name": "fake", "options": {"batch_size": 2}})
        self.batch_calls: list[list[str]] = []

    def op_x(self, item: dict[str, Any], **kw: Any) -> dict[str, Any]:
        if item["text"] == "bad":
            raise ValueError("bad item")
        return {"y": item["text"].upper(), "via": "item", "kw": kw}

    def batch_x(self, items: list[dict[str, Any]], **kw: Any) -> list[dict[str, Any]]:
        def chunk(sub):
            self.batch_calls.append([it["id"] for it in sub])
            if any(it["text"] in ("poison", "bad") for it in sub):
                raise RuntimeError("batch exploded")
            return [{"y": it["text"].upper(), "via": "batch", "kw": kw} for it in sub]

        return run_in_batches(items, self.sub_batch_size(), chunk, lambda it: self.op_x(it, **kw), label=self.name)


def test_model_impl_run_dispatches_to_batch_and_keeps_errors_per_item() -> None:
    m = FakeBatchModel()
    items = [{"id": k, "text": t} for k, t in enumerate(["a", "b", "poison", "c", "bad", "d", "e"])]
    res = m.run("x", items, lang="ja")
    assert [r["id"] for r in res] == list(range(7))
    assert [r.get("y") for r in res] == ["A", "B", "POISON", "C", None, "D", "E"]
    assert res[4]["error"] == "ValueError: bad item"
    # sub-batches of 2: (a,b) ok -> batch; (poison,c) raised -> item path; (bad,d) raised -> item path; (e) alone -> item path
    assert [r.get("via") for r in res] == ["batch", "batch", "item", "item", None, "item", "item"]
    assert res[0]["kw"] == {"lang": "ja"}
    assert m.batch_calls == [[0, 1], [2, 3], [4, 5]]


def test_sub_batch_size_option() -> None:
    assert ModelImpl({"options": {"batch_size": 3}}).sub_batch_size() == 3
    assert ModelImpl({"options": {"batch_size": "5"}}).sub_batch_size() == 5
    assert ModelImpl({"options": {"batch_size": 0}}).sub_batch_size() == 1
    assert ModelImpl({"options": {"batch_size": "x"}}).sub_batch_size() == 8
    assert ModelImpl({}).sub_batch_size() == 8


# ---------------------------------------------------------------------------
# adapters with stub library objects (no weights, no torch)
# ---------------------------------------------------------------------------


def _clip(tmp: Path, name: str, seconds: float) -> dict[str, Any]:
    return {"id": name, "wav": make_test_clip(tmp / f"{name}.wav", seconds=seconds), "text": "テスト", "lang": "ja", "out": str(tmp / f"{name}.npy")}


def test_sensevoice_batch_events_one_generate_call_truncation_and_bad_clip(tmp_path: Path) -> None:
    import soundfile as sf

    from adapters.funasr.sensevoice_impl import SenseVoice

    class Stub:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def generate(self, input, **kw):
            self.calls.append({"input": list(input) if isinstance(input, list) else [input], **kw})
            paths = input if isinstance(input, list) else [input]
            out = []
            for p in paths:
                dur = sf.info(p).duration
                tag = "<|HAPPY|><|Laughter|>" if dur > 1.5 else "<|SAD|><|Cry|>"
                out.append({"key": Path(p).stem, "text": f"<|ja|>{tag}<|woitn|>dur{dur:.1f}"})
            return out

    m = SenseVoice({"name": "sensevoice", "options": {"batch_size": 8, "max_sec": 2.0}})
    m.model = Stub()
    items = [_clip(tmp_path, "long", 3.0), _clip(tmp_path, "short", 1.0), {"id": "gone", "wav": str(tmp_path / "missing.wav")}, _clip(tmp_path, "mid", 2.0)]
    res = m.run("events", items)
    assert [r["id"] for r in res] == ["long", "short", "gone", "mid"]
    assert len(m.model.calls) == 1 and len(m.model.calls[0]["input"]) == 3 and m.model.calls[0]["batch_size"] == 3
    assert res[0]["truncated"] is True and "dur2.0" in res[0]["text"]  # cut to max_sec before the call
    assert res[0]["emotion"] == "happy" and res[0]["events"] == ["laughter"]
    assert res[1]["truncated"] is False and res[1]["emotion"] == "sad" and res[1]["events"] == ["cry"]
    assert res[3]["truncated"] is False
    assert "error" in res[2] and "id" in res[2]


def test_sensevoice_batch_falls_back_to_op_events_when_generate_fails(tmp_path: Path) -> None:
    from adapters.funasr.sensevoice_impl import SenseVoice

    class Stub:
        def generate(self, input, **kw):
            if isinstance(input, list):
                raise RuntimeError("CUDA out of memory")
            return [{"text": "<|ja|><|NEUTRAL|><|woitn|>x"}]

    m = SenseVoice({"name": "sensevoice", "options": {}})
    m.model = Stub()
    res = m.run("events", [_clip(tmp_path, "a", 1.0), _clip(tmp_path, "b", 1.2)])
    assert [r["emotion"] for r in res] == ["neutral", "neutral"]


def test_qwen3_asr_batch_transcribe_passes_per_item_languages(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from adapters.hf.asr_impl import Qwen3ASR

    class Stub:
        def __init__(self) -> None:
            self.calls: list[tuple[list[str], Any]] = []

        def transcribe(self, audio, language=None, **kw):
            audio = audio if isinstance(audio, list) else [audio]
            if "boom" in "".join(audio):
                raise RuntimeError("one unreadable file")
            self.calls.append((audio, language))
            langs = language if isinstance(language, list) else [language] * len(audio)
            return [SimpleNamespace(text=f"  {Path(a).stem}  ", language=lg or "Auto") for a, lg in zip(audio, langs)]

    m = Qwen3ASR({"name": "qwen3_asr", "options": {"batch_size": 2}})
    m.model = Stub()
    c = [_clip(tmp_path, n, s) for n, s in (("c", 3.0), ("a", 1.0), ("b", 2.0))]
    c[1]["lang"] = None
    res = m.run("transcribe", c, lang=None)
    assert [(r["id"], r["text"]) for r in res] == [("c", "c"), ("a", "a"), ("b", "b")]
    assert [r["lang"] for r in res] == ["Japanese", "Auto", "Japanese"]
    # sorted by duration: (a, b) share one call with per-item languages, c is alone and goes through op_transcribe
    assert m.model.calls[0][1] == [None, "Japanese"] and [Path(p).stem for p in m.model.calls[0][0]] == ["a", "b"]
    assert m.model.calls[1][1] == "Japanese"
    m.run("transcribe", c, lang="ru")  # the call-level language wins over the item's
    assert m.model.calls[2][1] == ["Russian", "Russian"] and m.model.calls[3][1] == "Russian"
    # one unreadable path inside a sub-batch: the sub-batch is redone per item and only that item errors
    bad = [dict(c[0], id="x", wav=str(tmp_path / "boom.wav")), c[2]]
    res = m.run("transcribe", bad)
    assert "error" in res[0] and res[1]["text"] == "b"


def test_qwen3_aligner_batch_align_units_and_ms_guard(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from adapters.hf.align_impl import Qwen3Aligner

    class Stub:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def align(self, audio, text, language):
            self.calls.append({"n": len(audio), "text": text, "language": language, "sr": {sr for _, sr in audio}})
            out = []
            for (y, sr), t in zip(audio, text):
                dur = len(y) / sr
                scale = 1000.0 if t == "ms" else 1.0  # this "model" answers in ms for one of the clips
                out.append([SimpleNamespace(text=t, start_time=0.0, end_time=dur * scale)])
            return out

    m = Qwen3Aligner({"name": "qwen3_aligner", "options": {}})
    m.model = Stub()
    a, b, c = _clip(tmp_path, "a", 1.0), _clip(tmp_path, "b", 2.0), _clip(tmp_path, "c", 1.5)
    a["text"], b["text"], c["text"] = "ab", "ms", ""
    d = {"id": "gone", "wav": str(tmp_path / "missing.wav"), "text": "x", "lang": "ja"}
    res = m.run("align", [a, b, d, c])
    assert [r["id"] for r in res] == ["a", "b", "gone", "c"]
    assert len(m.model.calls) == 1 and m.model.calls[0]["n"] == 3 and m.model.calls[0]["sr"] == {16000}
    assert m.model.calls[0]["language"] == ["Japanese"] * 3
    assert [ch["c"] for ch in res[0]["chars"]] == ["a", "b"] and res[0]["chars"][-1]["t1"] == pytest.approx(1.0, abs=0.01)
    assert res[1]["chars"][-1]["t1"] == pytest.approx(2.0, abs=0.01)  # ms output rescaled by the guard
    assert "error" in res[2]
    assert res[3]["chars"] == [] and res[3]["backend"] == "qwen3_aligner"


def test_utmosv2_batch_mos_groups_files_by_directory(tmp_path: Path) -> None:
    from adapters.score.mos_impl import UTMOSv2

    class Stub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, list[str]]] = []

        def predict(self, *, input_path=None, input_dir=None, val_list=None, device="cpu", batch_size=16, verbose=True):
            if input_path is not None:
                if not Path(input_path).exists():
                    raise FileNotFoundError(f"File not found: {input_path}")
                return float(len(Path(input_path).stem))
            self.calls.append((Path(input_dir).name, sorted(val_list)))
            want = {v.replace(".wav", "") for v in val_list}
            names = [p.name for p in sorted(Path(input_dir).glob("*.wav")) if p.stem in want]
            return [{"file_path": (Path(input_dir) / n).as_posix(), "predicted_mos": float(len(n) - 4)} for n in names]

    m = UTMOSv2({"name": "utmosv2", "options": {"batch_size": 8}})
    m.model = Stub()
    items = [_clip(tmp_path / "d1", "aa", 1.0), _clip(tmp_path / "d2", "bbb", 1.0), _clip(tmp_path / "d1", "cccc", 1.0), _clip(tmp_path / "d2", "d", 1.0), _clip(tmp_path / "d3", "ee", 1.0)]
    (tmp_path / "d1" / "unrelated.wav").write_bytes(b"")  # same directory, not requested: must not be scored
    res = m.run("mos", items)
    assert [r["id"] for r in res] == ["aa", "bbb", "cccc", "d", "ee"]
    assert [r["mos"] for r in res] == [2.0, 3.0, 4.0, 1.0, 2.0]
    assert sorted(m.model.calls) == [("d1", ["aa.wav", "cccc.wav"]), ("d2", ["bbb.wav", "d.wav"])]  # d3 held one file: single-file path
    m.model.calls.clear()
    gone = dict(items[0], id="gone", wav=str(tmp_path / "d1" / "gone.wav"))
    res = m.run("mos", [gone, items[0], items[2]])
    assert "FileNotFoundError" in res[0]["error"] and res[1]["mos"] == 2.0 and res[2]["mos"] == 4.0
    assert m.model.calls == [("d1", ["aa.wav", "cccc.wav"])]

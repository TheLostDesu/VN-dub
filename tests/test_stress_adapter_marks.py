"""adapters/asr/stress_impl.py + vndub/text/ru.py: unmarked words, lines with quotes, and the ``safe`` yo mode.

Mocked ruaccent output unless ``VNDUB_RUACCENT_TESTS=1`` (then the real ruaccent runs in the backend environment
``.venvs/main`` on ``models/ruaccent``: short texts, 2 threads)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

from adapters.asr import stress_impl
from vndub.text import ru

ROOT = Path(__file__).resolve().parents[1]


class FakeModel:
    def predict_yo_homographs(self, sentence):
        return [{"entity": "YO"}]  # would make every homograph ё


def fake_impl(raw: str, yo_words: dict | None = None, yo_homographs: dict | None = None):
    """A RuAccentStress whose ruaccent answers ``raw`` and records which yo data it saw."""
    seen: dict = {}
    acc = types.SimpleNamespace(
        yo_words=yo_words or {},
        yo_homographs=yo_homographs or {},
        yo_homograph_model=FakeModel(),
        accents={"даже": "д+аже", "тебя": "теб+я", "хотя": "хот+я", "таким": "так+им", "напоминает": "напомин+ает"},
        omographs={"замок": ["з+амок", "зам+ок"]},
        accent_model=types.SimpleNamespace(put_accent=lambda w: "+" + w if w == "образом" else w),
    )

    def process_all(text):
        seen["words"] = dict(acc.yo_words)
        seen["predict"] = acc.yo_homograph_model.predict_yo_homographs("x")
        return raw

    acc.process_all = process_all
    impl = stress_impl.RuAccentStress.__new__(stress_impl.RuAccentStress)
    impl.acc = acc
    return impl, seen


@pytest.mark.parametrize(
    "raw,want",
    [
        ("Даже +если я всё ещё ненав+ижу, теб+я", "Д+аже +если я всё ещё ненав+ижу, теб+я"),
        ("+Эта п+есня... напоминает мне", "+Эта п+есня... напомин+ает мне"),
        ("Хотя я не совс+ем", "Хот+я я не совс+ем"),
        ("Так+им +образом", "Так+им +образом"),  # nothing to add
        ("Таким образом", "Так+им +образом"),  # the dictionary has "таким", the accent network "образом"
        ("в на ли он чт+о", "в на ли он чт+о"),  # one vowel: left as it was
        ("Хот+я всё ёлка", "Хот+я всё ёлка"),  # ё is a stress by itself
    ],
)
def test_multi_vowel_words_left_unmarked_get_a_stress(raw, want):
    impl, _ = fake_impl(raw)
    assert impl.op_stress({"text": "x"}, fmt="plus", yo=True)["text"] == want


def test_an_unknown_word_without_a_network_answer_stays_as_it_is():
    assert stress_impl.fill_unmarked("кракозябра", {}, lambda w: w) == "кракозябра"


def test_safe_yo_turns_the_homograph_guess_off_and_drops_ambiguous_dictionary_words():
    yo_words = {"темном": "тёмном", "небу": "нёбу", "совершенный": "совершённый", "найдешь": "найдёшь"}
    impl, seen = fake_impl("x", yo_words, {"небу": "нёбу", "узнаем": "узнаём"})
    impl.op_stress({"text": "x"}, fmt="plus", yo="safe")
    assert seen["words"] == {"темном": "тёмном", "найдешь": "найдёшь"}  # небу is a homograph, совершенный is in our extra list
    assert seen["predict"] == []
    # restored for the next call, and the other modes see the whole data and the network
    impl.op_stress({"text": "x"}, fmt="plus", yo=True)
    assert seen["words"] == yo_words
    assert seen["predict"] == [{"entity": "YO"}]


def test_yo_false_still_removes_the_yo():
    impl, _ = fake_impl("вс+ё ещ+ё")
    assert impl.op_stress({"text": "x"}, fmt="plain", yo=False)["text"] == "все еще"


# ---- the "a stresser must not change the words" guard threw whole lines away ---------------------------------------


def quotes_stresser(text: str) -> str:
    """ruaccent: no straight quotes, the marks as '+'."""
    return (
        "Я над+еялась, чт+о +это ш+утка, н+о всё равн+о тв+ёрдо сказ+ала: Верн+и этим д+етям наз+ад!"
        if '"' in text
        else text
    )


def test_a_line_with_straight_quotes_keeps_its_marks_and_its_quotes():
    src = 'Я надеялась, что это шутка, но всё равно твёрдо сказала: "Верни этим детям назад!"'
    tts, plain = ru.prepare_tts_text(src, quotes_stresser, fmt="plus")
    assert tts == 'Я над+еялась, чт+о +это ш+утка, н+о всё равн+о тв+ёрдо сказ+ала: "Верн+и этим д+етям наз+ад!"'
    assert plain == src
    acute, _ = ru.prepare_tts_text(src, quotes_stresser, fmt="plain")
    assert acute == src


def test_graft_stress_refuses_other_words():
    assert ru.graft_stress("Привет, мир", "+Пока м+ир") is None
    assert ru.graft_stress("Привет *мир*", "Прив+ет м+ир") == "Прив+ет *м+ир*"
    assert ru.graft_stress('Он - "друг"', "Он - др+уг") == 'Он - "др+уг"'


# ---- the real ruaccent --------------------------------------------------------------------------------------------

REAL = pytest.mark.skipif(
    os.environ.get("VNDUB_RUACCENT_TESTS") != "1",
    reason="set VNDUB_RUACCENT_TESTS=1 to run the real ruaccent (.venvs/main, models/ruaccent)",
)

_PROBE = r"""
import json, os, sys
os.environ["OMP_NUM_THREADS"] = "2"; os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, sys.argv[1])
import onnxruntime as ort
_S = ort.InferenceSession
class S(_S):
    def __init__(self, *a, sess_options=None, **k):
        so = sess_options or ort.SessionOptions(); so.intra_op_num_threads = 2; so.inter_op_num_threads = 1
        super().__init__(*a, sess_options=so, **k)
ort.InferenceSession = S
from pathlib import Path
from adapters.asr.stress_impl import RuAccentStress, _load
impl = RuAccentStress.__new__(RuAccentStress)
impl.acc = _load(Path(sys.argv[1]) / "models" / "ruaccent", "turbo3.1", True)
texts, yo = json.load(open(sys.argv[2], encoding="utf-8"))
out = [impl.op_stress({"text": t}, fmt="plus", yo=yo)["text"] for t in texts]
sys.stdout.buffer.write(json.dumps(out, ensure_ascii=False).encode("utf-8"))
"""


def _real(tmp_path: Path, texts: list[str], yo) -> list[str]:
    py = ROOT / ".venvs" / "main" / "Scripts" / "python.exe"
    if not py.is_file():
        py = ROOT / ".venvs" / "main" / "bin" / "python"
    script, data = tmp_path / "probe.py", tmp_path / "in.json"
    script.write_bytes(_PROBE.encode("utf-8"))
    data.write_bytes(json.dumps([texts, yo], ensure_ascii=False).encode("utf-8"))
    r = subprocess.run([str(py), "-I", str(script), str(ROOT), str(data)], capture_output=True, timeout=600, cwd=tmp_path)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")[-800:]
    return json.loads(r.stdout.decode("utf-8"))


@REAL
def test_real_ruaccent_marks_every_multi_vowel_word(tmp_path):
    texts = [
        "Эта песня... напоминает мне особенный момент.",
        "Даже если я всё ещё ненавижу людей, я всё равно буду любить тебя, так что не волнуйся!",
        "Хотя я не совсем понимаю",
        "Таким образом",
    ]
    for got in _real(tmp_path, texts, "safe"):
        for tok in re.findall(r"[+а-яёА-ЯЁ]+", got):
            if sum(c in "аеёиоуыэюя" for c in tok.lower()) > 1 and "ё" not in tok.lower():
                assert "+" in tok, f"{tok!r} unmarked in {got!r}"


@REAL
def test_real_ruaccent_safe_yo_keeps_homographs_and_adds_the_sure_yo(tmp_path):
    texts = ["Мы можем гулять по небу", "когда мы лучше узнаем друг друга", "это самое весёлое", "в темном лесу насчет прически, найдешь еще"]
    safe = _real(tmp_path, texts, "safe")
    assert "нёбу" not in ru.strip_stress(safe[0]) and "узнаём" not in ru.strip_stress(safe[1]) and "самоё" not in ru.strip_stress(safe[2])
    assert safe[2].endswith("весёлое")  # the LLM's own ё is kept
    last = ru.strip_stress(safe[3])
    assert "тёмном" in last and "насчёт" in last and "найдёшь" in last and "причёск" in last and "ещё" in last
    assert "нёбу" in ru.strip_stress(_real(tmp_path, texts[:1], True)[0])  # the old mode still guesses (the reason for safe)

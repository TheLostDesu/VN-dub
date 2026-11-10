"""Russian stress on its way to a backend: the '+' variant, as is (F5-TTS_RUSSIAN) or as U+0301 marks (Chatterbox).

ruaccent stresses every Russian line at translate; only a backend whose ``stress_format`` asks for it is sent the marks. A project
translated while no such backend was enabled has no ``text_tts_plus``: synth makes it for the run from translate's stress cache.
"""

from __future__ import annotations

import types

import pytest

from tests.genproj import build
from tests.p3_helpers import watch_models
from tests.test_synth import _say, _sides, _with_caps
from vndub.backends import registry
from vndub.manifest.io import write_jsonl
from vndub.stages import genutil as gu
from vndub.stages import synth
from vndub.stages._opcache import OpCache
from vndub.text import laughter as lf
from vndub.text import ru


@pytest.fixture
def proj(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path / "game")
    build(ctx)
    return ctx


def _say_stressed(ctx, line_id: str, plus: str) -> None:
    """A line translated with its '+' variant, as translate writes it when a backend needs it."""
    trs = gu.load_translations(ctx)
    r = trs[line_id]
    r.text_tgt = r.text_tgt_short = r.text_tts = r.text_plain = gu.strip_stress(plus)
    r.text_tts_plus = r.text_tts_short_plus = plus
    write_jsonl(ctx.paths.translation(ctx.cfg.project.tgt_lang), list(trs.values()))


def _identity_stresser(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """No model for the lines the test does not look at; returns the texts it was asked to stress."""
    asked: list[list[str]] = []

    def make_stresser(ctx, texts):
        asked.append(list(texts))
        return (lambda t: t), "mock", ""

    monkeypatch.setattr("vndub.stages.translate.make_stresser", make_stresser)
    return asked


# ----------------------------------------------------------------------------
# laughs
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("Х+а-х+а-х+а! Н+у т+ы даёшь.", "Ха-ха-ха! Н+у т+ы даёшь."),
        ("Му+а-х+а-х+а-х+а!", "Муа-ха-ха-ха!"),
        ("Х+е-хе, всё в пор+ядке.", "Хе-хе, всё в пор+ядке."),
        ("(сме+ётся) +это", "(смеётся) +это"),  # a stage direction is a laugh too
        ("Ха-ха, +это", "Ха-ха, +это"),  # the first mark after a laugh is not in it
        ("В+ыход зам+ок", "В+ыход зам+ок"),  # no laugh
        ("Ха-ха-ха", "Ха-ха-ха"),  # no marks
        ("2+2", "2+2"),
        ("", ""),
    ],
)
def test_unstress_laughs(text: str, want: str) -> None:
    assert lf.unstress_laughs(text) == want


def test_a_stressed_laugh_is_found_once_its_marks_are_gone() -> None:
    assert not lf.find_laughs("Х+а-х+а-х+а!")  # why unstress_laughs exists
    assert [l.units for l in lf.find_laughs(lf.unstress_laughs("Х+а-х+а-х+а! Н+у"))] == [3]


@pytest.mark.parametrize(("fmt", "tail"), [("acute", "ну ты даёшь, де́вочка!"), ("plus", "ну ты даёшь, д+евочка!")])
def test_a_laugh_with_stress_marks_is_still_a_laugh(proj, monkeypatch, fmt, tail):
    ctx = proj
    _identity_stresser(monkeypatch)
    line = ctx.voiced_lines()[0]
    _say_stressed(ctx, line.line_id, "Х+а-х+а-х+а, ну ты даёшь, д+евочка!")
    _with_caps(monkeypatch, stress_format=fmt)
    s = synth.run(ctx)
    assert s["failed"] == 0 and s["laughter"]["modes"]["respell"] >= 1
    for side in _sides(ctx, line.line_id):
        assert side["text_used"] == "Ахахаха, " + tail  # the table's spelling; the rest of the line keeps its marks
        assert side["laughter"]["kind"] == "hearty"


# ----------------------------------------------------------------------------
# what a backend is sent
# ----------------------------------------------------------------------------


def test_chatterbox_asks_for_acute_marks() -> None:
    assert registry.get("chatterbox_ml").caps["stress_format"] == "acute"


def test_the_stress_format_decides_what_the_backend_is_sent(proj, monkeypatch):
    ctx = proj
    asked = _identity_stresser(monkeypatch)
    line = ctx.voiced_lines()[0]
    _say_stressed(ctx, line.line_id, "Он+а в+идит зам+ок.")
    synth.run(ctx)  # the mock TTS as it is: stress_format plain
    assert {s["text_used"] for s in _sides(ctx, line.line_id)} == {"Она видит замок."}
    assert not asked  # and a plain backend needs no stresser

    _with_caps(monkeypatch, stress_format="plus")
    synth.run(ctx)
    assert {s["text_used"] for s in _sides(ctx, line.line_id)} == {"Он+а в+идит зам+ок."}

    _with_caps(monkeypatch, stress_format="acute")
    synth.run(ctx)
    for side in _sides(ctx, line.line_id):
        assert side["text_used"] == "Она́ ви́дит замо́к."
        assert side["text_plain"] == "Она видит замок."  # the CER reference never has marks


def test_the_option_overrides_the_caps(make_ctx, tmp_path, monkeypatch):
    spec = types.SimpleNamespace(caps={"stress_format": "acute"})
    assert synth._stress_format(spec, {}) == "acute"
    assert synth._stress_format(spec, {"stress_format": "plain"}) == "plain"
    assert synth._stress_format(types.SimpleNamespace(caps=None), {}) == "plain"

    ctx = make_ctx(tmp_path / "game", sets={"synth.backends.mock_tts.stress_format": "plain"})
    build(ctx)
    asked = _identity_stresser(monkeypatch)
    line = ctx.voiced_lines()[0]
    _say_stressed(ctx, line.line_id, "Он+а в+идит зам+ок.")
    _with_caps(monkeypatch, stress_format="acute")
    synth.run(ctx)
    assert {s["text_used"] for s in _sides(ctx, line.line_id)} == {"Она видит замок."} and not asked


def test_a_user_text_keeps_its_own_marks_in_the_backend_format(proj, monkeypatch):
    """Round 18: a Review text is prepared like a translation. Marks the user typed are kept (no stresser), in the form the
    backend takes: the acute format drops the mark of a one-vowel word."""
    from vndub.review.app import write_override

    ctx = proj
    _identity_stresser(monkeypatch)
    _with_caps(monkeypatch, stress_format="acute")
    synth.run(ctx)
    line = ctx.voiced_lines()[0]
    write_override(ctx.paths, line.line_id, "set_text", {"text": "Д+а, зам+ок."})
    synth.run(ctx)
    over = [s for s in _sides(ctx, line.line_id) if s["text_variant"] == "override"]
    assert over and all(s["text_used"] == "Да, замо́к." for s in over)


def test_another_target_language_gets_no_marks(make_ctx, tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path / "game", sets={"project.tgt_lang": "en"})
    build(ctx)
    asked = _identity_stresser(monkeypatch)
    line = ctx.voiced_lines()[0]
    _say(ctx, line.line_id, "C+at sees C+o.")  # a '+' before a vowel is no stress mark here
    _with_caps(monkeypatch, stress_format="acute")
    assert synth.run(ctx)["failed"] == 0
    assert {s["text_used"] for s in _sides(ctx, line.line_id)} == {"C+at sees C+o."} and not asked


# ----------------------------------------------------------------------------
# a project translated without the variant
# ----------------------------------------------------------------------------


def test_the_variant_comes_from_translate_stress_cache_and_nothing_loads(proj, monkeypatch):
    """translate stressed the text anyway, for its plain variant; the team's projects were translated before Chatterbox asked."""
    ctx = proj
    line = ctx.voiced_lines()[0]
    _say(ctx, line.line_id, "Она видит замок.")
    before = ctx.paths.translation("ru").read_bytes()
    st = ctx.cfg.translate.stress
    spec, _ = ctx.models.resolve(st.tool)
    store = OpCache(ctx, f"translate_stress_{spec.name}")
    for r in gu.load_translations(ctx).values():
        for t in {ru.normalize_for_tts(x) for x in (r.text_tgt, r.text_tgt_short) if x}:
            store.put(t, store.key(spec.name, st.model, st.yo, t), "Он+а в+идит зам+ок." if t == "Она видит замок." else t)
    store.save()

    _with_caps(monkeypatch, stress_format="acute")
    counts = watch_models(ctx)
    assert synth.run(ctx)["failed"] == 0
    counts.restore()
    assert {s["text_used"] for s in _sides(ctx, line.line_id)} == {"Она́ ви́дит замо́к."}
    assert st.tool not in counts.acquired and spec.name not in counts.acquired
    assert ctx.paths.translation("ru").read_bytes() == before  # the translation file stays translate's


def test_only_the_lines_of_this_run_are_stressed(make_ctx, tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path / "game")
    build(ctx)
    first, second = ctx.voiced_lines()[:2]
    _say(ctx, first.line_id, "Первая реплика.")
    _say(ctx, second.line_id, "Вторая реплика.")
    asked = _identity_stresser(monkeypatch)
    _with_caps(monkeypatch, stress_format="acute")
    synth._backfill_stress(ctx, gu.load_translations(ctx), [first])
    assert asked == [["Первая реплика.", "Первая реплика."]]


def test_a_record_that_has_the_variant_is_left_alone(proj, monkeypatch):
    ctx = proj
    asked = _identity_stresser(monkeypatch)
    line = ctx.voiced_lines()[0]
    _say_stressed(ctx, line.line_id, "Он+а.")
    synth._backfill_stress(ctx, gu.load_translations(ctx), [line])
    assert asked == []


def test_no_stresser_means_plain_text_and_a_warning(proj, monkeypatch, caplog):
    ctx = proj
    line = ctx.voiced_lines()[0]
    _say(ctx, line.line_id, "Она видит замок.")
    monkeypatch.setattr("vndub.stages.translate.make_stresser", lambda c, texts: (None, "", "ruaccent is not available"))
    _with_caps(monkeypatch, stress_format="acute")
    with caplog.at_level("WARNING"):
        assert synth.run(ctx)["failed"] == 0
    assert {s["text_used"] for s in _sides(ctx, line.line_id)} == {"Она видит замок."}
    assert "without stress marks: ruaccent is not available" in caplog.text


# ----------------------------------------------------------------------------
# translate
# ----------------------------------------------------------------------------


def test_a_backend_that_wants_acute_marks_gets_the_plus_variant_at_translate(make_ctx, tmp_path):
    """synth makes the U+0301 marks from the '+' text, so a translation written now carries it."""
    from vndub.stages import translate

    assert translate.needs_plus(make_ctx(tmp_path / "plain")) is False
    assert translate.needs_plus(make_ctx(tmp_path / "chatterbox", sets={"synth.backends.chatterbox_ml.enabled": True})) is True

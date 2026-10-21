"""Voiced lines the game shows without a speaker name reach casting (all-mock ``apptest`` profile).

``renpy_mini`` gets an epilogue with four voiced lines that show no name: an off-screen line of Sakura in
the narrator box (``sak_`` file), a ``？？？`` line of Haruto (``har_`` file) and two voiced narrator lines
whose ``nar_`` files name nobody. Parse gives the first two to the character their voice-file prefix
names and the narrator lines to the ``voice_over`` speaker; through the app, ``voice_over`` is a
character in the summary and in casting, gets designs, is cast by automatic casting (route B: it has
original audio), and its lines are translated and dubbed. Unvoiced narration stays narration.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Iterator

import pytest
from app_helpers import FIXTURE, reset_app_state, run_job, setup_env, teardown_app_state, wait_job

from vndub.engines.base import VOICE_OVER, hidden_name, resolve_hidden_speakers, voice_name_parts
from vndub.manifest.io import LineRecord, TranslationRecord, load_characters, read_jsonl, save_characters
from vndub_app import appdirs, session

EPILOGUE = '''
    voice "voice/sak_0008.ogg"
    "待って、忘れ物だよ！"
    voice "voice/har_0008.ogg"
    "？？？" "おーい、待ってくれ！"
    voice "voice/nar_0001.ogg"
    "その日、三人の物語が始まった。"
    voice "voice/nar_0002.ogg"
    "けれど、それはまた別の話。"
    "夕焼けが静かに沈んでいった。"
'''
COPIES = {"sak_0008.ogg": "sak_0001.ogg", "har_0008.ogg": "har_0001.ogg", "nar_0001.ogg": "yui_0001.ogg", "nar_0002.ogg": "yui_0002.ogg"}


# ----------------------------------------------------------------------------
# unit: resolve_hidden_speakers
# ----------------------------------------------------------------------------


def _l(i: int, sid: str, voice: str | None, raw: str | None = None, flags: list[str] | None = None) -> LineRecord:
    return LineRecord(
        line_id=f"s_{i:03d}", scene="s", order=i, speaker_raw=sid if raw is None and sid != "narrator" else (raw or ""), speaker_id=sid,
        text_src="テキスト", voice_file=voice, has_voice=voice is not None, audio_id=voice, flags=list(flags or []),
    )


def _named(n: int = 3) -> list[LineRecord]:
    out = []
    for k in range(1, n + 1):
        out += [_l(2 * k - 1, "sakura", f"voice/sak_{k:04d}.ogg"), _l(2 * k, "haruto", f"voice/har_{k:04d}.ogg")]
    return out


def test_hidden_name_and_voice_parts() -> None:
    assert hidden_name("？？？") and hidden_name("???") and hidden_name("「…」") and hidden_name("――")
    assert not hidden_name("") and not hidden_name("さくら") and not hidden_name("Sakura?") and not hidden_name("Ｂ")
    assert voice_name_parts("voice/SAK_0012.ogg") == ("sak", 12) and voice_name_parts("voice/start_636ae3f5.ogg") is None


def test_resolve_by_prefix_and_voice_over() -> None:
    lines = _named() + [
        _l(20, "narrator", "voice/sak_0099.ogg"),               # off-screen Sakura
        _l(21, "narrator", "voice/nar_0001.ogg"),               # voiced narrator: nobody's prefix
        _l(22, "u_abcdef", "voice/har_0099.ogg", raw="？？？"),  # hidden name, Haruto's file
        _l(23, "u_123456", "voice/mob_0001.ogg", raw="？？？"),  # hidden name, unknown prefix: keeps its speaker
        _l(24, "narrator", None),                               # unvoiced narration: untouched
        _l(25, "narrator", None, flags=["menu"]),
    ]
    out = resolve_hidden_speakers(lines)
    by = {l.line_id: l for l in lines}
    assert out == {"from_voice": 2, "voice_over": 1}
    assert by["s_020"].speaker_id == "sakura" and "speaker_from_voice" in by["s_020"].flags and by["s_020"].speaker_raw == ""
    assert by["s_021"].speaker_id == VOICE_OVER and "offscreen_voice" in by["s_021"].flags
    assert by["s_022"].speaker_id == "haruto" and by["s_023"].speaker_id == "u_123456"
    assert by["s_024"].speaker_id == "narrator" and by["s_025"].speaker_id == "narrator" and not by["s_024"].flags


def test_prefixes_that_name_nobody_send_narrator_lines_to_voice_over() -> None:
    # one prefix for everybody (ch01_0001 style): the prefix says nothing about the speaker
    lines = [_l(i, "sakura" if i % 2 else "haruto", f"voice/ch_{i:04d}.ogg") for i in range(1, 9)]
    lines.append(_l(30, "narrator", "voice/ch_0030.ogg"))
    assert resolve_hidden_speakers(lines) == {"from_voice": 0, "voice_over": 1}
    assert lines[-1].speaker_id == VOICE_OVER


def test_parse_option_off_keeps_the_narrator(make_ctx, tmp_path: Path) -> None:
    from vndub.stages.common import run_stage

    game = _game(tmp_path / "game")
    ctx = make_ctx(game, sets={"parse.hidden_speakers": "off"})
    for st in ("detect", "extract", "parse"):
        run_stage(st, ctx)
    lines = {l.text_src: l for l in read_jsonl(ctx.paths.lines, LineRecord)}
    assert lines["待って、忘れ物だよ！"].speaker_id == "narrator" and VOICE_OVER not in load_characters(ctx.paths.characters)


def test_parse_resolves_hidden_speakers(make_ctx, tmp_path: Path) -> None:
    from vndub.stages.common import run_stage

    game = _game(tmp_path / "game")
    ctx = make_ctx(game)
    for st in ("detect", "extract", "parse"):
        summary = run_stage(st, ctx)
    assert summary["hidden_speakers"] == {"from_voice": 2, "voice_over": 2}
    lines = {l.text_src: l for l in read_jsonl(ctx.paths.lines, LineRecord)}
    assert lines["待って、忘れ物だよ！"].speaker_id == "sakura"
    assert lines["おーい、待ってくれ！"].speaker_id == "haruto"
    assert lines["その日、三人の物語が始まった。"].speaker_id == VOICE_OVER and lines["けれど、それはまた別の話。"].speaker_id == VOICE_OVER
    assert lines["夕焼けが静かに沈んでいった。"].speaker_id == "narrator" and not lines["夕焼けが静かに沈んでいった。"].has_voice
    chars = load_characters(ctx.paths.characters)
    assert chars[VOICE_OVER].names == {"src": "", "tgt": "Голос за кадром"} and chars[VOICE_OVER].stats.n_lines == 2
    assert chars["sakura"].stats.n_lines == 8 and chars["haruto"].stats.n_lines == 8
    assert "narrator" not in chars and not any(sid.startswith("u_") for sid in chars), "the ？？？ speaker lost its only line"
    # a second parse keeps the user's name for the voice-over
    chars[VOICE_OVER].names["tgt"] = "Рассказчик"
    save_characters(ctx.paths.characters, chars)
    ctx.force = True
    run_stage("parse", ctx)
    assert load_characters(ctx.paths.characters)[VOICE_OVER].names["tgt"] == "Рассказчик"


# ----------------------------------------------------------------------------
# the app: summary, casting, translation, dub
# ----------------------------------------------------------------------------


def _game(dst: Path) -> Path:
    shutil.copytree(FIXTURE, dst)
    script = dst / "game" / "script.rpy"
    script.write_text(script.read_text(encoding="utf-8").rstrip("\n") + "\n" + EPILOGUE, encoding="utf-8")
    vdir = dst / "game" / "voice"
    for new, src in COPIES.items():
        shutil.copyfile(vdir / src, vdir / new)
    return dst


@pytest.fixture(scope="module")
def app(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    from fastapi.testclient import TestClient

    from vndub_app.server import create_app

    root = tmp_path_factory.mktemp("offscreen")
    mp = pytest.MonkeyPatch()
    setup_env(mp, root)
    game = _game(root / "games" / "renpy_offscreen")
    reset_app_state()
    try:
        with TestClient(create_app(), raise_server_exceptions=False) as c:
            yield {"client": c, "root": root, "game": game}
    finally:
        teardown_app_state()
        mp.undo()


def _job(c: Any, path: str, body: Any = None) -> dict[str, Any]:
    j = run_job(c, c.post(path, json=body if body is not None else {}), 400.0)
    assert j["state"] == "done", j.get("error")
    return j


def test_app_voice_over_is_cast_and_dubbed(app: dict[str, Any]) -> None:
    c = app["client"]
    slug = c.post("/api/project/open", json={"game_dir": str(app["game"])}).json()["slug"]
    _job(c, "/api/project/import")
    work = app["root"] / "data" / "projects" / slug / "work"

    summary = c.get("/api/project/summary").json()
    chars = {x["speaker_id"]: x for x in summary["characters"]}
    assert VOICE_OVER in chars and "narrator" not in chars
    assert chars[VOICE_OVER]["name_tgt"] == "Голос за кадром" and chars[VOICE_OVER]["voiced_lines"] == 2 and chars[VOICE_OVER]["mode"] == "voiced"
    assert summary["voiced"] == 24 and chars["sakura"]["voiced_lines"] == 8 and chars["haruto"]["voiced_lines"] == 8

    _job(c, "/api/project/translate")
    translated = {r.line_id for r in read_jsonl(work / "translation" / "lines.ru.jsonl", TranslationRecord)}
    lines = list(read_jsonl(work / "manifest" / "lines.jsonl", LineRecord))
    vo_ids = [l.line_id for l in lines if l.speaker_id == VOICE_OVER]
    moved = [l.line_id for l in lines if "speaker_from_voice" in l.flags]
    assert len(vo_ids) == 2 and len(moved) == 2 and set(vo_ids + moved) <= translated

    cast = {x["speaker_id"]: x for x in c.get("/api/characters").json()}
    assert VOICE_OVER in cast and cast[VOICE_OVER]["name_tgt"] == "Голос за кадром" and cast[VOICE_OVER]["voiced_lines"] == 2
    clips = cast[VOICE_OVER]["clips"]
    assert [k["kind"] for k in clips] == ["neutral", "emotional"], "two voiced lines: two original clips, none twice"
    assert {k["line_id"] for k in clips} == set(vo_ids)
    glossary = c.get("/api/project/glossary").json()
    assert VOICE_OVER not in {x["speaker_id"] for x in glossary["characters"]}, "nothing to translate in a name that is not shown"

    res = _job(c, "/api/casting/auto")["result"]
    assert VOICE_OVER in res["chosen"], res
    cast = {x["speaker_id"]: x for x in c.get("/api/characters").json()}
    assert cast[VOICE_OVER]["chosen"]["route"] in ("A", "B")
    for j in c.get("/api/jobs", params={"project": slug, "active": 1}).json():
        wait_job(c, j["id"])

    ctx = session.read_ctx(appdirs.project_handle(slug))
    dub = {l.line_id for l in ctx.dub_lines(apply_only=False)}
    assert set(vo_ids + moved) <= dub, "the voice-over and the off-screen lines are dubbed"
    narration = [l.line_id for l in lines if l.speaker_id == "narrator" and not l.has_voice]
    assert narration and not (set(narration) & dub)

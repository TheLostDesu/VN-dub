"""Parsers of the secondary engines (unit level, no fixtures needed)."""

from __future__ import annotations

import json
from pathlib import Path

from vndub.engines.artemis import parse_ast, parse_lua
from vndub.engines.bgi import parse_bgi_json
from vndub.engines.catsystem2 import parse_cst_text
from vndub.engines.generic_prefix import group_by_prefix, infer_prefix_map
from vndub.engines.nscripter import parse_nscript, split_commands
from vndub.engines.siglus import koe_name, read_koe_master
from vndub.engines.tyrano import parse_tyrano
from vndub.engines.unity import parse_nani
from vndub.manifest.io import AudioIndexRecord


def test_nscript_parser() -> None:
    text = '*start\ndwave 0,"voice\\a01.wav":wait 100\n【澪】「やあ」@\n地の文。\\\nmov %0,1 ;comment\ndwave 1,"voice\\t01.wav"\n拓海「どうも」!w200\\\n*next\n`English line\\\n'
    rows = parse_nscript(text, r"^【(?P<name>[^】]+)】")
    assert [(r["label"], r["speaker"], r["text"], r["voice"]) for r in rows] == [
        ("start", "澪", "「やあ」", "voice/a01.wav"),
        ("start", "", "地の文。", None),
        ("start", "拓海", "「どうも」", "voice/t01.wav"),
        ("next", "", "English line", None),
    ]
    assert split_commands('dwave 0,"a:b.wav":wait 1') == ['dwave 0,"a:b.wav"', "wait 1"]


def test_siglus_csv(tmp_path: Path) -> None:
    p = tmp_path / "koe_master.csv"
    p.write_text("koe_no,character,text,duration,callsite\n100001,渚,あの……,1.2,s01:12\n\n200002,朋也,\"よう, 元気か\",0.8,s02:3\n", encoding="utf-8")
    rows = read_koe_master(p)
    assert [r["koe_no"] for r in rows] == ["100001", "200002"]
    assert rows[1]["text"] == "よう, 元気か"
    assert koe_name(100001) == "KOE(000100001)"


def test_bgi_json_shapes() -> None:
    data = [{"name": "ユキ", "message": "(yuki_000003)こんにちは"}, {"text": "地の文"}, [{"speaker": "ソウタ", "line": "よう", "voice_id": "sota_000002"}]]
    assert parse_bgi_json(data) == [
        {"speaker": "ユキ", "text": "こんにちは", "voice": "yuki_000003"},
        {"speaker": "", "text": "地の文", "voice": ""},
        {"speaker": "ソウタ", "text": "よう", "voice": "sota_000002"},
    ]
    assert parse_bgi_json({"entries": data[:1]})[0]["voice"] == "yuki_000003"


def test_catsystem2_cst_text_and_json() -> None:
    text = "0x21 さくら\n0x30 pcm sak_0001\n0x20 「こんにちは」\\n\n0x20 地の文\n"
    assert parse_cst_text(text) == [{"speaker": "さくら", "text": "「こんにちは」", "voice": "sak_0001"}, {"speaker": "", "text": "地の文", "voice": ""}]
    js = json.dumps([{"type": 33, "content": "ハル"}, {"type": "0x30", "content": "pcm har_01"}, {"type": 32, "content": "やあ"}])
    assert parse_cst_text(js) == [{"speaker": "ハル", "text": "やあ", "voice": "har_01"}]


AST = '''
astver = 2.0
ast = {
  block_00000 = {
    {"savetitle", text="序章"},
    text = {
      vo = { {"vo", file="sak_0001", ch="sakura"} },
      ja = { { name = {"さくら"}, "こんにちは", {"rt2"}, "、元気？", }, },
    },
    linknext = "block_00001",
  },
  block_00001 = {
    text = { ja = { { "地の文です。" } } },
  },
  ["block_00002"] = { {"bg", file="bg01"}, },
}
'''


def test_artemis_lua_ast() -> None:
    data = parse_lua(AST)
    assert data["astver"] == 2.0 and "block_00000" in data["ast"]
    rows = parse_ast(AST)
    assert [(r["name"], r["text"], r["voices"]) for r in rows] == [("さくら", "こんにちは、元気？", ["sak_0001"]), ("", "地の文です。", [])]


def test_tyrano_voconfig() -> None:
    text = (
        '[voconfig sebuf=2 name="akane" vostorage="akane_{number}.ogg" number=5]\n[vostart]\n'
        "#akane\nこんにちは[p]\n#akane:smile\nまたね[l]\n#\n地の文[p]\n[vostop]\n#akane\n無音[p]\n"
        '[playse storage="voice/extra_01.ogg" buf=2]\n#akane\n明示[p]\n'
    )
    rows = parse_tyrano(text, "s", "s.ks", {"jnames": {"akane": "あかね"}})
    assert [(r["speaker"], r["text"], r["voice"]) for r in rows] == [
        ("あかね", "こんにちは", "akane_5.ogg"),
        ("あかね", "またね", "akane_6.ogg"),
        ("", "地の文", None),
        ("あかね", "無音", None),
        ("あかね", "明示", "voice/extra_01.ogg"),
    ]


def test_naninovel() -> None:
    text = "# Start\n@voice Intro/Day/25\nKohaku: Hello there!\nJust narration.\nKohaku: Auto voiced line.|#2670eb4|\n@back Room\n"
    rows = parse_nani(text, "Script01")
    assert [(r["author"], r["text"], r["voice"]) for r in rows] == [
        ("Kohaku", "Hello there!", "Intro/Day/25"),
        ("", "Just narration.", None),
        ("Kohaku", "Auto voiced line.", "Naninovel/Voice/Script01/2670eb4"),
    ]


def test_prefix_grouping_and_inference() -> None:
    audio = [AudioIndexRecord(id=n, relpath=f"voice/{n}.ogg", src_path="") for n in ["nen_0002", "nen_0001", "har_0001", "nen_0010", "bgm01x"]]
    groups = group_by_prefix(audio, r"^(?P<prefix>[A-Za-z]+?)_?(?P<index>\d+)$")
    assert [a.id for a in groups["nen"]] == ["nen_0001", "nen_0002", "nen_0010"]
    assert set(groups) == {"nen", "har"}  # "bgm01x" does not match <prefix>_<index>
    mapping, notes = infer_prefix_map({"u_aaaa": 3, "u_bbbb": 1}, {"nen": groups["nen"], "har": groups["har"]}, {}, {})
    assert mapping == {"u_aaaa": "nen", "u_bbbb": "har"} and notes
    mapping, _ = infer_prefix_map({"x": 3}, groups, {"x": "HAR"}, {})
    assert mapping == {"x": "har"}


def test_vntextpatch_import(tmp_path: Path, make_ctx) -> None:
    game = tmp_path / "g"
    (game / "voice").mkdir(parents=True)
    for n in ["nen_0001", "nen_0002", "har_0001"]:
        (game / "voice" / f"{n}.ogg").write_bytes(b"OggS")
    jd = tmp_path / "json"
    jd.mkdir()
    (jd / "s01.json").write_text(json.dumps([{"message": "地の文"}, {"name": "寧々", "message": "一"}, {"name": "ハルキ", "message": "二"}, {"name": "寧々", "message": "三"}], ensure_ascii=False), encoding="utf-8")
    from engine_helpers import detect_and_parse

    ctx = make_ctx(game, extra={"import_vntextpatch": str(jd)}, sets={"parse.engine_opts.generic.prefix_map": {}})
    from vndub.engines.generic_prefix import import_vntextpatch

    _, _, audio, _ = detect_and_parse(ctx, engine="generic")
    res = import_vntextpatch(ctx, jd, audio)
    got = [(l.speaker_raw, l.text_src, l.voice_file, "prefix_guess" in l.flags) for l in res.lines]
    assert got == [("", "地の文", None, False), ("寧々", "一", "voice/nen_0001.ogg", True), ("ハルキ", "二", "voice/har_0001.ogg", True), ("寧々", "三", "voice/nen_0002.ogg", True)]


LAYOUTS = {
    "siglus": ["Scene.pck", "Gameexe.dat"],
    "catsystem2": ["scene.int", "pcm_a.int", "op.cst"],
    "artemis": ["root.pfs"],
    "yuris": ["pac/bn.ypf"],
    "majiro": ["scenario.arc", "voice.arc", "start.mjo"],
    "bgi": ["data01000.arc", "BGI.exe"],
    "unity": ["Game_Data/globalgamemanagers"],
    "tyrano": ["data/scenario/first.ks", "tyrano/libs.js"],
    "nscripter": ["nscript.dat", "arc.nsa"],
}


import pytest  # noqa: E402


@pytest.mark.parametrize("engine,files", sorted(LAYOUTS.items()))
def test_detect_layouts(tmp_path: Path, make_ctx, engine: str, files: list[str]) -> None:
    from vndub.stages import detect

    game = tmp_path / "g"
    for f in files:
        p = game / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"KIF\x00" if f.endswith(".int") else b"")
    s = detect.run(make_ctx(game))
    assert s["engine"] == engine, s


def test_detect_encrypted_xp3(tmp_path: Path, make_ctx) -> None:
    import struct

    from vndub.engines.xp3 import write_xp3
    from vndub.stages import detect

    game = tmp_path / "g"
    game.mkdir()
    p = game / "data.xp3"
    write_xp3(p, [("a.ogg", b"OggS" * 10), ("b.ks", "テキスト".encode())])
    raw = bytearray(p.read_bytes())
    index_ofs = struct.unpack_from("<Q", raw, 0x17 + 9)[0]
    for i in range(index_ofs + 17, len(raw)):
        raw[i] ^= 0x33
    p.write_bytes(bytes(raw))
    ctx = make_ctx(game)
    s = detect.run(ctx)
    assert s["engine"] == "kirikiri" and s["encrypted"] is True
    from vndub.engines import get_engine

    ex = get_engine("kirikiri").extract(ctx)
    assert any("encrypted" in w or "cannot read" in w for w in ex.warnings)

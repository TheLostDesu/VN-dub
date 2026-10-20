from __future__ import annotations

from pathlib import Path

from vndub.engines.kirikiri import KagParser, learn_macros, next_patch_name, parse_scn_json, parse_tag


def test_parse_tag_attrs() -> None:
    name, attrs = parse_tag('nm t="寧々" s=nen_0001 flag')
    assert name == "nm" and attrs == {"t": "寧々", "s": "nen_0001", "flag": "true"}
    assert parse_tag("playse storage='a b.ogg' buf=1") == ("playse", {"storage": "a b.ogg", "buf": "1"})


def test_macro_learning() -> None:
    macros = learn_macros(
        {
            "macro.ks": (
                "[macro name=nm]\n[ch text=\"【%t】\"][r]\n[playse storage=%s buf=1]\n[endmacro]\n"
                "[macro name=vo2]\n[voice storage=&mp.v]\n[emb exp=\"mp.who\"]\n[endmacro]\n"
                "[macro name=sakura]\n【さくら】\n[playse storage=%s|sak_0000]\n[endmacro]\n"
            )
        },
        ["cn"],
    )
    assert (macros["nm"].name_attr, macros["nm"].voice_attr) == ("t", "s")
    assert (macros["vo2"].name_attr, macros["vo2"].voice_attr) == ("who", "v")
    assert macros["sakura"].name_literal == "さくら" and macros["sakura"].voice_default == "sak_0000"
    assert "cn" in macros  # configured names are always recognised


def test_stream_parse_conventions() -> None:
    macros = learn_macros({"m.ks": "[macro name=nm][ch text=%t][playse storage=%s][endmacro]"})
    is_voice = lambda ref: not ref.startswith("se_")  # noqa: E731
    p = KagParser(macros, is_voice=is_voice)
    text = (
        "*start|title\n"
        "ナレーション。[p]\n"
        '[nm t="寧々" s=nen_0001]「一行目[r]二行目」[p]\n'
        "【ハルキ】[playse storage=har_0001]「やあ」[l][r]\n"
        "[playse storage=se_door]扉が開いた。[p]\n"
        "#ハルキ\n"
        "「名前行」[p]\n"
        "[ns]寧々[nse]「名前モード」\n"
        "\n"
        "@nm t=寧々 s=nen_0002\n"
        "「アット行」[p]\n"
    )
    rows = list(p.parse(text, "s01", "s01.ks"))
    got = [(r.speaker, r.text, r.voice) for r in rows]
    assert got == [
        ("", "ナレーション。", None),
        ("寧々", "「一行目二行目」", "nen_0001"),
        ("ハルキ", "「やあ」", "har_0001"),
        ("", "扉が開いた。", None),
        ("ハルキ", "「名前行」", None),
        ("寧々", "「名前モード」", None),
        ("寧々", "「アット行」", "nen_0002"),
    ]
    assert rows[1].label == "start"
    assert "[nm" in rows[1].raw


def test_break_on_r_option() -> None:
    p = KagParser(learn_macros({}), break_on_r=True)
    rows = list(p.parse("一行目[r]二行目[p]\n", "s", "s.ks"))
    assert [r.text for r in rows] == ["一行目", "二行目"]


def test_scn_json_defensive() -> None:
    data = {
        "scenes": [
            {
                "label": "*s01",
                "texts": [
                    ["nene", "寧々", "「本当に？」", [{"voice": "nen_0001"}]],
                    [None, None, "地の文。"],
                    ["haruki", "", [["haruki", "「やあ」"]], {"data": [{"voice": {"name": "har_0001"}}]}],
                    "garbage",
                ],
            }
        ]
    }
    rows = parse_scn_json(data, "s01")
    assert [(r["disp"], r["text"], r["voices"]) for r in rows] == [
        ("寧々", "「本当に？」", ["nen_0001"]),
        ("", "地の文。", []),
        ("haruki", "haruki", ["har_0001"]),
    ]


def test_kag_fixture_parse(make_ctx, fixtures_dir: Path) -> None:
    from engine_helpers import check_golden, detect_and_parse

    summary, ex, audio, res = detect_and_parse(make_ctx(fixtures_dir / "kag_mini"))
    assert summary["engine"] == "kirikiri"
    voiced = [l for l in res.lines if l.has_voice]
    assert [l.voice_file for l in voiced] == ["voice/nen_0001.ogg", "voice/har_0001.ogg", "voice/nen_0002.ogg", "voice/har_0002.ogg", "voice/nen_0003.ogg"]
    assert not any(l.voice_file and "se_wind" in l.voice_file for l in res.lines)  # SE playse ignored
    assert any("nm(name=t, voice=s)" in n for n in res.notes)
    check_golden("kag_mini", res.lines)


def test_next_patch_name(tmp_path: Path) -> None:
    assert next_patch_name(tmp_path) == "patch.xp3"
    (tmp_path / "patch.xp3").write_bytes(b"")
    assert next_patch_name(tmp_path) == "patch2.xp3"
    (tmp_path / "patch2.xp3").write_bytes(b"")
    assert next_patch_name(tmp_path) == "patch3.xp3"


def test_kirikiri_extract_from_xp3(tmp_path: Path, make_ctx) -> None:
    from vndub.engines.xp3 import write_xp3

    game = tmp_path / "g"
    game.mkdir()
    write_xp3(game / "data.xp3", [("scenario/a.ks", '[nm t="寧々" s=nen_0001]「中身」[p]\n'.encode()), ("voice/nen_0001.ogg", b"OggS" + bytes(100))])
    write_xp3(game / "patch.xp3", [("voice/nen_0001.ogg", b"OggS" + b"\x07" * 100)])  # patch overrides data.xp3
    from engine_helpers import detect_and_parse

    summary, ex, audio, res = detect_and_parse(make_ctx(game))
    assert summary["engine"] == "kirikiri" and not summary["encrypted"]
    assert len(audio) == 1 and Path(audio[0].src_path).read_bytes().endswith(b"\x07" * 10)
    (line,) = res.lines
    assert line.speaker_raw == "寧々" and line.voice_file == "voice/nen_0001.ogg"

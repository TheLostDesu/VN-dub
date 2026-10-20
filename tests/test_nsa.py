from __future__ import annotations

from pathlib import Path

import pytest

from vndub.engines import nsa


def test_nscript_xor_roundtrip() -> None:
    text = '*start\ndwave 0,"voice\\a001.wav"\n【澪】「こんにちは」\\\n'
    enc = nsa.encode_nscript(text)
    assert enc != text.encode("cp932")
    assert all(b == (x ^ 0x84) for b, x in zip(enc, text.encode("cp932")))
    assert nsa.decode_nscript(enc) == text


def test_fixture_nscript_dat_decodes(fixtures_dir: Path) -> None:
    text = nsa.decode_nscript((fixtures_dir / "nscripter_mini" / "nscript.dat").read_bytes())
    assert "dwave 0" in text and "【澪】" in text


FILES = [("voice/a001.wav", b"RIFF" + bytes(range(200))), ("voice/b002.ogg", b"OggS" * 40), ("bgm/x.ogg", b"OggS")]


@pytest.mark.parametrize("comp", [nsa.COMP_NONE, nsa.COMP_LZSS, nsa.COMP_NBZ])
def test_nsa_roundtrip(tmp_path: Path, comp: int) -> None:
    p = tmp_path / "arc.nsa"
    assert nsa.write_nsa(p, FILES, compression=comp) == 3
    with nsa.NSAReader(p) as r:
        assert [e.name for e in r.entries] == [n for n, _ in FILES]
        assert all(e.compression == comp for e in r.entries)
        for name, data in FILES:
            assert r.read(name) == data
        assert r.read("VOICE\\A001.WAV") == FILES[0][1]  # case/separator-insensitive lookup
        out = r.extract_all(tmp_path / "x", ["voice/b002.ogg"])
    assert [o.name for o in out] == ["b002.ogg"]


def test_sar_and_ns2_roundtrip(tmp_path: Path) -> None:
    for writer, ext in ((nsa.write_sar, ".sar"), (nsa.write_ns2, ".ns2")):
        p = tmp_path / f"arc{ext}"
        writer(p, FILES)
        with nsa.NSAReader(p) as r:
            assert {e.name for e in r.entries} == {n for n, _ in FILES}
            for name, data in FILES:
                assert r.read(name) == data


def test_lzss_backreference_decode() -> None:
    # literal 'A' then a back-reference to ring position 239 (where 'A' was written), length 0+2
    bits = [1] + [int(b) for b in format(ord("A"), "08b")]
    bits += [0] + [int(b) for b in format(239, "08b")] + [0, 0, 0, 0]
    while len(bits) % 8:
        bits.append(0)
    data = bytes(int("".join(map(str, bits[i : i + 8])), 2) for i in range(0, len(bits), 8))
    assert nsa.lzss_decode(data, 3) == b"AAA"
    assert nsa.lzss_decode(nsa.lzss_encode_literal(b"xyz" * 50), 150) == b"xyz" * 50


def test_archive_order(tmp_path: Path) -> None:
    for n in ("arc2.nsa", "arc.nsa", "arc1.nsa", "extra.sar"):
        (tmp_path / n).write_bytes(b"")
    assert [p.name for p in nsa.archive_files(tmp_path)] == ["arc.nsa", "arc1.nsa", "arc2.nsa", "extra.sar"]


def test_bad_archive(tmp_path: Path) -> None:
    p = tmp_path / "bad.nsa"
    p.write_bytes(b"\x00\x01\xff\xff\xff\xff")
    with pytest.raises(nsa.NSAError):
        nsa.NSAReader(p)

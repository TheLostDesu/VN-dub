from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest

from vndub.engines import xp3


def test_roundtrip_stored_and_compressed(tmp_path: Path) -> None:
    files = [
        ("scenario/first.ks", "[nm t=\"寧々\" s=nen_0001]「テスト」[p]\n".encode("utf-8")),
        ("voice/nen_0001.ogg", b"OggS" + bytes(range(256)) * 20),
        ("bare.txt", b"hello"),
    ]
    p = tmp_path / "patch.xp3"
    assert xp3.write_xp3(p, files) == 3
    with xp3.XP3Reader(p) as r:
        assert [e.name for e in r.entries] == [n for n, _ in files]
        for name, data in files:
            assert r.read(name) == data
        ks = r.by_name["scenario/first.ks"]
        assert ks.segments[0].compressed  # text members are zlib-compressed by default
        assert not r.by_name["voice/nen_0001.ogg"].segments[0].compressed
        assert ks.adler32 == zlib.adler32(files[0][1]) & 0xFFFFFFFF
        enc, why = xp3.looks_encrypted(r)
        assert not enc, why
    # header layout: magic, cushion pointer 0x17, minor version 1, continue flag
    raw = p.read_bytes()
    assert raw.startswith(xp3.MAGIC)
    assert struct.unpack_from("<Q", raw, 11)[0] == 0x17
    assert struct.unpack_from("<I", raw, 19)[0] == 1
    assert raw[0x17] == 0x80


def test_force_compress_and_extract(tmp_path: Path) -> None:
    p = tmp_path / "a.xp3"
    xp3.write_xp3(p, [("voice/x.wav", b"RIFF" + b"\0" * 100), ("../evil.txt", b"x")], compress=True)
    out = tmp_path / "out"
    with xp3.XP3Reader(p) as r:
        assert all(e.segments[0].compressed for e in r.entries)
        written = r.extract_all(out)
    assert (out / "voice" / "x.wav").read_bytes().startswith(b"RIFF")
    assert all(out in w.parents for w in written)  # '..' stripped
    assert xp3.list_names(p) == ["voice/x.wav", "../evil.txt"]


def test_old_style_header_without_cushion(tmp_path: Path) -> None:
    data = b"OggS" + b"\x01" * 50
    body = bytearray()
    name = "a.ogg"
    info = struct.pack("<IQQH", 0, len(data), len(data), len(name)) + name.encode("utf-16-le")
    segm = struct.pack("<IQQQ", 0, 19, len(data), len(data))
    file_body = b"info" + struct.pack("<Q", len(info)) + info + b"segm" + struct.pack("<Q", len(segm)) + segm
    body += b"File" + struct.pack("<Q", len(file_body)) + file_body
    index_ofs = 19 + len(data)
    raw = xp3.MAGIC + struct.pack("<Q", index_ofs) + data + bytes([0]) + struct.pack("<Q", len(body)) + bytes(body)
    p = tmp_path / "old.xp3"
    p.write_bytes(raw)
    with xp3.XP3Reader(p) as r:
        assert r.read("a.ogg") == data


def test_bad_magic_and_encrypted_index(tmp_path: Path) -> None:
    bad = tmp_path / "bad.xp3"
    bad.write_bytes(b"NOTXP3" + b"\0" * 64)
    with pytest.raises(xp3.XP3Error):
        xp3.XP3Reader(bad)
    p = tmp_path / "enc.xp3"
    xp3.write_xp3(p, [("a.ogg", b"OggS" * 10)])
    raw = bytearray(p.read_bytes())
    index_ofs = struct.unpack_from("<Q", raw, 0x17 + 9)[0]
    for i in range(index_ofs + 17, len(raw)):  # scramble the compressed index like a cipher would
        raw[i] ^= 0x5A
    p.write_bytes(bytes(raw))
    with pytest.raises(xp3.XP3Error, match="encrypted"):
        xp3.XP3Reader(p)


def test_looks_encrypted_on_scrambled_members(tmp_path: Path) -> None:
    files = [(f"voice/v{i}.ogg", bytes((b ^ 0xAA) for b in (b"OggS" + bytes(60)))) for i in range(4)]
    p = tmp_path / "s.xp3"
    xp3.write_xp3(p, files)
    with xp3.XP3Reader(p) as r:
        enc, why = xp3.looks_encrypted(r)
    assert enc and "do not match" in why


def test_fixture_archive_readable(fixtures_dir: Path) -> None:
    with xp3.XP3Reader(fixtures_dir / "kag_mini" / "xp3" / "sample.xp3") as r:
        names = {e.name for e in r.entries}
        assert names == {"scenario/test.ks", "voice/xp_0001.wav"}
        assert r.read("voice/xp_0001.wav").startswith(b"RIFF")
        assert "アーカイブ" in r.read("scenario/test.ks").decode("utf-8")

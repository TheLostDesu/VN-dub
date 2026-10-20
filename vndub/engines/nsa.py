"""NScripter / ONScripter containers: ``nscript.dat``, SAR, NSA, NS2.

* ``nscript.dat`` (also ``pscript.dat``): every byte XOR 0x84; Shift-JIS text.
* SAR: ``u16 count, u32 data_offset`` (big-endian), then per entry
  ``name\\0, u32 offset, u32 length``; offsets are relative to data_offset.
* NSA: like SAR plus a compression byte per entry:
  ``name\\0, u8 type, u32 offset, u32 packed, u32 original``;
  type 0 = stored, 1 = SPB (bitmap), 2 = LZSS, 4 = NBZ (u32 BE size + bzip2).
* NS2 (ONScripter-EN/ns2): ``u32 LE data_offset``, then ``"name"`` + ``u32 LE size``
  entries until ``e``; data follows in entry order.

Archives ``arc.nsa, arc1.nsa .. arc9.nsa`` (and ``arc.sar``) share one
namespace; loose files next to the exe override archive members. SPB bitmap
decoding is not implemented (images are never voices); members using it are
listed but skipped on extraction.
"""

from __future__ import annotations

import bz2
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

XOR_KEY = 0x84

COMP_NONE, COMP_SPB, COMP_LZSS, COMP_NBZ = 0, 1, 2, 4


class NSAError(ValueError):
    pass


# ----------------------------------------------------------------------------
# nscript.dat
# ----------------------------------------------------------------------------


def xor_bytes(data: bytes, key: int = XOR_KEY) -> bytes:
    return bytes(b ^ key for b in data)


def decode_nscript(data: bytes, encoding: str = "cp932") -> str:
    return xor_bytes(data).decode(encoding, errors="replace")


def encode_nscript(text: str, encoding: str = "cp932") -> bytes:
    return xor_bytes(text.encode(encoding))


# ----------------------------------------------------------------------------
# LZSS (EI=8, EJ=4, P=1 -> N=256, F=17, r = N - F = 239), MSB-first bit stream
# ----------------------------------------------------------------------------


class _BitReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0
        self.bit = 0

    def get(self, n: int) -> int:
        v = 0
        for _ in range(n):
            if self.pos >= len(self.data):
                return -1
            b = (self.data[self.pos] >> (7 - self.bit)) & 1
            self.bit += 1
            if self.bit == 8:
                self.bit = 0
                self.pos += 1
            v = (v << 1) | b
        return v


def lzss_decode(data: bytes, original_length: int) -> bytes:
    ring = bytearray(256)
    r = 239
    out = bytearray()
    br = _BitReader(data)
    while len(out) < original_length:
        flag = br.get(1)
        if flag < 0:
            break
        if flag:
            c = br.get(8)
            if c < 0:
                break
            out.append(c)
            ring[r] = c
            r = (r + 1) & 0xFF
        else:
            i = br.get(8)
            j = br.get(4)
            if i < 0 or j < 0:
                break
            for k in range(j + 2):
                c = ring[(i + k) & 0xFF]
                out.append(c)
                ring[r] = c
                r = (r + 1) & 0xFF
    return bytes(out[:original_length])


def lzss_encode_literal(data: bytes) -> bytes:
    """Valid LZSS stream using literals only (sufficient for writing test archives)."""
    bits = []
    for b in data:
        bits.append(1)
        bits.extend((b >> (7 - k)) & 1 for k in range(8))
    while len(bits) % 8:
        bits.append(0)
    out = bytearray()
    for i in range(0, len(bits), 8):
        v = 0
        for bit in bits[i : i + 8]:
            v = (v << 1) | bit
        out.append(v)
    return bytes(out)


# ----------------------------------------------------------------------------
# archives
# ----------------------------------------------------------------------------


@dataclass
class NSAEntry:
    name: str
    offset: int  # absolute file offset
    packed: int
    original: int
    compression: int = COMP_NONE


class NSAReader:
    """Reads .sar, .nsa and .ns2 archives (format chosen by extension, then sniffed)."""

    def __init__(self, path: str | Path, encoding: str = "cp932") -> None:
        self.path = Path(path)
        self.encoding = encoding
        self._f: BinaryIO = open(self.path, "rb")
        try:
            ext = self.path.suffix.lower()
            if ext == ".ns2":
                self.entries = self._read_ns2()
            else:
                self.entries = self._read_table(with_compression=(ext != ".sar"))
        except Exception:
            self._f.close()
            raise
        self.by_name = {e.name.lower(): e for e in self.entries}

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "NSAReader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _read_table(self, with_compression: bool) -> list[NSAEntry]:
        f = self._f
        size = self.path.stat().st_size
        head = f.read(6)
        if len(head) < 6:
            raise NSAError(f"{self.path}: too small")
        count, data_offset = struct.unpack(">HI", head)
        if not (6 <= data_offset <= size):
            raise NSAError(f"{self.path}: bad data offset {data_offset}")
        table = f.read(data_offset - 6)
        out, pos = [], 0
        for _ in range(count):
            end = table.find(b"\0", pos)
            if end < 0:
                raise NSAError(f"{self.path}: truncated entry table")
            name = table[pos:end].decode(self.encoding, errors="replace").replace("\\", "/")
            pos = end + 1
            if with_compression:
                comp = table[pos]
                off, packed, orig = struct.unpack_from(">III", table, pos + 1)
                pos += 13
            else:
                comp = COMP_NONE
                off, packed = struct.unpack_from(">II", table, pos)
                orig = packed
                pos += 8
            out.append(NSAEntry(name, data_offset + off, packed, orig, comp))
        return out

    def _read_ns2(self) -> list[NSAEntry]:
        f = self._f
        data_offset = struct.unpack("<I", f.read(4))[0]
        table = f.read(data_offset - 4)
        out, pos, cur = [], 0, data_offset
        while pos < len(table) and table[pos : pos + 1] == b'"':
            end = table.find(b'"', pos + 1)
            name = table[pos + 1 : end].decode(self.encoding, errors="replace").replace("\\", "/")
            size = struct.unpack_from("<I", table, end + 1)[0]
            pos = end + 5
            out.append(NSAEntry(name, cur, size, size, COMP_NONE))
            cur += size
        return out

    def read(self, name_or_entry: str | NSAEntry) -> bytes:
        e = self.by_name[name_or_entry.replace("\\", "/").lower()] if isinstance(name_or_entry, str) else name_or_entry
        self._f.seek(e.offset)
        raw = self._f.read(e.packed)
        if e.compression == COMP_NONE:
            return raw
        if e.compression == COMP_LZSS:
            return lzss_decode(raw, e.original)
        if e.compression == COMP_NBZ:
            return bz2.decompress(raw[4:])
        raise NSAError(f"{e.name}: compression type {e.compression} (SPB) not supported")

    def extract_all(self, dest: str | Path, names: Iterable[str] | None = None) -> list[Path]:
        from vndub.engines.xp3 import safe_relpath

        dest = Path(dest)
        wanted = {n.lower() for n in names} if names is not None else None
        out = []
        for e in self.entries:
            if wanted is not None and e.name.lower() not in wanted:
                continue
            if e.compression == COMP_SPB:
                continue
            target = dest / safe_relpath(e.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.read(e))
            out.append(target)
        return out


def write_sar(path: str | Path, files: Iterable[tuple[str, bytes]], encoding: str = "cp932") -> int:
    return _write_table(path, files, encoding, with_compression=False, compression=COMP_NONE)


def write_nsa(path: str | Path, files: Iterable[tuple[str, bytes]], compression: int = COMP_NONE, encoding: str = "cp932") -> int:
    if compression not in (COMP_NONE, COMP_LZSS, COMP_NBZ):
        raise NSAError("writer supports compression 0 (stored), 2 (LZSS) and 4 (NBZ)")
    return _write_table(path, files, encoding, with_compression=True, compression=compression)


def _write_table(path: str | Path, files: Iterable[tuple[str, bytes]], encoding: str, with_compression: bool, compression: int) -> int:
    items = [(n.replace("/", "\\"), d) for n, d in files]
    blobs = []
    for _, d in items:
        if compression == COMP_LZSS:
            blobs.append(lzss_encode_literal(d))
        elif compression == COMP_NBZ:
            blobs.append(struct.pack(">I", len(d)) + bz2.compress(d))
        else:
            blobs.append(d)
    names = [n.encode(encoding) for n, _ in items]
    per = 13 if with_compression else 8
    data_offset = 6 + sum(len(n) + 1 + per for n in names)
    table = bytearray(struct.pack(">HI", len(items), data_offset))
    off = 0
    for (_, d), nm, blob in zip(items, names, blobs):
        table += nm + b"\0"
        if with_compression:
            table += struct.pack(">BIII", compression, off, len(blob), len(d))
        else:
            table += struct.pack(">II", off, len(blob))
        off += len(blob)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        f.write(table)
        for blob in blobs:
            f.write(blob)
    return len(items)


def write_ns2(path: str | Path, files: Iterable[tuple[str, bytes]], encoding: str = "cp932") -> int:
    items = list(files)
    header = bytearray()
    for n, d in items:
        header += b'"' + n.replace("/", "\\").encode(encoding) + b'"' + struct.pack("<I", len(d))
    header += b"e"
    data_offset = 4 + len(header)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        f.write(struct.pack("<I", data_offset))
        f.write(header)
        for _, d in items:
            f.write(d)
    return len(items)


def archive_files(game_dir: Path) -> list[Path]:
    """NScripter archives in load order (later override earlier is NOT the rule:
    ONScripter searches loose files first, then arc.nsa, arc1..arc9; first hit wins)."""
    out: list[Path] = []
    for stem in ["arc"] + [f"arc{i}" for i in range(1, 10)]:
        for ext in (".nsa", ".sar", ".ns2"):
            p = game_dir / f"{stem}{ext}"
            if p.exists():
                out.append(p)
    for p in sorted(game_dir.glob("*.ns2")) + sorted(game_dir.glob("*.sar")) + sorted(game_dir.glob("*.nsa")):
        if p not in out:
            out.append(p)
    return out

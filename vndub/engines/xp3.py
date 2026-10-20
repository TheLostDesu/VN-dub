"""Pure-Python reader and writer for *unencrypted* KiriKiri XP3 archives.

Layout (krkr2 2.29+ / krkrz "version 2" header)::

    0x00  "XP3\\r\\n \\n\\x1a\\x8b\\x67\\x01"          11-byte magic
    0x0b  u64 = 0x17                              offset of the "cushion"
    0x13  u32 = 1                                 header minor version
    0x17  u8 0x80, u64 0, u64 index_offset        continue-flag cushion -> real index
    0x28  file data ...
    index: u8 flag (0 raw, 1 zlib), [u64 packed], u64 size, data

The index is a sequence of ``File`` chunks, each holding ``info`` (flags, sizes,
UTF-16LE name), ``segm`` (one or more segments: flags, offset, orig, packed;
flag 1 = zlib) and ``adlr`` (Adler-32 of the original data). Old-style
archives point the u64 at 0x0b straight at the index; both are read.
Unknown chunks (``time``, ``eliF``, vendor hashes) are skipped.

Encrypted archives (game-specific extraction filters) cannot be decrypted
here: :func:`looks_encrypted` detects them heuristically so the extractor can
point the user to KrkrDump / KirikiriTools ``version.dll`` dumps instead.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Iterable

MAGIC = b"XP3\r\n \n\x1a\x8b\x67\x01"
FLAG_ZLIB = 1
INDEX_CONTINUE = 0x80
FILE_PROTECTED = 0x80000000


class XP3Error(ValueError):
    pass


@dataclass
class Segment:
    flags: int
    offset: int
    orig_size: int
    packed_size: int

    @property
    def compressed(self) -> bool:
        return (self.flags & 7) == FLAG_ZLIB


@dataclass
class XP3Entry:
    name: str
    flags: int = 0
    orig_size: int = 0
    packed_size: int = 0
    adler32: int | None = None
    segments: list[Segment] = field(default_factory=list)

    @property
    def protected(self) -> bool:
        return bool(self.flags & FILE_PROTECTED)


def _u64(b: bytes, o: int) -> int:
    return struct.unpack_from("<Q", b, o)[0]


def _u32(b: bytes, o: int) -> int:
    return struct.unpack_from("<I", b, o)[0]


class XP3Reader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._f: BinaryIO = open(self.path, "rb")
        try:
            self.entries = self._read_index()
        except Exception:
            self._f.close()
            raise
        self.by_name = {e.name: e for e in self.entries}

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "XP3Reader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- index -----------------------------------------------------------------
    def _read_at(self, offset: int, n: int) -> bytes:
        self._f.seek(offset)
        data = self._f.read(n)
        if len(data) != n:
            raise XP3Error(f"{self.path}: truncated read at {offset} ({len(data)}/{n})")
        return data

    def _read_index(self) -> list[XP3Entry]:
        size = self.path.stat().st_size
        head = self._read_at(0, min(size, 64))
        if not head.startswith(MAGIC):
            raise XP3Error(f"{self.path}: not an XP3 archive (bad magic)")
        index_ofs = _u64(head, 11)
        entries: list[XP3Entry] = []
        guard = 0
        while True:
            guard += 1
            if guard > 64 or not (0 < index_ofs < size):
                raise XP3Error(f"{self.path}: index offset {index_ofs} out of range")
            flag = self._read_at(index_ofs, 1)[0]
            method = flag & 7
            if method == FLAG_ZLIB:
                packed, unpacked = struct.unpack("<QQ", self._read_at(index_ofs + 1, 16))
                raw = self._read_at(index_ofs + 17, packed)
                try:
                    data = zlib.decompress(raw)
                except zlib.error as e:
                    raise XP3Error(f"{self.path}: index does not decompress ({e}); archive is probably encrypted") from e
                if len(data) != unpacked:
                    raise XP3Error(f"{self.path}: index size mismatch")
                after = index_ofs + 17 + packed
            elif method == 0:
                n = struct.unpack("<Q", self._read_at(index_ofs + 1, 8))[0]
                data = self._read_at(index_ofs + 9, n) if n else b""
                after = index_ofs + 9 + n
            else:
                raise XP3Error(f"{self.path}: unknown index encoding {method}; archive is probably encrypted or customized")
            entries.extend(self._parse_index(data))
            if flag & INDEX_CONTINUE:
                # cushion: the next u64 after this (usually empty) index is the real index offset
                index_ofs = struct.unpack("<Q", self._read_at(after, 8))[0]
                continue
            break
        return entries

    def _parse_index(self, data: bytes) -> list[XP3Entry]:
        out: list[XP3Entry] = []
        pos = 0
        while pos + 12 <= len(data):
            tag = data[pos : pos + 4]
            n = _u64(data, pos + 4)
            body = data[pos + 12 : pos + 12 + n]
            pos += 12 + n
            if tag == b"File":
                out.append(self._parse_file(body))
        return out

    @staticmethod
    def _parse_file(body: bytes) -> XP3Entry:
        e = XP3Entry(name="")
        pos = 0
        while pos + 12 <= len(body):
            tag = body[pos : pos + 4]
            n = _u64(body, pos + 4)
            c = body[pos + 12 : pos + 12 + n]
            pos += 12 + n
            if tag == b"info":
                e.flags = _u32(c, 0)
                e.orig_size = _u64(c, 4)
                e.packed_size = _u64(c, 12)
                name_len = struct.unpack_from("<H", c, 20)[0]
                e.name = c[22 : 22 + 2 * name_len].decode("utf-16-le", errors="replace")
            elif tag == b"segm":
                for k in range(0, len(c) - 27, 28):
                    fl, off, orig, packed = struct.unpack_from("<IQQQ", c, k)
                    e.segments.append(Segment(fl, off, orig, packed))
            elif tag == b"adlr" and len(c) >= 4:
                e.adler32 = _u32(c, 0)
        return e

    # -- data ------------------------------------------------------------------
    def read(self, name_or_entry: str | XP3Entry, verify: bool = True) -> bytes:
        e = self.by_name[name_or_entry] if isinstance(name_or_entry, str) else name_or_entry
        parts = []
        for s in e.segments:
            raw = self._read_at(s.offset, s.packed_size)
            parts.append(zlib.decompress(raw) if s.compressed else raw)
        data = b"".join(parts)
        if verify and e.adler32 is not None and (zlib.adler32(data) & 0xFFFFFFFF) != e.adler32:
            raise XP3Error(f"{self.path}: checksum mismatch for {e.name} (encrypted data?)")
        return data

    def extract_all(self, dest: str | Path, names: Iterable[str] | None = None) -> list[Path]:
        dest = Path(dest)
        wanted = set(names) if names is not None else None
        out = []
        for e in self.entries:
            if wanted is not None and e.name not in wanted:
                continue
            target = dest / safe_relpath(e.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.read(e))
            out.append(target)
        return out


def safe_relpath(name: str) -> Path:
    """Archive member name -> safe relative path (no absolute paths / '..')."""
    parts = [p for p in name.replace("\\", "/").split("/") if p not in ("", ".", "..")]
    return Path(*parts) if parts else Path("_unnamed")


_SIGS = {".ogg": b"OggS", ".wav": b"RIFF", ".png": b"\x89PNG", ".opus": b"OggS", ".jpg": b"\xff\xd8", ".tlg": b"TLG"}


def looks_encrypted(reader: XP3Reader, sample: int = 24) -> tuple[bool, str]:
    """Heuristic: protected flags or known-format members whose bytes do not look right."""
    checked = bad = 0
    for e in reader.entries:
        if checked >= sample:
            break
        suf = Path(e.name).suffix.lower()
        if suf not in _SIGS and suf not in (".ks", ".tjs", ".txt", ".csv"):
            continue
        checked += 1
        try:
            data = reader.read(e, verify=True)[:4096]
        except (XP3Error, zlib.error):
            bad += 1
            continue
        if suf in _SIGS:
            if not data.startswith(_SIGS[suf]):
                bad += 1
        else:
            if not _textlike(data):
                bad += 1
    if any(e.protected for e in reader.entries) and bad:
        return True, "entries flagged protected and contents unreadable"
    if checked and bad / checked > 0.5:
        return True, f"{bad}/{checked} sampled entries do not match their format"
    return False, ""


def _textlike(data: bytes) -> bool:
    if data[:2] in (b"\xff\xfe", b"\xfe\xff") or data[:3] == b"\xef\xbb\xbf":
        return True
    if data[:5] == b"\xfe\xfe\x01\xff\xfe":  # KirikiriDescrambler-able scrambled text
        return False
    for enc in ("utf-8", "cp932"):
        try:
            data.decode(enc)
            return True
        except UnicodeDecodeError:
            continue
    return False


# ----------------------------------------------------------------------------
# writer
# ----------------------------------------------------------------------------


def write_xp3(path: str | Path, files: Iterable[tuple[str, bytes]], compress: bool | None = None) -> int:
    """Write an unencrypted version-2 XP3. ``files``: (archive name, data).

    ``compress=None`` compresses only text-like members (scripts); audio is
    stored as-is (already compressed). Returns the number of members.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    index = bytearray()
    n = 0
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<Q", 0x17))
        f.write(struct.pack("<I", 1))
        cushion_pos = f.tell()  # 0x17
        f.write(struct.pack("<BQQ", INDEX_CONTINUE, 0, 0))  # patched below
        for name, data in files:
            name = name.replace("\\", "/")
            do_zip = compress if compress is not None else Path(name).suffix.lower() in (".ks", ".tjs", ".txt", ".csv", ".json", ".scn")
            stored = zlib.compress(data, 9) if do_zip else data
            offset = f.tell()
            f.write(stored)
            enc_name = name.encode("utf-16-le")
            info = struct.pack("<IQQH", 0, len(data), len(stored), len(name)) + enc_name
            segm = struct.pack("<IQQQ", FLAG_ZLIB if do_zip else 0, offset, len(data), len(stored))
            adlr = struct.pack("<I", zlib.adler32(data) & 0xFFFFFFFF)
            body = _chunk(b"info", info) + _chunk(b"segm", segm) + _chunk(b"adlr", adlr)
            index += _chunk(b"File", body)
            n += 1
        index_ofs = f.tell()
        packed = zlib.compress(bytes(index), 9)
        f.write(struct.pack("<BQQ", FLAG_ZLIB, len(packed), len(index)))
        f.write(packed)
        f.seek(cushion_pos)
        f.write(struct.pack("<BQQ", INDEX_CONTINUE, 0, index_ofs))
    return n


def _chunk(tag: bytes, body: bytes) -> bytes:
    return tag + struct.pack("<Q", len(body)) + body


def list_names(path: str | Path) -> list[str]:
    with XP3Reader(path) as r:
        return [e.name for e in r.entries]

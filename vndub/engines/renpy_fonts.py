"""A small reader of the ``cmap`` table of a TrueType / OpenType font: does a font file have the glyphs of the Russian alphabet?

A Ren'Py game whose font has no Cyrillic letters shows the translated text as boxes. ``pack`` asks this module about the fonts the
game's scripts name. Only the formats the fonts of games use are read (4 = BMP segments, 12 = full range); a font with neither
subtable gives ``None`` ("cannot tell"), never a false "no". ``struct`` only: no font library.
"""

from __future__ import annotations

import struct
from pathlib import Path

# U+0410..U+044F (the letters) and the two yo's: what a Russian text needs besides ASCII
CYRILLIC = tuple(range(0x0410, 0x0450)) + (0x0401, 0x0451)


class _Unreadable(Exception):
    pass


def _u16(b: bytes, o: int) -> int:
    return struct.unpack_from(">H", b, o)[0]


def _u32(b: bytes, o: int) -> int:
    return struct.unpack_from(">I", b, o)[0]


def _table(fh, base: int, tag: bytes) -> tuple[int, int] | None:
    """(offset, length) of table ``tag`` of the font whose offset table starts at ``base``."""
    fh.seek(base)
    head = fh.read(12)
    if len(head) < 12 or head[:4] not in (b"\x00\x01\x00\x00", b"OTTO", b"true", b"typ1"):
        raise _Unreadable("not an sfnt font")
    n = _u16(head, 4)
    rec = fh.read(16 * n)
    if len(rec) < 16 * n:
        raise _Unreadable("truncated table directory")
    for i in range(n):
        if rec[16 * i : 16 * i + 4] == tag:
            return _u32(rec, 16 * i + 8), _u32(rec, 16 * i + 12)
    return None


def _format4(b: bytes, o: int, cps: tuple[int, ...]) -> set[int]:
    seg = _u16(b, o + 6) // 2
    end = o + 14
    start = end + 2 * seg + 2
    delta = start + 2 * seg
    rng = delta + 2 * seg
    found: set[int] = set()
    for cp in cps:
        if cp > 0xFFFF:
            continue
        for i in range(seg):
            if _u16(b, end + 2 * i) < cp:
                continue
            if _u16(b, start + 2 * i) > cp:
                break
            ro = _u16(b, rng + 2 * i)
            if ro == 0:
                gid = (cp + _u16(b, delta + 2 * i)) & 0xFFFF
            else:
                p = rng + 2 * i + ro + 2 * (cp - _u16(b, start + 2 * i))
                if p + 2 > len(b):
                    break
                gid = _u16(b, p)
                if gid:
                    gid = (gid + _u16(b, delta + 2 * i)) & 0xFFFF
            if gid:
                found.add(cp)
            break
    return found


def _format12(b: bytes, o: int, cps: tuple[int, ...]) -> set[int]:
    n = _u32(b, o + 12)
    groups = [struct.unpack_from(">III", b, o + 16 + 12 * i) for i in range(min(n, (len(b) - o - 16) // 12))]
    found: set[int] = set()
    for cp in cps:
        for s, e, g in groups:
            if s <= cp <= e:
                if g + (cp - s):
                    found.add(cp)
                break
    return found


def covered(path: str | Path, codepoints: tuple[int, ...] = CYRILLIC) -> set[int] | None:
    """The code points of ``codepoints`` that ``path`` has a glyph for, or None when the file cannot be read as a font / has no
    format 4 or 12 Unicode subtable."""
    try:
        with open(path, "rb") as fh:
            magic = fh.read(4)
            base = 0
            if magic == b"ttcf":  # a collection: the first font stands for it
                fh.seek(12)
                base = _u32(fh.read(4), 0)
            tab = _table(fh, base, b"cmap")
            if tab is None:
                return None
            off, length = tab
            fh.seek(off)
            b = fh.read(min(length, 4_000_000))
        n = _u16(b, 2)
        subs = []
        for i in range(n):
            plat, enc, so = _u16(b, 4 + 8 * i), _u16(b, 6 + 8 * i), _u32(b, 8 + 8 * i)
            if (plat, enc) in ((3, 10), (0, 4), (0, 6), (3, 1), (0, 3), (0, 2), (0, 1), (0, 0)) and so + 4 <= len(b):
                subs.append((0 if (plat, enc) in ((3, 10), (0, 4), (0, 6)) else 1, so))
        got: set[int] | None = None
        for _prio, so in sorted(subs):
            fmt = _u16(b, so)
            if fmt == 12:
                part = _format12(b, so, tuple(codepoints))
            elif fmt == 4:
                part = _format4(b, so, tuple(codepoints))
            else:
                continue
            got = (got or set()) | part
        return got
    except (OSError, struct.error, _Unreadable):
        return None


def has_cyrillic(path: str | Path) -> bool | None:
    """True when the font has every letter of the Russian alphabet (U+0410..U+044F, U+0401, U+0451), False when one is missing,
    None when the file cannot be read."""
    got = covered(path)
    if got is None:
        return None
    return all(cp in got for cp in CYRILLIC)

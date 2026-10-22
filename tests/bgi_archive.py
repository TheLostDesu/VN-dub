"""A tiny BGI / Ethornell game built from scratch for bgi-extractor: a real BURIKO ARC20 archive with
DSC-compressed BurikoCompiledScript files (used by tests/test_bgi_extractor_exe.py and packaging/smoke_exe.py)."""

from __future__ import annotations

import struct
from pathlib import Path


def _hash_update(h: int) -> tuple[int, int]:
    """The key stream of the DSC code-length table (BGI's 20021/346 generator)."""
    edx = 20021 * (h & 0xFFFF)
    eax = (20021 * (h >> 16) + 346 * h + (edx >> 16)) & 0xFFFFFFFF
    return (((eax & 0xFFFF) << 16) + (edx & 0xFFFF) + 1) & 0xFFFFFFFF, eax & 0x7FFF


def dsc(data: bytes, key: int = 0x12345678) -> bytes:
    """DSC FORMAT 1.00 with every literal at code length 8 and no back-references: the codes are the bytes themselves."""
    table = bytearray()
    h = key
    for n in range(512):
        h, r = _hash_update(h)
        table.append(((8 if n < 256 else 0) + r) & 0xFF)
    return b"DSC FORMAT 1.00\0" + struct.pack("<IIII", key, len(data), 0, 0) + bytes(table) + data


def bss(messages: list[tuple[str | None, str, str | None]]) -> bytes:
    """BurikoCompiledScriptVer1.00: per message [push voice, 0x1A5], [push speaker], push text, 0x140; then ret."""
    strings: list[bytes] = []
    code: list[tuple[int, int | None]] = []

    def push(s: str) -> None:
        code.append((0x003, len(strings)))
        strings.append(s.encode("cp932") + b"\0")

    for speaker, text, voice in messages:
        if voice:
            push(voice)
            code.append((0x1A5, None))
        if speaker:
            push(speaker)
        push(text)
        code.append((0x140, None))
    code.append((0x01B, None))
    offsets, pos = [], sum(8 if arg is not None else 4 for _, arg in code)
    for s in strings:
        offsets.append(pos)
        pos += len(s)
    body = b"".join(struct.pack("<I", op) + (struct.pack("<I", offsets[arg]) if arg is not None else b"") for op, arg in code)
    return b"BurikoCompiledScriptVer1.00\0" + struct.pack("<I", 4) + body + b"".join(strings)


def arc(entries: dict[str, bytes]) -> bytes:
    index, body = bytearray(), bytearray()
    for name, payload in entries.items():
        index += name.encode("ascii").ljust(0x60, b"\0") + struct.pack("<II", len(body), len(payload)) + bytes(0x18)
        body += payload
    return b"BURIKO ARC20" + struct.pack("<I", len(entries)) + bytes(index) + bytes(body)


SC01 = [("由岐", "「間宮卓司……」", "yuki_000003"), (None, "　街の喧噪の中に立っている時とは打って変わって。", None), ("卓司", "「なんでこんな所に？」", "taku_000001")]
SC02 = [("由岐", "「行こう」", "yuki_000004")]
VOICES = ("yuki_000003", "yuki_000004", "taku_000001")


def write_bgi_game(game: Path, voice: bytes = b"OggS") -> Path:
    """BGI.exe, data01701.arc (scripts sc01 + sc02: 4 lines, 3 voiced) and voice/*.ogg in ``game`` (``voice`` = their bytes)."""
    (game / "voice").mkdir(parents=True, exist_ok=True)
    (game / "BGI.exe").write_bytes(b"MZ")
    # data01701.arc has a chapter preset upstream (a Japanese folder name); the engine passes --no-presets
    (game / "data01701.arc").write_bytes(arc({"sc01": dsc(bss(SC01)), "sc02": dsc(bss(SC02))}))
    for v in VOICES:
        (game / "voice" / f"{v}.ogg").write_bytes(voice)
    return game

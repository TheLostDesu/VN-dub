"""Artemis engine: ``*.pfs`` archives (pfs-rs / pfs_unpacker), Lua ``.ast`` scenarios.

``.ast`` files are Lua tables, e.g. (V2, ``astver = 2.0``)::

    ast = {
      block_00000 = {
        {"vo", file="sak_0001", ch="sakura"},
        text = { ja = { { name = {"さくら"}, "こんにちは", }, }, vo = { {"vo", file="sak_0001"} } },
        linknext = "block_00001",
      },
    }

A small Lua-table reader turns the file into Python data; every block is
scanned for text under ``text.<lang>`` (``ja`` first), a ``name`` entry, and
voice references: any ``{"vo"|"voice", file=...}`` entry or ``vo``/``voice``
key anywhere in the block (voice key layout is UNVERIFIED across games).
Compiled ``.asb``/``.iet`` scripts are not supported (use VNTextPatch for text).
Packing: ``pfs-rs``/``pfs_unpacker`` can build v6/v8 packs; vndub writes loose
files + the command line.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vndub.engines.base import (
    BaseEngine,
    DetectResult,
    ExtractResult,
    LineBuilder,
    ParseResult,
    VoiceResolver,
    bind_line,
    build_characters,
    copy_loose,
    find_audio,
    read_text_any,
    write_install,
)
from vndub.manifest.io import AudioIndexRecord, FinalIndexRecord

if TYPE_CHECKING:
    from vndub.stages.common import StageContext


class LuaError(ValueError):
    pass


class _Lua:
    """Minimal Lua table-constructor reader (strings, numbers, booleans, nil, tables)."""

    TOKEN = re.compile(
        r"""\s*(?:(--\[\[.*?\]\]|--[^\n]*)|(\[\[.*?\]\])|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)|([A-Za-z_][\w]*)|(\[|\]|\{|\}|=|,|;))""",
        re.S,
    )

    def __init__(self, text: str) -> None:
        self.toks: list[tuple[str, Any]] = []
        pos = 0
        while pos < len(text):
            m = self.TOKEN.match(text, pos)
            if not m:
                if text[pos:].strip() == "":
                    break
                pos += 1  # skip unknown chars (operators in expressions we do not need)
                continue
            pos = m.end()
            if m.group(1):
                continue
            if m.group(2):
                self.toks.append(("str", m.group(2)[2:-2]))
            elif m.group(3):
                self.toks.append(("str", _unescape(m.group(3)[1:-1])))
            elif m.group(4):
                n = m.group(4)
                self.toks.append(("num", float(n) if "." in n or "e" in n.lower() else int(n)))
            elif m.group(5):
                self.toks.append(("name", m.group(5)))
            else:
                self.toks.append(("sym", m.group(6)))
        self.i = 0

    def peek(self, k: int = 0) -> tuple[str, Any] | None:
        j = self.i + k
        return self.toks[j] if j < len(self.toks) else None

    def take(self) -> tuple[str, Any]:
        t = self.peek()
        if t is None:
            raise LuaError("unexpected end of input")
        self.i += 1
        return t

    def value(self) -> Any:
        t = self.take()
        if t[0] in ("str", "num"):
            return t[1]
        if t[0] == "name":
            return {"true": True, "false": False, "nil": None}.get(t[1], t[1])
        if t == ("sym", "{"):
            return self.table()
        raise LuaError(f"unexpected token {t}")

    def table(self) -> Any:
        arr: list[Any] = []
        d: dict[Any, Any] = {}
        while True:
            t = self.peek()
            if t is None:
                raise LuaError("unterminated table")
            if t == ("sym", "}"):
                self.i += 1
                break
            if t in (("sym", ","), ("sym", ";")):
                self.i += 1
                continue
            if t[0] == "name" and self.peek(1) == ("sym", "="):
                self.i += 2
                d[t[1]] = self.value()
            elif t == ("sym", "["):
                self.i += 1
                k = self.value()
                if self.take() != ("sym", "]") or self.take() != ("sym", "="):
                    raise LuaError("bad [key] = value")
                d[k] = self.value()
            else:
                arr.append(self.value())
        if d and arr:
            d["__array__"] = arr
            return d
        return d if d else arr

    def chunk(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        while self.peek() is not None:
            t = self.take()
            if t[0] == "name" and self.peek() == ("sym", "="):
                self.i += 1
                out[t[1]] = self.value()
            elif t[0] == "name" and t[1] == "local":
                continue
        return out


def _unescape(s: str) -> str:
    return re.sub(r"\\(n|t|\\|\"|')", lambda m: {"n": "\n", "t": "\t"}.get(m.group(1), m.group(1)), s)


def parse_lua(text: str) -> dict[str, Any]:
    return _Lua(text).chunk()


def _texts(node: Any) -> list[str]:
    """Plain strings in a text node (skipping tag tables like {"rt2"} and name tables)."""
    out: list[str] = []
    if isinstance(node, str):
        out.append(node)
    elif isinstance(node, list):
        if node and isinstance(node[0], str) and len(node) <= 3 and re.fullmatch(r"[a-z_][\w]*", node[0] or ""):
            return out  # a tag like {"rt2"} / {"ruby", text=...}
        for x in node:
            out.extend(_texts(x))
    elif isinstance(node, dict):
        for k, v in node.items():
            if k in ("name", "vo", "voice"):
                continue
            if k == "__array__" or isinstance(k, int):
                out.extend(_texts(v))
    return out


def _name(node: Any) -> str:
    if isinstance(node, dict):
        if "name" in node:
            n = node["name"]
            if isinstance(n, list):
                return next((x for x in n if isinstance(x, str)), "")
            if isinstance(n, dict):
                for k in ("name", "ja", "__array__"):
                    v = n.get(k)
                    if isinstance(v, str):
                        return v
                    if isinstance(v, list) and v and isinstance(v[0], str):
                        return v[0]
            if isinstance(n, str):
                return n
        for v in node.values():
            r = _name(v)
            if r:
                return r
    elif isinstance(node, list):
        for v in node:
            r = _name(v)
            if r:
                return r
    return ""


def _voices(node: Any) -> list[str]:
    out: list[str] = []
    if isinstance(node, dict):
        arr = node.get("__array__")
        if isinstance(arr, list) and arr and arr[0] in ("vo", "voice") and isinstance(node.get("file"), str):
            out.append(node["file"])
        for k, v in node.items():
            if k in ("vo", "voice") and isinstance(v, str):
                out.append(v)
            else:
                out.extend(_voices(v))
    elif isinstance(node, list):
        for v in node:
            out.extend(_voices(v))
    return out


def parse_ast(text: str, lang: str = "ja") -> list[dict[str, Any]]:
    data = parse_lua(text)
    ast = data.get("ast") if isinstance(data.get("ast"), dict) else data
    rows = []
    for key in sorted((k for k in ast if isinstance(k, str) and k.startswith("block")), key=lambda k: k):
        block = ast[key]
        if not isinstance(block, dict):
            continue
        tnode = block.get("text")
        if not isinstance(tnode, dict):
            continue
        langnode = tnode.get(lang) or tnode.get("ja") or next((v for k, v in tnode.items() if k not in ("vo", "voice")), None)
        if langnode is None:
            continue
        text_s = "".join(_texts(langnode)).strip()
        if not text_s:
            continue
        rows.append({"block": key, "name": _name(langnode), "text": text_s, "voices": _voices(block)})
    return rows


class ArtemisEngine(BaseEngine):
    name = "artemis"
    out_name = "artemis"

    def detect(self, game_dir: Path) -> DetectResult | None:
        pfs = sorted(list(game_dir.glob("*.pfs")) + list(game_dir.glob("*.pfs.0*")))
        asts = list(game_dir.rglob("*.ast"))[:3]
        if not pfs and not asts:
            return None
        return DetectResult("artemis", 0.85 if pfs else 0.6, archives=[p.name for p in pfs], notes=["pfs archives: extract runs pfs-rs (tools.yaml: pfs_unpacker)"] if pfs else [])

    def extract(self, ctx: "StageContext") -> ExtractResult:
        game = ctx.game_dir
        raw = ctx.paths.raw / "artemis"
        warnings: list[str] = []
        asts = list(game.rglob("*.ast"))
        pfs = sorted(game.glob("*.pfs"))
        if pfs and not asts:
            if ctx.tools.resolve("pfs_unpacker") is None:
                warnings.append("pfs-rs/pfs_unpacker not configured (tools.yaml: pfs_unpacker)")
            else:
                for a in pfs:
                    r = ctx.tools.run("pfs_unpacker", ["unpack", str(a), str(raw)], check=False)
                    if r.returncode != 0:
                        warnings.append(f"pfs unpack failed on {a.name}: {r.stderr.strip()[-200:]}")
            asts = list(raw.rglob("*.ast")) if raw.exists() else []
        root = raw if raw.exists() and asts and raw in asts[0].parents else game
        return ExtractResult(script_files=sorted(asts), voice_root=root, voice_files=find_audio(root), warnings=warnings)

    def parse(self, ctx: "StageContext", extracted: ExtractResult, audio: list[AudioIndexRecord]) -> ParseResult:
        resolver = VoiceResolver(audio)
        lb = LineBuilder(ctx.cfg.parse.narrator_names)
        notes = []
        for p in extracted.script_files:
            try:
                rows = parse_ast(read_text_any(p), ctx.cfg.project.src_lang)
            except LuaError as e:
                notes.append(f"{p.name}: {e}")
                continue
            for row in rows:
                rec = lb.add(p.stem, row["text"], speaker_raw=row["name"], source_file=p.name, raw=row["block"])
                if row["voices"]:
                    bind_line(rec, resolver.resolve(row["voices"][0]), row["voices"][0])
        return ParseResult(lb.lines, build_characters(lb.lines), notes)

    def pack(self, ctx: "StageContext", finals: list[FinalIndexRecord]) -> dict[str, Any]:
        out = ctx.paths.out_engine("artemis")
        n = copy_loose(ctx, finals, out / "files")
        write_install(out, "artemis", [
            "Build a patch archive from `files/` with pfs-rs or pfs_unpacker (v6/v8), e.g. `pfs-rs pack files root.pfs.001`,",
            "name it so it loads after the original (Artemis reads root.pfs, root.pfs.000, .001 ... later wins), and copy it to the game dir.",
        ])
        return {"mode": "pfs_layout", "files": n, "out": str(out)}

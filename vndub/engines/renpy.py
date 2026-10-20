"""Ren'Py engine: ``.rpy`` scripts (``.rpyc`` via unrpyc, ``.rpa`` via unrpa).

Parsing covers what voice mapping needs:

* ``define x = Character("Name", voice_tag="tag")`` (also ``_("Name")``,
  ``Character(None)``/``kind=nvl``, ``DynamicCharacter``),
* labels (``label name:``, local ``.name``; ``hide`` labels and ``_``-labels
  do not change the translation label),
* say statements (``who [attrs] [@ tmp] "what" [nointeract] [id x] [(args)]``,
  narrator ``"what"``, string speakers ``"Name" "what"``, ``extend``),
* ``voice "file"`` (binds to the next say), ``voice sustain`` (the previous
  voice keeps playing over the next say: that line gets ``voice_sustain`` and
  the line owning the file gets ``sustain_head`` + ``sustain_lines``),
* ``menu:`` blocks (choice captions -> ``choices`` of a menu record),
* ``config.auto_voice`` (``define config.auto_voice = "voice/{id}.ogg"`` or
  inside ``init python``) -> file name from the dialogue identifier,
* ``dialogue.tab`` export from the launcher (``engine_opts.renpy.dialogue_tab``).

The dialogue identifier follows ``renpy/translation/__init__.py``: the md5 of
every statement in the translate block (``voice`` statements are translatable
and grouped with the following say) as ``get_code() + "\\r\\n"``, first 8 hex
digits, prefixed by the current translation label (dots -> underscores);
duplicates get ``_1``, ``_2`` ... Verified: the docs example
``e "Thank you for taking a look at the Ren'Py translation framework."``
under ``label start`` -> ``start_636ae3f5``.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import re
import shutil
from dataclasses import dataclass, field
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
    pick_voice_files,
    speaker_id_for,
    write_install,
)
from vndub.manifest.io import AudioIndexRecord, Character, FinalIndexRecord, LineRecord

if TYPE_CHECKING:
    from vndub.stages.common import StageContext

KEYWORDS = {
    "label", "menu", "jump", "call", "return", "scene", "show", "hide", "with", "play", "stop", "queue",
    "voice", "pause", "window", "define", "default", "init", "python", "if", "elif", "else", "while",
    "pass", "image", "transform", "screen", "style", "translate", "nvl", "camera", "layeredimage",
    "testcase", "at", "behind", "onlayer", "zorder", "expression", "rpy", "font",
}
# block statements whose bodies contain no dialogue we parse
SKIP_BLOCKS = re.compile(r"^(init\b.*python|python\b|screen\b|transform\b|style\b|translate\b|image\b.*:$|layeredimage\b|testcase\b|init\s+-?\d*\s*:$|init\s*:$)")

_STR = r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\''
SAY_RE = re.compile(
    rf"^(?P<who>(?:{_STR})|[A-Za-z_][\w\.]*)?(?P<attrs>(?:\s+(?!nointeract\b)(?!id\b)[A-Za-z_][\w\-]*)*)"
    rf"(?P<tmp>\s+@(?:\s+[A-Za-z_][\w\-]*)+)?\s*(?P<what>{_STR})(?P<rest>.*)$"
)
CHAR_DEF = re.compile(r"^define\s+(?P<var>[A-Za-z_][\w\.]*)\s*=\s*(?P<expr>(?:Dynamic)?Character\s*\(.*\))\s*$")
AUTO_VOICE = re.compile(r"(?:define\s+)?config\.auto_voice\s*=\s*(?P<val>.+?)\s*$")
VOICE_RE = re.compile(rf"^voice\s+(?P<arg>sustain\b|{_STR}|.+)$")
LABEL_RE = re.compile(r"^label\s+(?P<name>[\w\.]+)\s*(?:\((?P<params>[^)]*)\))?\s*(?P<hide>hide)?\s*:\s*$")
MENU_RE = re.compile(r"^menu(?:\s+(?P<name>[\w\.]+))?\s*(?:\([^)]*\))?\s*:\s*$")
CHOICE_RE = re.compile(rf"^(?P<cap>{_STR})(?:\s*\([^)]*\))?(?:\s+if\s+.+)?\s*:\s*$")


# ----------------------------------------------------------------------------
# string handling (mirrors renpy.lexer string processing and encode_say_string)
# ----------------------------------------------------------------------------


def dequote(raw: str) -> str:
    """Ren'Py string literal (with quotes) -> runtime string."""
    s = raw
    if s[:3] in ('"""', "'''") and len(s) >= 6:
        s = s[3:-3]
    else:
        s = s[1:-1]
    s = re.sub(r"[ \n]+", " ", s)

    def _dq(m: re.Match[str]) -> str:
        c = m.group(1)
        if c == "{":
            return "{{"
        if c == "[":
            return "[["
        if c == "%":
            return "%%"
        if c == "n":
            return "\n"
        if c[0] == "u":
            return chr(int(m.group(2), 16))
        return c

    return re.sub(r"\\(u([0-9a-fA-F]{1,4})|.)", _dq, s)


def encode_say_string(s: str) -> str:
    s = s.replace("\\", "\\\\")
    s = s.replace("\n", "\\n")
    s = s.replace('"', '\\"')
    s = re.sub(r"(?<= ) ", "\\ ", s)
    return '"' + s + '"'


def say_code(who: str | None, what: str, attributes: list[str] | None = None, temporary: list[str] | None = None,
             interact: bool = True, explicit_id: str | None = None, arguments: str = "") -> str:
    """``renpy.ast.Say.get_code()``."""
    rv: list[str] = []
    if who:
        rv.append(who)
    if attributes:
        rv.extend(attributes)
    if temporary:
        rv.append("@")
        rv.extend(temporary)
    rv.append(encode_say_string(what))
    if not interact:
        rv.append("nointeract")
    if explicit_id:
        rv.append("id")
        rv.append(explicit_id)
    if arguments:
        rv.append(arguments)
    return " ".join(rv)


def block_digest(codes: list[str]) -> str:
    md5 = hashlib.md5()
    for c in codes:
        md5.update((c + "\r\n").encode("utf-8"))
    return md5.hexdigest()[:8]


class IdAllocator:
    """``Translator.unique_identifier``: label_digest with _1, _2 suffixes on collision."""

    def __init__(self) -> None:
        self.used: set[str] = set()

    def allocate(self, label: str | None, digest: str) -> str:
        base = digest if label is None else label.replace(".", "_") + "_" + digest
        i, suffix = 0, ""
        while base + suffix in self.used:
            i += 1
            suffix = f"_{i}"
        ident = base + suffix
        self.used.add(ident)
        return ident


def dialogue_identifier(label: str | None, codes: list[str], alloc: IdAllocator | None = None) -> str:
    return (alloc or IdAllocator()).allocate(label, block_digest(codes))


def strip_text_tags(s: str) -> str:
    s = re.sub(r"\{\{", "\x00", s)
    s = re.sub(r"\{[^{}]*\}", "", s)
    s = s.replace("\x00", "{").replace("[[", "[")
    return s.replace("%%", "%").strip()


# ----------------------------------------------------------------------------
# parser
# ----------------------------------------------------------------------------


@dataclass
class CharDef:
    var: str
    name: str | None
    voice_tag: str | None = None
    kind: str = ""
    dynamic: bool = False


def parse_character_expr(var: str, expr: str) -> CharDef:
    try:
        node = ast.parse(expr.strip(), mode="eval").body
    except SyntaxError:
        m = re.search(r"\(\s*(?:_\()?\s*(['\"])(.*?)\1", expr)
        vt = re.search(r"voice_tag\s*=\s*(['\"])(.*?)\1", expr)
        return CharDef(var, m.group(2) if m else None, vt.group(2) if vt else None)
    assert isinstance(node, ast.Call)
    fname = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
    name: str | None = None
    if node.args:
        a = node.args[0]
        if isinstance(a, ast.Call) and getattr(a.func, "id", "") in ("_", "__") and a.args:
            a = a.args[0]
        if isinstance(a, ast.Constant) and isinstance(a.value, str):
            name = a.value
    kw = {k.arg: k.value for k in node.keywords if k.arg}
    vt = kw.get("voice_tag")
    kind = kw.get("kind")
    return CharDef(
        var=var,
        name=name,
        voice_tag=vt.value if isinstance(vt, ast.Constant) else None,
        kind=getattr(kind, "id", "") if kind is not None else "",
        dynamic=fname == "DynamicCharacter",
    )


@dataclass
class _Pending:
    voice: str | None = None
    voice_code: str | None = None
    sustain: bool = False
    sustain_code: str | None = None


@dataclass
class RpyParse:
    chars: dict[str, CharDef] = field(default_factory=dict)
    auto_voice: str | None = None
    says: list[dict[str, Any]] = field(default_factory=list)  # in script order


def _strip_comment(line: str) -> str:
    out, q = [], None
    i = 0
    while i < len(line):
        c = line[i]
        if q:
            out.append(c)
            if c == "\\" and i + 1 < len(line):
                out.append(line[i + 1])
                i += 2
                continue
            if c == q:
                q = None
        else:
            if c in "\"'":
                q = c
            elif c == "#":
                break
            out.append(c)
        i += 1
    return "".join(out).rstrip()


def _logical_lines(text: str) -> list[tuple[int, int, str]]:
    """(line_no, indent, content) with comments removed and bracket/quote continuations joined."""
    out = []
    raw = text.splitlines()
    i = 0
    while i < len(raw):
        line = raw[i]
        no = i + 1
        content = _strip_comment(line)
        stripped = content.strip()
        if not stripped:
            i += 1
            continue
        indent = len(content) - len(content.lstrip(" \t"))
        # join continuation lines while brackets are open or a string is unterminated
        while (stripped.count("(") > stripped.count(")") or stripped.count("[") > stripped.count("]") or _open_quote(stripped)) and i + 1 < len(raw):
            i += 1
            stripped += "\n" + _strip_comment(raw[i]).strip() if _open_quote(stripped) else " " + _strip_comment(raw[i]).strip()
        out.append((no, indent, stripped))
        i += 1
    return out


def _open_quote(s: str) -> bool:
    q = None
    i = 0
    while i < len(s):
        c = s[i]
        if q:
            if c == "\\":
                i += 2
                continue
            if c == q:
                q = None
        elif c in "\"'":
            q = c
        i += 1
    return q is not None


def parse_rpy_text(text: str, filename: str, state: RpyParse, alloc: IdAllocator, ctx_label: list[str | None], only_say: bool = False) -> None:
    """Parse one .rpy file into ``state`` (characters, auto_voice, says)."""
    lines = _logical_lines(text)
    skip_indent: int | None = None
    menu_stack: list[tuple[int, dict[str, Any]]] = []
    pending = _Pending()
    for no, indent, s in lines:
        if skip_indent is not None:
            if indent > skip_indent:
                m = AUTO_VOICE.search(s)
                if m:
                    state.auto_voice = _literal(m.group("val"))
                continue
            skip_indent = None
        while menu_stack and indent <= menu_stack[-1][0]:
            menu_stack.pop()
        m = CHAR_DEF.match(s)
        if m:
            state.chars[m.group("var")] = parse_character_expr(m.group("var"), m.group("expr"))
            continue
        m = AUTO_VOICE.match(s)
        if m and (s.startswith("define") or s.startswith("config.") or s.startswith("$")):
            state.auto_voice = _literal(m.group("val"))
            continue
        if s.startswith("$"):
            m = AUTO_VOICE.search(s)
            if m:
                state.auto_voice = _literal(m.group("val"))
            continue
        if SKIP_BLOCKS.match(s) and s.endswith(":"):
            skip_indent = indent
            continue
        m = LABEL_RE.match(s)
        if m:
            name = m.group("name")
            if name.startswith(".") and ctx_label[1]:
                name = f"{ctx_label[1]}{name}"
            elif "." not in name:
                ctx_label[1] = name
            if not (m.group("hide") or name.split(".")[-1].startswith("_")):
                ctx_label[0] = name
            state.says.append({"kind": "label", "label": name, "file": filename, "line": no})
            continue
        m = MENU_RE.match(s)
        if m:
            rec = {"kind": "menu", "choices": [], "file": filename, "line": no, "label": ctx_label[0]}
            state.says.append(rec)
            menu_stack.append((indent, rec))
            continue
        if menu_stack and indent > menu_stack[-1][0]:
            m = CHOICE_RE.match(s)
            if m:
                menu_stack[-1][1]["choices"].append(strip_text_tags(dequote(m.group("cap"))))
                state.says.append({"kind": "choice", "caption": strip_text_tags(dequote(m.group("cap"))), "file": filename, "line": no})
                continue
        m = VOICE_RE.match(s)
        if m:
            arg = m.group("arg").strip()
            if arg == "sustain":
                pending.sustain = True
                pending.sustain_code = s
            elif arg[0] in "\"'":
                pending.voice = dequote(arg)
                pending.voice_code = s
            else:
                pending.voice = None
                pending.voice_code = s  # expression voice: unresolvable statically
            continue
        first = s.split(None, 1)[0].rstrip(":")
        if first in KEYWORDS and first != "extend":
            continue
        m = SAY_RE.match(s)
        if not m:
            continue
        who = m.group("who")
        rest = m.group("rest").strip()
        if rest.endswith(":"):
            continue  # menu choice outside a recognised menu, or say with a block
        attrs = (m.group("attrs") or "").split()
        tmp = (m.group("tmp") or "").replace("@", " ").split()
        interact = True
        explicit_id = None
        args = ""
        toks = rest
        if toks.startswith("nointeract"):
            interact = False
            toks = toks[len("nointeract"):].strip()
        mid = re.match(r"^id\s+([\w\.]+)\s*(.*)$", toks)
        if mid:
            explicit_id = mid.group(1)
            toks = mid.group(2).strip()
        if toks.startswith("("):
            args = toks
        what = dequote(m.group("what"))
        codes = []
        if not only_say:
            if pending.voice_code:
                codes.append(pending.voice_code)
            if pending.sustain_code:
                codes.append(pending.sustain_code)
        codes.append(say_code(who, what, attrs, tmp, interact, explicit_id, args))
        ident = explicit_id or alloc.allocate(ctx_label[0], block_digest(codes))
        state.says.append(
            {
                "kind": "say",
                "who": who,
                "what": what,
                "voice": pending.voice,
                "voice_expr": bool(pending.voice_code) and pending.voice is None,
                "sustain": pending.sustain,
                "id": ident,
                "label": ctx_label[0],
                "file": filename,
                "line": no,
                "raw": s,
            }
        )
        pending = _Pending()


def _literal(expr: str) -> str | None:
    try:
        v = ast.literal_eval(expr.strip())
        return v if isinstance(v, str) else None
    except (ValueError, SyntaxError):
        return None


def parse_rpy_files(files: list[Path], only_say: bool = False) -> RpyParse:
    """Parse files in Ren'Py load order (sorted by path, as the launcher does)."""
    state = RpyParse()
    alloc = IdAllocator()
    for f in sorted(files, key=lambda p: p.as_posix().lower()):
        from vndub.engines.base import read_text_any

        ctx_label: list[str | None] = [None, None]  # [translation label, last global label]
        parse_rpy_text(read_text_any(f), f.name, state, alloc, ctx_label, only_say)
    return state


def read_dialogue_tab(path: Path) -> dict[str, dict[str, str]]:
    """Launcher 'Extract Dialogue' TSV: Identifier, Character, Dialogue, Filename, Line Number."""
    out: dict[str, dict[str, str]] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            ident = row.get("Identifier") or row.get("identifier")
            if ident:
                out[ident] = row
    return out


class RenpyEngine(BaseEngine):
    name = "renpy"
    out_name = "renpy"

    def detect(self, game_dir: Path) -> DetectResult | None:
        game = game_dir / "game"
        if not game.is_dir():
            return None
        rpa = sorted(game.glob("*.rpa"))
        rpy = list(game.rglob("*.rpy"))
        rpyc = list(game.rglob("*.rpyc"))
        if not (rpa or rpy or rpyc):
            return None
        conf = 0.75 + (0.2 if (game_dir / "renpy").is_dir() else 0.0) + (0.05 if rpy else 0.0)
        notes = []
        if rpyc and not rpy:
            notes.append("only compiled .rpyc scripts: extract will run unrpyc (tools.yaml: unrpyc)")
        if rpa:
            notes.append(f"{len(rpa)} .rpa archive(s): extract will run unrpa")
        sample = next(iter(pick_voice_files(game, ("voice", "voices", "audio/voice"))), None)
        return DetectResult("renpy", min(conf, 1.0), archives=[p.name for p in rpa], notes=notes, voice_sample=sample)

    def extract(self, ctx: "StageContext") -> ExtractResult:
        game = ctx.game_dir / "game"
        raw = ctx.paths.raw / "renpy"
        warnings: list[str] = []
        roots = [game]
        rpas = sorted(game.glob("*.rpa"))
        if rpas:
            dest = raw / "rpa"
            exe = ctx.tools.resolve("unrpa")
            if exe is None:
                warnings.append("unrpa not found: .rpa archives were not extracted (pip install unrpa; tools.yaml: unrpa)")
            else:
                dest.mkdir(parents=True, exist_ok=True)
                for a in rpas:
                    r = ctx.tools.run("unrpa", ["-mp", str(dest), str(a)], check=False)
                    if r.returncode != 0:
                        warnings.append(f"unrpa failed on {a.name}: {r.stderr.strip()[-300:]}")
                roots.append(dest)
        scripts: list[Path] = []
        for root in roots:
            scripts += [p for p in root.rglob("*.rpy")]
        rpyc_only = []
        for root in roots:
            for c in root.rglob("*.rpyc"):
                if not c.with_suffix(".rpy").exists():
                    rpyc_only.append(c)
        if rpyc_only:
            dest = raw / "decompiled"
            exe = ctx.tools.resolve("unrpyc")
            if exe is None:
                warnings.append(f"{len(rpyc_only)} .rpyc without .rpy and unrpyc is not configured (tools.yaml: unrpyc, v2.x): those scripts are not parsed")
            else:
                for c in rpyc_only:
                    out = dest / c.name
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(c, out)
                    args = [str(out)] if not str(exe).endswith(".py") else [str(out)]
                    r = ctx.tools.run("unrpyc", args, check=False)
                    if r.returncode != 0 or not out.with_suffix(".rpy").exists():
                        warnings.append(f"unrpyc failed on {c.name}: {r.stderr.strip()[-300:]}")
                    else:
                        scripts.append(out.with_suffix(".rpy"))
        scripts = [s for s in scripts if "/renpy/" not in s.as_posix() and not s.name.startswith("00")]
        voice_dirs = list(ctx.cfg.extract.voice_dirs) + ["voices", "audio/voice", "audio/voices"]
        voices: list[Path] = []
        seen: set[str] = set()
        for root in roots:
            for v in pick_voice_files(root, voice_dirs):
                key = v.relative_to(root).as_posix().lower()
                if key not in seen:
                    seen.add(key)
                    voices.append(v)
        return ExtractResult(script_files=sorted(scripts), voice_root=game, voice_files=voices, voice_roots=roots[1:], warnings=warnings, notes=[f"{len(scripts)} scripts"])

    def parse(self, ctx: "StageContext", extracted: ExtractResult, audio: list[AudioIndexRecord]) -> ParseResult:
        opts = ctx.cfg.parse.engine_opts.renpy
        state = parse_rpy_files(extracted.script_files, only_say=opts.tlid_only_considers_say)
        auto_voice = opts.auto_voice or state.auto_voice
        tab = read_dialogue_tab(Path(opts.dialogue_tab)) if opts.dialogue_tab else {}
        tab_by_pos: dict[tuple[str, int], str] = {}
        for ident, row in tab.items():
            try:
                tab_by_pos[(Path((row.get("Filename") or "").replace("\\", "/")).name, int(row.get("Line Number") or 0))] = ident
            except ValueError:
                continue
        resolver = VoiceResolver(audio)
        lb = LineBuilder(ctx.cfg.parse.narrator_names)
        notes: list[str] = []
        last_voiced: LineRecord | None = None
        last_speaker: tuple[str, str] = ("", "narrator")
        menu_rec: LineRecord | None = None
        scene = "start"
        auto_hits = 0
        for st in state.says:
            kind = st["kind"]
            if kind == "label":
                scene = st["label"].replace(".", "_")
                continue
            if kind == "menu":
                menu_rec = lb.add(scene, "", speaker_id="narrator", choices=[], flags=["menu"], source_file=st["file"], source_line=st["line"])
                menu_rec.choices = st["choices"]
                continue
            if kind == "choice":
                continue
            who = st["who"]
            speaker_raw, sid = "", "narrator"
            if who and who[0] in "\"'":
                speaker_raw = dequote(who)
                sid = speaker_id_for(speaker_raw, ctx.cfg.parse.narrator_names)
            elif who == "extend":
                speaker_raw, sid = last_speaker
            elif who:
                cd = state.chars.get(who)
                if cd is None:
                    speaker_raw, sid = who, speaker_id_for(who, ctx.cfg.parse.narrator_names)
                elif cd.name is None:
                    speaker_raw, sid = "", "narrator"
                else:
                    speaker_raw = cd.name
                    sid = speaker_id_for(cd.voice_tag) if cd.voice_tag else speaker_id_for(cd.name, ctx.cfg.parse.narrator_names)
            last_speaker = (speaker_raw, sid)
            text = strip_text_tags(st["what"])
            flags = []
            if who == "extend":
                flags.append("extend")
            rec = lb.add(scene, text, speaker_raw=speaker_raw, speaker_id=sid, raw=st["raw"], source_file=st["file"], source_line=st["line"], flags=flags)
            rec.dialogue_id = st["id"]  # type: ignore[attr-defined]
            ref = st["voice"]
            if st["sustain"] and last_voiced is not None and not ref:
                rec.flags.append("voice_sustain")
                rec.has_voice = False
                rec.voice_file = None
                head = last_voiced
                if "sustain_head" not in head.flags:
                    head.flags.append("sustain_head")
                sl = list(getattr(head, "sustain_lines", None) or [])
                sl.append(rec.line_id)
                head.sustain_lines = sl  # type: ignore[attr-defined]
                rec.sustain_of = head.line_id  # type: ignore[attr-defined]
                continue
            if not ref and st.get("voice_expr"):
                rec.flags.append("voice_expression")
            tab_id = tab_by_pos.get((st["file"], st["line"]))
            if tab_id:
                if tab_id != st["id"]:
                    rec.flags.append("dialogue_tab_id")  # launcher export overrides our computed id
                st["id"] = tab_id
                rec.dialogue_id = tab_id  # type: ignore[attr-defined]
            if not ref and auto_voice:
                cand = auto_voice.replace("{id}", st["id"])
                if resolver.resolve(cand) is not None:
                    ref = cand
                    auto_hits += 1
                    rec.flags.append("auto_voice")
            if ref:
                bind_line(rec, resolver.resolve(ref), ref)
            last_voiced = rec if rec.has_voice else None
        if auto_voice:
            notes.append(f"config.auto_voice={auto_voice!r}: {auto_hits} lines matched by dialogue identifier")
            notes += self._auto_voice_fallback(lb.lines, audio, auto_voice, resolver)
        if menu_rec is not None:
            pass
        chars = build_characters(lb.lines)
        for cd in state.chars.values():
            if cd.name is None:
                continue
            sid = speaker_id_for(cd.voice_tag) if cd.voice_tag else speaker_id_for(cd.name)
            c = chars.get(sid)
            if c is None:
                c = Character(names={"src": cd.name, "tgt": ""})
                chars[sid] = c
            c.names.setdefault("src", cd.name)
            if not c.names.get("src"):
                c.names["src"] = cd.name
        return ParseResult(lb.lines, chars, notes)

    @staticmethod
    def _auto_voice_fallback(lines: list[LineRecord], audio: list[AudioIndexRecord], pattern: str, resolver: VoiceResolver) -> list[str]:
        """Lines without a voice under auto_voice: match unclaimed files under the pattern's
        directory by order within the label (files sorted by name). Flags ``auto_voice_order``."""
        prefix = pattern.split("{id}")[0]
        claimed = {l.audio_id for l in lines if l.audio_id}
        pool = [a for a in audio if a.id not in claimed and a.relpath.lower().startswith(prefix.lower())]
        if not pool:
            return []
        by_label: dict[str, list[AudioIndexRecord]] = {}
        for a in pool:
            stem = Path(a.relpath).stem
            lab = re.sub(r"_[0-9a-f]{8}(?:_\d+)?$", "", stem)
            by_label.setdefault(lab, []).append(a)
        n = 0
        for l in lines:
            if l.has_voice or l.speaker_id == "narrator" or "voice_sustain" in l.flags or "menu" in l.flags:
                continue
            cands = by_label.get(l.scene)
            if cands:
                a = cands.pop(0)
                bind_line(l, a, a.relpath)
                l.flags.append("auto_voice_order")
                n += 1
        return [f"auto_voice fallback: {n} lines bound by script order within label (unverified ids)"] if n else []

    def pack(self, ctx: "StageContext", finals: list[FinalIndexRecord]) -> dict[str, Any]:
        out = ctx.paths.out_engine("renpy") / "game"
        n = copy_loose(ctx, finals, out)
        write_install(
            ctx.paths.out_engine("renpy"),
            "renpy",
            [
                f"Copy the `game` folder from here over the game's `game` folder ({ctx.game_dir / 'game'}).",
                "Ren'Py loads loose files before .rpa archives, so the new voices override the originals.",
                "If the game sets `config.force_archives = True`, loose files are ignored: repack the voices into an .rpa (rpatool -c) named so it sorts after the original archives.",
            ],
        )
        return {"mode": "loose", "files": n, "out": str(out)}

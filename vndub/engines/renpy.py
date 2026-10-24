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

Where a mistake silently silences a line (the file is named by an identifier Ren'Py never asks for), the details follow
``renpy/translation/__init__.py`` (``Restructurer``), ``renpy/parser.py`` and ``renpy/lexer.py`` of Ren'Py master:

* every file is restructured on its own: the translation label and the ``_1`` / ``_2`` suffixes start over per file;
* a ``Label`` node moves the translation label unless it is ``hide`` or its name starts with ``_`` (those only set
  an *alternate* label, the primary identifier is unchanged): ``label``, ``menu name:`` and ``call x from name`` all
  make one;
* a menu's own say (the line before the choices, ``e "Which?"``) is a ``nointeract`` say: its code ends in
  ``nointeract``; a bare string in a menu is a caption, not a say statement;
* a triple-quoted string is cut at blank lines and every paragraph is its own say (own click, own identifier);
* ``get_code`` ends with ``nointeract``, ``id x``, the argument list and the ``with`` clause.
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
# triple-quoted dialogue ("monologue mode"): one say statement per paragraph, see triple_paragraphs
_TSTR = r'"""(?:[^\\"]|\\.|"(?!""))*"""|\'\'\'(?:[^\\\']|\\.|\'(?!\'\'))*\'\'\''
SAY_RE = re.compile(
    # a string speaker must be followed by white space, or ``"""text"""`` would read as the empty speaker ``""`` + ``"text"``
    rf"^(?P<who>(?:{_STR})(?=\s)|[A-Za-z_][\w\.]*)?(?P<attrs>(?:\s+(?!nointeract\b)(?!id\b)(?!with\b)-?[A-Za-z_][\w\-]*)*)"
    rf"(?P<tmp>\s+@(?:\s+-?[A-Za-z_][\w\-]*)+)?\s*(?P<what>{_TSTR}|{_STR})(?P<rest>.*)$",
    re.S,
)
CALL_FROM_RE = re.compile(r"^call\b.*?\sfrom\s+(?P<name>\.?[A-Za-z_][\w\.]*)\s*$")
CHAR_DEF = re.compile(r"^define\s+(?P<var>[A-Za-z_][\w\.]*)\s*=\s*(?P<expr>(?:Dynamic)?Character\s*\(.*\))\s*$")
AUTO_VOICE = re.compile(r"(?:define\s+)?config\.auto_voice\s*=\s*(?P<val>.+?)\s*$")
VOICE_RE = re.compile(rf"^voice\s+(?P<arg>sustain\b|{_STR}|.+)$")
LABEL_RE = re.compile(r"^label\s+(?P<name>[\w\.]+)\s*(?:\((?P<params>[^)]*)\))?\s*(?P<hide>hide)?\s*:\s*$")
MENU_RE = re.compile(r"^menu(?:\s+(?P<name>[\w\.]+))?\s*(?:\([^)]*\))?\s*:\s*$")
CHOICE_RE = re.compile(rf"^(?P<cap>{_STR})(?:\s*\([^)]*\))?(?:\s+if\s+.+)?\s*:\s*$")


# ----------------------------------------------------------------------------
# string handling (mirrors renpy.lexer string processing and encode_say_string)
# ----------------------------------------------------------------------------


def _unescape(m: re.Match[str]) -> str:
    """The lexer's ``dequote`` for one backslash escape."""
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


def dequote(raw: str) -> str:
    """Ren'Py string literal (with quotes) -> runtime string."""
    s = raw
    if s[:3] in ('"""', "'''") and len(s) >= 6:
        s = s[3:-3]
    else:
        s = s[1:-1]
    s = re.sub(r"[ \n]+", " ", s)
    return re.sub(r"\\(u([0-9a-fA-F]{1,4})|.)", _unescape, s)


def triple_paragraphs(raw: str) -> list[tuple[int, str]]:
    """``Lexer.triple_string`` (monologue mode): a triple-quoted dialogue string is cut at blank lines
    (``config.monologue_delimiter``, ``"\\n\\n"``) and every paragraph is its own say statement, with its own click and
    its own dialogue identifier. Returns ``[(line offset inside the string, text)]``, empty paragraphs dropped."""
    s = re.sub(r" *\n *", "\n", raw[3:-3])
    delim = "\n\n"
    out: list[tuple[int, str]] = []
    pos = 0
    for part in s.split(delim):
        lead = len(part) - len(part.lstrip())
        offset = s.count("\n", 0, pos) + part.count("\n", 0, lead)
        pos += len(part) + len(delim)
        part = part.strip()
        if not part:
            continue
        out.append((offset, re.sub(r"\\(u([0-9a-fA-F]{1,4})|.)", _unescape, re.sub(r"[ \n]+", " ", part))))
    return out


def encode_say_string(s: str) -> str:
    s = s.replace("\\", "\\\\")
    s = s.replace("\n", "\\n")
    s = s.replace('"', '\\"')
    s = re.sub(r"(?<= ) ", "\\ ", s)
    return '"' + s + '"'


def say_code(who: str | None, what: str, attributes: list[str] | None = None, temporary: list[str] | None = None,
             interact: bool = True, explicit_id: str | None = None, arguments: str = "", with_: str | None = None) -> str:
    """``renpy.ast.Say.get_code()``: speaker, attributes, ``@ temporary``, the escaped text, ``nointeract``, ``id x``,
    the argument list and the ``with`` clause last. (A menu's own say, the line before the choices, is a
    ``nointeract`` one: the menu provides the interaction.)"""
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
    if with_:
        rv.append("with")
        rv.append(with_)
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


def _scan(line: str, q: str | None, depth: int) -> tuple[str, str | None, int]:
    """One physical line from the lexer state (``q``: the delimiter of the string still open, ``depth``: open brackets).
    Returns the line without its trailing comment and the state at its end. A ``#`` inside a string is text."""
    out: list[str] = []
    i, n = 0, len(line)
    while i < n:
        c = line[i]
        if q:
            if c == "\\" and i + 1 < n:
                out.append(line[i : i + 2])
                i += 2
            elif line.startswith(q, i):
                out.append(q)
                i += len(q)
                q = None
            else:
                out.append(c)
                i += 1
            continue
        if c in "\"'":
            q = c * 3 if line.startswith(c * 3, i) else c
            out.append(q)
            i += len(q)
            continue
        if c == "#":
            break
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        out.append(c)
        i += 1
    return "".join(out), q, depth


def _logical_lines(text: str) -> list[tuple[int, int, str]]:
    """(line_no, indent, content) with comments removed and bracket / string continuations joined (a string may run over
    several lines: triple-quoted dialogue keeps its blank lines, they separate its paragraphs)."""
    out = []
    raw = text.splitlines()
    i = 0
    while i < len(raw):
        no = i + 1
        content, q, depth = _scan(raw[i], None, 0)
        content = content.rstrip()
        stripped = content.strip()
        if not stripped:
            i += 1
            continue
        indent = len(content) - len(content.lstrip(" \t"))
        # join continuation lines while brackets are open or a string is unterminated
        while (q or depth > 0) and i + 1 < len(raw):
            i += 1
            more, q_next, depth = _scan(raw[i], q, depth)
            stripped += ("\n" if q else " ") + more.strip()
            q = q_next
        out.append((no, indent, stripped))
        i += 1
    return out


def _split_top(s: str) -> list[str]:
    """Split on commas outside brackets and strings."""
    parts: list[str] = []
    cur: list[str] = []
    depth, q, i = 0, None, 0
    while i < len(s):
        c = s[i]
        if q:
            cur.append(c)
            if c == "\\" and i + 1 < len(s):
                cur.append(s[i + 1])
                i += 1
            elif c == q:
                q = None
        elif c in "\"'":
            q = c
            cur.append(c)
        else:
            if c in "([{":
                depth += 1
            elif c in ")]}":
                depth -= 1
            if c == "," and depth == 0:
                parts.append("".join(cur))
                cur = []
            else:
                cur.append(c)
        i += 1
    parts.append("".join(cur))
    return parts


def normalize_arguments(text: str) -> str:
    """``ArgumentInfo.get_code()``: ``( a , key = b )`` -> ``(a, key=b)`` (the source text of each expression is kept)."""
    inner = text.strip()[1:-1]
    items = []
    for part in _split_top(inner):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^([A-Za-z_]\w*)\s*=(?!=)\s*(.*)$", part, re.S)
        items.append(f"{m.group(1)}={m.group(2).strip()}" if m else part)
    return "(" + ", ".join(items) + ")"


_WITH_EXPR = r"[A-Za-z_][\w\.]*(?:\([^()]*\))?"


def parse_say_tail(rest: str) -> tuple[bool, str | None, str, str | None]:
    """The clauses after a say statement's text, in any order (``renpy.parser.finish_say``): ``nointeract``,
    ``id name``, ``with expr``, one ``(arguments)``. Returns ``(interact, explicit id, normalized arguments, with)``.
    Something else (a ``with`` expression too complex for the pattern) ends the scan: the line stays a say statement,
    only the clauses after that point are not part of its code."""
    interact, ident, args, with_ = True, None, "", None
    r = rest.strip()
    while r:
        m = re.match(r"nointeract\b\s*", r)
        if m:
            interact, r = False, r[m.end():]
            continue
        m = re.match(r"id\s+([^\W\d]\w*)\s*", r)
        if m:
            ident, r = m.group(1), r[m.end():]
            continue
        m = re.match(r"with\s+(" + _WITH_EXPR + r")\s*", r)
        if m:
            with_, r = m.group(1), r[m.end():]
            continue
        if r.startswith("("):
            depth, q, end = 0, None, -1
            for k, c in enumerate(r):
                if q:
                    if c == q:
                        q = None
                elif c in "\"'":
                    q = c
                elif c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                    if depth == 0:
                        end = k
                        break
            if end < 0:
                break
            args, r = normalize_arguments(r[: end + 1]), r[end + 1:].lstrip()
            continue
        break
    return interact, ident, args, with_


def enter_label(name: str, hide: bool, ctx_label: list[str | None], declare: bool = True) -> str:
    """A ``Label`` node is met: returns the full label name and moves the translation label that prefixes the dialogue
    identifiers from here (``Restructurer.callback``: a hidden label, or one whose name starts with ``_``, does not).
    ``declare`` is for ``label`` / ``menu name`` (``Lexer.set_global_label``: the part before the first dot becomes the
    global label local ``.names`` hang on); the label a ``call ... from name`` makes leaves the global label alone."""
    if name.startswith("."):
        name = f"{ctx_label[1]}{name}" if ctx_label[1] else name
    elif declare:
        ctx_label[1] = name.split(".")[0]
    if not hide and not name.startswith("_"):
        ctx_label[0] = name
    return name


def parse_rpy_text(text: str, filename: str, state: RpyParse, alloc: IdAllocator, ctx_label: list[str | None], only_say: bool = False) -> None:
    """Parse one .rpy file into ``state`` (characters, auto_voice, says).

    ``alloc`` and ``ctx_label`` are one file's: Ren'Py restructures every script file on its own, so the identifiers it
    has handed out and the translation label both start empty in each (:func:`parse_rpy_files` does that)."""
    lines = _logical_lines(text)
    skip_indent: int | None = None
    menu_stack: list[list[Any]] = []  # [indent of the menu, its record, indent of its direct children (None until one is seen)]
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
            name = enter_label(m.group("name"), bool(m.group("hide")), ctx_label)
            state.says.append({"kind": "label", "label": name, "file": filename, "line": no})
            continue
        m = MENU_RE.match(s)
        if m:
            if m.group("name"):  # ``menu name:`` makes a Label node too
                enter_label(m.group("name"), False, ctx_label)
            rec = {"kind": "menu", "choices": [], "file": filename, "line": no, "label": ctx_label[0]}
            state.says.append(rec)
            menu_stack.append([indent, rec, None])
            continue
        m = CALL_FROM_RE.match(s)
        if m:  # ``call x from name`` makes a Label node too
            enter_label(m.group("name"), False, ctx_label, declare=False)
            continue
        direct_menu_child = False
        if menu_stack and indent > menu_stack[-1][0]:
            if menu_stack[-1][2] is None:
                menu_stack[-1][2] = indent
            direct_menu_child = indent == menu_stack[-1][2]
            m = CHOICE_RE.match(s) if direct_menu_child else None
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
        interact, explicit_id, args, with_ = parse_say_tail(rest)
        attrs = (m.group("attrs") or "").split()
        tmp = (m.group("tmp") or "").replace("@", " ").split()
        what_tok = m.group("what")
        if direct_menu_child:
            if who is None:
                # a bare string in a menu is a caption (a menu item without a block), not a say statement: no identifier
                state.says.append(
                    {"kind": "say", "who": None, "what": dequote(what_tok), "voice": None, "voice_expr": False, "sustain": False, "id": "",
                     "label": ctx_label[0], "file": filename, "line": no, "raw": s, "caption": True}
                )
                continue
            interact = False  # the say menuitem: ``finish_say(..., interact=False)``, the menu itself waits for the click
        if what_tok[:3] in ('"""', "'''"):  # monologue mode: one say statement per paragraph
            paragraphs = triple_paragraphs(what_tok)
        else:
            paragraphs = [(0, dequote(what_tok))]
        for n_para, (offset, what) in enumerate(paragraphs):
            first_para = n_para == 0
            codes = []
            if not only_say and first_para:
                if pending.voice_code:
                    codes.append(pending.voice_code)
                if pending.sustain_code:
                    codes.append(pending.sustain_code)
            codes.append(say_code(who, what, attrs, tmp, interact, explicit_id if first_para else None, args, with_))
            computed = alloc.allocate(ctx_label[0], block_digest(codes))  # an explicit id replaces it, the computed one stays taken
            ident = (explicit_id if first_para else None) or computed
            state.says.append(
                {
                    "kind": "say",
                    "who": who,
                    "what": what,
                    "voice": pending.voice if first_para else None,
                    "voice_expr": first_para and bool(pending.voice_code) and pending.voice is None,
                    "sustain": pending.sustain if first_para else False,
                    "id": ident,
                    "label": ctx_label[0],
                    "file": filename,
                    "line": no + offset,
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
    """Parse files in Ren'Py load order (sorted by path, as the launcher does).

    Every file gets its own :class:`IdAllocator` and translation label: ``Script.static_transforms`` calls
    ``renpy.translation.restructure`` once per file, so the ``_1`` / ``_2`` collision suffixes are per file."""
    state = RpyParse()
    for f in sorted(files, key=lambda p: p.as_posix().lower()):
        from vndub.engines.base import read_text_any

        ctx_label: list[str | None] = [None, None]  # [translation label, last global label]
        parse_rpy_text(read_text_any(f), f.name, state, IdAllocator(), ctx_label, only_say)
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
            if st.get("caption"):
                flags.append("menu_caption")  # a bare string in a menu: shown, never spoken, no dialogue identifier
            rec = lb.add(scene, text, speaker_raw=speaker_raw, speaker_id=sid, raw=st["raw"], source_file=st["file"], source_line=st["line"], flags=flags)
            if st["id"]:  # a menu caption is not a say statement and has no identifier
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
            if not ref and auto_voice and st["id"]:
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
        write_install(ctx.paths.out_engine("renpy"), "renpy", install_steps(ctx))
        return {"mode": "loose", "files": n, "out": str(out)}


def install_steps(ctx: "StageContext") -> list[str]:
    """INSTALL.md steps of a Ren'Py patch (also used by ``pack`` when the patch holds only lines without original voice)."""
    return [
        f"Copy the `game` folder from here over the game's `game` folder ({ctx.game_dir / 'game'}).",
        "Ren'Py loads loose files before .rpa archives, so the new voices override the originals.",
        "If the game sets `config.force_archives = True`, loose files are ignored: repack the voices into an .rpa (rpatool -c) named so it sorts after the original archives.",
    ]

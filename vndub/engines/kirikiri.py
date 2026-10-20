"""KiriKiri 2 / KiriKiri Z: KAG scenarios (``.ks``), KiriKiri Z ``.scn`` (PSB), XP3 archives.

Extraction
    plain (unencrypted) ``*.xp3`` -> built-in reader (:mod:`vndub.engines.xp3`);
    dumped folders (KirikiriTools ``version.dll`` ``unencrypted/``, KrkrDump
    output, a ``data/`` folder) are used as-is; encrypted archives -> clear
    error pointing at KrkrDump / version.dll; ``.scn`` -> FreeMote
    ``PsbDecompile`` (tools.yaml: freemote_psbdecompile) when configured.

Parsing (KAG, two phases)
    1. macro learning: ``[macro name=X] ... [endmacro]`` bodies are scanned for
       voice-playing tags (``playse``/``voice``/``playvoice``...) whose
       ``storage`` is a macro parameter (``%s``, ``%s|default``, ``&mp.s``),
       and for name output (``%t`` inside ``[ch text=%t]``, ``[emb exp="mp.t"]``,
       ``[eval exp="...mp.t..."]``); plus defaults for the common conventions
       ``[nm t="名前" s=voice]``, ``[cn ...]``, ``[name ...]``.
    2. streaming: per scenario file keep "current speaker" and "pending voice";
       text runs end at ``[l]``/``[p]``/``[s]``/``[cm]`` or a blank line (``[r]`` is
       a line break inside the message unless ``engine_opts.kag.break_on_r``).
       Speaker sources: name macros, ``[ns]名前[nse]``, ``【名前】`` inline,
       ``#名前`` line prefix. ``[playse storage=X]`` outside macros counts as a
       voice only when X resolves to an audio file in a voice dir or with a
       known ``<prefix>_<index>`` name (otherwise it is a sound effect).

Packing
    ``out/kirikiri/patchN.xp3`` (first unused N; flat root entries) with
    Xp3Pack when configured, else the built-in writer; ``mode: unencrypted_dir``
    writes ``out/kirikiri/unencrypted/``.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Iterator

from vndub.engines import xp3
from vndub.engines.base import (
    BaseEngine,
    DetectResult,
    ExtractResult,
    LineBuilder,
    ParseResult,
    VoiceResolver,
    bind_line,
    build_characters,
    find_audio,
    normalize_speaker,
    pick_voice_files,
    read_text_any,
    write_install,
)
from vndub.manifest.io import AudioIndexRecord, FinalIndexRecord

if TYPE_CHECKING:
    from vndub.stages.common import StageContext

TAG_RE = re.compile(r"\[(?P<body>[^\[\]]*)\]")
ATTR_RE = re.compile(r"""(?P<k>[A-Za-z_][\w\.\-]*)\s*(?:=\s*(?P<v>"[^"]*"|'[^']*'|[^\s\]]+))?""")
VOICE_TAGS = {"playse", "voice", "playvoice", "vo", "pv", "playvo", "se_voice", "pvoice"}
LINE_END_TAGS = {"l", "p", "s", "cm", "pg", "plc"}
DEFAULT_NAME_ATTRS = ("t", "name", "n", "chara", "text", "nm")
DEFAULT_VOICE_ATTRS = ("s", "voice", "v", "vo", "storage", "file")
VOICE_NAME_RE = re.compile(r"^[A-Za-z]{1,8}_?\d{2,}[A-Za-z]?$")


def parse_tag(body: str) -> tuple[str, dict[str, str]]:
    body = body.strip()
    if not body:
        return "", {}
    m = re.match(r"^([^\s=]+)", body)
    if not m:
        return "", {}
    name = m.group(1)
    attrs: dict[str, str] = {}
    for am in ATTR_RE.finditer(body[m.end():]):
        v = am.group("v")
        if v is None:
            attrs[am.group("k")] = "true"
        else:
            attrs[am.group("k")] = v[1:-1] if v[:1] in "\"'" else v
    return name, attrs


@dataclass
class MacroInfo:
    name: str
    voice_attr: str | None = None
    voice_default: str | None = None
    name_attr: str | None = None
    name_literal: str | None = None  # macro that always prints a fixed name
    voice_tag: str | None = None


def _param_ref(v: str) -> tuple[str | None, str | None]:
    """'%s' / '%s|def' / '&mp.s' / "&mp['s']" -> (param, default)."""
    v = v.strip()
    m = re.match(r"^%([\w]+)(?:\|(.*))?$", v)
    if m:
        return m.group(1), m.group(2)
    m = re.match(r"""^&?mp(?:\.(\w+)|\[['"](\w+)['"]\])$""", v)
    if m:
        return m.group(1) or m.group(2), None
    return None, None


def learn_macros(texts: dict[str, str], macro_names: list[str] | None = None) -> dict[str, MacroInfo]:
    """Phase 1: scan [macro name=X] ... [endmacro] bodies for voice/name parameters."""
    macros: dict[str, MacroInfo] = {}
    for text in texts.values():
        cur: MacroInfo | None = None
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line.startswith(";"):
                continue
            tags = [m.group("body") for m in TAG_RE.finditer(line)]
            if line.startswith("@"):
                tags.append(line[1:])
            for body in tags:
                tname, attrs = parse_tag(body)
                if tname == "macro" and attrs.get("name"):
                    cur = MacroInfo(attrs["name"])
                    macros[cur.name] = cur
                    continue
                if tname == "endmacro":
                    cur = None
                    continue
                if cur is None:
                    continue
                if tname in VOICE_TAGS and "storage" in attrs:
                    p, d = _param_ref(attrs["storage"])
                    if p:
                        cur.voice_attr, cur.voice_default, cur.voice_tag = p, d, tname
                for k, v in attrs.items():
                    if tname in ("ch", "nm_inner", "name_inner", "emb", "eval") or k in ("text", "exp", "name"):
                        if tname in VOICE_TAGS:
                            continue
                        mm = re.search(r"%(\w+)|mp\.(\w+)|mp\[['\"](\w+)['\"]\]", v)
                        if mm:
                            p = mm.group(1) or mm.group(2) or mm.group(3)
                            if p and not cur.name_attr and p != cur.voice_attr:
                                cur.name_attr = p
            if cur is not None and not tags and line and not cur.name_literal:
                # a macro body that prints text directly is a fixed-name macro: [macro name=sakura]【さくら】[endmacro]
                m = re.match(r"^【([^】]+)】$", line)
                if m:
                    cur.name_literal = m.group(1)
            if cur is not None:
                m = re.search(r"【([^】]+)】", line)
                if m and not cur.name_literal and not TAG_RE.sub("", line).strip().replace(m.group(0), ""):
                    cur.name_literal = m.group(1)
    for n in macro_names or []:
        macros.setdefault(n, MacroInfo(n))
    return macros


@dataclass
class KagLine:
    scene: str
    text: str
    speaker: str
    voice: str | None
    raw: str
    file: str
    line: int
    label: str = ""
    flags: list[str] = field(default_factory=list)


class KagParser:
    def __init__(self, macros: dict[str, MacroInfo], name_attr: str = "auto", voice_attr: str = "auto", break_on_r: bool = False, is_voice: Any = None) -> None:
        self.macros = macros
        self.name_attr = name_attr
        self.voice_attr = voice_attr
        self.break_on_r = break_on_r
        self.is_voice = is_voice or (lambda ref: True)

    def _name_macro(self, tname: str, attrs: dict[str, str]) -> tuple[str | None, str | None] | None:
        mi = self.macros.get(tname)
        if mi is None:
            return None
        name = voice = None
        if self.name_attr != "auto":
            name = attrs.get(self.name_attr)
        elif mi.name_attr and mi.name_attr in attrs:
            name = attrs[mi.name_attr]
        elif mi.name_literal:
            name = mi.name_literal
        else:
            for a in DEFAULT_NAME_ATTRS:
                if a in attrs and a != mi.voice_attr:
                    name = attrs[a]
                    break
        if self.voice_attr != "auto":
            voice = attrs.get(self.voice_attr)
        elif mi.voice_attr:
            voice = attrs.get(mi.voice_attr, mi.voice_default)
        else:
            for a in DEFAULT_VOICE_ATTRS:
                if a in attrs and attrs[a] != name:
                    voice = attrs[a]
                    break
        if name is None and voice is None and not mi.voice_attr and not mi.name_attr:
            return None
        return name, voice

    def parse(self, text: str, scene: str, filename: str) -> Iterator[KagLine]:
        speaker = ""
        voice: str | None = None
        buf: list[str] = []
        raw: list[str] = []
        start_line = 0
        label = ""
        in_ns = False
        ns_buf: list[str] = []
        in_macro = False
        in_script = False

        def emit(line_no: int) -> Iterator[KagLine]:
            nonlocal buf, raw, speaker, voice
            t = "".join(buf).strip()
            if t:
                sp = speaker
                m = re.match(r"^【([^】]+)】\s*(.*)$", t, re.S)
                if m:
                    sp, t = m.group(1), m.group(2).strip()
                yield KagLine(scene, t, normalize_speaker(sp), voice, " ".join(raw).strip(), filename, start_line or line_no, label)
                speaker, voice = "", None
            buf, raw = [], []

        for no, raw_line in enumerate(text.splitlines(), 1):
            line = raw_line.rstrip("\r\n")
            s = line.strip()
            if in_script:
                if s.startswith("[endscript]") or s == "@endscript":
                    in_script = False
                continue
            if not s:
                yield from emit(no)
                continue
            if s.startswith(";"):
                continue
            if s.startswith("*"):
                yield from emit(no)
                label = s[1:].split("|", 1)[0]
                continue
            if s.startswith("#"):
                yield from emit(no)
                speaker = s[1:].strip()
                continue
            if s.startswith("@"):
                tags = [s[1:]]
                rest_text = ""
            else:
                tags = None
                rest_text = s
            if tags is not None:
                tname, attrs = parse_tag(tags[0])
                if tname == "iscript":
                    in_script = True
                    continue
                for out in self._handle_tag(tname, attrs, tags[0], no):
                    if out == "END":
                        yield from emit(no)
                    elif isinstance(out, tuple):
                        if out[0] == "speaker":
                            speaker = out[1]
                        elif out[0] == "voice":
                            voice = out[1]
                        elif out[0] == "macro":
                            in_macro = out[1]
                continue
            pos = 0
            for m in TAG_RE.finditer(rest_text):
                chunk = rest_text[pos : m.start()]
                pos = m.end()
                if chunk and not in_macro:
                    if in_ns:
                        ns_buf.append(chunk)
                    else:
                        if not buf:
                            start_line = no
                        buf.append(chunk)
                tname, attrs = parse_tag(m.group("body"))
                if tname == "ns":
                    in_ns, ns_buf = True, []
                    continue
                if tname == "nse":
                    in_ns = False
                    speaker = "".join(ns_buf).strip()
                    continue
                if tname == "iscript":
                    in_script = True
                    break
                if not in_macro:
                    raw.append(m.group(0))
                for out in self._handle_tag(tname, attrs, m.group("body"), no):
                    if out == "END":
                        if not in_macro:
                            yield from emit(no)
                    elif out == "BR":
                        if self.break_on_r and not in_macro:
                            yield from emit(no)
                    elif isinstance(out, tuple):
                        if out[0] == "speaker" and not in_macro:
                            speaker = out[1]
                        elif out[0] == "voice" and not in_macro:
                            voice = out[1]
                        elif out[0] == "macro":
                            in_macro = out[1]
            tail = rest_text[pos:]
            if tail and not in_macro:
                if in_ns:
                    ns_buf.append(tail)
                else:
                    if not buf:
                        start_line = no
                    buf.append(tail)
        yield from emit(len(text.splitlines()) + 1)

    def _handle_tag(self, tname: str, attrs: dict[str, str], body: str, no: int) -> Iterator[Any]:
        if tname == "macro":
            yield ("macro", True)
            return
        if tname == "endmacro":
            yield ("macro", False)
            return
        if tname in LINE_END_TAGS:
            yield "END"
            return
        if tname == "r":
            yield "BR"
            return
        nm = self._name_macro(tname, attrs)
        if nm is not None:
            name, v = nm
            if name:
                yield ("speaker", name)
            if v and self.is_voice(v):
                yield ("voice", v)
            return
        if tname in VOICE_TAGS and attrs.get("storage"):
            ref = attrs["storage"]
            if tname != "playse" or self.is_voice(ref):
                yield ("voice", ref)


# ----------------------------------------------------------------------------
# KiriKiri Z .scn (FreeMote PsbDecompile JSON)
# ----------------------------------------------------------------------------


def _find_voices(obj: Any) -> list[str]:
    out: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "voice" and isinstance(v, str):
                out.append(v)
            elif k == "voice" and isinstance(v, (list, dict)):
                out.extend(_find_voices(v) or ([v["name"]] if isinstance(v, dict) and isinstance(v.get("name"), str) else []))
            else:
                out.extend(_find_voices(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_find_voices(v))
    return out


def parse_scn_json(data: Any, scene_default: str) -> list[dict[str, Any]]:
    """``scenes[].texts[]``: [name, display_name, text, ...nested dicts with "voice"]."""
    rows: list[dict[str, Any]] = []
    scenes = data.get("scenes", []) if isinstance(data, dict) else []
    for si, sc in enumerate(scenes):
        if not isinstance(sc, dict):
            continue
        label = str(sc.get("label") or sc.get("name") or f"{scene_default}_{si:03d}")
        for ti, entry in enumerate(sc.get("texts") or []):
            if not isinstance(entry, list) or len(entry) < 3:
                continue
            name = entry[0] if isinstance(entry[0], str) else ""
            disp = entry[1] if isinstance(entry[1], str) else ""
            text = entry[2] if isinstance(entry[2], str) else ""
            if not text:
                # some games nest the language variants: [[name, text, ...], ...]
                if isinstance(entry[2], list) and entry[2] and isinstance(entry[2][0], list):
                    first = entry[2][0]
                    text = next((x for x in first if isinstance(x, str) and x), "")
            voices = _find_voices(entry[3:]) if len(entry) > 3 else []
            rows.append({"label": label, "name": name, "disp": disp or name, "text": text, "voices": voices, "index": ti})
    return rows


# ----------------------------------------------------------------------------
# engine
# ----------------------------------------------------------------------------


def _strip_kag_text(t: str) -> str:
    t = re.sub(r"\[[^\]]*\]", "", t)
    return t.replace("　", " ").strip()


class KirikiriEngine(BaseEngine):
    name = "kirikiri"
    out_name = "kirikiri"

    def detect(self, game_dir: Path) -> DetectResult | None:
        xp3s = sorted(game_dir.glob("*.xp3"))
        ks = [p for p in game_dir.rglob("*.ks")][:50]
        tyrano = (game_dir / "data" / "scenario").is_dir() or any((game_dir / d).exists() for d in ("tyrano", "resources/app/tyrano"))
        scn = [p for p in game_dir.rglob("*.scn")][:5]
        if not (xp3s or (ks and not tyrano) or scn):
            return None
        conf = 0.0
        notes: list[str] = []
        encrypted = False
        version = ""
        if xp3s:
            conf = 0.9 if (game_dir / "data.xp3").exists() else 0.8
            for a in xp3s[:3]:
                try:
                    with xp3.XP3Reader(a) as r:
                        enc, why = xp3.looks_encrypted(r)
                        if enc:
                            encrypted = True
                            notes.append(f"{a.name}: looks encrypted ({why})")
                        if any(e.name.lower().endswith(".scn") for e in r.entries):
                            version = "krkrz"
                except xp3.XP3Error as e:
                    encrypted = True
                    notes.append(f"{a.name}: index unreadable ({e})")
        elif ks:
            conf = 0.6
            notes.append("loose .ks scenarios (dumped or unpacked game)")
        if scn:
            version = "krkrz"
            conf = max(conf, 0.7)
        for d in ("unencrypted", "KrkrDump", "extract"):
            if (game_dir / d).is_dir():
                notes.append(f"dump folder found: {d}/")
        if any(game_dir.glob("krkr*.exe")) or any(game_dir.glob("*.tpm")):
            conf = min(1.0, conf + 0.05)
        sample = next(iter(pick_voice_files(game_dir, ("voice", "unencrypted/voice"))), None)
        return DetectResult("kirikiri", conf, version=version or "krkr", encrypted=encrypted, archives=[p.name for p in xp3s], notes=notes, voice_sample=sample)

    # -- extract ---------------------------------------------------------------
    def extract(self, ctx: "StageContext") -> ExtractResult:
        game = ctx.game_dir
        raw = ctx.paths.raw / "xp3"
        warnings: list[str] = []
        notes: list[str] = []
        # roots in KiriKiri load order (later overrides earlier): the game folder itself
        # (loose "Data" files), dump folders, then data.xp3 < other xp3 < patch.xp3 < patch2.xp3 ...
        roots: list[Path] = [game]
        for d in ("unencrypted", "KrkrDump", "extract"):
            if (game / d).is_dir():
                roots.append(game / d)
                notes.append(f"using dump folder {d}/")
        archives = sorted(game.glob("*.xp3"), key=_xp3_order)
        for a in archives:
            dest = raw / a.stem
            marker = dest / ".extracted"
            if marker.exists() and marker.read_text() == str(a.stat().st_size):
                roots.append(dest)
                continue
            try:
                with xp3.XP3Reader(a) as r:
                    enc, why = xp3.looks_encrypted(r)
                    if enc:
                        warnings.append(
                            f"{a.name} looks encrypted ({why}); dump it at runtime with KrkrDump (tools.yaml: krkrdump) "
                            "or KirikiriTools version.dll + extract-unencrypted.txt, then point --project at the dump"
                        )
                        continue
                    if dest.exists():
                        shutil.rmtree(dest)
                    r.extract_all(dest)
                marker.write_text(str(a.stat().st_size))
                roots.append(dest)
                notes.append(f"{a.name}: extracted")
            except (xp3.XP3Error, OSError) as e:
                warnings.append(f"{a.name}: cannot read ({e}); encrypted or unsupported XP3 variant")
        skip_dirs = {"xp3", "unencrypted", "KrkrDump", "extract"}
        # later patches override earlier archives: search roots in reverse load order for scripts
        scripts: list[Path] = []
        seen: set[str] = set()
        for root in reversed(roots):
            cands = list(root.rglob("*.ks")) + list(root.rglob("*.scn")) + list(root.rglob("*.scn.json"))
            for p in sorted(cands):
                rel_parts = p.relative_to(root).parts
                if root == game and rel_parts and rel_parts[0] in skip_dirs:
                    continue  # covered by their own root (or test archives)
                key = p.relative_to(root).as_posix().lower()
                if key in seen:
                    continue
                seen.add(key)
                scripts.append(p)
        # .scn -> JSON via FreeMote
        scn = [p for p in scripts if p.suffix.lower() == ".scn"]
        if scn:
            if ctx.tools.resolve("freemote_psbdecompile") is None:
                warnings.append(f"{len(scn)} .scn files: FreeMote PsbDecompile not configured (tools.yaml: freemote_psbdecompile); they are not parsed")
            else:
                for p in scn:
                    out = p.with_suffix(".json")
                    if not out.exists():
                        r = ctx.tools.run("freemote_psbdecompile", [str(p)], check=False)
                        if r.returncode != 0:
                            warnings.append(f"PsbDecompile failed on {p.name}: {r.stderr.strip()[-200:]}")
                    if out.exists():
                        scripts.append(out)
            scripts = [p for p in scripts if p.suffix.lower() != ".scn"]
        voices: list[Path] = []
        vseen: set[str] = set()
        for root in reversed(roots):
            for v in find_audio(root):
                rel_parts = v.relative_to(root).parts
                if root == game and rel_parts and rel_parts[0] in skip_dirs:
                    continue
                key = v.relative_to(root).as_posix().lower()
                if key in vseen:
                    continue
                vseen.add(key)
                voices.append(v)
        voice_root = roots[-1]
        return ExtractResult(script_files=scripts, voice_root=voice_root, voice_files=voices, voice_roots=[r for r in reversed(roots) if r != voice_root], notes=notes, warnings=warnings)

    # -- parse -----------------------------------------------------------------
    def parse(self, ctx: "StageContext", extracted: ExtractResult, audio: list[AudioIndexRecord]) -> ParseResult:
        opts = ctx.cfg.parse.engine_opts.kag
        resolver = VoiceResolver(audio)
        lb = LineBuilder(ctx.cfg.parse.narrator_names)
        notes: list[str] = []
        ks_files = [p for p in extracted.script_files if p.suffix.lower() == ".ks"]
        texts = {p.as_posix(): read_text_any(p, opts.encoding) for p in ks_files}
        macros = learn_macros(texts, opts.macro_names)
        learned = {n: (m.name_attr, m.voice_attr) for n, m in macros.items() if m.name_attr or m.voice_attr or m.name_literal}
        if learned:
            notes.append("learned macros: " + ", ".join(f"{n}(name={a or '-'}, voice={b or '-'})" for n, (a, b) in sorted(learned.items())))
        prefixes = {re.match(r"^([A-Za-z]+)", Path(a.relpath).stem).group(1).lower() for a in audio if re.match(r"^([A-Za-z]+)", Path(a.relpath).stem)}

        def is_voice(ref: str) -> bool:
            a = resolver.resolve(ref)
            if a is None:
                return bool(VOICE_NAME_RE.match(PurePosixPath(ref.replace("\\", "/")).stem))
            rel = a.relpath.lower()
            if any(f"{d}/" in rel or rel.startswith(f"{d}/") for d in ("voice", "vo", "koe", "sound/voice")):
                return True
            stem = PurePosixPath(rel).stem
            m = re.match(r"^([a-z]+)_?\d{2,}", stem)
            return bool(m and m.group(1) in prefixes and VOICE_NAME_RE.match(stem))

        parser = KagParser(macros, opts.name_attr, opts.voice_attr, getattr(opts, "break_on_r", False), is_voice)
        for p in ks_files:
            if p.name in opts.macro_files and not _has_text(texts[p.as_posix()]):
                continue
            scene = p.stem
            for kl in parser.parse(texts[p.as_posix()], scene, p.name):
                text = _strip_kag_text(kl.text)
                if not text:
                    continue
                rec = lb.add(scene, text, speaker_raw=kl.speaker, raw=kl.raw, source_file=kl.file, source_line=kl.line)
                if kl.label:
                    rec.label = kl.label  # type: ignore[attr-defined]
                if kl.voice:
                    bind_line(rec, resolver.resolve(kl.voice), kl.voice)
        for p in extracted.script_files:
            if not p.name.endswith(".json"):
                continue
            try:
                data = json.loads(read_text_any(p))
            except json.JSONDecodeError as e:
                notes.append(f"{p.name}: invalid JSON ({e})")
                continue
            scene = p.name.split(".")[0]
            for row in parse_scn_json(data, scene):
                if not row["text"]:
                    continue
                rec = lb.add(scene, _strip_kag_text(row["text"]), speaker_raw=row["disp"] or row["name"], source_file=p.name, source_line=row["index"])
                rec.label = row["label"]  # type: ignore[attr-defined]
                if row["voices"]:
                    ref = row["voices"][0]
                    bind_line(rec, resolver.resolve(ref), ref)
                    if len(row["voices"]) > 1:
                        rec.flags.append("multi_voice")
        return ParseResult(lb.lines, build_characters(lb.lines), notes)

    # -- pack ------------------------------------------------------------------
    def pack(self, ctx: "StageContext", finals: list[FinalIndexRecord]) -> dict[str, Any]:
        opts = ctx.cfg.pack.kirikiri
        out = ctx.paths.out_engine("kirikiri")
        out.mkdir(parents=True, exist_ok=True)
        subdir = (opts.get("subdir") or "").strip("/")
        mode = opts.get("mode", "patch_xp3")
        members: list[tuple[str, Path]] = []
        for f in finals:
            src = ctx.paths.abs(f.encoded)
            if not src.exists():
                continue
            name = PurePosixPath(f.voice_file).name
            members.append((f"{subdir}/{name}" if subdir else name, src))
        proj = ctx.project
        encrypted = bool(proj and proj.encrypted)
        if mode == "unencrypted_dir":
            d = out / "unencrypted"
            for name, src in members:
                dst = d / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            write_install(out, "kirikiri", [
                "Install KirikiriTools `version.dll` next to the game exe (https://github.com/arcusmaximus/KirikiriTools/releases).",
                f"Copy the `unencrypted` folder next to the game exe ({ctx.game_dir}).",
                "version.dll makes the engine read files from `unencrypted/` before the archives.",
            ])
            return {"mode": mode, "files": len(members), "out": str(d)}
        patch_name = opts.get("patch_name") or next_patch_name(ctx.game_dir)
        target = out / patch_name
        used = "builtin"
        if ctx.tools.resolve("xp3pack") is not None:
            staging = out / "_xp3pack" / Path(patch_name).stem
            if staging.exists():
                shutil.rmtree(staging)
            for name, src in members:
                dst = staging / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            r = ctx.tools.run("xp3pack", [Path(patch_name).stem], cwd=staging.parent, check=False)
            produced = staging.parent / patch_name
            if r.returncode == 0 and produced.exists():
                shutil.move(str(produced), target)
                used = "Xp3Pack"
            shutil.rmtree(staging.parent, ignore_errors=True)
        if used == "builtin":
            xp3.write_xp3(target, ((name, src.read_bytes()) for name, src in members))
        steps = [
            f"Copy `{patch_name}` next to the game exe ({ctx.game_dir}). KiriKiri loads patch.xp3, patch2.xp3 ... after data.xp3; later archives override earlier ones.",
            "Entries are stored flat at the archive root (KiriKiri resolves storages by file name through its auto-paths).",
        ]
        if subdir:
            steps.append(f'Entries are under `{subdir}/`: add `Storages.addAutoPath(System.exePath + "{patch_name}>{subdir}/");` to the game\'s Config.tjs / startup.tjs override.')
        if encrypted:
            steps.append(
                "The original archives are encrypted: the engine may refuse an unencrypted patch. Install KirikiriTools `version.dll` "
                "(accepts unencrypted xp3, or use mode: unencrypted_dir), or rebuild with the game's cipher via KrkrExtract's Universal Patch."
            )
        write_install(out, "kirikiri", steps)
        return {"mode": mode, "files": len(members), "out": str(target), "writer": used}


def _has_text(text: str) -> bool:
    body = re.sub(r"\[macro\b.*?\[endmacro\]", "", text, flags=re.S)
    body = TAG_RE.sub("", body)
    return any(l.strip() and not l.strip().startswith((";", "@", "*")) for l in body.splitlines())


def _xp3_order(p: Path) -> tuple[int, str]:
    n = p.stem.lower()
    if n == "data":
        return (0, n)
    m = re.match(r"^patch(\d*)$", n)
    if m:
        return (2 + int(m.group(1) or 1), n)
    return (1, n)


def next_patch_name(game_dir: Path) -> str:
    existing = {p.name.lower() for p in game_dir.glob("patch*.xp3")} if game_dir.exists() else set()
    if "patch.xp3" not in existing:
        return "patch.xp3"
    n = 2
    while f"patch{n}.xp3" in existing:
        n += 1
    return f"patch{n}.xp3"

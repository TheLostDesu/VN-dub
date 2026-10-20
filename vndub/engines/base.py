"""Engine plugin protocol and shared helpers.

An engine knows how to recognise a game (``detect``), get scripts and voice
files out of it (``extract``), turn scripts into ``lines.jsonl`` +
``characters.yaml`` (``parse``) and put replaced voice files back (``pack``).

Paths: ``voice_file`` in lines.jsonl is the path *as the game addresses it*,
relative to the engine's voice root (Ren'Py: relative to ``game/``; KiriKiri:
archive-root relative storage name with the real extension; NScripter: relative
to the exe dir). ``audio_index.jsonl`` stores the same relpath for every
extracted voice file, and :class:`VoiceResolver` matches script references
(often extension-less, case-insensitive, with ``\\`` separators) to it.
"""

from __future__ import annotations

import re
import shutil
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Iterable

from vndub.manifest.io import AudioIndexRecord, Character, FinalIndexRecord, LineRecord, write_json
from vndub.util import slugify

if TYPE_CHECKING:
    from vndub.stages.common import StageContext

AUDIO_EXTS = (".ogg", ".wav", ".opus", ".mp3", ".flac", ".m4a", ".weba", ".hca", ".adx", ".at3", ".at9", ".fsb", ".nwa", ".wem", ".ovk", ".bank", ".acb", ".awb")


@dataclass
class DetectResult:
    engine: str
    confidence: float
    version: str = ""
    encrypted: bool = False
    drm: list[str] = field(default_factory=list)
    archives: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    voice_codec: dict[str, Any] | None = None


@dataclass
class ExtractResult:
    script_files: list[Path]
    voice_root: Path
    voice_files: list[Path]
    notes: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        write_json(
            path,
            {
                "script_files": [p.as_posix() for p in self.script_files],
                "voice_root": self.voice_root.as_posix(),
                "voice_files": [p.as_posix() for p in self.voice_files],
                "notes": self.notes,
                "extra": self.extra,
            },
        )

    @classmethod
    def load(cls, path: Path) -> "ExtractResult":
        from vndub.manifest.io import read_json

        d = read_json(path) or {}
        return cls(
            script_files=[Path(p) for p in d.get("script_files", [])],
            voice_root=Path(d.get("voice_root", ".")),
            voice_files=[Path(p) for p in d.get("voice_files", [])],
            notes=d.get("notes", []),
            extra=d.get("extra", {}),
        )


@dataclass
class ParseResult:
    lines: list[LineRecord]
    characters: dict[str, Character]
    notes: list[str] = field(default_factory=list)


class BaseEngine:
    """Defaults shared by engines; subclasses override what differs."""

    name = "generic"
    # where pack writes and what it is called in out/
    out_name = "generic"

    # -- detect ----------------------------------------------------------------
    def detect(self, game_dir: Path) -> DetectResult | None:
        return None

    # -- extract ---------------------------------------------------------------
    def extract(self, ctx: "StageContext") -> ExtractResult:
        """Default: no archives; scripts and voices are loose files in the game dir."""
        game = ctx.game_dir
        voices = find_audio(game)
        return ExtractResult(script_files=[], voice_root=game, voice_files=voices, notes=["loose files (no extraction)"])

    # -- parse -----------------------------------------------------------------
    def parse(self, ctx: "StageContext", extracted: ExtractResult, audio: list[AudioIndexRecord]) -> ParseResult:
        raise NotImplementedError(f"engine {self.name} has no script parser; use generic_prefix with --import-vntextpatch")

    # -- pack ------------------------------------------------------------------
    def pack(self, ctx: "StageContext", finals: list[FinalIndexRecord]) -> dict[str, Any]:
        """Default: loose files under out/<engine>/<voice relpath> + INSTALL.md."""
        out = ctx.paths.out_engine(self.out_name)
        n = copy_loose(ctx, finals, out)
        write_install(
            out,
            self.name,
            [
                f"Copy the contents of this folder into the game directory ({ctx.game_dir}).",
                "Files keep the original names, codec, sample rate and channel layout.",
                "If the engine does not prefer loose files over archives, repack with the engine tool listed in docs/ENGINES.md.",
            ],
        )
        return {"mode": "loose", "files": n, "out": str(out)}


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def find_audio(root: Path, exts: Iterable[str] = AUDIO_EXTS) -> list[Path]:
    exts = {e.lower() for e in exts}
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def norm_ref(ref: str) -> str:
    """Normalize a script voice reference for matching: posix, lowercase, no extension."""
    r = unicodedata.normalize("NFKC", ref.strip().strip('"').strip("'")).replace("\\", "/").lstrip("./")
    p = PurePosixPath(r)
    if p.suffix.lower() in AUDIO_EXTS:
        p = p.with_suffix("")
    return p.as_posix().lower()


class VoiceResolver:
    """Resolve script voice references to extracted audio files."""

    def __init__(self, audio: list[AudioIndexRecord]) -> None:
        self.by_full: dict[str, AudioIndexRecord] = {}
        self.by_base: dict[str, list[AudioIndexRecord]] = {}
        for a in audio:
            key = norm_ref(a.relpath)
            self.by_full[key] = a
            self.by_base.setdefault(key.rsplit("/", 1)[-1], []).append(a)

    def resolve(self, ref: str | None, prefer_dir: str = "") -> AudioIndexRecord | None:
        if not ref:
            return None
        key = norm_ref(ref)
        if key in self.by_full:
            return self.by_full[key]
        if prefer_dir:
            k2 = norm_ref(f"{prefer_dir}/{ref}")
            if k2 in self.by_full:
                return self.by_full[k2]
        cands = self.by_base.get(key.rsplit("/", 1)[-1], [])
        if len(cands) == 1:
            return cands[0]
        if cands and prefer_dir:
            pd = prefer_dir.lower().strip("/")
            for c in cands:
                if norm_ref(c.relpath).startswith(pd + "/"):
                    return c
        return cands[0] if cands else None


_BRACKETS = re.compile(r"[\s「」『』【】\[\]()（）<>＜＞〈〉《》\"'“”]")


def normalize_speaker(name: str) -> str:
    return _BRACKETS.sub("", unicodedata.normalize("NFKC", name or "")).strip()


def speaker_id_for(name: str, narrator_names: Iterable[str] = ("", "narrator")) -> str:
    n = normalize_speaker(name)
    if n.lower() in {x.lower() for x in narrator_names}:
        return "narrator"
    return slugify(n, fallback="narrator")


def bind_line(line: LineRecord, audio: AudioIndexRecord | None, ref: str | None) -> None:
    """Fill voice fields of ``line`` from a resolved audio record (or flag a dangling ref)."""
    if ref and audio is None:
        line.voice_file = ref
        line.has_voice = False
        if "voice_missing" not in line.flags:
            line.flags.append("voice_missing")
        return
    if audio is None:
        return
    line.voice_file = audio.relpath
    line.audio_id = audio.id
    line.has_voice = True
    line.src_wav = audio.wav or None
    line.src_wav16 = audio.wav16 or None


def build_characters(lines: list[LineRecord], existing: dict[str, Character] | None = None) -> dict[str, Character]:
    """Initial characters.yaml from parsed lines (keeps user edits in ``existing``)."""
    chars: dict[str, Character] = dict(existing or {})
    counts: dict[str, int] = {}
    prefixes: dict[str, dict[str, int]] = {}
    for l in lines:
        if l.speaker_id == "narrator" and not l.has_voice:
            continue
        c = chars.get(l.speaker_id)
        if c is None:
            c = Character(names={"src": l.speaker_raw or l.speaker_id, "tgt": ""})
            chars[l.speaker_id] = c
        if l.has_voice:
            counts[l.speaker_id] = counts.get(l.speaker_id, 0) + 1
            if l.voice_file:
                stem = PurePosixPath(l.voice_file).stem
                m = re.match(r"^([A-Za-z]+)", stem)
                if m:
                    pd = prefixes.setdefault(l.speaker_id, {})
                    pd[m.group(1)] = pd.get(m.group(1), 0) + 1
    for sid, c in chars.items():
        c.stats.n_lines = counts.get(sid, c.stats.n_lines)
        if not c.voice_prefix and prefixes.get(sid):
            c.voice_prefix = max(prefixes[sid].items(), key=lambda kv: kv[1])[0]
    return chars


def copy_loose(ctx: "StageContext", finals: list[FinalIndexRecord], out_root: Path) -> int:
    n = 0
    for f in finals:
        src = ctx.paths.abs(f.encoded)
        if not src.exists():
            continue
        dst = out_root / f.voice_file
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        n += 1
    return n


def write_install(out: Path, engine: str, steps: list[str], extra: str = "") -> Path:
    out.mkdir(parents=True, exist_ok=True)
    body = [f"# Installing the vndub voice patch ({engine})", ""]
    body += [f"{i}. {s}" for i, s in enumerate(steps, 1)]
    body += [
        "",
        "Back up the original files first. The patch contains only replaced voice files",
        "generated locally from your own copy of the game, for private/research use.",
    ]
    if extra:
        body += ["", extra]
    p = out / "INSTALL.md"
    p.write_text("\n".join(body) + "\n", encoding="utf-8")
    return p

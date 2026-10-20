"""Unity (incl. Naninovel). EXPERIMENTAL.

Extraction
    AssetRipper ``--headless`` (tools.yaml: ``assetripper``) exports the project;
    or, when the ``UnityPy`` package is importable (``uv sync --extra extract``),
    AudioClips are exported directly to ``work/raw/unity/audio/<name>.wav``.
    Already exported projects (``Assets/``) are used as-is.

Parsing (Naninovel ``.nani`` scripts)
    ``Author: text`` generic lines, plain narration lines, ``@voice Path/To/Clip``
    binds to the next line; ``# label`` lines; auto-voice: a text id marker
    ``|#id|`` (or the line's position id) maps to
    ``Naninovel/Voice/<ScriptPath>/<textId>`` when such a clip exists.

Packing
    Unity has no loose-file override: ``out/unity/`` gets the encoded clips,
    ``replace_audioclips.py`` (UnityPy: replaces AudioClip samples in the
    bundles listed in ``clips.json``) and a checklist. FSB5 re-encoding may be
    needed; clip metadata mismatches can crash the game -- test on a copy.
"""

from __future__ import annotations

import json
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
    find_audio,
    read_text_any,
    write_install,
)
from vndub.manifest.io import AudioIndexRecord, FinalIndexRecord

if TYPE_CHECKING:
    from vndub.stages.common import StageContext

AUTHOR_RE = re.compile(r"^(?P<author>[\w\.\-]+)(?:\.[\w\-]+)?\s*:\s*(?P<text>.+)$")
VOICE_RE = re.compile(r"^@voice\s+(?P<path>\S+)")
TEXTID_RE = re.compile(r"\|#(?P<id>[\w\-]+)\|")


def parse_nani(text: str, script_path: str) -> list[dict[str, Any]]:
    rows = []
    pending: str | None = None
    label = ""
    for no, raw in enumerate(text.splitlines(), 1):
        s = raw.strip()
        if not s or s.startswith(";"):
            continue
        if s.startswith("#"):
            label = s[1:].strip()
            continue
        if s.startswith("@"):
            m = VOICE_RE.match(s)
            if m:
                pending = m.group("path")
            continue
        tid_m = TEXTID_RE.search(s)
        tid = tid_m.group("id") if tid_m else None
        s2 = TEXTID_RE.sub("", s)
        author = ""
        m = AUTHOR_RE.match(s2)
        if m and not s2.startswith("「"):
            author, s2 = m.group("author"), m.group("text")
        s2 = re.sub(r"\[[^\]]*\]|<[^>]+>|\{[^}]*\}", "", s2).strip()
        if not s2:
            continue
        voice = pending
        flags = []
        if voice is None and tid:
            voice = f"Naninovel/Voice/{script_path}/{tid}"
            flags.append("auto_voice_textid")
        rows.append({"author": author, "text": s2, "voice": voice, "line": no, "label": label, "flags": flags})
        pending = None
    return rows


class UnityEngine(BaseEngine):
    name = "unity"
    out_name = "unity"

    def detect(self, game_dir: Path) -> DetectResult | None:
        data_dirs = [p for p in game_dir.glob("*_Data") if p.is_dir()]
        ggm = any((d / "globalgamemanagers").exists() for d in data_dirs)
        nani = list(game_dir.rglob("*.nani"))[:3]
        if not (ggm or data_dirs or nani):
            return None
        notes = ["EXPERIMENTAL: Unity has no loose-file override; see docs/ENGINES.md"]
        if nani:
            notes.append("Naninovel scripts found")
        return DetectResult("unity", 0.9 if ggm else 0.6, notes=notes)

    def extract(self, ctx: "StageContext") -> ExtractResult:
        game = ctx.game_dir
        raw = ctx.paths.raw / "unity"
        warnings: list[str] = []
        scripts = sorted(game.rglob("*.nani"))
        audio_root = raw / "audio"
        exported = list(game.rglob("Assets")) if not scripts else []
        if not scripts and not exported:
            try:
                import UnityPy  # type: ignore[import-not-found]

                audio_root.mkdir(parents=True, exist_ok=True)
                clips = {}
                for d in [p for p in game.glob("*_Data") if p.is_dir()]:
                    env = UnityPy.load(str(d))
                    for obj in env.objects:
                        if obj.type.name == "AudioClip":
                            clip = obj.read()
                            for name, data in clip.samples.items():
                                out = audio_root / name
                                out.write_bytes(data)
                                clips[name] = {"container": obj.assets_file.name, "path_id": obj.path_id, "clip": clip.m_Name}
                (raw / "clips.json").write_text(json.dumps(clips, ensure_ascii=False, indent=1), encoding="utf-8")
            except ImportError:
                if ctx.tools.resolve("assetripper") is not None:
                    r = ctx.tools.run("assetripper", ["--headless", "--output", str(raw / "export"), str(game)], check=False, timeout=24 * 3600)
                    if r.returncode != 0:
                        warnings.append(f"AssetRipper failed: {r.stderr.strip()[-200:]}")
                else:
                    warnings.append("neither UnityPy (uv sync --extra extract) nor AssetRipper (tools.yaml: assetripper) available")
            scripts = sorted(raw.rglob("*.nani")) if raw.exists() else []
        roots = [p for p in (audio_root, raw / "export", game) if p.exists()]
        voices = []
        for r in roots:
            voices += find_audio(r)
        return ExtractResult(script_files=scripts, voice_root=roots[0] if roots else game, voice_files=sorted(set(voices)), voice_roots=roots[1:], warnings=warnings, notes=["EXPERIMENTAL"])

    def parse(self, ctx: "StageContext", extracted: ExtractResult, audio: list[AudioIndexRecord]) -> ParseResult:
        resolver = VoiceResolver(audio)
        lb = LineBuilder(ctx.cfg.parse.narrator_names)
        for p in extracted.script_files:
            rel = p.stem
            for row in parse_nani(read_text_any(p), rel):
                rec = lb.add(p.stem, row["text"], speaker_raw=row["author"], source_file=p.name, source_line=row["line"], flags=list(row["flags"]))
                if row["voice"]:
                    a = resolver.resolve(row["voice"]) or resolver.resolve(row["voice"].rsplit("/", 1)[-1])
                    if a is None and "auto_voice_textid" in rec.flags:
                        continue
                    bind_line(rec, a, row["voice"])
        return ParseResult(lb.lines, build_characters(lb.lines))

    def pack(self, ctx: "StageContext", finals: list[FinalIndexRecord]) -> dict[str, Any]:
        out = ctx.paths.out_engine("unity")
        clips_dir = out / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        import shutil

        mapping = {}
        for f in finals:
            src = ctx.paths.abs(f.encoded)
            if src.exists():
                dst = clips_dir / Path(f.voice_file).name
                shutil.copy2(src, dst)
                mapping[Path(f.voice_file).stem] = dst.name
        meta = ctx.paths.raw / "unity" / "clips.json"
        info = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
        (out / "clips.json").write_text(json.dumps({"replacements": mapping, "source": info}, ensure_ascii=False, indent=1), encoding="utf-8")
        (out / "replace_audioclips.py").write_text(REPLACE_SCRIPT, encoding="utf-8")
        write_install(out, "unity (EXPERIMENTAL)", [
            "Back up the game's *_Data folder.",
            "pip install UnityPy, then run: python replace_audioclips.py <game>/<Name>_Data clips clips.json",
            "The script replaces AudioClip samples by clip name and saves the modified assets next to the originals (*.patched); swap them in after checking.",
            "If clips are FSB5-compressed and the game rejects the new data, re-encode with the original codec (e.g. via UABEA's import) instead.",
            "Alternative: a BepInEx plugin that swaps AudioClips at runtime (not provided).",
        ])
        return {"mode": "unitypy_script", "files": len(mapping), "out": str(out), "experimental": True}


REPLACE_SCRIPT = '''"""Replace Unity AudioClip samples with vndub output (EXPERIMENTAL).

usage: python replace_audioclips.py <Game_Data dir> <clips dir> <clips.json>
Writes modified asset files as <name>.patched next to the originals.
"""
import json
import sys
from pathlib import Path

import UnityPy

data_dir, clips_dir, meta = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
repl = json.loads(meta.read_text(encoding="utf-8"))["replacements"]
env = UnityPy.load(str(data_dir))
changed = set()
for obj in env.objects:
    if obj.type.name != "AudioClip":
        continue
    clip = obj.read()
    name = clip.m_Name
    if name not in repl:
        continue
    new = (clips_dir / repl[name]).read_bytes()
    try:
        clip.m_AudioData = new
        clip.save()
        changed.add(obj.assets_file.name)
        print("replaced", name)
    except Exception as e:  # typetree differences between Unity versions
        print("FAILED", name, e)
for f in env.files.values():
    if getattr(f, "name", None) in changed:
        out = data_dir / (f.name + ".patched")
        out.write_bytes(f.save())
        print("wrote", out)
'''

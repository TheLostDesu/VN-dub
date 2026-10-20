"""SiglusEngine: ``Scene.pck`` + ``Gameexe.dat``; voices in ``koe/*.ovk``.

Extraction uses SiglusSceneScriptUtility (tools.yaml: ``siglus_ssu``):
``siglus-ssu -x Scene.pck <out>`` (scripts) and
``siglus-ssu -k Scene.pck <ovk dir> <out>`` which dumps every voice as
``KOE(000000001).ogg`` into per-character folders and writes
``koe_master.csv`` (koe_no, character, text, duration, callsite). That CSV is
the mapping: parse reads it directly (a ``koe_master.csv`` already present in
the game dir or work/raw/siglus is used without running the tool).
Global KOE id = scene * 100000 + entry.

Packing: ``.ovk`` repacking is not provided by siglus-ssu; vndub writes the
replaced ``KOE(...)`` files in the dump layout and documents the options.
"""

from __future__ import annotations

import csv
import io
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

COLS = ("koe_no", "character", "text", "duration", "callsite")


def read_koe_master(path: Path) -> list[dict[str, str]]:
    text = read_text_any(path)
    sample = text[:2000]
    delim = "\t" if sample.count("\t") > sample.count(",") else ","
    rows = list(csv.reader(io.StringIO(text), delimiter=delim))
    if not rows:
        return []
    header = [h.strip().lower() for h in rows[0]]
    if "koe_no" in header or "koe" in header:
        idx = {c: header.index(c) for c in COLS if c in header}
        if "koe_no" not in idx and "koe" in header:
            idx["koe_no"] = header.index("koe")
        body = rows[1:]
    else:
        idx = {c: i for i, c in enumerate(COLS)}
        body = rows
    out = []
    for r in body:
        if not r or not any(x.strip() for x in r):
            continue
        out.append({c: (r[i].strip() if i < len(r) else "") for c, i in idx.items()})
    return out


def koe_name(koe_no: int) -> str:
    return f"KOE({koe_no:09d})"


def scene_of(row: dict[str, str], koe_no: int) -> str:
    cs = row.get("callsite", "")
    m = re.match(r"^\s*([^:@#\s]+)", cs)
    if m:
        return m.group(1)
    return f"scene{koe_no // 100000:04d}"


class SiglusEngine(BaseEngine):
    name = "siglus"
    out_name = "siglus"

    def detect(self, game_dir: Path) -> DetectResult | None:
        pck = game_dir / "Scene.pck"
        gexe = game_dir / "Gameexe.dat"
        csvp = self._find_master(game_dir, None)
        if not (pck.exists() or csvp):
            return None
        conf = 0.95 if (pck.exists() and gexe.exists()) else 0.7 if pck.exists() else 0.6
        notes = ["voices in koe/*.ovk: extract runs siglus-ssu -k"] if (game_dir / "koe").is_dir() else []
        if csvp:
            notes.append(f"koe_master.csv present: {csvp.name}")
        return DetectResult("siglus", conf, archives=["Scene.pck"] if pck.exists() else [], notes=notes)

    @staticmethod
    def _find_master(game: Path, raw: Path | None) -> Path | None:
        for base in [p for p in (raw, game) if p is not None]:
            hits = sorted(base.rglob("koe_master.csv")) if base.exists() else []
            if hits:
                return hits[0]
        return None

    def extract(self, ctx: "StageContext") -> ExtractResult:
        game = ctx.game_dir
        raw = ctx.paths.raw / "siglus"
        warnings: list[str] = []
        master = self._find_master(game, raw)
        if master is None and (game / "Scene.pck").exists():
            if ctx.tools.resolve("siglus_ssu") is None:
                warnings.append("siglus-ssu not configured (tools.yaml: siglus_ssu); cannot dump voices/koe_master.csv")
            else:
                raw.mkdir(parents=True, exist_ok=True)
                ovk = game / "koe"
                r = ctx.tools.run("siglus_ssu", ["-k", str(game / "Scene.pck"), str(ovk), str(raw / "koe")], check=False)
                if r.returncode != 0:
                    warnings.append(f"siglus-ssu -k failed: {r.stderr.strip()[-300:]}")
                r = ctx.tools.run("siglus_ssu", ["-x", str(game / "Scene.pck"), str(raw / "scene")], check=False)
                if r.returncode != 0:
                    warnings.append(f"siglus-ssu -x failed: {r.stderr.strip()[-300:]}")
                master = self._find_master(game, raw)
        voice_root = master.parent if master else game
        voices = find_audio(voice_root)
        return ExtractResult(script_files=[master] if master else [], voice_root=voice_root, voice_files=voices, warnings=warnings)

    def parse(self, ctx: "StageContext", extracted: ExtractResult, audio: list[AudioIndexRecord]) -> ParseResult:
        resolver = VoiceResolver(audio)
        lb = LineBuilder(ctx.cfg.parse.narrator_names)
        for p in extracted.script_files:
            for row in read_koe_master(p):
                try:
                    koe = int(re.sub(r"\D", "", row.get("koe_no", "")) or -1)
                except ValueError:
                    continue
                if koe < 0:
                    continue
                rec = lb.add(scene_of(row, koe), row.get("text", ""), speaker_raw=row.get("character", ""), source_file=p.name, raw=row.get("callsite", ""))
                rec.koe_no = koe  # type: ignore[attr-defined]
                ref = koe_name(koe)
                prefer = row.get("character", "")
                bind_line(rec, resolver.resolve(ref, prefer_dir=prefer), ref)
        return ParseResult(lb.lines, build_characters(lb.lines))

    def pack(self, ctx: "StageContext", finals: list[FinalIndexRecord]) -> dict[str, Any]:
        out = ctx.paths.out_engine("siglus")
        n = copy_loose(ctx, finals, out / "koe_dump")
        write_install(out, "siglus", [
            "The replaced voices are in `koe_dump/` with the same KOE(#########).ogg names and folders as the siglus-ssu dump.",
            "Siglus reads voices from koe/*.ovk; siglus-ssu does not repack .ovk. Rebuild the .ovk files with a tool that supports it "
            "(e.g. SiglusTools by AlexHatsuyuki, UNVERIFIED for ovk) keeping the entry order, then replace koe/*.ovk.",
        ])
        return {"mode": "dump_layout", "files": n, "out": str(out)}

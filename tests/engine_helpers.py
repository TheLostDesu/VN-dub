"""Helpers for engine tests: run detect + engine.extract + engine.parse without decoding audio."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import orjson

from vndub.manifest.io import AudioIndexRecord

GOLDEN_DIR = Path(__file__).resolve().parent / "fixtures" / "golden"
GOLDEN_FIELDS = ("line_id", "scene", "speaker_raw", "speaker_id", "text_src", "voice_file", "has_voice", "flags", "choices")


def detect_and_parse(ctx: Any, engine: str | None = None):
    """Returns (detect summary, ExtractResult, audio records, ParseResult)."""
    from vndub.stages import detect
    from vndub.stages.extract import assign_ids

    if engine:
        ctx.extra["engine"] = engine
    summary = detect.run(ctx)
    eng = ctx.engine()
    ex = eng.extract(ctx)
    files = [f for f in ex.voice_files if f.exists()]
    rels = [ex.relpath(f) for f in files]
    ids = assign_ids(rels)
    audio = [
        AudioIndexRecord(id=ids[r], relpath=r, src_path=str(f), wav=f"audio/src/{ids[r]}.wav", wav16=f"audio/src16/{ids[r]}.wav", duration=1.0)
        for f, r in zip(files, rels)
    ]
    res = eng.parse(ctx, ex, audio)
    return summary, ex, audio, res


def golden_rows(lines: list) -> list[dict[str, Any]]:
    rows = []
    for l in lines:
        d = l.model_dump(mode="json")
        rows.append({k: d.get(k) for k in GOLDEN_FIELDS})
    return rows


def check_golden(name: str, lines: list) -> None:
    rows = golden_rows(lines)
    path = GOLDEN_DIR / f"{name}.jsonl"
    data = b"\n".join(orjson.dumps(r) for r in rows) + b"\n"
    if os.environ.get("VNDUB_UPDATE_GOLDEN") or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        if not os.environ.get("VNDUB_UPDATE_GOLDEN"):
            raise AssertionError(f"golden {path.name} was missing and has been written; re-run the tests")
        return
    expected = [orjson.loads(x) for x in path.read_bytes().splitlines() if x.strip()]
    assert rows == expected, f"parser output differs from {path.name} (regenerate with VNDUB_UPDATE_GOLDEN=1 if intended)"

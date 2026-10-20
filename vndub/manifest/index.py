"""SQLite index over the JSONL manifests.

The index is a *cache*: it can always be deleted and rebuilt from ``work/``.
It exists for fast filtering in the review UI and the report (joins of lines,
link results, selection and translation by line_id).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import sqlite_utils

from vndub.manifest.io import iter_jsonl, read_json
from vndub.paths import WorkPaths


def _flat(row: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "_"))
        elif isinstance(v, list):
            out[key] = ",".join(str(x) for x in v)
        else:
            out[key] = v
    return out


def rebuild(paths: WorkPaths) -> Path:
    db_path = paths.index_db
    if db_path.exists():
        db_path.unlink()
    db = sqlite_utils.Database(db_path)

    def load(table: str, src: Path, pk: str = "line_id") -> None:
        rows = [_flat(r) for r in iter_jsonl(src)]
        if rows:
            db[table].insert_all(rows, pk=pk, replace=True, alter=True)

    load("lines", paths.lines)
    load("link", paths.link)
    load("selection", paths.selection)
    load("audio", paths.audio_index, pk="id")
    for tr in sorted(paths.translation_dir.glob("lines.*.jsonl")) if paths.translation_dir.exists() else []:
        lang = tr.name.split(".")[1]
        load(f"translation_{lang}", tr)

    # passports: a few scalar columns are enough for filtering
    rows = []
    if paths.analysis.exists():
        for p in paths.analysis.glob("*.json"):
            d = read_json(p) or {}
            emo = d.get("emotion", {})
            adv = emo.get("adv") or {}
            rows.append(
                {
                    "line_id": p.stem,
                    "duration_s": d.get("duration_s"),
                    "snr_db": d.get("snr_db"),
                    "bgm": d.get("bgm"),
                    "emotion": (d.get("style") or {}).get("emotion"),
                    "arousal": adv.get("arousal"),
                    "valence": adv.get("valence"),
                    "events": ",".join(d.get("events") or []),
                }
            )
    if rows:
        db["analysis"].insert_all(rows, pk="line_id", replace=True, alter=True)
    return db_path


def connect(paths: WorkPaths, rebuild_if_missing: bool = True) -> sqlite3.Connection:
    if rebuild_if_missing and not paths.index_db.exists():
        rebuild(paths)
    con = sqlite3.connect(paths.index_db)
    con.row_factory = sqlite3.Row
    return con


def query(paths: WorkPaths, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    con = connect(paths)
    try:
        return [dict(r) for r in con.execute(sql, params).fetchall()]
    finally:
        con.close()


def table_exists(paths: WorkPaths, name: str) -> bool:
    rows = query(paths, "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return bool(rows)

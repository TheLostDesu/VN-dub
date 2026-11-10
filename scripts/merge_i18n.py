#!/usr/bin/env python
"""Merge the i18n part files of the other tasks into the main UI tables.

The SPA reads two flat tables, ``app/i18n/ru.json`` (default) and ``app/i18n/en.json``. Other
tasks never edit them: they write ``app/i18n/parts/<task>.ru.json`` and ``<task>.en.json``. This
script folds every part into the main tables:

* a part key is added only when the main table does not have it yet (**main wins**); a part key
  that exists in the main table with a different text is reported as a conflict and ignored;
* keys are appended in part-file order, so the diff stays readable;
* after the merge it reports keys that exist in one language only (the i18n test fails on them);
* ``--delete`` removes the part files that were merged (done once, at the end);
* ``--dry-run`` only reports.

    uv run python scripts/merge_i18n.py              # merge, keep the part files
    uv run python scripts/merge_i18n.py --dry-run    # report only
    uv run python scripts/merge_i18n.py --delete     # merge and delete the part files

Exit code: 0 on success, 1 with ``--strict`` when there are conflicts or one-language keys,
2 when a file is not valid.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LANGS = ("ru", "en")
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = REPO_ROOT / "app" / "i18n"


class I18nFileError(Exception):
    """A table or part file is unreadable or not a flat key -> string mapping."""


@dataclass
class Report:
    added: dict[str, dict[str, int]] = field(default_factory=dict)  # part -> lang -> keys added
    conflicts: list[tuple[str, str, str, str, str]] = field(default_factory=list)  # lang, key, part, main text, part text
    only_in_one: dict[str, list[str]] = field(default_factory=dict)  # lang that LACKS the key -> keys
    merged_files: list[Path] = field(default_factory=list)
    deleted: list[Path] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.conflicts and not any(self.only_in_one.values())


def flatten(data: Any, prefix: str = "", where: str = "") -> dict[str, str]:
    """A part may be nested JSON; the tables are flat with dotted keys."""
    out: dict[str, str] = {}
    if not isinstance(data, dict):
        raise I18nFileError(f"{where}: expected a JSON object")
    for k, v in data.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, f"{key}.", where))
        elif isinstance(v, str):
            out[key] = v
        else:
            raise I18nFileError(f"{where}: value of {key!r} is not a string")
    return out


def read_table(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        return flatten(json.loads(path.read_text(encoding="utf-8")), where=str(path))
    except json.JSONDecodeError as e:
        raise I18nFileError(f"{path}: invalid JSON ({e})") from e


def write_table(path: Path, table: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(table, ensure_ascii=False, indent=2) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def part_files(i18n_dir: Path) -> list[tuple[str, str, Path]]:
    """(task, lang, path) of every ``parts/<task>.<lang>.json``, sorted."""
    out: list[tuple[str, str, Path]] = []
    parts = i18n_dir / "parts"
    if not parts.is_dir():
        return out
    for p in sorted(parts.glob("*.json")):
        stem, _, lang = p.stem.rpartition(".")
        if lang in LANGS and stem:
            out.append((stem, lang, p))
    return out


def merge(i18n_dir: Path = DEFAULT_DIR, *, delete: bool = False, dry_run: bool = False) -> Report:
    tables = {lang: read_table(i18n_dir / f"{lang}.json") for lang in LANGS}
    report = Report()
    for task, lang, path in part_files(i18n_dir):
        part = read_table(path)
        main = tables[lang]
        added = 0
        for key, text in part.items():
            if key not in main:
                main[key] = text
                added += 1
            elif main[key] != text:
                report.conflicts.append((lang, key, task, main[key], text))
        report.added.setdefault(task, {})[lang] = added
        report.merged_files.append(path)
    ru_keys, en_keys = set(tables["ru"]), set(tables["en"])
    report.only_in_one = {"en": sorted(ru_keys - en_keys), "ru": sorted(en_keys - ru_keys)}
    if not dry_run:
        for lang in LANGS:
            write_table(i18n_dir / f"{lang}.json", tables[lang])
        if delete:
            for path in report.merged_files:
                path.unlink()
                report.deleted.append(path)
            parts = i18n_dir / "parts"
            if parts.is_dir() and not any(parts.iterdir()):
                parts.rmdir()
    return report


def print_report(report: Report, out: Any = None) -> None:
    out = out or sys.stdout
    if not report.merged_files:
        print("no part files to merge", file=out)
    for task, per_lang in report.added.items():
        print(f"part {task}: " + ", ".join(f"{lang} +{n}" for lang, n in sorted(per_lang.items())), file=out)
    if report.conflicts:
        print(f"\nCONFLICTS ({len(report.conflicts)}): the main table wins, the part text was ignored", file=out)
        for lang, key, task, main_text, part_text in report.conflicts:
            print(f"  [{lang}] {key}  (part {task})\n      main: {main_text}\n      part: {part_text}", file=out)
    for lang, keys in report.only_in_one.items():
        if keys:
            print(f"\nkeys missing in {lang}.json ({len(keys)}):", file=out)
            for k in keys:
                print(f"  {k}", file=out)
    if report.deleted:
        print(f"\ndeleted {len(report.deleted)} part file(s)", file=out)
    if report.clean:
        print("\ntables are consistent", file=out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="i18n folder (default app/i18n)")
    ap.add_argument("--delete", action="store_true", help="delete the part files that were merged")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--strict", action="store_true", help="exit with 1 on conflicts or one-language keys")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
    try:
        report = merge(args.dir, delete=args.delete, dry_run=args.dry_run)
    except I18nFileError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print_report(report)
    return 1 if args.strict and not report.clean else 0


if __name__ == "__main__":
    sys.exit(main())

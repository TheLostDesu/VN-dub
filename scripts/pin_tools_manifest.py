"""Check or re-pin ``app/tools_manifest.json`` (needs a network; the app itself never runs this).

    uv run python scripts/pin_tools_manifest.py --check            # HEAD every URL, compare the pinned size
    uv run python scripts/pin_tools_manifest.py --hash llama_server rubberband
                                                                   # download, fill in size + sha256 (verify=sha256)
    uv run python scripts/pin_tools_manifest.py --hash all --write # same for every downloadable entry, rewrite the file

Use it after changing a ``url`` / ``version`` by hand (e.g. a newer llama.cpp build): put the new url,
filename and version into the manifest, run ``--hash <name> --write``, then ``--check``. The result is
validated with ``vndub_app.installer.validate_manifest`` before anything is written. Downloads go to a
temporary folder and are never executed.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "app" / "tools_manifest.json"
sys.path.insert(0, str(ROOT))


def _files(entry: dict[str, Any]) -> list[dict[str, Any]]:
    if entry["kind"] != "download" or entry.get("manual"):
        return []
    return [entry] + list(entry.get("extra") or [])


def head(url: str) -> tuple[int, int | None]:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "VNDub-pin/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            n = r.headers.get("Content-Length")
            return r.status, int(n) if n else None
    except urllib.error.HTTPError as e:
        return e.code, None
    except OSError:
        return 0, None


def check(data: dict[str, Any]) -> int:
    bad = 0
    for e in data["tools"]:
        for f in _files(e):
            status, size = head(f["url"])
            ok = status == 200 and (f.get("size") is None or size is None or f["size"] == size)
            bad += 0 if ok else 1
            print(f"{'ok ' if ok else 'BAD'} {e['name']:24} {f['filename']:50} http {status} size {size} (pinned {f.get('size')})")
    return 1 if bad else 0


def rehash(data: dict[str, Any], names: list[str]) -> None:
    from vndub_app import downloads

    chosen = [e for e in data["tools"] if "all" in names or e["name"] in names]
    if not chosen:
        raise SystemExit(f"no such tool: {names}; known: {[e['name'] for e in data['tools']]}")
    with tempfile.TemporaryDirectory(prefix="vndub-pin-") as tmp:
        for e in chosen:
            for i, f in enumerate(_files(e)):
                dest = Path(tmp) / f"{e['name']}-{i}-{f['filename']}"
                print(f"downloading {f['url']}")
                res = downloads.fetch(f["url"], dest)
                f["size"], f["sha256"], f["verify"] = res.size, downloads.sha256_file(dest), "sha256"
                f.pop("unverified_reason", None)
                print(f"  size {f['size']}  sha256 {f['sha256']}")
                dest.unlink()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="HEAD every download URL and compare sizes")
    p.add_argument("--hash", nargs="+", metavar="NAME", help="download these entries (or 'all') and pin size + sha256")
    p.add_argument("--write", action="store_true", help="write the manifest back (with --hash)")
    args = p.parse_args(argv)
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    code = 0
    if args.hash:
        rehash(data, args.hash)
        from vndub_app import installer

        problems = installer.validate_manifest(data)
        if problems:
            print("the manifest would be invalid:\n  " + "\n  ".join(problems))
            return 1
        if args.write:
            MANIFEST.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {MANIFEST}")
        else:
            print("(dry run: add --write to save)")
    if args.check:
        code = check(data)
    if not args.hash and not args.check:
        p.print_help()
    return code


if __name__ == "__main__":
    sys.exit(main())

"""The built SPA (``vndub_app/web/dist``) and whether it matches its sources.

The build is kept in the repository, so ``uv run vndub-app`` works from a fresh checkout without
Node.js. ``npm run build`` writes ``dist/build-stamp.json`` (``web/scripts/stamp.mjs``) with a hash of
the sources the bundle is made of; :func:`sources_hash` computes the same hash here. Line endings are
normalized (``\\r\\n`` -> ``\\n``) so a checkout with other line endings has the same hash.

* :func:`status` -- built / fresh, for the self-check and the tests (``tests/app/test_web_dist.py``
  fails when the sources changed and the build was not redone);
* :func:`ensure_built` -- at startup from source (never in the packaged exe): a missing or stale
  build is rebuilt with npm when Node.js is installed; without it the stale build is served and a
  warning is logged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger("vndub_app.webbuild")

WEB = Path(__file__).resolve().parent / "web"
REPO = WEB.parent.parent
#: what the bundle is made of (relative to ``web/``); keep in sync with SOURCES in web/scripts/stamp.mjs
SOURCES = ("src", "index.html", "package.json", "package-lock.json", "vite.config.ts", "tsconfig.json", "../../app/i18n/ru.json", "../../app/i18n/en.json")
STAMP = "build-stamp.json"
NODE_DIRS = (r"C:\Program Files\nodejs",)


def _files(web: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for rel in SOURCES:
        p = (web / rel).resolve()
        if p.is_dir():
            out += [(f"{rel}/{f.relative_to(p).as_posix()}", f) for f in p.rglob("*") if f.is_file()]
        elif p.is_file():
            out.append((rel, p))
    return sorted(out)


def sources_hash(web: Path = WEB) -> str:
    """sha256 over ``<relpath>\\0<sha256 of the LF-normalized bytes>\\n`` of every source file, sorted by path."""
    h = hashlib.sha256()
    for rel, p in _files(web):
        digest = hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        h.update(f"{rel}\0{digest}\n".encode())
    return h.hexdigest()


def read_stamp(web: Path = WEB) -> dict[str, Any] | None:
    try:
        return json.loads((web / "dist" / STAMP).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def status(web: Path = WEB) -> dict[str, Any]:
    """``{built, fresh, stamp}``; ``fresh`` is None when the sources are not there (packaged exe)."""
    built = (web / "dist" / "index.html").is_file()
    stamp = read_stamp(web)
    if not (web / "src").is_dir():
        return {"built": built, "fresh": None, "stamp": stamp}
    return {"built": built, "fresh": bool(built and stamp and stamp.get("sources_sha256") == sources_hash(web)), "stamp": stamp}


def find_npm() -> str | None:
    found = shutil.which("npm.cmd" if os.name == "nt" else "npm") or shutil.which("npm")
    if found:
        return found
    for d in NODE_DIRS:
        p = Path(d) / ("npm.cmd" if os.name == "nt" else "npm")
        if p.is_file():
            return str(p)
    return None


def build(web: Path = WEB, npm: str | None = None) -> None:
    """``npm ci`` (when ``node_modules`` is missing) + ``npm run build``; raises ``RuntimeError`` on failure."""
    npm = npm or find_npm()
    if not npm:
        raise RuntimeError("Node.js (npm) is not installed: https://nodejs.org")
    env = dict(os.environ)
    env["PATH"] = str(Path(npm).parent) + os.pathsep + env.get("PATH", "")
    steps = ([[npm, "ci"]] if not (web / "node_modules").is_dir() else []) + [[npm, "run", "build"]]
    for cmd in steps:
        r = subprocess.run(cmd, cwd=web, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd[1:])} failed ({r.returncode}): {(r.stdout + r.stderr)[-2000:]}")


def ensure_built(web: Path = WEB) -> dict[str, Any]:
    """Rebuild a missing or stale SPA when running from source; returns :func:`status` after it."""
    st = status(web)
    if getattr(sys, "frozen", False) or st["fresh"] is not False or os.environ.get("VNDUB_NO_WEB_BUILD"):
        return st
    npm = find_npm()
    if not npm:
        log.warning("the interface build is %s and Node.js is not installed: %s", "stale" if st["built"] else "missing",
                    "serving the old build" if st["built"] else "the app shows the 'not built' page")
        return st
    log.info("the interface sources changed: rebuilding vndub_app/web (npm run build)")
    try:
        build(web, npm)
    except RuntimeError as e:
        log.warning("interface build failed: %s", e)
    return status(web)

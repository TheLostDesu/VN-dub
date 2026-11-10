"""The build record of a backend env (``<env>/vndub_env.json``), standard library only.

``scripts/make_envs.*`` run this file with the env's own interpreter (``python envrecord.py write ...``) at the end of
every env build, and ``vndub_app.installer`` loads it by path to compare a built env with the requirement files of the
running app (``env_states``). One copy of the hash and of the torch-tag rules, so the writer and the reader cannot drift.

Record (``RECORD_VERSION`` 2; version 1 had only ``env, cpu, torch, created, extras_failed``)::

    {"env", "cpu", "torch", "created", "extras_failed": [file], "record_version": 2,
     "req_sha256": sha256 of the env's requirement files, "app_version": "1.1.0",
     "torch_kind": "cpu" | "cuda" | null (from the installed wheel's local tag, not from the -Cpu flag),
     "git": {checkout name: HEAD (a sha, or "ref: refs/heads/main")}}

The record is deleted when a build of the env starts, so an env whose build failed or was cancelled has none
(``complete`` false in ``env_states``) even though its ``pyvenv.cfg`` exists.

Command line::

    python envrecord.py torch-kind <venv>        prints cpu / cuda / none
    python envrecord.py write --venv <venv> --env <name> --req-dir <envs dir of the repo> --cpu 0|1
                              [--torch "<@torch line>"] [--extras-failed a.txt,b.txt] [--app-version X]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

RECORD = "vndub_env.json"
PROBE = "vndub_probe.json"
RECORD_VERSION = 2
_DIST = re.compile(r"^(?P<name>.+?)-(?P<version>[^-]+)\.dist-info$")


def req_files(req_dir: Path, env: str) -> list[Path]:
    """``<env>.txt``, its ``<env>+<extra>.txt`` files in name order, and ``build.constraints.txt``: what a build reads."""
    req_dir = Path(req_dir)
    files = [req_dir / f"{env}.txt", *sorted(req_dir.glob(f"{env}+*.txt"))]
    cons = req_dir / "build.constraints.txt"
    if cons.exists():
        files.append(cons)
    return [f for f in files if f.exists()]


def req_sha256(req_dir: Path, env: str) -> str:
    """Hash of the requirement files (names + text, line endings normalised: a checkout with CRLF hashes the same)."""
    h = hashlib.sha256()
    for f in req_files(req_dir, env):
        h.update(f.name.encode("utf-8") + b"\0")
        h.update(f.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()


def site_packages(venv: Path) -> Path | None:
    venv = Path(venv)
    win = venv / "Lib" / "site-packages"
    if win.is_dir():
        return win
    for p in sorted((venv / "lib").glob("python*/site-packages")) if (venv / "lib").is_dir() else []:
        return p
    return None


def norm_name(name: str) -> str:
    """PEP 503 normalisation (``ruamel.yaml``, ``ruamel_yaml`` and ``Ruamel-YAML`` are one project)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def installed(venv: Path) -> dict[str, str]:
    """``{normalised name: version}`` from the ``*.dist-info`` folder names (no interpreter started: ~20 ms an env)."""
    sp = site_packages(venv)
    out: dict[str, str] = {}
    if sp is None:
        return out
    try:
        names = os.listdir(sp)
    except OSError:
        return out
    for n in names:
        m = _DIST.match(n)
        if m:
            out[norm_name(m.group("name"))] = m.group("version")
    return out


def torch_kind(venv: Path, have: dict[str, str] | None = None) -> str | None:
    """``cpu`` / ``cuda`` from the installed torch's local version tag (``2.8.0+cpu``, ``2.8.0+cu128``); ``None`` without torch.

    A wheel without a tag came from PyPI, whose Windows torch is CPU-only and whose Linux torch is built with CUDA."""
    v = (have if have is not None else installed(venv)).get("torch")
    if not v:
        return None
    tag = v.partition("+")[2].lower()
    if tag.startswith("cu") or tag.startswith("rocm"):
        return "cuda"
    if tag == "cpu":
        return "cpu"
    return "cpu" if os.name == "nt" else "cuda"


def git_heads(venv: Path) -> dict[str, str]:
    """``{checkout: HEAD}`` of ``<env>/src/*`` (``# @git`` directives): a detached sha, or ``ref: refs/heads/<branch>``."""
    out: dict[str, str] = {}
    src = Path(venv) / "src"
    if not src.is_dir():
        return out
    for d in sorted(src.iterdir()):
        head = d / ".git" / "HEAD"
        try:
            out[d.name] = head.read_text(encoding="utf-8").strip()
        except OSError:
            continue
    return out


def app_version(req_dir: Path) -> str:
    """``VNDUB_APP_VERSION`` (the app passes it), else ``__version__`` of the checkout's ``vndub_app`` (command-line builds)."""
    v = os.environ.get("VNDUB_APP_VERSION")
    if v:
        return v
    try:
        text = (Path(req_dir).parent / "vndub_app" / "__init__.py").read_text(encoding="utf-8")
        m = re.search(r"__version__\s*=\s*['\"]([^'\"]+)", text)
        return m.group(1) if m else ""
    except OSError:
        return ""


def write_record(venv: Path, env: str, req_dir: Path, *, cpu_flag: bool, torch: str = "", extras_failed: list[str] | None = None, version: str | None = None) -> dict:
    venv = Path(venv)
    have = installed(venv)
    kind = torch_kind(venv, have)
    rec = {
        "env": env,
        "cpu": (kind == "cpu") if kind else bool(cpu_flag),  # what is installed, not what was asked for (round 18)
        "torch": torch,
        "torch_kind": kind,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "extras_failed": [x for x in (extras_failed or []) if x],
        "record_version": RECORD_VERSION,
        "req_sha256": req_sha256(req_dir, env),
        "app_version": version if version is not None else app_version(req_dir),
        "git": git_heads(venv),
    }
    tmp = venv / (RECORD + ".tmp")
    tmp.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    os.replace(tmp, venv / RECORD)
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="envrecord.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    tk = sub.add_parser("torch-kind")
    tk.add_argument("venv")
    w = sub.add_parser("write")
    w.add_argument("--venv", required=True)
    w.add_argument("--env", required=True)
    w.add_argument("--req-dir", required=True)
    w.add_argument("--cpu", default="0")
    w.add_argument("--torch", default="")
    w.add_argument("--extras-failed", default="")
    w.add_argument("--app-version", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "torch-kind":
        print(torch_kind(Path(a.venv)) or "none")
        return 0
    extras = [x.strip() for x in re.split(r"[,\s]+", a.extras_failed or "") if x.strip()]
    rec = write_record(Path(a.venv), a.env, Path(a.req_dir), cpu_flag=a.cpu in ("1", "true", "True"), torch=a.torch, extras_failed=extras, version=a.app_version or None)
    print(f"build record: {a.env} torch={rec['torch_kind'] or '-'} req_sha256={rec['req_sha256'][:12]} app={rec['app_version'] or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

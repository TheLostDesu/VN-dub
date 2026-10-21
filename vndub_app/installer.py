"""In-app installer: external tools (pinned manifest) and backend environments.

Endpoints (the router carries full ``/api/...`` paths; include it without a prefix):

* ``GET  /api/tools``           tools from ``app/tools_manifest.json`` with presence on this machine
* ``POST /api/tools/install``   ``{names}`` -> Job; per-file progress in ``job.result.files``
* ``GET  /api/envs``            backend virtual environments (present, size, CPU/CUDA torch)
* ``POST /api/envs/install``    ``{envs, torch: auto|cuda|cpu}`` -> Job (runs ``scripts/make_envs.*``)

Tool downloads go to ``<data>/tools/<name>/<version>/`` (resumable, sha256-checked, extracted
safely), then ``<data>/configs/tools.local.yaml`` (``{tools: {key: path}}``) is updated and the
generated project configs are refreshed. An unreachable URL (or a manual-only entry) raises
``AppError("download_unreachable", {url, filename, target_dir}, action="show_manual")``: the user
saves the file into ``target_dir`` and presses install again, the archive found there is verified
and extracted without another download.

The manifest validator (:func:`validate_manifest`) is strict: ``sha256: null`` is accepted only with
an explicit ``verify`` of ``size`` or ``none`` (with ``unverified_reason``); :func:`manifest_warnings`
lists every entry that is not hash-verified so nothing is unverified silently.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import threading
import time
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable

import yaml
from fastapi import APIRouter
from pydantic import BaseModel

from vndub_app import appdirs, downloads
from vndub_app.errors import AppError
from vndub_app.jobs import JobContext

log = logging.getLogger("vndub_app.installer")

router = APIRouter()

SCHEMA = 1
KINDS = ("download", "pip")
ARCHIVES = ("zip", "tar.gz", "tar.xz", "none")
VERIFY_MODES = ("sha256", "size", "none")
PLATFORMS = ("win32", "linux", "darwin")
PYTOOLS_PYTHON = "3.12"  # siglus-ssu needs >= 3.12; backend envs stay on 3.11
PYTOOLS_DIRNAME = "pytools"
MARKER = ".installed.json"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_PIN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*==[0-9A-Za-z][0-9A-Za-z.+!_-]*$")
_PIPHASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_ENTRY_KEYS = {
    "name", "version", "kind", "manual", "platforms", "engines", "required", "needs_pytools", "homepage",
    "license", "url", "filename", "archive", "size", "sha256", "verify", "unverified_reason", "provides",
    "extra", "wrapper", "pip", "notes",
}
_EXTRA_KEYS = {"url", "filename", "archive", "size", "sha256", "verify", "unverified_reason"}


# ----------------------------------------------------------------------------
# manifest: load + validate
# ----------------------------------------------------------------------------


class ManifestError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("invalid tools manifest:\n  " + "\n  ".join(errors))


def manifest_path() -> Path:
    return appdirs.resource("app", "tools_manifest.json")


def _safe_rel(p: Any) -> bool:
    if not isinstance(p, str) or not p or "\\" in p:
        return False
    pp = PurePosixPath(p)
    return not pp.is_absolute() and ".." not in pp.parts and not re.match(r"^[A-Za-z]:", p)


def _validate_file(ctx: str, d: dict[str, Any], errs: list[str], *, allow_http: bool, manual: bool, pip_entry: bool) -> None:
    """url / filename / archive / size / sha256 / verify rules shared by an entry and its ``extra`` files."""
    url = d.get("url")
    ok_schemes = ("https://", "http://", "file://") if allow_http else ("https://",)
    if not isinstance(url, str) or not url.startswith(ok_schemes):
        errs.append(f"{ctx}: url must start with {' or '.join(ok_schemes)}")
    fn = d.get("filename")
    if manual and fn is None:
        pass
    elif not isinstance(fn, str) or not fn or "/" in fn or "\\" in fn or fn in (".", ".."):
        errs.append(f"{ctx}: filename must be a plain file name")
    arch = d.get("archive")
    if arch not in ARCHIVES:
        errs.append(f"{ctx}: archive must be one of {ARCHIVES}")
    size = d.get("size")
    if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size <= 0):
        errs.append(f"{ctx}: size must be a positive integer or null")
    sha = d.get("sha256")
    if sha is not None and (not isinstance(sha, str) or not _HEX64.match(sha)):
        errs.append(f"{ctx}: sha256 must be 64 lowercase hex digits or null")
    verify = d.get("verify")
    if verify not in VERIFY_MODES:
        errs.append(f"{ctx}: verify must be one of {VERIFY_MODES}")
        return
    if pip_entry:
        return  # pip entries verify through pip.hashes (checked by the caller)
    if verify == "sha256" and sha is None:
        errs.append(f"{ctx}: sha256 is null but verify is 'sha256' (use verify 'size' or 'none' explicitly)")
    if verify == "size" and size is None:
        errs.append(f"{ctx}: verify 'size' needs a size")
    if verify == "none":
        if sha is not None:
            errs.append(f"{ctx}: verify 'none' contradicts a given sha256")
        if not (isinstance(d.get("unverified_reason"), str) and d["unverified_reason"].strip()):
            errs.append(f"{ctx}: verify 'none' needs a non-empty unverified_reason")


def validate_manifest(data: Any, *, allow_http: bool = False) -> list[str]:
    """Return a list of problems (empty = valid). ``allow_http`` also accepts http:// and file:// URLs (tests)."""
    errs: list[str] = []
    if not isinstance(data, dict):
        return ["manifest must be a JSON object"]
    if data.get("schema") != SCHEMA:
        errs.append(f"schema must be {SCHEMA}")
    tools = data.get("tools")
    if not isinstance(tools, list) or not tools:
        errs.append("tools must be a non-empty list")
        return errs
    from vndub.engines import ENGINES
    from vndub.tools import TOOL_INFO

    seen: set[str] = set()
    for i, e in enumerate(tools):
        if not isinstance(e, dict):
            errs.append(f"tools[{i}]: must be an object")
            continue
        name = e.get("name")
        ctx = f"tools[{i}] {name!r}"
        unknown = set(e) - _ENTRY_KEYS
        if unknown:
            errs.append(f"{ctx}: unknown keys {sorted(unknown)}")
        if not isinstance(name, str) or not _NAME.match(name):
            errs.append(f"tools[{i}]: name must match {_NAME.pattern}")
            continue
        if name in seen:
            errs.append(f"{ctx}: duplicate name")
        seen.add(name)
        if not isinstance(e.get("version"), str) or not e["version"]:
            errs.append(f"{ctx}: version must be a non-empty string")
        elif not re.match(r"^[0-9A-Za-z][0-9A-Za-z._+-]*$", e["version"]):
            errs.append(f"{ctx}: version must be usable as a folder name")
        kind = e.get("kind")
        if kind not in KINDS:
            errs.append(f"{ctx}: kind must be one of {KINDS}")
        plats = e.get("platforms")
        if not isinstance(plats, list) or not plats or any(p not in PLATFORMS for p in plats):
            errs.append(f"{ctx}: platforms must be a non-empty list of {PLATFORMS}")
        engines = e.get("engines")
        if not isinstance(engines, list) or not engines or any(x != "*" and x not in ENGINES for x in engines):
            errs.append(f"{ctx}: engines must be a non-empty list of '*' or engine names")
        if not isinstance(e.get("required"), bool):
            errs.append(f"{ctx}: required must be true/false")
        manual = bool(e.get("manual", False))
        prov = e.get("provides")
        if not isinstance(prov, dict) or not prov:
            errs.append(f"{ctx}: provides must map tool keys to relative paths")
        else:
            for key, rel in prov.items():
                if key not in TOOL_INFO:
                    errs.append(f"{ctx}: provides key {key!r} is not a tool in vndub.tools.TOOL_INFO")
                if not _safe_rel(rel):
                    errs.append(f"{ctx}: provides[{key!r}] must be a relative posix path without '..'")
        if not isinstance(e.get("extra", []), list):
            errs.append(f"{ctx}: extra must be a list")
        if kind == "pip":
            pip = e.get("pip")
            if not isinstance(pip, dict):
                errs.append(f"{ctx}: pip entries need a 'pip' object")
            else:
                if not isinstance(pip.get("requirement"), str) or not _PIN.match(pip["requirement"]):
                    errs.append(f"{ctx}: pip.requirement must be pinned as name==version")
                hashes = pip.get("hashes")
                if not isinstance(hashes, list) or not hashes or any(not isinstance(h, str) or not _PIPHASH.match(h) for h in hashes):
                    errs.append(f"{ctx}: pip.hashes must be a non-empty list of sha256:<64 hex>")
                if not isinstance(pip.get("script"), str) or not pip["script"]:
                    errs.append(f"{ctx}: pip.script (console script name) is required")
            if e.get("verify") != "sha256":
                errs.append(f"{ctx}: pip entries are verified through pip.hashes, set verify 'sha256'")
            if e.get("sha256") is not None:
                errs.append(f"{ctx}: pip entries keep sha256 null (hashes live in pip.hashes)")
            for k in ("url", "filename", "archive"):
                if e.get(k) is not None and k != "archive":
                    errs.append(f"{ctx}: pip entries have no {k}")
            if e.get("extra"):
                errs.append(f"{ctx}: pip entries have no extra files")
        elif kind == "download":
            _validate_file(ctx, e, errs, allow_http=allow_http, manual=manual, pip_entry=False)
            for j, x in enumerate(e.get("extra") or []):
                if not isinstance(x, dict):
                    errs.append(f"{ctx}: extra[{j}] must be an object")
                    continue
                bad = set(x) - _EXTRA_KEYS
                if bad:
                    errs.append(f"{ctx}: extra[{j}] unknown keys {sorted(bad)}")
                _validate_file(f"{ctx} extra[{j}]", x, errs, allow_http=allow_http, manual=False, pip_entry=False)
            if e.get("archive") == "none" and isinstance(prov, dict) and e.get("filename"):
                if any(rel != e["filename"] for rel in prov.values()):
                    errs.append(f"{ctx}: archive 'none' means the downloaded file is the tool: provides must equal filename")
            if manual and e.get("verify") == "sha256":
                errs.append(f"{ctx}: manual entries cannot claim sha256 verification")
        wr = e.get("wrapper")
        if wr is not None:
            if not isinstance(wr, dict) or not all(isinstance(wr.get(k), str) and wr[k] for k in ("name", "script", "python")):
                errs.append(f"{ctx}: wrapper needs name, script, python")
            else:
                if not _safe_rel(wr["script"]):
                    errs.append(f"{ctx}: wrapper.script must be a relative posix path")
                if wr["python"] != PYTOOLS_DIRNAME:
                    errs.append(f"{ctx}: wrapper.python must be {PYTOOLS_DIRNAME!r}")
                if not e.get("needs_pytools"):
                    errs.append(f"{ctx}: a wrapper needs needs_pytools: true")
                if isinstance(prov, dict) and wr["name"] not in prov.values():
                    errs.append(f"{ctx}: wrapper.name must be what provides points to")
    return errs


def manifest_warnings(data: dict[str, Any]) -> list[str]:
    """Entries a careful reader should know about: not hash-verified, manual, plain http."""
    out: list[str] = []
    for e in data.get("tools", []):
        name = e.get("name")
        if e.get("manual"):
            out.append(f"{name}: manual download only")
        if e.get("verify") != "sha256":
            out.append(f"{name}: not hash-verified (verify={e.get('verify')}): {e.get('unverified_reason', '')}")
        for x in e.get("extra") or []:
            if x.get("verify") != "sha256":
                out.append(f"{name}/{x.get('filename')}: not hash-verified (verify={x.get('verify')})")
        for u in [e.get("url")] + [x.get("url") for x in e.get("extra") or []]:
            if isinstance(u, str) and u.startswith("http://"):
                out.append(f"{name}: plain http url {u}")
    return out


_MANIFEST_CACHE: dict[str, Any] = {}


def load_manifest(path: Path | None = None, *, allow_http: bool = False) -> dict[str, Any]:
    p = Path(path) if path else manifest_path()
    key = f"{p}|{p.stat().st_mtime_ns if p.exists() else 0}|{allow_http}"
    if key in _MANIFEST_CACHE:
        return _MANIFEST_CACHE[key]
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ManifestError([f"cannot read {p}: {e}"]) from e
    errs = validate_manifest(data, allow_http=allow_http)
    if errs:
        raise ManifestError(errs)
    _MANIFEST_CACHE.clear()
    _MANIFEST_CACHE[key] = data
    return data


# manifest in use: tests point this at a fixture manifest (with file:// / http://127.0.0.1 URLs)
_MANIFEST_OVERRIDE: tuple[Path, bool] | None = None


def set_manifest_override(path: Path | None, *, allow_http: bool = True) -> None:
    global _MANIFEST_OVERRIDE
    _MANIFEST_OVERRIDE = (Path(path), allow_http) if path else None


def entries() -> list[dict[str, Any]]:
    if _MANIFEST_OVERRIDE:
        return load_manifest(_MANIFEST_OVERRIDE[0], allow_http=_MANIFEST_OVERRIDE[1])["tools"]
    return load_manifest()["tools"]


def get_entry(name: str) -> dict[str, Any]:
    for e in entries():
        if e["name"] == name:
            return e
    raise AppError("bad_request", params={"field": "names", "value": name}, details=f"unknown tool {name!r}", status=422)


# ----------------------------------------------------------------------------
# locations and local tool paths
# ----------------------------------------------------------------------------

_LOCAL_LOCK = threading.Lock()


def target_dir(entry: dict[str, Any]) -> Path:
    return appdirs.tools_dir() / entry["name"] / entry["version"]


def pytools_dir() -> Path:
    return appdirs.tools_dir() / PYTOOLS_DIRNAME


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _venv_script(venv: Path, script: str) -> Path:
    return venv / "Scripts" / f"{script}.exe" if os.name == "nt" else venv / "bin" / script


def _wrapper_file(entry: dict[str, Any]) -> Path:
    name = entry["wrapper"]["name"]
    return target_dir(entry) / (f"{name}.cmd" if os.name == "nt" else name)


def read_local_tools() -> dict[str, str]:
    f = appdirs.tools_local_yaml()
    if not f.exists():
        return {}
    try:
        data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    tools = data.get("tools", data) if isinstance(data, dict) else {}
    return {str(k): str(v) for k, v in (tools or {}).items() if v}


def update_local_tools(updates: dict[str, str]) -> dict[str, str]:
    """Merge ``updates`` into ``<data>/configs/tools.local.yaml`` (``{tools: {key: path}}``)."""
    with _LOCAL_LOCK:
        tools = read_local_tools()
        tools.update({k: str(v) for k, v in updates.items()})
        f = appdirs.tools_local_yaml()
        tmp = f.with_suffix(".yaml.tmp")
        tmp.write_text(
            "# written by the VNDub installer (tool name -> executable)\n" + yaml.safe_dump({"tools": tools}, allow_unicode=True, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp, f)
        return tools


def _find_file(base: Path, hint: str) -> Path | None:
    """``hint`` relative to ``base``; if absent, any file with that name below ``base`` (shallowest first)."""
    if not base.is_dir():
        return None
    exact = base / hint
    if exact.is_file():
        return exact
    want = PurePosixPath(hint).name.lower()
    best: tuple[int, Path] | None = None
    for root, dirs, files in os.walk(base):
        depth = len(Path(root).relative_to(base).parts)
        if depth > 6:
            dirs[:] = []
            continue
        for f in files:
            if f.lower() == want and (best is None or depth < best[0]):
                best = (depth, Path(root) / f)
    return best[1] if best else None


def provided_paths(entry: dict[str, Any]) -> dict[str, Path]:
    """Where this entry's executables are installed right now (only the ones that exist)."""
    out: dict[str, Path] = {}
    for key, hint in entry["provides"].items():
        p: Path | None
        if entry["kind"] == "pip":
            p = _venv_script(pytools_dir(), entry["pip"]["script"])
            p = p if p.is_file() else None
        elif entry.get("wrapper"):
            p = _wrapper_file(entry)
            p = p if p.is_file() else None
        else:
            p = _find_file(target_dir(entry), hint)
        if p is not None:
            out[key] = p
    return out


def discover_installed() -> dict[str, str]:
    """Tool paths of everything installed under ``<data>/tools`` (also files the user placed there by hand)."""
    found: dict[str, str] = {}
    for e in entries():
        if os.name != "nt" and "linux" not in e["platforms"]:
            continue
        for key, p in provided_paths(e).items():
            found[key] = str(p)
    return found


def tools_object():
    """``vndub.tools.Tools`` with the installer's results: local yaml first, then discovered files, then PATH."""
    from vndub.tools import Tools

    base = appdirs.resource("configs", "tools.yaml")
    overrides: dict[str, str] = dict(discover_installed())
    for k, v in read_local_tools().items():
        if Path(v).exists():  # a stale path must not hide a working install
            overrides[k] = v
    return Tools.load(base if base.exists() else None, overrides)


def _platform_ok(entry: dict[str, Any]) -> bool:
    plat = "win32" if sys.platform.startswith("win") else ("darwin" if sys.platform == "darwin" else "linux")
    return plat in entry["platforms"]


def list_tools() -> list[dict[str, Any]]:
    tools = tools_object()
    rows = []
    for e in entries():
        keys = list(e["provides"])
        resolved = {k: tools.resolve(k) for k in keys}
        total = [e.get("size")] + [x.get("size") for x in e.get("extra") or []]
        size_mb = round(sum(total) / 1e6, 2) if all(isinstance(t, int) for t in total) else None
        files_verified = e.get("verify") == "sha256" and all(x.get("verify") == "sha256" for x in e.get("extra") or [])
        manual = bool(e.get("manual")) or not _platform_ok(e)
        rows.append(
            {
                "name": e["name"],
                "version": e["version"],
                "kind": e["kind"],
                "present": all(p is not None for p in resolved.values()),
                "path": str(next(iter(p for p in resolved.values() if p), "")) or None,
                "required_for": list(e["engines"]),
                "required": bool(e["required"]),
                "size_mb": size_mb,
                "url": e.get("url") or (e.get("pip") or {}).get("requirement"),
                "filename": e.get("filename") or (e.get("pip") or {}).get("requirement"),
                "sha256": e.get("sha256") or ((e.get("pip") or {}).get("hashes") or [None])[0],
                "verify": e.get("verify"),
                "verified": bool(files_verified),
                "unverified_reason": e.get("unverified_reason"),
                "manual": manual,
                "platform_ok": _platform_ok(e),
                "target_dir": str(target_dir(e)),
                "needs_pytools": bool(e.get("needs_pytools") or e["kind"] == "pip"),
                "homepage": e.get("homepage"),
            }
        )
    return rows


# ----------------------------------------------------------------------------
# archives
# ----------------------------------------------------------------------------


def _check_member(dest: Path, name: str) -> Path:
    target = (dest / name).resolve()
    if not str(target).startswith(str(dest.resolve()) + os.sep) and target != dest.resolve():
        raise AppError("tool_install_failed", params={"reason": "unsafe_archive"}, action="show_manual", details=f"archive member escapes the target: {name}")
    return target


def extract_archive(archive: Path, dest: Path, kind: str) -> list[str]:
    """Extract ``archive`` into ``dest`` (created) refusing paths that escape it. Returns top-level names."""
    dest.mkdir(parents=True, exist_ok=True)
    if kind == "zip":
        with zipfile.ZipFile(archive) as z:
            for m in z.infolist():
                _check_member(dest, m.filename)
            z.extractall(dest)
    elif kind in ("tar.gz", "tar.xz"):
        with tarfile.open(archive, "r:*") as t:
            for m in t.getmembers():
                _check_member(dest, m.name)
                if m.issym() or m.islnk():
                    raise AppError("tool_install_failed", params={"reason": "unsafe_archive"}, action="show_manual", details=f"link in archive: {m.name}")
            t.extractall(dest)
    else:
        raise ValueError(f"cannot extract archive kind {kind!r}")
    return sorted(p.name for p in dest.iterdir())


def _install_extracted(archive: Path, tdir: Path, kind: str) -> None:
    """Extract into a scratch folder inside ``tdir`` and move the top-level entries up (all or nothing)."""
    scratch = tdir / f".x-{uuid.uuid4().hex[:8]}"
    try:
        names = extract_archive(archive, scratch, kind)
        for n in names:
            dst = tdir / n
            if dst.exists():
                shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
            shutil.move(str(scratch / n), str(dst))
    except (zipfile.BadZipFile, tarfile.TarError, OSError, EOFError) as e:
        raise AppError("tool_install_failed", params={"reason": "bad_archive", "filename": archive.name}, action="show_manual", details=f"{type(e).__name__}: {e}") from e
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# ----------------------------------------------------------------------------
# progress rows
# ----------------------------------------------------------------------------


class _Rows:
    """Per-file progress rows published through ``job.result.files`` (throttled)."""

    def __init__(self, ctx: JobContext) -> None:
        self.ctx = ctx
        self.rows: list[dict[str, Any]] = []
        self._last = 0.0

    def add(self, row_id: str, tool: str, filename: str, total: int | None = None) -> dict[str, Any]:
        row = {"id": row_id, "name": filename, "tool": tool, "filename": filename, "total": total or 0, "done": 0, "state": "pending", "verified": None, "skipped": False, "error": None}
        self.rows.append(row)
        return row

    def set(self, row: dict[str, Any], force: bool = False, **kv: Any) -> None:
        row.update(kv)
        now = time.monotonic()
        if force or now - self._last >= 0.3:
            self._last = now
            self.flush()

    def flush(self) -> None:
        self.ctx.set_result(files=[dict(r) for r in self.rows])
        total = sum(r["total"] for r in self.rows)
        done = sum(min(r["done"], r["total"]) for r in self.rows if r["total"])
        if total:
            # job-level counter in megabytes (bytes would be unreadable); exact bytes are in `files`
            t_mb = max(1, round(total / 1e6))
            self.ctx.progress((t_mb if done >= total else min(t_mb, int(done / 1e6))), t_mb, next((r["filename"] for r in self.rows if r["state"] in ("downloading", "verifying", "extracting", "installing")), None))
        else:
            self.ctx.checkpoint()


# ----------------------------------------------------------------------------
# subprocess helpers (uv, make_envs)
# ----------------------------------------------------------------------------


def find_uv() -> Path | None:
    """The bundled ``uv`` (packaged app), else one on PATH, else ``~/.local/bin`` (dev machines)."""
    exe = "uv.exe" if os.name == "nt" else "uv"
    for cand in (appdirs.resource("bin", exe), appdirs.resource(exe)):
        if cand.is_file():
            return cand
    found = shutil.which("uv")
    if found:
        return Path(found)
    home = Path.home() / ".local" / "bin" / exe
    return home if home.is_file() else None


def _child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    uv = find_uv()
    if uv is not None:
        env["PATH"] = str(uv.parent) + os.pathsep + env.get("PATH", "")
    env.setdefault("PYTHONUTF8", "1")
    env["VNDUB_ENVS_DIR"] = str(envs_dir())
    env.update(extra or {})
    return env


def _kill_tree(proc: subprocess.Popen[str]) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            proc.terminate()
    except Exception:
        pass


def run_stream(cmd: list[str], ctx: JobContext, *, cwd: Path | None = None, env: dict[str, str] | None = None, on_line: Callable[[str], None] | None = None) -> tuple[int, list[str]]:
    """Run ``cmd``, forward every output line to the job log, honour pause/cancel. Returns ``(exit code, last lines)``."""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    proc = subprocess.Popen(
        cmd, cwd=str(cwd) if cwd else None, env=env or _child_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", creationflags=flags, bufsize=1,
    )
    tail: list[str] = []
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\r\n")
            if not line.strip():
                continue
            ctx.log(line)
            tail.append(line)
            del tail[:-40]
            if on_line:
                on_line(line)
            ctx.checkpoint()
        rc = proc.wait()
    except BaseException:
        _kill_tree(proc)
        raise
    finally:
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass
    return rc, tail


_NET_ERR = re.compile(r"dns error|error sending request|failed to fetch|connection (?:refused|reset|timed out)|timed out|could not resolve|network is unreachable|no route to host|ssl", re.I)


def _looks_like_network(lines: list[str]) -> bool:
    return any(_NET_ERR.search(l) for l in lines)


# ----------------------------------------------------------------------------
# installing tools
# ----------------------------------------------------------------------------


def _manual_error(entry: dict[str, Any], reason: str = "manual") -> AppError:
    tdir = target_dir(entry)
    tdir.mkdir(parents=True, exist_ok=True)  # the card says "put the file here": the folder must exist and open
    return AppError(
        "download_unreachable",
        params={
            "url": entry.get("url") or (entry.get("pip") or {}).get("requirement", ""),
            "filename": entry.get("filename") or "",
            "target_dir": str(tdir),
            "tool": entry["name"],
            "reason": reason,
        },
        action="show_manual",
        details=f"{entry['name']}: {reason}",
        status=502,
    )


def _is_installed(entry: dict[str, Any]) -> bool:
    marker = target_dir(entry) / MARKER
    if entry["kind"] == "download" and not marker.exists():
        return False
    return len(provided_paths(entry)) == len(entry["provides"])


UTF8_SITECUSTOMIZE = (
    "# written by VNDub: tools print Japanese file names and text; a pipe would otherwise use the\n"
    "# console code page (cp1251 / cp932) and crash with UnicodeEncodeError. The app reads their output as UTF-8.\n"
    "import sys\n"
    "for _s in (sys.stdout, sys.stderr):\n"
    "    try:\n"
    "        _s.reconfigure(encoding='utf-8', errors='replace')\n"
    "    except Exception:\n"
    "        pass\n"
)


def ensure_utf8_output(py: Path) -> Path | None:
    """Make every script run by ``py`` write UTF-8 to pipes (``sitecustomize.py`` in its site-packages)."""
    try:
        r = subprocess.run(
            [str(py), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
            capture_output=True, text=True, timeout=60, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
        )
        site = Path(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 and r.stdout.strip() else None
    except (OSError, subprocess.SubprocessError, IndexError):
        site = None
    if site is None or not site.is_dir():
        return None
    f = site / "sitecustomize.py"
    if not f.exists() or f.read_text(encoding="utf-8", errors="replace") != UTF8_SITECUSTOMIZE:
        f.write_text(UTF8_SITECUSTOMIZE, encoding="utf-8")
    return f


def ensure_pytools(ctx: JobContext) -> Path:
    """The Python 3.12 tools environment (``<data>/tools/pytools``) used by pip-installed tools and unrpyc."""
    venv = pytools_dir()
    py = _venv_python(venv)
    if py.is_file() and (venv / "pyvenv.cfg").is_file():
        ensure_utf8_output(py)
        return py
    uv = find_uv()
    if uv is None:
        raise AppError("tool_missing", params={"tool": "uv"}, action="show_manual", details="uv was not found (not bundled, not on PATH)")
    if venv.exists():
        shutil.rmtree(venv, ignore_errors=True)
    ctx.message("installer.msg.pytools")
    rc, tail = run_stream([str(uv), "venv", str(venv), "--python", PYTOOLS_PYTHON], ctx)
    if rc != 0 or not py.is_file():
        if _looks_like_network(tail):
            raise AppError("download_unreachable", params={"url": "https://github.com/astral-sh/python-build-standalone/releases", "filename": f"Python {PYTOOLS_PYTHON}", "target_dir": str(venv)}, action="show_manual", details="\n".join(tail))
        raise AppError("tool_install_failed", params={"tool": "pytools"}, action="retry", details="\n".join(tail))
    ensure_utf8_output(py)
    return py


def _write_wrapper(entry: dict[str, Any], py: Path) -> Path:
    script = _find_file(target_dir(entry), entry["wrapper"]["script"])
    if script is None:
        raise AppError("tool_install_failed", params={"tool": entry["name"], "reason": "script_missing"}, action="show_manual", details=entry["wrapper"]["script"])
    w = _wrapper_file(entry)
    if os.name == "nt":
        w.write_text(f'@echo off\r\n"{py}" "{script}" %*\r\n', encoding="utf-8")
    else:
        w.write_text(f'#!/bin/sh\nexec "{py}" "{script}" "$@"\n', encoding="utf-8")
        w.chmod(w.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return w


def _download_files(entry: dict[str, Any]) -> list[dict[str, Any]]:
    primary = {k: entry.get(k) for k in ("url", "filename", "archive", "size", "sha256", "verify")}
    return [primary] + [dict(x) for x in entry.get("extra") or []]


def _install_download(entry: dict[str, Any], ctx: JobContext, rows: _Rows, row_ids: list[dict[str, Any]]) -> None:
    tdir = target_dir(entry)
    tdir.mkdir(parents=True, exist_ok=True)
    for spec, row in zip(_download_files(entry), row_ids):
        dest = tdir / spec["filename"]
        verify = spec.get("verify")
        sha = spec.get("sha256") if verify == "sha256" else None
        size = spec.get("size") if verify in ("sha256", "size") else None
        rows.set(row, force=True, state="downloading", total=spec.get("size") or 0)

        def cb(done: int, total: int, _row: dict[str, Any] = row) -> None:
            if total and not _row["total"]:
                _row["total"] = total
            rows.set(_row, done=done)

        res = downloads.fetch(
            spec["url"], dest, expected_sha256=sha, expected_size=size, progress=cb, checkpoint=ctx.checkpoint, target_dir=tdir,
        )
        rows.set(row, force=True, done=res.size, total=res.size, state="installing" if spec["archive"] == "none" else "extracting", verified=bool(res.verified))
        if verify == "none":
            ctx.log(f"WARNING: {spec['filename']} cannot be hash-verified ({entry.get('unverified_reason', '')})")
        if spec["archive"] != "none":
            _install_extracted(dest, tdir, spec["archive"])
            dest.unlink(missing_ok=True)
        rows.set(row, force=True, state="done")


def _install_pip(entry: dict[str, Any], ctx: JobContext, rows: _Rows, row: dict[str, Any]) -> None:
    pip = entry["pip"]
    rows.set(row, force=True, state="installing")
    py = ensure_pytools(ctx)
    uv = find_uv()
    assert uv is not None  # ensure_pytools checked
    req = appdirs.sub("tools") / f".req-{entry['name']}-{uuid.uuid4().hex[:6]}.txt"
    req.write_text(pip["requirement"] + "".join(f" --hash={h}" for h in pip["hashes"]) + "\n", encoding="utf-8")
    try:
        ctx.message("installer.msg.pip", tool=entry["name"])
        rc, tail = run_stream([str(uv), "pip", "install", "--python", str(py), "--no-deps", "--require-hashes", "-r", str(req)], ctx)
    finally:
        req.unlink(missing_ok=True)
    if rc != 0:
        if any("hash" in l.lower() and ("mismatch" in l.lower() or "do not match" in l.lower()) for l in tail):
            raise AppError("checksum_mismatch", params={"filename": pip["requirement"], "expected": ",".join(pip["hashes"]), "actual": ""}, action="retry", details="\n".join(tail))
        if _looks_like_network(tail):
            raise AppError("download_unreachable", params={"url": f"https://pypi.org/project/{pip['requirement'].split('==')[0]}/", "filename": pip["requirement"], "target_dir": str(pytools_dir())}, action="show_manual", details="\n".join(tail))
        raise AppError("tool_install_failed", params={"tool": entry["name"]}, action="retry", details="\n".join(tail))
    rows.set(row, force=True, state="done", verified=True)


def install_tools(names: list[str], ctx: JobContext) -> dict[str, Any]:
    """Job body for ``POST /api/tools/install`` (also used by the Doctor's ``download_ffmpeg``)."""
    chosen = [get_entry(n) for n in dict.fromkeys(names)]
    rows = _Rows(ctx)
    plan: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    need_bytes = 0
    for e in chosen:
        if e["kind"] == "pip":
            plan.append((e, [rows.add(f"{e['name']}:{e['pip']['requirement']}", e["name"], e["pip"]["requirement"])]))
            continue
        specs = _download_files(e)
        plan.append((e, [rows.add(f"{e['name']}:{s['filename']}", e["name"], s["filename"], s.get("size")) for s in specs]))
        if not (e.get("manual") or not _platform_ok(e)) and not _is_installed(e):
            need_bytes += sum(s.get("size") or 0 for s in specs)
    rows.flush()
    # room for the download plus the extracted copy
    free = downloads.free_space_gb(appdirs.tools_dir())
    if need_bytes and free * 1e9 < need_bytes * 2.2:
        raise AppError("disk_full", params={"need_gb": round(need_bytes * 2.2 / 1e9, 1), "free_gb": round(free, 1), "path": str(appdirs.tools_dir())}, action="open_data_dir")

    installed: list[str] = []
    failed: list[dict[str, Any]] = []
    unverified: list[str] = []
    first_error: AppError | None = None
    for e, row_list in plan:
        ctx.checkpoint()
        ctx.message("installer.msg.tool", tool=e["name"])
        py: Path | None = None
        try:
            placed_by_hand = bool(e.get("manual")) and len(provided_paths(e)) == len(e["provides"])
            if (e.get("manual") or not _platform_ok(e)) and not placed_by_hand:
                raise _manual_error(e, "manual" if e.get("manual") else "platform")
            if placed_by_hand or _is_installed(e):
                for r in row_list:
                    rows.set(r, force=True, state="done", skipped=True, done=r["total"])
            elif e["kind"] == "pip":
                _install_pip(e, ctx, rows, row_list[0])
            else:
                if e.get("needs_pytools"):
                    py = ensure_pytools(ctx)
                _install_download(e, ctx, rows, row_list)
                if e.get("wrapper"):
                    _write_wrapper(e, py)
                (target_dir(e) / MARKER).write_text(
                    json.dumps({"name": e["name"], "version": e["version"], "installed": time.strftime("%Y-%m-%dT%H:%M:%S"), "verify": e["verify"]}), encoding="utf-8"
                )
            paths = provided_paths(e)
            missing = [k for k in e["provides"] if k not in paths]
            if missing:
                raise AppError("tool_install_failed", params={"tool": e["name"], "reason": "executable_missing", "missing": missing}, action="show_manual", details=f"not found after install: {missing}")
            update_local_tools({k: str(p) for k, p in paths.items()})
            installed.append(e["name"])
            if e.get("verify") != "sha256":
                unverified.append(e["name"])
        except AppError as err:
            if err.code == "cancelled":
                raise
            for r in row_list:
                if r["state"] != "done":
                    rows.set(r, force=True, state="failed", error=err.code)
            failed.append({"name": e["name"], "code": err.code, "params": err.params})
            first_error = first_error or err
            ctx.log(f"ERROR {e['name']}: {err.code} {err.details.splitlines()[0] if err.details else ''}")
    rows.flush()
    ctx.set_result(files=[dict(r) for r in rows.rows], installed=installed, failed=failed, unverified=unverified)
    if installed:
        refresh_project_configs()
    if first_error is not None:
        raise first_error
    return {"installed": installed, "unverified": unverified}


def refresh_project_configs() -> None:
    """Regenerate every project's pipeline.yaml so new tool paths are picked up."""
    for h in appdirs.list_projects():
        try:
            appdirs.write_project_config(h)
        except Exception as e:  # a broken project must not fail an installer job
            log.warning("cannot refresh config of %s: %s", h.slug, e)


# ----------------------------------------------------------------------------
# backend environments
# ----------------------------------------------------------------------------


def repo_envs_dir() -> Path:
    return appdirs.resource_root() / ".venvs"


def configure_environment() -> Path:
    """Decide where backend venvs live and export it as ``VNDUB_ENVS_DIR`` (idempotent; keeps an explicit setting).

    Packaged app: ``<data>/envs``. Dev checkout: the repo's ``.venvs`` when it already holds environments
    (so ``uv run vndub-app`` sees what ``scripts/make_envs`` built), else ``<data>/envs``.
    """
    env = os.environ.get("VNDUB_ENVS_DIR")
    if env:
        return Path(env)
    chosen = appdirs.envs_dir()
    if not getattr(sys, "frozen", False):
        legacy = repo_envs_dir()
        if legacy.is_dir() and any((d / "pyvenv.cfg").exists() for d in legacy.iterdir() if d.is_dir()):
            chosen = legacy
    os.environ["VNDUB_ENVS_DIR"] = str(chosen)
    return chosen


def envs_dir() -> Path:
    """Where backend venvs live (also exports ``VNDUB_ENVS_DIR`` so the pipeline in this process agrees)."""
    env = os.environ.get("VNDUB_ENVS_DIR")
    return Path(env) if env else configure_environment()


def _env_requirement_text(name: str) -> str | None:
    f = appdirs.resource("envs", f"{name}.txt")
    try:
        return f.read_text(encoding="utf-8")
    except OSError:
        return None


def env_names() -> list[str]:
    """Environments that have something to install (``envs/<name>.txt`` with requirements or directives)."""
    d = appdirs.resource("envs")
    names = []
    if d.is_dir():
        for f in sorted(d.glob("*.txt")):
            if "+" in f.stem or "." in f.stem:
                continue
            txt = f.read_text(encoding="utf-8", errors="replace")
            if any(re.match(r"^\s*[^#\s]", l) or re.match(r"^\s*#\s*@", l) for l in txt.splitlines()):
                names.append(f.stem)
    return names


def env_needs_git(name: str) -> bool:
    txt = _env_requirement_text(name) or ""
    return bool(re.search(r"^\s*#\s*@git\s", txt, re.M))


def _dir_size_gb(p: Path) -> float:
    total = 0
    stack = [p]
    while stack:
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(Path(e.path))
                        elif e.is_file(follow_symlinks=False):
                            total += e.stat(follow_symlinks=False).st_size
                    except OSError:
                        pass
        except OSError:
            pass
    return total / 1e9


_SIZE_CACHE: dict[str, tuple[float, float]] = {}


def _env_torch(name: str) -> str | None:
    f = envs_dir() / name / "vndub_env.json"
    try:
        data = json.loads(f.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    if "cpu" in data:
        return "cpu" if data["cpu"] else "cuda"
    return None


def list_envs(with_size: bool = True) -> list[dict[str, Any]]:
    from vndub.backends import registry
    from vndub.backends.runner import env_exists

    users: dict[str, list[str]] = {}
    for s in registry.all_specs().values():
        if s.env:
            users.setdefault(s.env, []).append(s.name)
    rows = []
    for name in env_names():
        present = env_exists(name)
        size = None
        if present and with_size:
            key = f"{envs_dir()}|{name}"
            cached = _SIZE_CACHE.get(key)
            if cached and time.monotonic() - cached[0] < 120:
                size = cached[1]
            else:
                size = round(_dir_size_gb(envs_dir() / name), 2)
                _SIZE_CACHE[key] = (time.monotonic(), size)
        rows.append(
            {
                "name": name,
                "present": present,
                "size_gb": size,
                "torch": _env_torch(name) if present else None,
                "needs_git": env_needs_git(name),
                "backends": sorted(users.get(name, [])),
            }
        )
    return rows


def resolve_torch(choice: str) -> str:
    if choice in ("cuda", "cpu"):
        return choice
    from vndub_app import doctor

    return doctor.choose_torch()


def build_make_envs_command(envs: list[str], torch: str, envs_path: Path) -> tuple[list[str], dict[str, str]]:
    """Command line + extra environment for ``scripts/make_envs.ps1`` (Windows) / ``.sh``."""
    scripts = appdirs.resource("scripts")
    if os.name == "nt":
        ps = shutil.which("powershell") or "powershell.exe"
        cmd = [ps, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(scripts / "make_envs.ps1"), "-Env", ",".join(envs), "-EnvsDir", str(envs_path)]
        if torch == "cpu":
            cmd.append("-Cpu")
        return cmd, {}
    extra = {"ENVS_DIR": str(envs_path)}
    if torch == "cpu":
        extra["CPU"] = "1"
    return ["bash", str(scripts / "make_envs.sh"), *envs], extra


_ENV_HEADER = re.compile(r"^==\s+env\s+(\S+)")


def install_envs(envs: list[str], torch: str, ctx: JobContext) -> dict[str, Any]:
    """Job body for ``POST /api/envs/install``: run make_envs, stream its output, verify the result."""
    from vndub.backends.runner import env_exists

    known = env_names()
    names = list(known) if "all" in envs else list(dict.fromkeys(envs))
    bad = [n for n in names if n not in known]
    if bad or not names:
        raise AppError("bad_request", params={"field": "envs", "value": ",".join(bad)}, details=f"unknown envs: {bad}; known: {known}", status=422)
    torch_kind = resolve_torch(torch)
    if shutil.which("git") is None:
        needing = [n for n in names if env_needs_git(n)]
        if needing:
            raise AppError("git_missing", params={"envs": needing}, action="show_manual", details=f"git not found on PATH; needed by {needing}")
    uv = find_uv()
    if uv is None:
        raise AppError("tool_missing", params={"tool": "uv"}, action="show_manual", details="uv was not found (not bundled, not on PATH)")
    path = envs_dir()
    path.mkdir(parents=True, exist_ok=True)
    rows = _Rows(ctx)
    row_by = {n: rows.add(f"env:{n}", n, n) for n in names}
    ctx.message("installer.msg.envs", torch=torch_kind, count=len(names))
    cmd, extra = build_make_envs_command(names, torch_kind, path)
    current: list[str | None] = [None]

    def on_line(line: str) -> None:
        m = _ENV_HEADER.match(line)
        if m and m.group(1) in row_by:
            prev = current[0]
            if prev:
                rows.set(row_by[prev], force=True, state="done")
            current[0] = m.group(1)
            rows.set(row_by[m.group(1)], force=True, state="installing")
            ctx.message("installer.msg.env", env=m.group(1))
            ctx.progress(names.index(m.group(1)), len(names), m.group(1))

    rc, tail = run_stream(cmd, ctx, env=_child_env(extra), on_line=on_line)
    created = [n for n in names if env_exists(n)]
    for n in names:
        rows.set(row_by[n], force=True, state="done" if n in created else "failed", error=None if n in created else "env_install_failed")
    ctx.set_result(files=[dict(r) for r in rows.rows], envs=created, torch=torch_kind)
    _SIZE_CACHE.clear()
    if rc != 0 or len(created) != len(names):
        if _looks_like_network(tail):
            raise AppError("download_unreachable", params={"url": "https://download.pytorch.org/whl/", "filename": ", ".join(names), "target_dir": str(path)}, action="show_manual", details="\n".join(tail))
        raise AppError("env_install_failed", params={"env": current[0] or ",".join(names), "exit_code": rc}, action="retry", details="\n".join(tail))
    ctx.progress(len(names), len(names), None)
    return {"envs": created, "torch": torch_kind}


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------


class ToolsInstallBody(BaseModel):
    names: list[str]


class EnvsInstallBody(BaseModel):
    envs: list[str]
    torch: str = "auto"


@router.get("/api/tools")
def api_tools() -> list[dict[str, Any]]:
    return list_tools()


@router.post("/api/tools/install")
def api_tools_install(body: ToolsInstallBody) -> dict[str, Any]:
    from vndub_app import jobs

    if not body.names:
        raise AppError("bad_request", params={"field": "names"}, details="names must not be empty", status=422)
    for n in body.names:
        get_entry(n)  # unknown name -> 422 before a job is created
    names = list(body.names)
    job = jobs.runner().submit("tools_install", lambda ctx: install_tools(names, ctx), params={"names": names}, gpu=False)
    return job.info()


@router.get("/api/envs")
def api_envs() -> list[dict[str, Any]]:
    return list_envs()


@router.post("/api/envs/install")
def api_envs_install(body: EnvsInstallBody) -> dict[str, Any]:
    from vndub_app import jobs

    if body.torch not in ("auto", "cuda", "cpu"):
        raise AppError("bad_request", params={"field": "torch", "value": body.torch}, status=422)
    known = env_names()
    bad = [n for n in body.envs if n != "all" and n not in known]
    if bad or not body.envs:
        raise AppError("bad_request", params={"field": "envs", "value": ",".join(bad)}, details=f"unknown envs {bad}", status=422)
    envs, torch = list(body.envs), body.torch
    job = jobs.runner().submit("envs_install", lambda ctx: install_envs(envs, torch, ctx), params={"envs": envs, "torch": torch}, gpu=False)
    return job.info()

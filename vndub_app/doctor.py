r"""Diagnostics ("Doctor", wizard step 0): green / yellow / red rows with a fix where one exists.

``GET  /api/doctor``                   ``{summary, gpu, rows, torch, mock, profile, data_dir}``
``POST /api/doctor/fix/{action}``      ``download_ffmpeg`` (Job) | ``install_vcredist`` (Job) | ``open_data_dir``
                                      | ``open_driver_page`` | ``open_vcredist_page`` | ``add_defender_exclusion_help``
                                      | ``kill_processes`` (the first two answer with a Job, ``kill_processes`` with
                                      ``{ok, killed}`` at once, the others with ``{ok, ...}``)

The router carries full ``/api/...`` paths; include it without a prefix.

Rows (ids): ``gpu driver vram disk ffmpeg vcredist git internet defender envs models processes``. Each row is
``{id, status: green|yellow|red, value, params, fix: {action, label_key}|null}``; the UI shows
``doctor.<id>.title`` and ``doctor.<id>.<status>`` interpolated with ``params``. The backend sends
no display text. ``fix.action`` is either one of the fix actions above (button label
``doctor.fix.<action>``) or a navigation action from ``errors.ACTIONS`` (``open_tools`` /
``open_models``, label ``actions.<action>``).

Thresholds (also in docs/decisions/O.md):

* VRAM (nvidia-smi MiB / 1024): green >= 7.9 GB (an "8 GB" card reports 8188..8192 MiB), yellow
  5.9..7.9 GB, red below or without an NVIDIA GPU.
* Driver (CUDA version nvidia-smi reports): green >= 12.8 (the cu128 wheels of most envs), yellow
  12.1..12.8 (older wheels only), red below 12.1 / no driver.
* Disk: needed = what is still to download for the default setup (+10 GB working room);
  green >= needed, yellow >= 60 % of it, red below.
* Visual C++ runtime (Windows only): green when HKLM (64-bit view) ``SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64``
  says ``Installed=1`` with 14.40 or newer (binaries built with VS 2022 17.10+ need that), or when ``msvcp140.dll`` and
  ``vcruntime140_1.dll`` in System32 are 14.40+; red otherwise (no model starts without it). The fix runs the
  ``redist\vc_redist.x64.exe`` next to ``VNDub.exe`` (a job), or, without that file (a dev run), opens Microsoft's page.
* Git: green when ``git`` is on PATH or bundled (``git\cmd\git.exe`` next to ``VNDub.exe``); yellow otherwise, naming the
  envs whose installation clones repositories.
* Processes: yellow while orphans are running (``winproc`` ledger: a backend ``python.exe`` / ``llama-server`` / tool under the
  environments or tools folders whose parent is gone; ``value`` and ``params.count`` = how many, ``params.names`` =
  ``"python.exe x2, llama-server.exe"``), green otherwise. The fix ``kill_processes`` terminates them and returns
  ``{ok: true, killed: n}``; the processes of a running job are not orphans and are never touched.

Every probe is injectable (:class:`Probes`) so tests need neither a GPU nor a network. The Windows
Defender row is a manual card: the app never changes security settings, it only shows the command.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter

from vndub_app import appdirs, openers
from vndub_app.errors import AppError

log = logging.getLogger("vndub_app.doctor")

router = APIRouter()

NVIDIA_DRIVER_URL = "https://www.nvidia.com/Download/index.aspx"
INTERNET_HOSTS: tuple[tuple[str, int], ...] = (("huggingface.co", 443), ("github.com", 443))
INTERNET_TIMEOUT_S = 3.0
INTERNET_CACHE_S = 30.0

VRAM_GREEN_GB = 7.9
VRAM_YELLOW_GB = 5.9
VRAM_NEEDED_GB = 8
CUDA_GREEN = (12, 8)
CUDA_MIN = (12, 1)
WORK_ROOM_GB = 10.0
ENV_ESTIMATE_GB = 4.0
ENV_ESTIMATE_BY_NAME_GB = {"main": 9.0}  # torch cu128 + the asr/hf/score packages in one venv; the others are smaller
DISK_YELLOW_SHARE = 0.6
DISK_FLOOR_GB = 8.0

VC_REG_KEY = r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"
VC_MIN = (14, 40)  # binaries built with VS 2022 17.10+ (llama.cpp b11469, torch 2.8) need runtime 14.40+
VC_DLLS = ("msvcp140.dll", "vcruntime140_1.dll")

FIX_ACTIONS = ("download_ffmpeg", "install_vcredist", "open_data_dir", "open_driver_page", "open_vcredist_page", "add_defender_exclusion_help", "kill_processes")
ROW_IDS = ("gpu", "driver", "vram", "disk", "ffmpeg", "vcredist", "git", "internet", "defender", "envs", "models", "processes")


# ----------------------------------------------------------------------------
# probes
# ----------------------------------------------------------------------------


@dataclass
class GpuInfo:
    name: str
    vram_gb: float | None
    driver: str
    cuda: str | None
    count: int = 1

    def public(self) -> dict[str, Any]:
        return {"name": self.name, "vram_gb": self.vram_gb, "driver": self.driver, "cuda": self.cuda}


def _no_window() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform.startswith("win") else 0


def find_nvidia_smi() -> str | None:
    found = shutil.which("nvidia-smi")
    if found:
        return found
    if sys.platform.startswith("win"):
        for c in (r"C:\Windows\System32\nvidia-smi.exe", r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"):
            if os.path.exists(c):
                return c
    return None


def parse_nvidia_smi(csv_text: str, header_text: str = "") -> GpuInfo | None:
    """``--query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits`` (+ the plain header for CUDA)."""
    rows = [l.strip() for l in csv_text.strip().splitlines() if l.strip()]
    if not rows:
        return None
    pieces = rows[0].rsplit(",", 2)  # the name itself may contain commas
    if len(pieces) < 3:
        return None
    name, mem, driver = (x.strip() for x in pieces)
    try:
        vram: float | None = round(float(mem) / 1024.0, 2)
    except ValueError:  # "[N/A]" on some mobile parts
        vram = None
    m = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", header_text)
    return GpuInfo(name=name or "NVIDIA GPU", vram_gb=vram, driver=driver, cuda=m.group(1) if m else None, count=len(rows))


def probe_gpu() -> GpuInfo | None:
    exe = find_nvidia_smi()
    if exe is None:
        return None
    try:
        q = subprocess.run(
            [exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, creationflags=_no_window(),
        )
        if q.returncode != 0 or not q.stdout.strip():
            return None
        header = ""
        try:
            h = subprocess.run([exe], capture_output=True, text=True, timeout=10, creationflags=_no_window())
            header = h.stdout if h.returncode == 0 else ""
        except (OSError, subprocess.SubprocessError):
            pass
        return parse_nvidia_smi(q.stdout, header)
    except (OSError, subprocess.SubprocessError) as e:
        log.info("nvidia-smi failed: %s", e)
        return None


def probe_disk_free_gb(path: Path) -> float:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def probe_internet() -> bool:
    """TCP connect only (no request is sent) to the hosts the downloads use."""
    for host, port in INTERNET_HOSTS:
        try:
            with socket.create_connection((host, port), timeout=INTERNET_TIMEOUT_S):
                return True
        except OSError:
            continue
    return False


def probe_tools() -> dict[str, dict[str, Any]]:
    """Required tools via ``vndub tools check`` logic (``Tools.check``), including ``<data>/tools``."""
    from vndub_app import installer

    rows = installer.tools_object().check()
    return {r["tool"]: {"present": bool(r["present"]), "path": r["path"] or None, "required": bool(r["required"])} for r in rows}


def probe_envs() -> dict[str, Any]:
    """Envs the default profile needs and which of them exist."""
    from vndub.backends import registry
    from vndub.backends.runner import env_exists

    needed = sorted({s.env for s in registry.for_profile("default") if s.env})
    return {"needed": needed, "present": [e for e in needed if env_exists(e)]}


def probe_models() -> dict[str, Any]:
    from vndub_app import models_api

    return models_api.summary()


def probe_profile() -> str:
    return appdirs.load_settings().profile


def probe_processes() -> list[Any]:
    """Orphans: processes under the app's environments / tools folders that nobody who is still running started (``winproc.ProcInfo``)."""
    from vndub_app import installer, winproc

    return [i for i in winproc.list_processes(installer.process_roots()) if i.orphan and not i.ours]


@dataclass
class VcRuntime:
    """What :func:`decide_vcruntime` found: ``ok`` = 14.40 or newer is there; ``version`` = ``"14.44.35211"`` (or ``None``)."""

    ok: bool
    version: str | None
    source: str  # registry | files | none
    applicable: bool = True


def _fmt_version(v: tuple[int, ...]) -> str:
    return ".".join(str(x) for x in v[:3])


def decide_vcruntime(reg: Callable[[str], int | None], dll_version: Callable[[str], tuple[int, int, int, int] | None]) -> VcRuntime:
    """Is the Visual C++ 2015-2022 runtime there and new enough?

    ``reg(name)`` reads a DWORD of the ``Runtimes\\x64`` key, ``dll_version(name)`` the file version of a System32 DLL
    (both injectable for tests). The registry decides; the DLLs are the fallback when it says nothing useful.
    """
    major, minor = reg("Major"), reg("Minor")
    registry: tuple[int, int, int] | None = None
    if reg("Installed") == 1 and major is not None and minor is not None:
        registry = (major, minor, reg("Bld") or 0)
        if (major, minor) >= VC_MIN:
            return VcRuntime(True, _fmt_version(registry), "registry")
    dlls = [dll_version(n) for n in VC_DLLS]  # the registry value can be missing although the files are there
    if all(v is not None and v[:2] >= VC_MIN for v in dlls):
        return VcRuntime(True, _fmt_version(min(v for v in dlls if v)), "files")
    if registry is not None:
        return VcRuntime(False, _fmt_version(registry), "registry")
    found = [v for v in dlls if v]
    if found:
        return VcRuntime(False, _fmt_version(min(found)), "files")
    return VcRuntime(False, None, "none")


def _read_vc_registry() -> dict[str, int]:
    import winreg

    out: dict[str, int] = {}
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, VC_REG_KEY, 0, winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as key:
            for name in ("Installed", "Major", "Minor", "Bld"):
                try:
                    value, kind = winreg.QueryValueEx(key, name)
                except OSError:
                    continue
                if kind == winreg.REG_DWORD:
                    out[name] = int(value)
    except OSError:
        pass
    return out


def _file_version(path: str) -> tuple[int, int, int, int] | None:
    """``FileVersion`` of a PE file (version.dll); ``None`` when the file is missing or has no version resource."""
    import ctypes
    from ctypes import wintypes

    try:
        ver = ctypes.WinDLL("version", use_last_error=True)  # type: ignore[attr-defined]
        ver.GetFileVersionInfoSizeW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
        ver.GetFileVersionInfoSizeW.restype = wintypes.DWORD
        ver.GetFileVersionInfoW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p]
        ver.GetFileVersionInfoW.restype = wintypes.BOOL
        ver.VerQueryValueW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.UINT)]
        ver.VerQueryValueW.restype = wintypes.BOOL
        size = ver.GetFileVersionInfoSizeW(path, None)
        if not size:
            return None
        buf = ctypes.create_string_buffer(size)
        if not ver.GetFileVersionInfoW(path, 0, size, buf):
            return None
        ptr, length = ctypes.c_void_p(), wintypes.UINT()
        if not ver.VerQueryValueW(buf, "\\", ctypes.byref(ptr), ctypes.byref(length)) or not ptr.value or length.value < 16:
            return None
        info = (wintypes.DWORD * 4).from_address(ptr.value)  # VS_FIXEDFILEINFO: signature, struct version, FileVersionMS, FileVersionLS
        ms, ls = info[2], info[3]
        return (ms >> 16, ms & 0xFFFF, ls >> 16, ls & 0xFFFF)
    except (OSError, AttributeError, ValueError):
        return None


def probe_vcredist() -> VcRuntime:
    """Registry first (HKLM, 64-bit view), then the DLLs in System32; not applicable off Windows."""
    if not sys.platform.startswith("win"):
        return VcRuntime(True, None, "none", applicable=False)
    sysdir = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    return decide_vcruntime(_read_vc_registry().get, lambda name: _file_version(str(sysdir / name)))


def probe_vcredist_bundled() -> bool:
    """Is the redistributable the installer carries (``redist\\vc_redist.x64.exe`` next to ``VNDub.exe``) there to run?"""
    from vndub_app import installer

    return installer.bundled_vcredist() is not None


def probe_git() -> dict[str, Any]:
    """``{path, source: system|bundled|None, frozen, envs}``: PATH first, then the MinGit next to ``VNDub.exe``."""
    from vndub_app import installer

    found = installer.find_git()
    bundled = installer.bundled_git()
    source = None if found is None else ("bundled" if bundled is not None and found == bundled else "system")
    return {"path": str(found) if found else None, "source": source, "frozen": bool(getattr(sys, "frozen", False)), "envs": installer.envs_using_git()}


@dataclass
class Probes:
    """Everything Doctor looks at; replace any field in tests."""

    gpu: Callable[[], GpuInfo | None] = probe_gpu
    disk_free_gb: Callable[[Path], float] = probe_disk_free_gb
    internet: Callable[[], bool] = probe_internet
    tools: Callable[[], dict[str, dict[str, Any]]] = probe_tools
    vcredist: Callable[[], VcRuntime] = probe_vcredist
    vcredist_bundled: Callable[[], bool] = probe_vcredist_bundled
    git: Callable[[], dict[str, Any]] = probe_git
    envs: Callable[[], dict[str, Any]] = probe_envs
    models: Callable[[], dict[str, Any]] = probe_models
    data_dir: Callable[[], Path] = appdirs.data_dir
    profile: Callable[[], str] = probe_profile
    processes: Callable[[], list[Any]] = probe_processes
    platform: str = sys.platform
    cache_internet: bool = False  # only the default probes cache the network check


_PROBES: Probes | None = None
_NET_CACHE: tuple[float, bool] | None = None
_NET_LOCK = threading.Lock()


def set_probes(p: Probes | None) -> None:
    """Tests: replace the probes used by the HTTP endpoint (``None`` restores the real ones)."""
    global _PROBES
    _PROBES = p


def current_probes() -> Probes:
    return _PROBES or Probes(cache_internet=True)


def _cached_internet(probes: Probes, refresh: bool) -> bool:
    global _NET_CACHE
    if not probes.cache_internet:
        return probes.internet()
    with _NET_LOCK:
        if not refresh and _NET_CACHE and time.monotonic() - _NET_CACHE[0] < INTERNET_CACHE_S:
            return _NET_CACHE[1]
    ok = probes.internet()
    with _NET_LOCK:
        _NET_CACHE = (time.monotonic(), ok)
    return ok


# ----------------------------------------------------------------------------
# rows
# ----------------------------------------------------------------------------


def _fix(action: str, label_key: str | None = None) -> dict[str, str]:
    return {"action": action, "label_key": label_key or (f"doctor.fix.{action}" if action in FIX_ACTIONS else f"actions.{action}")}


def _row(rid: str, status: str, value: Any = None, params: dict[str, Any] | None = None, fix: dict[str, str] | None = None) -> dict[str, Any]:
    return {"id": rid, "status": status, "value": value, "params": params or {}, "fix": fix}


def _ver(s: str | None) -> tuple[int, int] | None:
    if not s:
        return None
    m = re.match(r"^\s*(\d+)(?:\.(\d+))?", s)
    return (int(m.group(1)), int(m.group(2) or 0)) if m else None


def row_gpu(g: GpuInfo | None) -> dict[str, Any]:
    if g is None:
        return _row("gpu", "red", None, {"reason": "no_nvidia"})
    return _row("gpu", "green", g.name, {"name": g.name, "count": g.count})


def row_driver(g: GpuInfo | None) -> dict[str, Any]:
    if g is None:
        return _row("driver", "red", None, {"reason": "no_driver"}, _fix("open_driver_page"))
    cuda = _ver(g.cuda)
    params = {"driver": g.driver, "cuda": g.cuda or "", "min_cuda": f"{CUDA_GREEN[0]}.{CUDA_GREEN[1]}"}
    value = f"{g.driver} (CUDA {g.cuda})" if g.cuda else g.driver  # a plain value: the UI interpolates primitives only
    if cuda is None:
        return _row("driver", "yellow", value, {**params, "reason": "cuda_unknown"}, _fix("open_driver_page"))
    if cuda >= CUDA_GREEN:
        return _row("driver", "green", value, params)
    if cuda >= CUDA_MIN:
        return _row("driver", "yellow", value, {**params, "reason": "old_driver"}, _fix("open_driver_page"))
    return _row("driver", "red", value, {**params, "reason": "too_old"}, _fix("open_driver_page"))


def row_vram(g: GpuInfo | None) -> dict[str, Any]:
    need = {"need_gb": VRAM_NEEDED_GB}
    if g is None:
        return _row("vram", "red", None, {**need, "vram_gb": 0, "reason": "no_nvidia"})
    if g.vram_gb is None:
        return _row("vram", "yellow", None, {**need, "vram_gb": 0, "reason": "unknown"})
    shown = round(g.vram_gb, 1)
    p = {**need, "vram_gb": shown}
    if g.vram_gb >= VRAM_GREEN_GB:
        return _row("vram", "green", shown, p)
    if g.vram_gb >= VRAM_YELLOW_GB:
        return _row("vram", "yellow", shown, {**p, "reason": "tight"})
    return _row("vram", "red", shown, {**p, "reason": "low_vram"})


def disk_need_gb(models: dict[str, Any], envs: dict[str, Any]) -> float:
    present = set(envs.get("present", []))
    missing = [e for e in envs.get("needed", []) if e not in present]
    envs_gb = sum(ENV_ESTIMATE_BY_NAME_GB.get(e, ENV_ESTIMATE_GB) for e in missing)
    return round(float(models.get("default_missing_gb", 0.0)) + envs_gb + WORK_ROOM_GB, 1)


def row_disk(free_gb: float, need_gb: float, drive: str) -> dict[str, Any]:
    p = {"free_gb": round(free_gb, 1), "need_gb": round(need_gb, 1), "drive": drive}
    if free_gb >= need_gb:
        return _row("disk", "green", round(free_gb, 1), p)
    if free_gb >= max(DISK_FLOOR_GB, need_gb * DISK_YELLOW_SHARE):
        return _row("disk", "yellow", round(free_gb, 1), p, _fix("open_data_dir"))
    return _row("disk", "red", round(free_gb, 1), p, _fix("open_data_dir"))


def row_ffmpeg(tools: dict[str, dict[str, Any]]) -> dict[str, Any]:
    need = [t for t, i in tools.items() if i.get("required")] or ["ffmpeg", "ffprobe"]
    missing = [t for t in need if not tools.get(t, {}).get("present")]
    if not missing:
        return _row("ffmpeg", "green", tools.get("ffmpeg", {}).get("path"), {"path": tools.get("ffmpeg", {}).get("path") or ""})
    return _row("ffmpeg", "red", None, {"missing": ", ".join(missing)}, _fix("download_ffmpeg"))


def row_vcredist(platform: str, rt: VcRuntime | None, bundled: bool) -> dict[str, Any]:
    """Red without the Visual C++ runtime (no model starts); the fix runs the bundled redistributable, or opens Microsoft's page."""
    if not platform.startswith("win") or (rt is not None and not rt.applicable):
        return _row("vcredist", "green", None, {"applicable": False, "reason": "not_applicable"})
    min_version = f"{VC_MIN[0]}.{VC_MIN[1]}"
    if rt is None:  # the probe itself failed
        return _row("vcredist", "yellow", None, {"min_version": min_version})
    params = {"version": rt.version or "", "min_version": min_version, "source": rt.source, "bundled": bundled}
    if rt.ok:
        return _row("vcredist", "green", rt.version, params)
    fix = _fix("install_vcredist") if bundled else _fix("open_vcredist_page")
    return _row("vcredist", "red", rt.version, {**params, "reason": "too_old" if rt.version else "missing"}, fix)


def row_git(g: dict[str, Any]) -> dict[str, Any]:
    """Green when git is found (PATH or bundled); yellow, naming the envs that clone repositories, when it is not."""
    envs = ", ".join(g.get("envs") or [])
    if g.get("path"):
        params: dict[str, Any] = {"path": g["path"], "source": g.get("source") or "system", "envs": envs}
        return _row("git", "green", g["path"], {**params, "reason": "bundled"} if g.get("source") == "bundled" else params)
    return _row("git", "yellow", None, {"envs": envs, **({"reason": "bundled_missing"} if g.get("frozen") else {})})


def row_internet(ok: bool) -> dict[str, Any]:
    return _row("internet", "green" if ok else "yellow", ok)


def row_defender(platform: str, data_dir: Path) -> dict[str, Any]:
    if not platform.startswith("win"):
        return _row("defender", "green", None, {"applicable": False})
    return _row("defender", "yellow", str(data_dir), {"path": str(data_dir)}, _fix("add_defender_exclusion_help"))


def row_envs(envs: dict[str, Any], mock: bool) -> dict[str, Any]:
    needed, present = envs.get("needed", []), envs.get("present", [])
    p = {"present": len(present), "total": len(needed), "missing": ", ".join(e for e in needed if e not in present)}
    if mock:
        return _row("envs", "green", {"present": len(present), "total": len(needed)}, {**p, "mock": True})
    val = {"present": len(present), "total": len(needed), "missing": [e for e in needed if e not in present]}
    if len(present) >= len(needed):
        return _row("envs", "green", val, p)
    return _row("envs", "yellow" if present else "red", val, p, _fix("open_tools"))


def row_models(models: dict[str, Any], mock: bool) -> dict[str, Any]:
    d_tot, d_pres = models.get("default_total", 0), models.get("default_present", 0)
    m_tot, m_pres = models.get("min_total", 0), models.get("min_present", 0)
    p = {"present": d_pres, "total": d_tot, "min_present": m_pres, "min_total": m_tot, "missing_gb": round(float(models.get("default_missing_gb", 0.0)), 1)}
    if mock:
        return _row("models", "green", {"present": d_pres, "total": d_tot}, {**p, "mock": True})
    val = {"present": d_pres, "total": d_tot, "min_present": m_pres, "min_total": m_tot}
    if d_pres >= d_tot:
        return _row("models", "green", val, p)
    if m_pres >= m_tot:
        return _row("models", "yellow", val, p, _fix("open_models"))
    return _row("models", "red", val, p, _fix("open_models"))


def row_processes(procs: list[Any]) -> dict[str, Any]:
    """Yellow while orphan backend processes run (they keep VRAM and file locks); the fix terminates them."""
    from vndub_app import winproc

    n = len(procs)
    return _row("processes", "green" if n == 0 else "yellow", n, {"count": n, "names": winproc.summarize_names(procs)}, _fix("kill_processes") if n else None)


def summarize(rows: list[dict[str, Any]]) -> str:
    """``fail`` if any row is red, else ``warn`` if any is yellow. The Defender row is a standing hint and does not count."""
    counted = [r for r in rows if r["id"] != "defender"]
    if any(r["status"] == "red" for r in counted):
        return "fail"
    if any(r["status"] == "yellow" for r in counted):
        return "warn"
    return "ok"


def choose_torch(gpu: GpuInfo | None | bool = False) -> str:
    """``cuda`` when an NVIDIA GPU with a driver for CUDA >= 12.1 is present, else ``cpu`` (what ``torch: auto`` means)."""
    g = current_probes().gpu() if gpu is False else gpu
    if g is None:
        return "cpu"
    cuda = _ver(g.cuda)
    return "cuda" if cuda is None or cuda >= CUDA_MIN else "cpu"


def _safe(fn: Callable[[], Any], default: Any, what: str) -> Any:
    try:
        return fn()
    except Exception as e:  # a failing probe becomes an "unknown" row, never an HTTP 500
        log.warning("doctor probe %s failed: %s", what, e)
        return default


def run_doctor(probes: Probes | None = None, *, refresh: bool = False) -> dict[str, Any]:
    probes = probes or current_probes()
    data_dir = _safe(probes.data_dir, appdirs.default_data_dir(), "data_dir")
    profile = _safe(probes.profile, "default", "profile")
    mock = profile in ("test", "apptest")
    with ThreadPoolExecutor(max_workers=9, thread_name_prefix="doctor") as pool:
        f_gpu = pool.submit(_safe, probes.gpu, None, "gpu")
        f_net = pool.submit(_safe, lambda: _cached_internet(probes, refresh), False, "internet")
        f_tools = pool.submit(_safe, probes.tools, {}, "tools")
        f_envs = pool.submit(_safe, probes.envs, {"needed": [], "present": []}, "envs")
        f_models = pool.submit(_safe, probes.models, {}, "models")
        f_disk = pool.submit(_safe, lambda: probes.disk_free_gb(Path(data_dir)), 0.0, "disk")
        f_vc = pool.submit(_safe, probes.vcredist, None, "vcredist")
        f_vcb = pool.submit(_safe, probes.vcredist_bundled, False, "vcredist_bundled")
        f_git = pool.submit(_safe, probes.git, {"path": None, "source": None, "frozen": False, "envs": []}, "git")
        f_procs = pool.submit(_safe, probes.processes, [], "processes")
        gpu, net, tools, envs, models, free = (f.result() for f in (f_gpu, f_net, f_tools, f_envs, f_models, f_disk))
        vc, vc_bundled, git, procs = (f.result() for f in (f_vc, f_vcb, f_git, f_procs))
    drive = os.path.splitdrive(str(data_dir))[0] or str(data_dir)
    rows = [
        row_gpu(gpu),
        row_driver(gpu),
        row_vram(gpu),
        row_disk(float(free), disk_need_gb(models, envs), drive),
        row_ffmpeg(tools),
        row_vcredist(probes.platform, vc, bool(vc_bundled)),
        row_git(git),
        row_internet(bool(net)),
        row_defender(probes.platform, Path(data_dir)),
        row_envs(envs, mock),
        row_models(models, mock),
        row_processes(procs),
    ]
    return {
        "summary": summarize(rows),
        "gpu": gpu.public() if gpu else None,
        "rows": rows,
        "torch": choose_torch(gpu),
        "mock": mock,
        "profile": profile,
        "data_dir": str(data_dir),
    }


# ----------------------------------------------------------------------------
# fixes
# ----------------------------------------------------------------------------


def defender_command(path: Path) -> str:
    """The exact PowerShell line the user may run (as administrator) - the app never runs it."""
    return "Add-MpPreference -ExclusionPath '" + str(path).replace("'", "''") + "'"


def apply_fix(action: str) -> dict[str, Any]:
    if action not in FIX_ACTIONS:
        raise AppError("bad_request", params={"field": "action", "value": action}, details=f"unknown fix {action!r}", status=404)
    data = appdirs.data_dir()
    if action == "open_data_dir":
        openers.open_path(data)
        return {"ok": True, "path": str(data)}
    if action == "open_driver_page":
        return {"ok": True, "url": NVIDIA_DRIVER_URL}
    if action == "open_vcredist_page":
        from vndub_app import installer

        return {"ok": True, "url": installer.VCREDIST_URL}
    if action == "kill_processes":  # synchronous, no job: the orphans only (what the row counts)
        from vndub_app import installer, winproc

        killed = winproc.kill_orphans(installer.process_roots(), reason="terminated from Diagnostics")
        return {"ok": True, "killed": len(killed)}
    if action == "add_defender_exclusion_help":
        return {
            "ok": True,
            "path": str(data),
            "command": defender_command(data),
            "settings_path_key": "doctor.defender.settings_path",
            "admin_required": True,
            "applicable": sys.platform.startswith("win"),
        }
    from vndub_app import installer, jobs

    if action == "install_vcredist":
        exe = installer.bundled_vcredist()
        if exe is None:  # the row offers this fix only while the file is there; it can have been deleted since
            raise AppError("file_not_found", params={"path": r"redist\vc_redist.x64.exe"}, details="the bundled vc_redist.x64.exe was not found", status=404)
        job = jobs.runner().submit("vcredist_install", lambda ctx: installer.install_vcredist(ctx), params={"file": str(exe)}, gpu=False)
        return job.info()
    job = jobs.runner().submit("ffmpeg_download", lambda ctx: installer.install_tools(["ffmpeg"], ctx), params={"names": ["ffmpeg"]}, gpu=False)
    return job.info()


@router.get("/api/doctor")
def api_doctor(refresh: bool = False) -> dict[str, Any]:
    return run_doctor(refresh=refresh)


@router.post("/api/doctor/fix/{action}")
def api_doctor_fix(action: str) -> dict[str, Any]:
    return apply_fix(action)

"""Processes of the windowed app on Windows: no console flashes, children that die with the app, and a ledger of leftovers.

``VNDub.exe`` is a GUI program without a console. A GUI process that starts a console program (ffmpeg,
ffprobe, nvidia-smi, vgmstream, rubberband ...) without ``CREATE_NO_WINDOW`` makes Windows open a console
window for every call, and a dub runs thousands of them. The pipeline's own ``subprocess`` calls live in
modules the app does not own, so :func:`hide_child_consoles` adds the flag to every ``subprocess.Popen``
of this process instead (``subprocess.run`` / ``check_output`` / asyncio subprocesses all go through it).
Calls that ask for a console of their own (``CREATE_NEW_CONSOLE`` / ``DETACHED_PROCESS``) are left alone.

Everything else here is about the processes the app starts and must not leave behind (backend adapters, ``llama-server``,
install scripts, tools). They hold the GPU, so a leftover is not harmless. Four layers, from the most to the least direct:

1. :func:`kill_tree` ends a child and everything it started (``taskkill /T /F``; the job runner calls it the moment a job
   is cancelled, see ``JobContext.watch_process``).
2. :func:`kill_children_on_exit` puts the app into a Windows Job Object that kills what is still alive when ``VNDub.exe``
   ends. It only helps once ``VNDub.exe`` really ends (``shell`` finishes with :func:`end_process`) and only when the job
   could be installed (:func:`job_object_status` says why not: an outer job that forbids nesting, for one).
3. The *ledger* (:func:`list_processes`, :func:`kill_orphans`, :func:`kill_all_under`): every running process whose image
   file lies under the app's folders (the environments, uv's Pythons, the tools) with pid, parent, role and whether it is
   ours or an orphan. It does not depend on the process tree, which breaks when an intermediate parent dies first, nor on
   the Job Object. ``shell`` sweeps with it at start, at exit and before every model job; the installer before it touches
   an environment; Doctor shows it.
4. The *reaper* (:func:`start_reaper`, :func:`reaper_main`): a tiny helper process that outlives the app, waits for it to
   end however it ends (crash, kill, hung exit) and then runs the same sweep. It is the answer to "the Job Object did not
   do its job on this machine".

The ledger is Windows only (off Windows every function returns an empty result). Nothing here raises: a kill that fails
is logged (``vndub_app.winproc``: INFO per terminated process with pid, exe and reason, WARNING for a failure).
"""

from __future__ import annotations

import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, NamedTuple

log = logging.getLogger("vndub_app.winproc")

CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_CONSOLE = 0x00000010
DETACHED_PROCESS = 0x00000008
CREATE_BREAKAWAY_FROM_JOB = 0x01000000

_ORIGINAL: Callable[..., None] | None = None


def hide_child_consoles() -> Callable[[], None]:
    """Install the patch (idempotent; no-op off Windows). Returns a function that removes it."""
    global _ORIGINAL
    if os.name != "nt":
        return lambda: None
    if _ORIGINAL is not None:
        return unhide_child_consoles
    original = subprocess.Popen.__init__

    def init(self: subprocess.Popen, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        flags = int(kwargs.get("creationflags", 0) or 0)
        if not flags & (CREATE_NEW_CONSOLE | DETACHED_PROCESS):
            kwargs["creationflags"] = flags | CREATE_NO_WINDOW
        original(self, *args, **kwargs)

    _ORIGINAL = original
    subprocess.Popen.__init__ = init  # type: ignore[method-assign]
    return unhide_child_consoles


def unhide_child_consoles() -> None:
    global _ORIGINAL
    if _ORIGINAL is not None:
        subprocess.Popen.__init__ = _ORIGINAL  # type: ignore[method-assign]
        _ORIGINAL = None


# ----------------------------------------------------------------------------
# killing a child and everything it started
# ----------------------------------------------------------------------------


def kill_tree(proc: "subprocess.Popen[Any] | int", timeout: float = 30.0) -> bool:
    """End ``proc`` (a Popen or a pid) and every process it started. Never raises; returns True when a kill was sent.

    Windows: ``taskkill /T /F`` walks the parent-child tree (PowerShell -> uv -> python ...). Elsewhere the process
    group is killed when the child leads one, else the process alone. A process that already ended is left alone.
    """
    pid = proc if isinstance(proc, int) else getattr(proc, "pid", None)
    if not pid:
        return False
    if not isinstance(proc, int) and proc.poll() is not None:
        return False
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True, timeout=timeout, creationflags=CREATE_NO_WINDOW,
            )
        else:
            try:
                # only a child that leads its own group is killed as a group: killpg on the parent's group would
                # take the app down with it
                if os.getpgid(pid) == pid:
                    os.killpg(pid, signal.SIGKILL)
                else:
                    os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # already gone
        return True
    except Exception as e:  # a kill that fails must never take the caller down
        log.warning("kill_tree(%s) failed: %s", pid, e)
        return False


# ----------------------------------------------------------------------------
# Win32 bindings (built once, Windows only)
# ----------------------------------------------------------------------------

PROCESS_TERMINATE = 0x0001
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
SYNCHRONIZE = 0x00100000
TH32CS_SNAPPROCESS = 0x00000002
STILL_ACTIVE = 259
WAIT_OBJECT_0 = 0
INFINITE = 0xFFFFFFFF
ERROR_INVALID_PARAMETER = 87
UV_ENV_DIR = "_uv"  # <envs>/_uv holds uv's CPython builds and cache: the real interpreter behind every env's launcher


class Meta(NamedTuple):
    """What one ``OpenProcess`` tells about a process: full image path, creation time (FILETIME ticks), still running."""

    image: str
    created: int
    alive: bool


class _Api:
    """ctypes bindings for the ledger and the window helpers. ``_api()`` builds it on first use (Windows only)."""

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        self.ct, self.wt = ctypes, wintypes
        k = self.k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        u = self.u32 = ctypes.WinDLL("user32", use_last_error=True)
        H, D, B = wintypes.HANDLE, wintypes.DWORD, wintypes.BOOL

        class PROCESSENTRY32W(ctypes.Structure):
            _fields_ = [
                ("dwSize", D), ("cntUsage", D), ("th32ProcessID", D), ("th32DefaultHeapID", ctypes.c_void_p), ("th32ModuleID", D),
                ("cntThreads", D), ("th32ParentProcessID", D), ("pcPriClassBase", wintypes.LONG), ("dwFlags", D), ("szExeFile", wintypes.WCHAR * 260),
            ]

        class FLASHWINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.UINT), ("hwnd", H), ("dwFlags", D), ("uCount", wintypes.UINT), ("dwTimeout", D)]

        self.PROCESSENTRY32W, self.FLASHWINFO = PROCESSENTRY32W, FLASHWINFO
        self.ENUMPROC = ctypes.WINFUNCTYPE(B, H, wintypes.LPARAM)
        k.CreateToolhelp32Snapshot.restype, k.CreateToolhelp32Snapshot.argtypes = H, [D, D]
        k.Process32FirstW.argtypes = k.Process32NextW.argtypes = [H, ctypes.POINTER(PROCESSENTRY32W)]
        k.OpenProcess.restype, k.OpenProcess.argtypes = H, [D, B, D]
        k.CloseHandle.argtypes = [H]
        k.TerminateProcess.argtypes = [H, wintypes.UINT]
        k.WaitForSingleObject.restype, k.WaitForSingleObject.argtypes = D, [H, D]
        k.GetExitCodeProcess.argtypes = [H, ctypes.POINTER(D)]
        k.QueryFullProcessImageNameW.argtypes = [H, D, wintypes.LPWSTR, ctypes.POINTER(D)]
        k.GetProcessTimes.argtypes = [H] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        k.GetCurrentProcess.restype = H
        u.GetForegroundWindow.restype = H
        u.GetAncestor.restype, u.GetAncestor.argtypes = H, [H, wintypes.UINT]
        u.IsIconic.argtypes = [H]
        u.IsWindowVisible.argtypes = [H]
        u.GetWindow.restype, u.GetWindow.argtypes = H, [H, wintypes.UINT]
        u.GetWindowThreadProcessId.argtypes = [H, ctypes.POINTER(D)]
        u.GetWindowTextW.argtypes = [H, wintypes.LPWSTR, ctypes.c_int]
        u.EnumWindows.argtypes = [self.ENUMPROC, wintypes.LPARAM]
        u.FlashWindowEx.argtypes = [ctypes.POINTER(FLASHWINFO)]

    def snapshot(self) -> dict[int, tuple[int, str]]:
        """``{pid: (parent pid, exe file name)}`` of every running process, from one Toolhelp32 snapshot."""
        ct = self.ct
        snap = self.k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
        out: dict[int, tuple[int, str]] = {}
        if not snap or snap == self.wt.HANDLE(-1).value:
            return out
        try:
            entry = self.PROCESSENTRY32W()
            entry.dwSize = ct.sizeof(self.PROCESSENTRY32W)
            ok = self.k32.Process32FirstW(snap, ct.byref(entry))
            while ok:
                out[int(entry.th32ProcessID)] = (int(entry.th32ParentProcessID), str(entry.szExeFile))
                ok = self.k32.Process32NextW(snap, ct.byref(entry))
        finally:
            self.k32.CloseHandle(snap)
        return out

    def meta_of(self, handle: Any) -> Meta | None:
        """Image path, creation time and liveness of an open process handle (needs query rights)."""
        ct, wt = self.ct, self.wt
        buf = ct.create_unicode_buffer(4096)
        size = wt.DWORD(len(buf))
        if not self.k32.QueryFullProcessImageNameW(handle, 0, buf, ct.byref(size)):
            return None
        created, a, b, c = wt.FILETIME(), wt.FILETIME(), wt.FILETIME(), wt.FILETIME()
        ticks = 0
        if self.k32.GetProcessTimes(handle, ct.byref(created), ct.byref(a), ct.byref(b), ct.byref(c)):
            ticks = (int(created.dwHighDateTime) << 32) | int(created.dwLowDateTime)
        code = wt.DWORD()
        alive = not self.k32.GetExitCodeProcess(handle, ct.byref(code)) or code.value == STILL_ACTIVE
        return Meta(buf.value, ticks, alive)

    def meta(self, pid: int) -> Meta | None:
        """:class:`Meta` of ``pid``; ``None`` when it cannot be opened (gone, or a process of another user / a protected one)."""
        h = self.k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return None
        try:
            return self.meta_of(h)
        finally:
            self.k32.CloseHandle(h)


_API: _Api | None = None
_API_LOCK = threading.Lock()


def _api() -> _Api:
    global _API
    with _API_LOCK:
        if _API is None:
            _API = _Api()
        return _API


def process_alive(pid: int) -> bool:
    """True while ``pid`` is a running process (Windows; elsewhere ``os.kill(pid, 0)``). Never raises."""
    try:
        if os.name == "nt":
            m = _api().meta(int(pid))
            return m is not None and m.alive
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def end_process(code: int) -> None:
    """End *this* process now, without interpreter finalization and without DLL detach notifications.

    ``os._exit`` skips Python's finalization but still calls the C runtime's ``ExitProcess``, which runs the detach routine
    of every loaded DLL; the CLR and WebView2 DLLs of a pywebview window can hang there and leave a closed app (and the
    Job Object handle that keeps its children alive) running for good. ``TerminateProcess`` on ourselves has no such step.
    Falls back to ``os._exit``. Does not return.
    """
    if os.name == "nt":
        try:
            k = _api().k32
            k.TerminateProcess(k.GetCurrentProcess(), int(code) & 0xFFFFFFFF)
        except Exception:
            pass
    os._exit(code)


# ----------------------------------------------------------------------------
# killing the descendants of a process
# ----------------------------------------------------------------------------

_REAPER_PID: int | None = None  # the helper started by start_reaper: a child by parentage, but it must outlive the app
#: processes of the window itself: WebView2 is shutting down after the window closed and flushes the page's storage (localStorage
#: is committed lazily) while it does; killing it in the same instant would lose the last writes. It ends with the app (or its Job Object).
SPARED_NAMES = ("msedgewebview2.exe",)


def kill_descendants(pid: int | None = None) -> int:
    """Terminate every descendant of ``pid`` (default: this process) by walking the process tree. Returns the count.

    The tree is read with a Toolhelp32 snapshot (parent pid per process) and the descendants are terminated leaves
    first, so a backend adapter or ``llama-server`` started by the pipeline never outlives the window. A process whose
    recorded parent pid has been reused (it is older than the "parent") is not a descendant. The reaper started by
    :func:`start_reaper` and whatever it started is spared, and so is WebView2 (``SPARED_NAMES``). Windows only; never raises.
    This only reaches processes whose chain of parents is intact; :func:`kill_orphans` finds the rest by image path.
    """
    if os.name != "nt":
        return 0
    try:
        api = _api()
        snap = api.snapshot()
        children: dict[int, list[int]] = {}
        for p, (pp, _name) in snap.items():
            children.setdefault(pp, []).append(p)
        root = os.getpid() if pid is None else int(pid)
        root_meta = api.meta(root)
        spared: set[int] = set()
        todo = ([_REAPER_PID] if _REAPER_PID is not None else []) + [p for p, (_pp, n) in snap.items() if n.lower() in SPARED_NAMES]
        while todo:
            cur = todo.pop()
            if cur not in spared:
                spared.add(cur)
                todo.extend(children.get(cur, []))
        order: list[tuple[int, str]] = []
        seen = {root}
        todo = [(root, root_meta.created if root_meta else 0)]
        while todo:
            cur, cur_created = todo.pop()
            for c in children.get(cur, []):
                if c in seen or c in spared:
                    continue
                seen.add(c)
                m = api.meta(c)
                if m is not None and cur_created and m.created and m.created < cur_created:
                    continue  # older than its "parent": the parent pid was reused, this is somebody else's child
                order.append((c, m.image if m else snap[c][1]))
                todo.append((c, m.created if m else 0))
        killed = 0
        for c, image in reversed(order):  # leaves first, so a parent cannot respawn a child it is watching
            h = api.k32.OpenProcess(PROCESS_TERMINATE, False, c)
            if h:
                if api.k32.TerminateProcess(h, 1):
                    killed += 1
                    log.info("terminated pid %d (%s): descendant of pid %d at the end of the app", c, image, root)
                api.k32.CloseHandle(h)
        return killed
    except Exception as e:
        log.warning("kill_descendants failed: %s", e)
        return 0


# ----------------------------------------------------------------------------
# children die with the app
# ----------------------------------------------------------------------------

_JOB_HANDLE: int | None = None
JOB_OBJECT_ERROR: str | None = None  # why the last kill_children_on_exit() failed (None: it worked or was never tried)
_JOB_STRUCTS: tuple[Any, Any, Any] | None = None
JOB_LIMIT_BREAKAWAY_OK = 0x0800
JOB_LIMIT_SILENT_BREAKAWAY_OK = 0x1000
JOB_LIMIT_KILL_ON_JOB_CLOSE = 0x2000


def job_object_status() -> dict[str, Any]:
    """``{"installed": bool, "error": str | None}``: whether this process is in its kill-on-close Job Object, and if not, why.

    Both are falsy / ``None`` before :func:`kill_children_on_exit` ran and on systems without Job Objects, so a caller
    that wants to log only real failures checks ``error``.
    """
    return {"installed": _JOB_HANDLE is not None, "error": JOB_OBJECT_ERROR}


def _job_structs() -> tuple[Any, Any, Any]:
    """``(BASIC, IO, EXTENDED)``: ctypes layouts of ``JOBOBJECT_EXTENDED_LIMIT_INFORMATION`` and its parts."""
    global _JOB_STRUCTS
    if _JOB_STRUCTS is None:
        import ctypes
        from ctypes import wintypes

        class BASIC(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER), ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD), ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD),
            ]

        class IO(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class EXTENDED(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BASIC), ("IoInfo", IO), ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        _JOB_STRUCTS = (BASIC, IO, EXTENDED)
    return _JOB_STRUCTS


def _outer_job_note(k32: Any) -> str:
    """Why an ``AssignProcessToJobObject`` may have failed: a note when this process already sits in another job (and its limits)."""
    try:
        import ctypes
        from ctypes import wintypes

        k32.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
        k32.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
        inside = wintypes.BOOL(0)
        if not k32.IsProcessInJob(k32.GetCurrentProcess(), None, ctypes.byref(inside)) or not inside.value:
            return ""
        info, got = _job_structs()[2](), wintypes.DWORD()
        if k32.QueryInformationJobObject(None, 9, ctypes.byref(info), ctypes.sizeof(info), ctypes.byref(got)):
            flags = int(info.BasicLimitInformation.LimitFlags)
            names = [n for bit, n in ((JOB_LIMIT_KILL_ON_JOB_CLOSE, "kill-on-close"), (JOB_LIMIT_BREAKAWAY_OK, "breakaway-ok"), (JOB_LIMIT_SILENT_BREAKAWAY_OK, "silent-breakaway-ok")) if flags & bit]
            return f"; this process already runs inside another job (limit flags 0x{flags:x}{': ' + ', '.join(names) if names else ''})"
        return "; this process already runs inside another job"
    except Exception:
        return ""


def kill_children_on_exit() -> bool:
    """Put this process into a Windows Job Object that kills every process in it when the last handle closes.

    The pipeline starts backend adapter processes and ``llama-server`` (they hold the GPU). Closing the window
    stops the job runner, which closes them, but a job that is deep inside a model call may not notice for a while
    and the app exits regardless (or crashes): without this the orphans would keep the VRAM. Children inherit the
    job membership (a child that asks for ``CREATE_BREAKAWAY_FROM_JOB`` is refused: the job does not allow breakaway
    except for the instant :func:`start_reaper` needs), the handle is closed by the OS when the app process ends.
    Returns True when installed; when it cannot be, the failing step and the Windows error go to :data:`JOB_OBJECT_ERROR`
    (see :func:`job_object_status`; for a refused assignment the note says that the process already runs inside another
    job) and the app starts regardless: this is a safety net, never a reason to refuse to run. The ledger sweep and the
    reaper cover the machines where it cannot be installed.
    """
    global _JOB_HANDLE, JOB_OBJECT_ERROR
    if os.name != "nt":
        return False
    if _JOB_HANDLE is not None:
        return True
    import ctypes
    from ctypes import wintypes

    k32: Any = None
    job: Any = None

    def fail(step: str, note: Callable[[], str] | None = None) -> bool:
        """Record ``step`` and the Windows error code of the call that just failed; release a half-built job."""
        global JOB_OBJECT_ERROR
        err = ctypes.get_last_error()  # first thing after the failing call: CloseHandle below would overwrite it
        extra = note() if note is not None else ""
        if job and k32 is not None:
            k32.CloseHandle(job)
        JOB_OBJECT_ERROR = f"{step} failed: {ctypes.FormatError(err).strip()} (error {err}){extra}"
        return False

    try:
        extended = _job_structs()[2]
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        job = k32.CreateJobObjectW(None, None)
        if not job:
            return fail("CreateJobObjectW")
        info = extended()
        info.BasicLimitInformation.LimitFlags = JOB_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):  # JobObjectExtendedLimitInformation
            return fail("SetInformationJobObject")
        if not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
            return fail("AssignProcessToJobObject", lambda: _outer_job_note(k32))  # e.g. the process already sits in a job that forbids nesting
    except Exception as e:  # ctypes problems must not stop the app from starting
        JOB_OBJECT_ERROR = f"{type(e).__name__}: {e}"
        return False
    _JOB_HANDLE = int(job)  # never closed on purpose: the OS closes it when this process ends
    JOB_OBJECT_ERROR = None
    return True


def _allow_breakaway(allow: bool) -> bool:
    """Switch ``JOB_OBJECT_LIMIT_BREAKAWAY_OK`` on our Job Object on or off (False when there is no job or it cannot be set)."""
    if _JOB_HANDLE is None or os.name != "nt":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
        info = _job_structs()[2]()
        info.BasicLimitInformation.LimitFlags = JOB_LIMIT_KILL_ON_JOB_CLOSE | (JOB_LIMIT_BREAKAWAY_OK if allow else 0)
        return bool(k32.SetInformationJobObject(ctypes.c_void_p(_JOB_HANDLE), 9, ctypes.byref(info), ctypes.sizeof(info)))
    except Exception:
        return False


# ----------------------------------------------------------------------------
# the process ledger
# ----------------------------------------------------------------------------


@dataclass
class ProcInfo:
    """One running process whose image file lies under one of the ledger's roots (see :func:`list_processes`)."""

    pid: int
    ppid: int
    exe: str  # full image path
    name: str  # file name of the image
    env: str | None  # folder directly under the envs root (``main``, ``_uv`` ...) when the image lies under <envs>/<env>/
    role: str  # adapter | llama_server | tool | other
    ours: bool  # a descendant of this process
    orphan: bool  # its parent is gone, or no chain of live parents leads to a VNDub
    started: int = 0  # creation time (FILETIME ticks): kill_processes uses it to be sure the pid still means this process

    def public(self) -> dict[str, Any]:
        """The shape of ``GET /api/system/processes``."""
        return {"pid": self.pid, "ppid": self.ppid, "exe": self.exe, "name": self.name, "env": self.env, "role": self.role, "ours": self.ours, "orphan": self.orphan}


_PYTHON_NAME = re.compile(r"^pythonw?\d*(\.\d+)?\.exe$")
_SWEEP_LOCK = threading.RLock()


def _norm(path: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def _root_forms(root: str | os.PathLike[str]) -> list[str]:
    """The ways a folder can be spelled in an image path: as given, and with links / short names resolved."""
    forms = [_norm(root)]
    try:
        real = _norm(os.path.realpath(str(root)))
        if real not in forms:
            forms.append(real)
    except OSError:
        pass
    return [f.rstrip("\\/") for f in forms]


class _Scan:
    """One look at the machine: the snapshot, the processes under the roots, and the open-once metadata of everything we asked."""

    def __init__(self, roots: Iterable[str | os.PathLike[str]]) -> None:
        self.api = _api()
        self.roots = [_root_forms(r) for r in roots]
        self.snap = self.api.snapshot()
        self._meta: dict[int, Meta | None] = {}
        self.me = os.getpid()
        self.me_meta = self.meta(self.me)
        self.self_image = _norm(self.me_meta.image) if self.me_meta else ""

    def meta(self, pid: int) -> Meta | None:
        if pid not in self._meta:
            self._meta[pid] = self.api.meta(pid)
        return self._meta[pid]

    def locate(self, image: str) -> tuple[int, str] | None:
        """``(index of the root, path relative to it)`` when ``image`` lies under one of the roots."""
        img = os.path.normpath(image)
        low = os.path.normcase(img)
        for i, forms in enumerate(self.roots):
            for f in forms:
                if f and low.startswith(f + os.sep):
                    return i, img[len(f) + 1:]
        return None

    def is_vndub(self, image: str) -> bool:
        """Is this a running VNDub: a ``VNDub.exe`` or the same program as this process (a source run)? Never something under the roots:
        an environment's interpreter can be the same Python build as the app's and is not the app."""
        same = os.path.normcase(os.path.basename(image)) == "vndub.exe" or (bool(self.self_image) and _norm(image) == self.self_image)
        return same and self.locate(image) is None

    def judge(self, pid: int, created: int) -> tuple[bool, bool]:
        """``(ours, orphan)``: climb the chain of parents up to this process or to the end of the chain.

        Reaching this process: ours. A chain that ends (the parent is gone, or its pid now belongs to a younger process) is an
        orphan, unless it passed a running VNDub on the way up (another instance owns it: neither ours nor an orphan).
        """
        cur, cur_created, saw_vndub = pid, created, False
        for _ in range(64):
            ppid = self.snap.get(cur, (0, ""))[0]
            if ppid <= 0 or ppid == cur or ppid not in self.snap:
                return False, not saw_vndub
            pm = self.meta(ppid)
            if pm is None:
                return False, False  # a live parent we may not look at (another user, protected): do not guess
            if not pm.alive or (cur_created and pm.created and pm.created > cur_created):
                return False, not saw_vndub
            if ppid == self.me:
                return True, False
            saw_vndub = saw_vndub or self.is_vndub(pm.image)
            cur, cur_created = ppid, pm.created
        return False, False

    def candidates(self) -> list[ProcInfo]:
        out: list[ProcInfo] = []
        for pid, (ppid, name) in self.snap.items():
            if pid in (0, 4, self.me):
                continue
            m = self.meta(pid)
            if m is None or not m.alive:
                continue
            where = self.locate(m.image)
            if where is None:
                continue
            idx, rel = where
            low = name.lower()
            env = rel.split(os.sep)[0] if idx == 0 and os.sep in rel else None
            if low.startswith("llama-server"):
                role = "llama_server"
            elif idx == 0:
                role = "adapter" if _PYTHON_NAME.match(low) else "other"
            else:
                role = "tool"
            ours, orphan = self.judge(pid, m.created)
            out.append(ProcInfo(pid, ppid, os.path.normpath(m.image), name, env, role, ours, orphan, m.created))
        by_pid = {i.pid: i for i in out}
        for i in out:  # the real interpreter behind an env's launcher lives in <envs>/_uv: it belongs to its launcher's env
            if i.env == UV_ENV_DIR and i.ppid in by_pid and by_pid[i.ppid].env not in (None, UV_ENV_DIR):
                i.env = by_pid[i.ppid].env
        out.sort(key=lambda i: i.pid)
        return out

    def descendants_of(self, pids: Iterable[int]) -> list[ProcInfo]:
        """Live descendants of ``pids`` (whatever their image) that are not in ``pids`` themselves."""
        children: dict[int, list[int]] = {}
        for p, (pp, _n) in self.snap.items():
            children.setdefault(pp, []).append(p)
        known = set(pids)
        out: list[ProcInfo] = []
        todo = list(known)
        while todo:
            cur = todo.pop()
            cm = self.meta(cur)
            for c in children.get(cur, []):
                if c in known or c == self.me:
                    continue
                m = self.meta(c)
                if m is None or not m.alive or (cm and cm.created and m.created and m.created < cm.created):
                    continue
                known.add(c)
                todo.append(c)
                out.append(ProcInfo(c, cur, os.path.normpath(m.image), self.snap[c][1], None, "other", False, False, m.created))
        return out


def list_processes(roots: Iterable[str | os.PathLike[str]]) -> list[ProcInfo]:
    """Every running process whose full image path lies under one of ``roots``, sorted by pid. Never raises (``[]`` off Windows).

    ``roots[0]`` is the environments folder (``<envs>/<env>/Scripts/python.exe`` gives ``env`` = ``<env>`` and the role
    ``adapter``; ``<envs>/_uv/python/...`` is the real interpreter behind a launcher and takes the env of that launcher);
    the other roots are tool folders (role ``tool``; ``llama-server*.exe`` is ``llama_server`` anywhere). One snapshot of the
    process table, every candidate opened once.

    ``ours``: a descendant of this process. ``orphan``: its parent is gone, or the chain of its live parents does not lead
    to a running VNDub (this process or another ``VNDub.exe``), so nobody who started it is still around to use it. A
    parent pid that has been reused by a younger process counts as gone (the creation times are compared), which also
    keeps the orphans of an earlier run from posing as children of this one when Windows hands out the same pid again.
    """
    if os.name != "nt":
        return []
    try:
        roots = [r for r in roots if str(r)]
        if not any(os.path.isdir(r) for r in roots):  # nothing can run from folders that do not exist (and no snapshot is needed to know it)
            return []
        return _Scan(roots).candidates()
    except Exception as e:
        log.warning("list_processes failed: %s", e)
        return []


def _depth(info: ProcInfo, by_pid: dict[int, ProcInfo]) -> int:
    d, cur, seen = 0, info, {info.pid}
    while cur.ppid in by_pid and cur.ppid not in seen:
        seen.add(cur.ppid)
        cur = by_pid[cur.ppid]
        d += 1
    return d


def kill_processes(infos: Iterable[ProcInfo], reason: str = "requested") -> list[int]:
    """Terminate the listed processes, leaves first; returns the pids that were terminated. Never raises.

    Only the listed processes are touched: each pid is opened again and must still be the process that was listed (same
    image path and creation time), so a pid that was reused since the listing is left alone. Never this process.
    Every kill is logged at INFO (pid, exe, env, role, ``reason``), every failure at WARNING.
    """
    infos = list(infos)
    if os.name != "nt" or not infos:
        return []
    killed: list[int] = []
    try:
        api = _api()
        me = os.getpid()
        by_pid = {i.pid: i for i in infos}
        deadline = time.monotonic() + 15.0
        for info in sorted(infos, key=lambda i: _depth(i, by_pid), reverse=True):
            if info.pid in (0, 4, me):
                continue
            h = api.k32.OpenProcess(PROCESS_TERMINATE | PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, False, info.pid)
            if not h:
                err = api.ct.get_last_error()
                if err != ERROR_INVALID_PARAMETER:  # that one is "already gone"
                    log.warning("cannot terminate pid %d (%s): %s (error %d)", info.pid, info.exe, api.ct.FormatError(err).strip(), err)
                continue
            try:
                m = api.meta_of(h)
                if m is None or not m.alive:
                    continue
                if _norm(m.image) != _norm(info.exe) or (info.started and m.created and m.created != info.started):
                    log.info("not terminating pid %d: it is no longer %s", info.pid, info.exe)
                    continue
                if api.k32.TerminateProcess(h, 1):
                    killed.append(info.pid)
                    log.info("terminated pid %d (%s) env=%s role=%s: %s", info.pid, info.exe, info.env, info.role, reason)
                    wait_ms = int(max(0.2, min(3.0, deadline - time.monotonic())) * 1000)
                    if api.k32.WaitForSingleObject(h, wait_ms) != WAIT_OBJECT_0:
                        log.warning("pid %d (%s) is still running %d ms after it was terminated", info.pid, info.exe, wait_ms)
                else:
                    err = api.ct.get_last_error()
                    log.warning("could not terminate pid %d (%s): %s (error %d)", info.pid, info.exe, api.ct.FormatError(err).strip(), err)
            finally:
                api.k32.CloseHandle(h)
    except Exception as e:
        log.warning("kill_processes failed: %s", e)
    return killed


def kill_orphans(roots: Iterable[str | os.PathLike[str]], reason: str = "orphan left by an earlier run") -> list[ProcInfo]:
    """Terminate every process under ``roots`` that is an orphan and not ours; returns what was terminated. Never raises."""
    with _SWEEP_LOCK:
        victims = [i for i in list_processes(roots) if i.orphan and not i.ours]
        done = set(kill_processes(victims, reason))
        return [i for i in victims if i.pid in done]


def kill_all_under(roots: Iterable[str | os.PathLike[str]], *, with_descendants: bool = False, reason: str = "requested") -> list[ProcInfo]:
    """Terminate every process under ``roots`` (ours too), except this one; returns what was terminated. Never raises.

    ``with_descendants`` adds the descendants of those processes whatever their image: the installer needs it because the
    real interpreter behind an environment's launcher lives in ``<envs>/_uv`` and still has the environment's DLLs open.
    """
    if os.name != "nt":
        return []
    with _SWEEP_LOCK:
        try:
            roots = [r for r in roots if str(r)]
            if not any(os.path.isdir(r) for r in roots):
                return []
            scan = _Scan(roots)
            victims = scan.candidates()
            if with_descendants and victims:
                victims += scan.descendants_of(i.pid for i in victims)
            done = set(kill_processes(victims, reason))
            return [i for i in victims if i.pid in done]
        except Exception as e:
            log.warning("kill_all_under failed: %s", e)
            return []


def describe(infos: Iterable[ProcInfo], limit: int = 12) -> str:
    """``pid 12 C:\\x\\python.exe, pid 15 ...`` for log lines (capped)."""
    items = [f"pid {i.pid} {i.exe}" for i in infos]
    return ", ".join(items[:limit]) + (f", ... (+{len(items) - limit})" if len(items) > limit else "")


def summarize_names(infos: Iterable[ProcInfo]) -> str:
    """``python.exe x2, llama-server.exe``: the names of the processes with a count, in order of first appearance."""
    counts: dict[str, int] = {}
    for i in infos:
        counts[i.name] = counts.get(i.name, 0) + 1
    return ", ".join(n if c == 1 else f"{n} x{c}" for n, c in counts.items())


# ----------------------------------------------------------------------------
# the reaper: a helper that outlives the app
# ----------------------------------------------------------------------------


def reaper_command(parent_pid: int, roots: Iterable[str | os.PathLike[str]], log_file: str | os.PathLike[str]) -> list[str]:
    """Command line of the reaper: ``VNDub.exe --reaper ...`` when packaged, ``python -m vndub_app --reaper ...`` from source."""
    started = 0
    try:
        m = _api().meta(parent_pid) if os.name == "nt" else None
        started = m.created if m else 0
    except Exception:
        pass
    args = ["--reaper", "--parent", str(parent_pid), "--parent-started", str(started), "--roots", os.pathsep.join(str(r) for r in roots), "--log", str(log_file)]
    base = [sys.executable] if getattr(sys, "frozen", False) else [sys.executable, "-m", "vndub_app"]
    return base + args


def start_reaper(roots: Iterable[str | os.PathLike[str]], log_file: str | os.PathLike[str]) -> dict[str, Any]:
    """Start the reaper for this process; returns ``{"pid", "breakaway", "error"}`` for the log. Windows only. Never raises.

    The reaper must not be a member of our Job Object (it would die with us, exactly when it is needed), so it is created
    with ``CREATE_BREAKAWAY_FROM_JOB`` while the job allows breakaway for that one call (:func:`_allow_breakaway`). If the
    breakaway is refused (an outer job of whoever started VNDub forbids it) it is started without: it then ends with the job
    but the ``error`` field says so. It has no window and no console (``DETACHED_PROCESS``: the windowed ``VNDub.exe`` has none
    anyway, a console ``python.exe`` of a source run would otherwise bring a ``conhost.exe`` along), no inherited handles and
    its output goes nowhere.
    """
    global _REAPER_PID
    result: dict[str, Any] = {"pid": None, "breakaway": False, "error": None}
    if os.name != "nt":
        return result
    try:
        roots = [str(r) for r in roots if str(r)]
        cmd = reaper_command(os.getpid(), roots, log_file)
        kw: dict[str, Any] = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
        proc = None
        opened = _allow_breakaway(True)
        try:
            proc = subprocess.Popen(cmd, creationflags=DETACHED_PROCESS | CREATE_BREAKAWAY_FROM_JOB, **kw)
            result["breakaway"] = True
        except OSError as e:
            result["error"] = f"breakaway from the job refused ({e}); the reaper runs inside the app's job"
        finally:
            if opened:
                _allow_breakaway(False)
        if proc is None:
            proc = subprocess.Popen(cmd, creationflags=DETACHED_PROCESS, **kw)
        result["pid"] = proc.pid
        _REAPER_PID = proc.pid
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    return result


REAPER_LOG_MAX_BYTES = 256_000  # the log gets a line or two per app run; past this size only the newest 200 lines are kept


def _reaper_line(log_file: str, text: str) -> None:
    try:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if path.stat().st_size > REAPER_LOG_MAX_BYTES:
                tail = path.read_text(encoding="utf-8", errors="replace").splitlines()[-200:]
                path.write_text("\n".join(tail) + "\n", encoding="utf-8")
        except OSError:
            pass
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + text + "\n")
    except OSError:
        pass


def reaper_main(argv: list[str]) -> int:
    """Body of ``VNDub.exe --reaper --parent PID [--parent-started TICKS] --roots A;B --log FILE``.

    Waits until the parent process ends, however it ends, then terminates every orphan under the roots (the ledger's
    definition: a process nobody who is still alive started) and appends one timestamped line to the log file. It never
    takes the instance lock and never starts the server. If the parent cannot be opened it exits (after one log line),
    except when the parent is already gone (the app died before the reaper got going): then it sweeps at once.
    Imports only the standard library (about 20 MB of memory for the life of the app; no FastAPI, no GUI).
    """
    import argparse

    p = argparse.ArgumentParser(prog="vndub-reaper", add_help=False)
    p.add_argument("--reaper", action="store_true")
    p.add_argument("--parent", type=int, required=True)
    p.add_argument("--parent-started", type=int, default=0)
    p.add_argument("--roots", default="")
    p.add_argument("--log", default="")
    a, _unknown = p.parse_known_args(argv)
    roots = [r for r in a.roots.split(os.pathsep) if r]
    logf = a.log or os.path.join(os.environ.get("TEMP", "."), "vndub-reaper.log")
    if os.name != "nt":
        return 0
    try:
        api = _api()
        h = api.k32.OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, False, a.parent)
        if not h:
            err = api.ct.get_last_error()
            if err != ERROR_INVALID_PARAMETER:
                _reaper_line(logf, f"cannot open parent {a.parent} ({api.ct.FormatError(err).strip()}, error {err}); exiting")
                return 0
            # no such process: the parent was gone before we got going, sweep at once
        else:
            try:
                m = api.meta_of(h)
                if m is None or not (a.parent_started and m.created and m.created != a.parent_started):  # a changed creation time: the pid is a younger process, our parent is long gone
                    api.k32.WaitForSingleObject(h, INFINITE)
            finally:
                api.k32.CloseHandle(h)
        time.sleep(0.5)  # the Job Object (when there is one) kills its members now: let it finish before looking
        killed = kill_orphans(roots, reason=f"orphan after parent {a.parent} ended")
        _reaper_line(logf, f"parent {a.parent} ended; terminated {len(killed)}" + (": " + describe(killed, 30) if killed else ""))
        time.sleep(1.5)  # a process that was being spawned while the parent died
        again = kill_orphans(roots, reason=f"orphan after parent {a.parent} ended (second pass)")
        if again:
            _reaper_line(logf, f"second pass terminated {len(again)}: {describe(again, 30)}")
    except Exception as e:
        _reaper_line(logf, f"reaper failed: {type(e).__name__}: {e}")
    return 0


# ----------------------------------------------------------------------------
# window attention (taskbar flash)
# ----------------------------------------------------------------------------

FLASHW_ALL = 0x00000003
FLASHW_TIMERNOFG = 0x0000000C
FLASHW_STOP = 0


def flash_window(hwnd: int | None, count: int = 5) -> bool:
    """Flash the taskbar button of ``hwnd`` until the window comes to the foreground (``FLASHW_ALL | FLASHW_TIMERNOFG``).

    With ``FLASHW_TIMERNOFG`` Windows ignores ``count`` and keeps flashing until the user activates the window, which is
    what "a job finished while you were elsewhere" wants. Returns True when the call was made. Never raises.
    """
    if os.name != "nt" or not hwnd:
        return False
    try:
        api = _api()
        info = api.FLASHWINFO(api.ct.sizeof(api.FLASHWINFO), api.ct.c_void_p(int(hwnd)), FLASHW_ALL | FLASHW_TIMERNOFG, int(count), 0)
        api.u32.FlashWindowEx(api.ct.byref(info))
        return True
    except Exception as e:
        log.debug("FlashWindowEx failed: %s", e)
        return False


def is_foreground(hwnd: int | None) -> bool:
    """Is ``hwnd`` (or a child of it) the foreground window and not minimized? ``False`` whenever that cannot be told, so the caller flashes."""
    if os.name != "nt" or not hwnd:
        return False
    try:
        api = _api()
        if api.u32.IsIconic(api.ct.c_void_p(int(hwnd))):
            return False
        fg = api.u32.GetForegroundWindow()
        if not fg:
            return False
        root = api.u32.GetAncestor(fg, 2) or fg  # GA_ROOT: a focused child belongs to its top-level window
        return int(root) == int(hwnd) or int(fg) == int(hwnd)
    except Exception:
        return False


def find_own_window(title: str = "VNDub") -> int | None:
    """Handle of a visible, unowned top-level window of this process whose title is ``title`` (else the first visible one)."""
    if os.name != "nt":
        return None
    try:
        api = _api()
        ct, wt = api.ct, api.wt
        mine = os.getpid()
        found: list[tuple[bool, int]] = []

        def visit(hwnd: Any, _lparam: Any) -> bool:
            pid = wt.DWORD()
            api.u32.GetWindowThreadProcessId(hwnd, ct.byref(pid))
            if pid.value == mine and api.u32.IsWindowVisible(hwnd) and not api.u32.GetWindow(hwnd, 4):  # GW_OWNER
                buf = ct.create_unicode_buffer(256)
                api.u32.GetWindowTextW(hwnd, buf, 256)
                found.append((buf.value == title, int(hwnd)))
            return True

        cb = api.ENUMPROC(visit)
        api.u32.EnumWindows(cb, 0)
        found.sort(key=lambda t: not t[0])
        return found[0][1] if found else None
    except Exception:
        return None

"""Keep console windows from flashing when the windowed app starts command-line tools (Windows only).

``VNDub.exe`` is a GUI program without a console. A GUI process that starts a console program (ffmpeg,
ffprobe, nvidia-smi, vgmstream, rubberband ...) without ``CREATE_NO_WINDOW`` makes Windows open a console
window for every call, and a dub runs thousands of them. The pipeline's own ``subprocess`` calls live in
modules the app does not own, so :func:`hide_child_consoles` adds the flag to every ``subprocess.Popen``
of this process instead (``subprocess.run`` / ``check_output`` / asyncio subprocesses all go through it).
Calls that ask for a console of their own (``CREATE_NEW_CONSOLE`` / ``DETACHED_PROCESS``) are left alone.

Two more things live here because they are about the same processes: :func:`kill_tree` ends a child and
everything it started (``taskkill /T /F``; the job runner calls it the moment a job is cancelled, see
``JobContext.watch_process``), and :func:`kill_children_on_exit` puts the app into a Windows Job Object so that
whatever is still alive dies with ``VNDub.exe`` itself.

That safety net only works once ``VNDub.exe`` really ends, which is why ``shell.main`` finishes with a direct
``os._exit`` (a hung GUI / .NET thread must not keep the process, and with it the Job Object handle, alive).
When the Job Object cannot be installed the reason is kept (:func:`job_object_status`) and ``shell`` writes it to
``app.log`` once logging exists, so a user's log says whether the "children die with the app" fallback was in place.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
from typing import Any, Callable

log = logging.getLogger("vndub_app.winproc")

CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_CONSOLE = 0x00000010
DETACHED_PROCESS = 0x00000008

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


def kill_descendants(pid: int | None = None) -> int:
    """Terminate every descendant of ``pid`` (default: this process) by walking the process tree. Returns the count.

    The fallback for the moment the app ends when :func:`kill_children_on_exit` could not install the Job Object (it is
    logged in app.log): the tree is read with a Toolhelp32 snapshot (parent pid per process) and the descendants are
    terminated leaves first, so a backend adapter or ``llama-server`` started by the pipeline never outlives the window.
    Windows only (the Job Object and this walk are both Windows mechanisms); never raises.
    """
    if os.name != "nt":
        return 0
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESSENTRY32W(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_void_p), ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG), ("dwFlags", wintypes.DWORD),
                ("szExeFile", wintypes.WCHAR * 260),
            ]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        k32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
        k32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        snap = k32.CreateToolhelp32Snapshot(0x00000002, 0)  # TH32CS_SNAPPROCESS
        if not snap or snap == wintypes.HANDLE(-1).value:
            return 0
        children: dict[int, list[int]] = {}
        try:
            entry = PROCESSENTRY32W()
            entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
            ok = k32.Process32FirstW(snap, ctypes.byref(entry))
            while ok:
                children.setdefault(int(entry.th32ParentProcessID), []).append(int(entry.th32ProcessID))
                ok = k32.Process32NextW(snap, ctypes.byref(entry))
        finally:
            k32.CloseHandle(snap)
        root = os.getpid() if pid is None else int(pid)
        todo, seen, order = [root], {root}, []
        while todo:  # the snapshot is a tree: every pid is listed once with one parent
            cur = todo.pop()
            for c in children.get(cur, []):
                if c not in seen:
                    seen.add(c)
                    order.append(c)
                    todo.append(c)
        killed = 0
        for c in reversed(order):  # leaves first, so a parent cannot respawn a child it is watching
            h = k32.OpenProcess(0x0001, False, c)  # PROCESS_TERMINATE
            if h:
                if k32.TerminateProcess(h, 1):
                    killed += 1
                k32.CloseHandle(h)
        return killed
    except Exception as e:
        log.warning("kill_descendants failed: %s", e)
        return 0


# ----------------------------------------------------------------------------
# children die with the app
# ----------------------------------------------------------------------------

_JOB_HANDLE: int | None = None
JOB_OBJECT_ERROR: str | None = None  # why the last kill_children_on_exit() failed (None: it worked or was never tried)


def job_object_status() -> dict[str, Any]:
    """``{"installed": bool, "error": str | None}``: whether this process is in its kill-on-close Job Object, and if not, why.

    Both are falsy / ``None`` before :func:`kill_children_on_exit` ran and on systems without Job Objects, so a caller
    that wants to log only real failures checks ``error``.
    """
    return {"installed": _JOB_HANDLE is not None, "error": JOB_OBJECT_ERROR}


def kill_children_on_exit() -> bool:
    """Put this process into a Windows Job Object that kills every process in it when the last handle closes.

    The pipeline starts backend adapter processes and ``llama-server`` (they hold the GPU). Closing the window
    stops the job runner, which closes them, but a job that is deep inside a model call may not notice for a while
    and the app exits regardless (or crashes): without this the orphans would keep the VRAM. Children inherit the
    job membership, the handle is closed by the OS when the app process ends. Returns True when installed; when it
    cannot be, the failing step and the Windows error go to :data:`JOB_OBJECT_ERROR` (see :func:`job_object_status`)
    and the app starts regardless: this is a safety net, never a reason to refuse to run.
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

    def fail(step: str) -> bool:
        """Record ``step`` and the Windows error code of the call that just failed; release a half-built job."""
        global JOB_OBJECT_ERROR
        err = ctypes.get_last_error()  # first thing after the failing call: CloseHandle below would overwrite it
        if job and k32 is not None:
            k32.CloseHandle(job)
        JOB_OBJECT_ERROR = f"{step} failed: {ctypes.FormatError(err).strip()} (error {err})"
        return False

    try:
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
        info = EXTENDED()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):  # JobObjectExtendedLimitInformation
            return fail("SetInformationJobObject")
        if not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
            return fail("AssignProcessToJobObject")  # e.g. the process already sits in a job that forbids nesting
    except Exception as e:  # ctypes problems must not stop the app from starting
        JOB_OBJECT_ERROR = f"{type(e).__name__}: {e}"
        return False
    _JOB_HANDLE = int(job)  # never closed on purpose: the OS closes it when this process ends
    JOB_OBJECT_ERROR = None
    return True

"""Keep console windows from flashing when the windowed app starts command-line tools (Windows only).

``VNDub.exe`` is a GUI program without a console. A GUI process that starts a console program (ffmpeg,
ffprobe, nvidia-smi, vgmstream, rubberband ...) without ``CREATE_NO_WINDOW`` makes Windows open a console
window for every call, and a dub runs thousands of them. The pipeline's own ``subprocess`` calls live in
modules the app does not own, so :func:`hide_child_consoles` adds the flag to every ``subprocess.Popen``
of this process instead (``subprocess.run`` / ``check_output`` / asyncio subprocesses all go through it).
Calls that ask for a console of their own (``CREATE_NEW_CONSOLE`` / ``DETACHED_PROCESS``) are left alone.
"""

from __future__ import annotations

import os
import subprocess
from typing import Callable

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
# children die with the app
# ----------------------------------------------------------------------------

_JOB_HANDLE: int | None = None


def kill_children_on_exit() -> bool:
    """Put this process into a Windows Job Object that kills every process in it when the last handle closes.

    The pipeline starts backend adapter processes and ``llama-server`` (they hold the GPU). Closing the window
    stops the job runner, which closes them, but a job that is deep inside a model call may not notice for a while
    and the app exits regardless (or crashes): without this the orphans would keep the VRAM. Children inherit the
    job membership, the handle is closed by the OS when the app process ends. Returns True when installed.
    """
    global _JOB_HANDLE
    if os.name != "nt":
        return False
    if _JOB_HANDLE is not None:
        return True
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

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    job = k32.CreateJobObjectW(None, None)
    if not job:
        return False
    info = EXTENDED()
    info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))  # JobObjectExtendedLimitInformation
    if not ok or not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
        k32.CloseHandle(job)
        return False
    _JOB_HANDLE = int(job)  # never closed on purpose: the OS closes it when this process ends
    return True

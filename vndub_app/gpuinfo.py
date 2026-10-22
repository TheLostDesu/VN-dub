"""GPU status through ``nvidia-smi`` (no torch): used by ``GET /api/gpu`` and the progress WebSocket."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from typing import Any

from vndub.gpu import find_nvidia_smi

_LOCK = threading.Lock()
_CACHE: tuple[float, dict[str, Any] | None] = (0.0, None)
_TTL_S = 1.0


def available() -> bool:
    return find_nvidia_smi() is not None


def query(max_age: float = _TTL_S) -> dict[str, Any] | None:
    """``{name, total_gb, used_gb}`` of GPU 0, or ``None`` without NVIDIA / when the query fails."""
    global _CACHE
    with _LOCK:
        t, val = _CACHE
        if time.monotonic() - t < max_age:
            return val
    exe = find_nvidia_smi()
    val = None
    if exe:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform.startswith("win") else 0
        try:
            r = subprocess.run(
                [exe, "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits", "--id=0"],
                capture_output=True, text=True, timeout=5, creationflags=flags, check=False,
            )
            if r.returncode == 0 and r.stdout.strip():
                name, total, used = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")[:3]]
                val = {"name": name, "total_gb": round(float(total) / 1024.0, 2), "used_gb": round(float(used) / 1024.0, 2)}
        except (OSError, ValueError, subprocess.SubprocessError):
            val = None
    with _LOCK:
        _CACHE = (time.monotonic(), val)
    return val

"""Spawn backend adapters in their own virtual environments.

Protocol (one JSON object per line, UTF-8, over the child's stdin/stdout)::

    -> {"jsonrpc": "2.0", "id": 7, "method": "load|run|unload|selftest|peak_vram|shutdown", "params": {...}}
    <- {"jsonrpc": "2.0", "id": 7, "result": {...}}
    <- {"jsonrpc": "2.0", "id": 7, "error": {"code": -32001, "message": "...", "data": {...}}}

Error codes: -32001 weights missing, -32002 backend unavailable (import error,
no CUDA, unsupported platform), -32000 generic adapter error.

The adapter process's stderr is streamed to ``work/logs/adapter_<env>.log``.
A crashed adapter (EOF / broken pipe / non-zero exit) is restarted once and the
request retried; a timeout kills the process and raises :class:`AdapterTimeout`.
"""

from __future__ import annotations

import itertools
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import orjson

from vndub.backends.base import BackendSpec, BackendUnavailable, WeightsMissing, _RoleMixin
from vndub.backends.weights import resolved
from vndub.util import repo_root

ERR_WEIGHTS = -32001
ERR_UNAVAILABLE = -32002


class AdapterError(RuntimeError):
    def __init__(self, name: str, message: str, data: Any = None) -> None:
        self.name = name
        self.data = data
        super().__init__(f"{name}: {message}")


class AdapterTimeout(AdapterError):
    pass


class AdapterCrashed(AdapterError):
    pass


def env_python(env: str, root: Path | None = None) -> Path:
    root = root or repo_root()
    base = root / ".venvs" / env
    if sys.platform.startswith("win"):
        return base / "Scripts" / "python.exe"
    return base / "bin" / "python"


def env_exists(env: str, root: Path | None = None) -> bool:
    """A usable env has its interpreter and ``pyvenv.cfg`` (an interrupted ``uv venv`` leaves only the exe)."""
    py = env_python(env, root)
    return py.exists() and (py.parent.parent / "pyvenv.cfg").exists()


class AdapterProcess:
    """One adapter subprocess (one per loaded backend instance)."""

    def __init__(
        self,
        env: str,
        name: str,
        root: Path | None = None,
        log_path: Path | None = None,
        extra_env: dict[str, str] | None = None,
        python: Path | None = None,
        module: str | None = None,
    ) -> None:
        self.env = env
        self.name = name
        self.root = root or repo_root()
        self.python = python or env_python(env, self.root)
        self.module = module or f"adapters.{env}.main"
        self.log_path = log_path
        self.extra_env = extra_env or {}
        self.proc: subprocess.Popen[bytes] | None = None
        self._ids = itertools.count(1)
        self._responses: "queue.Queue[dict[str, Any] | None]" = queue.Queue()
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []

    # -- lifecycle -------------------------------------------------------------
    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> None:
        if self.alive:
            return
        if not self.python.exists():
            raise BackendUnavailable(
                self.name,
                f"backend env '{self.env}' not created ({self.python} missing)",
                f"run scripts/make_envs.ps1 -Env {self.env} (or scripts/make_envs.sh {self.env})",
            )
        env = os.environ.copy()
        env.update(
            {
                "PYTHONPATH": str(self.root) + os.pathsep + env.get("PYTHONPATH", ""),
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUTF8": "1",
                "PYTHONUNBUFFERED": "1",
            }
        )
        env.update(self.extra_env)
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform.startswith("win") else 0
        self.proc = subprocess.Popen(
            [str(self.python), "-u", "-m", self.module],
            cwd=str(self.root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            creationflags=creationflags,
        )
        self._responses = queue.Queue()
        t_out = threading.Thread(target=self._read_stdout, name=f"{self.name}-stdout", daemon=True)
        t_err = threading.Thread(target=self._read_stderr, name=f"{self.name}-stderr", daemon=True)
        t_out.start()
        t_err.start()
        self._threads = [t_out, t_err]

    def _read_stdout(self) -> None:
        assert self.proc and self.proc.stdout
        for raw in self.proc.stdout:
            raw = raw.strip()
            if not raw:
                continue
            try:
                self._responses.put(orjson.loads(raw))
            except orjson.JSONDecodeError:
                self._log(b"[non-protocol stdout] " + raw)
        self._responses.put(None)  # EOF marker

    def _read_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        for raw in self.proc.stderr:
            self._log(raw.rstrip())

    def _log(self, raw: bytes) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "ab") as f:
            f.write(f"[{self.name}] ".encode() + raw + b"\n")

    def close(self, timeout: float = 15.0) -> None:
        if self.proc is None:
            return
        if self.alive:
            try:
                self._send({"jsonrpc": "2.0", "id": 0, "method": "shutdown", "params": {}})
                self.proc.wait(timeout=timeout)
            except Exception:
                pass
        if self.alive:
            self.proc.kill()
            try:
                self.proc.wait(timeout=5)
            except Exception:
                pass
        for s in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            try:
                if s:
                    s.close()
            except Exception:
                pass
        self.proc = None

    # -- requests --------------------------------------------------------------
    def _send(self, msg: dict[str, Any]) -> None:
        assert self.proc and self.proc.stdin
        self.proc.stdin.write(orjson.dumps(msg) + b"\n")
        self.proc.stdin.flush()

    def request(self, method: str, params: dict[str, Any] | None = None, timeout: float = 600.0) -> Any:
        with self._lock:
            if not self.alive:
                raise AdapterCrashed(self.name, "adapter process is not running")
            rid = next(self._ids)
            try:
                self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
            except (BrokenPipeError, OSError) as e:
                raise AdapterCrashed(self.name, f"broken pipe: {e}") from e
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.close(timeout=1)
                    raise AdapterTimeout(self.name, f"'{method}' timed out after {timeout:.0f}s")
                try:
                    msg = self._responses.get(timeout=min(remaining, 1.0))
                except queue.Empty:
                    if not self.alive and self._responses.empty():
                        code = self.proc.returncode if self.proc else None
                        raise AdapterCrashed(self.name, f"adapter exited (code {code}) during '{method}'")
                    continue
                if msg is None:
                    code = self.proc.poll() if self.proc else None
                    raise AdapterCrashed(self.name, f"adapter closed stdout (exit {code}) during '{method}'")
                if msg.get("id") != rid:
                    continue  # stale response from a timed-out call
                if "error" in msg:
                    err = msg["error"] or {}
                    code = err.get("code")
                    text = err.get("message", "adapter error")
                    if code == ERR_WEIGHTS:
                        raise WeightsMissing(self.name, err.get("data", {}).get("detail", "") if isinstance(err.get("data"), dict) else "")
                    if code == ERR_UNAVAILABLE:
                        raise BackendUnavailable(self.name, text, (err.get("data") or {}).get("hint", "") if isinstance(err.get("data"), dict) else "")
                    raise AdapterError(self.name, text, err.get("data"))
                return msg.get("result")


class RemoteBackend(_RoleMixin):
    """Handle for a backend running in its env (see module docstring)."""

    def __init__(
        self,
        spec: BackendSpec,
        options: dict[str, Any] | None = None,
        *,
        device: str = "cuda:0",
        models_dir: Path | None = None,
        log_dir: Path | None = None,
        root: Path | None = None,
        process_factory: Callable[..., AdapterProcess] | None = None,
    ) -> None:
        assert spec.env, f"{spec.name} is a local backend"
        self.spec = spec
        self.options = {**spec.options, **(options or {})}
        self.device = device
        self.models_dir = models_dir
        self.root = root or repo_root()
        log_path = (log_dir / f"adapter_{spec.env}.log") if log_dir else None
        # run time is offline: weights come only from models/ (pulled once)
        extra = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}
        if models_dir is not None:
            extra["VNDUB_MODELS_DIR"] = str(models_dir)
            extra.setdefault("HF_HOME", str(models_dir / "hf_home"))
            extra.setdefault("MODELSCOPE_CACHE", str(models_dir / "modelscope"))
            extra.setdefault("TORCH_HOME", str(models_dir / "torch_home"))
        factory = process_factory or AdapterProcess
        self.proc = factory(spec.env, spec.name, root=self.root, log_path=log_path, extra_env=extra)
        self.load_info: dict[str, Any] = {}
        self._loaded = False
        self.timeout_load = float(self.options.get("timeout_load_s", 1800))
        self.timeout_item = float(self.options.get("timeout_item_s", 300))

    def _load_params(self) -> dict[str, Any]:
        return {
            "name": self.spec.name,
            "model": self.spec.adapter,
            "options": self.options,
            "device": self.device,
            "models_dir": str(self.models_dir) if self.models_dir else None,
            "weights": resolved(self.models_dir, self.spec),
            "caps": self.spec.caps,
        }

    def load(self) -> dict[str, Any]:
        self.proc.start()
        self.load_info = self.proc.request("load", self._load_params(), timeout=self.timeout_load) or {}
        self._loaded = True
        return self.load_info

    def _with_restart(self, method: str, params: dict[str, Any], timeout: float) -> Any:
        try:
            return self.proc.request(method, params, timeout=timeout)
        except AdapterCrashed:
            # restart once, reload, retry
            self.proc.close(timeout=1)
            self.proc.start()
            if self._loaded:
                self.proc.request("load", self._load_params(), timeout=self.timeout_load)
            return self.proc.request(method, params, timeout=timeout)

    def call(self, op: str, items: list[dict[str, Any]] | None = None, **kwargs: Any) -> Any:
        if not self._loaded:
            self.load()
        n = len(items) if items else 1
        timeout = float(kwargs.pop("timeout_s", 120 + self.timeout_item * n))
        params = {"op": op, "items": items, **kwargs}
        return self._with_restart("run", params, timeout)

    def unload(self) -> None:
        if self.proc.alive:
            try:
                self.proc.request("unload", {}, timeout=120)
            except Exception:
                pass
        self.proc.close()
        self._loaded = False

    def selftest(self) -> dict[str, Any]:
        self.proc.start()
        try:
            return self.proc.request("selftest", self._load_params(), timeout=self.timeout_load + 600) or {}
        finally:
            self.proc.close()

    def peak_vram(self) -> float | None:
        if not self.proc.alive:
            return None
        try:
            res = self.proc.request("peak_vram", {}, timeout=30) or {}
            return res.get("peak_vram_gb")
        except Exception:
            return None

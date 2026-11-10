"""Spawn backend adapters in their own virtual environments.

Protocol (one JSON object per line, UTF-8, over the child's stdin/stdout)::

    -> {"jsonrpc": "2.0", "id": 7, "method": "load|run|unload|selftest|peak_vram|shutdown", "params": {...}}
    <- {"jsonrpc": "2.0", "id": 7, "result": {...}}
    <- {"jsonrpc": "2.0", "id": 7, "error": {"code": -32001, "message": "...", "data": {...}}}

Error codes: -32001 weights missing, -32002 backend unavailable (import error,
no CUDA, unsupported platform), -32000 generic adapter error.

The adapter process's stderr is streamed to ``work/logs/adapter_<env>.log``, one ``HH:MM:SS [name] line`` per line
(the file also gets a ``===`` line when the process starts and when it ends). Starts, exits, kills and restarts are
logged to ``app.log`` through the ``vndub.backends.runner`` logger.
A crashed adapter (EOF / broken pipe / non-zero exit) is restarted once and the
request retried; a timeout kills the process and raises :class:`AdapterTimeout`.
An adapter killed on purpose (:meth:`AdapterProcess.abort`, a job was cancelled or the app is closing) is NOT a crash:
the request that was waiting on it raises :class:`AdapterAborted`, nothing is restarted and nothing is re-sent.
"""

from __future__ import annotations

import itertools
import logging
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

log = logging.getLogger("vndub.backends.runner")

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


class AdapterAborted(AdapterCrashed):
    """The adapter was killed by :meth:`AdapterProcess.abort` (cancel / app exit): not a crash, never restarted."""


def _lenient_reply(raw: bytes) -> dict[str, Any] | None:
    """A reply that orjson rejects only because the adapter's ``json.dumps`` wrote ``NaN`` / ``Infinity`` (a MOS or a cosine
    of a silent take): parsed with the standard library, the non-finite numbers as ``None``. It used to be logged as noise and
    the call waited for its timeout (82 minutes for a chunk of 16), then lost the whole chunk."""
    if not raw.startswith(b'{"jsonrpc"'):
        return None
    import json

    try:
        msg = json.loads(raw.decode("utf-8", "replace"), parse_constant=lambda _c: None)
    except ValueError:
        return None
    return msg if isinstance(msg, dict) else None


def envs_root(root: Path | None = None) -> Path:
    """Directory that holds the backend virtual environments.

    An explicit ``root`` always means ``<root>/.venvs`` (tests, other checkouts). Without it
    ``$VNDUB_ENVS_DIR`` wins (the desktop app keeps its envs in ``<data>/envs``), else
    ``<repo>/.venvs`` as before.
    """
    if root is None:
        env = os.environ.get("VNDUB_ENVS_DIR")
        if env:
            return Path(env)
        root = repo_root()
    return root / ".venvs"


def adapters_root() -> Path:
    """Folder that holds the ``adapters`` package: the adapter processes run there (cwd, ``PYTHONPATH``).

    The repo root from a checkout. In the packaged app (PyInstaller) a folder that holds nothing but
    ``adapters/`` (``<bundle>/backend``): the bundle folder itself is full of the core's CPython DLLs and
    extension modules, which a backend env's interpreter must not pick up ahead of its own.
    """
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)) / "backend"
    return repo_root()


def env_python(env: str, root: Path | None = None) -> Path:
    base = envs_root(root) / env
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
        self.root = root or adapters_root()
        # the interpreter follows envs_root(): an explicit root means <root>/.venvs, else $VNDUB_ENVS_DIR (the
        # desktop app's <data>/envs), else <repo>/.venvs -- the same place env_exists() and the installer use
        self.python = python or env_python(env, root)
        self.module = module or f"adapters.{env}.main"
        self.log_path = log_path
        self.extra_env = extra_env or {}
        self.proc: subprocess.Popen[bytes] | None = None
        self._ids = itertools.count(1)
        self._responses: "queue.Queue[dict[str, Any] | None]" = queue.Queue()
        self._lock = threading.Lock()
        self._log_lock = threading.Lock()  # stdout and stderr readers append to the same file
        self._threads: list[threading.Thread] = []
        self._started_at: float | None = None  # monotonic time of the latest start (for "exited after Ns")
        self.aborted = False  # set by abort(): the process died on purpose, requests fail with AdapterAborted

    # -- lifecycle -------------------------------------------------------------
    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> None:
        if self.alive:
            return
        self.aborted = False  # a fresh process (the caller decided to start one) is not the aborted one
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
        self._started_at = time.monotonic()
        log.info("adapter %s started pid %s: %s -m %s (env %s)", self.name, self.proc.pid, self.python, self.module, self.env)
        self._log(f"=== started pid {self.proc.pid}: {self.python} -m {self.module}".encode("utf-8", "replace"))
        self._responses = queue.Queue()
        t_out = threading.Thread(target=self._read_stdout, name=f"{self.name}-stdout", daemon=True)
        t_err = threading.Thread(target=self._read_stderr, name=f"{self.name}-stderr", daemon=True)
        t_out.start()
        t_err.start()
        self._threads = [t_out, t_err]

    def _read_stdout(self) -> None:
        proc = self.proc
        assert proc and proc.stdout
        started = self._started_at
        try:
            for raw in proc.stdout:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    self._responses.put(orjson.loads(raw))
                except orjson.JSONDecodeError:
                    msg = _lenient_reply(raw)
                    if msg is not None:
                        self._responses.put(msg)
                    else:
                        self._log(b"[non-protocol stdout] " + raw)
        except (OSError, ValueError):  # close() shut the pipe under the reader
            pass
        self._responses.put(None)  # EOF marker
        self._log_exit(proc, started)

    def _log_exit(self, proc: "subprocess.Popen[bytes]", started: float | None) -> None:
        """The stdout pipe closed, so the process is over (or about to be): record its exit code and run time."""
        try:
            code = proc.wait(timeout=10)
        except Exception:
            code = proc.poll()
        ran = f" after {time.monotonic() - started:.1f}s" if started is not None else ""
        if code in (0, None):
            log.info("adapter %s exited code %s%s (pid %s)", self.name, code, ran, proc.pid)
        else:
            log.warning("adapter %s exited code %s%s (pid %s)", self.name, code, ran, proc.pid)
        self._log(f"=== exited code {code}{ran}".encode("utf-8", "replace"))

    def _read_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        for raw in self.proc.stderr:
            self._log(raw.rstrip())

    def _log(self, raw: bytes) -> None:
        """Append ``HH:MM:SS [name] line`` to the adapter log file (stderr of the process and our own ``===`` lines)."""
        if self.log_path is None:
            return
        stamp = f"{time.strftime('%H:%M:%S')} [{self.name}] ".encode()
        try:
            with self._log_lock:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_path, "ab") as f:
                    f.write(stamp + raw + b"\n")
        except OSError:  # a full disk or a locked file must not stop the adapter's output being read
            pass

    def _kill_tree(self) -> None:
        """Kill the adapter AND everything it started (an Applio training, a tool): ``Popen.kill`` alone would
        orphan the grandchild, which then keeps its VRAM while the next model loads next to it."""
        assert self.proc is not None
        log.warning("adapter %s (pid %s) did not stop by itself: killing its process tree", self.name, self.proc.pid)
        if sys.platform.startswith("win"):
            try:
                subprocess.run(
                    ["taskkill", "/T", "/F", "/PID", str(self.proc.pid)], capture_output=True, timeout=30,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False,
                )
            except (OSError, subprocess.SubprocessError):
                pass
        try:
            self.proc.kill()
        except OSError:
            pass

    def abort(self) -> None:
        """Kill the adapter and its children at once (a job was cancelled); the call waiting on it fails fast."""
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        self.aborted = True  # before the kill: the request that sees the process die must know it was on purpose
        log.info("adapter %s (pid %s) abort: killing its process tree (job cancelled)", self.name, proc.pid)
        if sys.platform.startswith("win"):
            try:
                subprocess.run(
                    ["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, timeout=30,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False,
                )
            except (OSError, subprocess.SubprocessError):
                pass
        try:
            proc.kill()
        except OSError:
            pass

    def close(self, timeout: float = 15.0) -> None:
        if self.proc is None:
            self.aborted = False
            return
        if self.alive:
            try:
                self._send({"jsonrpc": "2.0", "id": 0, "method": "shutdown", "params": {}})
                self.proc.wait(timeout=timeout)
            except Exception:
                pass
        if self.alive:
            self._kill_tree()
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
        self.aborted = False

    # -- requests --------------------------------------------------------------
    def _send(self, msg: dict[str, Any]) -> None:
        assert self.proc and self.proc.stdin
        self.proc.stdin.write(orjson.dumps(msg) + b"\n")
        self.proc.stdin.flush()

    def _aborted_error(self, method: str) -> AdapterAborted:
        log.info("adapter %s aborted (pid %s): request %s dropped", self.name, self.proc.pid if self.proc else "?", method)
        return AdapterAborted(self.name, f"adapter was aborted (job cancelled) during '{method}'")

    def request(self, method: str, params: dict[str, Any] | None = None, timeout: float = 600.0) -> Any:
        with self._lock:
            if self.aborted:
                raise self._aborted_error(method)
            if not self.alive:
                raise AdapterCrashed(self.name, "adapter process is not running")
            rid = next(self._ids)
            try:
                self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
            except (BrokenPipeError, OSError) as e:
                if self.aborted:
                    raise self._aborted_error(method) from e
                raise AdapterCrashed(self.name, f"broken pipe: {e}") from e
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    log.warning("adapter %s: '%s' timed out after %.0fs, closing the process", self.name, method, timeout)
                    self.close(timeout=1)
                    raise AdapterTimeout(self.name, f"'{method}' timed out after {timeout:.0f}s")
                try:
                    msg = self._responses.get(timeout=min(remaining, 1.0))
                except queue.Empty:
                    if not self.alive and self._responses.empty():
                        if self.aborted:
                            raise self._aborted_error(method)
                        code = self.proc.returncode if self.proc else None
                        raise AdapterCrashed(self.name, f"adapter exited (code {code}) during '{method}'")
                    continue
                if msg is None:
                    if self.aborted:
                        raise self._aborted_error(method)
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
        self.root = root or adapters_root()
        log_path = (log_dir / f"adapter_{spec.env}.log") if log_dir else None
        # run time is offline: weights come only from models/ (pulled once)
        extra = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}
        if models_dir is not None:
            extra["VNDUB_MODELS_DIR"] = str(models_dir)
            extra.setdefault("HF_HOME", str(models_dir / "hf_home"))
            extra.setdefault("MODELSCOPE_CACHE", str(models_dir / "modelscope"))
            extra.setdefault("TORCH_HOME", str(models_dir / "torch_home"))
        factory = process_factory or AdapterProcess
        self.proc = factory(spec.env, spec.name, root=root, log_path=log_path, extra_env=extra)
        self.load_info: dict[str, Any] = {}
        self._loaded = False
        self._aborted = False  # abort() was called: no new process, no new request until unload()
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

    def _check_not_aborted(self, what: str) -> None:
        if self._aborted:
            raise AdapterAborted(self.spec.name, f"adapter was aborted (job cancelled): '{what}' not started")

    def load(self) -> dict[str, Any]:
        self._check_not_aborted("load")
        t0 = time.monotonic()
        self.proc.start()
        self.load_info = self.proc.request("load", self._load_params(), timeout=self.timeout_load) or {}
        self._loaded = True
        log.info("backend %s loaded in %.1fs (device %s)", self.spec.name, time.monotonic() - t0, self.device)
        return self.load_info

    def _with_restart(self, method: str, params: dict[str, Any], timeout: float) -> Any:
        try:
            return self.proc.request(method, params, timeout=timeout)
        except AdapterAborted:
            raise  # killed on purpose (cancel / app exit): a new process would only become an orphan
        except AdapterCrashed as e:
            # restart once, reload, retry
            log.warning("adapter %s crashed (%s): restarting it once and retrying '%s'", self.spec.name, e, method)
            self.proc.close(timeout=1)
            self._check_not_aborted(method)  # Cancel arrived while the crash was being handled: no new process
            self.proc.start()
            if self._loaded:
                self.proc.request("load", self._load_params(), timeout=self.timeout_load)
            return self.proc.request(method, params, timeout=timeout)

    def call(self, op: str, items: list[dict[str, Any]] | None = None, **kwargs: Any) -> Any:
        self._check_not_aborted(op)
        if not self._loaded:
            self.load()
        n = len(items) if items else 1
        timeout = float(kwargs.pop("timeout_s", 120 + self.timeout_item * n))
        params = {"op": op, "items": items, **kwargs}
        return self._with_restart("run", params, timeout)

    def abort(self) -> None:
        self._aborted = True  # a call() that arrives after the kill must not start a process of its own
        self.proc.abort()

    def unload(self) -> None:
        if self.proc.alive:
            try:
                self.proc.request("unload", {}, timeout=120)
            except Exception:
                pass
        self.proc.close()
        self._loaded = False
        self._aborted = False

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

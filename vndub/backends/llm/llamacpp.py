"""Local LLM through llama.cpp's ``llama-server`` (OpenAI-compatible HTTP API).

The backend runs in-process: it starts the server binary (path from
``tools.yaml: llama_server``) bound to 127.0.0.1, waits for ``/health``, and
talks to ``/v1/chat/completions`` with :mod:`urllib` (no third-party client).
The server process holds the model in VRAM, so this backend is *heavy* and is
acquired through the GPU registry like any other model; ``unload`` kills it.

The model never runs half from system RAM by accident. Recent llama.cpp builds
(the b11469 the app installs among them) treat ``-ngl -1`` as *auto* and have
``--fit on`` by default: when the weights plus the context do not leave a 1 GiB
margin of free video memory, the server quietly keeps some layers on the CPU
and the translation crawls. So the server is started with every layer requested
(``-ngl 999``), ``--fit off`` (when the build knows the flag) and an explicit
slot count, and after the start its log is read back: fewer than all layers on
the GPU (no usable CUDA device, not enough memory) is a
:class:`BackendUnavailable`, which makes the registry's fallback chain move on
to a smaller model instead of a slow one. A non-negative
``translate.llm.n_gpu_layers`` is the user's explicit choice of a partial
offload and is passed through unchanged.

Nothing ever leaves localhost.

Logs: ``<log_dir>/llama_server.log`` is the server's own output; every start is announced by a ``=== YYYY-MM-DD
HH:MM:SS <command line>`` line and every stop by ``=== stopped HH:MM:SS (exit code N)``. The app log gets one line per
start (pid, port, model, layers on the GPU) and per stop, through the ``vndub.backends.llm.llamacpp`` logger.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from vndub.backends.base import BackendSpec, BackendUnavailable, LocalBackend, WeightSpec, WeightsMissing
from vndub.backends.weights import local_dir

logger = logging.getLogger("vndub.backends.llm.llamacpp")

_COMMON = dict(role="llm", env=None, adapter="llamacpp", heavy=True, local_impl="vndub.backends.llm.llamacpp:LlamaCppBackend")

# What the registry's budget check costs an LLM (``base.estimate_vram_gb``, GiB): ``options["vram"]`` =
# weights that land in VRAM (the GGUF minus the token embedding, plus the output copy when the model ties them)
# + ``kv_bytes_per_token`` x ctx x parallel (f16 K+V of the full-attention layers) + ``kv_fixed_gib`` (sliding
# windows, recurrent states) + ``overhead_gib`` (compute buffer ~0.5 + the CUDA context ~0.4, which nvidia-smi
# counts but torch does not). ``vram_estimate_gb`` is that number at the spec's own ctx. From the HF configs,
# 2026-10-22:
#   gemma4_12b  IQ4_XS 6.38 GB -> 5.9 GiB; 8 global layers (1 kv head x 512) = 16 KiB/token, 40 sliding layers
#               (8 kv x 256, window 1024 + a 512 batch) = 0.47 GiB fixed -> 7.5 at 16K. Does NOT fit an 8 GB card
#               next to the desktop: the budget check skips it there (12 GB cards run it).
#   qwen35_9b   Q4_K_M 5.68 GB -> ~4.8 GiB in VRAM; 8 full-attention layers (4 kv x 256) = 32 KiB/token, 24
#               linear-attention layers ~0.05 GiB -> 6.3 at 16K.
#   hy_mt15_7b  Q4_K_M 4.62 GB -> ~4.1 GiB; 32 layers (8 kv x 128) = 128 KiB/token -> 6.0 at 8K, 7.0 at 16K.
SPECS = [
    BackendSpec(
        name="gemma4_12b",
        license="Apache-2.0",
        commercial_ok=True,
        vram_estimate_gb=7.5,
        weights=(WeightSpec("hf_file", repo="unsloth/gemma-4-12b-it-GGUF", allow=("*IQ4_XS*.gguf",), size_gb=6.4),),
        # Gemma 4 thinks before answering unless the template is told not to: the thought block would eat the
        # token budget of a JSON answer (truncated answers, "covered 0/N lines")
        options={
            "ctx": 16384,
            "chat_template_kwargs": {"enable_thinking": False},
            "vram": {"weights_gib": 5.9, "kv_bytes_per_token": 16384, "kv_fixed_gib": 0.47, "overhead_gib": 0.9},
        },
        fallback="qwen35_9b",
        mock="mock_translator",
        description="Gemma 4 12B-it IQ4_XS: default ja->ru translator (more than 8 GB of VRAM at 16K context)",
        homepage="https://huggingface.co/unsloth/gemma-4-12b-it-GGUF",
        **_COMMON,
    ),
    BackendSpec(
        name="qwen35_9b",
        license="Apache-2.0",
        commercial_ok=True,
        vram_estimate_gb=6.3,
        weights=(WeightSpec("hf_file", repo="unsloth/Qwen3.5-9B-GGUF", allow=("*Q4_K_M*.gguf",), size_gb=5.7),),
        options={
            "ctx": 16384,
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_budget": 0,
            "vram": {"weights_gib": 4.8, "kv_bytes_per_token": 32768, "kv_fixed_gib": 0.05, "overhead_gib": 0.9},
        },
        fallback="hy_mt15_7b",
        mock="mock_translator",
        description="Qwen3.5-9B Q4_K_M (thinking off): the 8 GB choice, lighter fallback",
        homepage="https://huggingface.co/unsloth/Qwen3.5-9B-GGUF",
        **_COMMON,
    ),
    BackendSpec(
        name="hy_mt15_7b",
        license="Tencent HY-MT license (see model card)",
        commercial_ok=True,
        vram_estimate_gb=6.0,
        weights=(WeightSpec("hf_file", repo="tencent/HY-MT1.5-7B-GGUF", allow=("*Q4_K_M*.gguf",), size_gb=4.6),),
        options={"ctx": 8192, "vram": {"weights_gib": 4.1, "kv_bytes_per_token": 131072, "overhead_gib": 0.9}},
        mock="mock_translator",
        description="HY-MT1.5-7B: sentence-level MT specialist with terminology intervention",
        homepage="https://huggingface.co/tencent/HY-MT1.5-7B-GGUF",
        **_COMMON,
    ),
]


def find_gguf(models_dir: Path, spec: BackendSpec) -> Path:
    for w in spec.weights:
        d = local_dir(models_dir, w, spec.name)
        if not d.exists():
            continue
        pats = w.allow or ("*.gguf",)
        hits = sorted(p for p in d.rglob("*.gguf") if any(fnmatch.fnmatch(p.name, pat) for pat in pats))
        # multi-part GGUFs: llama.cpp wants the first shard; a vision projector (mmproj) is never the language model
        hits = [p for p in hits if ("-of-" not in p.name or "-00001-of-" in p.name) and "mmproj" not in p.name.lower()]
        if hits:
            return hits[0]
    raise WeightsMissing(spec.name, f"no GGUF matching {[w.allow for w in spec.weights]} under {models_dir}")


def _free_port(preferred: int) -> int:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]


#: ``-ngl`` value that means "every layer" in every llama.cpp build (clamped to n_layer + 1). ``-1`` is "auto"
#: (fit to free memory) in the builds that have ``--fit`` and "the library default" (0 on CUDA) in older ones.
ALL_LAYERS = "999"

_OFFLOAD_RE = re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers to GPU")
_help_cache: dict[str, str] = {}


def _no_window() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform.startswith("win") else 0


def server_help(binary: Path) -> str:
    """``llama-server --help`` (cached per binary; ``""`` when it cannot run): which flags this build knows."""
    key = str(binary)
    if key not in _help_cache:
        text = ""
        try:
            r = subprocess.run([str(binary), "--help"], capture_output=True, timeout=30, creationflags=_no_window(), check=False)
            text = (r.stdout or b"").decode("utf-8", "replace") + (r.stderr or b"").decode("utf-8", "replace")
        except (OSError, subprocess.SubprocessError):
            text = ""
        _help_cache[key] = text
    return _help_cache[key]


def build_args(binary: Path, gguf: Path, port: int, options: dict[str, Any], help_text: str = "") -> list[str]:
    """The server command line: every layer on the GPU, no automatic fitting, an explicit slot count."""
    try:
        ngl = int(options.get("n_gpu_layers", -1))
    except (TypeError, ValueError):
        ngl = -1
    # N slots share one context window: each keeps `ctx` tokens, requests on different slots decode together
    parallel = max(1, int(options.get("parallel", 1) or 1))
    args = [
        str(binary), "-m", str(gguf),
        "-c", str(int(options.get("ctx", 16384)) * parallel),
        "-ngl", ALL_LAYERS if ngl < 0 else str(ngl),
        # always explicit: the newer builds default to ``-np auto`` and may split the context between slots
        "-np", str(parallel),
        "--host", "127.0.0.1", "--port", str(port),
        "--jinja", "--no-webui",
    ]
    if "--fit" in help_text:
        args += ["--fit", "off"]
    if options.get("reasoning_budget") is not None:
        args += ["--reasoning-budget", str(options["reasoning_budget"])]
    args += [str(a) for a in options.get("extra_args", []) or []]
    return args


def parse_offload(log_text: str) -> tuple[int, int] | None:
    """``(on_gpu, total)`` layers from the last ``offloaded X/Y layers to GPU`` line of the server log, or None."""
    hits = _OFFLOAD_RE.findall(log_text)
    if not hits:
        return None
    done, total = hits[-1]
    return int(done), int(total)


def offload_problem(requested_layers: int, want_gpu: bool, offload: tuple[int, int] | None) -> str | None:
    """Why a started server must not be used, or None.

    Only when the GPU was wanted and every layer was asked for (``requested_layers < 0``): a user-chosen number
    of layers is their decision, and a log without the line (another build, no log dir) proves nothing.
    """
    if not want_gpu or requested_layers >= 0 or offload is None:
        return None
    done, total = offload
    if done >= total:
        return None
    if done == 0:
        return f"no layer is on the GPU (0/{total} offloaded): llama-server found no usable CUDA device"
    return f"only {done}/{total} layers fit into video memory; the rest would run from system RAM (slow)"


def _read_from(path: Path, offset: int) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


class LlamaCppBackend(LocalBackend):
    """``context`` keys: ``tools`` (Tools), ``models_dir`` (Path), ``log_dir`` (Path), ``device`` (the registry's)."""

    def __init__(self, spec: BackendSpec, options: dict[str, Any] | None = None, context: dict[str, Any] | None = None) -> None:
        super().__init__(spec, options, context)
        self.proc: subprocess.Popen[bytes] | None = None
        self.port = 0
        self.gguf: Path | None = None
        self._tls = threading.local()  # per-thread record of the latest answer (slots are driven from several threads)
        self.load_info: dict[str, Any] = {}  # args, gpu_layers ... of the running server
        self._t_spawn: float | None = None  # monotonic time the server process was started

    @property
    def last(self) -> dict[str, Any]:
        """``finish_reason`` / ``reasoning_chars`` / ``usage`` of the latest chat answer on this thread."""
        return getattr(self._tls, "last", {})

    # -- server lifecycle -----------------------------------------------------
    def _binary(self) -> Path:
        tools = self.context.get("tools")
        path = tools.resolve("llama_server") if tools is not None else None
        if path is None:
            raise BackendUnavailable(
                self.spec.name,
                "llama-server binary not found",
                "set tools.llama_server in configs/tools.yaml (llama.cpp release with CUDA for Windows)",
            )
        return path

    def _want_gpu(self) -> bool:
        return str(self.context.get("device") or "cuda:0").startswith("cuda")

    def load(self) -> dict[str, Any]:
        if self.proc is not None and self.proc.poll() is None:
            return {"port": self.port, **self.load_info}
        models_dir = Path(self.context.get("models_dir") or "models")
        self.gguf = Path(self.options["gguf"]) if self.options.get("gguf") else find_gguf(models_dir, self.spec)
        if not self.gguf.exists():
            raise WeightsMissing(self.spec.name, str(self.gguf))
        binary = self._binary()
        self.port = _free_port(int(self.options.get("port", 8089)))
        args = build_args(binary, self.gguf, self.port, self.options, server_help(binary))
        log_dir = self.context.get("log_dir")
        log_path = Path(log_dir) / "llama_server.log" if log_dir else None
        log_start = 0
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "ab") as f:  # the command line first, so the log says what was asked for
                f.write((f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} " + " ".join(args) + "\n").encode("utf-8", "replace"))
                log_start = f.tell()
        log = open(log_path, "ab") if log_path is not None else subprocess.DEVNULL
        try:
            self.proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, creationflags=_no_window())
        finally:
            if log is not subprocess.DEVNULL:
                log.close()  # the child holds its own handle
        self._t_spawn = time.monotonic()
        logger.info(
            "llama-server started pid %s port %s gguf %s ctx %s parallel %s (%s)",
            self.proc.pid, self.port, self.gguf, int(self.options.get("ctx", 16384)), max(1, int(self.options.get("parallel", 1) or 1)),
            "gpu" if self._want_gpu() else "cpu",
        )
        deadline = time.monotonic() + float(self.options.get("timeout_load_s", 900))
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                logger.warning("llama-server pid %s exited with code %s before it was ready", self.proc.pid, self.proc.returncode)
                raise BackendUnavailable(self.spec.name, f"llama-server exited with code {self.proc.returncode}", "see work/logs/llama_server.log")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if r.status == 200:
                        return self._started(args, log_path, log_start)
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            time.sleep(1.0)
        logger.warning("llama-server pid %s did not become healthy in %ss", getattr(self.proc, "pid", "?"), self.options.get("timeout_load_s", 900))
        self.unload()
        raise BackendUnavailable(self.spec.name, "llama-server did not become healthy in time")

    def _started(self, args: list[str], log_path: Path | None, log_start: int) -> dict[str, Any]:
        """The server answers: check from its log that the model really sits on the GPU, then report."""
        offload = parse_offload(_read_from(log_path, log_start)) if log_path is not None else None
        try:
            requested = int(self.options.get("n_gpu_layers", -1))
        except (TypeError, ValueError):
            requested = -1
        problem = offload_problem(requested, self._want_gpu(), offload)
        if problem:
            logger.warning("llama-server pid %s refused: %s", getattr(self.proc, "pid", "?"), problem)
            self.unload()
            raise BackendUnavailable(
                self.spec.name,
                problem,
                "close other programs that use the graphics card or pick a smaller model; "
                "translate.llm.n_gpu_layers: <number> allows a partial offload on purpose",
            )
        self.loaded = True
        took = f" in {time.monotonic() - self._t_spawn:.1f}s" if self._t_spawn is not None else ""
        logger.info(
            "llama-server ready pid %s port %s gpu layers %s%s",
            getattr(self.proc, "pid", "?"), self.port, f"{offload[0]}/{offload[1]}" if offload else "unknown", took,
        )
        self.load_info = {
            "gguf": str(self.gguf),
            "args": args[1:],
            "gpu_layers": f"{offload[0]}/{offload[1]}" if offload else None,
            "device": "cuda" if self._want_gpu() else "cpu",
        }
        return {"port": self.port, **self.load_info}

    def abort(self) -> None:
        """Kill the server at once (a job was cancelled): a request in flight fails instead of running to its end."""
        proc = self.proc
        if proc is not None and proc.poll() is None:
            logger.warning("llama-server pid %s killed (job cancelled)", getattr(proc, "pid", "?"))
            try:
                proc.kill()
            except OSError:
                pass

    def unload(self) -> None:
        if self.proc is not None:
            proc, pid = self.proc, getattr(self.proc, "pid", "?")
            proc.terminate()
            code: int | None
            try:
                code = proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                logger.warning("llama-server pid %s did not stop in 20s: killing it", pid)
                proc.kill()
                try:
                    code = proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    code = None
            self.proc = None
            logger.info("llama-server stopped pid %s exit code %s", pid, code)
            log_dir = self.context.get("log_dir")
            if log_dir:  # the server's log ends with the exit, so a log read later says the process is gone
                try:
                    with open(Path(log_dir) / "llama_server.log", "ab") as f:
                        f.write(f"\n=== stopped {time.strftime('%H:%M:%S')} (exit code {code})\n".encode("utf-8"))
                except OSError:
                    pass
        self.loaded = False
        self.load_info = {}

    # -- ops ------------------------------------------------------------------
    def op_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        json_mode: bool = True,
        seed: int | None = None,
        timeout_s: float = 600.0,
        **_: Any,  # protocol extras (e.g. tgt=) used only by the mock translator
    ) -> str:
        if not self.loaded:
            self.load()
        body: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if seed is not None:
            body["seed"] = seed
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.options.get("chat_template_kwargs"):
            body["chat_template_kwargs"] = self.options["chat_template_kwargs"]
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            data = json.loads(r.read().decode("utf-8"))
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        # the stage reads these after the call: a "length" finish means the JSON answer was cut off
        self._tls.last = {
            "finish_reason": choice.get("finish_reason"),
            "reasoning_chars": len(message.get("reasoning_content") or ""),
            "usage": data.get("usage") or {},
        }
        return message.get("content") or ""

    def selftest(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        self.load()
        gpu_layers = self.load_info.get("gpu_layers")
        try:
            out = self.op_chat(
                [
                    {"role": "system", "content": "You translate Japanese to Russian. Reply with JSON {\"ru\": \"...\"}."},
                    {"role": "user", "content": "こんにちは"},
                ],
                temperature=0.0,
                max_tokens=64,
            )
            ok = "ru" in json.loads(out)
        finally:
            self.unload()
        return {"ok": ok, "output": out, "elapsed_s": round(time.perf_counter() - t0, 2), "peak_vram_gb": None, "gpu_layers": gpu_layers}

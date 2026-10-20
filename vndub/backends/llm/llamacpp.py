"""Local LLM through llama.cpp's ``llama-server`` (OpenAI-compatible HTTP API).

The backend runs in-process: it starts the server binary (path from
``tools.yaml: llama_server``) bound to 127.0.0.1, waits for ``/health``, and
talks to ``/v1/chat/completions`` with :mod:`urllib` (no third-party client).
The server process holds the model in VRAM, so this backend is *heavy* and is
acquired through the GPU registry like any other model; ``unload`` kills it.

Nothing ever leaves localhost.
"""

from __future__ import annotations

import fnmatch
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from vndub.backends.base import BackendSpec, BackendUnavailable, LocalBackend, WeightSpec, WeightsMissing
from vndub.backends.weights import local_dir

_COMMON = dict(role="llm", env=None, adapter="llamacpp", heavy=True, local_impl="vndub.backends.llm.llamacpp:LlamaCppBackend")

SPECS = [
    BackendSpec(
        name="gemma4_12b",
        license="Apache-2.0",
        commercial_ok=True,
        vram_estimate_gb=7.2,
        weights=(WeightSpec("hf_file", repo="unsloth/gemma-4-12b-it-GGUF", allow=("*IQ4_XS*.gguf",), size_gb=6.4),),
        options={"ctx": 16384, "chat_template_kwargs": {}},
        fallback="qwen35_9b",
        mock="mock_translator",
        description="Gemma 4 12B-it IQ4_XS: default ja->ru translator",
        homepage="https://huggingface.co/unsloth/gemma-4-12b-it-GGUF",
        **_COMMON,
    ),
    BackendSpec(
        name="qwen35_9b",
        license="Apache-2.0",
        commercial_ok=True,
        vram_estimate_gb=6.5,
        weights=(WeightSpec("hf_file", repo="unsloth/Qwen3.5-9B-GGUF", allow=("*Q4_K_M*.gguf",), size_gb=5.7),),
        options={"ctx": 16384, "chat_template_kwargs": {"enable_thinking": False}, "reasoning_budget": 0},
        fallback="hy_mt15_7b",
        mock="mock_translator",
        description="Qwen3.5-9B Q4_K_M (thinking off): lighter fallback",
        homepage="https://huggingface.co/unsloth/Qwen3.5-9B-GGUF",
        **_COMMON,
    ),
    BackendSpec(
        name="hy_mt15_7b",
        license="Tencent HY-MT license (see model card)",
        commercial_ok=True,
        vram_estimate_gb=5.5,
        weights=(WeightSpec("hf_file", repo="tencent/HY-MT1.5-7B-GGUF", allow=("*Q4_K_M*.gguf",), size_gb=4.6),),
        options={"ctx": 8192},
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
        # multi-part GGUFs: llama.cpp wants the first shard
        hits = [p for p in hits if "-of-" not in p.name or "-00001-of-" in p.name]
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


class LlamaCppBackend(LocalBackend):
    """``context`` keys: ``tools`` (Tools), ``models_dir`` (Path), ``log_dir`` (Path)."""

    def __init__(self, spec: BackendSpec, options: dict[str, Any] | None = None, context: dict[str, Any] | None = None) -> None:
        super().__init__(spec, options, context)
        self.proc: subprocess.Popen[bytes] | None = None
        self.port = 0
        self.gguf: Path | None = None

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

    def load(self) -> dict[str, Any]:
        if self.proc is not None and self.proc.poll() is None:
            return {"port": self.port}
        models_dir = Path(self.context.get("models_dir") or "models")
        self.gguf = Path(self.options["gguf"]) if self.options.get("gguf") else find_gguf(models_dir, self.spec)
        if not self.gguf.exists():
            raise WeightsMissing(self.spec.name, str(self.gguf))
        binary = self._binary()
        self.port = _free_port(int(self.options.get("port", 8089)))
        args = [
            str(binary), "-m", str(self.gguf),
            "-c", str(self.options.get("ctx", 16384)),
            "-ngl", str(self.options.get("n_gpu_layers", -1)),
            "--host", "127.0.0.1", "--port", str(self.port),
            "--jinja", "--no-webui",
        ]
        if self.options.get("reasoning_budget") is not None:
            args += ["--reasoning-budget", str(self.options["reasoning_budget"])]
        args += [str(a) for a in self.options.get("extra_args", [])]
        log_dir = self.context.get("log_dir")
        log = open(Path(log_dir) / "llama_server.log", "ab") if log_dir else subprocess.DEVNULL
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform.startswith("win") else 0
        self.proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, creationflags=creationflags)
        deadline = time.monotonic() + float(self.options.get("timeout_load_s", 900))
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise BackendUnavailable(self.spec.name, f"llama-server exited with code {self.proc.returncode}", "see work/logs/llama_server.log")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if r.status == 200:
                        self.loaded = True
                        return {"port": self.port, "gguf": str(self.gguf)}
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            time.sleep(1.0)
        self.unload()
        raise BackendUnavailable(self.spec.name, "llama-server did not become healthy in time")

    def unload(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None
        self.loaded = False

    # -- ops ------------------------------------------------------------------
    def op_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        json_mode: bool = True,
        seed: int | None = None,
        timeout_s: float = 600.0,
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
        return data["choices"][0]["message"]["content"]

    def selftest(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        self.load()
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
        return {"ok": ok, "output": out, "elapsed_s": round(time.perf_counter() - t0, 2), "peak_vram_gb": None}

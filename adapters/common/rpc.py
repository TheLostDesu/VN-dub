"""Adapter-side JSON-RPC server (runs inside ``.venvs/<env>``).

Each env's ``adapters/<env>/main.py`` builds a ``{adapter_key: ModelImpl subclass}``
table and calls :func:`serve`. The core process (``vndub.backends.runner``)
then sends ``load`` / ``run`` / ``unload`` / ``selftest`` / ``peak_vram`` /
``pull`` / ``shutdown`` requests, one JSON object per line.

Only the standard library is imported at module level so this file works in
every env; torch is imported lazily for VRAM accounting.

stdout belongs to the protocol: :func:`serve` duplicates the real stdout for
responses and points fd 1 at stderr, so prints from model libraries (even from
C extensions) cannot corrupt the stream.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

ERR_GENERIC = -32000
ERR_WEIGHTS = -32001
ERR_UNAVAILABLE = -32002


class WeightsMissingError(RuntimeError):
    def __init__(self, name: str, detail: str = "") -> None:
        self.name = name
        self.detail = detail
        super().__init__(f"weights missing: run `vndub models pull {name}`" + (f" ({detail})" if detail else ""))


class UnavailableError(RuntimeError):
    def __init__(self, message: str, hint: str = "") -> None:
        self.hint = hint
        super().__init__(message)


# ----------------------------------------------------------------------------
# helpers shared by adapters
# ----------------------------------------------------------------------------


def torch_device(requested: str) -> str:
    """'cuda:0' if requested and available, else 'cpu'."""
    if not requested.startswith("cuda"):
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return requested
    except ImportError:
        pass
    return "cpu"


def reset_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def peak_vram_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    except Exception:
        pass
    return None


def free_cuda() -> None:
    try:
        import gc

        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def weight_dir(params: dict[str, Any], idx: int = 0, require: bool = True) -> Path:
    """Local dir of the idx-th weight; raises WeightsMissingError if not pulled."""
    ws = params.get("weights") or []
    if idx >= len(ws):
        if require:
            raise WeightsMissingError(params.get("name", "?"), "no weight entry")
        return Path(".")
    w = ws[idx]
    d = Path(w.get("local_dir") or "")
    if require and w.get("kind") != "builtin" and not w.get("present", d.joinpath(".vndub_complete").exists()):
        raise WeightsMissingError(params.get("name", "?"), str(d))
    return d


def find_weight(params: dict[str, Any], basename: str, require: bool = True) -> Path:
    """Local dir of the weight whose repo/subdir basename is ``basename``."""
    for i, w in enumerate(params.get("weights") or []):
        name = (w.get("subdir") or (w.get("repo") or "").split("/")[-1])
        if name == basename:
            return weight_dir(params, i, require)
    if require:
        raise WeightsMissingError(params.get("name", "?"), basename)
    return Path(".")


def load_audio(path: str, sr: int | None = 16000, mono: bool = True):
    """(numpy float32 array, sr). Uses soundfile, resamples with scipy/librosa if needed."""
    import numpy as np
    import soundfile as sf

    y, file_sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(axis=1) if mono else y.T
    if sr and file_sr != sr:
        try:
            import librosa

            y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
        except ImportError:
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(int(sr), int(file_sr))
            y = resample_poly(y, sr // g, file_sr // g).astype("float32")
        file_sr = sr
    return np.ascontiguousarray(y, dtype=np.float32), file_sr


def save_audio(path: str | Path, y, sr: int) -> None:
    import numpy as np
    import soundfile as sf

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(y, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
        arr = arr.T
    sf.write(str(path), arr, int(sr), subtype="PCM_16")


def make_test_clip(path: str | Path, seconds: float = 2.0, sr: int = 16000, f0: float = 220.0) -> str:
    """Speech-like 2 s clip (harmonics with syllable envelope + light noise)."""
    import numpy as np

    n = int(seconds * sr)
    t = np.arange(n) / sr
    f = f0 * 2 ** ((0.8 * np.sin(2 * np.pi * 1.3 * t)) / 12)
    ph = 2 * np.pi * np.cumsum(f) / sr
    y = sum((0.5 / k) * np.sin(k * ph) for k in range(1, 10))
    env = np.clip(np.sin(2 * np.pi * 2.5 * t), 0, None) ** 0.7
    y = 0.2 * env * y / 2 + np.random.default_rng(0).normal(0, 0.003, n)
    save_audio(path, y, sr)
    return str(path)


# ----------------------------------------------------------------------------
# model implementation base
# ----------------------------------------------------------------------------


class ModelImpl:
    """One loaded model. Subclasses implement ``load`` and ``op_<name>`` methods.

    ``op_<name>(self, item, **kw) -> dict`` handles one item; batch-native
    implementations may instead define ``batch_<name>(self, items, **kw) -> list``.
    """

    #: op used by the default selftest and its extra kwargs
    selftest_op: str = ""
    selftest_kwargs: dict[str, Any] = {}

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        self.name: str = params.get("name", "?")
        self.options: dict[str, Any] = params.get("options") or {}
        self.device: str = torch_device(params.get("device") or "cpu")
        self.caps: dict[str, Any] = params.get("caps") or {}

    # lifecycle
    def load(self) -> dict[str, Any]:
        return {}

    def unload(self) -> None:
        for k in list(vars(self)):
            if k not in ("params", "name", "options", "device", "caps"):
                setattr(self, k, None)
        free_cuda()

    # dispatch
    def run(self, op: str, items: list[dict[str, Any]] | None, **kw: Any) -> Any:
        batch = getattr(self, f"batch_{op}", None)
        if batch is not None and items is not None:
            return batch(items, **kw)
        fn = getattr(self, f"op_{op}", None)
        if fn is None:
            raise UnavailableError(f"{self.name} does not support op '{op}'")
        if items is None:
            return fn(**kw)
        out = []
        for it in items:
            try:
                res = fn(it, **kw)
                out.append({"id": it.get("id"), **(res or {})})
            except (WeightsMissingError, UnavailableError):
                raise
            except Exception as e:
                sys.stderr.write(f"[{self.name}] item {it.get('id')} failed: {e}\n{traceback.format_exc()}\n")
                out.append({"id": it.get("id"), "error": f"{type(e).__name__}: {e}"})
        return out

    def selftest_items(self, tmp: Path) -> list[dict[str, Any]]:
        clip = make_test_clip(tmp / "selftest_in.wav")
        return [{"id": "selftest", "wav": clip, "src": clip, "out": str(tmp / "selftest_out.wav"), "text": "テスト", "lang": "ja"}]

    def selftest(self) -> dict[str, Any]:
        import tempfile

        t0 = time.perf_counter()
        reset_peak()
        info = self.load()
        load_s = time.perf_counter() - t0
        with tempfile.TemporaryDirectory() as td:
            items = self.selftest_items(Path(td))
            t1 = time.perf_counter()
            res = self.run(self.selftest_op, items, **self.selftest_kwargs) if self.selftest_op else []
            run_s = time.perf_counter() - t1
        peak = peak_vram_gb()
        self.unload()
        errors = [r.get("error") for r in (res or []) if isinstance(r, dict) and r.get("error")]
        return {
            "ok": not errors,
            "op": self.selftest_op,
            "load_s": round(load_s, 2),
            "run_s": round(run_s, 2),
            "peak_vram_gb": peak,
            "device": self.device,
            "info": info,
            "result": _short(res),
            "errors": errors,
        }

    @classmethod
    def pull(cls, params: dict[str, Any]) -> dict[str, Any]:
        """Env-side download/post-processing (``models pull``). Default: nothing to do."""
        return {"ok": True, "detail": "nothing to pull"}


def _short(obj: Any, limit: int = 400) -> Any:
    s = json.dumps(obj, ensure_ascii=False, default=str)
    return obj if len(s) <= limit else s[:limit] + "..."


# ----------------------------------------------------------------------------
# server loop
# ----------------------------------------------------------------------------


def _error(rid: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message, "data": data}}


def serve(table: dict[str, Callable[[dict[str, Any]], ModelImpl]], env: str) -> None:
    proto = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    current: ModelImpl | None = None

    def send(msg: dict[str, Any]) -> None:
        proto.write(json.dumps(msg, ensure_ascii=False, default=str) + "\n")
        proto.flush()

    def impl_for(params: dict[str, Any]) -> ModelImpl:
        key = params.get("model")
        if key not in table:
            raise UnavailableError(f"env '{env}' has no adapter '{key}' (known: {', '.join(sorted(table))})")
        return table[key](params)

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except json.JSONDecodeError as e:
            sys.stderr.write(f"bad request: {e}\n")
            continue
        rid, method, params = req.get("id"), req.get("method"), req.get("params") or {}
        try:
            if method == "shutdown":
                if current is not None:
                    current.unload()
                send({"jsonrpc": "2.0", "id": rid, "result": {"ok": True}})
                break
            if method == "load":
                if current is not None:
                    current.unload()
                current = impl_for(params)
                reset_peak()
                t0 = time.perf_counter()
                info = current.load() or {}
                info.update(load_s=round(time.perf_counter() - t0, 2), device=current.device, peak_vram_gb=peak_vram_gb())
                send({"jsonrpc": "2.0", "id": rid, "result": info})
            elif method == "run":
                if current is None:
                    raise UnavailableError("no model loaded")
                op = params.pop("op")
                items = params.pop("items", None)
                send({"jsonrpc": "2.0", "id": rid, "result": current.run(op, items, **params)})
            elif method == "unload":
                if current is not None:
                    current.unload()
                    current = None
                send({"jsonrpc": "2.0", "id": rid, "result": {"ok": True}})
            elif method == "peak_vram":
                send({"jsonrpc": "2.0", "id": rid, "result": {"peak_vram_gb": peak_vram_gb()}})
            elif method == "selftest":
                impl = impl_for(params)
                send({"jsonrpc": "2.0", "id": rid, "result": impl.selftest()})
            elif method == "pull":
                key = params.get("model")
                if key not in table:
                    raise UnavailableError(f"env '{env}' has no adapter '{key}'")
                cls = table[key]
                send({"jsonrpc": "2.0", "id": rid, "result": cls.pull(params)})  # type: ignore[attr-defined]
            else:
                send(_error(rid, ERR_GENERIC, f"unknown method {method!r}"))
        except WeightsMissingError as e:
            send(_error(rid, ERR_WEIGHTS, str(e), {"detail": e.detail}))
        except UnavailableError as e:
            send(_error(rid, ERR_UNAVAILABLE, str(e), {"hint": e.hint}))
        except ImportError as e:
            send(_error(rid, ERR_UNAVAILABLE, f"import failed in env '{env}': {e}", {"hint": f"recreate the env: scripts/make_envs.ps1 -Env {env}"}))
        except Exception as e:
            tb = traceback.format_exc()
            sys.stderr.write(tb)
            msg = str(e)
            if "out of memory" in msg.lower():
                msg = f"CUDA out of memory: {msg}"
            send(_error(rid, ERR_GENERIC, f"{type(e).__name__}: {msg}", {"traceback": tb[-2000:]}))


def main(table: dict[str, Callable[[dict[str, Any]], ModelImpl]], env: str) -> None:
    """Entry point: JSON-RPC server, or ``--selftest [keys]`` from the command line."""
    ap = argparse.ArgumentParser(prog=f"python -m adapters.{env}.main")
    ap.add_argument("--selftest", nargs="*", metavar="ADAPTER", help="run selftests (all adapters if none given)")
    ap.add_argument("--models-dir", default=os.environ.get("VNDUB_MODELS_DIR", str(Path(__file__).resolve().parents[2] / "models")))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.selftest is None:
        serve(table, env)
        return
    keys = args.selftest or sorted(table)
    results = {}
    for key in keys:
        params = _cli_params(key, env, Path(args.models_dir), args.device)
        try:
            results[key] = table[key](params).selftest()
        except WeightsMissingError as e:
            results[key] = {"ok": False, "error": str(e)}
        except UnavailableError as e:
            results[key] = {"ok": False, "error": str(e), "hint": e.hint}
        except Exception as e:
            results[key] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    sys.exit(0 if all(r.get("ok") for r in results.values()) else 1)


def _cli_params(key: str, env: str, models_dir: Path, device: str) -> dict[str, Any]:
    """Build load params for a CLI selftest using the core registry if importable."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from vndub.backends import registry  # light imports only
        from vndub.backends.weights import resolved

        for spec in registry.all_specs().values():
            if spec.env == env and spec.adapter == key:
                return {"name": spec.name, "model": key, "options": spec.options, "device": device, "models_dir": str(models_dir), "weights": resolved(models_dir, spec), "caps": spec.caps}
    except Exception as e:  # registry needs core deps (numpy etc.) that may be absent in the env
        sys.stderr.write(f"registry unavailable ({e}); selftest without weight info\n")
    return {"name": key, "model": key, "options": {}, "device": device, "models_dir": str(models_dir), "weights": []}

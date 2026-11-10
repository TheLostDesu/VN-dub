"""Adapter-side JSON-RPC server (runs inside ``.venvs/<env>``).

Each env keeps its dispatch table in a light module ``adapters/<env>/table.py``
(``TABLE = {adapter_key: "module:Class"}``); ``adapters/<env>/main.py`` wraps
it with :func:`lazy_table` and calls :func:`main`. Implementation modules are
imported only when a model is loaded, so a missing package surfaces as a clear
"backend unavailable" error for that model instead of breaking the whole env.

The core process (``vndub.backends.runner``) sends ``load`` / ``run`` /
``unload`` / ``selftest`` / ``probe`` / ``peak_vram`` / ``pull`` / ``shutdown`` requests, one
JSON object per line. Responses never carry NaN / Infinity (:func:`json_safe`): those are not JSON
and the core's parser would refuse the whole line.

``--probe [keys]`` (and the ``probe`` request) imports what each implementation's ``load`` imports, with
its shims, but no weights and no device (:meth:`ModelImpl.probe`): a broken or stale env shows in seconds
instead of after minutes of weight loading (rounds 11, 13 and 14 each failed inside a vendor import).

Only the standard library is imported at module level so this file works in
every env; torch is imported lazily for VRAM accounting.

stdout belongs to the protocol: :func:`serve` duplicates the real stdout for
responses and points fd 1 at stderr, so prints from model libraries (even from
C extensions) cannot corrupt the stream.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
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

LANG_NAMES = {
    "ru": "Russian", "ja": "Japanese", "en": "English", "zh": "Chinese", "ko": "Korean",
    "de": "German", "fr": "French", "es": "Spanish", "it": "Italian", "pt": "Portuguese",
}


def lang_name(code: str | None, default: str = "Auto") -> str:
    if not code:
        return default
    return LANG_NAMES.get(code.lower().split("-")[0], code)


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


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    import random

    random.seed(int(seed))
    try:
        import numpy as np

        np.random.seed(int(seed) % (2**32))
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except ImportError:
        pass


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


def is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"


def weight_dir(params: dict[str, Any], idx: int = 0, require: bool = True) -> Path:
    """Local dir of the idx-th weight; raises WeightsMissingError if not pulled."""
    ws = params.get("weights") or []
    if idx >= len(ws):
        if require:
            raise WeightsMissingError(params.get("name", "?"), "no weight entry")
        return Path(".")
    w = ws[idx]
    d = Path(w.get("local_dir") or "")
    present = w.get("present")
    if present is None:
        present = d.joinpath(".vndub_complete").exists()
    if require and w.get("kind") != "builtin" and not present:
        raise WeightsMissingError(params.get("name", "?"), str(d))
    return d


def find_weight(params: dict[str, Any], basename: str, require: bool = True) -> Path:
    """Local dir of the weight whose repo/subdir basename is ``basename``."""
    for i, w in enumerate(params.get("weights") or []):
        name = w.get("subdir") or (w.get("repo") or "").split("/")[-1]
        if name == basename:
            return weight_dir(params, i, require)
    if require:
        raise WeightsMissingError(params.get("name", "?"), basename)
    return Path(".")


def models_dir(params: dict[str, Any]) -> Path:
    md = params.get("models_dir") or os.environ.get("VNDUB_MODELS_DIR")
    if md:
        return Path(md)
    return Path(__file__).resolve().parents[2] / "models"


def env_src_dir(env: str, name: str) -> Path:
    """Source checkout cloned by scripts/make_envs (``<envs dir>/<env>/src/<name>``).

    The envs dir is ``$VNDUB_ENVS_DIR`` when set (desktop app: ``<data>/envs``), else ``<repo>/.venvs``.
    """
    base = os.environ.get("VNDUB_ENVS_DIR")
    root = Path(base) if base else Path(__file__).resolve().parents[2] / ".venvs"
    return root / env / "src" / name


def load_audio(path: str, sr: int | None = 16000, mono: bool = True):
    """(numpy float32 array, sr). soundfile + scipy polyphase resampling (librosa's soxr_hq
    default measured ~150x slower on the build machine)."""
    import numpy as np
    import soundfile as sf

    y, file_sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(axis=1) if mono else y.T
    if sr and file_sr != sr:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(sr), int(file_sr))
        y = resample_poly(y, int(sr) // g, int(file_sr) // g, axis=-1, window=("kaiser", 8.0)).astype("float32")
        file_sr = sr
    return np.ascontiguousarray(y, dtype=np.float32), file_sr


def save_audio(path: str | Path, y, sr: int) -> None:
    import numpy as np
    import soundfile as sf

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(y, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
        arr = arr.T
    arr = np.clip(arr, -1.0, 1.0)
    sf.write(str(path), arr, int(sr), subtype="PCM_16")


def to_numpy(wav: Any):
    """torch tensor / list / ndarray -> 1-D float32 numpy array."""
    import numpy as np

    if hasattr(wav, "detach"):
        wav = wav.detach().float().cpu().numpy()
    arr = np.asarray(wav, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim > 1:
        arr = arr.mean(axis=0) if arr.shape[0] < arr.shape[-1] else arr.mean(axis=-1)
    return arr


def make_test_clip(path: str | Path, seconds: float = 2.0, sr: int = 16000, f0: float = 220.0) -> str:
    """Speech-like clip (harmonics with syllable envelope + light noise)."""
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
# sub-batching with per-item fallback (used by every ``batch_<op>`` method)
# ----------------------------------------------------------------------------


def audio_seconds(item: dict[str, Any], key: str = "wav") -> float:
    """Duration of ``item[key]`` from the file header (0.0 when unreadable); a cheap sort key for batching."""
    try:
        import soundfile as sf

        return float(sf.info(item[key]).duration)
    except Exception:
        return 0.0


def load_audios(items: list[dict[str, Any]], sr: int | None = 16000, key: str = "wav") -> tuple[list[Any], list[BaseException | None]]:
    """Load ``item[key]`` of every item; a clip that cannot be read (or is empty) yields its exception
    in the second list instead of failing the whole batch. Returns ``(arrays, errors)``, aligned with ``items``."""
    ys: list[Any] = []
    errs: list[BaseException | None] = []
    for it in items:
        try:
            y, _ = load_audio(it[key], sr)
            if len(y) == 0:
                raise ValueError(f"empty audio: {it[key]}")
            ys.append(y)
            errs.append(None)
        except Exception as e:
            ys.append(None)
            errs.append(e)
    return ys, errs


def _error_row(item: dict[str, Any], exc: BaseException, label: str) -> dict[str, Any]:
    sys.stderr.write(f"[{label}] item {item.get('id')} failed: {exc}\n")
    return {"id": item.get("id"), "error": f"{type(exc).__name__}: {exc}"}


def run_in_batches(
    items: list[dict[str, Any]],
    batch_size: int,
    batch_fn: Callable[[list[dict[str, Any]]], list[Any]],
    item_fn: Callable[[dict[str, Any]], dict[str, Any] | None],
    *,
    label: str = "?",
    sort_key: Callable[[dict[str, Any]], Any] | None = None,
) -> list[dict[str, Any]]:
    """Run ``items`` through ``batch_fn`` in sub-batches of at most ``batch_size``; errors stay per item.

    * ``batch_fn(sub_items) -> list`` returns one entry per sub-item, in order: a payload dict, or an
      exception instance for an item that failed on its own (unreadable clip, bad text, ...).
    * If ``batch_fn`` raises (including CUDA OOM, after ``free_cuda()``) or returns the wrong number of
      entries, the sub-batch is redone item by item with ``item_fn`` (the plain ``op_<name>`` path), so one
      bad clip never fails its neighbours. ``WeightsMissingError`` / ``UnavailableError`` propagate, as in
      :meth:`ModelImpl.run`. A sub-batch of one item goes straight to ``item_fn``.
    * ``sort_key`` (e.g. clip duration) orders the items before they are cut into sub-batches, which keeps
      padding small; the result list is always in the original item order.

    Returns one dict per item, in item order: ``{"id", **payload}`` or ``{"id", "error"}``.
    """
    n = len(items)
    results: list[dict[str, Any] | None] = [None] * n
    order = list(range(n))
    size = max(1, int(batch_size or 1))
    if sort_key is not None and size > 1 and n > 1:
        try:
            order.sort(key=lambda i: sort_key(items[i]))
        except Exception:
            order = list(range(n))
    for start in range(0, n, size):
        idx = order[start : start + size]
        sub = [items[i] for i in idx]
        outs: list[Any] | None = None
        if len(sub) > 1:
            try:
                outs = list(batch_fn(sub))
                if len(outs) != len(sub):
                    raise RuntimeError(f"batch returned {len(outs)} results for {len(sub)} items")
                outs = [RuntimeError("batch produced no result for this item") if o is None else o for o in outs]
            except (WeightsMissingError, UnavailableError):
                raise
            except Exception as e:
                if is_oom(e):
                    free_cuda()
                sys.stderr.write(f"[{label}] batch of {len(sub)} failed ({type(e).__name__}: {e}); retrying item by item\n{traceback.format_exc()}\n")
                outs = None
        if outs is None:
            outs = []
            for it in sub:
                try:
                    outs.append(item_fn(it) or {})
                except (WeightsMissingError, UnavailableError):
                    raise
                except Exception as e:
                    if is_oom(e):
                        free_cuda()
                    sys.stderr.write(f"[{label}] item {it.get('id')} traceback:\n{traceback.format_exc()}\n")
                    outs.append(e)
        for i, it, o in zip(idx, sub, outs):
            if isinstance(o, BaseException):
                results[i] = _error_row(it, o, label)
            elif isinstance(o, dict) and o.get("error"):
                results[i] = {"id": it.get("id"), **o}
            else:
                results[i] = {"id": it.get("id"), **(o or {})}
    return [r if r is not None else {"id": items[i].get("id"), "error": "RuntimeError: no result"} for i, r in enumerate(results)]


# ----------------------------------------------------------------------------
# model implementation base
# ----------------------------------------------------------------------------


class ModelImpl:
    """One loaded model. Subclasses implement ``load`` and ``op_<name>`` methods.

    ``op_<name>(self, item, **kw) -> dict`` handles one item; batch-native
    implementations may instead define ``batch_<name>(self, items, **kw) -> list``.
    A ``batch_<name>`` should build its result with :func:`run_in_batches` (sub-batches of
    :meth:`sub_batch_size`, per-item fallback to ``op_<name>`` when a batch fails or runs out of memory),
    so that an error never spreads from one item to its neighbours.
    """

    #: op used by the default selftest and its extra kwargs
    selftest_op: str = ""
    selftest_kwargs: dict[str, Any] = {}
    #: vendor modules :meth:`load` imports; :meth:`probe` imports them without weights (keep in step with ``load``)
    probe_imports: tuple[str, ...] = ()

    @classmethod
    def probe_setup(cls, params: dict[str, Any]) -> None:
        """What ``load`` does before its imports (a checkout on sys.path, compat shims). Default: nothing."""

    @classmethod
    def probe_checks(cls, params: dict[str, Any]) -> dict[str, Any] | None:
        """Checks beyond plain imports (a vendor that swallows an ImportError, a config parser). Default: none."""
        return None

    @classmethod
    def probe(cls, params: dict[str, Any]) -> dict[str, Any]:
        """Import what a load imports, no weights, no device: ``{"ok": True, "s", "imports", ...}``; raises on a broken env."""
        t0 = time.perf_counter()
        cls.probe_setup(params)
        for mod in cls.probe_imports:
            importlib.import_module(mod)
        extra = cls.probe_checks(params) or {}
        return {"ok": True, "s": round(time.perf_counter() - t0, 2), "imports": list(cls.probe_imports), **extra}

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        self.name: str = params.get("name", "?")
        self.options: dict[str, Any] = params.get("options") or {}
        self.device: str = torch_device(params.get("device") or "cpu")
        self.caps: dict[str, Any] = params.get("caps") or {}

    # lifecycle
    def load(self) -> dict[str, Any]:
        return {}

    def sub_batch_size(self, default: int = 8) -> int:
        """Items per model call inside a ``batch_<op>`` method (``options['batch_size']``, at least 1)."""
        try:
            return max(1, int(self.options.get("batch_size", default)))
        except (TypeError, ValueError):
            return default

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
                if is_oom(e):
                    free_cuda()
                sys.stderr.write(f"[{self.name}] item {it.get('id')} failed: {e}\n{traceback.format_exc()}\n")
                out.append({"id": it.get("id"), "error": f"{type(e).__name__}: {e}"})
        return out

    def selftest_items(self, tmp: Path) -> list[dict[str, Any]]:
        clip = make_test_clip(tmp / "selftest_in.wav")
        return [{"id": "selftest", "wav": clip, "src": clip, "out": str(tmp / "selftest_out.npy"), "text": "テスト", "lang": "ja"}]

    def selftest(self) -> dict[str, Any]:
        import tempfile

        t0 = time.perf_counter()
        reset_peak()
        info = self.load() or {}
        load_s = time.perf_counter() - t0
        with tempfile.TemporaryDirectory() as td:
            items = self.selftest_items(Path(td))
            t1 = time.perf_counter()
            res = self.run(self.selftest_op, items, **self.selftest_kwargs) if self.selftest_op else []
            run_s = time.perf_counter() - t1
        peak = peak_vram_gb()
        self.unload()
        rows = res if isinstance(res, list) else [res]
        errors = [r.get("error") for r in rows if isinstance(r, dict) and r.get("error")]
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


class TTSImpl(ModelImpl):
    """TTS base: ``generate(item) -> (wave, sr)``; this class handles seeds, timing and IO.

    Item keys: id, text, lang, timbre_ref, timbre_ref_text, emotion_ref, instruct,
    knobs, seed, target_duration, out.
    """

    selftest_op = "synthesize"
    selftest_text = "Привет! Это короткая проверка синтеза речи."

    def generate(self, item: dict[str, Any]):  # pragma: no cover - abstract
        raise NotImplementedError

    def design_generate(self, item: dict[str, Any]):
        """Voice design from ``item['description']`` (or ``instruct``); default: unsupported."""
        raise UnavailableError(f"{self.name} does not support voice design")

    def _finish(self, item: dict[str, Any], wav: Any, sr: int, t0: float) -> dict[str, Any]:
        y = to_numpy(wav)
        if y.size == 0:
            raise RuntimeError("model returned empty audio")
        save_audio(item["out"], y, sr)
        return {"path": item["out"], "sr": int(sr), "duration": round(len(y) / sr, 3), "elapsed_s": round(time.perf_counter() - t0, 3)}

    def op_synthesize(self, item: dict[str, Any], **_: Any) -> dict[str, Any]:
        set_seed(item.get("seed"))
        t0 = time.perf_counter()
        wav, sr = self.generate(item)
        return self._finish(item, wav, sr, t0)

    def op_design(self, item: dict[str, Any], **_: Any) -> dict[str, Any]:
        set_seed(item.get("seed"))
        t0 = time.perf_counter()
        wav, sr = self.design_generate(item)
        return self._finish(item, wav, sr, t0)

    def selftest_items(self, tmp: Path) -> list[dict[str, Any]]:
        ref = make_test_clip(tmp / "ref.wav", seconds=3.0, sr=24000)
        return [
            {
                "id": "selftest", "text": self.selftest_text, "lang": "ru", "timbre_ref": ref, "timbre_ref_text": "",
                "emotion_ref": None, "instruct": None, "knobs": {}, "seed": 1234, "target_duration": 2.5,
                "out": str(tmp / "tts_out.wav"), "description": "A young woman with a clear, bright voice.",
            }
        ]


def _short(obj: Any, limit: int = 400) -> Any:
    s = json.dumps(obj, ensure_ascii=False, default=str)
    return obj if len(s) <= limit else s[:limit] + "..."


# ----------------------------------------------------------------------------
# lazy dispatch tables
# ----------------------------------------------------------------------------


class LazyImpl:
    """Factory for ``module:Class`` imported on first use (keeps the env table light)."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._cls: type[ModelImpl] | None = None

    def cls(self) -> type[ModelImpl]:
        if self._cls is None:
            mod, _, name = self.path.partition(":")
            self._cls = getattr(importlib.import_module(mod), name)
        return self._cls

    def __call__(self, params: dict[str, Any]) -> ModelImpl:
        return self.cls()(params)

    def pull(self, params: dict[str, Any]) -> dict[str, Any]:
        return self.cls().pull(params)

    def probe(self, params: dict[str, Any]) -> dict[str, Any]:
        return self.cls().probe(params)  # importing the implementation module is part of the check


def lazy_table(table: dict[str, str]) -> dict[str, LazyImpl]:
    return {k: LazyImpl(v) for k, v in table.items()}


# ----------------------------------------------------------------------------
# server loop
# ----------------------------------------------------------------------------


def _error(rid: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message, "data": data}}


def json_safe(obj: Any) -> Any:
    """``obj`` with every non-finite float (NaN, +-Infinity, numpy's included) replaced by ``None``.

    ``json.dumps`` writes them as the bare words ``NaN`` / ``Infinity``, which are not JSON: a single silent clip
    (a pitch track of NaN, a score of -inf) made the core refuse the whole response line."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    kind = getattr(getattr(obj, "dtype", None), "kind", None)
    if kind == "f" and getattr(obj, "ndim", None) == 0:  # a numpy float scalar (np.float32 is no float subclass)
        v = float(obj)
        return v if math.isfinite(v) else None
    return obj


def _env_hint(env: str) -> str:
    """The way out of a broken env, for app users first (the hint reaches the app's error panel), then the command line."""
    return f"rebuild the environment '{env}' (in the app: Diagnostics -> Rebuild environments, or Models -> Environments; command line: scripts/make_envs.ps1 -Env {env})"


def serve(table: dict[str, Callable[[dict[str, Any]], ModelImpl]], env: str) -> None:
    proto = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    current: ModelImpl | None = None

    def send(msg: dict[str, Any]) -> None:
        proto.write(json.dumps(json_safe(msg), ensure_ascii=False, default=str) + "\n")
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
                    current = None
                impl = impl_for(params)
                reset_peak()
                t0 = time.perf_counter()
                info = impl.load() or {}
                current = impl
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
            elif method == "probe":
                key = params.get("model")
                if key not in table:
                    raise UnavailableError(f"env '{env}' has no adapter '{key}'")
                probe = getattr(table[key], "probe", None)
                if probe is None:
                    raise UnavailableError(f"adapter '{key}' of env '{env}' has no probe")
                send({"jsonrpc": "2.0", "id": rid, "result": probe(params)})
            elif method == "pull":
                key = params.get("model")
                if key not in table:
                    raise UnavailableError(f"env '{env}' has no adapter '{key}'")
                factory = table[key]
                pull = getattr(factory, "pull", None)
                send({"jsonrpc": "2.0", "id": rid, "result": pull(params) if pull else {"ok": True}})
            else:
                send(_error(rid, ERR_GENERIC, f"unknown method {method!r}"))
        except WeightsMissingError as e:
            send(_error(rid, ERR_WEIGHTS, str(e), {"detail": e.detail}))
        except UnavailableError as e:
            send(_error(rid, ERR_UNAVAILABLE, str(e), {"hint": e.hint}))
        except ImportError as e:
            send(_error(rid, ERR_UNAVAILABLE, f"import failed in env '{env}': {e}", {"hint": _env_hint(env)}))
        except Exception as e:
            tb = traceback.format_exc()
            sys.stderr.write(tb)
            msg = str(e)
            if is_oom(e):
                free_cuda()
                msg = f"CUDA out of memory: {msg}"
            send(_error(rid, ERR_GENERIC, f"{type(e).__name__}: {msg}", {"traceback": tb[-2000:]}))


def main(table: dict[str, Callable[[dict[str, Any]], ModelImpl]], env: str) -> None:
    """Entry point: JSON-RPC server, ``--selftest [keys]`` or ``--probe [keys]`` from the command line."""
    ap = argparse.ArgumentParser(prog=f"python -m adapters.{env}.main")
    ap.add_argument("--selftest", nargs="*", metavar="ADAPTER", help="run selftests (all adapters if none given)")
    ap.add_argument("--probe", nargs="*", metavar="ADAPTER", help="import-level check of each adapter, no weights (all adapters if none given)")
    ap.add_argument("--models-dir", default=os.environ.get("VNDUB_MODELS_DIR", str(Path(__file__).resolve().parents[2] / "models")))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.probe is not None:
        # as in serve: what the vendor packages print at import (even from C) goes to stderr, the result is one JSON line
        out = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8")
        sys.stdout.flush()
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        sys.stdout = sys.stderr
        results_p = run_probes(table, env, args.probe or sorted(table), Path(args.models_dir))
        out.write(json.dumps(json_safe(results_p), ensure_ascii=False, default=str) + "\n")
        out.flush()
        sys.exit(0 if all(r.get("ok") for r in results_p.values()) else 1)
    if args.selftest is None:
        serve(table, env)
        return
    os.environ.setdefault("HF_HOME", str(Path(args.models_dir) / "hf_home"))
    os.environ.setdefault("TORCH_HOME", str(Path(args.models_dir) / "torch_home"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    keys = args.selftest or sorted(table)
    results = {}
    for key in keys:
        params = _cli_params(key, env, Path(args.models_dir), args.device)
        adapter = params["model"]
        if adapter not in table:
            results[key] = {"ok": False, "error": f"env '{env}' has no adapter or backend '{key}'"}
            continue
        try:
            results[key] = table[adapter](params).selftest()
        except WeightsMissingError as e:
            results[key] = {"ok": False, "error": str(e)}
        except UnavailableError as e:
            results[key] = {"ok": False, "error": str(e), "hint": e.hint}
        except ImportError as e:
            results[key] = {"ok": False, "error": f"import failed in env '{env}': {e}"}
        except Exception as e:
            results[key] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    sys.exit(0 if all(r.get("ok") for r in results.values()) else 1)


def run_probes(table: dict[str, Any], env: str, keys: list[str], mdir: Path) -> dict[str, dict[str, Any]]:
    """``{adapter: {"ok", "s", ...} | {"ok": False, "error"}}``: :meth:`ModelImpl.probe` of each key, never raises."""
    results: dict[str, dict[str, Any]] = {}
    for key in keys:
        t0 = time.perf_counter()
        if key not in table:
            results[key] = {"ok": False, "error": f"env '{env}' has no adapter '{key}'"}
            continue
        params = {"name": key, "model": key, "options": {}, "device": "cpu", "models_dir": str(mdir), "weights": []}
        try:
            results[key] = table[key].probe(params)
        except Exception as e:  # ErrorDuringImport (pydoc), TypeError from a vendor that hid its ImportError ...
            sys.stderr.write(traceback.format_exc())
            results[key] = {"ok": False, "s": round(time.perf_counter() - t0, 2), "error": f"{type(e).__name__}: {e}"}
    return results


def _cli_params(key: str, env: str, mdir: Path, device: str) -> dict[str, Any]:
    """Load params for a CLI selftest. ``key`` is a backend name or an adapter key;
    for an adapter key shared by several backends, one with weights present wins."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from vndub.backends import registry  # light imports only (numpy, soundfile)
        from vndub.backends.weights import missing, resolved

        specs = [s for s in registry.all_specs().values() if s.env == env and (s.name == key or s.adapter == key)]
        specs.sort(key=lambda s: (s.name != key, bool(missing(mdir, s))))
        if specs:
            spec = specs[0]
            return {"name": spec.name, "model": spec.adapter, "options": spec.options, "device": device, "models_dir": str(mdir), "weights": resolved(mdir, spec), "caps": spec.caps}
    except Exception as e:  # registry needs core deps (numpy etc.) that may be absent in the env
        sys.stderr.write(f"registry unavailable ({e}); selftest without weight info\n")
    return {"name": key, "model": key, "options": {}, "device": device, "models_dir": str(mdir), "weights": []}

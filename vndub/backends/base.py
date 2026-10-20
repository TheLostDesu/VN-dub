"""Backend protocols, specs and the two implementations of a backend handle.

A *backend* is one model (or model family entry) with a role (ASR, SER, TTS,
VC ...). Stages never import model libraries; they get a handle from
``ctx.models.acquire(name)`` and call role methods on it. A handle is either

* :class:`RemoteBackend` -- the model runs in ``.venvs/<env>`` as a subprocess
  (``adapters/<env>/main.py``) speaking JSON-RPC over stdio, or
* :class:`LocalBackend` -- a light in-process implementation (mocks, CPU DSP).

Items and results are plain JSON dicts. Audio always travels by file path.
Every item has an ``id``; every result echoes it and may carry ``error``
instead of a payload (one bad clip never fails a whole batch).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict, runtime_checkable

# ----------------------------------------------------------------------------
# roles and item shapes (documentation + static typing; runtime uses dicts)
# ----------------------------------------------------------------------------

ROLES = (
    "asr", "align", "ser", "adv", "events", "embedder", "caption", "f0", "vad",
    "stress", "llm", "tts", "vc", "mos", "aesthetics",
)


class AudioItem(TypedDict, total=False):
    id: str
    wav: str  # absolute path
    out: str  # absolute output path (npy / wav) where the op writes a file
    text: str  # reference text (aligner), expected text
    lang: str


class TTSItem(TypedDict, total=False):
    id: str
    text: str
    lang: str
    timbre_ref: str | None
    timbre_ref_text: str | None
    emotion_ref: str | None
    instruct: str | None
    knobs: dict[str, Any]
    seed: int
    target_duration: float | None
    out: str


class VCItem(TypedDict, total=False):
    id: str
    src: str
    out: str
    target_ref: str | None
    model: str | None
    index: str | None
    f0_transpose: int
    params: dict[str, Any]


class Result(TypedDict, total=False):
    id: str
    error: str


# ----------------------------------------------------------------------------
# spec
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class WeightSpec:
    """One downloadable artifact for ``vndub models pull``.

    kind: ``hf`` (snapshot of a repo, optionally ``allow`` patterns),
    ``hf_file`` (single file from a repo), ``url`` (direct download),
    ``modelscope`` (FunASR model id; downloaded inside the env),
    ``builtin`` (bundled with a pip package, nothing to pull).
    ``post``: optional post-processing step run in the backend env after the
    download (e.g. ``ct2_convert`` for Whisper -> CTranslate2).
    """

    kind: str
    repo: str = ""
    filename: str = ""
    url: str = ""
    allow: tuple[str, ...] = ()
    revision: str | None = None
    subdir: str = ""
    post: str = ""
    size_gb: float = 0.0


@dataclass(frozen=True)
class BackendSpec:
    name: str
    role: str
    env: str | None  # None -> in-process LocalBackend
    adapter: str  # key the env adapter dispatches on
    license: str
    commercial_ok: bool
    vram_estimate_gb: float
    heavy: bool = True  # occupies the GPU; subject to the one-resident rule
    cpu_ok: bool = False  # adapter can run on CPU when CUDA is absent
    weights: tuple[WeightSpec, ...] = ()
    profiles: tuple[str, ...] = ("default",)  # default | test | optional
    options: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    homepage: str = ""
    fallback: str | None = None  # name of the backend to try if this one is unusable
    mock: str | None = None  # name of the mock used by profile=test fallback
    platforms: tuple[str, ...] = ("win32", "linux", "darwin")
    local_impl: str = ""  # "module:Class" for env=None backends
    # Capabilities the stages consult (TTS: instruct style, tags, knobs,
    # duration control, stress format, ref requirements; VC: trainable ...).
    caps: dict[str, Any] = field(default_factory=dict)

    @property
    def is_local(self) -> bool:
        return self.env is None


class BackendUnavailable(RuntimeError):
    """The backend cannot run here (env missing, weights missing, no CUDA ...)."""

    def __init__(self, name: str, reason: str, hint: str = "") -> None:
        self.name = name
        self.reason = reason
        self.hint = hint
        super().__init__(f"{name}: {reason}" + (f" -- {hint}" if hint else ""))


class WeightsMissing(BackendUnavailable):
    def __init__(self, name: str, detail: str = "") -> None:
        super().__init__(name, f"weights missing{': ' + detail if detail else ''}", f"run `vndub models pull {name}`")


# ----------------------------------------------------------------------------
# handle protocol
# ----------------------------------------------------------------------------


@runtime_checkable
class Backend(Protocol):
    spec: BackendSpec

    def load(self) -> dict[str, Any]: ...

    def unload(self) -> None: ...

    def call(self, op: str, items: list[dict[str, Any]] | None = None, **kwargs: Any) -> Any: ...

    def selftest(self) -> dict[str, Any]: ...

    def peak_vram(self) -> float | None: ...


class _RoleMixin:
    """Typed convenience wrappers; every one maps to ``call(op, items)``."""

    def call(self, op: str, items: list[dict[str, Any]] | None = None, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    # analysis
    def transcribe(self, items: list[AudioItem], lang: str, **opts: Any) -> list[dict[str, Any]]:
        return self.call("transcribe", items, lang=lang, **opts)

    def align(self, items: list[AudioItem], **opts: Any) -> list[dict[str, Any]]:
        return self.call("align", items, **opts)

    def emotion(self, items: list[AudioItem], **opts: Any) -> list[dict[str, Any]]:
        return self.call("emotion", items, **opts)

    def adv(self, items: list[AudioItem], **opts: Any) -> list[dict[str, Any]]:
        return self.call("adv", items, **opts)

    def events(self, items: list[AudioItem], **opts: Any) -> list[dict[str, Any]]:
        return self.call("events", items, **opts)

    def embed(self, items: list[AudioItem], **opts: Any) -> list[dict[str, Any]]:
        return self.call("embed", items, **opts)

    def caption(self, items: list[AudioItem], prompt: str, **opts: Any) -> list[dict[str, Any]]:
        return self.call("caption", items, prompt=prompt, **opts)

    def f0(self, items: list[AudioItem], **opts: Any) -> list[dict[str, Any]]:
        return self.call("f0", items, **opts)

    def vad(self, items: list[AudioItem], **opts: Any) -> list[dict[str, Any]]:
        return self.call("vad", items, **opts)

    def stress(self, texts: list[str], fmt: str = "plain", yo: bool = True) -> list[str]:
        res = self.call("stress", [{"id": str(i), "text": t} for i, t in enumerate(texts)], fmt=fmt, yo=yo)
        return [r.get("text", texts[i]) for i, r in enumerate(res)]

    # scoring
    def mos(self, items: list[AudioItem], **opts: Any) -> list[dict[str, Any]]:
        return self.call("mos", items, **opts)

    def aesthetics(self, items: list[AudioItem], **opts: Any) -> list[dict[str, Any]]:
        return self.call("aesthetics", items, **opts)

    # generation
    def synthesize(self, items: list[TTSItem], **opts: Any) -> list[dict[str, Any]]:
        return self.call("synthesize", items, **opts)

    def design(self, items: list[dict[str, Any]], **opts: Any) -> list[dict[str, Any]]:
        return self.call("design", items, **opts)

    def convert(self, items: list[VCItem], **opts: Any) -> list[dict[str, Any]]:
        return self.call("convert", items, **opts)

    def train(self, job: dict[str, Any]) -> dict[str, Any]:
        return self.call("train", None, job=job)

    # llm
    def chat(self, messages: list[dict[str, str]], **opts: Any) -> str:
        return self.call("chat", None, messages=messages, **opts)


class LocalBackend(_RoleMixin):
    """In-process backend. Subclasses implement ``op_<name>`` methods."""

    def __init__(self, spec: BackendSpec, options: dict[str, Any] | None = None, context: dict[str, Any] | None = None) -> None:
        self.spec = spec
        self.options = {**spec.options, **(options or {})}
        self.context = context or {}
        self.loaded = False

    def load(self) -> dict[str, Any]:
        self.loaded = True
        return {"device": "cpu", "local": True}

    def unload(self) -> None:
        self.loaded = False

    def call(self, op: str, items: list[dict[str, Any]] | None = None, **kwargs: Any) -> Any:
        fn = getattr(self, f"op_{op}", None)
        if fn is None:
            raise NotImplementedError(f"{self.spec.name} does not support op '{op}'")
        if items is None:
            return fn(**kwargs)
        out = []
        for it in items:
            try:
                out.append({"id": it.get("id"), **fn(it, **kwargs)})
            except Exception as e:  # one bad item never fails the batch
                out.append({"id": it.get("id"), "error": f"{type(e).__name__}: {e}"})
        return out

    def selftest(self) -> dict[str, Any]:
        return {"ok": True, "peak_vram_gb": 0.0, "detail": "local backend"}

    def peak_vram(self) -> float | None:
        return 0.0

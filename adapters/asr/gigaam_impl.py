"""GigaAM-v3 e2e CTC (Russian ASR) via Transformers ``trust_remote_code``.

The remote ``modeling_gigaam.py`` loads audio by calling the ``ffmpeg`` binary;
we replace that module-level loader with a soundfile one so the env does not
depend on ffmpeg being on PATH. ``transcribe`` is limited to 25 s per call, so
longer clips are cut into <= 24 s windows and the texts joined.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

from adapters.common import rpc
from adapters.common.rpc import ModelImpl, load_audio, save_audio, weight_dir

MAX_S = 24.0


OPTIONAL_REMOTE_IMPORTS = ("pyannote", "torchcodec")


def _allow_optional_imports() -> None:
    """GigaAM's remote code imports pyannote/torchcodec for long-form segmentation only.

    Transformers refuses to load a remote module when any imported package is
    missing; we never call the long-form path, so those two are tolerated.
    """
    import transformers.dynamic_module_utils as dmu

    if getattr(dmu.check_imports, "_vndub", False):
        return
    orig = dmu.check_imports

    def check_imports(filename):  # type: ignore[no-untyped-def]
        try:
            return orig(filename)
        except ImportError as e:
            if any(p in str(e) for p in OPTIONAL_REMOTE_IMPORTS):
                return dmu.get_relative_imports(filename)
            raise

    check_imports._vndub = True  # type: ignore[attr-defined]
    dmu.check_imports = check_imports


def _sf_loader(audio_path: str, sample_rate: int = 16000):
    import torch

    y, _ = rpc.load_audio(audio_path, sample_rate)
    return torch.from_numpy(y)


class GigaAMASR(ModelImpl):
    selftest_op = "transcribe"
    selftest_kwargs = {"lang": "ru"}

    probe_imports = ("torch", "transformers")  # what load() imports (ModelImpl.probe)

    def load(self) -> dict[str, Any]:
        import torch
        from transformers import AutoModel

        d = weight_dir(self.params, 0)
        _allow_optional_imports()
        self.model = AutoModel.from_pretrained(str(d), trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()
        patched = 0
        for mod_name, mod in list(sys.modules.items()):
            if "gigaam" in mod_name.lower() and not mod_name.startswith("adapters.") and hasattr(mod, "load_audio"):
                mod.load_audio = _sf_loader
                patched += 1
        torch.set_grad_enabled(False)
        return {"model_dir": str(d), "patched_loaders": patched}

    def op_transcribe(self, item: dict[str, Any], lang: str | None = "ru", **_: Any) -> dict[str, Any]:
        y, sr = load_audio(item["wav"], 16000)
        max_n = int(float(self.options.get("max_sec", 25.0)) * sr) - sr  # keep a 1 s margin
        win = int(min(MAX_S * sr, max_n))
        if len(y) <= win:
            return {"text": str(self.model.transcribe(item["wav"])).strip(), "lang": "ru"}
        texts = []
        with tempfile.TemporaryDirectory() as td:
            for i, start in enumerate(range(0, len(y), win)):
                chunk = y[start : start + win]
                if len(chunk) < sr // 4:
                    continue
                p = Path(td) / f"chunk{i}.wav"
                save_audio(p, chunk, sr)
                texts.append(str(self.model.transcribe(str(p))).strip())
        return {"text": " ".join(t for t in texts if t), "lang": "ru", "chunks": len(texts)}

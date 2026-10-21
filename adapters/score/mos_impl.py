"""Naturalness MOS: UTMOSv2 and UTMOS22 (SpeechMOS).

UTMOSv2 downloads its checkpoint with ``wget`` into a user cache, which does
not exist on Windows; ``models pull utmosv2`` fetches
``fold0_s42_best_model.pth`` with huggingface_hub into ``<models_dir>/utmosv2``
instead and instantiates the model once so its SSL backbone lands in HF_HOME.

UTMOS22 comes from ``torch.hub`` (``tarepan/SpeechMOS:v1.2.0``); with
``TORCH_HOME`` inside models_dir the hub repo and checkpoint are cached by
``models pull utmos22`` and loaded from the local cache afterwards.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from adapters.common.rpc import ModelImpl, WeightsMissingError, load_audio, models_dir, weight_dir

UTMOSV2_REPO = "sarulab-speech/UTMOSv2"
UTMOSV2_CKPT = "fold0_s42_best_model.pth"
SPEECHMOS = "tarepan/SpeechMOS:v1.2.0"


def _torch_home(params: dict[str, Any]) -> Path:
    return Path(os.environ.get("TORCH_HOME") or (models_dir(params) / "torch_home"))


class UTMOSv2(ModelImpl):
    selftest_op = "mos"

    def load(self) -> dict[str, Any]:
        import utmosv2

        d = weight_dir(self.params, 0)
        ckpt = d / UTMOSV2_CKPT
        if not ckpt.exists():
            raise WeightsMissingError(self.name, str(ckpt))
        try:
            self.model = utmosv2.create_model(pretrained=True, checkpoint_path=str(ckpt), device=self.device)
        except TypeError:  # older signature without device
            self.model = utmosv2.create_model(pretrained=True, checkpoint_path=str(ckpt))
        return {"checkpoint": str(ckpt)}

    def op_mos(self, item: dict[str, Any], **_: Any) -> dict[str, Any]:
        val = self.model.predict(input_path=item["wav"], device=self.device) if _accepts(self.model.predict, "device") else self.model.predict(input_path=item["wav"])
        return {"mos": round(float(val), 4)}

    @classmethod
    def pull(cls, params: dict[str, Any]) -> dict[str, Any]:
        from huggingface_hub import hf_hub_download

        d = Path(params["local_dir"])
        d.mkdir(parents=True, exist_ok=True)
        p = hf_hub_download(repo_id=UTMOSV2_REPO, filename=UTMOSV2_CKPT, local_dir=str(d))
        import utmosv2

        utmosv2.create_model(pretrained=True, checkpoint_path=p)  # caches the SSL backbone
        return {"ok": True, "detail": p}


def _accepts(fn: Any, name: str) -> bool:
    import inspect

    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _speechmos_cache(params: dict[str, Any]) -> Path:
    return _torch_home(params) / "hub" / SPEECHMOS.replace("/", "_").replace(":", "_")


class UTMOS22(ModelImpl):
    selftest_op = "mos"

    def load(self) -> dict[str, Any]:
        import torch

        weight_dir(self.params, 0)  # marker written by `models pull utmos22`
        cache = _speechmos_cache(self.params)
        if not cache.exists():
            raise WeightsMissingError(self.name, f"torch.hub cache {cache} missing")
        self.model = torch.hub.load(str(cache), "utmos22_strong", source="local", trust_repo=True)
        self.model.to(self.device).eval()
        self._torch = torch
        return {"hub_cache": str(cache)}

    def op_mos(self, item: dict[str, Any], **_: Any) -> dict[str, Any]:
        torch = self._torch
        y, sr = load_audio(item["wav"], 16000)
        with torch.inference_mode():
            score = self.model(torch.from_numpy(y).unsqueeze(0).to(self.device), sr)
        return {"mos": round(float(score.reshape(-1)[0]), 4)}

    @classmethod
    def pull(cls, params: dict[str, Any]) -> dict[str, Any]:
        import torch

        home = _torch_home(params)
        torch.hub.set_dir(str(home / "hub"))
        model = torch.hub.load(SPEECHMOS, "utmos22_strong", trust_repo=True)
        del model
        return {"ok": True, "detail": str(_speechmos_cache(params))}

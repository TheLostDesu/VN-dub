"""Speaker embeddings written as ``.npy`` (ECAPA 192-d, WavLM-SV 512-d), L2-normalized."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from adapters.common.rpc import ModelImpl, load_audio, weight_dir


def _save(out: str, vec) -> dict[str, Any]:
    import numpy as np

    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n > 0:
        v = v / n
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.save(out, v)
    return {"path": out, "dim": int(v.size)}


class EcapaEmbedder(ModelImpl):
    selftest_op = "embed"

    def load(self) -> dict[str, Any]:
        import torch

        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:  # speechbrain < 1.0
            from speechbrain.pretrained import EncoderClassifier  # type: ignore
        d = weight_dir(self.params, 0)
        # savedir == source: speechbrain uses the pulled files in place (no copies, no symlinks,
        # and no failure on the optional custom.py that this repo does not ship)
        # hyperparams.yaml points pretrained_path at the hub repo: redirect it to the local snapshot
        kwargs: dict[str, Any] = {"source": str(d), "savedir": str(d), "run_opts": {"device": self.device}, "overrides": {"pretrained_path": str(d)}}
        try:
            from speechbrain.utils.fetching import LocalStrategy

            kwargs["local_strategy"] = LocalStrategy.COPY  # no symlinks on Windows
        except ImportError:
            pass
        self.model = EncoderClassifier.from_hparams(**kwargs)
        self.model.eval()
        self._torch = torch
        return {"source": str(d)}

    def op_embed(self, item: dict[str, Any], **_: Any) -> dict[str, Any]:
        torch = self._torch
        y, _ = load_audio(item["wav"], 16000)
        with torch.inference_mode():
            emb = self.model.encode_batch(torch.from_numpy(y).unsqueeze(0).to(self.device))
        return _save(item["out"], emb.squeeze().cpu().numpy())


class WavLMSV(ModelImpl):
    selftest_op = "embed"

    def load(self) -> dict[str, Any]:
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector

        d = weight_dir(self.params, 0)
        self.fe = AutoFeatureExtractor.from_pretrained(str(d))
        self.model = WavLMForXVector.from_pretrained(str(d)).to(self.device).eval()
        self._torch = torch
        return {"source": str(d)}

    def op_embed(self, item: dict[str, Any], **_: Any) -> dict[str, Any]:
        torch = self._torch
        y, _ = load_audio(item["wav"], 16000)
        inputs = self.fe(y, sampling_rate=16000, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            emb = self.model(**inputs).embeddings
        return _save(item["out"], emb.squeeze().cpu().numpy())

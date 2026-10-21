"""Qwen3-ASR (``qwen-asr`` package, transformers backend)."""

from __future__ import annotations

from typing import Any

from adapters.common.rpc import ModelImpl, lang_name, weight_dir


class Qwen3ASR(ModelImpl):
    selftest_op = "transcribe"
    selftest_kwargs = {"lang": "ja"}

    def load(self) -> dict[str, Any]:
        import torch
        from qwen_asr import Qwen3ASRModel

        d = weight_dir(self.params, 0)
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.model = Qwen3ASRModel.from_pretrained(str(d), dtype=dtype, device_map=self.device, max_inference_batch_size=8, max_new_tokens=256)
        return {"model_dir": str(d)}

    def op_transcribe(self, item: dict[str, Any], lang: str | None = None, **_: Any) -> dict[str, Any]:
        lg = lang or item.get("lang")
        res = self.model.transcribe(audio=item["wav"], language=lang_name(lg, default=None) if lg else None)
        r = res[0]
        return {"text": str(r.text).strip(), "lang": getattr(r, "language", lg)}

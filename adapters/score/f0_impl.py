"""torchcrepe F0 at a 10 ms hop; frames with periodicity below the threshold -> 0 Hz."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from adapters.common.rpc import ModelImpl, load_audio

SR = 16000
HOP = 160  # 10 ms


class TorchCrepeF0(ModelImpl):
    selftest_op = "f0"

    def load(self) -> dict[str, Any]:
        import torch
        import torchcrepe

        self._torch, self._crepe = torch, torchcrepe
        return {"model": self.options.get("model", "full")}

    def op_f0(self, item: dict[str, Any], **_: Any) -> dict[str, Any]:
        import numpy as np

        torch, crepe = self._torch, self._crepe
        y, _ = load_audio(item["wav"], SR)
        audio = torch.from_numpy(y).unsqueeze(0)
        pitch, periodicity = crepe.predict(
            audio, SR, HOP,
            float(self.options.get("fmin", 50.0)), float(self.options.get("fmax", 1100.0)),
            self.options.get("model", "full"), batch_size=512, device=self.device, return_periodicity=True,
        )
        periodicity = crepe.filter.median(periodicity, 3)
        pitch = crepe.filter.mean(pitch, 3)
        f0 = pitch.squeeze(0).cpu().numpy().astype(np.float32)
        per = periodicity.squeeze(0).cpu().numpy()
        f0[per < float(self.options.get("periodicity_threshold", 0.21))] = 0.0
        out = Path(item["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, f0)
        return {"path": str(out), "hop_s": HOP / SR, "n": int(f0.size), "voiced_ratio": round(float((f0 > 0).mean()) if f0.size else 0.0, 3)}

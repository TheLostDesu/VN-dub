"""Audiobox Aesthetics (CE / CU / PC / PQ) from the local ``checkpoint.pt``."""

from __future__ import annotations

from typing import Any

from adapters.common.rpc import ModelImpl, WeightsMissingError, weight_dir


class AudioboxAesthetics(ModelImpl):
    selftest_op = "aesthetics"

    def load(self) -> dict[str, Any]:
        from audiobox_aesthetics.infer import AesPredictor

        d = weight_dir(self.params, 0)
        ckpt = d / "checkpoint.pt"
        if not ckpt.exists():
            raise WeightsMissingError(self.name, str(ckpt))
        self.pred = AesPredictor(checkpoint_pth=str(ckpt), data_col="path")
        self.device = str(getattr(self.pred, "device", self.device))
        return {"checkpoint": str(ckpt)}

    def batch_aesthetics(self, items: list[dict[str, Any]], **_: Any) -> list[dict[str, Any]]:
        out = []
        for it in items:
            try:
                r = self.pred.forward([{"path": it["wav"]}])[0]
                out.append({"id": it["id"], "ce": round(r["CE"], 4), "cu": round(r["CU"], 4), "pc": round(r["PC"], 4), "pq": round(r["PQ"], 4)})
            except Exception as e:
                out.append({"id": it["id"], "error": f"{type(e).__name__}: {e}"})
        return out

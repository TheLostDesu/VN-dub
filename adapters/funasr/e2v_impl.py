"""emotion2vec+ through FunASR: 9-class probabilities + 768-d utterance embedding.

FunASR returns labels like ``"生气/angry"``; they are mapped to the English
emotion2vec set (``E2V_LABELS`` in vndub.backends.ser.emotion2vec).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from adapters.common.rpc import ModelImpl, weight_dir

E2V_LABELS = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown"]


def english_label(raw: str) -> str:
    s = str(raw).split("/")[-1].strip().lower()
    if s in ("<unk>", "unk", ""):
        return "unknown"
    return s


def funasr_model(local_dir: Path, device: str):
    from funasr import AutoModel

    return AutoModel(model=str(local_dir), device=device, disable_update=True, log_level="ERROR")


class Emotion2Vec(ModelImpl):
    """No ``batch_emotion`` on purpose: ``AutoModel.generate(input=[paths])`` ends up in ``Emotion2vec.inference``,
    which loops over the clips one by one (``extract_features`` on a single (1, T) tensor), so a list call gives
    identical outputs and no speed-up."""

    selftest_op = "emotion"

    probe_imports = ("funasr",)  # what load() imports (ModelImpl.probe)

    def load(self) -> dict[str, Any]:
        d = weight_dir(self.params, 0)
        self.model = funasr_model(d, self.device)
        return {"model_dir": str(d)}

    def op_emotion(self, item: dict[str, Any], **_: Any) -> dict[str, Any]:
        import numpy as np

        want_emb = bool(item.get("out"))
        res = self.model.generate(item["wav"], granularity="utterance", extract_embedding=want_emb)
        r = res[0] if isinstance(res, list) else res
        labels = [english_label(l) for l in r.get("labels", [])]
        scores = [float(s) for s in r.get("scores", [])]
        probs = {l: 0.0 for l in E2V_LABELS}
        for l, s in zip(labels, scores):
            probs[l] = probs.get(l, 0.0) + s
        tot = sum(probs.values()) or 1.0
        probs = {k: round(v / tot, 5) for k, v in probs.items()}
        out: dict[str, Any] = {"probs": probs, "label": max(probs, key=probs.get)}
        if want_emb:
            feats = r.get("feats")
            if feats is None:
                raise RuntimeError("emotion2vec returned no embedding")
            v = np.asarray(feats, dtype=np.float32).reshape(-1)
            p = Path(item["out"])
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, v)
            out["embedding_path"] = str(p)
            out["dim"] = int(v.size)
        return out

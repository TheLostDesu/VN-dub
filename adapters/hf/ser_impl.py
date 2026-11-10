"""Speech emotion: litagin anime SER (10 classes) and audeering A/D/V (regression)."""

from __future__ import annotations

import sys
from typing import Any

from adapters.common.rpc import ModelImpl, audio_seconds, load_audio, load_audios, run_in_batches, weight_dir

ANIME_LABELS = ["Angry", "Disgusted", "Embarrassed", "Fearful", "Happy", "Sad", "Surprised", "Neutral", "Sexual1", "Sexual2"]


class AnimeSER(ModelImpl):
    """litagin/anime_speech_emotion_classification via the audio-classification pipeline."""

    selftest_op = "emotion"

    probe_imports = ("transformers",)  # what load() imports (ModelImpl.probe)

    def load(self) -> dict[str, Any]:
        from transformers import pipeline

        d = weight_dir(self.params, 0)
        dev = 0 if self.device.startswith("cuda") else -1
        if self.device.startswith("cuda:"):
            dev = int(self.device.split(":")[1])
        self.pipe = pipeline("audio-classification", model=str(d), feature_extractor=str(d), trust_remote_code=True, device=dev)
        # padded batching is only exact when the extractor returns an attention mask AND the conv feature
        # encoder normalises per frame ("layer"); a "group" norm (wav2vec2/WavLM base style) normalises over the
        # padded time axis, so padding would change the probabilities -> keep the per-clip path for those
        fe, cfg = self.pipe.feature_extractor, self.pipe.model.config
        self._padded_batch_ok = bool(getattr(fe, "return_attention_mask", False)) and getattr(cfg, "feat_extract_norm", None) == "layer"
        sys.stderr.write(f"[{self.name}] padded batching {'on' if self._padded_batch_ok else 'off'} (return_attention_mask={getattr(fe, 'return_attention_mask', None)}, feat_extract_norm={getattr(cfg, 'feat_extract_norm', None)})\n")
        return {"model_dir": str(d)}

    def op_emotion(self, item: dict[str, Any], **_: Any) -> dict[str, Any]:
        y, sr = load_audio(item["wav"], 16000)
        res = self.pipe({"raw": y, "sampling_rate": sr}, top_k=None)
        probs = {l: 0.0 for l in ANIME_LABELS}
        for r in res:
            probs[str(r["label"])] = round(float(r["score"]), 5)
        return {"label": max(probs, key=probs.get), "probs": probs}

    def batch_emotion(self, items: list[dict[str, Any]], **kw: Any) -> list[dict[str, Any]]:
        """Several clips per forward pass: ``feature_extractor(list, padding=True, return_attention_mask=True)``
        then one ``model(**inputs)`` (the same computation the audio-classification pipeline does per clip).

        Clips are sorted by duration and cut into sub-batches of ``options['batch_size']`` (default 8). The
        extractor normalises every clip with its own length and the attention mask keeps padded frames out of
        the transformer and out of the mean pooling of ``*ForSequenceClassification``; the conv feature encoder
        has no padding, so the probabilities equal the single-clip ones up to float noise. This only holds for
        models whose extractor returns an attention mask and whose conv encoder uses per-frame layer norm; that
        is checked in :meth:`load` and otherwise the clips run one by one (same numbers as :meth:`op_emotion`).
        A failing sub-batch (incl. CUDA OOM) is redone clip by clip with :meth:`op_emotion`.
        """
        size = self.sub_batch_size() if getattr(self, "_padded_batch_ok", False) else 1
        return run_in_batches(items, size, self._emotion_chunk, lambda it: self.op_emotion(it, **kw), label=self.name, sort_key=audio_seconds)

    def _emotion_chunk(self, items: list[dict[str, Any]]) -> list[Any]:
        import torch

        ys, out = load_audios(items)
        ok = [i for i, e in enumerate(out) if e is None]
        if ok:
            pipe = self.pipe
            enc = pipe.feature_extractor([ys[i] for i in ok], sampling_rate=16000, padding=True, return_attention_mask=True, return_tensors="pt")
            enc = {k: v.to(pipe.device) for k, v in enc.items()}
            with torch.inference_mode():
                logits = pipe.model(**enc).logits
            scores = logits.float().softmax(-1).cpu()
            id2label = pipe.model.config.id2label
            for r, i in enumerate(ok):
                probs = {l: 0.0 for l in ANIME_LABELS}
                for k, lab in id2label.items():
                    probs[str(lab)] = round(float(scores[r, int(k)]), 5)
                out[i] = {"label": max(probs, key=probs.get), "probs": probs}
        return out


def _emotion_model_class():
    import torch
    import torch.nn as nn
    from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

    class RegressionHead(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, features):
            x = self.dropout(features)
            x = torch.tanh(self.dense(x))
            x = self.dropout(x)
            return self.out_proj(x)

    class EmotionModel(Wav2Vec2PreTrainedModel):
        """Model-card head of audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim."""

        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead(config)
            if hasattr(self, "post_init"):
                self.post_init()
            else:  # pragma: no cover - old transformers
                self.init_weights()

        def forward(self, input_values):
            hidden = self.wav2vec2(input_values)[0]
            hidden = torch.mean(hidden, dim=1)
            return hidden, self.classifier(hidden)

    return EmotionModel


class AudeeringADV(ModelImpl):
    """Arousal / dominance / valence (~0..1). Batch size 1 by design (known batching issue)."""

    selftest_op = "adv"

    probe_imports = ("torch", "transformers")  # what load() imports (ModelImpl.probe)

    def load(self) -> dict[str, Any]:
        import torch
        from transformers import Wav2Vec2Processor

        d = weight_dir(self.params, 0)
        self.processor = Wav2Vec2Processor.from_pretrained(str(d))
        self.model = _emotion_model_class().from_pretrained(str(d)).to(self.device).eval()
        self._torch = torch
        return {"model_dir": str(d)}

    def op_adv(self, item: dict[str, Any], **_: Any) -> dict[str, Any]:
        torch = self._torch
        y, sr = load_audio(item["wav"], 16000)
        x = self.processor(y, sampling_rate=sr)["input_values"][0]
        x = torch.as_tensor(x, dtype=torch.float32).reshape(1, -1).to(self.device)
        with torch.inference_mode():
            _, logits = self.model(x)
        a, d, v = (float(t) for t in logits[0].cpu())
        clip = lambda z: round(min(1.0, max(0.0, z)), 4)  # noqa: E731
        return {"arousal": clip(a), "dominance": clip(d), "valence": clip(v)}

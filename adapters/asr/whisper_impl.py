"""Whisper checkpoints through faster-whisper (CTranslate2).

* CTranslate2 checkpoints (``Systran/faster-whisper-*``, ``mobiuslabsgmbh/...``)
  are used as downloaded.
* Transformers checkpoints (``litagin/anime-whisper``,
  ``antony66/whisper-large-v3-russian``) carry ``post: ct2_convert``; ``models
  pull`` converts them once into ``<local_dir>/ct2`` (float16 weights; int8 is
  applied at load time through ``compute_type``).

anime-whisper rules from its model card: no initial prompt,
``no_repeat_ngram_size=5``, ``condition_on_previous_text=False``.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

from adapters.common.rpc import ModelImpl, WeightsMissingError, load_audio, weight_dir

_TOKENIZER_FILES = ["tokenizer.json", "preprocessor_config.json", "vocabulary.json", "vocabulary.txt"]


def _model_dir(params: dict[str, Any]) -> Path:
    d = weight_dir(params, 0)
    w = (params.get("weights") or [{}])[0]
    if w.get("post") == "ct2_convert":
        d = d / "ct2"
    if not (d / "model.bin").exists():
        raise WeightsMissingError(params.get("name", "?"), f"no CTranslate2 model.bin in {d}")
    return d


class FasterWhisperASR(ModelImpl):
    """No ``batch_transcribe`` on purpose: ``WhisperModel.transcribe`` takes one audio; ``BatchedInferencePipeline``
    only batches the 30 s windows of ONE long audio (VAD-cut), it has no cross-file batching for short clips and
    would change the decoding setup (no ``condition_on_previous_text`` chain, VAD windows)."""

    selftest_op = "transcribe"
    selftest_kwargs = {"lang": "ja"}

    def load(self) -> dict[str, Any]:
        from faster_whisper import WhisperModel

        d = _model_dir(self.params)
        compute = self.options.get("compute_type", "int8")
        if self.device == "cpu" and "float16" in compute:
            compute = "int8"
        if self.device.startswith("cuda") and compute == "int8":
            compute = "int8_float16"
        dev, idx = ("cuda", int(self.device.split(":")[1]) if ":" in self.device else 0) if self.device.startswith("cuda") else ("cpu", 0)
        self.model = WhisperModel(str(d), device=dev, device_index=idx, compute_type=compute, cpu_threads=int(self.options.get("cpu_threads", 0)))
        return {"model_dir": str(d), "compute_type": compute}

    def op_transcribe(self, item: dict[str, Any], lang: str | None = None, **_: Any) -> dict[str, Any]:
        audio, _sr = load_audio(item["wav"], 16000)
        language = lang or item.get("lang") or self.options.get("language")
        kwargs: dict[str, Any] = {
            "language": language,
            "beam_size": int(self.options.get("beam_size", 5)),
            "condition_on_previous_text": bool(self.options.get("condition_on_previous_text", False)),
            "initial_prompt": None,  # never prompt (anime-whisper hallucinates with prompts)
            "vad_filter": bool(self.options.get("vad_filter", False)),
            "without_timestamps": False,
        }
        if self.options.get("no_repeat_ngram_size"):
            kwargs["no_repeat_ngram_size"] = int(self.options["no_repeat_ngram_size"])
        if self.options.get("repetition_penalty"):
            kwargs["repetition_penalty"] = float(self.options["repetition_penalty"])
        segments, info = self.model.transcribe(audio, **kwargs)
        segs = [{"t0": round(s.start, 3), "t1": round(s.end, 3), "text": s.text, "avg_logprob": round(s.avg_logprob, 4), "no_speech_prob": round(s.no_speech_prob, 4)} for s in segments]
        text = "".join(s["text"] for s in segs).strip()
        return {"text": text, "lang": info.language, "lang_prob": round(float(info.language_probability), 4), "segments": segs}

    @classmethod
    def pull(cls, params: dict[str, Any]) -> dict[str, Any]:
        """post=ct2_convert: Transformers Whisper -> CTranslate2 into <local_dir>/ct2."""
        if params.get("post") != "ct2_convert":
            return {"ok": True, "detail": "nothing to do"}
        src = Path(params["local_dir"])
        out = src / "ct2"
        if (out / "model.bin").exists():
            return {"ok": True, "detail": f"already converted: {out}"}
        from ctranslate2.converters import TransformersConverter

        copy = [f for f in _TOKENIZER_FILES if (src / f).exists()]
        tmp = src / "ct2.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        TransformersConverter(str(src), copy_files=copy, load_as_float16=True).convert(str(tmp), quantization="float16", force=True)
        if not (tmp / "tokenizer.json").exists():
            # faster-whisper falls back to downloading a tokenizer from the hub when this is
            # missing, which would break offline runs: materialize it from the HF tokenizer.
            from transformers import WhisperTokenizerFast

            WhisperTokenizerFast.from_pretrained(str(src)).backend_tokenizer.save(str(tmp / "tokenizer.json"))
        tmp.rename(out)
        sys.stderr.write(f"converted {src} -> {out}\n")
        return {"ok": True, "detail": f"converted to {out}"}

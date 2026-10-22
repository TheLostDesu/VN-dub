"""Qwen3-ASR (``qwen-asr`` package, transformers backend)."""

from __future__ import annotations

from typing import Any

from adapters.common.rpc import ModelImpl, audio_seconds, lang_name, run_in_batches, weight_dir


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

    def batch_transcribe(self, items: list[dict[str, Any]], lang: str | None = None, **kw: Any) -> list[dict[str, Any]]:
        """Several clips per ``Qwen3ASRModel.transcribe(audio=[paths], language=[names])`` call.

        The library builds one padded batch (processor ``padding=True``, attention masks for text and audio
        features) and runs ``generate`` on it, so lengths are masked by the model itself; a language list
        carries the per-item language (``None`` = auto-detect). Audio is passed as paths, as in
        :meth:`op_transcribe`, so loading/resampling is the library's own and identical. Clips are sorted by
        duration and cut into sub-batches of ``options['batch_size']`` (default 8, same as the model's
        ``max_inference_batch_size``). Greedy decoding gives the same text as the single-clip call except for
        rare ties that bf16 batched kernels can flip. A failing sub-batch (incl. CUDA OOM, or one unreadable
        clip) is redone clip by clip with :meth:`op_transcribe`.
        """
        return run_in_batches(items, self.sub_batch_size(), lambda sub: self._transcribe_chunk(sub, lang), lambda it: self.op_transcribe(it, lang=lang, **kw), label=self.name, sort_key=audio_seconds)

    def _transcribe_chunk(self, items: list[dict[str, Any]], lang: str | None) -> list[Any]:
        langs = [lang or it.get("lang") for it in items]
        names = [lang_name(lg, default=None) if lg else None for lg in langs]
        res = self.model.transcribe(audio=[it["wav"] for it in items], language=names)
        if len(res) != len(items):
            raise RuntimeError(f"Qwen3-ASR returned {len(res)} results for {len(items)} clips")
        return [{"text": str(r.text).strip(), "lang": getattr(r, "language", lg)} for r, lg in zip(res, langs)]

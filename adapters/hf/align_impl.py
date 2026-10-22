"""Forced alignment -> per-character timestamps ``[{c, t0, t1}]``.

* ``qwen3_aligner``: Qwen3-ForcedAligner-0.6B through the ``qwen-asr`` package
  (units are characters for ja/zh; multi-character units are split evenly).
* ``ctc_align``: WhisperX's algorithm with a character CTC model
  (jonatasgrosman/wav2vec2-large-xlsr-53-japanese): ``whisperx.align`` when
  whisperx is installed, otherwise ``torchaudio.functional.forced_align``.
  Characters missing from the CTC vocabulary get interpolated times.
"""

from __future__ import annotations

import unicodedata
from typing import Any

from adapters.common.rpc import ModelImpl, audio_seconds, lang_name, load_audio, load_audios, run_in_batches, weight_dir


def _split_unit(text: str, t0: float, t1: float) -> list[dict[str, Any]]:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return []
    step = (t1 - t0) / len(chars)
    return [{"c": c, "t0": round(t0 + i * step, 3), "t1": round(t0 + (i + 1) * step, 3)} for i, c in enumerate(chars)]


class Qwen3Aligner(ModelImpl):
    selftest_op = "align"

    def load(self) -> dict[str, Any]:
        import torch
        from qwen_asr import Qwen3ForcedAligner

        d = weight_dir(self.params, 0)
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.model = Qwen3ForcedAligner.from_pretrained(str(d), dtype=dtype, device_map=self.device)
        return {"model_dir": str(d)}

    def op_align(self, item: dict[str, Any], **_: Any) -> dict[str, Any]:
        y, sr = load_audio(item["wav"], 16000)
        dur = len(y) / sr
        res = self.model.align(audio=(y, sr), text=item.get("text") or "", language=lang_name(item.get("lang", "ja")))
        units = res[0] if res else []
        scale = 1.0
        if units and float(units[-1].end_time) > dur * 20:  # milliseconds
            scale = 0.001
        chars: list[dict[str, Any]] = []
        for u in units:
            chars += _split_unit(str(u.text), float(u.start_time) * scale, float(u.end_time) * scale)
        return {"chars": chars, "backend": "qwen3_aligner"}

    def batch_align(self, items: list[dict[str, Any]], **kw: Any) -> list[dict[str, Any]]:
        """Several clips per ``Qwen3ForcedAligner.align(audio=[(y, sr)], text=[...], language=[...])`` call.

        ``align`` runs ONE forward pass over the whole list (processor ``padding=True`` with attention masks,
        argmax of the thinker logits at the ``<timestamp>`` tokens of each clip), so lengths are masked by the
        model itself; the list is therefore cut into sub-batches of ``options['batch_size']`` (default 8) to
        bound memory. Clips are loaded with :func:`load_audio` exactly like :meth:`op_align` (an unreadable clip
        only fails itself), sorted by duration, and the ms/s unit guard is applied per clip. The timestamps are
        discrete (80 ms steps), so the result equals the single-clip call except for rare argmax ties under
        bf16. A failing sub-batch (incl. CUDA OOM) is redone clip by clip with :meth:`op_align`.
        """
        return run_in_batches(items, self.sub_batch_size(), self._align_chunk, lambda it: self.op_align(it, **kw), label=self.name, sort_key=audio_seconds)

    def _align_chunk(self, items: list[dict[str, Any]]) -> list[Any]:
        ys, out = load_audios(items)
        ok = [i for i, e in enumerate(out) if e is None]
        if ok:
            res = self.model.align(
                audio=[(ys[i], 16000) for i in ok],
                text=[items[i].get("text") or "" for i in ok],
                language=[lang_name(items[i].get("lang", "ja")) for i in ok],
            )
            if len(res) != len(ok):
                raise RuntimeError(f"Qwen3-ForcedAligner returned {len(res)} results for {len(ok)} clips")
            for units, i in zip(res, ok):
                dur = len(ys[i]) / 16000
                scale = 1.0
                if len(units) and float(units[-1].end_time) > dur * 20:  # milliseconds
                    scale = 0.001
                chars: list[dict[str, Any]] = []
                for u in units:
                    chars += _split_unit(str(u.text), float(u.start_time) * scale, float(u.end_time) * scale)
                out[i] = {"chars": chars, "backend": "qwen3_aligner"}
        return out


class CTCAligner(ModelImpl):
    selftest_op = "align"

    def load(self) -> dict[str, Any]:
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        d = weight_dir(self.params, 0)
        self.processor = Wav2Vec2Processor.from_pretrained(str(d))
        self.model = Wav2Vec2ForCTC.from_pretrained(str(d)).to(self.device).eval()
        self.vocab = {k: v for k, v in self.processor.tokenizer.get_vocab().items()}
        self.blank = self.processor.tokenizer.pad_token_id or 0
        self._torch = torch
        try:
            import whisperx  # noqa: F401

            self.whisperx = True
        except ImportError:
            self.whisperx = False
        return {"model_dir": str(d), "whisperx": self.whisperx}

    def _whisperx(self, y, sr: int, text: str) -> list[dict[str, Any]]:
        import whisperx

        meta = {"language": "ja", "dictionary": {c.lower(): i for c, i in self.vocab.items()}, "type": "huggingface"}
        res = whisperx.align([{"start": 0.0, "end": len(y) / sr, "text": text}], self.model, meta, y, self.device, return_char_alignments=True)
        out = []
        for seg in res.get("segments", []):
            for ch in seg.get("chars", []):
                if ch.get("start") is not None and str(ch.get("char", "")).strip():
                    out.append({"c": ch["char"], "t0": round(ch["start"], 3), "t1": round(ch["end"], 3)})
        return out

    def op_align(self, item: dict[str, Any], **_: Any) -> dict[str, Any]:
        torch = self._torch
        import torchaudio.functional as AF

        y, sr = load_audio(item["wav"], 16000)
        text = unicodedata.normalize("NFKC", item.get("text") or "")
        chars = [c for c in text if not c.isspace()]
        if not chars:
            return {"chars": [], "backend": "ctc_align"}
        if self.whisperx:
            try:
                return {"chars": self._whisperx(y, sr, text), "backend": "whisperx"}
            except Exception:
                pass
        inputs = self.processor(y, sampling_rate=sr, return_tensors="pt").input_values.to(self.device)
        with torch.inference_mode():
            emissions = torch.log_softmax(self.model(inputs).logits, dim=-1).cpu()
        known = [(i, c) for i, c in enumerate(chars) if c in self.vocab]
        frame_s = (len(y) / sr) / emissions.shape[1]
        times: dict[int, tuple[float, float]] = {}
        if known and emissions.shape[1] >= len(known):
            targets = torch.tensor([[self.vocab[c] for _, c in known]], dtype=torch.int32)
            aligned, _ = AF.forced_align(emissions, targets, blank=self.blank)
            spans = AF.merge_tokens(aligned[0], torch.zeros(aligned.shape[1]))
            for (idx, _c), span in zip(known, spans):
                times[idx] = (span.start * frame_s, span.end * frame_s)
        # interpolate characters missing from the vocabulary (kanji/punctuation not in the CTC set)
        out, last = [], 0.0
        for i, c in enumerate(chars):
            if i in times:
                t0, t1 = times[i]
            else:
                nxt = next((times[j][0] for j in range(i + 1, len(chars)) if j in times), len(y) / sr)
                t0, t1 = last, max(last, (last + nxt) / 2)
            out.append({"c": c, "t0": round(t0, 3), "t1": round(t1, 3)})
            last = t1
        return {"chars": out, "backend": "ctc_align"}

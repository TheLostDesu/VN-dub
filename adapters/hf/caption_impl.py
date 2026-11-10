"""Qwen2-Audio-7B-Instruct, 4-bit NF4 (bitsandbytes): free-text delivery captions.

Fit in 8 GB is UNVERIFIED. An OOM during load is reported as "backend
unavailable" so the analyze stage disables captions and logs the reason.
"""

from __future__ import annotations

from typing import Any

from adapters.common.rpc import ModelImpl, UnavailableError, free_cuda, is_oom, load_audio, weight_dir

DEFAULT_PROMPT = (
    "Describe how this line is spoken: emotion, intensity, tempo, voice quality "
    "(e.g. breathy, tense, whispery, bright), and notable events (laughter, sigh, "
    "crying, breath). Answer in English in at most 40 words."
)


class Qwen2AudioCaptioner(ModelImpl):
    selftest_op = "caption"
    selftest_kwargs = {"prompt": DEFAULT_PROMPT}

    probe_imports = ("torch", "transformers")  # what load() imports (ModelImpl.probe)

    def load(self) -> dict[str, Any]:
        import torch
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        d = weight_dir(self.params, 0)
        quant = self.options.get("quantization", "nf4")
        kwargs: dict[str, Any] = {}
        if quant == "nf4":
            if not self.device.startswith("cuda"):
                raise UnavailableError("Qwen2-Audio NF4 needs CUDA (bitsandbytes 4-bit)", "disable analyze.caption on CPU-only machines")
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
            kwargs["device_map"] = self.device
        else:
            kwargs["torch_dtype"] = torch.float16 if self.device.startswith("cuda") else torch.float32
        try:
            self.processor = AutoProcessor.from_pretrained(str(d))
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(str(d), **kwargs)
        except Exception as e:
            if is_oom(e):
                free_cuda()
                raise UnavailableError(f"Qwen2-Audio does not fit: {e}", "keep analyze.caption: false") from e
            raise
        if quant != "nf4":
            self.model.to(self.device)
        self._torch = torch
        return {"model_dir": str(d), "quantization": quant}

    def op_caption(self, item: dict[str, Any], prompt: str = DEFAULT_PROMPT, **_: Any) -> dict[str, Any]:
        torch = self._torch
        sr = self.processor.feature_extractor.sampling_rate
        y, _ = load_audio(item["wav"], sr)
        conv = [{"role": "user", "content": [{"type": "audio", "audio_url": item["wav"]}, {"type": "text", "text": prompt or DEFAULT_PROMPT}]}]
        text = self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        try:
            inputs = self.processor(text=text, audio=[y], sampling_rate=sr, return_tensors="pt", padding=True)
        except TypeError:  # older processors use `audios=`
            inputs = self.processor(text=text, audios=[y], sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            gen = self.model.generate(**inputs, max_new_tokens=int(self.options.get("max_new_tokens", 80)), do_sample=False)
        gen = gen[:, inputs["input_ids"].shape[1] :]
        cap = self.processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return {"caption": cap.strip()}

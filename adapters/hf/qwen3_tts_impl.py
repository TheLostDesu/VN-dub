"""Qwen3-TTS 12Hz (``qwen-tts`` package).

Modes (``options.mode``: auto | clone | custom):

* clone  -- Base model, ``generate_voice_clone`` from ``timbre_ref``.
  ``x_vector_only_mode=True`` (DEFAULT, or whenever no reference transcript is
  given) uses only the speaker embedding, which limits language leakage from a
  Japanese reference. The Base model has no instruct channel: an instruction on
  a clone item is recorded as ignored.
* custom -- CustomVoice model, ``generate_custom_voice`` with a built-in speaker
  and the natural-language ``instruct`` (route-B performer that VC re-timbres).
* auto   -- clone when the item has a timbre reference, custom otherwise
  (``knobs.qwen_mode`` overrides per item).

Only one 1.7B model is resident at a time (8 GB budget): items are grouped by
mode and the models are swapped between groups.
"""

from __future__ import annotations

import sys
import time
from typing import Any

from adapters.common.rpc import (
    TTSImpl, UnavailableError, find_weight, free_cuda, lang_name, set_seed, weight_dir,
)

FEMALE = "Serena"
MALE = "Aiden"
TOKENS_PER_S = 12.5


def _max_new_tokens(text: str, margin: float, target: float | None) -> int:
    est_s = max(2.0, len(text) * 0.09)  # ~11 chars/s for Russian dialogue
    if target:
        est_s = max(est_s, float(target) * 1.2)
    return int(est_s * TOKENS_PER_S * max(1.0, margin) * 1.5) + 32


class _QwenBase(TTSImpl):
    def _load_model(self, path: str):
        import torch
        from qwen_tts import Qwen3TTSModel

        dtype = getattr(torch, str(self.options.get("dtype", "bfloat16"))) if self.device.startswith("cuda") else torch.float32
        kwargs: dict[str, Any] = {"device_map": self.device, "dtype": dtype}
        if self.options.get("attn_implementation"):
            kwargs["attn_implementation"] = self.options["attn_implementation"]
        return Qwen3TTSModel.from_pretrained(path, **kwargs)


class Qwen3TTS(_QwenBase):
    def load(self) -> dict[str, Any]:
        base = self.options.get("base_model") or "Qwen3-TTS-12Hz-1.7B-Base"
        self.base_dir = find_weight(self.params, base)
        inst = self.options.get("instruct_model") or ""
        self.custom_dir = find_weight(self.params, inst, require=False) if inst else None
        self.current = ""
        self.model = None
        self._ensure("clone")
        return {"base": str(self.base_dir), "custom": str(self.custom_dir) if self.custom_dir else None}

    def _ensure(self, mode: str) -> None:
        if self.current == mode:
            return
        path = self.base_dir if mode == "clone" else self.custom_dir
        if path is None or not path.exists():
            raise UnavailableError(f"{self.name}: no {'Base' if mode == 'clone' else 'CustomVoice'} model available")
        self.model = None
        free_cuda()
        self.model = self._load_model(str(path))
        self.current = mode

    def _mode(self, item: dict[str, Any]) -> str:
        m = (item.get("knobs") or {}).get("qwen_mode") or self.options.get("mode", "auto")
        if m == "auto":
            m = "clone" if item.get("timbre_ref") else "custom"
        if m == "custom" and self.custom_dir is None:
            m = "clone"
        return m

    def batch_synthesize(self, items: list[dict[str, Any]], **_: Any) -> list[dict[str, Any]]:
        order = {it["id"]: i for i, it in enumerate(items)}
        groups: dict[str, list[dict[str, Any]]] = {}
        for it in items:
            groups.setdefault(self._mode(it), []).append(it)
        out: list[dict[str, Any]] = []
        for mode in sorted(groups, key=lambda m: m != self.current):
            self._ensure(mode)
            for it in groups[mode]:
                try:
                    set_seed(it.get("seed"))
                    t0 = time.perf_counter()
                    wav, sr, meta = self._gen(it, mode)
                    out.append({"id": it["id"], **self._finish(it, wav, sr, t0), "mode": mode, **meta})
                except Exception as e:
                    sys.stderr.write(f"[{self.name}] {it.get('id')}: {e}\n")
                    out.append({"id": it["id"], "error": f"{type(e).__name__}: {e}"})
        out.sort(key=lambda r: order.get(r["id"], 0))
        return out

    def generate(self, item: dict[str, Any]):
        mode = self._mode(item)
        self._ensure(mode)
        wav, sr, _ = self._gen(item, mode)
        return wav, sr

    def _gen(self, item: dict[str, Any], mode: str):
        text = item["text"]
        lang = lang_name(item.get("lang") or "ru")
        mnt = _max_new_tokens(text, float(self.options.get("max_new_tokens_margin", 1.3)), item.get("target_duration"))
        if mode == "clone":
            ref = item.get("timbre_ref")
            if not ref:
                raise ValueError("clone mode needs timbre_ref")
            ref_text = item.get("timbre_ref_text") or None
            xvec = bool(self.options.get("x_vector_only_mode", True)) or not ref_text
            wavs, sr = self.model.generate_voice_clone(
                text=text, language=lang, ref_audio=ref, ref_text=None if xvec else ref_text,
                x_vector_only_mode=xvec, max_new_tokens=mnt,
            )
            return wavs[0], sr, {"x_vector_only_mode": xvec, "instruct_ignored": bool(item.get("instruct"))}
        speaker = self.options.get("custom_speaker", "auto")
        if speaker == "auto":
            speaker = MALE if (item.get("knobs") or {}).get("gender") == "m" else FEMALE
        wavs, sr = self.model.generate_custom_voice(text=text, language=lang, speaker=speaker, instruct=item.get("instruct") or None, max_new_tokens=mnt)
        return wavs[0], sr, {"speaker": speaker}


class Qwen3TTSVoiceDesign(_QwenBase):
    """VoiceDesign: a voice from a text description (route-B performer design)."""

    def load(self) -> dict[str, Any]:
        d = weight_dir(self.params, 0)
        self.model = self._load_model(str(d))
        return {"model_dir": str(d)}

    def design_generate(self, item: dict[str, Any]):
        desc = item.get("description") or item.get("instruct") or ""
        if not desc:
            raise ValueError("voice design needs a description")
        mnt = _max_new_tokens(item["text"], 1.3, item.get("target_duration"))
        wavs, sr = self.model.generate_voice_design(text=item["text"], language=lang_name(item.get("lang") or "ru"), instruct=desc, max_new_tokens=mnt)
        return wavs[0], sr

    def generate(self, item: dict[str, Any]):
        return self.design_generate(item)

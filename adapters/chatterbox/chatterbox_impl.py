"""Chatterbox Multilingual (V3 T3 checkpoint when present, else V2) from the local snapshot.

Emotion via ``exaggeration`` (the style renderer maps arousal into
``exaggeration_range``); ``cfg_weight=0`` for cross-lingual references is the
vendor's documented accent mitigation, applied by the renderer
(``knobs.cfg_weight``). Seeds through ``torch.manual_seed``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from adapters.common.rpc import TTSImpl, weight_dir

T3_FILES = ["t3_mtl23ls_v3.safetensors", "t3_mtl23ls_v2.safetensors"]


class ChatterboxML(TTSImpl):
    def load(self) -> dict[str, Any]:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        d = weight_dir(self.params, 0)
        t3 = self.options.get("t3_model") or next((f for f in T3_FILES if (Path(d) / f).exists()), None)
        try:
            self.model = ChatterboxMultilingualTTS.from_local(str(d), self.device, t3_model=t3) if t3 else ChatterboxMultilingualTTS.from_local(str(d), self.device)
        except TypeError:  # package versions without the t3_model argument (V2 only)
            self.model = ChatterboxMultilingualTTS.from_local(str(d), self.device)
            t3 = "t3_mtl23ls_v2.safetensors"
        self.sr = int(self.model.sr)
        return {"model_dir": str(d), "t3": t3, "sr": self.sr}

    def generate(self, item: dict[str, Any]):
        knobs = item.get("knobs") or {}
        lo, hi = self.options.get("exaggeration_range", [0.3, 1.2])
        exag = float(knobs.get("exaggeration", 0.5))
        exag = min(float(hi), max(float(lo), exag))
        wav = self.model.generate(
            (item.get("text_prefix") or "") + item["text"],
            language_id=(item.get("lang") or "ru").split("-")[0],
            audio_prompt_path=item.get("timbre_ref"),
            exaggeration=exag,
            cfg_weight=float(knobs.get("cfg_weight", self.options.get("cfg_weight", 0.5))),
            temperature=float(knobs.get("temperature", self.options.get("temperature", 0.8))),
        )
        return wav, self.sr

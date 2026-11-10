"""Chatterbox Multilingual (V3 T3 checkpoint when present, else V2) from the local snapshot.

Emotion via ``exaggeration`` (the style renderer maps arousal into
``exaggeration_range``); ``cfg_weight=0`` for cross-lingual references is the
vendor's documented accent mitigation, applied by the renderer
(``knobs.cfg_weight``). Seeds through ``torch.manual_seed``. Russian stress
marks come with the text (``stress_format: acute``), the vendor's own stresser
stays out of it.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from adapters.common.pkg_resources_compat import ensure_pkg_resources
from adapters.common.rpc import TTSImpl, weight_dir

T3_FILES = ["t3_mtl23ls_v3.safetensors", "t3_mtl23ls_v2.safetensors"]


def require_watermarker() -> None:
    """Raise the import error ``perth`` swallowed, instead of its ``'NoneType' object is not callable`` at model build.

    ``perth/__init__.py`` turns any ImportError of its watermarker into ``PerthImplicitWatermarker = None``, and
    ``ChatterboxMultilingualTTS.__init__`` calls that name after all the weights have been loaded.
    """
    import perth

    if perth.PerthImplicitWatermarker is None:
        from perth.perth_net.perth_net_implicit.perth_watermarker import PerthImplicitWatermarker  # noqa: F401


class ChatterboxML(TTSImpl):
    probe_imports = ("chatterbox.models.tokenizers.tokenizer", "chatterbox.mtl_tts")  # what load() imports (ModelImpl.probe)

    @classmethod
    def probe_setup(cls, params: dict[str, Any]) -> None:
        ensure_pkg_resources()
        require_watermarker()  # round 14: perth hides this ImportError; the real load met it after 249 s of weights

    def load(self) -> dict[str, Any]:
        ensure_pkg_resources()  # resemble-perth 1.0.1 imports pkg_resources, which uv envs / setuptools >= 82 lack
        require_watermarker()
        from chatterbox.models.tokenizers import tokenizer as vendor_tokenizer
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        vendor_tokenizer.add_russian_stress = lambda text: text  # without russian_text_stresser it warns on every line; with it, marks twice

        d = weight_dir(self.params, 0)
        t3 = self.options.get("t3_model") or next((f for f in T3_FILES if (Path(d) / f).exists()), None)
        if t3 and "t3_model" in inspect.signature(ChatterboxMultilingualTTS.from_local).parameters:
            self.model = ChatterboxMultilingualTTS.from_local(str(d), self.device, t3_model=t3)
        else:  # package versions without the t3_model argument (V2 only)
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

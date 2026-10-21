"""ruaccent: Russian stress placement and yo-fication (ONNX on CPU).

ruaccent downloads its dictionaries and networks on first ``load``; ``models
pull ruaccent`` does that once into ``<models_dir>/ruaccent`` (``workdir``) so
runs are offline. Output formats: ``plus`` keeps ruaccent's ``+`` before the
stressed vowel (F5-TTS_RUSSIAN), ``acute`` converts to U+0301 after the vowel,
``plain`` removes the marks (yo-fication is kept when ``yo``).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from adapters.common.rpc import ModelImpl, WeightsMissingError, weight_dir

# config name -> ruaccent omograph model size
MODEL_SIZES = {"turbo": "turbo3.1", "turbo3.1": "turbo3.1", "turbo3": "turbo3", "turbo2": "turbo2", "tiny": "tiny2.1", "big": "big_poetry"}
_PLUS_VOWEL = re.compile(r"\+([аеёиоуыэюяАЕЁИОУЫЭЮЯ])")


def convert(text: str, fmt: str = "plain", yo: bool = True) -> str:
    out = text
    if not yo:
        out = out.replace("ё", "е").replace("Ё", "Е")
    if fmt == "plus":
        return out
    if fmt == "acute":
        return _PLUS_VOWEL.sub(lambda m: m.group(1) + "́", out).replace("+", "")
    return out.replace("+", "")


def _load(workdir: Path, size: str, use_dictionary: bool):
    from ruaccent import RUAccent

    acc = RUAccent()
    acc.load(omograph_model_size=size, use_dictionary=use_dictionary, device="CPU", workdir=str(workdir))
    return acc


class RuAccentStress(ModelImpl):
    selftest_op = "stress"
    selftest_kwargs = {"fmt": "plus"}

    def _size(self) -> str:
        return MODEL_SIZES.get(str(self.options.get("model", "turbo")), str(self.options.get("model", "turbo3.1")))

    def load(self) -> dict[str, Any]:
        workdir = weight_dir(self.params, 0)
        size = self._size()
        if not (workdir / "dictionary").exists() or not (workdir / "nn" / "nn_omograph" / size).exists():
            raise WeightsMissingError(self.name, f"ruaccent models not in {workdir}")
        self.device = "cpu"
        self.acc = _load(workdir, size, bool(self.options.get("use_dictionary", True)))
        return {"workdir": str(workdir), "omograph_model": size}

    def op_stress(self, item: dict[str, Any], fmt: str = "plain", yo: bool = True, **_: Any) -> dict[str, Any]:
        text = item.get("text") or ""
        if not text.strip():
            return {"text": text}
        stressed = self.acc.process_all(text)
        return {"text": convert(stressed, fmt, yo), "stressed": stressed}

    def selftest_items(self, tmp: Path) -> list[dict[str, Any]]:
        return [{"id": "selftest", "text": "На двери висит замок, а вдали виден старый замок."}]

    @classmethod
    def pull(cls, params: dict[str, Any]) -> dict[str, Any]:
        workdir = Path(params["local_dir"])
        workdir.mkdir(parents=True, exist_ok=True)
        size = MODEL_SIZES.get(str((params.get("options") or {}).get("model", "turbo")), "turbo3.1")
        _load(workdir, size, True)
        return {"ok": True, "detail": f"ruaccent models in {workdir} ({size})"}

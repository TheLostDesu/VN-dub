"""ruaccent: Russian stress placement and yo-fication (ONNX on CPU).

ruaccent downloads its dictionaries and networks on first ``load``; ``models
pull ruaccent`` does that once into ``<models_dir>/ruaccent`` (``workdir``) so
runs are offline. Output formats: ``plus`` keeps ruaccent's ``+`` before the
stressed vowel (F5-TTS_RUSSIAN), ``acute`` converts to U+0301 after the vowel,
``plain`` removes the marks (yo-fication is kept when ``yo``).

ruaccent keeps one part of its data, the koziev morphology files (rulemma, rupostagger: ~190 MB),
inside its own package folder instead of ``workdir``, and downloads them on ``load`` when that
folder is missing. An environment recreated after ``models pull`` had lost them, and the next run
reached for the hub under ``HF_HUB_OFFLINE`` ("OfflineModeIsEnabled: Cannot reach
.../ruaccent/accentuator") although the weights folder was marked complete. So the pull keeps a
copy in ``<workdir>/koziev`` and :func:`sync_koziev` restores the package copy from it on load
(and saves one from an environment that already has it).
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path
from typing import Any

from adapters.common.rpc import ModelImpl, UnavailableError, WeightsMissingError, weight_dir

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


#: the koziev files ruaccent's rule engine opens (data and the modules it imports); all present = a complete copy
KOZIEV_FILES = (
    "rulemma/rulemma.dat",
    "rulemma/rulemma.py",
    "rupostagger/rupostagger.model",
    "rupostagger/rupostagger.py",
    "rupostagger/database/ruword2tags.db",
)


def _module_koziev() -> Path:
    """Where ruaccent looks for the koziev data: ``<site-packages>/ruaccent/koziev``."""
    import ruaccent

    return Path(ruaccent.__file__).resolve().parent / "koziev"


def _koziev_complete(d: Path) -> bool:
    return all((d / f).is_file() for f in KOZIEV_FILES)


def sync_koziev(workdir: Path, module: Path | None = None, name: str = "ruaccent") -> str:
    """Make sure ruaccent's package copy of the koziev data is complete, keeping a copy in ``<workdir>/koziev``.

    Returns where the data came from: ``env`` (both copies there), ``restored`` (package copy rebuilt from the
    workdir), ``saved`` (workdir copy made from the package). Raises :class:`WeightsMissingError` when neither
    copy is complete: the hub is out of reach at run time, so only ``models pull`` can fetch it.
    """
    module = module if module is not None else _module_koziev()
    saved = workdir / "koziev"
    have_module, have_saved = _koziev_complete(module), _koziev_complete(saved)
    if have_module and have_saved:
        return "env"
    if have_saved:
        try:
            shutil.copytree(saved, module, dirs_exist_ok=True)
        except OSError as e:
            raise UnavailableError(f"ruaccent: cannot restore the koziev data into {module}: {e}", hint="make that environment folder writable, or recreate the environment and run `vndub models pull ruaccent` again") from e
        return "restored"
    if have_module:
        try:
            shutil.copytree(module, saved, dirs_exist_ok=True)
        except OSError as e:  # the copy is a safety net for a later env rebuild: its failure must not stop this run
            sys.stderr.write(f"ruaccent: could not keep a copy of the koziev data in {saved}: {e}\n")
        return "saved"
    raise WeightsMissingError(name, f"ruaccent's koziev data is neither in the environment ({module}) nor in {saved}")


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
        koziev = sync_koziev(workdir, name=self.name)
        self.device = "cpu"
        self.acc = _load(workdir, size, bool(self.options.get("use_dictionary", True)))
        return {"workdir": str(workdir), "omograph_model": size, "koziev": koziev}

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
        _load(workdir, size, True)  # online: the dictionaries and networks go to workdir, the koziev data to the package
        koziev = sync_koziev(workdir, name=str(params.get("name") or "ruaccent"))
        return {"ok": True, "detail": f"ruaccent models in {workdir} ({size}, koziev data {koziev})"}

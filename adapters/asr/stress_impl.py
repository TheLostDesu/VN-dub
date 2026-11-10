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
_VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
_WORD = re.compile(r"(?:\+?[а-яёА-ЯЁ])+")

#: e-spellings that are also real words next to their ё-spelling but are missing from ruaccent's own list of such
#: homographs (``yo_homographs.json.gz``, 637 words): the ё is never added to these in the ``safe`` yo mode
EXTRA_YO_HOMOGRAPHS = frozenset(
    "совершенный совершенная совершенное совершенные совершенного совершенной совершенному совершенным совершенных "
    "совершенном совершенную совершенными "
    "полет полете".split()
)


def convert(text: str, fmt: str = "plain", yo: bool = True) -> str:
    out = text
    if not yo:
        out = out.replace("ё", "е").replace("Ё", "Е")
    if fmt == "plus":
        return out
    if fmt == "acute":
        return _PLUS_VOWEL.sub(lambda m: m.group(1) + "́", out).replace("+", "")
    return out.replace("+", "")


def fill_unmarked(text: str, accents: dict[str, str], put_accent: Any = None, skip: Any = ()) -> str:
    """Mark the multi-vowel words ruaccent left without a stress.

    ruaccent asks a small network (stress usage predictor) which words of a sentence take a stress at all, and
    that network answers "no" for plenty of ordinary words ("Даже", "тебя", "Хотя", "Таким образом", a verb after
    "..."), which then reach the TTS bare. A word of two or more vowels always has a stress, so such a word gets
    it from ruaccent's dictionary, or from its accent network (``put_accent``) when the dictionary lacks the word.
    Words with ё, with a mark already and one-vowel words are left alone; ``skip`` holds words not to touch.
    """

    def word(m: re.Match[str]) -> str:
        w = m.group(0)
        low = w.lower()
        if "+" in w or "ё" in low or sum(c in _VOWELS for c in low) < 2 or low in skip:
            return w
        known = accents.get(low)
        if known and "+" in known and known.replace("+", "") == low:
            out, i = [], 0
            for ch in known:
                if ch == "+":
                    out.append("+")
                else:
                    out.append(w[i])
                    i += 1
            return "".join(out)
        if put_accent is not None:
            try:
                got = put_accent(w)
            except Exception:
                return w
            if isinstance(got, str) and "+" in got and got.replace("+", "") == w:
                return got
        return w

    return _WORD.sub(word, text)


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

    probe_imports = ("ruaccent",)  # what load() imports (ModelImpl.probe)

    def load(self) -> dict[str, Any]:
        workdir = weight_dir(self.params, 0)
        size = self._size()
        if not (workdir / "dictionary").exists() or not (workdir / "nn" / "nn_omograph" / size).exists():
            raise WeightsMissingError(self.name, f"ruaccent models not in {workdir}")
        koziev = sync_koziev(workdir, name=self.name)
        self.device = "cpu"
        self.acc = _load(workdir, size, bool(self.options.get("use_dictionary", True)))
        return {"workdir": str(workdir), "omograph_model": size, "koziev": koziev}

    def _safe_yo(self):
        """ruaccent with the ё of the unambiguous words only (``yo: safe``).

        ruaccent adds ё in two ways: ``yo_words`` (85k forms that exist only with ё: "темном" -> "тёмном", "насчет",
        "найдешь") and a network that picks ё for ``yo_homographs`` (637 e/ё pairs: "небу" -> "нёбу", "узнаем" ->
        "узнаём", "все" -> "всё"), which guesses wrong on short chat lines. Safe mode turns the network off and
        takes from ``yo_words`` every word that is not also a homograph (250 of its keys are).
        """
        if not hasattr(self, "_safe_words"):
            homographs = set(self.acc.yo_homographs) | EXTRA_YO_HOMOGRAPHS
            self._full_words = self.acc.yo_words
            self._safe_words = {k: v for k, v in self.acc.yo_words.items() if k not in homographs}
        return self._safe_words

    def op_stress(self, item: dict[str, Any], fmt: str = "plain", yo: Any = True, **_: Any) -> dict[str, Any]:
        text = item.get("text") or ""
        if not text.strip():
            return {"text": text}
        model = self.acc.yo_homograph_model
        saved = (self.acc.yo_words, model.predict_yo_homographs)
        if yo == "safe":
            self.acc.yo_words = self._safe_yo()
            model.predict_yo_homographs = lambda sentence: []
        try:
            stressed = self.acc.process_all(text)
        finally:
            self.acc.yo_words, model.predict_yo_homographs = saved
        stressed = fill_unmarked(stressed, self.acc.accents, self.acc.accent_model.put_accent, self.acc.omographs)
        return {"text": convert(stressed, fmt, bool(yo)), "stressed": stressed}

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

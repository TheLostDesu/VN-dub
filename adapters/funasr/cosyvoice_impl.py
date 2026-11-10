"""Fun-CosyVoice3-0.5B-2512 from the CosyVoice repo checkout (``.venvs/funasr/src/CosyVoice``).

* no instruction -> ``inference_cross_lingual`` with the text prefixed by
  ``You are a helpful assistant.<|endofprompt|>`` (CosyVoice3 prompt format; the
  prompt wav gives the timbre, its language does not have to match);
* instruction    -> ``inference_instruct2(text, "You are a helpful assistant. <instr><|endofprompt|>", prompt_wav)``.

Fine-grained inline tags (``[breath]`` ...) pass through in the text.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from adapters.common.pkg_resources_compat import ensure_pkg_resources
from adapters.common.rpc import TTSImpl, UnavailableError, env_src_dir, weight_dir
from adapters.common.yamlcompat import hyperpyyaml_smoke, ruamel_old_api_compat

SYSTEM = "You are a helpful assistant."
EOP = "<|endofprompt|>"


def normalize_instruct(instr: str) -> str:
    s = instr.strip()
    if EOP not in s:
        s = s + EOP
    if not s.startswith(SYSTEM):
        s = f"{SYSTEM} {s}"
    return s


EXAMPLE_CONFIG = ("examples", "libritts", "cosyvoice3", "conf", "cosyvoice3.yaml")
_TARGET = re.compile(r"!(?:new|name):([A-Za-z_][\w.]*)")


def _repo_on_path(options: dict[str, Any]) -> Path:
    """The CosyVoice checkout (with its Matcha-TTS submodule) on sys.path, as a load needs it."""
    repo = Path(options["repo_dir"]) if options.get("repo_dir") else env_src_dir("funasr", "CosyVoice")
    if not (repo / "cosyvoice").exists():
        raise UnavailableError(f"CosyVoice checkout not found at {repo}", "rebuild the environment 'funasr' (it installs the +cosyvoice extra): Diagnostics or Models -> Environments; command line: scripts/make_envs.ps1 -Env funasr")
    for p in (repo, repo / "third_party" / "Matcha-TTS"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return repo


class CosyVoice3TTS(TTSImpl):
    @classmethod
    def probe_setup(cls, params: dict[str, Any]) -> None:
        _repo_on_path(params.get("options") or {})
        ruamel_old_api_compat()
        ensure_pkg_resources()

    @classmethod
    def probe_checks(cls, params: dict[str, Any]) -> dict[str, Any] | None:
        """Locate every ``!new:`` / ``!name:`` target of the model config, the way HyperPyYAML does at load.

        Importing ``cosyvoice.cli.cosyvoice`` alone passed in round 13's broken env: the failing import
        (``cosyvoice.flow.flow_matching`` -> matcha -> lightning -> pkg_resources) is reached only through the config.
        The pulled model's own yaml is used when it is there, else the checkout's example config."""
        import pydoc

        repo = _repo_on_path(params.get("options") or {})
        smoke = hyperpyyaml_smoke()
        cfg = None
        try:
            cfg = next(iter(sorted(weight_dir(params, 0).glob("cosyvoice*.yaml"))), None)
        except Exception:  # no weights in a probe: fine, the example config names the same classes
            cfg = None
        cfg = cfg or repo.joinpath(*EXAMPLE_CONFIG)
        targets = sorted(set(_TARGET.findall(cfg.read_text(encoding="utf-8"))))
        for t in targets:
            if pydoc.locate(t) is None:  # an import error inside raises pydoc.ErrorDuringImport with the real cause
                raise ImportError(f"cannot locate {t} named by {cfg.name}")
        return {**smoke, "config": str(cfg), "targets": len(targets)}

    def load(self) -> dict[str, Any]:
        _repo_on_path(self.options)
        d = weight_dir(self.params, 0)
        ruamel_old_api_compat()  # the model yaml goes through HyperPyYAML's old ruamel loader
        ensure_pkg_resources()  # lightning 2.2 (via Matcha-TTS) imports pkg_resources, which uv envs / setuptools >= 82 lack
        try:
            from cosyvoice.cli.cosyvoice import AutoModel

            self.model = AutoModel(model_dir=str(d))
        except ImportError:
            from cosyvoice.cli.cosyvoice import CosyVoice3  # type: ignore

            self.model = CosyVoice3(str(d))
        self.sr = int(self.model.sample_rate)
        return {"model_dir": str(d), "sr": self.sr}

    def generate(self, item: dict[str, Any]):
        import torch

        ref = item.get("timbre_ref") or item.get("emotion_ref")
        if not ref:
            raise ValueError("cosyvoice3 needs a prompt wav (timbre_ref)")
        text = (item.get("text_prefix") or "") + item["text"]
        speed = float((item.get("knobs") or {}).get("speed", self.options.get("speed", 1.0)))
        instr = item.get("instruct")
        if instr:
            gen = self.model.inference_instruct2(text, normalize_instruct(instr), ref, stream=False, speed=speed)
        else:
            gen = self.model.inference_cross_lingual(f"{SYSTEM}{EOP}{text}", ref, stream=False, speed=speed)
        chunks = [j["tts_speech"] for j in gen]
        if not chunks:
            raise RuntimeError("cosyvoice returned no audio")
        return torch.cat(chunks, dim=-1), self.sr

"""Fun-CosyVoice3-0.5B-2512 from the CosyVoice repo checkout (``.venvs/funasr/src/CosyVoice``).

* no instruction -> ``inference_cross_lingual`` with the text prefixed by
  ``You are a helpful assistant.<|endofprompt|>`` (CosyVoice3 prompt format; the
  prompt wav gives the timbre, its language does not have to match);
* instruction    -> ``inference_instruct2(text, "You are a helpful assistant. <instr><|endofprompt|>", prompt_wav)``.

Fine-grained inline tags (``[breath]`` ...) pass through in the text.
"""

from __future__ import annotations

import sys
from typing import Any

from adapters.common.rpc import TTSImpl, UnavailableError, env_src_dir, weight_dir

SYSTEM = "You are a helpful assistant."
EOP = "<|endofprompt|>"


def normalize_instruct(instr: str) -> str:
    s = instr.strip()
    if EOP not in s:
        s = s + EOP
    if not s.startswith(SYSTEM):
        s = f"{SYSTEM} {s}"
    return s


class CosyVoice3TTS(TTSImpl):
    def load(self) -> dict[str, Any]:
        repo = env_src_dir("funasr", "CosyVoice")
        if self.options.get("repo_dir"):
            from pathlib import Path

            repo = Path(self.options["repo_dir"])
        if not (repo / "cosyvoice").exists():
            raise UnavailableError(f"CosyVoice checkout not found at {repo}", "scripts/make_envs.ps1 -Env funasr (installs the +cosyvoice extra)")
        for p in (repo, repo / "third_party" / "Matcha-TTS"):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        d = weight_dir(self.params, 0)
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

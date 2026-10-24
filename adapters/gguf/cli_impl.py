"""Native GGUF TTS runners driven as subprocesses (UNVERIFIED command lines).

Fish S2 Pro (audio.cpp), Zonos2 (zonos2.cpp) and MOSS-TTS (llama.cpp path) ship
Windows CLIs whose flags were not verifiable on the build machine. Each
wrapper builds its command from ``options.args`` -- a list of strings with
placeholders -- so the exact flags can be fixed in ``pipeline.yaml`` without
code changes:

    {binary} {model} {model_dir} {text} {text_file} {ref} {ref_text} {out}
    {seed} {duration} {tokens} {emotion} {valence} {arousal} {strength}

The binary comes from ``tools.yaml`` (injected as ``options.binary`` by the
registry). The GGUF file is the first match of ``options.model_glob`` in the
weight dir. A missing binary/model is a clear "unavailable"/"weights missing"
error; a non-zero exit returns the tail of the tool's stderr.
"""

from __future__ import annotations

import fnmatch
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from adapters.common.rpc import TTSImpl, UnavailableError, WeightsMissingError, load_audio, weight_dir


class _CLI(TTSImpl):
    tool = ""
    hint = ""
    default_args: list[str] = []
    default_glob = "*.gguf"

    def load(self) -> dict[str, Any]:
        binary = self.options.get("binary") or ""
        if not binary or not Path(binary).exists():
            raise UnavailableError(f"{self.name}: CLI '{self.tool}' not configured", f"set tools.yaml {self.options.get('tool', self.tool)} ({self.hint})")
        d = weight_dir(self.params, 0)
        pat = self.options.get("model_glob", self.default_glob)
        models = sorted(p for p in d.rglob("*.gguf") if fnmatch.fnmatch(p.name, pat))
        if not models:
            raise WeightsMissingError(self.name, f"no GGUF matching {pat} in {d}")
        self.binary, self.model_dir, self.model_file = Path(binary), d, models[0]
        self.device = "cuda:0" if self.options.get("gpu", True) else "cpu"
        return {"binary": str(self.binary), "model": str(self.model_file)}

    def _values(self, item: dict[str, Any], text_file: Path) -> dict[str, str]:
        knobs = item.get("knobs") or {}
        emo = knobs.get("emotion")
        return {
            "binary": str(self.binary), "model": str(self.model_file), "model_dir": str(self.model_dir),
            "text": (item.get("text_prefix") or "") + item["text"], "text_file": str(text_file),
            "ref": str(item.get("timbre_ref") or ""), "ref_text": str(item.get("timbre_ref_text") or ""),
            "out": str(item["out"]), "seed": str(item.get("seed") or 0),
            "duration": f"{float(item.get('target_duration') or 0):.2f}",
            "tokens": str(int(float(item.get("target_duration") or 0) * float(self.options.get("tokens_per_second", 12.5)))),
            "emotion": json.dumps(emo) if isinstance(emo, (list, dict)) else str(emo or ""),
            "valence": str(knobs.get("valence", "")), "arousal": str(knobs.get("arousal", "")),
            "strength": str(knobs.get("emotion_strength", "")),
        }

    def command(self, values: dict[str, str]) -> list[str]:
        tmpl = self.options.get("args") or self.default_args
        cmd = [str(self.binary)]
        for a in tmpl:
            s = str(a).format(**values)
            if s.strip():
                cmd.append(s)
        return cmd

    def generate(self, item: dict[str, Any]):
        Path(item["out"]).parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as td:
            tf = Path(td) / "text.txt"
            tf.write_text((item.get("text_prefix") or "") + item["text"], encoding="utf-8")
            cmd = self.command(self._values(item, tf))
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=float(self.options.get("timeout_s", 600)))
        if r.returncode != 0 or not Path(item["out"]).exists():
            raise RuntimeError(f"{self.tool} failed ({r.returncode}): {(r.stderr or r.stdout)[-600:]}")
        return load_audio(item["out"], None)


class FishS2CLI(_CLI):
    tool = "audio.cpp"
    hint = "https://github.com/0xShug0/audio.cpp/releases"
    default_glob = "*Q4_K*.gguf"
    default_args = ["--model", "{model}", "--text", "{text}", "--reference", "{ref}", "--reference-text", "{ref_text}", "--seed", "{seed}", "--output", "{out}"]


class Zonos2CLI(_CLI):
    tool = "zonos2.cpp"
    hint = "https://github.com/Zyphra/zonos2.cpp/releases"
    default_glob = "*Q4_K*.gguf"
    default_args = ["--model", "{model}", "--text", "{text}", "--speaker", "{ref}", "--emotion", "{emotion}", "--valence", "{valence}", "--arousal", "{arousal}", "--seed", "{seed}", "--out", "{out}"]


class MossTTSCLI(_CLI):
    tool = "moss-tts"
    hint = "https://github.com/OpenMOSS/MOSS-TTS"
    default_glob = "*Q4_K_M*.gguf"
    default_args = ["--model", "{model}", "--text", "{text}", "--prompt-audio", "{ref}", "--prompt-text", "{ref_text}", "--target-tokens", "{tokens}", "--seed", "{seed}", "--output", "{out}"]

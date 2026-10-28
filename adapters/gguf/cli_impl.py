"""Native GGUF TTS runners driven as subprocesses (not run on the build machine: no GPU).

Fish S2 Pro and MOSS-TTS (audio.cpp) and Zonos2 (zonos2.cpp) ship Windows CLIs.
Each wrapper builds its command from ``options.args`` -- a list of strings with
placeholders -- so the exact flags can be fixed in ``pipeline.yaml`` without
code changes:

    {binary} {model} {model_dir} {text} {text_file} {ref} {ref_text} {out}
    {seed} {duration} {tokens} {emotion} {valence} {arousal} {strength}
    {gpu} {emotion_dir}   (Zonos2 only: ``--gpu`` / ``--cpu``, the emotion_directions folder)

A flag whose value renders empty is left out together with its value (``--reference-text ""``
would otherwise swallow the next flag). The command lines in the specs come from the runtimes'
own docs / sources (audio.cpp ``docs/models/*.md``; zonos2.cpp ``src/main.cpp`` and README).

The binary comes from ``tools.yaml`` (injected as ``options.binary`` by the
registry). The GGUF file is the first match of ``options.model_glob`` in the
weight dir. A missing binary/model is a clear "unavailable"/"weights missing"
error; a non-zero exit returns the tail of the tool's stderr.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from adapters.common.rpc import TTSImpl, UnavailableError, WeightsMissingError, load_audio, weight_dir

_FLAG = re.compile(r"^--?[A-Za-z]")


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
        tmpl = [str(a) for a in (self.options.get("args") or self.default_args)]
        cmd = [str(self.binary)]
        skip = False
        for i, a in enumerate(tmpl):
            if skip:
                skip = False
                continue
            nxt = tmpl[i + 1] if i + 1 < len(tmpl) else ""
            # a literal flag followed by a value that renders empty goes out WITH it: the CLIs take the next token as the
            # value (or exit 1 on it), so a lone ``--reference-text`` would swallow ``--seed``
            if _FLAG.match(a) and "{" not in a and "{" in nxt and not nxt.format(**values).strip():
                skip = True
                continue
            s = a.format(**values)
            if s.strip():
                cmd.append(s)
        return cmd

    def child_env(self) -> dict[str, str] | None:
        """Environment of the CLI process (``None``: inherit ours)."""
        return None

    def generate(self, item: dict[str, Any]):
        Path(item["out"]).parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as td:
            tf = Path(td) / "text.txt"
            tf.write_text((item.get("text_prefix") or "") + item["text"], encoding="utf-8")
            cmd = self.command(self._values(item, tf))
            # stdin=DEVNULL: the child must not inherit the RPC request pipe (a CLI that reads stdin would swallow requests)
            r = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=float(self.options.get("timeout_s", 600)),
                stdin=subprocess.DEVNULL, env=self.child_env(),
            )
        if r.returncode != 0 or not Path(item["out"]).exists():
            raise RuntimeError(f"{self.tool} failed ({r.returncode}): {(r.stderr or r.stdout)[-600:]}")
        return load_audio(item["out"], None)


class FishS2CLI(_CLI):
    tool = "audio.cpp"
    hint = "https://github.com/0xShug0/audio.cpp/releases"
    default_glob = "*Q4_K*.gguf"
    default_args = ["--model", "{model}", "--text", "{text}", "--reference", "{ref}", "--reference-text", "{ref_text}", "--seed", "{seed}", "--output", "{out}"]


# order of the ``knobs.emotion`` vector (``vndub.text.style_prompt._ZONOS_ORDER``; a test keeps the two equal)
ZONOS_EMOTION_ORDER = ("happy", "sad", "disgusted", "fearful", "surprised", "angry", "other", "neutral")
# the named directions zonos2.cpp ships (``emotion_directions/manifest.json``, next to ``valence`` / ``arousal`` axes);
# the other entries of the vector have no direction and are left out
ZONOS2_SLIDERS = ("happy", "sad", "angry", "surprised")


def _centered(x: Any) -> float:
    """A 0..1 knob (0.5 = neutral) as the signed axis of zonos2.cpp (0 = unchanged), clipped to [-1, 1]."""
    try:
        return max(-1.0, min(1.0, (float(x) - 0.5) * 2.0))
    except (TypeError, ValueError):
        return 0.0


class Zonos2CLI(_CLI):
    """``zonos2-cli <model.gguf> --tts "<text>" <out.wav> [--clone ref.wav] [--emotion happy=1] ... [--seed N] [--gpu]``.

    From zonos2.cpp ``src/main.cpp`` and README; the CLI exits 1 on a flag it does not know (the guessed
    ``--model --text --speaker --out`` of the first version). ``dac.gguf`` and ``spk-encoder.gguf`` are found next to
    the backbone GGUF (the weight dir holds all three). ``--clone`` encodes the reference in-process through ffmpeg
    (``$ZONOS2_FFMPEG``, next to the binary, or PATH); emotion control needs that speaker embedding and reads the
    ``emotion_directions`` folder that ships next to the binary (the CLI default is relative to the working dir).
    ``--speaking-rate`` is a bucket index in the CLI (the rate -> bucket map lives in its server), so the
    ``speaking_rate`` knob is not passed.
    """

    tool = "zonos2.cpp"
    hint = "https://github.com/Zyphra/zonos2.cpp/releases"
    default_glob = "*Q4_K*.gguf"
    default_args = [
        "{model}", "--tts", "{text}", "{out}", "--clone", "{ref}",
        "--emotion", "{emotion}", "--emotion-valence", "{valence}", "--emotion-arousal", "{arousal}", "--emotion-strength", "{strength}",
        "--emotion-dir", "{emotion_dir}", "--seed", "{seed}", "{gpu}",
    ]

    def _emotion_dir(self) -> str:
        d = self.options.get("emotion_dir")
        if d:
            return str(d)
        near = Path(self.binary).parent / "emotion_directions"
        return str(near) if near.is_dir() else ""

    def _values(self, item: dict[str, Any], text_file: Path) -> dict[str, str]:
        v = super()._values(item, text_file)
        knobs = item.get("knobs") or {}
        emo = knobs.get("emotion")
        if isinstance(emo, (list, tuple)):
            emo = dict(zip(ZONOS_EMOTION_ORDER, emo))
        sliders: dict[str, float] = {}
        if isinstance(emo, dict):
            for k in ZONOS2_SLIDERS:
                try:
                    w = float(emo.get(k) or 0.0)
                except (TypeError, ValueError):
                    w = 0.0
                if w > 0.0:
                    sliders[k] = w
        valence, arousal = _centered(knobs.get("valence", 0.5)), _centered(knobs.get("arousal", 0.5))
        v["emotion"] = ",".join(f"{k}={w:.3f}" for k, w in sliders.items())
        v["valence"] = f"{valence:.3f}" if abs(valence) >= 0.01 else ""
        v["arousal"] = f"{arousal:.3f}" if abs(arousal) >= 0.01 else ""
        if not (v["emotion"] or v["valence"] or v["arousal"]) or not v["ref"]:
            # neutral: nothing to ask for. Without a reference there is no speaker embedding, and the CLI refuses any
            # emotion control then ("requires --speaker or --clone"), so plain synthesis is all it can do
            v["emotion"] = v["valence"] = v["arousal"] = v["strength"] = ""
        else:
            s = knobs.get("emotion_strength")
            v["strength"] = "" if s in (None, "") else f"{float(s):.3f}"
        v["emotion_dir"] = self._emotion_dir()
        v["gpu"] = "--gpu" if self.options.get("gpu", True) else "--cpu"
        return v

    def child_env(self) -> dict[str, str] | None:
        # the installed ffmpeg is not necessarily on PATH: registry.inject_tools hands the configured one over as options.ffmpeg
        ff = self.options.get("ffmpeg")
        if not ff or os.environ.get("ZONOS2_FFMPEG"):
            return None
        return {**os.environ, "ZONOS2_FFMPEG": str(ff)}


class MossTTSCLI(_CLI):
    tool = "moss-tts"
    hint = "https://github.com/OpenMOSS/MOSS-TTS"
    default_glob = "*Q4_K_M*.gguf"
    default_args = ["--model", "{model}", "--text", "{text}", "--prompt-audio", "{ref}", "--prompt-text", "{ref_text}", "--target-tokens", "{tokens}", "--seed", "{seed}", "--output", "{out}"]

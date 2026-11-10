"""F5-TTS with the Misha24-10/F5-TTS_RUSSIAN checkpoint (F5TTS_v1_Base architecture).

* ``fix_duration`` = reference duration + ``target_duration`` (F5 counts the
  prompt in the total; capped at 30 s) -- the only v1 route with real length control.
* ``+`` stress marks (ruaccent ``plus`` format) pass through to the model.
* F5 needs the reference transcript (``timbre_ref_text``); without it F5 would
  run its own Whisper, which is not available offline, so the item fails clearly.
* The Vocos vocoder (``charactr/vocos-mel-24khz``) is a weight of the spec and is loaded from its local directory
  (``vocoder_local_path``); f5-tts would otherwise download it at load time, which the offline adapter env forbids.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

from adapters.common.rpc import TTSImpl, WeightsMissingError, find_weight, load_audio, weight_dir

MAX_TOTAL_S = 30.0
VOCODER_DIR = "vocos-mel-24khz"  # weight dir of charactr/vocos-mel-24khz (second weight of the f5_ru spec)


def _find(root: Path, pattern: str) -> Path | None:
    hits = sorted(p for p in root.rglob("*") if p.is_file() and fnmatch.fnmatch(p.relative_to(root).as_posix(), pattern))
    hits = [p for p in hits if "all-checkpoints" not in p.as_posix()]
    return hits[-1] if hits else None


class F5Russian(TTSImpl):
    selftest_text = "Прив+ет! Это кор+откая пров+ерка с+интеза р+ечи."

    probe_imports = ("f5_tts.api",)  # what load() imports (ModelImpl.probe)

    def load(self) -> dict[str, Any]:
        from f5_tts.api import F5TTS

        d = weight_dir(self.params, 0)
        ckpt = _find(d, self.options.get("ckpt_glob", "F5TTS_v1_Base_v4_winter/*.safetensors"))
        vocab = _find(d, self.options.get("vocab_glob", "F5TTS_v1_Base/vocab.txt"))
        if ckpt is None or vocab is None:
            raise WeightsMissingError(self.name, f"checkpoint/vocab not found under {d}")
        # f5-tts 1.1.9 reads <dir>/config.yaml + <dir>/pytorch_model.bin when vocoder_local_path is set; without it
        # load_vocoder calls hf_hub_download("charactr/vocos-mel-24khz"), which cannot work with HF_HUB_OFFLINE=1
        vocoder = find_weight(self.params, VOCODER_DIR)
        self.model = F5TTS(
            model=self.options.get("arch", "F5TTS_v1_Base"), ckpt_file=str(ckpt), vocab_file=str(vocab),
            vocoder_local_path=str(vocoder), device=self.device,
        )
        return {"ckpt": str(ckpt), "vocab": str(vocab), "vocoder": str(vocoder)}

    def generate(self, item: dict[str, Any]):
        ref = item.get("timbre_ref")
        ref_text = (item.get("timbre_ref_text") or "").strip()
        if not ref or not ref_text:
            raise ValueError("f5_ru needs timbre_ref and timbre_ref_text (offline: no auto-transcription)")
        fix = None
        if item.get("target_duration"):
            ref_y, ref_sr = load_audio(ref, None)
            fix = min(MAX_TOTAL_S, len(ref_y) / ref_sr + float(item["target_duration"]))
        wav, sr, _spec = self.model.infer(
            ref_file=ref, ref_text=ref_text, gen_text=item["text"],
            nfe_step=int(self.options.get("nfe_step", 32)), cfg_strength=float(self.options.get("cfg_strength", 2.0)),
            speed=float((item.get("knobs") or {}).get("speed", 1.0)), fix_duration=fix,
            seed=item.get("seed"), remove_silence=False, show_info=lambda *a, **k: None,
        )
        return wav, sr

    def selftest_items(self, tmp: Path) -> list[dict[str, Any]]:
        items = super().selftest_items(tmp)
        items[0]["timbre_ref_text"] = "Это тестовая фраза."
        return items

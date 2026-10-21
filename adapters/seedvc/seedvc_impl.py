"""Seed-VC v1 zero-shot timbre conversion (checkout ``.venvs/seedvc/src/seed-vc``).

``SeedVCWrapper.convert_voice`` is a generator even with ``stream_output=False``
(it *returns* the audio), so the value is taken from ``StopIteration``. The
v1 model converts timbre only, i.e. the equivalent of v2's
``convert_style=False``: intonation and timing of the source are kept.
``f0_condition=False`` -> 22.05 kHz output (the 44.1 kHz F0 model is for singing).

Checkpoints are fetched by the wrapper through huggingface_hub into the
checkout's ``checkpoints/`` and HF_HOME; ``models pull seedvc`` instantiates it
once online so later runs are offline.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

from adapters.common.rpc import ModelImpl, UnavailableError, env_src_dir, make_test_clip, save_audio, to_numpy, weight_dir


def repo_dir(options: dict[str, Any]) -> Path:
    d = Path(options.get("repo_dir") or env_src_dir("seedvc", "seed-vc"))
    if not (d / "seed_vc_wrapper.py").exists():
        raise UnavailableError(f"Seed-VC checkout not found at {d}", "scripts/make_envs.ps1 -Env seedvc")
    return d


def _wrapper(d: Path, device: str):
    import torch

    os.chdir(d)  # checkpoints/ and configs/ are resolved relative to the checkout
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    from seed_vc_wrapper import SeedVCWrapper

    return SeedVCWrapper(device=torch.device(device))


class SeedVC(ModelImpl):
    selftest_op = "convert"

    def load(self) -> dict[str, Any]:
        weight_dir(self.params, 0)  # marker of `models pull seedvc`
        d = repo_dir(self.options)
        self.vc = _wrapper(d, self.device)
        return {"repo": str(d)}

    def op_convert(self, item: dict[str, Any], **_: Any) -> dict[str, Any]:
        ref = item.get("target_ref")
        if not ref:
            raise ValueError("seedvc needs target_ref (the character's clean reference clip)")
        p = {**self.options, **(item.get("params") or {})}
        f0_cond = bool(p.get("f0_condition", False))
        t0 = time.perf_counter()
        gen = self.vc.convert_voice(
            source=item["src"], target=ref,
            diffusion_steps=int(p.get("diffusion_steps", 30)),
            length_adjust=float(p.get("length_adjust", 1.0)),
            inference_cfg_rate=float(p.get("inference_cfg_rate", 0.7)),
            f0_condition=f0_cond, auto_f0_adjust=bool(p.get("auto_f0_adjust", False)),
            pitch_shift=int(round(float(item.get("f0_transpose") or 0))) if f0_cond else 0,
            stream_output=False,
        )
        audio = None
        try:
            while True:
                next(gen)
        except StopIteration as stop:
            audio = stop.value
        if audio is None:
            raise RuntimeError("seed-vc returned no audio")
        sr = 44100 if f0_cond else 22050
        y = to_numpy(audio)
        save_audio(item["out"], y, sr)
        return {"path": item["out"], "sr": sr, "duration": round(len(y) / sr, 3), "elapsed_s": round(time.perf_counter() - t0, 3)}

    def selftest_items(self, tmp: Path) -> list[dict[str, Any]]:
        src = make_test_clip(tmp / "src.wav", seconds=2.0, sr=22050, f0=180)
        ref = make_test_clip(tmp / "ref.wav", seconds=3.0, sr=22050, f0=260)
        return [{"id": "selftest", "src": src, "target_ref": ref, "out": str(tmp / "out.wav"), "params": {"diffusion_steps": 5}}]

    @classmethod
    def pull(cls, params: dict[str, Any]) -> dict[str, Any]:
        d = repo_dir(params.get("options") or {})
        _wrapper(d, "cpu")
        return {"ok": True, "detail": f"Seed-VC checkpoints cached under {d / 'checkpoints'} and HF_HOME"}

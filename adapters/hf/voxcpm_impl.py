"""VoxCPM2 (``voxcpm`` package): controllable cloning, voice design, LoRA (route C).

* synthesize: ``generate(text="(style)" + text, reference_wav_path=timbre_ref)``;
  with ``options.ultimate`` and a reference transcript the "ultimate cloning"
  continuation mode is used (``prompt_wav_path`` + ``prompt_text``).
* design: ``generate(text="(description)" + text)`` without a reference.
* train / train_lora: writes the ``{"audio", "text"}`` manifest and a LoRA config
  from the repo template and runs the official ``scripts/train_voxcpm_finetune.py``
  from the VoxCPM checkout (``.venvs/hf/src/VoxCPM``) as a subprocess, after
  freeing the inference model's VRAM.
* ``options.lora``: path of trained LoRA weights to load (route C inference).

``torch.compile`` (``optimize``) is off by default: it needs Triton, which is not
generally available on Windows.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from adapters.common.rpc import TTSImpl, UnavailableError, env_src_dir, free_cuda, weight_dir


def _style_prefix(item: dict[str, Any]) -> str:
    p = (item.get("text_prefix") or "").strip()
    if not p and (item.get("instruct") or "").strip().startswith("("):
        p = item["instruct"].strip()
    return p


class VoxCPM2TTS(TTSImpl):
    def load(self) -> dict[str, Any]:
        from voxcpm import VoxCPM

        d = weight_dir(self.params, 0)
        kwargs: dict[str, Any] = {"load_denoiser": False, "optimize": bool(self.options.get("optimize", False))}
        if self.options.get("lora"):
            kwargs["lora_weights_path"] = str(self.options["lora"])
        try:
            self.model = VoxCPM.from_pretrained(str(d), local_files_only=True, device=self.device, **kwargs)
        except TypeError:
            self.model = VoxCPM.from_pretrained(str(d), **kwargs)
        self.sr = int(self.model.tts_model.sample_rate)
        return {"model_dir": str(d), "sr": self.sr, "lora": self.options.get("lora")}

    def _gen(self, text: str, **kw: Any):
        if getattr(self, "model", None) is None:  # freed by a training job
            self.load()
        args = {
            "text": text,
            "cfg_value": float(self.options.get("cfg_value", 2.0)),
            "inference_timesteps": int(self.options.get("inference_timesteps", 10)),
            "normalize": bool(self.options.get("normalize", True)),
        }
        args.update({k: v for k, v in kw.items() if v is not None})
        return self.model.generate(**args), self.sr

    def generate(self, item: dict[str, Any]):
        text = _style_prefix(item) + item["text"]
        ref = item.get("timbre_ref")
        kw: dict[str, Any] = {"reference_wav_path": ref, "seed": item.get("seed")}
        if self.options.get("ultimate") and ref and item.get("timbre_ref_text"):
            kw.update(prompt_wav_path=ref, prompt_text=item["timbre_ref_text"])
        return self._gen(text, **kw)

    def design_generate(self, item: dict[str, Any]):
        desc = (item.get("description") or item.get("instruct") or "").strip().strip("()")
        return self._gen(f"({desc}){item['text']}", seed=item.get("seed"))

    # -- route C -----------------------------------------------------------------
    def op_train_lora(self, job: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        job = job or {}
        repo = Path(self.options.get("repo_dir") or env_src_dir("hf", "VoxCPM"))
        script = repo / "scripts" / "train_voxcpm_finetune.py"
        template = repo / "conf" / "voxcpm_v2" / "voxcpm_finetune_lora.yaml"
        if not script.exists():
            raise UnavailableError(f"VoxCPM training script not found at {script}", "scripts/make_envs.ps1 -Env hf (the +voxcpm extra clones the repo)")
        out = Path(job["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        manifest = out / "train.jsonl"
        with open(manifest, "w", encoding="utf-8") as f:
            for p in job.get("pairs", []):
                f.write(json.dumps({"audio": str(p["wav"]), "text": p["text"]}, ensure_ascii=False) + "\n")
        import yaml

        cfg = yaml.safe_load(template.read_text(encoding="utf-8")) if template.exists() else {}
        cfg.update(
            pretrained_path=str(weight_dir(self.params, 0)),
            train_manifest=str(manifest),
            val_manifest=None,
            save_path=str(out / "checkpoints"),
            tensorboard=str(out / "tb"),
            batch_size=int(job.get("batch_size", 1)),
            grad_accum_steps=int(job.get("grad_accum_steps", 16)),
            num_workers=0,
            preprocessing_num_workers=1,
            num_iters=int(job.get("num_iters", 1000)),
            max_steps=int(job.get("num_iters", 1000)),
            learning_rate=float(job.get("learning_rate", 1e-4)),
        )
        cfg_path = out / "lora_config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
        self.model = None
        free_cuda()
        log = out / "train.log"
        with open(log, "w", encoding="utf-8") as lf:
            r = subprocess.run([sys.executable, str(script), "--config_path", str(cfg_path)], cwd=str(repo), stdout=lf, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            raise RuntimeError(f"VoxCPM LoRA training failed (exit {r.returncode}); see {log}")
        ckpts = sorted((out / "checkpoints").rglob("lora_weights*"), key=lambda p: p.stat().st_mtime)
        adapter = ckpts[-1].parent if ckpts else out / "checkpoints"
        return {"adapter": str(adapter), "log": str(log), "n_pairs": len(job.get("pairs", []))}

    op_train = op_train_lora

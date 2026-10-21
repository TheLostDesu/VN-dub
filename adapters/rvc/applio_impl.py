"""RVC v2 through Applio 3.7 (``.venvs/rvc/src/Applio`` or ``tools.yaml: applio_dir``).

* convert: ``core.run_infer_script`` (pitch = ``f0_transpose`` semitones,
  index_rate, protect, f0_method rmvpe, contentvec embedder); keeps the
  performer's F0 contour and timing.
* train: preprocess -> extract (rmvpe) -> train -> index in a child process
  (``applio_job.py``) with ``APPLIO_LOGS_DIR=<out_dir>/applio_logs``; the newest
  ``.pth`` and the ``.index`` are copied to ``<out_dir>/model.pth|model.index``.
  Applio 3.7's CLI has no overtraining detector, so ``overfit_patience`` is
  reported but not enforced; the epoch budget comes from the config.
* f0 (``rmvpe`` backend): Applio's ``RMVPE0Predictor`` at 100 frames/s.
* pull: Applio's prerequisites download (rmvpe, contentvec, HiFi-GAN pretrains).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from adapters.common.rpc import ModelImpl, UnavailableError, WeightsMissingError, env_src_dir, load_audio, make_test_clip, weight_dir

JOB_SCRIPT = Path(__file__).with_name("applio_job.py")


def applio_dir(options: dict[str, Any]) -> Path:
    d = Path(options.get("applio_dir") or env_src_dir("rvc", "Applio"))
    if not (d / "core.py").exists():
        raise UnavailableError(f"Applio checkout not found at {d}", "scripts/make_envs.ps1 -Env rvc, or set tools.yaml applio_dir")
    return d


def _prereqs_ok(d: Path) -> bool:
    return (d / "rvc" / "models" / "predictors" / "rmvpe.pt").exists() and (d / "rvc" / "models" / "embedders" / "contentvec" / "pytorch_model.bin").exists()


def run_job(d: Path, job: dict[str, Any], logs_dir: Path | None, log_file: Path, timeout: float | None = None) -> dict[str, Any]:
    env = os.environ.copy()
    if logs_dir is not None:
        env["APPLIO_LOGS_DIR"] = str(logs_dir)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(job, f)
        job_path = f.name
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as lf:
            r = subprocess.run([sys.executable, str(JOB_SCRIPT), job_path], cwd=str(d), env=env, stdout=subprocess.PIPE, stderr=lf, text=True, timeout=timeout)
        last = next((l for l in reversed(r.stdout.splitlines()) if l.strip().startswith("{")), "")
        res = json.loads(last) if last else {"ok": False, "messages": [r.stdout[-500:]]}
        res["returncode"] = r.returncode
        return res
    finally:
        os.unlink(job_path)


class ApplioRVC(ModelImpl):
    selftest_op = "convert"

    def load(self) -> dict[str, Any]:
        weight_dir(self.params, 0)  # marker of `models pull rvc`
        self.dir = applio_dir(self.options)
        if not _prereqs_ok(self.dir):
            raise WeightsMissingError(self.name, f"Applio prerequisites missing in {self.dir}")
        os.chdir(self.dir)  # Applio resolves model paths relative to its checkout
        if str(self.dir) not in sys.path:
            sys.path.insert(0, str(self.dir))
        import core  # noqa: F401  (imports torch, Applio modules)

        self.core = core
        return {"applio_dir": str(self.dir)}

    def op_convert(self, item: dict[str, Any], **_: Any) -> dict[str, Any]:
        model = item.get("model")
        if not model or not Path(model).exists():
            raise FileNotFoundError(f"no RVC model for this character ({model})")
        p = {**self.options, **(item.get("params") or {})}
        out = str(Path(item["out"]).with_suffix(".wav"))
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        self.core.run_infer_script(
            pitch=int(round(float(item.get("f0_transpose") or 0))),
            index_rate=float(p.get("index_rate", 0.5)),
            volume_envelope=float(p.get("volume_envelope", 1.0)),
            protect=float(p.get("protect", 0.33)),
            f0_method=str(p.get("f0_method", "rmvpe")),
            input_path=str(item["src"]),
            output_path=out,
            pth_path=str(model),
            index_path=str(item.get("index") or ""),
            split_audio=False,
            f0_autotune=False,
            f0_autotune_strength=1.0,
            proposed_pitch=False,
            proposed_pitch_threshold=155.0,
            export_format="WAV",
            embedder_model="contentvec",
        )
        if not Path(out).exists():
            raise RuntimeError("Applio produced no output")
        import soundfile as sf

        info = sf.info(out)
        return {"path": out, "sr": int(info.samplerate), "duration": round(info.frames / info.samplerate, 3), "elapsed_s": round(time.perf_counter() - t0, 3)}

    def op_train(self, job: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        job = dict(job or {})
        out = Path(job["out_dir"])
        name = str(job.get("model_name") or job.get("speaker_id") or out.parent.name or "voice")
        dataset = out / "dataset"
        dataset.mkdir(parents=True, exist_ok=True)
        for i, clip in enumerate(job.get("clips", [])):
            shutil.copy2(clip, dataset / f"{i:05d}{Path(clip).suffix or '.wav'}")
        logs = out / "applio_logs"
        log_file = out / "train.log"
        t0 = time.perf_counter()
        res = run_job(
            self.dir,
            {
                "mode": "train", "model_name": name, "dataset": str(dataset),
                "sample_rate": int(job.get("sample_rate", 40000)), "epochs": int(job.get("epochs", 300)),
                "batch_size": int(job.get("batch_size", 6)), "save_every": int(job.get("save_every", 10)),
                "f0_method": job.get("f0_method", "rmvpe"), "gpu": "0" if self.device.startswith("cuda") else "-",
            },
            logs,
            log_file,
        )
        if not res.get("ok"):
            raise RuntimeError(f"Applio training failed: {res.get('messages')} (see {log_file})")
        mdir = logs / name
        pths = sorted(mdir.glob(f"{name}*.pth"), key=lambda p: p.stat().st_mtime) or sorted(mdir.rglob("*.pth"), key=lambda p: p.stat().st_mtime)
        idx = sorted((p for p in mdir.rglob("*.index") if not p.name.startswith("trained")), key=lambda p: p.stat().st_mtime)
        if not pths:
            raise RuntimeError(f"no .pth produced in {mdir}")
        model = out / "model.pth"
        shutil.copy2(pths[-1], model)
        index = None
        if idx:
            index = out / "model.index"
            shutil.copy2(idx[-1], index)
        return {
            "model": str(model), "index": str(index) if index else None, "source_pth": str(pths[-1]),
            "epochs": int(job.get("epochs", 300)), "sample_rate": int(job.get("sample_rate", 40000)),
            "overfit_patience": job.get("overfit_patience"), "overfit_guard": "not available in Applio 3.7 CLI",
            "elapsed_s": round(time.perf_counter() - t0, 1), "log": str(log_file), "messages": res.get("messages"),
        }

    def selftest_items(self, tmp: Path) -> list[dict[str, Any]]:
        clip = make_test_clip(tmp / "in.wav", seconds=2.0, sr=40000)
        model = self.options.get("selftest_model") or ""
        return [{"id": "selftest", "src": clip, "out": str(tmp / "out.wav"), "model": model, "index": "", "f0_transpose": 0}]

    def selftest(self) -> dict[str, Any]:
        res = super().selftest()
        if not self.options.get("selftest_model"):
            # inference needs a trained per-character model; loading Applio is the verifiable part
            res["note"] = "loaded Applio; conversion needs options.selftest_model (a trained .pth)"
            res["ok"] = all("no RVC model" in str(e) for e in res.get("errors", [])) or not res.get("errors")
        return res

    @classmethod
    def pull(cls, params: dict[str, Any]) -> dict[str, Any]:
        d = applio_dir(params.get("options") or {})
        if _prereqs_ok(d):
            return {"ok": True, "detail": "prerequisites present"}
        res = run_job(d, {"mode": "prereq"}, None, Path(params["local_dir"]) / "pull.log")
        if not res.get("ok") or not _prereqs_ok(d):
            raise RuntimeError(f"Applio prerequisites download failed: {res}")
        return {"ok": True, "detail": res.get("messages")}


class RMVPEF0(ModelImpl):
    selftest_op = "f0"

    def load(self) -> dict[str, Any]:
        weight_dir(self.params, 0)
        d = applio_dir(self.options)
        ckpt = d / "rvc" / "models" / "predictors" / "rmvpe.pt"
        if not ckpt.exists():
            raise WeightsMissingError(self.name, str(ckpt))
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
        from rvc.lib.predictors.RMVPE import RMVPE0Predictor

        self.pred = RMVPE0Predictor(str(ckpt), device=self.device)
        return {"checkpoint": str(ckpt)}

    def op_f0(self, item: dict[str, Any], **_: Any) -> dict[str, Any]:
        import numpy as np

        y, _ = load_audio(item["wav"], 16000)
        f0 = np.asarray(self.pred.infer_from_audio(y, thred=float(self.options.get("threshold", 0.03))), dtype=np.float32)
        out = Path(item["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, f0)
        return {"path": str(out), "hop_s": 0.01, "n": int(f0.size)}

    @classmethod
    def pull(cls, params: dict[str, Any]) -> dict[str, Any]:
        return ApplioRVC.pull(params)

"""Run one Applio job in a fresh process: ``python applio_job.py <job.json>``.

Applio resolves its logs dir (``APPLIO_LOGS_DIR``) and relative model paths
at import time and from the working directory, so training and the
prerequisites download run here, with ``cwd`` = the Applio checkout and the env
var set by the caller, instead of inside the long-lived adapter process.

job: {"mode": "prereq"} or
     {"mode": "train", "model_name", "dataset", "sample_rate", "epochs", "batch_size",
      "save_every", "f0_method", "cpu_cores", "gpu", "vocoder", "index_algorithm"}
Prints one final JSON line: {"ok": bool, "messages": [...]}.
"""

from __future__ import annotations

import json
import os
import sys


def main() -> None:
    job = json.load(open(sys.argv[1], encoding="utf-8"))
    sys.path.insert(0, os.getcwd())
    import core  # Applio's core.py

    msgs: list[str] = []
    ok = True
    if job["mode"] == "prereq":
        msgs.append(core.run_prerequisites_script(pretraineds_hifigan=True, models=True, exe=False))
    elif job["mode"] == "train":
        name = job["model_name"]
        sr = int(job.get("sample_rate", 40000))
        cores = int(job.get("cpu_cores", max(1, (os.cpu_count() or 2) - 1)))
        gpu = str(job.get("gpu", "0"))
        m = core.run_preprocess_script(name, job["dataset"], sr, cores, "Automatic", False, 3.0, 0.3, "none")
        msgs.append(m)
        ok = "successfully" in m
        if ok:
            m = core.run_extract_script(name, job.get("f0_method", "rmvpe"), cores, gpu, sr, "contentvec", None, 2)
            msgs.append(m)
            ok = "successfully" in m
        if ok:
            m = core.run_train_script(
                name, int(job.get("save_every", 10)), True, True, int(job.get("epochs", 300)), sr,
                int(job.get("batch_size", 6)), gpu, True, False, job.get("index_algorithm", "Auto"),
                False, False, None, None, job.get("vocoder", "HiFi-GAN"), False, False,
            )
            msgs.append(m)
            ok = "successfully" in m
    else:
        ok = False
        msgs.append(f"unknown mode {job['mode']}")
    print(json.dumps({"ok": ok, "messages": msgs}))


if __name__ == "__main__":
    main()

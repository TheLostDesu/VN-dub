# Troubleshooting

**`backend env 'hf' not created`** — the backend's virtual environment is missing:
`powershell -ExecutionPolicy Bypass -File scripts/make_envs.ps1 -Env hf` (add `-Cpu` on a
machine without an NVIDIA GPU). Logs of the adapter process: `work/logs/adapter_<env>.log`.

**`weights missing: run vndub models pull <name>`** — download once:
`uv run vndub models pull <name>` (or `--profile default`). At run time adapters are
offline (`HF_HUB_OFFLINE=1`), so nothing is fetched implicitly.

**`needs ~X GB > budget` / `only Y GB free`** — the GPU registry refuses to load a model that
does not fit `gpu.vram_budget_gb - gpu.safety_margin_gb` or the free VRAM. Close other GPU
programs (browsers, games, other CUDA processes), or disable that backend
(`--set synth.backends.voxcpm2.enabled=false`). Do not raise the budget beyond the real card.

**`CUDA not available`** — no NVIDIA driver/GPU visible (`nvidia-smi` must work). CPU-capable
adapters can run with `--set gpu.allow_cpu_fallback=true` (very slow); the test profile
enables it.

**CUDA out of memory during a run** — the adapter reports it as an error for the batch; the
items are recorded with the error and the stage continues. Reduce `synth.n_candidates`, keep
VoxCPM2/Fish S2 disabled, or set `analyze.caption: false` (captioning disables itself on OOM).

**`work dir ... belongs to game ...`** — each game needs its own work dir:
`--work-dir work_<game> --out-dir out_<game>`.

**Encrypted KiriKiri archives** (`xp3 index is not plain`) — the built-in reader handles only
unencrypted XP3. Dump files at runtime with KrkrDump (KiriKiri Z) or KirikiriTools
`version.dll` + `extract-unencrypted.txt`, then point `--project` at the dump (see ENGINES.md).
For patches, use `version.dll` so the game accepts an unencrypted `patchN.xp3`.

**Steam/DMM DRM** — injection dumpers may be blocked; detect prints a warning when
`steam_api*.dll` or SoftDenchi markers are present. Use a DRM-free copy you own.

**Ren'Py `.rpyc` only** — install unrpyc v2 and set `tools.unrpyc`; `.rpa` archives need
`unrpa` (`uv sync --extra extract`).

**Many lines in `review` after `link`** — check `work/review_queue.jsonl`. Typical causes:
BGM under the voice (ASR fails), names spelled differently in the script, a wrong
`engine_opts.kag.voice_attr`. Raising `link.review_score` only hides the problem.

**Japanese accent in the Russian output** — route A (direct cloning from the Japanese clip)
carries the accent by design; prefer route B (native performer + RVC) or C, and compare in the
report's route table.

**Wrong Russian stress** — check `text_tts` in `translation/lines.ru.jsonl`; ruaccent must be
present (`vndub models pull ruaccent`). F5-TTS_RUSSIAN is the only backend that takes explicit
`+` stress marks.

**ffmpeg not found** — install it (`winget install Gyan.FFmpeg`) or set `tools.ffmpeg` and
`tools.ffprobe` in `configs/tools.yaml`; check with `uv run vndub tools check`.

**The Gradio review UI does not open** — it binds to `127.0.0.1:7861` by default; use
`--port` if busy and open the printed URL manually (`--browser` opens it automatically).

**Starting over** — delete `work/` (or one stage's cache file `work/cache/<stage>.json`, or use
`--force` on that stage). `work/index.sqlite` can always be deleted.

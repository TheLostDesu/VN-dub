# Troubleshooting

**`backend env 'main' not created`** — the backend's virtual environment is missing:
`powershell -ExecutionPolicy Bypass -File scripts/make_envs.ps1 -Env main` (add `-Cpu` on a
machine without an NVIDIA GPU; the other envs are `funasr`, `chatterbox`, `f5`, `gguf`, `higgs`, `rvc`,
`seedvc`). Logs of the adapter process: `work/logs/adapter_<env>.log`.

**`ruaccent: OfflineModeIsEnabled: Cannot reach https://huggingface.co/api/models/ruaccent/accentuator`** —
the `main` environment was recreated after `models pull ruaccent`: ruaccent keeps its koziev morphology data
(~190 MB) inside its own package folder, which went with the old environment, while the weights folder stayed
marked complete. Since 1.0.0 the pull keeps a copy in `models/ruaccent/koziev` and the adapter restores the
package copy from it; an installation pulled by an older version has no such copy yet: run
`uv run vndub models pull ruaccent` once more (the app: Models -> ruaccent -> download again).

**`cosyvoice3: ErrorDuringImport: problem in cosyvoice.flow.flow_matching - ModuleNotFoundError: No module named 'pkg_resources'`** —
the `funasr` environment has no setuptools (uv creates environments without it) or one newer than 81 (setuptools 82 removed
`pkg_resources`), and lightning 2.2, which CosyVoice's Matcha-TTS imports, needs the module at import time. Since 1.0.1 the
adapter registers a stand-in when the module is missing and the `+cosyvoice` extra pins setuptools 80.10.2; on an older
version install it by hand: `uv pip install --python <envs>/funasr/Scripts/python.exe setuptools==80.10.2`.

**`chatterbox_ml: TypeError: 'NoneType' object is not callable`** (in the adapter log: `self.watermarker = perth.PerthImplicitWatermarker()`) —
the same missing `pkg_resources`, in the `chatterbox` environment: resemble-perth 1.0.1 imports it for its watermarker and hides
the failure (`PerthImplicitWatermarker = None`), so the error appears only after all the weights have loaded. The line
`from_local() got an unexpected keyword argument 't3_model'` above it in logs before 1.0.2 was not the cause. Since 1.0.2 the
adapter registers the stand-in and `envs/chatterbox.txt` pins setuptools 80.10.2; on an older version install it by hand:
`uv pip install --python <envs>/chatterbox/Scripts/python.exe setuptools==80.10.2`.

**Chatterbox's first load downloads `spacy_ontonotes.zip`** (34 MB, from GitHub, to `~/.pkuseg`) —
`chatterbox-tts` builds its Chinese word segmenter on every load, whatever the language, and that model is not part of
`vndub models pull`. It happens once per user account. Without network at that moment the load fails, even for Russian.

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

**`no kernel image is available for execution on the device`** — the card is newer than the torch build of that
environment: the `funasr` (torch 2.3.1, cu121), `seedvc` and `chatterbox` (torch 2.4/2.6, cu124) environments carry no
kernels for compute capability 12.0 (RTX 50xx, Blackwell), so CosyVoice 3, emotion2vec, SenseVoice, Chatterbox and
Seed-VC fail at their first kernel although CUDA itself is available. The `main` env (cu128 torch) is fine. Until those
environments move to a cu128 torch, use the other backends on such a card (Qwen3-TTS, GigaAM, the llama.cpp translator).

**CUDA out of memory during a run** — the adapter reports it as an error for the batch; the
items are recorded with the error and the stage continues. Reduce `synth.n_candidates`, keep
VoxCPM2/Fish S2 disabled, or set `analyze.caption: false` (captioning disables itself on OOM).

**`llm answer covered 0/N lines` over and over, `translate_failed` lines** — the model's JSON answer was
unusable. Look at `work/logs/llm_answers.jsonl`: every incomplete answer is stored there with its
`finish_reason`. `length` means the token budget ran out (a thinking model spent it on a thought block:
Gemma 4 and Qwen 3.x get `enable_thinking: false` from the backend spec; a custom GGUF with another
switch needs it in `chat_template_kwargs`), `stop` with an empty or off-format `content` means the model
did not follow the format (try the fallback model). Check `src_lang` in `work/project.yaml` too: a
script that is not Japanese used to be translated "from Japanese" (now the language is detected at parse;
`translate.llm` logs `translating <src> -> <tgt>`).

**The game's text is English (or another language), not Japanese** — nothing to do: `parse` detects the
language of the script text (`project.src_lang: auto`) and `project.yaml` shows it (`src_lang`,
`src_lang_detected`). Force one with `--set project.src_lang=en` or by editing `project.yaml`. When the
voices are still Japanese (a translated release), set `project.audio_lang: ja`: the ASR check and the aligner
are skipped (they would compare Japanese speech with English text) and link uses the cheap signals.

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

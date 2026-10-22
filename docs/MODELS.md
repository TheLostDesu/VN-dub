# Models

Every backend vndub can use, as registered in `vndub/backends/` (`uv run vndub models list`).
VRAM numbers are estimates for the 8 GB target (one heavy model resident at a time, enforced
by `ModelRegistry`); they come from model cards or community reports (see the research notes)
and must be confirmed with `vndub models selftest <name>` on the GPU machine, which reports
`torch.cuda.max_memory_allocated` as `peak_vram_gb`.

## Commands

```bash
uv run vndub models list                      # all backends: role, env, license, VRAM estimate
uv run vndub models pull --profile default    # the v1 model set (~25 GB incl. the 12B GGUF)
uv run vndub models pull --profile test       # small CPU set for the test profile (~1.8 GB)
uv run vndub models pull whisper_tiny ecapa   # individual backends
uv run vndub models status --profile default  # present / size on disk / usable / why not
uv run vndub models selftest qwen3_tts        # load, 2 s clip, peak VRAM, unload
```

Each env can also self-test without the core: `.venvs\<env>\Scripts\python.exe -m adapters.<env>.main --selftest [backend...]`.

Since 2026-10-22 the former `asr`, `hf` and `score` envs are ONE env, `main` (`envs/main.txt`: torch 2.8.0 cu128,
transformers 4.57.3; the adapter code keeps its packages `adapters/asr`, `adapters/hf`, `adapters/score` and is dispatched by
`adapters/main/table.py`). The "Env" column below says `main` for all of them; the build-machine notes further down still name the
envs as they were built on 2026-10-21.

Weights live in `models/` (or `VNDUB_MODELS_DIR`): one directory per repo
(`models/<repo-basename>/`, `models/<name>-ct2` conversions inside it), plus `models/hf_home`
and `models/torch_home` for packages that cache on their own (SSL backbones, torch.hub).
A finished pull writes `.vndub_complete`; run time is offline (`HF_HUB_OFFLINE=1` for adapters).

Profiles: **default** = v1 pipeline; **test** = CPU/tiny profile (`configs/pipeline.test.yaml`);
**optional** = implemented, disabled by default. `license_policy: commercial` drops every
backend whose "Commercial" column is "no" (falling back or marking lines for review).

## Backends

| Backend | Role | Model / repo | Env | VRAM est. GB | License | Commercial | Fallback | Profile | Status on build machine |
|---|---|---|---|---|---|---|---|---|---|
| [anime_whisper](https://huggingface.co/litagin/anime-whisper) | asr | `litagin/anime-whisper` | main | 2.5 | MIT | yes | whisper_large_v3_turbo | default | not verified: weights missing: run `vndub models pull anime_whisper` |
| [gigaam_v3](https://huggingface.co/ai-sage/GigaAM-v3) | asr | `ai-sage/GigaAM-v3`@e2e_ctc | main | 1.2 | MIT | yes | whisper_ru | default | **selftest OK on CPU** (load 35.5 s, 2 s clip 1.61 s) |
| mock_asr | asr | - | core | 0.0 | MIT | yes | - | test | core (always available) |
| [qwen3_asr](https://github.com/QwenLM/Qwen3-ASR) | asr | `Qwen/Qwen3-ASR-1.7B` | main | 4.0 | Apache-2.0 | yes | - | optional | not verified: env 'main' not created |
| [whisper_large_v3_turbo](https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo) | asr | `mobiuslabsgmbh/faster-whisper-large-v3-turbo` | main | 1.8 | MIT | yes | - | default | not verified: weights missing: run `vndub models pull whisper_large_v3_turbo` |
| [whisper_ru](https://huggingface.co/antony66/whisper-large-v3-russian) | asr | `antony66/whisper-large-v3-russian` | main | 3.0 | MIT | yes | - | default | not verified: weights missing: run `vndub models pull whisper_ru` |
| [whisper_tiny](https://huggingface.co/Systran/faster-whisper-tiny) | asr | `Systran/faster-whisper-tiny` | main | 0.3 | MIT | yes | - | test | **selftest OK on CPU** (load 11.17 s, 2 s clip 0.94 s) |
| mock_aligner | align | - | core | 0.0 | MIT | yes | - | test | core (always available) |
| [qwen3_aligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) | align | `Qwen/Qwen3-ForcedAligner-0.6B` | main | 1.5 | Apache-2.0 | yes | whisperx_ja | default | not verified: env 'main' not created |
| [whisperx_ja](https://github.com/m-bain/whisperX) | align | `jonatasgrosman/wav2vec2-large-xlsr-53-japanese` | main | 1.3 | Apache-2.0 (model) / BSD-2 (whisperx) | yes | - | default | not verified: env 'main' not created |
| [anime_ser](https://huggingface.co/litagin/anime_speech_emotion_classification) | ser | `litagin/anime_speech_emotion_classification` | main | 0.8 | MIT | yes | - | default | not verified: env 'main' not created |
| [emotion2vec_plus_base](https://github.com/ddlBoJack/emotion2vec) | ser | `emotion2vec/emotion2vec_plus_base` | funasr | 0.8 | MIT (code); emotion2vec model license (commercial terms UNVERIFIED) | no | - | default | not verified: weights missing: run `vndub models pull emotion2vec_plus_base` |
| [emotion2vec_plus_large](https://github.com/ddlBoJack/emotion2vec) | ser | `emotion2vec/emotion2vec_plus_large` | funasr | 1.8 | MIT (code); emotion2vec model license (commercial terms UNVERIFIED) | no | emotion2vec_plus_base | default | not verified: weights missing: run `vndub models pull emotion2vec_plus_large` |
| [emotion2vec_plus_seed](https://github.com/ddlBoJack/emotion2vec) | ser | `emotion2vec/emotion2vec_plus_seed` | funasr | 0.8 | MIT (code); emotion2vec model license (commercial terms UNVERIFIED) | no | - | test | **selftest OK on CPU** (load 46.95 s, 2 s clip 1.25 s) |
| mock_anime_ser | ser | - | core | 0.0 | MIT | yes | - | test | core (always available) |
| mock_ser | ser | - | core | 0.0 | MIT | yes | - | test | core (always available) |
| [audeering_adv](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim) | adv | `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` | main | 1.0 | CC-BY-NC-SA-4.0 | no | - | default | not verified: env 'main' not created |
| mock_adv | adv | - | core | 0.0 | MIT | yes | - | test | core (always available) |
| mock_events | events | - | core | 0.0 | MIT | yes | - | test | core (always available) |
| [sensevoice](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) | events | `FunAudioLLM/SenseVoiceSmall` | funasr | 1.0 | FunASR model license (maintainers state commercial use OK) | yes | - | default | not verified: weights missing: run `vndub models pull sensevoice` |
| mock_caption | caption | - | core | 0.0 | MIT | yes | - | test | core (always available) |
| [qwen2_audio](https://github.com/QwenLM/Qwen2-Audio) | caption | `Qwen/Qwen2-Audio-7B-Instruct` | main | 5.8 | Apache-2.0 | yes | - | optional | not verified: env 'main' not created |
| librosa_f0 | f0 | - | core | 0.0 | ISC (librosa) | yes | - | default,test | core (always available) |
| [rmvpe](https://github.com/Dream-High/RMVPE) | f0 | downloaded by the env (`models pull`) | rvc | 0.6 | Apache-2.0 (RMVPE) / MIT (Applio) | yes | torchcrepe | default | not verified: env 'rvc' not created |
| [torchcrepe](https://github.com/maxrmorrison/torchcrepe) | f0 | bundled with the pip package | main | 0.5 | MIT | yes | librosa_f0 | default | **selftest OK on CPU** (load 16.28 s, 2 s clip 19.92 s) |
| energy_vad | vad | - | core | 0.0 | MIT | yes | - | default,test | core (always available) |
| [silero](https://github.com/snakers4/silero-vad) | vad | bundled with the pip package | main | 0.0 | MIT | yes | energy_vad | default | **selftest OK on CPU** (load 10.71 s, 2 s clip 0.22 s) |
| mock_stress | stress | - | core | 0.0 | MIT | yes | - | test | core (always available) |
| [ruaccent](https://github.com/Den4ikAI/ruaccent) | stress | downloaded by the env (`models pull`) | main | 0.0 | MIT | yes | - | default,test | **selftest OK on CPU** (load 56.25 s, 2 s clip 0.28 s) |
| [ecapa](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | embedder | `speechbrain/spkrec-ecapa-voxceleb` | main | 0.4 | Apache-2.0 | yes | - | default,test | **selftest OK on CPU** (load 17.68 s, 2 s clip 1.74 s) |
| mock_embedder | embedder | - | core | 0.0 | MIT | yes | - | test | core (always available) |
| [wavlm_sv](https://huggingface.co/microsoft/wavlm-base-plus-sv) | embedder | `microsoft/wavlm-base-plus-sv` | main | 0.8 | MIT | yes | - | optional | not verified: weights missing: run `vndub models pull wavlm_sv` |
| mock_mos | mos | - | core | 0.0 | MIT | yes | - | test | core (always available) |
| [utmos22](https://github.com/tarepan/SpeechMOS) | mos | downloaded by the env (`models pull`) | main | 0.5 | MIT | yes | - | default,test | **selftest OK on CPU** (load 16.62 s, 2 s clip 9.89 s) |
| [utmosv2](https://github.com/sarulab-speech/UTMOSv2) | mos | downloaded by the env (`models pull`) | main | 1.5 | MIT | yes | utmos22 | default | **selftest OK on CPU** (load 51.85 s, 2 s clip 39.08 s) |
| [audiobox](https://github.com/facebookresearch/audiobox-aesthetics) | aesthetics | `facebook/audiobox-aesthetics` | main | 1.0 | CC-BY-4.0 | yes | - | default | **selftest OK on CPU** (load 16.84 s, 2 s clip 8.61 s) |
| mock_aesthetics | aesthetics | - | core | 0.0 | MIT | yes | - | test | core (always available) |
| [gemma4_12b](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF) | llm | `unsloth/gemma-4-12b-it-GGUF` | core | 7.5 | Apache-2.0 | yes | qwen35_9b | default | not verified: GGUF not pulled (`vndub models pull gemma4_12b`); also needs `tools.llama_server` |
| [hy_mt15_7b](https://huggingface.co/tencent/HY-MT1.5-7B-GGUF) | llm | `tencent/HY-MT1.5-7B-GGUF` | core | 6.0 | Tencent HY-MT license (see model card) | yes | - | default | not verified: GGUF not pulled (`vndub models pull hy_mt15_7b`); also needs `tools.llama_server` |
| mock_translator | llm | - | core | 0.0 | MIT | yes | - | test | core (always available) |
| [qwen35_9b](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) | llm | `unsloth/Qwen3.5-9B-GGUF` | core | 6.3 | Apache-2.0 | yes | hy_mt15_7b | default | not verified: GGUF not pulled (`vndub models pull qwen35_9b`); also needs `tools.llama_server` |
| [chatterbox_ml](https://github.com/resemble-ai/chatterbox) | tts | `ResembleAI/chatterbox` | chatterbox | 4.0 | MIT (Perth watermark embedded) | yes | - | default | not verified: env 'chatterbox' not created |
| [cosyvoice3](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) | tts | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | funasr | 3.8 | Apache-2.0 | yes | - | default | not verified: weights missing: run `vndub models pull cosyvoice3` |
| [f5_ru](https://huggingface.co/Misha24-10/F5-TTS_RUSSIAN) | tts | `Misha24-10/F5-TTS_RUSSIAN` | f5 | 3.0 | CC-BY-NC-4.0 (Russian checkpoint) | no | - | optional | not verified: env 'f5' not created |
| [fish_s2](https://huggingface.co/fishaudio/s2-pro) | tts | `fishaudio/s2-pro-gguf` | gguf | 7.5 | Fish Audio Research License (non-commercial) | no | - | optional | not verified: env 'gguf' not created |
| [higgs_tts3](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b) | tts | `bosonai/higgs-audio-v3-tts-4b` | higgs | 6.0 | Boson Research & Non-Commercial License | no | - | optional | not verified: env 'higgs' not created |
| mock_tts | tts | - | core | 0.0 | MIT | yes | - | test | core (always available) |
| [moss_tts](https://github.com/OpenMOSS/MOSS-TTS) | tts | `OpenMOSS-Team/MOSS-TTS-v1.5-GGUF` | gguf | 7.0 | Apache-2.0 | yes | - | optional | not verified: env 'gguf' not created |
| [qwen3_tts](https://github.com/QwenLM/Qwen3-TTS) | tts | `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | main | 4.5 | Apache-2.0 | yes | - | default | not verified: env 'main' not created |
| [qwen3_tts_06b](https://github.com/QwenLM/Qwen3-TTS) | tts | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | main | 2.5 | Apache-2.0 | yes | - | optional | not verified: env 'main' not created |
| [qwen3_tts_voicedesign](https://github.com/QwenLM/Qwen3-TTS) | tts | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | main | 4.5 | Apache-2.0 | yes | - | default | not verified: env 'main' not created |
| sapi_tts | tts | - | core | 0.0 | Windows component (test use only) | no | - | test | core (always available) |
| [voxcpm2](https://github.com/OpenBMB/VoxCPM) | tts | `openbmb/VoxCPM2` | main | 7.8 | Apache-2.0 | yes | - | default | not verified: env 'main' not created |
| [zonos2](https://github.com/Zyphra/zonos2.cpp) | tts | `Zyphra/ZONOS2-GGUF` | gguf | 6.5 | MIT/Apache-2.0 (conflicting statements, both permissive) | yes | - | optional | not verified: env 'gguf' not created |
| mock_vc | vc | - | core | 0.0 | MIT | yes | - | test | core (always available) |
| [rvc](https://github.com/IAHispano/Applio) | vc | downloaded by the env (`models pull`) | rvc | 2.0 | MIT | yes | - | default | not verified: env 'rvc' not created |
| [seedvc](https://github.com/Plachtaa/seed-vc) | vc | downloaded by the env (`models pull`) | seedvc | 3.0 | GPL-3.0 | yes | - | default | not verified: env 'seedvc' not created |

Mocks (`mock_*`) and `sapi_tts` (Windows SAPI voices, test only) run in-process; they exist so the
test profile exercises every stage without downloads. The test profile substitutes a backend's mock
only when the real one is unusable, and every substitution is logged and reported.

## Verified on the build machine (2026-10-21)

Machine: Windows 11 Pro, Intel Core i7-1185G7, 48 GB RAM, Intel Iris Xe -- **no NVIDIA GPU**, so
everything below ran on CPU and `peak_vram_gb` is `null`. Raw output: [selftest_results.json](selftest_results.json).

Environments created with `scripts/make_envs.ps1 -Cpu`:

| Env | Torch | Build time | Notes |
|---|---|---|---|
| asr | 2.8.0+cpu | a few minutes (not cleanly timed: the first runs exposed script bugs that were fixed) | `ctranslate2` must be >= 4.7 (4.6.0 imports the removed `pkg_resources`) |
| score | 2.8.0+cpu | 11.1 min | incl. the `+utmosv2` extra (UTMOSv2 from git, timm, audiobox_aesthetics) |
| funasr | 2.3.1+cpu | 2.3 min base + 1.1 min `+cosyvoice` extra | CosyVoice repo cloned at the pinned commit; `openai-whisper` needs `envs/build.constraints.txt` (setuptools<70) |

Selftests that passed (2 s synthetic clip; CPU). First runs used librosa's `soxr_hq` resampler,
which took ~1.2 s per clip on this machine; after switching to scipy polyphase resampling the
same selftests were re-run ("after" column, `vndub models selftest ...`, 2026-10-21):

| Backend | Load s | Run s (before) | Run s (after fix) | Output |
|---|---|---|---|---|
| whisper_tiny | 4-11 | 0.4-0.9 | 0.25 | transcript (empty on the synthetic tone) |
| gigaam_v3 | 35 | 1.6-2.0 | 0.25 | transcript |
| silero | 5-11 | 0.1-0.2 | 0.27 | speech segments |
| ruaccent | 21-56 | 0.1-0.3 | 0.08 | `На двер+и вис+ит зам+ок, а вдал+и виден ст+арый з+амок.` (homographs замок/замок resolved) |
| ecapa | 5-18 | 0.2-1.7 | 0.16 | 192-d embedding |
| utmos22 | 3-17 | 5-10 | 1.5 | MOS |
| utmosv2 | 20-52 | 36-39 | 7.44 | MOS (slow on CPU: multi-crop SSL + spectrogram branches) |
| audiobox | 7-17 | 3-9 | 2.1 | CE/CU/PC/PQ |
| torchcrepe | 8-16 | 20-24 | 5.56 | F0 at 10 ms (`full` model is slow on CPU) |
| emotion2vec_plus_seed | 43-47 | 0.6-1.3 | 0.28 | 9 probs + 768-d embedding |

Real-speech check through the JSON-RPC path (Windows SAPI voices as the source, script in the build log):

| Check | Result |
|---|---|
| GigaAM-v3 e2e_ctc on a Russian SAPI sentence | CER **0.042** (only "сто" -> "100" differs: e2e output normalizes numbers) |
| whisper tiny, same Russian sentence | CER 0.097 |
| whisper tiny on a Japanese SAPI line `今日はね、すごくいいことがあったの！` | exact text (`。` instead of `！`) |
| ruaccent | `Сег+одня случ+илось кое-что +очень хор+ошее ...` |
| emotion2vec+ seed on the cheerful Japanese line | `happy` 0.97 |
| UTMOS22 on SAPI Russian | 2.66 (robotic TTS, as expected) |

Not verified here (no NVIDIA GPU, and multi-GB downloads were out of scope): every GPU backend in
the hf, chatterbox, rvc, seedvc, f5, gguf and higgs envs, CosyVoice3 inference, the LLMs.
Their adapters are implemented against the upstream APIs pinned in `envs/*.txt`; `selftest`
reports a clear "env not created" / "weights missing: run `vndub models pull <name>`" instead
of a traceback.

## Env notes and known risks

* **Default env split deviations.** The spec's `hf` env cannot hold Chatterbox (it pins torch 2.6,
  numpy<2 and its own transformers/diffusers), so Chatterbox has its own `chatterbox` env.
  F5-TTS (`f5`), the GGUF CLI wrappers (`gguf`) and Higgs (`higgs`) also get their own envs.
* **hf env pins** follow `qwen-tts==0.1.1` (transformers 4.57.3). `qwen-asr` pins 4.57.6, so it is
  installed with `--no-deps` in the `+aligner` extra; its `nagisa` dependency may lack Windows wheels
  -- the extra then fails without breaking the env (Qwen3-ForcedAligner falls back to `whisperx_ja`).
* **funasr env** uses CosyVoice's own pins (torch 2.3.1, numpy 1.26, transformers 4.51.3) so the
  FunASR models and CosyVoice3 share one env, as the spec's DEFAULT split asks.
* **UTMOSv2** downloads its checkpoint with `wget` (absent on Windows); `models pull utmosv2` fetches
  `fold0_s42_best_model.pth` with huggingface_hub instead and passes it as `checkpoint_path`.
* **ECAPA (speechbrain)**: the hub `hyperparams.yaml` points `pretrained_path` at the hub; the adapter
  overrides it to the local snapshot and uses the snapshot dir as `savedir` (no symlinks on Windows).
* **GigaAM-v3** remote code imports `pyannote`/`torchcodec` for long-form only; the adapter tolerates
  their absence and replaces the ffmpeg-based loader with soundfile. Clips > 24 s are chunked.
* **RVC / Applio 3.7.0**: its CLI no longer has an overtraining detector, so `voicebank.rvc.overfit_patience`
  is recorded but not enforced. Training runs as a child process (Applio reads `APPLIO_LOGS_DIR` at import).
* **Seed-VC**: archived upstream (GPL-3.0); v1 wrapper = timbre-only conversion (equivalent of v2
  `convert_style=false`). `SeedVCWrapper.convert_voice` is a generator even without streaming.
* **Fish S2 / Zonos2 / MOSS-TTS**: command lines are UNVERIFIED; they are templates in
  `options.args` (see `adapters/gguf/cli_impl.py`) so the flags can be fixed from `pipeline.yaml`.
* **Higgs TTS 3** ships for SGLang-Omni (Linux); the adapter is a client of a server on 127.0.0.1
  (WSL2/Docker on the same machine). Non-loopback URLs are refused.
* **VoxCPM2** (~8 GB official) stays disabled until its selftest passes on the target GPU;
  `torch.compile` (`optimize`) is off by default on Windows.

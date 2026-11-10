# Backend environments

Each group of heavy models runs in a virtual environment `.venvs/<env>` (Python 3.11, created with
`uv`) as a JSON-RPC subprocess (`python -m adapters.<env>.main`). The core `vndub` package never
imports these libraries. Models that agree on torch share one env: `main` holds speech recognition,
the Transformers models (Qwen3-TTS, SER, aligners) and quality scoring, all on torch 2.8.0 cu128
(until 2026-10-22 these were three envs, `asr`, `hf` and `score`, each with its own copy of torch).
The other envs exist only because their pins conflict (torch 2.3 / 2.4 / 2.6 / 2.11).

Create them with:

```powershell
scripts\make_envs.ps1 -Env main           # one env, CUDA torch
scripts\make_envs.ps1 -Env main,funasr -Cpu # CPU torch (no NVIDIA GPU / CI)
scripts\make_envs.ps1 -Env all
```

```bash
scripts/make_envs.sh main funasr        # CUDA
CPU=1 scripts/make_envs.sh main funasr  # CPU torch
```

### Where the environments go

By default `.venvs/<env>` in the checkout. The desktop app keeps them in its data folder (`%LOCALAPPDATA%\VNDub\envs`):
both scripts take the target (`-EnvsDir <dir>` / `ENVS_DIR=<dir>`, default `$VNDUB_ENVS_DIR`, else `<repo>/.venvs`) and the
pipeline finds them through the same variable `VNDUB_ENVS_DIR` (`vndub.backends.runner.envs_root`). `-Cpu` / `CPU=1` selects
CPU torch wheels; the app's "Install environments" button runs these scripts with the choice the Doctor makes (`torch: auto`).

```powershell
$env:VNDUB_ENVS_DIR = "D:ndub-envs"; scripts\make_envs.ps1 -Env main  # creates D:ndub-envs\main; the pipeline uses it
```

The desktop app also keeps uv's managed Pythons and its wheel cache next to the envs (`<envs>/_uv/python`, `<envs>/_uv/cache`,
`UV_PYTHON_INSTALL_DIR` / `UV_CACHE_DIR` set for every uv it runs) so that a wheel two envs share is downloaded once and hardlinked, and
passes the concrete interpreter (`-Python <path>`) instead of a version; see `vndub_app/installer.py`.
From the command line the plain `make_envs` calls above use uv's defaults.

### Build record and health checks

Every env the scripts build gets `<env>/vndub_env.json` (written by `adapters/common/envrecord.py` with the env's own
interpreter): the sha256 of its requirement files (`<env>.txt`, `<env>+*.txt`, `build.constraints.txt`), the app version,
the HEAD of every `# @git` checkout, the installed torch kind (from the wheel's `+cpu` / `+cuXXX` tag) and the extras that
failed. The record is deleted when a build of the env starts, so an interrupted build has none. The app compares it with
the current files (`vndub_app/installer.py: env_states`): unmet `==` pins ("built by an older version"), a failed extra, a
CPU torch on a machine with an NVIDIA card. A rebuild over an existing env is incremental (uv moves the pins); a CPU torch
asked to become CUDA is reinstalled (`--reinstall-package`, since `torch==X` is met by `X+cpu`), and a `# @git` checkout
pinned to a commit is moved to the new commit.

Import check without weights or a GPU (what each adapter's `load` imports, with its shims; CosyVoice 3 locates every class
its model config names):

```powershell
.venvs\funasr\Scripts\python.exe -m adapters.funasr.main --probe          # all adapters of the env, JSON on one line
.venvs\chatterbox\Scripts\python.exe -m adapters.chatterbox.main --probe chatterbox_ml
```

The app runs it after every env install and from Diagnostics ("Check environments"), and `packaging/build_app.ps1` runs it
for every env in `.venvs` before a release (`-SkipEnvProbe "<reason>"` to build without).

## File format

`envs/<env>.txt` is a pip requirements file with directives in comments:

| Directive | Meaning |
|---|---|
| `# @torch torch==X torchaudio==Y` | installed first from the PyTorch index (`-Cpu` -> `/whl/cpu`) |
| `# @cuda cu128` | CUDA wheel tag of that index for GPU installs |
| `# @git <name> <url> <ref> [--recursive]` | clone into `.venvs/<env>/src/<name>` at the pinned ref |
| `# @req-from <name> <file>` | also install `<file>` from that checkout |
| `# @nodeps <requirement>` | installed with `--no-deps` (pin conflicts upstream) |

`envs/<env>+<extra>.txt` are optional extras installed after the base file; a failing
extra is reported and skipped (the base env stays usable), unless `-Strict`.
`envs/<env>.lock.txt` (when present) is the `uv pip freeze` of an env that was
built and verified on the build machine.

| Env | Torch | Backends |
|---|---|---|
| main | 2.8.0 cu128 (+ torchvision 0.23.0) | faster-whisper (anime-whisper, whisper-*), GigaAM-v3, silero VAD, ruaccent; anime SER, audeering A/D/V, CTC aligner, Qwen3-TTS; UTMOS22, ECAPA, WavLM-SV, torchcrepe. Extras: `+aligner`: Qwen3-ForcedAligner / Qwen3-ASR; `+voxcpm`: VoxCPM2; `+caption`: Qwen2-Audio NF4; `+utmosv2`: UTMOSv2, Audiobox Aesthetics |
| funasr | 2.3.1 cu121 | emotion2vec+, SenseVoice; `+cosyvoice`: Fun-CosyVoice3 (CosyVoice pins) |
| chatterbox | 2.6.0 cu124 | Chatterbox Multilingual (strict pins: own env) |
| rvc | 2.11.0 cu128 | Applio 3.7.0 (RVC v2 train/infer, RMVPE) |
| seedvc | 2.4.0 cu124 | Seed-VC |
| f5 | 2.8.0 cu128 | F5-TTS + F5-TTS_RUSSIAN (its own env until its pins are verified against main) |
| gguf | - | CLI wrappers: Fish S2 (audio.cpp), Zonos2 (zonos2.cpp), MOSS-TTS |
| higgs | 2.8.0 cu128 | Higgs TTS 3 (4-bit) |
| llm | - | nothing: llama-server is a binary, the client lives in core |

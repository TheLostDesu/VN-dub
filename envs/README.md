# Backend environments

Each heavy model family runs in its own virtual environment `.venvs/<env>` (Python 3.11,
created with `uv`) as a JSON-RPC subprocess (`python -m adapters.<env>.main`). The core
`vndub` package never imports these libraries.

Create them with:

```powershell
scripts\make_envs.ps1 -Env asr            # one env, CUDA torch
scripts\make_envs.ps1 -Env asr,score -Cpu # CPU torch (no NVIDIA GPU / CI)
scripts\make_envs.ps1 -Env all
```

```bash
scripts/make_envs.sh asr score          # CUDA
CPU=1 scripts/make_envs.sh asr score    # CPU torch
```

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
| asr | 2.8.0 cu128 | faster-whisper (anime-whisper, whisper-*), GigaAM-v3, silero VAD, ruaccent |
| funasr | 2.3.1 cu121 | emotion2vec+, SenseVoice; `+cosyvoice`: Fun-CosyVoice3 (CosyVoice pins) |
| hf | 2.8.0 cu128 | anime SER, audeering A/D/V, CTC aligner, Qwen3-TTS; `+aligner`: Qwen3-ForcedAligner / Qwen3-ASR; `+voxcpm`: VoxCPM2; `+caption`: Qwen2-Audio NF4 |
| chatterbox | 2.6.0 cu124 | Chatterbox Multilingual (strict pins: own env) |
| rvc | 2.11.0 cu128 | Applio 3.7.0 (RVC v2 train/infer, RMVPE) |
| seedvc | 2.4.0 cu124 | Seed-VC |
| score | 2.8.0 cu128 | UTMOSv2, UTMOS22, Audiobox Aesthetics, ECAPA, WavLM-SV, torchcrepe |
| f5 | 2.8.0 cu128 | F5-TTS + F5-TTS_RUSSIAN |
| gguf | - | CLI wrappers: Fish S2 (audio.cpp), Zonos2 (zonos2.cpp), MOSS-TTS |
| higgs | 2.8.0 cu128 | Higgs TTS 3 (4-bit) |
| llm | - | nothing: llama-server is a binary, the client lives in core |

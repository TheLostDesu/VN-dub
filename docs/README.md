# vndub quickstart

## 1. Requirements

- Windows 11 (Linux works for the core and most envs), Python 3.11 via [uv](https://docs.astral.sh/uv/).
- One NVIDIA GPU with 8 GB VRAM and a CUDA 12.x driver for the default profile.
  Without a GPU only the CPU test profile is practical.
- ffmpeg/ffprobe on PATH or in `configs/tools.yaml` (`winget install Gyan.FFmpeg`).
- Disk: ~60 GB for the default model set, plus several GB per game for `work/`.

## 2. Install

```bash
uv sync                               # core package + CLI (light: no torch)
uv run vndub --help
uv run vndub tools check              # which external tools are present, where to get the rest
```

Backend environments (each heavy model family is isolated in `.venvs/<env>`):

```bash
powershell -ExecutionPolicy Bypass -File scripts/make_envs.ps1 -Env all        # CUDA 12 torch
powershell -ExecutionPolicy Bypass -File scripts/make_envs.ps1 -Env asr -Cpu   # CPU torch (no GPU)
```

Linux/macOS: `scripts/make_envs.sh all` (or `scripts/make_envs.sh asr --cpu`).

## 3. Download models (once; the pipeline then runs offline)

```bash
uv run vndub models pull --profile default    # the v1 model set (~60 GB)
uv run vndub models pull --profile test       # tiny CPU models for the test profile
uv run vndub models status                    # what is present, VRAM estimate, why not usable
uv run vndub models selftest qwen3_tts        # load, synthesize 2 s, report peak VRAM, unload
```

The translator needs llama.cpp's `llama-server` (CUDA build) set in `configs/tools.yaml`.

## 4. Try it on the bundled fixture (CPU, no big downloads)

```bash
uv run vndub run --config configs/pipeline.test.yaml --project tests/fixtures/renpy_mini
```

Outputs: `out/renpy/game/voice/*.ogg` (the patch), `work/report/index.html` (the report).
The test profile substitutes deterministic mocks for every backend whose env or weights
are missing and lists each substitution in the report.

## 5. A real game

```bash
uv run vndub detect --project "D:/Games/SomeVN"          # writes work/project.yaml (edit if needed)
uv run vndub extract --project "D:/Games/SomeVN"
uv run vndub parse --project "D:/Games/SomeVN"
uv run vndub link --project "D:/Games/SomeVN"            # check work/review_queue.jsonl
uv run vndub analyze --project "D:/Games/SomeVN"
uv run vndub translate --project "D:/Games/SomeVN" --seed-glossary   # then edit configs/glossary.yaml
uv run vndub translate --project "D:/Games/SomeVN"
uv run vndub voicebank --project "D:/Games/SomeVN"
uv run vndub synth --project "D:/Games/SomeVN"
uv run vndub score --project "D:/Games/SomeVN"
uv run vndub master --project "D:/Games/SomeVN"
uv run vndub pack --project "D:/Games/SomeVN"
uv run vndub report --project "D:/Games/SomeVN"
```

or simply `uv run vndub run --project "D:/Games/SomeVN"`. Every stage is resumable: rerun
after a crash or a config change and only affected items are recomputed. Use a separate
`--work-dir/--out-dir` per game (vndub refuses to mix two games in one work dir).

Useful flags on every stage: `--only s01_000123,s01_000124`, `--limit 50`, `--force`,
`--set synth.n_candidates=6`.

## 6. Review

```bash
uv run vndub review --project "D:/Games/SomeVN"     # http://127.0.0.1:7861
```

Filter lines, listen to the original and every candidate, accept / reject / regenerate with
an edited text or style prompt, pick another reference clip, keep the original. Decisions
go to `work/overrides.jsonl`; then `vndub synth --only <ids>` and `vndub score --only <ids>`
(or `vndub run --from synth`) apply them, followed by `master` and `pack`.

## 7. Where things are

```
work/project.yaml            engine, codec, encryption flags (editable)
work/manifest/lines.jsonl    one record per script line
work/manifest/characters.yaml  per character: names, gender, stats, refs, performer, rvc, route
work/link.jsonl              ASR verification of the mapping; work/review_queue.jsonl
work/analysis/<line>.json    expression passport
work/translation/lines.ru.jsonl
work/voicebank/<speaker>/    refs, performer, rvc, lora
work/synth/<line>/           candidates + sidecars;  work/scores/<line>.jsonl metrics
work/selection.jsonl         winners;  work/final_index.jsonl encoded finals
work/report/index.html       report;   work/logs/<stage>.log
out/<engine>/                the patch + INSTALL.md + manifest_patch.json
```

See [PIPELINE.md](PIPELINE.md) for every stage and config key.

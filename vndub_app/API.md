# VNDub desktop app — internal contract (API, data, pipeline hooks)

This file was the single agreement between the app's parts while they were built in parallel: the pipeline (`vndub/`), the
backend (`vndub_app/`), the SPA (`vndub_app/web/`) and packaging. Spec:
the desktop app specification. If you need to
change a shape below, keep it backward compatible and note it in your final report.

## 0. Shared core (already written — use, do not rewrite)

| Module | What |
|---|---|
| `vndub_app/appdirs.py` | data folder, `Settings`, `ProjectHandle` (`work`, `out`, `config_file`, `glossary_file`, `state_file`, `game_dir`), `create_project`, `list_projects`, `read_app_state` / `write_app_state`, `write_project_config` (generates `<project>/pipeline.yaml` that `extends` the bundled profile; sets `link.mode: no_asr` unless Advanced `asr_check`), `resource()` for bundled files |
| `vndub_app/errors.py` | `AppError(code, params=, action=, details=, status=)`, `error_from_exception`, `install_handlers(app)`; action vocabulary `ACTIONS` |
| `vndub_app/jobs.py` | `runner()` (process-wide `JobRunner`), `JobContext` (`set_stages`, `begin_stage`, `progress`, `message`, `log`, `set_result`, `checkpoint`, `stage_progress_cb()`), `EtaEstimator`, `ThroughputStore`, `JobCancelled` |
| `vndub/stages/common.py` | `StageContext.progress(done, total, item=None)` — calls `ctx.extra["progress_cb"](stage, done, total, item)` when set (may block = pause, may raise = cancel) |
| `vndub/config.py` | new keys: `link.mode` (`asr` \| `no_asr`, CLI default `asr`), `link.recover_text` (bool), `link.recover_asr`; section `text_emotion` (`enabled`, `cues`, `narration_window`, `llm` auto/on/off, `llm_context_lines`, `min_confidence`, `only_unvoiced`) |
| `configs/pipeline.apptest.yaml` | all-mock profile (seconds per full run on `tests/fixtures/renpy_mini`); select with `VNDUB_APP_PROFILE=apptest` |

Profiles: `VNDUB_APP_PROFILE` = `default` | `test` | `apptest`. Data folder: `VNDUB_DATA`
(tests always set it to a tmp dir). Models dir env for backends: `VNDUB_MODELS_DIR`; backend
venvs dir: `VNDUB_ENVS_DIR` (added by the installer task in `vndub/backends/runner.py`); user-added models:
`VNDUB_CUSTOM_MODELS` = path of a JSON list (the app sets it to `<data>/custom_models.json` at startup; unset = the CLI's
registry is exactly the built-in one; see the `/api/models/hf/*` and `/api/models/custom` rows of section 3).

## 1. Building a pipeline context from the app

```python
from vndub.cli import make_ctx            # existing
from vndub.stages.common import run_stage  # existing
h = appdirs.project_handle(slug); appdirs.write_project_config(h)
ctx = make_ctx(h.game_dir, h.config_file, only="id1,id2" | None, limit=None, force=False,
               sets=["synth.n_candidates=1"], extra={"progress_cb": job_ctx.stage_progress_cb()})
try:
    for st in stages: job_ctx.begin_stage(st); run_stage(st, ctx)   # score -> cli._regen_loop(ctx)
finally:
    ctx.models.close()
```
Stage log lines: attach a `logging.Handler` to logger `vndub` (stage loggers are children,
`vndub.<stage>`) that forwards records to `job_ctx.log()`.

## 2. Pipeline additions (owned by the pipeline tasks; the app calls exactly these)

STAGES order becomes: `detect extract parse link analyze translate text_emotion voicebank synth score master pack report`.
`text_emotion` is cheap and cached; the app also runs it in Step 1 (rule + narration cues only,
before any translation), and again after `translate` (merges the LLM layer).

### 2.1 link / mode C (task P1)
* `link.mode: no_asr` — no ASR backend is acquired; `LinkRecord.status` from cheap signals; reasons
  are machine-readable strings prefixed with the signal: `spk_outlier:<x>`, `duration_implausible:<ratio>`,
  `order_gap`, …; summary gains `mode`, `confidence` (0..1, share of lines with all signals ok),
  `outliers` (count), `suggest_asr_check` (bool: many outliers).
* Mode C: `link.recover_text: true` transcribes only voiced lines with empty `text_src`; result stored
  in `link.jsonl` (`text_recovered`) and applied by `StageContext.lines()` / `voiced_lines()` as
  `text_src` with `flags += ["text_from_asr"]`. User edits of such text: override action `set_src_text`
  (payload `{"text": ...}`), applied the same way (later wins).
* Helper for the app: `vndub.stages.link.no_text_lines(ctx) -> list[str]` (voiced line_ids without text).
* Mixed fixture: `tests/fixtures/renpy_mini` gains an unvoiced protagonist (speaker id `hero`) and a few
  unvoiced narration lines around dialogue, incl. speech-verb cues (`と叫んだ`, `ささやいた` …).

### 2.2 text_emotion (task P2)
* `vndub/stages/text_emotion.py` (`NAME="text_emotion"`), CLI `vndub text-emotion`; writes
  `analysis/<line_id>.json` (Passport schema) with extra fields
  `source: "text"`, `delivery: {"emotion", "intensity", "tempo": slow|normal|fast, "loudness": soft|normal|loud,
  "events": [...], "confidence", "layer": "rule"|"cue"|"llm"|"neutral", "cues": [matched cue strings]}`
  and `style` = StyleSpec dict rendered through `text/style_prompt.py`.
* Lines processed: speaker lines without original audio (mode B). Narration lines are context only.
* LLM layer: `translate` asks the same llama-server session for one more JSON field per line
  (`delivery`) for those lines; stored in `TranslationRecord.delivery` (extra field). `text_emotion`
  merges rule → cue → llm with `min_confidence`. Mock translator returns a deterministic delivery.
* `configs/delivery_cues.yaml` (bilingual cue table) — the app edits it via
  `vndub.stages.text_emotion.load_cues(path) -> dict` / `save_cues(path, data)`.
* `vndub.stages.translate.preview_import(ctx, path: str, n: int = 5) -> dict`
  → `{"format", "matched", "total", "preview": [{"line_id", "text_src", "text_tgt"}]}` (no writes).

### 2.3 casting, tuning, mode-B synthesis (task P3)
* Route `"P"` = performer voice only, no VC (used for characters without original audio). Allowed routes
  become `A`, `B`, `C`, `P`. Characters with no voiced lines get `route: "P"` by default.
* `Character.description` (free text, e.g. "строгая учительница, низкий голос") feeds the voice-design prompt.
* `characters.yaml.<sid>.performer.tuning = {pitch_st: -6..6, speed: 0.8..1.2, expressiveness: 0..1,
  energy: -1..1, timbre_preset: ""|"younger"|"older"|"softer"|"sharper", character_note: str}`; defaults
  `{0, 1.0, 0.5 (=passport as is), 0, "", ""}`. Per-line override: override action `tuning`
  (payload = tuning dict). `vndub.text.style_prompt.apply_tuning(spec: StyleSpec, tuning) -> StyleSpec`
  + `tuning_knobs(tuning, caps) -> dict` (backend-specific extras). Tuning is part of the synth cache key.
* Voice designs (in `vndub/stages/voicebank.py`):
  ```python
  list_designs(ctx, speaker_id) -> list[Design]
  propose_designs(ctx, speaker_id, n=3, description=None, sample_line_ids=None) -> list[Design]  # GPU; adds n
  add_reference_design(ctx, speaker_id, wav_path) -> Design      # user sample 5-20 s (wav/mp3)
  choose_design(ctx, speaker_id, design_id, route) -> dict        # writes characters.yaml performer + route
  auto_cast(ctx) -> dict[speaker_id, design_id]                   # top-ranked for everyone
  sample_lines(ctx, speaker_id, n=2) -> list[str]                 # translated lines used for samples
  import_design(ctx, speaker_id, design: dict, wav_paths) -> Design  # reuse a voice-library entry
  train_rvc(ctx, speaker_id) -> dict                              # route B training (job rvc_train)
  ```
  `Design = {"design_id", "kind": "voice_design"|"reference"|"clone", "backend", "prompt", "description",
  "samples": [{"line_id", "wav" (work-relative), "text"}], "spk_sim": float|None, "f0_delta_st": float|None,
  "stars": 1..5|None (None for characters without audio), "rank_score": float, "license", "commercial_ok",
  "created"}`; stored in `voicebank/<sid>/designs.json` + `voicebank/<sid>/designs/<design_id>/*.wav`.
  `kind: "clone"` = route A (zero-shot from the original clips), always offered for voiced characters.
* Synthesis of lines without audio: `synth`/`score`/`master` include speaker lines with
  `has_voice == False` when `synth.voice_unvoiced: true` (app sets it; CLI default false). Score renormalizes
  the composite over available terms (no spk_sim-to-original / emo_sim / prosody for them). Master writes
  them; pack places them (Ren'Py: generated `voice` statements file; other engines: `out/<engine>/unvoiced/`
  + INSTALL.md note — manual).
* Stars: `stars = clamp(round(1 + 4 * (spk_sim - thr_lo) / (thr_hi - thr_lo)), 1, 5)` with the calibrated
  threshold as the 3-star point (document the exact mapping in DECISIONS).

## 3. HTTP API (FastAPI, `127.0.0.1:<port>`, JSON; errors per `errors.py`)

All paths below exist; "→ Job" means the response is a `JobInfo` (section 4) and the result arrives via
the job's `result` when `state == "done"`. Audio URLs are always produced by the server
(`/api/audio/<slug>/<work-relative path>` or `/api/audio/<slug>/_out/<out-relative path>`), never built by the SPA.

### General / settings / system (server task S)
| Method | Path | Body → Response |
|---|---|---|
| GET | `/api/health` | `{ok, version, profile, mock: bool}` |
| GET/PUT | `/api/settings` | `Settings` + `{data_dir, paths}` (PUT partial). `paths` = `{data, models, envs, tools, python, uv_cache}`, the effective absolute folders (`python` / `uv_cache` are uv's CPython builds and download cache, always `<envs>/_uv/python` and `<envs>/_uv/cache`, so the cache and the venvs share a volume and uv can hardlink wheels). `models_dir` / `envs_dir`: `""` = the default (`<data>/models`, `<data>/envs`), else an absolute folder the app can create and write to (relative / not a string → 422 `bad_request`, not creatable or writable → `permission_denied` with action `open_data_dir`); nothing is moved. `envs_dir` is refused with 409 `job_busy` while any job is active, and takes effect at once (`VNDUB_ENVS_DIR` is re-exported for the pipeline and the installer); a `VNDUB_ENVS_DIR` set from outside the app (shell, tests) wins over it, and a dev checkout that already has `<repo>/.venvs` keeps it while no `envs_dir` is set (`paths.envs` always shows what is in use). `translation_llm`: `""` = the default (`gemma4_12b`), or the name of a non-mock `llm` backend (built in, or `hf_<slug>` of a model added by link; anything else → 422); a non-default value is written to every project's `pipeline.yaml` as `translate.llm.backend` |
| POST | `/api/settings/data-dir` | `{path}` → Settings |
| POST | `/api/system/pick-folder` | `{purpose}` → `{path: str|null, unsupported?: true}` (native dialog via pywebview; browser mode → unsupported, UI shows a path field) |
| POST | `/api/system/pick-file` | `{purpose, filters?: ["*.wav", ...]}` → `{path|null, unsupported?}` |
| POST | `/api/system/open` | `{kind: data_dir|logs|patch|backups|work|path, path?}` → `{ok}` |
| POST | `/api/system/report` | → Job, result `{path}` (zip: logs + stage_status.json + project.yaml + app_state.json, no audio) |
| POST | `/api/system/shutdown-pc` | internal (used when "Выключить компьютер по завершении" is set) |
| GET | `/api/help` | `[{id, title_key}]`; GET `/api/help/{id}` → `{markdown}` (bundled docs, `docs/APP.md` first) |
| GET | `/api/audio/{slug}/{path:path}` | audio file (wav/ogg/mp3), range requests OK; 404 code `file_not_found` |

### Project (S)
| Method | Path | Body → Response |
|---|---|---|
| GET | `/api/projects` | `[ProjectInfo-lite {slug, name, game_dir, engine, step, updated}]` |
| POST | `/api/project/open` | `{game_dir}` or `{slug}` → `ProjectInfo` (creates if new, becomes current) |
| GET | `/api/project` | `ProjectInfo {slug, name, game_dir, engine, project_yaml|null, state (app_state), stage_status, active_job: JobInfo|null, resume: {available, job: JobInfo|null, next_stage|null}}` (404 `no_project`) |
| PUT | `/api/project/state` | partial app_state (deep-merged) → app_state; regenerates pipeline.yaml |
| POST | `/api/project/resume` | → Job (re-submits the interrupted job's kind+params) |
| POST | `/api/project/import` | Step 1 → Job (detect, extract, parse, link no_asr, analyze, text_emotion) |
| POST | `/api/project/recover-text` | mode C → Job (link with `recover_text`) |
| POST | `/api/project/asr-check` | → Job (link with `mode: asr`) |
| GET | `/api/project/summary` | Step 1 summary: `{engine: {id, label, version, encrypted, drm[], warnings[]}, tools_missing: [{tool, url, pip?, manual_key?}], voice_files, lines_total, voiced, unvoiced, characters: [{speaker_id, name_src, name_tgt, lines, voiced_lines, unvoiced_lines, minutes, mode: voiced|unvoiced|mixed}], mapping: {mode, ok, review, unmatched, confidence, suggest_asr_check}, no_text_lines, problem_files: [{line_id, status, reasons[]}], analysis: {audio_passports, text_passports}, languages: {src: code|null, src_detected: bool, audio: code|null, tgt: code}}`. `languages.src` is the language of the script text (null before the import; `src_detected` is true when it came from the language detector, false when the user chose it), `audio` the language of the voice recordings when known (the UI shows it only when it differs from `src`), `tgt` the target language; codes: ja, en, zh, ko, ru, uk, de, fr, es, it, pt |
| POST | `/api/stages/{name}/run` | `{only?: [ids], limit?, force?, sets?: {dotted: value}}` → Job |

### Translation & settings — Step 2 (S)
| GET/PUT | `/api/project/translation` | `{tgt_lang, src_lang: auto|ja|en|zh|ko|ru|uk|de|fr|es|it|pt, src_lang_effective, src_lang_detected: {lang, confidence: 0..1, letters, lines}|null, source: auto|import, import_path, honorifics: drop|keep|adapt, address_default: ты|вы, license_policy, llm_available: bool, advanced: {n_candidates, routes: {A,B,C}, asr_check, backends: [{name, enabled, vram_gb, license, commercial_ok, role}]}}`. `src_lang` is the user's choice (`auto` by default: the pipeline detects the language of the script text; PUT accepts one of the listed codes); `src_lang_effective` is the code the pipeline uses right now (the detected one when the choice is `auto`, else the choice); `src_lang_detected` is the detection record over the parsed lines (`lang`, `confidence`, number of `letters` and `lines` it is based on), null before the game was imported or when nothing could be detected. When the source language equals `tgt_lang` the lines are kept as they are and the translation model is not started. `tgt_lang` stays as before. The same `src_lang` (default `auto`) is kept in `app_state.translation` |
| POST | `/api/project/translation/preview-import` | `{path}` → `translate.preview_import` result |
| GET/PUT | `/api/project/glossary` | `{characters: [{speaker_id, src, tgt, gender: m|f|unknown}], terms: [{src, tgt, note}]}` (writes characters.yaml names/gender + project glossary.yaml) |
| POST | `/api/project/translate` | → Job (translate, text_emotion) |
| GET/PUT | `/api/project/delivery-cues` | cue table (text_emotion.load_cues/save_cues) |

### Casting — Step 3 / 3b (S, calling P3 functions)
| GET | `/api/characters` | `[Character {speaker_id, name_src, name_tgt, gender, age_hint, description, lines, voiced_lines, minutes, mode, emotion_hist: {label: share}, clips: [{kind: neutral|emotional|longest, line_id, url, text_src, emotion}], chosen: {design_id, route, kind}|null, tuning, rvc: {state: none|queued|training|done|failed, progress}}]` sorted by line count |
| PUT | `/api/characters/{sid}` | `{gender?, age_hint?, description?, name_tgt?}` → Character |
| GET | `/api/casting/{sid}/proposals` | `[Design + samples[].url + samples[].text_tgt]` |
| POST | `/api/casting/{sid}/generate` | `{n: 3, description?}` → Job, result `{designs: [...]}` |
| POST | `/api/casting/{sid}/upload` | multipart `file` (wav/mp3, 5–20 s; else `sample_too_short`/`sample_too_long`/`sample_bad_format`) → Design |
| POST | `/api/casting/{sid}/choose` | `{design_id, route}` → Character (starts RVC training job for route B in background) |
| POST | `/api/casting/auto` | `{n?}` → Job `casting_auto` (designs for characters that have none, then the top-ranked voice for everybody, route-B training queued), result `{chosen: {sid: design_id}, generated: [sid]}` |
| POST | `/api/casting/{sid}/listen` | `{n: 3}` → Job, result `{items: [{line_id, text_src, text_tgt, original_url|null, result_url}]}` |
| GET | `/api/voices` | library `[{id, name, gender, description, sample_url, source_project}]` |
| POST | `/api/voices` | `{speaker_id, name}` → library entry (saves the chosen design) |
| POST | `/api/casting/{sid}/from-library` | `{voice_id}` → Design |
| GET/PUT | `/api/characters/{sid}/tuning` | tuning dict |
| POST | `/api/characters/{sid}/tuning/listen` | `{tuning}` → Job, result `{items: [{line_id, kind: neutral|emotional, url, text_tgt}]}` (target ≤ 10 s) |
| POST | `/api/characters/{sid}/tuning/apply-similar` | → `{updated: [sid]}` (same gender + age_hint) |

### Preview — Step 4, full run — Step 5 (S)
| POST | `/api/preview/run` | `{mode: first_scene|first_n, n: 40}` → Job (translate, text_emotion, voicebank, synth, score, master — only those lines) |
| GET | `/api/preview/items` | `{line_ids, items: [{line_id, speaker_id, name, text_src, text_tgt, original_url|null, dub_url|null, status}]}` |
| POST | `/api/run/full` | `{shutdown_when_done: bool}` → Job (translate … report, regen loop) |
| GET | `/api/gpu` | `{name, total_gb, used_gb}` or `{available: false}` |

### Review — Step 6 (S)
| GET | `/api/review/lines` | query `speaker, scene, status (ok|check|failed), q, min, max` → `{items: [{line_id, speaker_id, name, scene, status: ok|check|failed, composite, text_src, text_tgt, winner_label, flags[], delivery: {source: audio|text, layer}|null, touched: bool}], facets: {speakers, scenes}}` — `check` first |
| GET | `/api/review/lines/{id}` | `{line, original_url|null, candidates: [{id, label: A|B|C…, url, backend, route, composite, passed, is_winner, verdicts: [Verdict]}], why: [{gate, key, params}], style_prompt, text_tts, refs: [{path, url, emotion}], tuning_override|null}` |
| POST | `/api/review/lines/{id}/action` | `{action: accept|reject|regen|set_text|set_ref|keep_original|set_src_text|tuning, payload}` → `{ok, touched}` |
| POST | `/api/review/prefer-route` | `{route: "A", speaker_id?}` → `{updated: n}` (accept that route's candidate where it passed its gates) |
| POST | `/api/review/apply` | → Job (synth, score, master for touched lines), result `{changed, lines}` |

`Verdict = {metric: voice|emotion|clarity|length|quality, key, level: good|ok|bad, value, tooltip_params}`.
Keys (i18n `verdict.<metric>.<key>`): voice `similar|close|different`; emotion `close|partly|far`;
clarity `good|ok|poor`; length `ok|longish|too_long|shortish|too_short`; quality `good|ok|poor`.
Mapping lives in `vndub_app/verdicts.py` (unit-tested) and uses the same thresholds as the score gates.
`why[].key` (i18n `why.<key>`): `cer`, `utmos`, `spk_sim`, `dur_ratio`, `text_from_asr`, `no_candidates`, `override`.

### Doctor, installer, models, patch install (task O)
| GET | `/api/doctor` | `{summary: ok|warn|fail, gpu: {name, vram_gb, driver, cuda}|null, rows: [{id, status: green|yellow|red, value, params, fix: {action, label_key}|null}]}` — row ids: `gpu`, `driver`, `vram`, `disk`, `ffmpeg`, `vcredist`, `git`, `internet`, `defender`, `envs`, `models` (i18n `doctor.<id>.title`, `doctor.<id>.<status>`, and with `params.reason` also `doctor.<id>.<status>.<reason>`). `vcredist` (Windows only): green when the Microsoft Visual C++ 2015-2022 runtime is 14.40+ (HKLM 64-bit view of `SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64`, else `msvcp140.dll` + `vcruntime140_1.dll` in System32), red (`reason` `missing` / `too_old`) otherwise, with fix `install_vcredist` (runs the bundled `redist\vc_redist.x64.exe`) or, without that file (a dev run), `open_vcredist_page`; `git`: green (`params.source` `system` or `bundled`: PATH first, then `<exe dir>\git\cmd\git.exe`), yellow with `params.envs` (the envs whose installation clones repositories) otherwise |
| POST | `/api/doctor/fix/{action}` | `download_ffmpeg | install_vcredist | open_data_dir | open_driver_page | open_vcredist_page | add_defender_exclusion_help` → Job (`download_ffmpeg` → `ffmpeg_download`, `install_vcredist` → `vcredist_install`, result `{exit_code, reboot_required}`) or `{ok, ...}` (`url` for the two `open_*_page` actions: the SPA opens it) |
| GET | `/api/tools` | `[{name, version, present, required_for: [engines], size_mb, url, filename, sha256, manual: bool}]` (a tool the packaged app bundles, manifest `bundled`, is `present` with `path` next to `VNDub.exe` unless the tools folder holds an own copy) |
| POST | `/api/tools/install` | `{names}` → Job (per-file progress in `result.files`; unreachable URL → error `download_unreachable` params `{url, filename}` action `show_manual`) |
| GET | `/api/envs` | `[{name, present, size_gb, torch: cuda|cpu|null}]`; POST `/api/envs/install` `{envs, torch: auto|cuda|cpu}` → Job (first installs CPython 3.11 with uv into `<envs>/_uv/python` and hands that interpreter to `scripts/make_envs.*`; a Windows refusal to follow uv's link folder, os error 448, is the error `python_junction_blocked` with `params.path` = that folder) |
| GET | `/api/models` | `{rows: [{name, role, env, size_gb, vram_gb, license, commercial_ok, present, usable, reason, presets: [min|recommended|all], selftest: {ok, peak_vram_gb, at}|null}], presets: {min: [..], recommended: [..], all: [..]}, disk: {free_gb, data_dir}}` (i18n `models.<name>.name`, `models.<name>.purpose`) |
| POST | `/api/models/download` | `{names}` → Job (resume-capable, per-file progress in `result.files`) |
| POST | `/api/models/check` | `{names}` → Job (selftest; result rows with `peak_vram_gb`) |
| POST | `/api/models/hf/inspect` | `{url}` → `{repo, revision, sha, page_url, files: [{name, size}], total_size, license|null, gated, private, kind: builtin|gguf_llm|rvc|files, suggested: [names], suggested_size, builtin: <backend name>|null, warnings: [{code, params}], free_gb, focus}`. `url` = `https://huggingface.co/<owner>/<repo>` (also `/tree/<rev>[/<dir>]`, `/blob|resolve/<rev>/<path>`, `hf.co/...`, no scheme, `?download=true`, or a bare `owner/repo`); any other host → 422 `hf_bad_url`; unknown repo → 404 `hf_not_found`; gated and not accessible → 403 `hf_gated` (`params.page_url`); no network → `download_unreachable`. Warning codes (i18n `models.hf.warn.<code>`): `gated`, `license_check`, `unknown_license`, `too_big`, `not_text_model` |
| POST | `/api/models/hf/add` | `{repo, revision?, files: [names], kind: gguf_llm|rvc|files, label?}` → Job `models_hf_add` (`gpu=false`, `params.id` = the entry id; per-file rows in `result.files` like `models_download`; `result.entry` when done). Downloads into `<models_dir>/custom/<owner>__<repo>/`, then records `{id: "hf_<slug>", repo, revision, files, kind, dir, size_gb, license, label, added_at}` in `<data>/custom_models.json`. Resumable (finished files are skipped, a half-downloaded file continues from its `.part`); a second add of the same repo while one runs → 409 `job_busy` |
| GET | `/api/models/custom` | `[entry + {present, disk_gb, usable|null, in_use}]` (`usable` is the registry's verdict for `gguf_llm`) |
| DELETE | `/api/models/custom/{id}` | `{ok}`; removes the files and the entry (and clears `Settings.translation_llm` when it was that model); 404 `bad_request` for an unknown id; 409 `bad_request` with `params.reason: "in_use"` while a download of it or a pipeline job that translates with it is active |
| GET | `/api/models/hf/recommended` | `{items: [{repo, url, kind, file, size_gb, license, title: {ru, en}, note: {ru, en}}], checked}`, served from `app/recommended_models.json` (edit it without rebuilding the SPA) |
| GET | `/api/install/plan` | `{engine, auto_possible, manual_reason_key|null, install_md, files: [{target, source, exists}], patch_dir, backups: [{id, created, files}]}` |
| POST | `/api/install/apply` | `{backup: true}` → Job (`install_manual_only` error when not automatic — never claims success) |
| POST | `/api/install/rollback` | `{backup_id}` → Job |
| POST | `/api/install/zip` | → Job, result `{path}` |

## 4. Jobs and progress

`JobInfo` = `JobRecord.info()`: `{id, kind, title_key, params, project, gpu, state: queued|running|paused|done|failed|cancelled|interrupted, created, started, finished, stages[], stage, stage_index, stage_count, done, total, item, eta_s, stage_eta_s, rate_per_min, message_key, message_params, error: ErrorInfo|null, result}`.
`GET /api/jobs?project=&active=1`, `GET /api/jobs/{id}` (+ `log_tail`), `POST /api/jobs/{id}/pause|resume|cancel`.
`WS /api/progress` sends `{"type": "job", "job": JobInfo}`, `{"type": "log", "job_id", "line"}`,
`{"type": "gpu", "used_gb", "total_gb"}` (every 2 s while a GPU job runs; omitted without NVIDIA) and
`{"type": "hello", "jobs": [JobInfo active]}` on connect.
Job kinds (i18n `jobs.<kind>`): `import, recover_text, asr_check, translate, stage, casting_generate, casting_auto,
casting_listen, tuning_listen, rvc_train, preview, full, review_apply, report, models_download,
models_check, models_hf_add, tools_install, envs_install, install_apply, install_rollback, install_zip, ffmpeg_download, vcredist_install`.
Stage names for progress labels: i18n `stages.<name>`.

## 5. i18n

* One table per language: `app/i18n/ru.json` (DEFAULT) and `app/i18n/en.json`, flat dotted keys
  (`"errors.tool_missing.title": "..."`), `{param}` interpolation. Owned by the SPA task.
* Other tasks never hard-code UI text; they emit codes. If they need new keys (error codes, model names,
  doctor rows, cue labels) they write `app/i18n/parts/<task>.ru.json` + `.en.json`; `scripts/merge_i18n.py`
  merges parts into the main tables (the parts are deleted after the merge).
* Error keys: `errors.<code>.title` (what happened, one sentence) + `errors.<code>.hint` (one thing to do);
  buttons `actions.<action>`. Codes used so far: `internal, bad_request, cancelled, tool_missing,
  weights_missing, env_missing, backend_unavailable, stage_failed, file_not_found, permission_denied,
  project_not_found, no_project, job_not_found, job_busy, sample_too_short, sample_too_long,
  sample_bad_format, download_unreachable, checksum_mismatch, install_manual_only, no_gpu, low_vram,
  disk_full, game_dir_invalid, nothing_to_review, nothing_to_install, hf_bad_url, hf_not_found, hf_gated, vcredist_failed,
  python_junction_blocked`.

## 6. Tests

* Python: `uv run pytest` (repo root `B:\VN-Dub-Research\vndub`; uv at `C:\Users\user\.local\bin\uv.exe`).
  App tests live in `tests/app/` and always set `VNDUB_DATA` to `tmp_path`, `VNDUB_APP_PROFILE=apptest`,
  copy `tests/fixtures/renpy_mini` into `tmp_path` and call `jobs.reset_runner(JobRunner(None, None))`.
* SPA: `npm run build` in `vndub_app/web` → `vndub_app/web/dist` (committed build output is not required;
  the server serves it when present and shows a plain "SPA not built" page otherwise).
* Playwright (Python, `tests/app/test_wizard_e2e.py`, marker `e2e`): walks Steps 1→7 on the fixture in
  apptest mode; screenshots to `docs/img/app/*.png`. `tests/app/test_models_hf_e2e.py` drives the "Add a model by link"
  card with fake Hugging Face clients (`08b-models-hf.png`).
* Real Hugging Face (small downloads, marker `network`, skipped by default):
  `VNDUB_NETWORK_TESTS=1 uv run pytest tests/app/test_hf_network.py -m network -s`.

## 7. As built

The implementation follows this contract with backward-compatible additions (extra fields such as
review `counts`/`total`, summary `no_text_sample`/`recover_offer`, doctor `params.reason` variants,
tool/model row extras, `GET /api/progress/sse` as an SSE twin of the WebSocket, `?slug=` on the install
routes). They are listed with their rationale in `docs/DECISIONS.md`, section "Desktop app". Review
action payloads: accept/reject `{candidate: <candidate id = work-relative wav>}` (or `{label}`), regen
`{text, style_prompt}`, set_ref `{ref: <refs[].path>}`, set_src_text `{text}`, tuning = tuning dict.

Models by Hugging Face link (`vndub_app/hf_custom.py`): file bytes of every Hugging Face download (`models_download` too) are
streamed by `vndub_app/downloads.py` after `huggingface_hub` has resolved the pre-signed link, size and sha256
(`models_api.HfHub.fetch`), so a cancel acts within a second and a half-downloaded file continues from `<file>.part`;
`huggingface_hub` 2.x does not resume across calls and `hf_xet` swallows exceptions raised by progress callbacks.
`downloads.fetch(identity=...)` keeps a `.part` valid although the signed URL changes. Details: `docs/DECISIONS.md`, "HF links".

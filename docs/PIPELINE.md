# Pipeline: stages, inputs/outputs, config keys

```
detect -> extract -> parse -> link -> analyze -> translate -> voicebank -> synth -> score -> master -> pack -> report
                                                                              ^         |
                                                                              +- regen -+        review (UI) -> overrides.jsonl
```

Every stage: `vndub <stage> --project <dir> [--config pipeline.yaml] [--force] [--only id,...] [--limit N] [--work-dir] [--out-dir] [--set key=value]`.
`vndub run` executes all stages in order, including up to `score.regen_rounds` rounds of
`synth --regen` + `score` for lines whose candidates failed the gates.

Each stage logs to `work/logs/<stage>.log`, records `started/finished/hash/summary/error`
in `work/stage_status.json` and prints a one-screen summary. Item-level results are cached
in `work/cache/<stage>.json` by `sha256(stage version, relevant config section, inputs,
content hashes of input files)`, so reruns recompute only what changed and a crash loses at
most the items in flight. `--force` clears the stage's cache.

## Principles

- **Manifests are the source of truth** (`work/**/*.jsonl|yaml|json`). `work/index.sqlite`
  is a rebuildable cache (`vndub index rebuild`).
- **Additive fields:** a stage writes its own files and never rewrites fields of a file it
  does not own. `link` does not edit `lines.jsonl`; later stages apply its rebindings via
  `StageContext.voiced_lines()`.
- **One heavy model at a time:** stages call `ctx.models.acquire(name)` once per model and
  process all items under it. Acquiring another heavy model unloads the resident one
  (adapter `unload` + process exit). The registry checks `vram_estimate_gb` against
  `gpu.vram_budget_gb - gpu.safety_margin_gb` and the free memory reported by `nvidia-smi`.
- **No silent downgrades:** an unavailable backend is replaced only by its configured
  fallback; if none works, the affected lines are marked `review` with the reason. The
  test profile alone may substitute mocks, and every substitution is logged and reported.
- **Offline:** adapters run with `HF_HUB_OFFLINE=1`; weights come only from `models/`.

## Backend adapters

`vndub/backends/registry.py` maps a backend name to a `BackendSpec` (role, env, adapter key,
license, commercial flag, VRAM estimate, weights, fallback, mock, capabilities). Remote
backends run `adapters/<env>/main.py` inside `.venvs/<env>` and speak JSON-RPC over stdio:

```
-> {"jsonrpc":"2.0","id":7,"method":"load|run|unload|selftest|peak_vram|pull|shutdown","params":{...}}
<- {"jsonrpc":"2.0","id":7,"result":{...}}      or  {"error":{"code":-32001|-32002|-32000,"message":...}}
```

`-32001` = weights missing ("run `vndub models pull <name>`"), `-32002` = unavailable
(import error, no CUDA, missing binary). Audio always travels by file path. A crashed
adapter is restarted once and the call retried; timeouts kill the process. Role operations:

| role | op | item -> result |
|---|---|---|
| asr | transcribe | {id, wav} -> {text} |
| align | align | {id, wav, text} -> {chars:[{c,t0,t1}]} |
| ser | emotion | {id, wav, out?} -> {probs, label, embedding_path?} |
| adv | adv | {id, wav} -> {arousal, dominance, valence} |
| events | events | {id, wav} -> {events:[...], raw} |
| embedder | embed | {id, wav, out} -> {path} (.npy) |
| f0 | f0 | {id, wav, out} -> {path, hop_s} (Hz, 0 = unvoiced, 10 ms hop) |
| vad | vad | {id, wav} -> {segments:[[t0,t1]]} |
| stress | stress | {id, text} -> {text} (fmt plain/plus, yo) |
| caption | caption | {id, wav} -> {caption} |
| mos | mos | {id, wav} -> {mos} |
| aesthetics | aesthetics | {id, wav} -> {pq, ce, cu, pc} |
| tts | synthesize / design | {id, text, lang, timbre_ref, timbre_ref_text, emotion_ref, instruct, knobs, seed, target_duration, out} -> {path, sr, duration} |
| vc | convert / train | {id, src, out, target_ref, model, index, f0_transpose, params} -> {path}; train(job) -> {model, index} |
| llm | chat | messages -> content string (JSON mode) |

## Stages

### detect
In: `--project` (game dir, or a dir with a hand-written `project.yaml` containing `game_dir`).
Out: `work/project.yaml` (`ProjectYaml`: engine, engine_version, voice_codec {container,
codec, sr, ch, bitrate_k}, encrypted, drm, archives, warnings). Engine by file signatures
(see ENGINES.md); `--engine` forces one. Encryption and DRM markers produce warnings.
Config: `parse.engine` (auto = trust detect).

### extract
In: game dir, `project.yaml`. Out: `work/raw/` (extracted archives/scripts),
`work/raw/extract.json`, `work/audio/src/<id>.wav` (original sr, mono),
`work/audio/src_stereo/<id>.wav` (stereo originals), `work/audio/src16/<id>.wav` (16 kHz
mono), `work/audio_index.jsonl` (`AudioIndexRecord`: id, relpath, src_path, duration, sr,
ch, codec, bytes, wav, wav16, error). Decoding via ffmpeg, `vgmstream-cli` for proprietary
formats. Config: `extract.*`, `tools.*`.

### parse
In: extracted scripts + `audio_index.jsonl`. Out: `work/manifest/lines.jsonl`
(`LineRecord`: line_id, scene, order, speaker_raw, speaker_id, text_src, voice_file,
has_voice, audio_id, src_wav, src_wav16, context_prev, choices, flags, raw, source_file,
source_line) and `work/manifest/characters.yaml` (user edits are preserved on re-parse).
`voice sustain` (Ren'Py): the head line owns the file (`flags: [sustain_head]`,
`sustain_lines: [...]`), continuation lines carry `flags: [voice_sustain]`.
Config: `parse.engine_opts.<engine>.*`, `parse.import_vntextpatch`, `parse.narrator_names`.

### link
In: lines, `src16` audio. Out: `work/link.jsonl` (`LinkRecord`: line_id, audio_id,
asr_text, match_score, best_alt_line_id, best_alt_score, spk_cluster, spk_consistency,
status ok|review|unmatched, reasons, relinked_from), `work/review_queue.jsonl`,
`work/analysis/spk/<audio_id>.npy`.
1. ASR (`link.asr`, fallback `link.asr_fallback`): anime-whisper via faster-whisper, no
   initial prompt, `no_repeat_ngram_size=5`, no conditioning on previous text.
2. NFKC + punctuation/ruby stripping, RapidFuzz normalized similarity against the bound
   line and +-`neighbour_window` neighbours; `>= accept_score` ok, `[review_score,
   accept_score)` review, else a global top-`global_topk` search (blocked by speaker);
   `auto_repair` moves a file to a clearly better line.
3. Speaker embeddings (`link.speaker_embedder`), agglomerative clustering
   (`cluster_distance`), `spk_consistency` = cosine to the speaker centroid;
   `< outlier_consistency` -> review.

### analyze
In: voiced lines. Out: `work/analysis/<line_id>.json` (the expression passport, schema
`manifest.io.Passport`), `analysis/emb/<line_id>.npy` (emotion2vec+ embedding),
`analysis/feat/<line_id>.{f0,energy}.npy`, updated `characters.yaml` stats
(n_lines, total_sec, f0_median_hz, f0_iqr_st, energy_db_median, lufs_median, rate_median,
emotion_hist) and `ref_candidates`. Models run grouped by model (`analyze.e2v`,
`anime_ser`, `adv`, `events`, `aligner`, `f0`, `vad`, optional `caption`). A second pass
fills `passport.style` (StyleSpec) relative to the character medians.

Passport example:
```json
{"line_id":"s01_000123","duration_s":2.84,"snr_db":31.2,"bgm":false,
 "emotion":{"e2v":{"label":"happy","probs":{"happy":0.71,"neutral":0.1,"...":0},"embedding_path":"analysis/emb/s01_000123.npy"},
            "anime":{"label":"Happy","probs":{}},"adv":{"arousal":0.63,"dominance":0.55,"valence":0.78}},
 "events":["laughter"],
 "prosody":{"f0_median_hz":271,"f0_std_st":3.1,"f0_range_st":9.4,"energy_db_mean":-21.3,"rate_mora_per_s":7.9,
            "pauses":[{"t":1.12,"d":0.31}],"leading_silence_s":0.08,"trailing_silence_s":0.21,"arc":"rising_end",
            "f0_contour_path":"analysis/feat/s01_000123.f0.npy","energy_contour_path":"analysis/feat/s01_000123.energy.npy","lufs":-22.4},
 "align":{"chars":[{"c":"本","t0":0.41,"t1":0.52}]},
 "caption":null,
 "style":{"emotion":"happy","intensity":0.63,"tempo":"fast","loudness":"normal","events":["laughter"],"arc":"rising_end"},
 "backends":{"e2v":"emotion2vec_plus_large"},"missing":{}}
```

### translate
In: lines, passports (emotion per line), `configs/glossary.yaml`. Out:
`work/translation/lines.<tgt>.jsonl` (`TranslationRecord`: text_tgt, text_tgt_short,
text_tts, text_tts_short, text_plain (stress-free CER reference), syllables, notes,
source import|llm:<model>|mock, glossary_hits, flags).
- `translate.import`: VNTextPatch JSON/XLSX dir or a line_id -> text map; skips the LLM.
- Else `llama-server` with `translate.llm.backend` (Gemma 4 12B IQ4_XS; fallbacks), per scene,
  batches of `batch_lines` as JSON `[{id, speaker, gender, emotion, text}]` with the last
  `context_window_lines` translations and a scene summary; system prompt = glossary (names,
  terms present in the batch, honorific policy, ты/вы matrix, genders) + rules. Output JSON
  `{"lines":[{"id","ru","ru_short","notes"}]}`, validated strictly; retry at temperature 0,
  then halve the batch.
- Interjections (`configs/interjections.yaml`) map to natural equivalents.
- Post-processing: ruaccent stress (+ yo), numbers -> words, abbreviations expanded.
- `--seed-glossary` writes/extends `configs/glossary.yaml` from `characters.yaml`.

### voicebank
Out: `work/voicebank/<speaker_id>/refs/*.wav` + `refs.json` (`RefClip`: path, line_id,
emotion, arousal, valence, snr, duration, usage timbre|emotion|both, text),
`performer.yaml` + `performer/*.wav`, `rvc/` (model, index, `meta.json` with
`f0_transpose`), `lora/`, `synthetic/`, and `work/voicebank/spk_calibration.json` (EER
threshold of same- vs different-character cosine). Config: `voicebank.refs.*`,
`voicebank.performer.*` (voice design from gender/age/pitch/emotion profile, picked by ECAPA
similarity + F0 within `max_f0_delta_semitones`), `voicebank.rvc.*` (Applio training with
`>= min_minutes` of clean audio), `voicebank.route_c.*`.

### synth
Out: `work/synth/<line_id>/<backend>_<route>_<k>.wav` + `.json` sidecar (`SynthSidecar`:
backend, route, k, seed, params, style_prompt, style, ref_paths, text_used, text_variant,
elapsed_s, peak_vram_gb, pre_vc_wav, round).
- Style: `text/style_prompt.py` turns the passport into a `StyleSpec` (emotion = argmax of
  0.6 e2v + 0.4 anime SER, intensity = arousal, tempo/loudness vs the character median, arc,
  events) and renders it per backend capability: natural-language `instruct` (Qwen3-TTS
  CustomVoice), CosyVoice3 instruct before `<|endofprompt|>`, VoxCPM2 parenthesized style,
  Fish S2 / Higgs inline tags, Chatterbox `exaggeration`/`cfg_weight`, Zonos2 sliders.
- Route A: the character's own (Japanese) emotion-matched reference, `x_vector_only_mode`
  on Qwen3-TTS, `cfg_weight=0` on Chatterbox, explicit target language.
- Route B: the native performer voice + emotion instruction, then VC into the character
  (RVC with `f0_transpose`, index_rate 0.5, protect 0.33, rmvpe; Seed-VC zero-shot when no
  RVC model). The pre-VC wav is kept.
- Route C: per-character LoRA (when trained).
- N = `synth.n_candidates` (`n_candidates_short` for interjections, short and high-arousal
  lines); `text_tgt_short` after a duration ratio above `use_short_text_if_ratio_above`;
  the original audio is added as an `orig` candidate for interjections when
  `keep_original_for_interjections`. `--regen` re-synthesizes only lines marked `regen`.

### score
Out: `work/scores/<line_id>.jsonl` (`CandidateScore` with every metric persisted),
`work/selection.jsonl` (`SelectionRecord`: winner, scores, status auto|regen|review|override,
reason, round). Metrics (grouped by model): UTMOSv2 (`score.mos`, fallback UTMOS22),
Audiobox PQ (`score.aesthetics`), speaker similarity (ECAPA max cosine to the character's
timbre refs), emotion similarity (cosine of emotion2vec+ embeddings, same model as analyze)
and |dA| + |dV|, CER (GigaAM-v3, fallback Whisper-ru) against the stress-free target text,
duration ratio, F0 and energy correlation after DTW (100 points), pause-pattern Jaccard.
Composite = weighted sum (`score.weights`, renormalized over available metrics). Gates:
`score.gates.*` (spk_sim from the EER calibration). Best gate-passing candidate -> `auto`;
otherwise `regen` (up to `regen_rounds`) then `review`. Overrides are honoured.
`vndub score --reweight` recomputes composites and selection from persisted metrics only.

### master
Out: `work/final/<line_id>.wav`, `work/final/encoded/<voice_file>`, `work/final_index.jsonl`.
Trim to the original's silence profile, stretch (Rubber Band, ffmpeg `atempo` fallback)
only outside `master.stretch.target_window` and within `max_stretch`, loudness-normalize to
the character's median LUFS with a true-peak limiter at `true_peak_dbtp`, resample to the
source sr/channels, encode to the source codec/bitrate, same file name.
`master.include_review_lines: provisional` also masters review lines (marked provisional).

### pack
Out: `out/<engine>/...` + `INSTALL.md` + `manifest_patch.json` (file, sha256, line_id,
backend, route). Ren'Py: loose `game/<voice>`; NScripter: loose files beside the exe;
KiriKiri: `patchN.xp3` (Xp3Pack or the built-in unencrypted writer) or `unencrypted/`;
Unity: UnityPy replacement script + checklist (experimental); others: directory layout +
tool command lines. See ENGINES.md.

### report
Out: `work/report/index.html`, `work/report/report.md`: mapping coverage, per-character
stats and emotion histograms, score distributions per backend/route, A/B/C comparison,
review queue with reasons, backend substitutions, stage timings, peak VRAM per model.

### review (UI)
`vndub review --project <dir>`: Gradio app on `review.host:review.port`; writes
`work/overrides.jsonl` (`OverrideRecord`: line_id, action accept|reject|regen|set_text|
set_ref|keep_original, payload, ts).

## Config sections (configs/pipeline.yaml)

| Section | Keys |
|---|---|
| top level | `profile`, `extends`, `fallback_to_mock` (test only), `license_policy` research/commercial, `tools_file`, `tools` |
| `project` | `game_dir`, `work_dir`, `out_dir`, `models_dir`, `src_lang`, `tgt_lang` |
| `gpu` | `vram_budget_gb`, `safety_margin_gb`, `device`, `allow_cpu_fallback` |
| `extract` | `keep_stereo_copy`, `analysis_sr`, `workers`, `voice_exts`, `vgmstream_exts`, `voice_dirs` |
| `parse` | `engine`, `engine_opts.{kag,nscripter,renpy,tyrano,generic}`, `import_vntextpatch`, `narrator_names` |
| `link` | `asr`, `asr_fallback`, `asr_compute`, `accept_score`, `review_score`, `neighbour_window`, `global_topk`, `auto_repair`, `speaker_embedder`, `cluster_distance`, `outlier_consistency` |
| `analyze` | `e2v`, `anime_ser`, `adv`, `events`, `aligner`, `aligner_fallback`, `f0`, `vad`, `caption`, `caption_backend`, `ref_candidates_per_emotion`, `e2v_weight`, `anime_weight` |
| `translate` | `import`, `llm.{backend,fallbacks,ctx,n_gpu_layers,temperature,port,max_retries,real_backend}`, `batch_lines`, `context_window_lines`, `glossary`, `short_variant_min_words`, `stress.{tool,model,yo,format}`, `interjections`, `mock_dictionary` |
| `voicebank` | `refs.*`, `performer.*`, `rvc.*`, `route_c.*`, `calibrate_spk`, `min_minutes_for_voicebank` |
| `synth` | `n_candidates`, `n_candidates_short`, `routes`, `backends.<name>.{enabled,...}`, `vc.{primary,fallback,rvc,seedvc}`, `keep_original_for_interjections`, `use_short_text_if_ratio_above`, `interjection_max_syllables`, `high_arousal`, `seed_base` |
| `score` | `weights`, `gates.{utmos_min,cer_max,dur_ratio,spk_sim_min,spk_sim_default}`, `mos`, `mos_fallback`, `aesthetics`, `speaker`, `speaker_extra`, `cer_asr`, `cer_asr_fallback`, `emotion`, `adv`, `regen_rounds`, `interjection_cer_max` |
| `master` | `stretch.{enabled,target_window,max_stretch,engine}`, `loudness.{target,true_peak_dbtp,default_lufs}`, `match_silence_profile`, `include_review_lines` |
| `pack` | `kirikiri`, `renpy`, `nscripter`, `unity`, `other` |
| `review` | `host`, `port`, `batch_accept_threshold` |
| `report` | `compare_routes`, `markdown` |

`configs/pipeline.test.yaml` (`extends: pipeline.yaml`) is the CPU/tiny profile used by the
fixture run and CI.

# Pipeline: stages, inputs/outputs, config keys

```
detect -> extract -> parse -> link -> analyze -> translate -> text_emotion -> voicebank -> synth -> score -> master -> pack -> report
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
Mixed games are normal: the mode is per line (`has_voice`, presence of `text_src`), not per game;
an unvoiced protagonist is a character with `stats.n_lines == 0` and unvoiced lines.
Mode C (voice files but no script text, see `link`): with `link.recover_text: true` a script that
yields no lines (the generic engine without a text import, an unparsed/encrypted script) no longer
stops parse; it writes one voiced line with empty `text_src` per voice file (`flags: [voice_only]`,
speaker = file prefix, scene = parent folder). With the option off the engine's error is raised as before.
Voiced lines shown without a speaker name (`parse.hidden_speakers: auto`, the default): a voiced line in
the narrator box (an off-screen voice, a voiced narrator) or under a name that hides the speaker
(punctuation only: `？？？`, `…`) goes to the character its voice-file prefix names, when prefixes identify
speakers in this game (`sak_0012` -> the speaker of most named `sak_` lines; same purity test as link's
prefix check; flag `speaker_from_voice`). Voiced narrator-box lines that no prefix names go to the
`voice_over` speaker (flag `offscreen_voice`; `characters.yaml` name `tgt` = "Голос за кадром"), which is
cast, translated and dubbed like any character; a `？？？` line no prefix names keeps its own speaker.
Unvoiced narration is never touched and never dubbed. `off` keeps the script's speaker (a voiced narrator
then has no performer and only route A).
Config: `parse.engine_opts.<engine>.*`, `parse.import_vntextpatch`, `parse.narrator_names`, `parse.hidden_speakers`.

### link
In: lines, `src16` audio. Out: `work/link.jsonl` (`LinkRecord`: line_id, audio_id,
asr_text, match_score, best_alt_line_id, best_alt_score, spk_cluster, spk_consistency,
status ok|review|unmatched, reasons, relinked_from; extras `signals` (no_asr) and
`text_recovered` (mode C)), `work/review_queue.jsonl`, `work/analysis/spk/<audio_id>.npy`.

The script text is the source of truth; the original audio is only an intonation reference.
`link.mode` selects how the line <-> voice mapping is verified:

* `asr` (CLI default): the steps below. `link.recover_text` aside, behaviour is unchanged.
* `no_asr` ("intonation-check only", the desktop app's default): **no ASR backend is acquired**
  (not even the mock; `work/gpu_events.jsonl` shows no `role: asr` load). Cheap signals only:
  1. speaker-embedding consistency (`link.speaker_embedder`, steps 3 below; reasons
     `spk_outlier:<cosine>` below `outlier_consistency`, `spk_cluster:<speaker>` when the clip
     clusters with another speaker),
  2. duration vs text length: morae of the (sustain-extended) text (`text/ja.count_morae`) against the
     file duration; `ratio = duration / (morae / sqrt(rate_min * rate_max))`, 1.0 = typical rate;
     implausible when shorter than `morae / rate_max` or longer than `max(morae / rate_min, 1.5 s) +
     silence_tolerance_s` (reason `duration_implausible:<ratio>`; defaults 3 and 13 mora/s, 1.0 s),
  3. voice-file prefix / index order for names like `sak_0012`: inside a scene the indexes of one
     prefix must not run backwards (`order_gap`, on both lines of the inversion) and a speaker's file
     prefix must be the one in `characters.yaml` `voice_prefix` (several allowed, `a, b`) or the
     speaker's majority prefix (`prefix_mismatch:<prefix>`); only when prefixes identify speakers
     in the game (>= 80 % purity), never for the narrator, never for hashed names.

  A line is `ok` when every available signal agrees, `review` otherwise (reasons above, also in
  `review_queue.jsonl` with the `signals` dict `{dur_ratio, morae, voice_prefix, voice_index, no_text}`),
  `unmatched` only when the audio is missing (`audio_missing:<ref>`: the decoded clip is gone or the
  script references a voice file that does not exist). No auto-repair: nothing is moved, orphan files
  stay unbound. Summary adds `mode`, `confidence` (share of judged lines with all signals ok, 0..1),
  `outliers` (review + unmatched), `suggest_asr_check` (true when >= 5 lines were judged and the
  outlier share is at least `suggest_asr_share`, default 0.2: a healthy game flags a few percent;
  a fifth means a systematic offset, so the opt-in ASR verification is worth its time) and `signals`
  (flag count per signal). In `asr` mode the summary carries `mode`, `confidence` (= `coverage_ok`),
  `outliers` and `suggest_asr_check: false` as well.
* Mode C, `link.recover_text: true` (either mode): voiced lines whose text is empty (no script text)
  are transcribed with `link.recover_asr` (fallback `link.asr_fallback`) -- only those clips, nothing
  else. The text is stored as `text_recovered` in the line's `link.jsonl` record and applied by
  `StageContext.lines()` / `voiced_lines()` as `text_src` with `flags += ["text_from_asr"]` (script
  text always wins over a stale recovery; `--force` transcribes again). The user's corrections are
  override actions `set_src_text` (`overrides.jsonl`, payload `{"text": ...}`, the latest wins) applied by
  the same function with `flags += ["text_edited"]` (an edited ASR line carries both flags).
  `lines.jsonl` is never rewritten. Every stage reads text through `ctx.lines()`, so translation
  batches (their cache key contains the source text), the aligner and the CER/ref texts follow edits.
  `vndub.stages.link.no_text_lines(ctx)` lists the voiced lines still without text;
  `text_recovered_lines(ctx)` those that came from ASR. A recovered line is not compared with its own
  transcript in `asr` mode (record `ok`, match 1.0).

ASR mode steps:
1. ASR (`link.asr`, fallback `link.asr_fallback`): anime-whisper via faster-whisper, no
   initial prompt, `no_repeat_ngram_size=5`, no conditioning on previous text.
2. NFKC + punctuation/ruby stripping, RapidFuzz normalized similarity against the bound
   line and +-`neighbour_window` neighbours; `>= accept_score` ok, `[review_score,
   accept_score)` review, else a global top-`global_topk` search (blocked by speaker);
   `auto_repair` moves a file to a clearly better line.
3. Speaker embeddings (`link.speaker_embedder`), agglomerative clustering
   (`cluster_distance`), `spk_consistency` = cosine to the speaker centroid;
   `< outlier_consistency` -> review.

Progress: `ctx.progress(done, total, item)` after each ASR / embedding chunk and each judged line.

### analyze
In: voiced lines. Out: `work/analysis/<line_id>.json` (the expression passport, schema
`manifest.io.Passport`), `analysis/emb/<line_id>.npy` (emotion2vec+ embedding),
`analysis/feat/<line_id>.{f0,energy}.npy`, updated `characters.yaml` stats
(n_lines, total_sec, f0_median_hz, f0_iqr_st, energy_db_median, lufs_median, rate_median,
emotion_hist) and `ref_candidates`. Models run grouped by model (`analyze.e2v`,
`anime_ser`, `adv`, `events`, `aligner`, `f0`, `vad`, optional `caption`). A second pass
fills `passport.style` (StyleSpec) relative to the character medians.

analyze never recognises speech. The forced aligner (`qwen3_aligner`, fallback `whisperx_ja`, both
role `align`) aligns the *known* text (`ctx.lines()`: script text, recovered or edited); a name of
another role set as `aligner`/`aligner_fallback` (an ASR model) is refused, the line then falls
back to VAD-only timing (mora rate over the VAD speech time): `passport.backends.timing =
"vad_only"` and the reason in `passport.missing.align`. SenseVoice is an event tagger here; its
recognised text is dropped before the result is cached (`raw` keeps only the `<|tag|>` markers).

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
- **Unvoiced speaker lines** (mode B): `translate.include_unvoiced` (default `auto` = on only when
  `synth.voice_unvoiced` is on, which the desktop app sets, so the CLI keeps translating voiced lines only; `on` =
  always; `off` = never) also translates speaker lines without original
  audio (the unvoiced protagonist; never narration, which stays context, never `voice_sustain` continuations, which
  are translated anyway as part of their head line).
- **Delivery field** (LLM layer of `text_emotion`): for those lines the request row carries
  `"want_delivery": true` and the answer entry gets one more field,
  `"delivery": {"emotion": angry|disgusted|fearful|happy|neutral|other|sad|surprised|unknown|embarrassed,
  "intensity": 0..1, "tempo": slow|normal|fast, "loudness": soft|normal|loud,
  "events": [laugh|sigh|cry|breath|whisper], "confidence": 0..1}`. The model also gets the script around the
  line (`scene_lines`: +-`text_emotion.llm_context_lines` lines, narration included, source text), the
  character sheet (age, description) and the rule "when nothing in the text or the surrounding narration
  indicates a delivery, answer neutral, 0.3, low confidence; do not invent drama". Validation is strict
  (enums, ranges, foreign `id`): an invalid object is dropped and never fails, retries or halves the batch;
  `text_emotion` then falls back to its rule and cue layers. Stored as the extra field
  `TranslationRecord.delivery`. These lines go in their own batches after the scene's regular ones
  (cache key `DELIVERY_PROTOCOL`, batch, context, scene context), so cached batches of voiced lines are never
  invalidated by switching them on. `text_emotion.llm: off` -> not requested.
  With `translate.import`: no LLM is started for delivery when `text_emotion.llm` is `auto`; `on` runs a
  delivery-only request (`{"lines":[{"id","delivery"}]}`, same client, batching and cache) and keeps the
  imported text as is; an unavailable model only means "no llm delivery".
- `vndub.stages.translate.preview_import(ctx, path, n=5)` -> `{format, matched, total, preview: [{line_id,
  text_src, text_tgt}], files, in_file}`: parses a translation file/folder with the importers above and matches it
  to the lines that would be translated, without writing anything. `format`: `jsonl`, `json_map`, `json_rows`,
  `json_pairs`, `vntextpatch`, `csv`, `tsv`, `xlsx`, `folder:<suffixes>`.

### text_emotion
Delivery from text for lines without original audio (mode B: the unvoiced protagonist, unvoiced
choices-as-speech, a whole unvoiced game). `vndub text-emotion` (stage `text_emotion`, run by `vndub run` right after
`translate`; the app also runs it before `translate` exists). Cheap, cached, no model loaded.

In: `ctx.lines()` (script text with link rebindings / recovered / edited text), `characters.yaml`, the
translation when there is one, `configs/delivery_cues.yaml`. Target lines = speaker lines with no original audio
after the link rebindings; narrator lines, `voice_sustain` continuations and menu records are not targets.
Out: `work/analysis/<line_id>.json` (schema `manifest.io.Passport`, extra fields allowed):

```json
{"line_id": "start_000012", "source": "text", "duration_s": 0.0,
 "delivery": {"emotion": "neutral", "intensity": 0.25, "tempo": "slow", "loudness": "soft", "events": ["whisper"],
              "confidence": 0.95, "layer": "cue", "cues": ["……", "？", "ささやいた", "小さな声"],
              "arc": "rising_end", "notes": "hesitant, with pauses", "layers": {"rule": {"accepted": true, "...": "..."}, "cue": {"accepted": true, "...": "..."}}},
 "style": {"emotion": "neutral", "intensity": 0.25, "tempo": "slow", "loudness": "soft", "events": ["whisper"], "arc": "rising_end", "notes": "hesitant, with pauses", "...": "StyleSpec.to_dict()"},
 "backends": {"text_emotion": "rule+cue"}, "missing": {"audio": "no original audio: delivery derived from text"}}
```

`delivery.layer` is `rule`, `cue`, `llm` or `neutral` (the winning layer, `neutral` = generic); `delivery.events`
uses `laugh|sigh|cry|breath|whisper` (the StyleSpec in `style.events` says `laughter` instead of `laugh`: the
vocabulary of `text/style_prompt.py`); `style` is `StyleSpec.to_dict()` of the merged delivery, so the style
renderers work unchanged; `delivery.layers` records every layer that took part with its confidence and
whether it was accepted. `analysis/<id>.json` of a line with original audio (written by `analyze`) is never
touched, even if the line is unvoiced now (`kept_audio_passport` in the summary).

Layers, merged rule -> cue -> llm (`vndub/text/delivery.py`):
1. **rule pass** (no model), on the Japanese text and on the translation when there is one (the same mark in
   both counts once): `！` raises intensity (one 0.55, `！！` 0.7 and loud), `？` rising arc, `！？` surprise, `……` / `…`
   slower tempo + "pauses" (a single `…` is too weak), elongation (`ー`, `〜`, repeated vowels / small kana /
   `っっ`), ALL-CAPS, `♪` / `♡` / `☆`, laughter (`ははは` `あはは` `えへへ` `ふふ` `くすくす`, `(笑)`, "ха-ха" "хи-хи"),
   sobbing (`うう…` `ぐすっ` `ひっく` `しくしく`, "хнык"), whisper markers (`（小声）` `(ひそひそ)`), a line in
   parentheses = inner voice (soft), leading interjections (`えっ` `ひっ` `ちっ` `やった` `はあ` `しーっ` ...), stammering
   (`わ、わたし`, "п-п-подожди"), the interjection lexicon `configs/interjections.yaml` (its Russian equivalent adds
   its onomatopoeia when there is no translation yet). Each piece of evidence has a weight; confidence = noisy-OR,
   reduced when named emotions disagree. A plain `。` is no evidence: neutral, confidence 0.2. `！` alone without a named emotion gives
   `other` ("expressive") at intensity >= 0.5.
2. **narration cues**: the narrator lines around the line (`narration_window`, +-3 lines, same scene) are scanned
   with the table `configs/delivery_cues.yaml` (`ja` entries: substrings, kana-insensitive; other languages: word
   stems matched as prefixes, so gender/tense forms match, reflexive `-ся/-сь` ignored; scanned in the source
   language and, when the narration has a translation, the target language). Entry:
   `{cue, emotion?, intensity?, tempo?, loudness?, events?, confidence=0.75, position=any|after|before, notes?}`.
   **Attribution** (a cue describes one line): the speaker whose name occurs in the narration (first mention
   wins; a narration naming only characters who are not around is ignored); otherwise the nearest dialogue line,
   ties decided by `position` (`after` -> preceding line, `before` -> following, `any` -> preceding). Candidates
   are all speaker lines, voiced ones included, so a cue about a voiced heroine never lands on the protagonist.
   Confidence is adjusted: +0.1 name match, -0.05 per line of distance beyond the first, -0.1 tie without a name,
   -0.05/-0.1 when the line is on the side the cue's position argues against.
   `vndub.stages.text_emotion.load_cues(path) -> dict` / `save_cues(path, data)` read and write the table
   (`ValueError` naming the entry for unknown emotions, ranges, enums, duplicates; a failed save leaves the file).
3. **LLM layer**: `TranslationRecord.delivery` written by `translate` (see above); this stage only reads it
   (`text_emotion.llm: off` ignores it).

**Merge**: a layer takes effect only when its confidence is `>= text_emotion.min_confidence` (0.6); later layers
override the fields they set (events, notes and cues accumulate). No accepted layer: generic = neutral, intensity
0.3, normal tempo and loudness, `layer: neutral`, confidence 0.2: nothing indicates a delivery, so none is invented.

Cache: `ctx.cache` per line, key = line text, speaker, translation, the translation's `delivery`, the cues attributed
to it, the narration texts within the window, cue-table hash, lexicon hash, config section
(`min_confidence`, `narration_window`, `llm`, ...); `--force` clears. A rerun with nothing changed writes nothing.
Progress: `ctx.progress(i, n, line_id)` per line. Summary: `lines, written, cached, kept_audio_passport,
layers, emotions, narration_lines_with_cues, llm, cues`.
Config: `text_emotion.{enabled, cues, narration_window, llm (auto|on|off), llm_context_lines, min_confidence,
only_unvoiced}` (`only_unvoiced: false` also gives voiced lines that have *no* passport yet a stop-gap text passport).

### voicebank
Out: `work/voicebank/<speaker_id>/refs/*.wav` + `refs.json` (`RefClip`: path, line_id,
emotion, arousal, valence, snr, duration, usage timbre|emotion|both, text),
`performer.yaml` + `performer/*.wav`, `rvc/` (model, index, `meta.json` with
`f0_transpose`), `lora/`, `synthetic/`, and `work/voicebank/spk_calibration.json` (EER
threshold of same- vs different-character cosine). Config: `voicebank.refs.*`,
`voicebank.performer.*` (voice design from gender/age/pitch/emotion profile, picked by ECAPA
similarity + F0 within `max_f0_delta_semitones`), `voicebank.rvc.*` (Applio training with
`>= min_minutes` of clean audio), `voicebank.route_c.*`.

**Voice designs (casting).** Every design a character is offered is kept: `voicebank/<sid>/designs.json`
(`{"version": 1, "designs": [Design]}`) and `voicebank/<sid>/designs/<design_id>/*.wav`. `Design`: `design_id`
(`vd01` generated, `ref01` user sample, `clone`, `imp01` from the voice library), `kind` `voice_design` |
`reference` | `clone` (route A: zero-shot from the original clips, offered for characters with audio; the
stage renders it only with `voicebank.performer.offer_clone`, the app's `propose_designs` always), `backend`,
`prompt`, `description`, `samples` (the audition clips: `voicebank.performer.sample_lines` translated lines of
the character, spoken in that voice), `ref_clips` (generic test sentences up to `test_sentences` clips; with the
samples they become the performer's timbre references), `spk_sim`, `f0_delta_st`, `stars`, `rank_score`,
`license`, `commercial_ok`, `created`, `origin` (`stage` | `user` | `library`). Designs are ranked by
`rank_score`: inside the F0 window (`max_f0_delta_semitones`) before outside, then speaker similarity to the
character centroid, then F0 distance. Stars (1-5, computed on read): `floor(3 + 2 * (spk_sim - thr) / w + 0.5)`
clamped to 1..5 with `thr` the speaker-similarity gate threshold (calibrated EER threshold when available) as the
3-star point and `w = clamp((same_mean - diff_mean) / 2, 0.10, 0.35)` from the calibration (0.20 without);
`None` without original audio. The stage generates `n_designs`, picks the performer (the design the user chose,
`characters.yaml performer.design_id` + `chosen_by: user`, else the top-ranked non-clone) and writes
`performer.yaml` + emotion variants as before; user designs and the chosen one survive reruns. Characters with
dub lines but no original audio (only with `synth.voice_unvoiced`) are designed from the character sheet (gender,
age, names, `Character.description`, the tuning's character note / timbre preset), have no similarity or stars,
and default to `route: "P"`. API used by the casting screen: `list_designs`, `propose_designs`,
`add_reference_design` (5-20 s wav/mp3; `ValueError` messages `sample_too_short` / `sample_too_long` /
`sample_bad_format`), `import_design`, `choose_design`, `auto_cast`, `sample_lines`, `train_rvc` (route B model of
one character), `set_tuning`.

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
- Route P (performer voice only, no VC): speaker lines without original audio (`synth.voice_unvoiced`, off for
  the CLI; narration is never dubbed) and characters with `route: P`. The same performer TTS as route B,
  written straight to the candidate path. The style comes from the text passport (`source: "text"`, stage
  `text_emotion`), neutral when there is none; there is no source duration (no `target_duration`). Characters
  without voicebank refs get route P automatically.
- Voice tuning (`characters.yaml.<sid>.performer.tuning` = `{pitch_st -6..6, speed 0.8..1.2, expressiveness 0..1
  (0.5 = passport as is, 0 = always calm, 1 = amplified), energy -1..1, timbre_preset "" | younger | older |
  softer | sharper, character_note}`; a per-line `tuning` override in overrides.jsonl replaces it for that
  line). `text/style_prompt.py`: `apply_tuning` changes the StyleSpec, so instruct-TTS gets the words
  (Qwen / Cosy instruct, VoxCPM2 prefix), Chatterbox `exaggeration` and Zonos strength follow the scaled
  intensity, Fish / Higgs tags follow energy / calm; `tuning_knobs` adds native knobs (CosyVoice `speed`, Zonos
  `speaking_rate`); `tuning_plan` splits the rest: RVC `f0_transpose += pitch_st` (route B), a speed on
  duration-controllable backends via `target_duration`, and for everything without a native knob the `master`
  pitch / tempo pass (Rubber Band, ffmpeg fallback). The plan is recorded in the sidecar
  (`params.tuning`) and the tuning is part of every candidate's cache key: changing it regenerates only that
  character's (or that line's) candidates in place; candidates of an older tuning that are no longer planned are removed.
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
Lines without original audio (route P): `emo_sim`, `adv_delta`, `prosody` and `dur_ratio` are not computed, so
those terms are missing and the composite is renormalized over `utmos` / `pq` / `spk_sim` / `cer`; no gate needs a
reference metric (gates of unavailable metrics are never evaluated). `spk_sim` is measured against the performer's
reference clips (voice consistency with the chosen design; same threshold) when the character has no own refs or the
route is P. A speed the TTS backend applied itself (tuning) moves the expected duration of the `dur_ratio` check.
`vndub score --reweight` recomputes composites and selection from persisted metrics only.

### master
Out: `work/final/<line_id>.wav`, `work/final/encoded/<voice_file>`, `work/final_index.jsonl`.
Trim to the original's silence profile, stretch (Rubber Band, ffmpeg `atempo` fallback)
only outside `master.stretch.target_window` and within `max_stretch`, loudness-normalize to
the character's median LUFS with a true-peak limiter at `true_peak_dbtp`, resample to the
source sr/channels, encode to the source codec/bitrate, same file name.
`master.include_review_lines: provisional` also masters review lines (marked provisional).
Lines without original audio: no source duration to stretch to, default silence profile, loudness of the
character, else the median of the characters' loudness, else `default_lufs`; file `unvoiced/<dialogue id | line
id>.<ext>` (`FinalIndexRecord.unvoiced`). The tuning's leftovers from the sidecar are applied before the
duration step: pitch / tempo pass for backends without a native knob (the requested speed is not limited by
`max_stretch`; the source-duration check then compares with `source / speed`), and energy -> loudness offset
(-1..1 -> -3..+3 dB).

### pack
Out: `out/<engine>/...` + `INSTALL.md` + `manifest_patch.json` (file, sha256, line_id,
backend, route). Ren'Py: loose `game/<voice>`; NScripter: loose files beside the exe;
KiriKiri: `patchN.xp3` (Xp3Pack or the built-in unencrypted writer) or `unencrypted/`;
Unity: UnityPy replacement script + checklist (experimental); others: directory layout +
tool command lines. See ENGINES.md.
Lines that had no original voice (`unvoiced` finals) have no file to replace: Ren'Py gets
`game/<dir>/<dialogue id>.<ext>` plus a generated `game/vndub_auto_voice.rpy` that sets `config.auto_voice` for
that folder (`pack.renpy.unvoiced_dir`, default `voice_dub`), or, when the game already defines
`config.auto_voice`, the files under its pattern and no rpy; every other engine gets
`out/<engine>/unvoiced/<file>` and a note in INSTALL.md that they are NOT installed automatically.
`manifest_patch.json` marks them (`unvoiced`, `renpy_auto_voice` / `manual`).

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
| `parse` | `engine`, `engine_opts.{kag,nscripter,renpy,tyrano,generic}`, `import_vntextpatch`, `narrator_names`, `hidden_speakers` |
| `link` | `asr`, `asr_fallback`, `asr_compute`, `accept_score`, `review_score`, `neighbour_window`, `global_topk`, `auto_repair`, `speaker_embedder`, `cluster_distance`, `outlier_consistency`, `mode` (`asr` \| `no_asr`), `recover_text`, `recover_asr`, `rate_min`, `rate_max`, `silence_tolerance_s`, `suggest_asr_share` |
| `analyze` | `e2v`, `anime_ser`, `adv`, `events`, `aligner`, `aligner_fallback`, `f0`, `vad`, `caption`, `caption_backend`, `ref_candidates_per_emotion`, `e2v_weight`, `anime_weight` |
| `translate` | `import`, `llm.{backend,fallbacks,ctx,n_gpu_layers,temperature,port,max_retries,real_backend}`, `batch_lines`, `context_window_lines`, `glossary`, `short_variant_min_words`, `stress.{tool,model,yo,format}`, `interjections`, `mock_dictionary`, `include_unvoiced` (`on` default \| `off` \| `auto`) |
| `text_emotion` | `enabled`, `cues`, `narration_window`, `llm` (`auto` \| `on` \| `off`), `llm_context_lines`, `min_confidence`, `only_unvoiced` |
| `voicebank` | `refs.*`, `performer.{kind,backend,n_designs,test_sentences,max_f0_delta_semitones,reference_dir,sample_lines,offer_clone}`, `rvc.*`, `route_c.*`, `calibrate_spk`, `min_minutes_for_voicebank` |
| `synth` | `n_candidates`, `n_candidates_short`, `routes` (A, B, C, P), `backends.<name>.{enabled,...}`, `vc.{primary,fallback,rvc,seedvc}`, `keep_original_for_interjections`, `use_short_text_if_ratio_above`, `interjection_max_syllables`, `high_arousal`, `seed_base`, `voice_unvoiced` |
| `score` | `weights`, `gates.{utmos_min,cer_max,dur_ratio,spk_sim_min,spk_sim_default}`, `mos`, `mos_fallback`, `aesthetics`, `speaker`, `speaker_extra`, `cer_asr`, `cer_asr_fallback`, `emotion`, `adv`, `regen_rounds`, `interjection_cer_max` |
| `master` | `stretch.{enabled,target_window,max_stretch,engine}`, `loudness.{target,true_peak_dbtp,default_lufs}`, `match_silence_profile`, `include_review_lines` |
| `pack` | `kirikiri`, `renpy`, `nscripter`, `unity`, `other` |
| `review` | `host`, `port`, `batch_accept_threshold` |
| `report` | `compare_routes`, `markdown` |

`configs/pipeline.test.yaml` (`extends: pipeline.yaml`) is the CPU/tiny profile used by the
fixture run and CI.

### The renpy_mini fixture (mixed game)
`tests/fixtures/renpy_mini` (generator `scripts/make_fixtures.py renpy`, data `scripts/fixture_data.py`,
description `FIXTURE.md`, MockTranslator dictionary `tests/fixtures/mock_dictionary.yaml`, golden parser
output `tests/fixtures/golden/renpy_mini.jsonl`) is a mixed game: 20 voiced lines (sakura, haruto, yui; SAPI
audio, never re-rendered unless `--rerender-voices`/`--synthetic`), an UNVOICED protagonist `hero` (name 蓮,
`mc`, 6 lines) and 13 narrator lines, 5 of them next to hero lines with Japanese delivery cues
(`小さな声でささやいた`, `笑いながら言った`, `泣きながら言った`, `…と叫んだ`, `怒鳴った`). 41 lines in
scenes `start` (hero lines before the menu), `after_menu`, `epilogue`. Hero lines: `start_000012`,
`start_000015`, `start_000016`, `epilogue_000002`, `epilogue_000006`, `epilogue_000009`; cue narration:
`start_000013` (whisper), `start_000014` (laughing), `start_000017` (crying), `epilogue_000007` (shout),
`epilogue_000008` (angry); `epilogue_000002` (`みんな、帰ったのか。`) has no cue within three narration lines.
Regenerate after editing the data: `uv run python scripts/make_fixtures.py renpy` then
`VNDUB_UPDATE_GOLDEN=1 uv run pytest tests/test_engine_goldens.py`.

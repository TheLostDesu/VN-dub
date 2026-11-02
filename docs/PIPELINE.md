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
adapter is restarted once and the call retried; timeouts kill the process. An adapter handles
the items of a `run` call one by one (`op_<name>`) or, where the library takes a batch, all at
once (`batch_<name>` built on `rpc.run_in_batches`: sub-batches of `options.batch_size`, a
failing sub-batch redone item by item, results in item order; SenseVoice, anime-SER, Qwen3-ASR,
the Qwen3 aligner and UTMOSv2 do, see DECISIONS.md "Batched adapters"). Role operations:

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
Out: `work/project.yaml` (`ProjectYaml`: engine, engine_version, src_lang, audio_lang, tgt_lang,
voice_codec {container, codec, sr, ch, bitrate_k}, encrypted, drm, archives, warnings). Engine by
file signatures (see ENGINES.md); `--engine` forces one. Encryption and DRM markers produce warnings.
Languages come from the config: `project.src_lang` (`auto` by default: `parse` detects it from the
script text and writes the code plus a `src_lang_detected` record `{lang, confidence, letters,
lines, scripts}` into `project.yaml`), `project.audio_lang` (`""` = the voices speak the text's
language; set a code for a translated release that kept the original voices) and `project.tgt_lang`.
Re-running detect keeps the languages of an existing `project.yaml` (a detected code is kept only
while the config still says `auto`; a code set by the user is never replaced).
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
Source language (`project.src_lang: auto`): after the lines are written parse runs `text/lang.detect_language`
over them (letters counted by script: kana among the CJK letters = ja, han alone = zh, hangul = ko, Cyrillic =
ru / uk, Latin = en / de / fr / es / it / pt by a stop-word vote; no model) and stores the code with the
`src_lang_detected` record in `project.yaml`; every later stage reads `StageContext.src_lang` (and
`audio_lang`). A code set by hand is kept, with a warning when the text clearly looks like another language.
Config: `parse.engine_opts.<engine>.*`, `parse.import_vntextpatch`, `parse.narrator_names`, `parse.hidden_speakers`.

### link
In: lines, `src16` audio. Out: `work/link.jsonl` (`LinkRecord`: line_id, audio_id,
asr_text, match_score, best_alt_line_id, best_alt_score, spk_cluster, spk_consistency,
status ok|review|unmatched, reasons, relinked_from; extras `signals` (no_asr) and
`text_recovered` (mode C)), `work/review_queue.jsonl`, `work/analysis/spk/<audio_id>.npy`.

The script text is the source of truth; the original audio is only an intonation reference.
`link.mode` selects how the line <-> voice mapping is verified:

* `asr` (CLI default): the steps below. `link.recover_text` aside, behaviour is unchanged. The ASR runs in
  the voices' language (`audio_lang`); a backend built for one language (`anime_whisper`, `options.language:
  ja`) is skipped for voices in another one. When `audio_lang` differs from `src_lang` (a translated release
  that kept the original voices) the ASR text cannot be compared with the script: the stage falls back to the
  cheap signals below and says so (summary `note`, `mode: no_asr`).
* `no_asr` ("intonation-check only", the desktop app's default): **no ASR backend is acquired**
  (not even the mock; `work/gpu_events.jsonl` shows no `role: asr` load). Cheap signals only:
  1. speaker-embedding consistency (`link.speaker_embedder`, steps 3 below; reasons
     `spk_outlier:<cosine>` below `outlier_consistency`, `spk_cluster:<speaker>` when the clip
     clusters with another speaker),
  2. duration vs text length: morae of the (sustain-extended) text (`text/ja.count_morae`; for a text in
     another language mora equivalents = estimated syllables x 1.5, `text/lang.speech_units`) against the
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
  script references a voice file that does not exist); a clip of zero length is `review` with `audio_empty`. No auto-repair: nothing is moved, orphan files
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
- The source language is `StageContext.src_lang` (detected by parse, see above): the prompt says
  "translating a visual novel from <language>"; the import readers also accept the code / name of
  that language as the source column. A script that already is in the target language is kept as it
  is (`source: same_lang`), no model is started.
- `translate.import`: VNTextPatch JSON/XLSX dir or a line_id -> text map; skips the LLM.
- Else `llama-server` with `translate.llm.backend` (Gemma 4 12B IQ4_XS; fallbacks), per scene,
  batches of `batch_lines` as JSON `[{id, speaker, gender, emotion, text}]` with the last
  `context_window_lines` translations and a scene summary; system prompt = glossary (names,
  terms present in the batch, honorific policy, ты/вы matrix, genders) + rules. Output JSON
  `{"lines":[{"id","ru","ru_short","notes"}]}`, validated strictly; retry at temperature 0,
  then halve the batch. An answer cut off by the token budget (`finish_reason: length`) keeps
  the lines that were complete, only the rest is retried; thinking is switched off in the chat
  template (`enable_thinking: false`, Gemma 4 / Qwen 3.x) so the budget goes to the answer.
  Every incomplete answer is appended raw to `work/logs/llm_answers.jsonl` (ids, coverage,
  finish reason, reasoning chars, content). `translate.llm.parallel: N` starts llama-server with
  `N` slots (`-np N`, context `ctx x N`) and translates `N` scenes at once from threads (continuous
  batching on the GPU); the default 1 fits an 8 GB card with the 12B model.
- Interjections (`configs/interjections.yaml`, Japanese -> Russian) map to natural equivalents;
  the lexicon and the kana-only interjection flag apply to Japanese scripts only.
- Laughter flags (any language, `vndub.text.laughter`): a line whose script text or translation contains a laugh
  ("Haha, no way!", "Хе-хе", `*laughs*`) gets the flag `laughter`; a line that is nothing but laughter ("Haha!",
  "Ха-ха!", `(laughs)`) gets `interjection` as well, so it takes the interjection path of synth (`n_candidates_short`,
  the original audio as a candidate, `interjection` text variant) and score (laughter-only lines have no CER gate, see
  score). The summary counts them (`laughter`, `interjections`).
- Post-processing: ruaccent stress (+ yo), numbers -> words, abbreviations expanded. `text_tts` has no marks;
  `text_tts_plus` / `text_tts_short_plus` keep ruaccent's `+` and are written when an enabled backend asks for stress
  (`stress_format`: `plus` for F5-TTS_RUSSIAN, `acute` for Chatterbox).
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
   `っっ`), ALL-CAPS, `♪` / `♡` / `☆`, laughter (see below), sobbing (`うう…` `ぐすっ` `ひっく` `しくしく`, "хнык"), whisper markers (`（小声）` `(ひそひそ)`), a line in
   parentheses = inner voice (soft), leading interjections (`えっ` `ひっ` `ちっ` `やった` `はあ` `しーっ` ...), stammering
   (`わ、わたし`, "п-п-подожди"), the interjection lexicon `configs/interjections.yaml` (its Russian equivalent adds
   its onomatopoeia when there is no translation yet). `rule_pass(text_src, text_tgt, lexicon, src_lang)`: `src_lang`
   (`ctx.src_lang`) says which laughter scripts the source text can contain. Each piece of evidence has a weight; confidence = noisy-OR,
   reduced when named emotions disagree. A plain `。` is no evidence: neutral, confidence 0.2. `！` alone without a named emotion gives
   `other` ("expressive") at intensity >= 0.5.
   **Laughter** (`vndub/text/laughter.py`, shared with synth): written laughter in the source text (any script: kana
   `あはは` `えへへ` `ふふ` `くすくす` `くくく` `ぷっ`; Cyrillic `ха-ха` `хе-хе` `хи-хи` `муа-ха-ха` `пф-ф`; Latin `haha` `hehe` `hihi` `mwahaha`
   `ehehe...` `pfft`, also hyphenated / spaced `ha-ha`, `heh heh heh`) and in the translation gives the `laugh` event and a
   kind: `hearty`, `chuckle`, `giggle`, `evil`, `nervous` (a hearty laugh or chuckle trailing into `...`), `snort`. Per kind: emotion
   (hearty / chuckle / giggle `happy`, nervous `embarrassed`, evil and snort `other`; `other` names no direction, so it only wins
   when nothing more specific is named), an intensity bump (short laugh 0.10-0.25, long laugh of 3+ syllables 0.15-0.35) and a
   note ("a hearty laugh", "chuckling", "giggling", "a nervous laugh", "a sinister laugh", "a snort"). A bracketed stage direction
   (`*laughs*`, `(giggles)`, `(смеётся)`, `(笑)`) is evidence too, at weight 0.65 instead of 0.75 and half the intensity bump (a line
   that is only `(笑)` is a stage direction, not an inner voice). The longest spoken laugh sets the kind. `delivery.laughter` =
   `{kind, units, stage, only}` (`only`: what the dub speaks is nothing but laughter; `StyleSpec.laughter` is the same dict);
   `Delivery.to_dict` has the key only when a laugh was found. The LLM layer may answer `events: ["laugh"]` plus an optional
   `laugh_kind` (also `events: ["giggling"]`; an unknown word is ignored, the object stays valid); a kind from the rule pass
   (it read the actual text) wins over the LLM's. `text_emotion` stage `VERSION` is 2: passports of version 1 have no
   `delivery.laughter` and are rewritten (no model, cheap). How the laugh is then spoken is decided in synth.
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
  Tag syntaxes follow the model cards: Fish S2 free-form `[laughing]` `[chuckling]` `[inhale]` `[clearing throat]` (the
  parenthesis form `(laughing)` is the deprecated S1); Higgs v3 `<|emotion:amusement|>` `<|style:whispering|>` (no
  `happiness` / `neutral`), `<|sfx:laughter|>` paired with its onomatopoeia right after it.
- **Laughter** (`synth.laughter`, `configs/laughter.yaml`, `text/laughter.py`). An LLM-based TTS reads a written "Ха-ха-ха"
  letter by letter unless it is spelled the way the model met laughter, or replaced by its own laugh token; Qwen3-TTS Base (the
  zero-shot cloner: it ignores `instruct`) has nothing but the text to work with. After the text variant (`text_tts`,
  `text_tts_short`, plus variants) is picked, every laugh in it (kana, Cyrillic, Latin, whatever script the translator wrote)
  is rewritten for the backend, per laugh and per backend:
  * `respell` (default; every backend without a laugh token): the table spelling for the target language, per kind, `short`
    (1-2 syllables) or `long` (3+): hearty ru `Ахаха!` / `Ахахаха!`, chuckle `Хехе`, giggle `Хихи`, evil `Муахаха!`, nervous
    `Ахаха...`, snort `Пфф` (vowel-run forms on purpose: the hyphenated dictionary spelling "Ха-ха" is what a TTS reads syllable by
    syllable); the sentence mark of the spelling yields to the punctuation of the line ("Ха-ха-ха, ну" ->
    "Ахахаха, ну"); the capital of a sentence-initial laugh is kept;
  * `inline` (caps `tags: cosy_inline`): CosyVoice `[laughter]` replaces the laugh (a tokenizer special token; the CosyVoice2 example
    writes it inline, the CosyVoice3 example only shows `[breath]`); a comma after the token is dropped; a line that is only a
    laugh keeps a spelling next to it (`backends.cosyvoice3.only: tag`);
  * `tag` (caps `tags: fish` / `higgs`): the token goes in front of the spelled laugh, at the laugh: `[chuckling] Хе-хе`,
    `<|sfx:laughter|>Haha` (Higgs: spelled in English, `backends.higgs_tts3.spelling`); the Fish / Higgs prefix then leaves its
    own laughter tag out;
  * `keep`: spoken laughs are left (`backends.<name>.mode: keep`; `synth.laughter.enabled: false` switches the whole thing off,
    stage directions included).
  Stage directions (`*laughs*`, `(смеётся)`, `(笑)`) never reach the TTS as words: removed (`stage: remove`, default), spoken as a
  laugh of their kind (`stage: respell`), or the backend token (`inline` / `tag`); a stage direction next to a spoken laugh is
  dropped; a line that is only a stage direction is spoken as a laugh, never sent empty. A `set_text` override is sent exactly as
  typed. `text_plain` (the CER reference) is never respelled. The laugh kind and length also reach the style: `render()` gets a
  `StyleSpec.laughter` made from the TTS text (event `laughter` added when the passport had none, unless the tuning is "always
  calm"), so the instruct says "laughing heartily" / "giggling softly" / "a short, nervous laugh" / "cackling maniacally"
  (kind x strength) instead of "with a light laugh", a laughter-only line gets an instruction about the laugh alone ("Only
  laugh, no words: make a short, nervous laugh ..."), VoxCPM2 gets "(happy, giggling)", Fish `[giggling]` / `[laughing nervously]`.
  `render()` also returns `laughter` = the decision for the text (`respell` | `inline:<token>` | `tag:<token>`). The sidecar has
  `laughter: {kind, n, mode, only, stage}` (absent for a line without laughs); the stage summary `laughter: {lines, modes}`.
  `configs/laughter.yaml` is meant to be edited per voice: `<lang>: {<kind>: {short, long}}`, `default`, `stage`, and a `backends:`
  block per backend name (`mode`, `token` / `tokens`, `only`, `spelling`, `stage`, per-language overrides). The setting is left out
  of the synth cache hash (the spelled text and the laugh phrase are part of every candidate key), so enabling it, or editing the
  table, regenerates only the candidates of lines with a laugh.
- Russian stress: a backend with `stress_format: plus` is sent `text_tts_plus` as it is; `acute` (Chatterbox) turns it into U+0301
  marks after the laughs are written, without marks on a word with one vowel or on ё. Laughs lose their marks first, or the
  laugh patterns would not see them; `text_used` shows what was sent, `text_plain` (the CER reference) never has marks, a `set_text`
  override goes as written. A project translated while no such backend was enabled has no variant: synth makes it for the run from
  translate's stress cache (the translation file is not touched; without a stresser the lines stay plain, with a warning).
  `synth.backends.<name>.stress_format: plain` turns the marks off for one backend.
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
otherwise `regen` (up to `regen_rounds`) then `review`. Overrides are honoured. A line whose most intelligible take
still misreads the text by more than `score.regen_cer_max` (0.2, never below the line's CER gate) goes to `review` at
once ("a new take would not help"): a new seed changes the delivery, not the words. A take the ASR heard no words in
gets no CER (note `cer failed: the ASR heard no words in this take`): the gate counts as not checked, never as a CER
of 1.0.
**Laughter** (`vndub.text.laughter`) is not text an ASR can spell, so the CER is computed over the words only: laughs
are taken out of both the reference and the transcript (`"Haha, no way!"` is checked as `"no way!"`; a stage direction
like `(laughs)` counts as a laugh). The plain reading of the same texts is a floor, so a laugh the ASR heard as a stray
syllable ("ха да ну" for "Ха-ха, да ну!") is never held against the line. A line that is nothing but laughter ("Haha!",
"Хе-хе.") has no words to check: its candidates get `cer` in `na` (like the `orig` route) and the note
`laughter: no words to check`, the ASR is not asked for them, no CER gate exists for them, and they are accepted or sent
to regen/review on the other gates (utmos, spk_sim, dur) alone; such a line is never "every take misreads the text", and
its failure text names the failing gate and says `laughter only, no words to check`. A take whose transcript holds only
a laugh while the line has words is "the ASR heard no words" (unchecked gate), as before. The compared texts are stored
on the candidate (`cer_ref`, `asr_text`, `cer_hyp` when a laugh was taken out of the transcript, `cer_rule`); a CER
stored by an older run (no `cer_rule`) for a line that has laughter is recomputed on the next `score`.
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
Lines whose selection is `review` (no take passed the gates, the regen rounds used up or pointless) or still `regen`
(the rounds' new takes were never made) are mastered from their best take and marked `provisional`
(`master.include_review_lines`, default `provisional`): a mediocre take in the game beats a line that stays silent
(on a route-P project most lines can end in review, and `skip` used to give a patch of the third that passed).
`skip` leaves them out; the summary then counts them under `skipped.review` / `skipped.regen`. A line without any winner
(`skipped."no winner"`) has nothing to install. The summary carries `provisional` (and `provisional_regen`), the stage
log says it in a warning, and the final index records of provisional lines carry `selection_status` and `reason`.
`pack` and `report` turn this into the patch's own statement of what it covers (below).
Lines without original audio: no source duration to stretch to, default silence profile, loudness of the
character, else the median of the characters' loudness, else `default_lufs`; file `unvoiced/<dialogue id | line
id>.<ext>` (`FinalIndexRecord.unvoiced`). The tuning's leftovers from the sidecar are applied before the
duration step: pitch / tempo pass for backends without a native knob (the requested speed is not limited by
`max_stretch`; the source-duration check then compares with `source / speed`), and energy -> loudness offset
(-1..1 -> -3..+3 dB).

### pack
Out: `out/<engine>/...` + `INSTALL.md` + `manifest_patch.json` (file, sha256, line_id,
backend, route, status; `reason` for provisional lines). Both state what the patch covers: how many of the dubbed
lines have a file, how many of those are provisional ("best take installed, check it in Review"), how many are LEFT
OUT and why (`review` / `regen` with `include_review_lines: skip`, `no winner`, `not mastered`; each listed in
`manifest_patch.json` under `coverage.left_out_lines`), how many narration lines are never dubbed, and (Ren'Py) how
many speaker lines are NOT INSTALLABLE because the script gives them a `voice` statement of their own (`own_voice`,
`own_voice_lines`: `config.auto_voice` is never asked for them). INSTALL.md is rebuilt on every pack (its sections
never pile up). A full pack also removes the files of the previous patch that the new one does not contain (nothing
else under `out/<engine>` is touched; `--only` never deletes). The pack summary carries `provisional` / `left_out` /
`own_voice` / `unnamed`.
Ren'Py: loose `game/<voice>`; NScripter: loose files beside the exe;
KiriKiri: `patchN.xp3` (Xp3Pack or the built-in unencrypted writer) or `unencrypted/`;
Unity: UnityPy replacement script + checklist (experimental); others: directory layout +
tool command lines. See ENGINES.md.
Lines that had no original voice (`unvoiced` finals) have no file to replace: Ren'Py gets
`game/<dir>/<dialogue id>.<ext>` plus a generated `game/vndub_auto_voice.rpy` that sets `config.auto_voice` for
that folder (`pack.renpy.unvoiced_dir`, default `voice_dub`), or, when the game already defines
`config.auto_voice`, the files under its pattern and no rpy; every other engine gets
`out/<engine>/unvoiced/<file>` and a note in INSTALL.md that they are NOT installed automatically.
`manifest_patch.json` marks them (`unvoiced`, `renpy_auto_voice` / `manual`).
Ren'Py voice track (1.0.3): the dub plays on Ren'Py's `voice` channel, whose mixer Ren'Py binds at `init 1600` only when
`config.has_voice` is True, so a second generated script `game/vndub_voice_mix.rpy` (`init 1000`, after the game's
`options.rpy`) sets `config.has_voice`, a `config.sample_voice` (when the game has none: the standard preferences screen
then shows a "Voice Volume" slider), Ren'Py's ducking (`config.emphasize_audio_channels = ["voice"]`, the `emphasize_audio`
preference switched on once: music and sfx drop to `duck` while a line plays) and an in-game panel (screen
`vndub_voice_panel`: voice volume, music level during a line, ducking and mute toggles, test) on a hotkey and on a button in
the preferences screen. Config `pack.renpy.voice_mix`: `enabled: auto | true | false` (auto = the patch holds lines that had
no original voice; a bare `voice_mix: true|false` is `enabled`), `duck: 0.3` (1.0 = no ducking), `key: K_F8`.
`manifest_patch.json` carries a top-level `voice_mix` (`{rpy, key, keysym, duck, enabled}` or `null`), INSTALL.md the section
"The voice track in the game", the pack summary `voice_mix: true|false`. A full pack removes a `vndub_voice_mix.rpy` it
generated earlier when the feature is now off (same rule as the stale auto_voice script). See ENGINES.md.

### report
Out: `work/report/index.html`, `work/report/report.md`: mapping coverage, per-character
stats and emotion histograms, score distributions per backend/route, A/B/C comparison,
patch coverage (dubbed / in the patch / provisional / left out with reasons / narration), review queue with
reasons and where each review line ended up (`provisional` or `left out`), backend substitutions, stage timings,
peak VRAM per model.

### review (UI)
`vndub review --project <dir>`: Gradio app on `review.host:review.port`; writes
`work/overrides.jsonl` (`OverrideRecord`: line_id, action accept|reject|regen|set_text|
set_ref|keep_original, payload, ts).

## Config sections (configs/pipeline.yaml)

| Section | Keys |
|---|---|
| top level | `profile`, `extends`, `fallback_to_mock` (test only), `license_policy` research/commercial, `tools_file`, `tools` |
| `project` | `game_dir`, `work_dir`, `out_dir`, `models_dir`, `src_lang` (`auto` = detected by parse \| a code), `audio_lang` (`""` = the text's language), `tgt_lang` |
| `gpu` | `vram_budget_gb`, `safety_margin_gb`, `device`, `allow_cpu_fallback` |
| `extract` | `keep_stereo_copy`, `analysis_sr`, `workers`, `voice_exts`, `vgmstream_exts`, `voice_dirs` |
| `parse` | `engine`, `engine_opts.{kag,nscripter,renpy,tyrano,generic}`, `import_vntextpatch`, `narrator_names`, `hidden_speakers` |
| `link` | `asr`, `asr_fallback`, `asr_compute`, `accept_score`, `review_score`, `neighbour_window`, `global_topk`, `auto_repair`, `speaker_embedder`, `cluster_distance`, `outlier_consistency`, `mode` (`asr` \| `no_asr`), `recover_text`, `recover_asr`, `rate_min`, `rate_max`, `silence_tolerance_s`, `suggest_asr_share` |
| `analyze` | `e2v`, `anime_ser`, `adv`, `events`, `aligner`, `aligner_fallback`, `f0`, `vad`, `caption`, `caption_backend`, `ref_candidates_per_emotion`, `e2v_weight`, `anime_weight` |
| `translate` | `import`, `llm.{backend,fallbacks,ctx,n_gpu_layers,temperature,port,max_retries,parallel,real_backend}`, `batch_lines`, `context_window_lines`, `glossary`, `short_variant_min_words`, `stress.{tool,model,yo,format}`, `interjections`, `mock_dictionary`, `include_unvoiced` (`on` default \| `off` \| `auto`) |
| `text_emotion` | `enabled`, `cues`, `narration_window`, `llm` (`auto` \| `on` \| `off`), `llm_context_lines`, `min_confidence`, `only_unvoiced` (the laughter language is `project.src_lang`) |
| `voicebank` | `refs.*`, `performer.{kind,backend,n_designs,test_sentences,max_f0_delta_semitones,reference_dir,sample_lines,offer_clone}`, `rvc.*`, `route_c.*`, `calibrate_spk`, `min_minutes_for_voicebank` |
| `synth` | `n_candidates`, `n_candidates_short`, `routes` (A, B, C, P), `backends.<name>.{enabled,...}`, `vc.{primary,fallback,rvc,seedvc}`, `keep_original_for_interjections`, `use_short_text_if_ratio_above`, `interjection_max_syllables`, `high_arousal`, `seed_base`, `voice_unvoiced`, `laughter.{enabled,table}` |
| `score` | `weights`, `gates.{utmos_min,cer_max,dur_ratio,spk_sim_min,spk_sim_default}`, `mos`, `mos_fallback`, `aesthetics`, `speaker`, `speaker_extra`, `cer_asr`, `cer_asr_fallback`, `emotion`, `adv`, `regen_rounds`, `regen_cer_max`, `interjection_cer_max` |
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

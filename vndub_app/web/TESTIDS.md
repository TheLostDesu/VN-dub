# data-testid reference of the VNDub SPA

For the Playwright walk of the wizard (Steps 1 -> 7) and for any other UI test. Every key control has a
`data-testid`. Dynamic ids contain a placeholder in `<angle brackets>`; the value comes from the data
(`<sid>` speaker id such as `haruto`, `<id>` line id, `<label>` candidate label `A`/`B`/`C`).

State is exposed as attributes where a test needs to wait for it:

| Attribute | Where | Values |
|---|---|---|
| `data-state` | every `job-progress` root (`*-job`, `job-progress`) | `queued running paused done failed cancelled interrupted` |
| `data-kind` | same | job kind (`import`, `translate`, `preview`, `full`, ...) |
| `data-chosen` | `char-<sid>`, `design-<sid>-<design_id>` | `true` / `false` |
| `data-status` | `doctor-row-<id>`, `review-row-<id>` | `green yellow red` / `ok check failed` |
| `data-playing`, `data-ready` | every `WavePlayer` root | `true` / `false` |
| `data-value` | every `[role=progressbar]` | percent 0..100 or empty (indeterminate) |
| `data-line` | `review-detail` | id of the shown line |

Routes (HashRouter): `/#/step/<doctor|game|translate|casting|preview|run|review|install>`, `/#/models`, `/#/tools`,
`/#/settings`, `/#/help[/<id>]`. `/` opens the step stored in the project (first run: Diagnostics).
Each step screen has `screen-<step>` (`screen-game`, ...).

## The wizard walk for the Playwright test (mock profile, `tests/fixtures/renpy_mini`)

Browser mode: `POST /api/system/pick-folder` answers `unsupported`, so the SPA shows a path field.

1. Open `/#/step/game` (go straight there: `/` would show Diagnostics first on a fresh browser profile).
   Click `game-pick-folder`, fill `game-path-input` with the game folder, click `game-path-open` (or press Enter).
   The import starts by itself: wait for `game-summary` (or `game-job` with `data-state=done`).
   Check `stat-lines`, `stat-voiced`, `stat-unvoiced`, `game-characters-table`. Click `nav-next`.
2. Step 2: click `translate-run-button`, wait for `translate-done-chip`, click `nav-next`.
3. Step 3: click `casting-auto` (a GPU job of kind `casting_auto`: it proposes designs for every character that has none,
   picks the top-ranked voice for everybody and queues RVC training). Wait for `casting-auto-job` `data-state=done`;
   `casting-progress` then reads "N из N" and every `char-<sid>` has `data-chosen=true`; click `nav-next` (disabled while the
   job runs). With a character left unchosen `nav-next` opens `casting-confirm`; `casting-confirm-auto` starts the same job
   and goes on to Step 4 by itself when it is done.
4. Step 4: click `preview-run`, wait for `preview-job` `data-state=done` and `preview-items`, click `preview-approve`.
5. Step 5: the full run starts by itself (the preview approval carries `autostart`). Wait for `run-done`;
   click `run-to-review` (or `nav-next`).
6. Step 6: wait for `review-detail`; optional keys: `3` plays variant B, `Enter` accepts the selected variant and moves on.
   Click `review-apply`, wait for `review-apply-result`, click `nav-next`.
7. Step 7: click `install-apply`, wait for `install-done` (it only appears when the job state is `done`).
   For engines without automatic install `install-manual` is shown instead and `install-apply` does not exist.

## Shell

| id | what |
|---|---|
| `nav-home` | logo / back to the wizard |
| `project-name` | current project name in the top bar |
| `nav-wizard`, `nav-models`, `nav-settings`, `nav-help`, `nav-doctor` | top bar links |
| `topbar-job` | chip with the active job (title, stage, percent); a button that opens the job's screen when it has one |
| `stepper` | the step list; `step-<doctor|game|translate|casting|preview|run|review|install>` buttons (`aria-current=step` on the current one) |
| `nav-back`, `nav-next` | step footer buttons (`nav-next` is `preview-approve` on Step 4); `nav-next-hint` explains a disabled Next |
| `active-job-bar`, `active-job-show` | banner for a job running on another screen; "show" opens the page that has the job (wizard step, `/models`, `/tools`, `/#/step/doctor` for ffmpeg / Visual C++, `/settings` for the problem report); the bar is hidden on that page |
| `resume-dialog`, `resume-continue`, `resume-dismiss` | "Продолжить с того места, где остановились?" (continues the job named in `project.resume.job`: the newest interrupted heavy one) |
| `resume-keep-shutdown` | checkbox "Выключить компьютер по завершении" in that dialog, only when the interrupted job had `shutdown_when_done`; unchecked by default, a continue without it drops the shutdown, ticked it sends `keep_shutdown: true` |
| `resume-candidates` | line "Ещё прерваны: ..." (one title per kind) when `resume.candidates` is not empty; those jobs stay interrupted and are offered next time |
| `toasts` | live region of short messages |
| `splash` | start screen while settings load / when the server is unreachable |
| `app-error`, `app-error-text`, `app-error-reload` | the card that replaces a screen which threw while rendering (`app-error-text` is the error message, `app-error-reload` reloads the window); a navigation clears it |
| `settings-paths` | Settings: the "Folders" card; rows `settings-path-<data|models|envs|tools>` with the path in `settings-path-<key>-value`; `settings-open-<models|envs|tools>` opens the folder; `settings-change-<models|envs>` picks a new one (dialog `settings-path-dialog`, `settings-path-confirm`; without a native dialog the text field `settings-path-input` + `settings-path-apply`); `settings-reset-<models|envs>` goes back to the default |

## Shared pieces

| id | what |
|---|---|
| `error-panel` (`data-code`) | any error; inside: `error-title`, `error-hint`, `error-hf-page` (link to the model page for `hf_gated` / `hf_not_found`), `error-action`, `error-retry`, `error-dismiss`, `error-details-toggle`, `error-details` (technical text, collapsed) |
| `manual-download-card`, `manual-copy-url`, `manual-open-data` | manual download card (unreachable download) |
| `manual-target-dir`, `manual-copy-target`, `manual-open-target` | same card when the error carries `params.target_dir` ("put the file into this folder") |
| `report-problem`, `report-button`, `report-done`, `report-open-folder` | "Собрать отчёт о проблеме" |
| `job-progress` and variants `game-job`, `translate-job`, `preview-job`, `run-job`, `models-download-job`, `models-hf-job`, `models-check-job`, `tools-install-job`, `envs-install-job`, `install-apply-job`, `install-rollback-job`, `install-zip-job`, `review-apply-job`, `casting-gen-job-<sid>`, `casting-listen-job-<sid>` | job panel; inside: `job-title`, `job-stage`, `job-bar`, `job-overall`, `job-counter`, `job-eta`, `job-rate`, `job-item`, `job-stage-list`, `job-pause`, `job-resume`, `job-cancel`, `job-continue`, `job-warnings`, `job-log-toggle`, `job-log`, `job-log-copy`, `job-control-error` (`job-stage` and `job-overall` are absent once a multi-stage job is `done`: one bar then). `job-warnings` is the warn callout of a run that substituted a model or skipped a part (`result.warnings`: substitution / unavailable / mixed_models / settings_changed), one `li` per warning with `data-kind`; it is shown for active and finished jobs, above the log toggle, and absent when there are none |
| `files-progress`, `file-row-<id>` (`data-state`) | per-file rows of download / install jobs: bytes from `result.files[]`, environments show a state only. Download jobs (`models_download`, `tools_install`, `ffmpeg_download`) count megabytes, `envs_install` environments, `install_zip` megabytes (`job-counter` reads "N of M MB"), `install_apply` / `install_rollback` a percent only (their totals are not a file count: the backup pass and the copy pass share one); `job-rate` ("реплик в минуту") is only shown for line-based jobs |
| `save-indicator` (`data-status`) | autosave state (`saving`, `saved`, `error`) |
| players | `<base>` root with `<base>-play` (button), `<base>-volume` (range): `clip-<sid>-<neutral|emotional|longest>`, `sample-<sid>-<design_id>-<n>`, `review-original`, `candidate-player-<label>`, `library-player-<voice id>`, `listen-<sid>-<line_id>` (A/B: `-a`, `-b`, `-seq`), `preview-ab-<id>` (same suffixes) |

## Step 0 Diagnostics (`screen-doctor`)

`doctor-title`, `doctor-summary`, `doctor-gpu-warning`, `doctor-rows`, `doctor-row-<gpu|driver|vram|disk|ffmpeg|internet|defender|envs|models>`,
`doctor-fix-<action>` (`download_ffmpeg`, `open_data_dir`, `open_driver_page`, `add_defender_exclusion_help`, `open_tools`, `open_models`),
`doctor-refresh`, `defender-help` (dialog with `defender-settings-path`, `defender-path`, `defender-copy-path`, `defender-command` (the exact PowerShell line), `defender-copy-command`).

## Step 1 Game (`screen-game`)

| id | what |
|---|---|
| `game-title`, `game-picker` | heading, picker card |
| `game-pick-folder` | "Выбрать папку игры" (native dialog; `unsupported` in a browser) |
| `game-manual-toggle`, `game-path-input`, `game-path-open` | manual path entry |
| `game-recent`, `game-recent-<slug>` | recent games |
| `game-card`, `game-name`, `game-engine` | open game, engine chip |
| `game-import` | "Разобрать игру" (shown only when nothing was imported and no import job runs) |
| `game-job` | import job panel (stage label "Слушаю интонации / читаю текст" for analyze + text_emotion) |
| `game-tools-missing`, `tool-missing-<tool>`, `tool-install-<tool>`, `tools-job`, `game-reimport`, `krkrdump-guide`, `krkrdump-copy-url` | missing tools (install button or manual path), 3-step KrkrDump guide |
| `game-summary` | whole result block |
| `game-engine-warnings`, `game-encrypted`, `game-drm`, `game-engine-warning` | plain-words warnings |
| `game-stats`, `stat-lines`, `stat-voiced`, `stat-unvoiced`, `stat-voice-files`, `game-analysis`, `game-mode-b` | numbers; `game-mode-b` is the "no voice acting" message |
| `game-mapping`, `mapping-bar`, `mapping-confidence`, `mapping-text` | mapping confidence |
| `game-asr-suggest`, `game-asr-check` | ASR check offer when the mapping looks wrong |
| `game-no-text`, `game-recover-text` | mode C: "Распознать текст из озвучки" |
| `game-characters`, `game-characters-table`, `game-char-<sid>` (`data-mode`), `game-unvoiced-list` | characters, voiced / unvoiced split |
| `game-problems-toggle`, `game-problems` | collapsible "Проблемные файлы" |
| `game-advanced-toggle`, `game-asr-check-advanced`, `game-reimport-advanced` | Advanced |
| `game-empty` | import finished but nothing was found |
| `game-import-stale`, `game-import-stale-run` | warn callout + button when the game was imported by an older version of the app (`project.stale`) |

## Step 2 Translation (`screen-translate`)

`translate-title`, `translate-lang`, `translate-source` (`translate-source-auto`, `translate-source-import`),
`translate-no-llm`, `translate-open-models`, `translate-import-box`, `translate-import-path`, `translate-pick-file`,
`translate-pick-folder`, `translate-preview-button`, `translate-preview` (5-line table),
`translate-honorifics` (`honorifics-drop|keep|adapt`), `translate-address` (`address-ty`, `address-vy`),
`translate-license` (`license-research`, `license-commercial`),
`translate-glossary`, `glossary`, `glossary-characters`, `glossary-char-<sid>`, `glossary-tgt-<sid>`, `glossary-gender-<sid>`,
`glossary-terms`, `glossary-term-src-<i>`, `glossary-term-tgt-<i>`, `glossary-term-remove-<i>`, `glossary-add-term`,
`translate-advanced-toggle`, `translate-advanced`, `advanced-n-candidates`, `advanced-route-<A|B|C>`,
`advanced-backends`, `advanced-backend-<name>`, `advanced-asr-check`, `cues-toggle`, `cues-editor`, `cues-filter`, `cues-table`, `cues-save`,
`translate-run`, `translate-run-button`, `translate-job`, `translate-done-chip`, `translate-stale`.

## Step 3 Casting (`screen-casting`)

| id | what |
|---|---|
| `casting-title`, `casting-toolbar`, `casting-auto`, `casting-progress` | global bar; progress text "Голос выбран у N из M" |
| `casting-auto-job` (`data-state`, `data-kind=casting_auto`), `casting-auto-result` | automatic casting job panel (stays after it is done) and its summary line |
| `casting-tab-characters`, `casting-tab-library` | tabs |
| `casting-characters`, `casting-no-characters` | list / empty state |
| `char-<sid>` (`data-chosen`, `data-mode`), `char-toggle-<sid>` (expand), `char-chosen-<sid>` ("голос выбран"), `rvc-chip-<sid>` | character card |
| `char-gender-<sid>`, `char-age-<sid>`, `char-hist-<sid>`, `char-clips-<sid>`, `clip-<sid>-<kind>` | left side |
| `char-noaudio-<sid>`, `char-description-<sid>` | characters without original audio |
| `designs-<sid>`, `design-<sid>-<design_id>`, `design-choose-<sid>-<design_id>`, `design-stars-<sid>-<design_id>` (`data-stars`), `sample-<sid>-<design_id>-<n>` | proposed performers |
| `casting-more-<sid>` ("+ Ещё 3"), `casting-regen-<sid>`, `casting-show-all-<sid>`, `casting-upload-<sid>`, `casting-upload-input-<sid>` (file input), `casting-empty-<sid>` | actions |
| `clone-box-<sid>`, `casting-clone-<sid>` | "Клонировать напрямую (маршрут A)" |
| `casting-listen-<sid>`, `listen-results-<sid>` | "Прослушать, как будет звучать" |
| `casting-save-voice-<sid>`, `casting-voice-name-<sid>`, `casting-voice-save-<sid>` | save to the voice library |
| `char-<sid>-tuning` (+ `-toggle`, `-pitch`, `-speed`, `-expressiveness`, `-energy`, `-timbre-<younger|older|softer|sharper>`, `-note`, `-listen`, `-reset`, `-apply-similar`, `-tuning-results`) | Step 3b panel under the chosen performer |
| `voice-library`, `library-empty`, `library-item-<id>`, `library-target-<id>`, `library-use-<id>`, `library-player-<id>` | "Библиотека голосов" |
| `casting-confirm`, `casting-confirm-back`, `casting-confirm-auto` | dialog when Next is pressed with unchosen characters |

## Step 4 Preview (`screen-preview`)

`preview-title`, `preview-setup`, `preview-mode` (`preview-mode-scene`, `preview-mode-n`), `preview-n`, `preview-run`, `preview-job`,
`preview-items`, `preview-item-<id>`, `preview-ab-<id>`, `preview-empty`, `preview-decision`,
`preview-change-voices`, `preview-change-settings`, `preview-approve` ("Мне нравится, запускай всё" = the Next button).

## Step 5 Full run (`screen-run`)

`run-title`, `run-setup`, `run-shutdown` (checkbox, default off), `run-shutdown-warning`, `run-start`, `run-closable`,
`run-progress`, `run-job` (with `job-stage-list`, `job-log`), `run-gpu` (only while `gpu` events arrive), `run-done`, `run-to-review`,
`run-coverage` (callout under `run-done`: what the patch holds; warn tone when something is provisional or left out; inside `run-coverage-installed`,
`run-coverage-provisional` + `run-coverage-review` (button to the review), `run-coverage-left-out`, `run-coverage-narration`, `run-coverage-own-voice` (Ren'Py lines with a `voice` statement of their own, not installable); the last four only when the number is above 0),
`run-stale-import` + `run-stale-import-go` (warn callout when the project was imported by an older version; button to Step 1).

## Step 6 Review (`screen-review`)

| id | what |
|---|---|
| `review-title`, `review-toolbar` | |
| `review-filter-speaker`, `review-filter-scene`, `review-filter-status`, `review-filter-min`, `review-filter-max`, `review-search` | filters |
| `review-prefer-a`, `review-apply`, `review-apply-job`, `review-apply-result` | bulk "Пометить всех, как A..." and "Применить изменения" (the badge is the server's count of every touched line, whatever the filters show; a running `review_apply` is picked up again when the screen is reopened) |
| `review-counts`, `review-list`, `review-row-<id>` (`data-status`), `review-empty` | list (virtualized: only rows near the viewport exist in the DOM) |
| `review-prev`, `review-next` | previous / next line |
| `review-detail` (`data-line`), `review-detail-status`, `review-why`, `review-asr-badge`, `review-delivery-chip`, `review-detail-patch` | selected line (`review-detail-patch`: "в патче как предварительная" / "нет в патче", absent for a line that is in the patch normally) |
| `review-row-patch-<id>` | chip in a list row: "предварительная" (the line is in the patch with a take that failed the checks) or "нет в патче" |
| `review-text-src` (only for `text_from_asr` lines: editable) / `review-text-src-readonly`, `review-src-edit`, `review-save-src` | source text |
| `review-text-tgt`, `review-save-text` | translated text |
| `review-original`, `review-candidates`, `candidate-<label>` (`data-selected`), `candidate-label-<label>`, `candidate-player-<label>`, `candidate-verdicts-<label>`, `verdict-<voice|emotion|clarity|length|quality>` (`data-level`), `candidate-accept-<label>`, `review-no-candidates` | original and variants |
| `review-actions`, `review-accept`, `review-regen`, `review-reject`, `review-keep-original`, `review-notice` | actions (accepting the last row of the list reloads the detail instead of moving on) |
| `review-style-toggle`, `review-style`, `review-refs-toggle`, `review-use-ref` | style text, other reference sample |
| `review-line-tuning` (+ `-toggle`, `-pitch`, `-speed`, `-expressiveness`, `-energy`, `-timbre-*`, `-note`, `-reset`), `review-emotion`, `review-line-tuning-apply` | "Только для этой реплики" (the sliders start from the line's own tuning, else from the character's; reset returns to the character's) |

Keyboard (not in a text field): `Space` play / pause the last used player, `1` original, `2` A, `3` B, `4` C,
`Enter` accept the selected variant (and go to the next line), `R` regenerate, arrow keys previous / next line.
They use physical keys (`event.code`), so they also work on the Russian layout.

## Step 7 Install (`screen-install`)

`install-title`, `install-plan`, `install-engine`, `install-engine-steps` (plain-language, engine-specific explanation from i18n `install.engine.<engine>.steps`),
`install-md-toggle` (collapsible "Подробная техническая инструкция (на английском)"), `install-md` (rendered INSTALL.md; it stays in the DOM, hidden, while the section is collapsed), `install-files-toggle` (the title counts every file of the patch: `plan.file_count`), `install-files` (the first 300 only), `install-files-truncated` (hint above the list when the plan holds more files than it lists),
`install-notes` (info callout with the plan `notes`: `install-note-<unvoiced_manual|encrypted>`, and the plan `helpers`: `install-helper-<name>` with `data-in-game`; absent when there are none),
`install-manual` (manual engines: never a success message), `install-actions`, `install-backup` (checkbox, default on),
`install-apply`, `install-apply-job`, `install-done`, `install-backup-select`, `install-no-backups`, `install-rollback`,
`install-rollback-dialog`, `install-rollback-confirm`, `install-rollback-job`, `install-rollback-partial` (warn callout under a finished rollback that could not restore some files: they were replaced without a backup), `install-open-patch`, `install-zip`, `install-zip-job`,
`install-zip-done`, `install-zip-open`, `install-coverage` (three chips beside the file count: `install-coverage-installed`, `install-coverage-provisional`, `install-coverage-left-out`; absent for a patch built before the coverage existed).

## Models and tools (`/#/models`, `/#/tools`)

`models-title`, `tab-models`, `tab-tools`, `models-presets`, `models-preset-<min|recommended|all>`, `models-table-card`, `models-table`,
`model-row-<name>` (`data-present`), `model-check-<name>` (checkbox), `model-selftest-<name>`, `models-total`, `models-no-space`,
`models-download`, `models-download-job`, `files-progress`, `models-check`, `models-check-job`,
`models-env-needed` (shown when a selected weight's row has `reason: env_missing`; those rows are left out of download and check), `models-install-envs` (starts `POST /api/envs/install` for the needed envs), `models-open-tools`, `models-envs-job`;
`model-recommended-<name>` (chip "Рекомендуется" on the rows of the `recommended` preset);
**Add a model by link** (card above the table): `models-hf-card`, `models-hf-url` (text field, placeholder `https://huggingface.co/owner/model`), `models-hf-inspect` ("Проверить"),
`models-hf-recommended` + `models-hf-reco-<i>` (`data-repo`; clicking one only fills `models-hf-url`), `models-hf-result` (`data-kind`: `builtin|gguf_llm|rvc|files`, `data-repo`),
`models-hf-kind` (chip), `models-hf-warn-<code>` (`gated`, `license_check`, `unknown_license`, `too_big`, `not_text_model`), `models-hf-builtin` ("Показать в списке": selects and scrolls to `model-row-<builtin>`),
`models-hf-files` (checklist), `models-hf-file-<i>` (checkbox of the i-th file of the repo listing), `models-hf-pick-suggested`, `models-hf-pick-none`, `models-hf-total`, `models-hf-no-space`,
`models-hf-add` ("Скачать"), `models-hf-job` (job panel, kind `models_hf_add`, with `files-progress`), `models-hf-done`;
added models: `models-custom-list` (`<ul>`; `models-custom-empty` instead when there are none), `models-custom-row-<id>` (`data-kind`, `data-present`), `models-custom-delete-<id>`,
`models-custom-delete-dialog`, `models-custom-delete-confirm`, `models-custom-in-use` (delete refused: a job uses it);
`tools-card`, `tools-table`, `tool-row-<name>` (`data-present`), `tools-install`, `tools-install-job`, `tools-manual-toggle`,
`envs-card`, `envs-list`, `env-row-<name>`, `envs-torch`, `envs-torch-<auto|cuda|cpu>` (radio buttons), `envs-install`, `envs-install-job`.

**UI persistence** (leaving a screen must not lose its work; `tests/app/test_ui_persistence_e2e.py`): `/#/models` and `/#/tools` are two
routes, so switching the tab unmounts the tab. Jobs live on the server and in the global progress store, therefore the job panels
(`models-download-job`, `models-check-job`, `models-envs-job`, `tools-install-job`, `envs-install-job`, `models-hf-job` and `models-hf-done`,
`game-job`, `install-*-job`, the Diagnostics fix job, `report-done`) come back by themselves when the screen is opened again:
a running job first, else the finished panel (done / failed) the UI has seen in this session. The ticked `model-check-<name>` boxes and the
`envs-torch-*` choice are kept in memory until the page is reloaded; the pre-selection of the missing recommended models happens only on the
first visit. A page reload keeps only jobs that are still active on the server.

## Settings and help

`settings-title`, `settings-data`, `settings-data-dir`, `settings-open-data`, `settings-change-data`, `settings-data-path`,
`settings-data-apply`, `settings-data-dialog`, `settings-data-confirm`, `settings-language`, `settings-lang` (select `ru`/`en`),
`settings-gpu`, `settings-gpu-budget`, `settings-translation-llm` (card), `settings-translation-llm-select` (value `""` = default, a built-in `llm` backend name or `hf_<slug>`), `settings-license`, `settings-license-policy`, `settings-license-research`,
`settings-license-commercial`, `settings-report`, `settings-log` (the Log card: `settings-debug-log` checkbox, `settings-open-logs`), `settings-version`;
`help-title`, `help-nav`, `help-doc-<id>`, `help-body`, `help-markdown`.

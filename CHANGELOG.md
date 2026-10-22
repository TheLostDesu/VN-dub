# Changelog

| Version | What was added |
|---------|----------------|
| 0.2.0 | The installer carries everything it needs: VC++ runtime, MinGit, bgi-extractor; adapters start from the right environment; smoke test of the exe |
| 0.3.0 | Round 3: processes end when the window closes, the interface keeps jobs across tabs, install folders, one shared torch environment `main` (an incompatible change: old environments have to be recreated) |
| 0.4.0 | Round 4: the script's language is detected, diagnostics for empty model answers, batched adapters |
| 0.5.0 | Round 5: models don't silently end up running from system RAM |
| 0.6.0 | Round 6: the Cancel button works everywhere: model processes are killed at once, the translate stage has progress, pause and cancel, the button shows "Cancelling..." |
| 0.6.1 | Round 6.1: translation runs once (step 2); the preview and the full run no longer translate again |
| 0.6.2 | Round 6.2: the time estimate is honest: the speed is the average since the stage began (model loading and the wait for a batch count), shown in lines a minute; cached results and pauses no longer skew it, stages are added up per line, the first estimate comes from the previous run |
| 0.6.3 | Round 7: model processes never outlive the app (process ledger, startup sweep, a reaper that survives a crash, no adapter restart after Cancel), every job log line has a time and the whole log is kept in a file, app.log tells what the app did (jobs, stages, processes, errors, settings; a detailed mode in Settings), the interface is aligned, a job chip in the top bar, the window title and taskbar say when a job is done |
| 0.6.4 | Round 8: a model that does not fit is a visible warning, not a silent downgrade (synth stops instead of deleting candidates, score sends unchecked lines to review, the translate cache knows which model answered); an interrupted run keeps its score metrics and route-B voices; Continue picks the heavy job, keeps the other interrupted ones, drops the shutdown flag unless re-ticked and names changed settings; the speed estimate no longer collapses after Continue |

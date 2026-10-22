# Changelog

| Version | What was added |
|---------|----------------|
| 0.2.0 | The installer carries everything it needs: VC++ runtime, MinGit, bgi-extractor; adapters start from the right environment; smoke test of the exe |
| 0.3.0 | Round 3: processes end when the window closes, the interface keeps jobs across tabs, install folders, one shared torch environment `main` (an incompatible change: old environments have to be recreated) |
| 0.4.0 | Round 4: the script's language is detected, diagnostics for empty model answers, batched adapters |
| 0.5.0 | Round 5: models don't silently end up running from system RAM |
| 0.6.0 | Round 6: the Cancel button works everywhere: model processes are killed at once, the translate stage has progress, pause and cancel, the button shows "Cancelling..." |
| 0.6.1 | Round 6.1: translation runs once (step 2); the preview and the full run no longer translate again |

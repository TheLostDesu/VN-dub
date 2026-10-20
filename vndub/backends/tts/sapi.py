"""Windows SAPI 5 voices through PowerShell ``System.Speech`` (test/dev only).

Not a dubbing-quality backend: it exists so that the CPU test profile on
Windows produces real Russian speech (which makes CER/UTMOS meaningful) and so
``scripts/make_fixtures.py`` can voice the Japanese fixtures with an installed
ja-JP voice. One PowerShell process handles a whole batch (startup is ~1 s).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from vndub.backends.base import BackendSpec, BackendUnavailable, LocalBackend

SPECS = [
    BackendSpec(
        name="sapi_tts",
        role="tts",
        env=None,
        adapter="sapi",
        license="Windows component (test use only)",
        commercial_ok=False,
        vram_estimate_gb=0.0,
        heavy=False,
        cpu_ok=True,
        profiles=("test",),
        platforms=("win32",),
        local_impl="vndub.backends.tts.sapi:SapiTTS",
        mock="mock_tts",
        caps={"instruct": "none", "tags": "none", "knobs": "sapi", "duration": False, "stress_format": "plain", "design": True, "sr": 22050, "langs": ["ru", "ja", "en"]},
        description="Windows SAPI voices (Irina ru-RU, Haruka ja-JP ...): real speech on CPU for tests",
    ),
]

_CULTURE = {"ru": "ru-RU", "ja": "ja-JP", "en": "en-US"}

_PS_SCRIPT = r"""
param([string]$JobFile)
Add-Type -AssemblyName System.Speech
$jobs = Get-Content -Raw -Encoding UTF8 $JobFile | ConvertFrom-Json
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voices = $synth.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo }
foreach ($j in $jobs) {
  try {
    $v = $voices | Where-Object { $_.Culture.Name -eq $j.culture } | Select-Object -First 1
    if ($j.voice) { $named = $voices | Where-Object { $_.Name -like "*$($j.voice)*" } | Select-Object -First 1; if ($named) { $v = $named } }
    if (-not $v) { Write-Output ("ERR`t" + $j.id + "`tno voice for " + $j.culture); continue }
    $synth.SelectVoice($v.Name)
    $fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(22050, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
    $synth.SetOutputToWaveFile($j.out, $fmt)
    $text = [System.Security.SecurityElement]::Escape($j.text)
    $ssml = "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='" + $j.culture + "'><prosody pitch='" + $j.pitch + "' rate='" + $j.rate + "' volume='" + $j.volume + "'>" + $text + "</prosody></speak>"
    $synth.SpeakSsml($ssml)
    $synth.SetOutputToNull()
    Write-Output ("OK`t" + $j.id + "`t" + $v.Name)
  } catch { $synth.SetOutputToNull(); Write-Output ("ERR`t" + $j.id + "`t" + $_.Exception.Message) }
}
"""


def sapi_available() -> bool:
    return sys.platform.startswith("win")


def synthesize_batch(jobs: list[dict[str, Any]], timeout: float = 600) -> dict[str, tuple[bool, str]]:
    """jobs: {id, text, lang, out, pitch ('+0%'), rate ('medium'|'+10%'), volume ('medium'), voice?}."""
    if not sapi_available():
        raise BackendUnavailable("sapi_tts", "Windows SAPI is only available on Windows")
    payload = []
    for j in jobs:
        Path(j["out"]).parent.mkdir(parents=True, exist_ok=True)
        payload.append(
            {
                "id": j["id"],
                "text": j["text"],
                "culture": _CULTURE.get(j.get("lang", "ru"), j.get("lang", "ru-RU")),
                "out": str(Path(j["out"]).resolve()),
                "pitch": j.get("pitch", "+0%"),
                "rate": j.get("rate", "medium"),
                "volume": j.get("volume", "medium"),
                "voice": j.get("voice", ""),
            }
        )
    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "sapi.ps1"
        script.write_text(_PS_SCRIPT, encoding="utf-8-sig")
        job_file = Path(td) / "jobs.json"
        job_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(script), str(job_file)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
        )
    results: dict[str, tuple[bool, str]] = {}
    for line in r.stdout.splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 3:
            results[parts[1]] = (parts[0] == "OK", parts[2])
    for j in jobs:
        results.setdefault(j["id"], (False, (r.stderr or "no output").strip()[-300:]))
    return results


def _pitch_for(knobs: dict[str, Any]) -> str:
    # SAPI pitch is relative; map the requested F0 against a ~200 Hz female / 120 Hz male default
    f0 = knobs.get("f0_hz")
    if not f0:
        return "+0%"
    base = 200.0 if knobs.get("gender", "f") != "m" else 120.0
    pct = max(-50, min(50, int(round((float(f0) / base - 1.0) * 100))))
    return f"{pct:+d}%"


class SapiTTS(LocalBackend):
    def load(self) -> dict[str, Any]:
        if not sapi_available():
            raise BackendUnavailable(self.spec.name, "Windows SAPI is only available on Windows")
        return super().load()

    def call(self, op: str, items: list[dict[str, Any]] | None = None, **kwargs: Any) -> Any:
        if op not in ("synthesize", "design"):
            return super().call(op, items, **kwargs)
        jobs = []
        for it in items or []:
            knobs = it.get("knobs") or {}
            rate = {"fast": "fast", "slow": "slow"}.get(knobs.get("tempo", ""), "medium")
            volume = {"loud": "loud", "soft": "soft"}.get(knobs.get("loudness", ""), "medium")
            jobs.append({"id": it["id"], "text": it["text"], "lang": it.get("lang", "ru"), "out": it["out"], "pitch": _pitch_for(knobs), "rate": rate, "volume": volume, "voice": knobs.get("sapi_voice", "")})
        res = synthesize_batch(jobs)
        out = []
        for it in items or []:
            ok, detail = res[it["id"]]
            if ok and Path(it["out"]).exists():
                out.append({"id": it["id"], "path": it["out"], "sr": 22050, "voice": detail})
            else:
                out.append({"id": it["id"], "error": f"sapi: {detail}"})
        return out

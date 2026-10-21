"""Higgs TTS 3 4B through a *local* SGLang-Omni server (research & non-commercial license).

Higgs Audio v3 is released for SGLang-Omni (Linux; on Windows run it in WSL2 or
Docker, on this machine, with the 4-bit weights from ``models/higgs-audio-v3-tts-4b``).
This adapter is the client: ``POST <server>/v1/audio/speech`` with
``{"input", "references": [{"audio_path", "text"}], "temperature", "max_new_tokens"}``.
Only loopback addresses are accepted, so nothing leaves the machine. Delivery
tags (``<|emotion:...|>``, ``<|sfx:...|>``) arrive in ``text_prefix`` from the
style renderer. 8 GB fit of the 4-bit model is UNVERIFIED.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from adapters.common.rpc import TTSImpl, UnavailableError, load_audio

LOOPBACK = {"127.0.0.1", "localhost", "::1"}


class HiggsTTS3(TTSImpl):
    def load(self) -> dict[str, Any]:
        self.url = str(self.options.get("server_url", "http://127.0.0.1:8000")).rstrip("/")
        host = urllib.parse.urlparse(self.url).hostname or ""
        if host not in LOOPBACK:
            raise UnavailableError(f"higgs_tts3 server must be local (got {host}); vndub never calls remote services")
        try:
            urllib.request.urlopen(self.url + "/health", timeout=5)
        except (urllib.error.URLError, OSError) as e:
            if not isinstance(e, urllib.error.HTTPError):
                raise UnavailableError(f"no SGLang-Omni server at {self.url}: {e}", "start: sgl-omni serve --model-path models/higgs-audio-v3-tts-4b --port 8000 (WSL2/Docker)") from e
        self.device = "server"
        return {"server": self.url}

    def generate(self, item: dict[str, Any]):
        body: dict[str, Any] = {
            "input": (item.get("text_prefix") or "") + item["text"],
            "temperature": float(self.options.get("temperature", 0.7)),
            "max_new_tokens": int(self.options.get("max_new_tokens", 1024)),
        }
        if item.get("seed") is not None:
            body["seed"] = int(item["seed"])
        if item.get("timbre_ref"):
            ref = {"audio_path": str(Path(item["timbre_ref"]).resolve())}
            if item.get("timbre_ref_text"):
                ref["text"] = item["timbre_ref_text"]
            body["references"] = [ref]
        req = urllib.request.Request(self.url + "/v1/audio/speech", data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=float(self.options.get("timeout_s", 600))) as r:
            data = r.read()
        Path(item["out"]).parent.mkdir(parents=True, exist_ok=True)
        Path(item["out"]).write_bytes(data)
        return load_audio(item["out"], None)

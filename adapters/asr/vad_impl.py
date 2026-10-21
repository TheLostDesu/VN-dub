"""silero-vad through the ONNX model bundled with faster-whisper (CPU)."""

from __future__ import annotations

from typing import Any

from adapters.common.rpc import ModelImpl, load_audio

SR = 16000


class SileroVAD(ModelImpl):
    selftest_op = "vad"

    def load(self) -> dict[str, Any]:
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        self._get = get_speech_timestamps
        self._opts = VadOptions(
            threshold=float(self.options.get("threshold", 0.5)),
            min_silence_duration_ms=int(self.options.get("min_silence_duration_ms", 120)),
            speech_pad_ms=int(self.options.get("speech_pad_ms", 20)),
        )
        self.device = "cpu"
        return {"engine": "faster_whisper.vad (silero ONNX)"}

    def op_vad(self, item: dict[str, Any], **_: Any) -> dict[str, Any]:
        y, _ = load_audio(item["wav"], SR)
        ts = self._get(y, vad_options=self._opts)
        segs = [[round(t["start"] / SR, 3), round(t["end"] / SR, 3)] for t in ts]
        return {"segments": segs, "duration": round(len(y) / SR, 3)}

"""SenseVoice-Small: audio-event and emotion tags read from the raw output.

Raw output looks like ``<|ja|><|HAPPY|><|Laughter|><|woitn|>text``; tags are parsed
*before* ``rich_transcription_postprocess`` (which strips them). Inputs are
limited to 30 s; longer clips are truncated (events in a VN line are near the
start anyway) and flagged.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any

from adapters.common.rpc import ModelImpl, load_audio, save_audio, weight_dir
from adapters.funasr.e2v_impl import funasr_model

EVENTS = {"Laughter": "laughter", "Cry": "cry", "Breath": "breath", "BGM": "bgm", "Cough": "cough", "Sneeze": "sneeze", "Applause": "applause"}
EMOTIONS = {"HAPPY": "happy", "SAD": "sad", "ANGRY": "angry", "NEUTRAL": "neutral", "FEARFUL": "fearful", "DISGUSTED": "disgusted", "SURPRISED": "surprised", "EMO_UNKNOWN": "unknown"}
_TAG = re.compile(r"<\|([^|]+)\|>")


def parse_tags(raw: str) -> dict[str, Any]:
    tags = _TAG.findall(raw or "")
    events = [EVENTS[t] for t in tags if t in EVENTS]
    emo = next((EMOTIONS[t] for t in tags if t in EMOTIONS), None)
    lang = tags[0] if tags else None
    return {"events": sorted(set(events), key=events.index), "emotion": emo, "lang_tag": lang, "tags": tags}


class SenseVoice(ModelImpl):
    selftest_op = "events"

    def load(self) -> dict[str, Any]:
        d = weight_dir(self.params, 0)
        self.model = funasr_model(d, self.device)
        return {"model_dir": str(d)}

    def op_events(self, item: dict[str, Any], **_: Any) -> dict[str, Any]:
        max_s = float(self.options.get("max_sec", 30.0))
        y, sr = load_audio(item["wav"], 16000)
        truncated = len(y) > max_s * sr
        with tempfile.TemporaryDirectory() as td:
            path = item["wav"]
            if truncated:
                path = str(Path(td) / "clip.wav")
                save_audio(path, y[: int(max_s * sr)], sr)
            res = self.model.generate(input=path, cache={}, language=self.options.get("language", "auto"), use_itn=False, batch_size_s=60)
        raw = (res[0] if isinstance(res, list) else res).get("text", "")
        out = parse_tags(raw)
        try:
            from funasr.utils.postprocess_utils import rich_transcription_postprocess

            out["text"] = rich_transcription_postprocess(raw)
        except ImportError:
            out["text"] = _TAG.sub("", raw)
        out["raw"] = raw
        out["truncated"] = truncated
        return out

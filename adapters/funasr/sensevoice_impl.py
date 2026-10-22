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

from adapters.common.rpc import ModelImpl, audio_seconds, load_audio, load_audios, run_in_batches, save_audio, weight_dir
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

    def batch_events(self, items: list[dict[str, Any]], **kw: Any) -> list[dict[str, Any]]:
        """Several clips per ``AutoModel.generate(input=[paths], batch_size=N)`` call.

        FunASR (no VAD model) cuts the path list into batches of ``batch_size`` samples and runs
        ``SenseVoiceSmall.inference`` once per batch: the frontend computes fbank/LFR/CMVN per clip from its own
        length, the features are zero-padded and the SANM encoder gets the true lengths as a mask (the FSMN conv
        multiplies by the mask), CTC decoding stops at each clip's own output length. So padding does not change
        the tags (``batch_size_s`` is NOT used: it only applies together with a VAD model, which would segment the
        clips). Clips are sorted by duration, cut into sub-batches of ``options['batch_size']`` (default 8), and
        longer than ``max_sec`` clips are truncated into a temp dir exactly like :meth:`op_events`. A failing
        sub-batch (incl. CUDA OOM) is redone clip by clip with :meth:`op_events`.
        """
        return run_in_batches(items, self.sub_batch_size(), self._events_chunk, lambda it: self.op_events(it, **kw), label=self.name, sort_key=audio_seconds)

    def _events_chunk(self, items: list[dict[str, Any]]) -> list[Any]:
        max_s = float(self.options.get("max_sec", 30.0))
        ys, out = load_audios(items)
        ok = [i for i, e in enumerate(out) if e is None]
        if not ok:
            return out
        with tempfile.TemporaryDirectory() as td:
            paths: list[str] = []
            trunc: list[bool] = []
            for i in ok:
                y = ys[i]
                cut = len(y) > max_s * 16000
                path = items[i]["wav"]
                if cut:
                    path = str(Path(td) / f"clip_{i}.wav")
                    save_audio(path, y[: int(max_s * 16000)], 16000)
                paths.append(path)
                trunc.append(cut)
            res = self.model.generate(input=paths, cache={}, language=self.options.get("language", "auto"), use_itn=False, batch_size=len(paths))
        if not isinstance(res, list) or len(res) != len(paths):
            raise RuntimeError(f"SenseVoice returned {len(res) if isinstance(res, list) else type(res).__name__} results for {len(paths)} clips")
        try:
            from funasr.utils.postprocess_utils import rich_transcription_postprocess
        except ImportError:
            rich_transcription_postprocess = None
        for r, i, cut in zip(res, ok, trunc):
            raw = r.get("text", "")
            row = parse_tags(raw)
            row["text"] = rich_transcription_postprocess(raw) if rich_transcription_postprocess else _TAG.sub("", raw)
            row["raw"] = raw
            row["truncated"] = cut
            out[i] = row
        return out

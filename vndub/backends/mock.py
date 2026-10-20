"""Deterministic in-process mock backends for the test profile and CI.

They implement the same ops and result shapes as the real adapters, computed
from cheap DSP features, so every stage runs end to end without downloads.
Results are flagged ``"mock": true`` and stages record the substitution.
"""

from __future__ import annotations

import json
import re
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from vndub.audio import features, vad
from vndub.backends.base import BackendSpec, LocalBackend
from vndub.backends.ser.anime_ser import ANIME_LABELS
from vndub.backends.ser.emotion2vec import E2V_LABELS


def _spec(name: str, role: str, cls: str, description: str, caps: dict | None = None, options: dict | None = None) -> BackendSpec:
    return BackendSpec(
        options=options or {},
        name=name,
        role=role,
        env=None,
        adapter=name,
        license="MIT",
        commercial_ok=True,
        vram_estimate_gb=0.0,
        heavy=False,
        cpu_ok=True,
        profiles=("test",),
        local_impl=f"vndub.backends.mock:{cls}",
        description=description,
        caps=caps or {},
    )


SPECS = [
    _spec("mock_asr", "asr", "MockASR", "returns the text recorded in a synth sidecar, else empty"),
    _spec("mock_aligner", "align", "MockAligner", "spreads characters uniformly over VAD speech"),
    _spec("mock_ser", "ser", "MockSER", "heuristic emotion2vec-style probs + feature embedding"),
    _spec("mock_anime_ser", "ser", "MockSER", "heuristic anime-SER-style probs (10 labels)", options={"labels": "anime"}),
    _spec("mock_adv", "adv", "MockADV", "heuristic arousal/dominance/valence from energy and F0"),
    _spec("mock_events", "events", "MockEvents", "heuristic breath/laughter tags"),
    _spec("mock_embedder", "embedder", "MockEmbedder", "MFCC statistics as a speaker embedding"),
    _spec("mock_mos", "mos", "MockMOS", "MOS-like score from SNR and voicing"),
    _spec("mock_aesthetics", "aesthetics", "MockAesthetics", "PQ-like score from spectral flatness"),
    _spec("mock_caption", "caption", "MockCaption", "template caption from features"),
    _spec("mock_stress", "stress", "MockStress", "no stress marks (identity)"),
    _spec(
        "mock_tts",
        "tts",
        "MockTTS",
        "sine-vocoded placeholder with the right duration and an F0 arc",
        caps={"instruct": "natural", "tags": "none", "knobs": "chatterbox", "duration": True, "stress_format": "plain", "design": True, "sr": 24000, "langs": ["*"]},
    ),
    _spec("mock_vc", "vc", "MockVC", "pitch shift by f0_transpose (no timbre change)", caps={"trainable": True, "zero_shot": True}),
    _spec("mock_translator", "llm", "MockTranslator", "dictionary + passthrough translator speaking the translate JSON protocol"),
]


def _seed_of(*parts: Any) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode("utf-8")) & 0xFFFFFFFF


def _feat(path: str) -> dict[str, float]:
    y, sr = features.load(path, sr=16000)
    e = features.rms_db(y, sr)
    segs = vad.energy_vad(y, sr)
    mask = features.mask_from_segments(e.size, segs)
    f0 = features.f0_pyin(y, sr) if y.size < 16000 * 15 else np.zeros(1)
    st = features.f0_stats(f0)
    snr = features.snr_db(e, mask)
    return {
        "energy": float(np.mean(e[mask])) if mask.any() else float(np.mean(e)),
        "f0_med": float(st["f0_median_hz"] or 0.0),
        "f0_std": float(st["f0_std_st"] or 0.0),
        "voiced": float(st["voiced_ratio"] or 0.0),
        "snr": float(snr if snr is not None else 0.0),
        "tilt": features.spectral_tilt(y, sr),
        "dur": len(y) / sr,
    }


def _softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max())
    return z / z.sum()


class MockASR(LocalBackend):
    def op_transcribe(self, item: dict[str, Any], lang: str = "ja", **_: Any) -> dict[str, Any]:
        side = Path(item["wav"]).with_suffix(".json")
        text = ""
        if side.exists():
            try:
                text = json.loads(side.read_text(encoding="utf-8")).get("text_used", "")
            except (json.JSONDecodeError, OSError):
                text = ""
        return {"text": text, "lang": lang, "mock": True}


class MockAligner(LocalBackend):
    def op_align(self, item: dict[str, Any], **_: Any) -> dict[str, Any]:
        y, sr = features.load(item["wav"], sr=16000)
        segs = vad.energy_vad(y, sr) or [(0.0, len(y) / sr)]
        chars = [c for c in (item.get("text") or "") if not c.isspace()]
        total = sum(b - a for a, b in segs) or 1e-3
        out, i = [], 0
        for a, b in segs:
            n = round(len(chars) * (b - a) / total) if segs[-1] != (a, b) else len(chars) - i
            step = (b - a) / max(1, n)
            for k in range(n):
                if i >= len(chars):
                    break
                out.append({"c": chars[i], "t0": round(a + k * step, 3), "t1": round(a + (k + 1) * step, 3)})
                i += 1
        return {"chars": out, "mock": True}


class MockSER(LocalBackend):
    """Emotion from (energy, pitch variability, tilt); embedding = normalized feature vector."""

    def _probs(self, f: dict[str, float], labels: list[str], happy: str, angry: str, sad: str, neutral: str, surprised: str) -> dict[str, float]:
        logits = np.full(len(labels), -2.0)
        idx = {l: i for i, l in enumerate(labels)}
        arousal = np.clip((f["energy"] + 40) / 30, 0, 1) * 0.6 + np.clip(f["f0_std"] / 4, 0, 1) * 0.4
        logits[idx[neutral]] = 1.5 - 2.0 * arousal
        logits[idx[happy]] = 2.5 * arousal - 0.5 + (0.5 if f["tilt"] > -3 else 0.0)
        logits[idx[angry]] = 2.5 * arousal - 1.0 + (0.6 if f["tilt"] < -3 else -0.3)
        logits[idx[sad]] = 1.2 - 3.0 * arousal
        logits[idx[surprised]] = 2.0 * np.clip(f["f0_std"] / 5, 0, 1) - 0.8
        p = _softmax(logits)
        return {l: round(float(v), 4) for l, v in zip(labels, p)}

    def op_emotion(self, item: dict[str, Any], **_: Any) -> dict[str, Any]:
        f = _feat(item["wav"])
        if self.options.get("labels") == "anime":
            probs = self._probs(f, ANIME_LABELS, "Happy", "Angry", "Sad", "Neutral", "Surprised")
            return {"label": max(probs, key=probs.get), "probs": probs, "mock": True}
        probs = self._probs(f, E2V_LABELS, "happy", "angry", "sad", "neutral", "surprised")
        res: dict[str, Any] = {"probs": probs, "label": max(probs, key=probs.get), "mock": True}
        if item.get("out"):
            vec = np.array([f["energy"] / 40, f["f0_med"] / 300, f["f0_std"] / 4, f["voiced"], f["snr"] / 40, f["tilt"] / 10, *probs.values()], dtype=np.float32)
            out = Path(item["out"])
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, vec)
            res["embedding_path"] = str(out)
        return res


class MockADV(LocalBackend):
    def op_adv(self, item: dict[str, Any], **_: Any) -> dict[str, Any]:
        f = _feat(item["wav"])
        arousal = float(np.clip((f["energy"] + 40) / 30, 0, 1) * 0.6 + np.clip(f["f0_std"] / 4, 0, 1) * 0.4)
        dominance = float(np.clip((f["energy"] + 35) / 30, 0, 1))
        valence = float(np.clip(0.5 + (f["tilt"] + 3) / 20 + (f["f0_med"] - 220) / 800, 0, 1))
        return {"arousal": round(arousal, 4), "dominance": round(dominance, 4), "valence": round(valence, 4), "mock": True}


class MockEvents(LocalBackend):
    def op_events(self, item: dict[str, Any], **_: Any) -> dict[str, Any]:
        f = _feat(item["wav"])
        ev = []
        if f["voiced"] < 0.15 and f["energy"] > -45:
            ev.append("breath")
        return {"events": ev, "raw": "", "mock": True}


class MockEmbedder(LocalBackend):
    def op_embed(self, item: dict[str, Any], **_: Any) -> dict[str, Any]:
        import librosa

        y, sr = features.load(item["wav"], sr=16000)
        if y.size < 2048:
            y = np.pad(y, (0, 2048 - y.size))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        f0 = features.f0_pyin(y, sr) if y.size < 16000 * 15 else np.zeros(1)
        st = features.f0_stats(f0)
        logf0 = np.log((st["f0_median_hz"] or 150.0) / 150.0)
        vec = np.concatenate([mfcc.mean(axis=1) / 50, mfcc.std(axis=1) / 50, [logf0 * 3]]).astype(np.float32)
        out = Path(item["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, vec)
        return {"path": str(out), "dim": int(vec.size), "mock": True}


class MockMOS(LocalBackend):
    def op_mos(self, item: dict[str, Any], **_: Any) -> dict[str, Any]:
        f = _feat(item["wav"])
        score = 1.0 + 4.0 / (1.0 + np.exp(-(f["snr"] - 15) / 6)) * (0.6 + 0.4 * min(1.0, f["voiced"] * 2))
        return {"mos": round(float(np.clip(score, 1, 5)), 3), "mock": True}


class MockAesthetics(LocalBackend):
    def op_aesthetics(self, item: dict[str, Any], **_: Any) -> dict[str, Any]:
        import librosa

        y, sr = features.load(item["wav"], sr=16000)
        flat = float(np.mean(librosa.feature.spectral_flatness(y=y if y.size >= 2048 else np.pad(y, (0, 2048 - y.size)))))
        pq = float(np.clip(9.0 - 12.0 * flat, 1, 10))
        return {"pq": round(pq, 3), "ce": round(pq * 0.8, 3), "cu": round(pq * 0.9, 3), "pc": round(3.0 + flat, 3), "mock": True}


class MockCaption(LocalBackend):
    def op_caption(self, item: dict[str, Any], prompt: str = "", **_: Any) -> dict[str, Any]:
        f = _feat(item["wav"])
        energy = "loud" if f["energy"] > -20 else "soft" if f["energy"] < -35 else "moderate"
        pitch = "lively" if f["f0_std"] > 3 else "steady"
        return {"caption": f"{energy.capitalize()}, {pitch} delivery; {f['dur']:.1f} s.", "mock": True}


class MockStress(LocalBackend):
    def op_stress(self, item: dict[str, Any], fmt: str = "plain", yo: bool = True, **_: Any) -> dict[str, Any]:
        return {"text": item["text"], "mock": True}


_VOWELS = re.compile(r"[aeiouyаеёиоуыэюяあいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン]", re.I)


class MockTTS(LocalBackend):
    """Harmonic 'speech-like' signal: one envelope bump per syllable, F0 arc from knobs."""

    def op_synthesize(self, item: dict[str, Any], **_: Any) -> dict[str, Any]:
        sr = 24000
        text = item.get("text", "")
        n_syl = max(1, len(_VOWELS.findall(text)))
        knobs = item.get("knobs") or {}
        target = item.get("target_duration")
        rng = np.random.default_rng(_seed_of(text, item.get("seed", 0), self.spec.name))
        tempo = {"fast": 0.85, "slow": 1.2}.get(knobs.get("tempo", ""), 1.0)
        dur = float(target) * float(rng.uniform(0.93, 1.07)) if target else n_syl * 0.17 * tempo
        dur = float(np.clip(dur, 0.3, 30.0))
        n = int(dur * sr)
        t = np.arange(n) / sr
        f0_base = float(knobs.get("f0_hz") or 180.0) * float(rng.uniform(0.97, 1.03))
        intensity = float(knobs.get("intensity", 0.5))
        arc_kind = knobs.get("arc", "flat")
        arc_curve = {"rising_end": np.clip((t / dur - 0.66) * 3, 0, 1) * 3, "falling_end": -np.clip((t / dur - 0.66) * 3, 0, 1) * 3}.get(arc_kind, 0.0)
        vibrato = 0.6 * intensity * np.sin(2 * np.pi * 5.5 * t)
        f0 = f0_base * 2 ** ((arc_curve + vibrato) / 12)
        phase = 2 * np.pi * np.cumsum(f0) / sr
        y = sum((0.6 / k) * np.sin(k * phase) for k in range(1, 9))
        # syllable envelope with short gaps
        lead, trail = 0.05, 0.08
        env = np.zeros(n)
        body = max(1e-3, dur - lead - trail)
        for s in range(n_syl):
            a = lead + body * s / n_syl
            b = lead + body * (s + 0.82) / n_syl
            m = (t >= a) & (t < b)
            if m.any():
                x = (t[m] - a) / (b - a)
                env[m] = np.sin(np.pi * x) ** 0.6
        noise = rng.normal(0, 0.004, n)
        y = (0.25 + 0.35 * intensity) * env * y / 3.0 + noise
        out = Path(item["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        features.save(out, y.astype(np.float32), sr)
        return {"path": str(out), "sr": sr, "duration": round(dur, 3), "mock": True}

    def op_design(self, item: dict[str, Any], **kw: Any) -> dict[str, Any]:
        return self.op_synthesize(item, **kw)


class MockVC(LocalBackend):
    def op_convert(self, item: dict[str, Any], **_: Any) -> dict[str, Any]:
        import librosa

        y, sr = features.load(item["src"])
        steps = float(item.get("f0_transpose") or 0)
        if abs(steps) > 0.01:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
        out = Path(item["out"])
        features.save(out, y, sr)
        return {"path": str(out), "sr": sr, "mock": True}

    def op_train(self, job: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        job = job or {}
        out = Path(job["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        model = out / "model.pth"
        model.write_bytes(b"mock-rvc")
        (out / "model.index").write_bytes(b"mock-index")
        return {"model": str(model), "index": str(out / "model.index"), "epochs": 0, "mock": True}


class MockTranslator(LocalBackend):
    """Speaks the translate-stage JSON protocol: reads the batch from the last user
    message and answers ``{"lines": [{"id", "<tgt>", "<tgt>_short", "notes"}]}``.

    ``options.dictionary``: path to a YAML/JSON mapping source text -> target text;
    unknown lines pass through unchanged (flagged in notes).
    """

    def __init__(self, spec: BackendSpec, options: dict[str, Any] | None = None, context: dict[str, Any] | None = None) -> None:
        super().__init__(spec, options, context)
        self.dictionary: dict[str, str] = {}
        path = self.options.get("dictionary")
        if path and Path(path).exists():
            import yaml

            self.dictionary = {str(k): str(v) for k, v in (yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}).items()}

    def op_chat(self, messages: list[dict[str, str]], tgt: str = "ru", **_: Any) -> str:
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "{}")
        start = user.find("[")
        batch = json.loads(user[start : user.rfind("]") + 1]) if start >= 0 else []
        out = []
        for row in batch:
            src = row.get("text", "")
            tr = self.dictionary.get(src)
            note = row.get("emotion") or ""
            if tr is None:
                tr = src
                note = "passthrough"
            words = tr.split()
            short = " ".join(words[: max(1, int(len(words) * 0.75))]) if len(words) > 8 else tr
            out.append({"id": row.get("id"), tgt: tr, f"{tgt}_short": short, "notes": note})
        return json.dumps({"lines": out}, ensure_ascii=False)

"""Manifest schemas and atomic JSONL/YAML IO.

Manifests are the source of truth. Rules:

* Each stage owns its output files; it never rewrites fields of a file it does
  not own. Later stages *add* files (``link.jsonl``, ``analysis/*.json`` ...).
* Writes are atomic (temp file + ``os.replace``) so a crash never leaves a
  half-written manifest.
* All schemas allow extra fields (``extra="allow"``) so newer stages can add
  data without breaking older readers.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Type, TypeVar

import orjson
import yaml
from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T", bound=BaseModel)


class Record(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


# ----------------------------------------------------------------------------
# schemas
# ----------------------------------------------------------------------------


class AudioIndexRecord(Record):
    id: str
    relpath: str  # relative to the voice root inside work/raw (or the game dir)
    src_path: str  # absolute path of the extracted original file
    duration: float = 0.0
    sr: int = 0
    ch: int = 0
    codec: str = ""
    bytes: int = 0
    wav: str = ""  # work-relative audio/src/<id>.wav
    wav16: str = ""  # work-relative audio/src16/<id>.wav
    wav_stereo: str = ""
    error: str = ""


class LineRecord(Record):
    line_id: str
    scene: str
    order: int
    speaker_raw: str = ""
    speaker_id: str = "narrator"
    text_src: str = ""
    voice_file: str | None = None
    has_voice: bool = False
    audio_id: str | None = None
    src_wav: str | None = None
    src_wav16: str | None = None
    context_prev: list[str] = Field(default_factory=list)
    choices: list[str] | None = None
    flags: list[str] = Field(default_factory=list)
    raw: str | None = None  # raw script source of the line (debugging)
    source_file: str | None = None
    source_line: int | None = None


class CharacterStats(Record):
    n_lines: int = 0
    total_sec: float = 0.0
    f0_median_hz: float | None = None
    f0_iqr_st: float | None = None
    energy_db_median: float | None = None
    lufs_median: float | None = None
    rate_median: float | None = None
    emotion_hist: dict[str, float] = Field(default_factory=dict)


class Character(Record):
    names: dict[str, str] = Field(default_factory=dict)  # {src, tgt}
    gender: Literal["m", "f", "unknown"] = "unknown"
    age_hint: str = ""
    voice_prefix: str = ""
    stats: CharacterStats = Field(default_factory=CharacterStats)
    refs: list[str] = Field(default_factory=list)
    ref_candidates: list[str] = Field(default_factory=list)
    performer: dict[str, Any] = Field(default_factory=dict)
    rvc: dict[str, Any] = Field(default_factory=dict)
    lora: dict[str, Any] = Field(default_factory=dict)
    # A clone the original, B performer + VC, C LoRA, P performer only (no original audio)
    route: Literal["A", "B", "C", "P"] | None = None
    notes: str = ""
    description: str = ""  # free-text voice description for voice design ("строгая учительница, низкий голос")


class LinkRecord(Record):
    line_id: str
    audio_id: str | None = None
    asr_text: str = ""
    match_score: float = 0.0
    best_alt_line_id: str | None = None
    best_alt_score: float = 0.0
    spk_cluster: int | None = None
    spk_consistency: float | None = None
    status: Literal["ok", "review", "unmatched"] = "review"
    reasons: list[str] = Field(default_factory=list)
    relinked_from: str | None = None


class Passport(Record):
    """``analysis/<line_id>.json`` -- the expression passport.

    emotion: {"e2v": {"probs", "label", "embedding_path"}, "anime": {"label", "probs"},
              "adv": {"arousal", "dominance", "valence"}}
    prosody: {"f0_median_hz", "f0_std_st", "f0_range_st", "f0_iqr_st", "voiced_ratio",
              "energy_db_mean", "energy_contour_path", "f0_contour_path", "rate_mora_per_s",
              "pauses": [{"t", "d"}], "leading_silence_s", "trailing_silence_s", "speech_s",
              "arc", "spectral_tilt_db_khz", "lufs"}
    align:   {"chars": [{"c", "t0", "t1"}], "backend"}
    style:   StyleSpec.to_dict() (filled after per-character aggregation)
    backends: part -> backend actually used; missing: part -> reason it is absent
    """

    line_id: str
    duration_s: float = 0.0
    snr_db: float | None = None
    bgm: bool = False
    emotion: dict[str, Any] = Field(default_factory=dict)
    events: list[str] = Field(default_factory=list)
    prosody: dict[str, Any] = Field(default_factory=dict)
    align: dict[str, Any] = Field(default_factory=dict)
    caption: str | None = None
    style: dict[str, Any] = Field(default_factory=dict)
    backends: dict[str, str] = Field(default_factory=dict)
    missing: dict[str, str] = Field(default_factory=dict)


class RefClip(Record):
    """Entry of ``voicebank/<speaker_id>/refs.json``."""

    path: str  # work-relative wav
    line_id: str
    emotion: str = "neutral"
    arousal: float | None = None
    valence: float | None = None
    snr: float | None = None
    duration: float = 0.0
    usage: Literal["timbre", "emotion", "both"] = "both"
    text: str = ""


class TranslationRecord(Record):
    line_id: str
    text_tgt: str
    text_tgt_short: str = ""
    text_tts: str = ""
    text_tts_short: str = ""
    text_plain: str = ""  # stress-free, normalized: the CER reference
    syllables: int = 0
    notes: str = ""
    source: str = ""  # import | llm:<model> | mock
    glossary_hits: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class SynthSidecar(Record):
    line_id: str
    backend: str
    route: str
    k: int
    seed: int | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    style_prompt: str = ""
    style: dict[str, Any] = Field(default_factory=dict)
    ref_paths: list[str] = Field(default_factory=list)
    text_used: str = ""
    text_variant: str = "full"  # full | short | interjection | orig
    elapsed_s: float = 0.0
    peak_vram_gb: float | None = None
    pre_vc_wav: str | None = None
    round: int = 0
    wav: str = ""


class CandidateScore(Record):
    line_id: str
    wav: str
    backend: str = ""
    route: str = ""
    metrics: dict[str, float | None] = Field(default_factory=dict)
    gates: dict[str, bool] = Field(default_factory=dict)
    composite: float | None = None
    passed: bool = False
    notes: list[str] = Field(default_factory=list)


class SelectionRecord(Record):
    line_id: str
    winner: str | None = None
    backend: str | None = None
    route: str | None = None
    scores: dict[str, float | None] = Field(default_factory=dict)
    status: Literal["auto", "regen", "review", "override"] = "review"
    reason: str = ""
    round: int = 0
    n_candidates: int = 0


class OverrideRecord(Record):
    line_id: str
    # set_src_text: user-corrected source text (mode C lines recovered by ASR)
    # tuning: per-line voice tuning override (payload = tuning dict, see style_prompt.apply_tuning)
    action: Literal["accept", "reject", "regen", "set_text", "set_ref", "keep_original", "set_src_text", "tuning"]
    payload: dict[str, Any] = Field(default_factory=dict)
    ts: str = ""


class FinalIndexRecord(Record):
    line_id: str
    voice_file: str
    final_wav: str
    encoded: str
    source: str  # winner path or "orig"
    status: str = "auto"
    stretch: float = 1.0
    lufs: float | None = None
    duration_s: float = 0.0


# ----------------------------------------------------------------------------
# atomic IO
# ----------------------------------------------------------------------------


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _to_json(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json", exclude_none=False)
    return obj


def dumps(obj: Any) -> bytes:
    return orjson.dumps(_to_json(obj), option=orjson.OPT_SERIALIZE_NUMPY, default=_default)


def _default(o: Any) -> Any:
    if isinstance(o, Path):
        return o.as_posix()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, BaseModel):
        return o.model_dump(mode="json")
    raise TypeError(f"not JSON serializable: {type(o)!r}")


def write_json(path: Path, obj: Any, indent: bool = True) -> None:
    opt = orjson.OPT_SERIALIZE_NUMPY | (orjson.OPT_INDENT_2 if indent else 0)
    _atomic_write_bytes(Path(path), orjson.dumps(_to_json(obj), option=opt, default=_default))


def read_json(path: Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return orjson.loads(path.read_bytes())


def write_jsonl(path: Path, records: Iterable[Any]) -> int:
    lines = [dumps(r) for r in records]
    _atomic_write_bytes(Path(path), b"\n".join(lines) + (b"\n" if lines else b""))
    return len(lines)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return
    with open(path, "rb") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield orjson.loads(line)
            except orjson.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: invalid JSON: {e}") from e


def read_jsonl(path: Path, model: Type[T] | None = None) -> list[Any]:
    rows = list(iter_jsonl(path))
    if model is None:
        return rows
    return [model.model_validate(r) for r in rows]


def append_jsonl(path: Path, record: Any) -> None:
    """Append one record (not atomic across records, but each line is whole)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "ab") as f:
        f.write(dumps(record) + b"\n")
        f.flush()


def write_yaml(path: Path, obj: Any) -> None:
    data = _to_json(obj)
    if isinstance(data, dict):
        data = {k: _to_json(v) for k, v in data.items()}
    text = yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=120)
    _atomic_write_bytes(Path(path), text.encode("utf-8"))


def read_yaml(path: Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------
# typed accessors
# ----------------------------------------------------------------------------


def by_key(records: Iterable[Any], key: str = "line_id") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for r in records:
        k = r[key] if isinstance(r, dict) else getattr(r, key)
        out[k] = r
    return out


def merge_by_key(base: list[dict[str, Any]], updates: Iterable[dict[str, Any]], key: str = "line_id") -> list[dict[str, Any]]:
    """Shallow-merge ``updates`` into ``base`` by key; new keys are appended."""
    idx = {r[key]: i for i, r in enumerate(base)}
    out = [dict(r) for r in base]
    for u in updates:
        k = u[key]
        if k in idx:
            out[idx[k]].update(u)
        else:
            idx[k] = len(out)
            out.append(dict(u))
    return out


def load_lines(path: Path) -> list[LineRecord]:
    return read_jsonl(path, LineRecord)


def load_characters(path: Path) -> dict[str, Character]:
    raw = read_yaml(path, {}) or {}
    return {k: Character.model_validate(v or {}) for k, v in raw.items()}


def save_characters(path: Path, chars: dict[str, Character]) -> None:
    write_yaml(path, {k: v.model_dump(mode="json") for k, v in chars.items()})


def load_overrides(path: Path) -> dict[str, list[OverrideRecord]]:
    """Overrides grouped by line_id in file order (last one wins per action)."""
    out: dict[str, list[OverrideRecord]] = {}
    for r in read_jsonl(path, OverrideRecord):
        out.setdefault(r.line_id, []).append(r)
    return out

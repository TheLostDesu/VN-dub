"""Pydantic models for ``pipeline.yaml`` and ``project.yaml``.

``pipeline.yaml`` describes *how* to process (models, thresholds, weights);
``project.yaml`` (written by ``detect`` into the work dir, editable) describes
*what* is processed (game dir, engine, codec).

A pipeline config may ``extends: other.yaml`` (path relative to the file); the
base is loaded first and the child is deep-merged over it.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from vndub.util import hash_obj


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class _Open(BaseModel):
    """Section that accepts arbitrary extra keys (backend parameters)."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)


# ----------------------------------------------------------------------------
# project / gpu / tools
# ----------------------------------------------------------------------------


class ProjectSection(_Section):
    game_dir: str = ""
    work_dir: str = "work"
    out_dir: str = "out"
    models_dir: str = "models"
    # auto = detect the language of the script text (parse writes it into work/project.yaml), or a code: ja, en, zh, ko, ru ...
    src_lang: str = "auto"
    # language of the voice recordings when it differs from the text (a translated release with the original voices); "" = same
    audio_lang: str = ""
    tgt_lang: str = "ru"


class GpuSection(_Section):
    vram_budget_gb: float = 7.5
    safety_margin_gb: float = 0.5
    device: str = "cuda:0"
    # When CUDA is absent, heavy backends whose adapters support CPU run there
    # (slow). The test profile enables this; the default profile refuses.
    allow_cpu_fallback: bool = False


# ----------------------------------------------------------------------------
# extract / parse
# ----------------------------------------------------------------------------


class ExtractSection(_Section):
    keep_stereo_copy: bool = True
    analysis_sr: int = 16000
    workers: int = 4
    voice_exts: list[str] = Field(
        default_factory=lambda: [".ogg", ".wav", ".opus", ".mp3", ".flac", ".m4a", ".weba"]
    )
    # Proprietary formats decoded with vgmstream-cli instead of ffmpeg.
    vgmstream_exts: list[str] = Field(
        default_factory=lambda: [
            ".hca", ".adx", ".at3", ".at9", ".fsb", ".bank", ".nwa", ".nwk", ".ovk",
            ".wem", ".bnk", ".acb", ".awb", ".xwb", ".ahx", ".kvs", ".ktss",
        ]
    )
    # Directory names that usually contain voice files (used by generic scans).
    voice_dirs: list[str] = Field(default_factory=lambda: ["voice", "vo", "koe", "sound/voice", "data/voice"])


class KagOpts(_Section):
    macro_files: list[str] = Field(default_factory=lambda: ["macro.ks", "first.ks"])
    name_attr: str = "auto"
    voice_attr: str = "auto"
    macro_names: list[str] = Field(default_factory=lambda: ["nm", "cn", "name", "ns"])
    scenario_glob: str = "**/*.ks"
    encoding: str = "auto"
    # [r] ends a dialogue line (default: [r] is a line break inside the message)
    break_on_r: bool = False


class NScripterOpts(_Section):
    speaker_regex: str = r"^【(?P<name>[^】]+)】"
    encoding: str = "cp932"


class RenpyOpts(_Section):
    dialogue_tab: str = ""
    # Overrides config.auto_voice detected in the scripts, e.g. "voice/{id}.ogg".
    auto_voice: str = ""
    # Ren'Py >= 8.5.1 config.tlid_only_considers_say (default True in 8.6): voice
    # statements do not enter the dialogue identifier
    tlid_only_considers_say: bool = False


class TyranoOpts(_Section):
    scenario_dir: str = "data/scenario"


class GenericOpts(_Section):
    prefix_regex: str = r"^(?P<prefix>[A-Za-z]+?)_?(?P<index>\d+)$"
    # speaker_id -> voice prefix, when it cannot be inferred
    prefix_map: dict[str, str] = Field(default_factory=dict)


class EngineOpts(_Section):
    kag: KagOpts = Field(default_factory=KagOpts)
    nscripter: NScripterOpts = Field(default_factory=NScripterOpts)
    renpy: RenpyOpts = Field(default_factory=RenpyOpts)
    tyrano: TyranoOpts = Field(default_factory=TyranoOpts)
    generic: GenericOpts = Field(default_factory=GenericOpts)


class ParseSection(_Section):
    engine: str = "auto"
    engine_opts: EngineOpts = Field(default_factory=EngineOpts)
    import_vntextpatch: str = ""
    narrator_names: list[str] = Field(default_factory=lambda: ["", "narrator", "ナレーション", "地の文"])
    # Voiced lines shown without a speaker name (narrator box, or a name like ？？？): auto = give them to the
    # character their voice-file prefix names, else (narrator box) to the cast-and-dubbed "voice_over"
    # speaker; off = keep the script's speaker (a voiced narrator then gets no performer, route A only).
    hidden_speakers: Literal["auto", "off"] = "auto"


# ----------------------------------------------------------------------------
# link / analyze
# ----------------------------------------------------------------------------


class LinkSection(_Section):
    asr: str = "anime_whisper"
    asr_fallback: str = "whisper_large_v3_turbo"
    asr_compute: str = "int8"
    accept_score: float = 0.80
    review_score: float = 0.60
    neighbour_window: int = 5
    global_topk: int = 3
    auto_repair: bool = True
    speaker_embedder: str = "ecapa"
    cluster_distance: float = 0.35
    outlier_consistency: float = 0.50
    # asr: verify every clip by speech recognition (CLI default, unchanged behaviour).
    # no_asr: "intonation-check only" -- cheap signals only (speaker-embedding consistency,
    # duration vs text length, prefix/index order); no ASR backend is loaded (app default).
    mode: Literal["asr", "no_asr"] = "asr"
    # Mode C: voiced lines with no script text get their source text from ASR
    # (flag text_from_asr). Runs even with mode no_asr; nothing else is transcribed.
    recover_text: bool = False
    recover_asr: str = "anime_whisper"
    # no_asr duration check: speech slower than rate_min or faster than rate_max (mora/s) is
    # implausible for the line's text; silence_tolerance_s is added to the slow bound
    # (leading/trailing silence and pauses are part of the file).
    rate_min: float = 3.0
    rate_max: float = 13.0
    silence_tolerance_s: float = 1.0
    # no_asr: summary.suggest_asr_check is true when at least this share of the judged lines
    # (and at least 5 lines) is flagged by a cheap signal
    suggest_asr_share: float = 0.2


class AnalyzeSection(_Section):
    e2v: str = "emotion2vec_plus_large"
    anime_ser: str = "anime_ser"
    adv: str = "audeering_adv"
    events: str = "sensevoice"
    aligner: str = "qwen3_aligner"
    aligner_fallback: str = "whisperx_ja"
    f0: str = "rmvpe"  # rmvpe | torchcrepe | librosa
    vad: str = "silero"  # silero | energy
    caption: bool = False
    caption_backend: str = "qwen2_audio"
    ref_candidates_per_emotion: int = 2
    e2v_weight: float = 0.6
    anime_weight: float = 0.4


# ----------------------------------------------------------------------------
# translate
# ----------------------------------------------------------------------------


class TextEmotionSection(_Section):
    """Synthetic expression passports for lines without original audio (stage text_emotion)."""

    enabled: bool = True
    cues: str = "configs/delivery_cues.yaml"
    narration_window: int = 3        # unvoiced narration lines scanned before/after a line
    llm: Literal["auto", "on", "off"] = "auto"   # auto = on when translate's local LLM is available
    llm_context_lines: int = 5
    min_confidence: float = 0.6      # a later layer overrides an earlier one only at >= this
    only_unvoiced: bool = True       # lines with original audio keep their audio passport


class LlmOpts(_Section):
    backend: str = "gemma4_12b"
    fallbacks: list[str] = Field(default_factory=lambda: ["qwen35_9b", "hy_mt15_7b"])
    ctx: int = 16384
    # -1 = every layer on the GPU and never a silent partial offload (llama-server's own --fit is switched off; a
    # model that does not fit fails over to the next fallback); a number = that many layers, a partial offload on purpose
    n_gpu_layers: int = -1
    temperature: float = 0.3
    port: int = 8089
    max_retries: int = 2
    # more llama-server flags, e.g. ["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0"] (a quantized KV cache: half the size)
    extra_args: list[str] = Field(default_factory=list)
    # llama-server slots: this many scenes are translated at the same time (continuous batching). The KV cache
    # grows with ctx x parallel, so 2 needs a card with VRAM to spare beyond the model (12 GB+ for the 12B model).
    parallel: int = 1
    # used by `vndub translate --real` when backend is the mock (test profile)
    real_backend: str = "gemma4_12b"


class StressOpts(_Section):
    tool: str = "ruaccent"  # ruaccent | none
    model: str = "turbo"
    yo: bool = True
    # plus = "+" before the stressed vowel (F5-TTS_RUSSIAN); plain = no marks
    format: Literal["auto", "plus", "plain"] = "auto"


class TranslateSection(_Section):
    import_path: str = Field("", alias="import")
    llm: LlmOpts = Field(default_factory=LlmOpts)
    batch_lines: int = 20
    context_window_lines: int = 15
    glossary: str = "configs/glossary.yaml"
    short_variant_min_words: int = 8
    stress: StressOpts = Field(default_factory=StressOpts)
    interjections: str = "configs/interjections.yaml"
    # MockTranslator dictionary (test profile): source text -> target text
    mock_dictionary: str = ""
    # Also translate speaker lines that have no original audio (mode B; the unvoiced protagonist); narration never.
    # auto (default) = on only when synth.voice_unvoiced is on (the app; the CLI keeps voiced lines only)
    # | on = always translate speaker lines without audio too | off = voiced lines only.
    include_unvoiced: Literal["auto", "on", "off"] = "auto"


# ----------------------------------------------------------------------------
# voicebank
# ----------------------------------------------------------------------------


class RefsOpts(_Section):
    min_snr_db: float = 25.0
    min_sec: float = 3.0
    max_sec: float = 10.0
    per_emotion: int = 2
    max_total: int = 10
    min_refs: int = 5


class PerformerOpts(_Section):
    kind: Literal["voice_design", "reference"] = "voice_design"
    backend: str = "qwen3_tts_voicedesign"
    n_designs: int = 4
    test_sentences: int = 5
    max_f0_delta_semitones: float = 3.0
    # kind=reference: <reference_dir>/<speaker_id>/*.wav native target-language clips
    reference_dir: str = ""
    # Casting screen: how many of a design's clips are the character's own translated lines
    # (the clips the user auditions); the rest of ``test_sentences`` are generic test sentences.
    sample_lines: int = 2
    # Also render a "clone" design (route A, zero-shot from the original clips) in the voicebank
    # stage. Off for the CLI; the app's propose_designs always offers it for characters with audio.
    offer_clone: bool = False


class RvcOpts(_Section):
    enabled: bool = True
    backend: str = "rvc"
    min_minutes: float = 10.0
    max_minutes: float = 40.0
    f0_method: str = "rmvpe"
    sample_rate: int = 40000
    batch_size: int = 6
    epochs: int = 300
    overfit_patience: int = 30


class RouteCOpts(_Section):
    enabled: bool = False
    backend: str = "voxcpm2"
    min_minutes: float = 5.0
    max_minutes: float = 30.0
    composite_threshold: float = 0.75


class VoicebankSection(_Section):
    refs: RefsOpts = Field(default_factory=RefsOpts)
    performer: PerformerOpts = Field(default_factory=PerformerOpts)
    rvc: RvcOpts = Field(default_factory=RvcOpts)
    route_c: RouteCOpts = Field(default_factory=RouteCOpts)
    calibrate_spk: bool = True
    min_minutes_for_voicebank: float = 3.0


# ----------------------------------------------------------------------------
# synth / score / master / pack / review / report
# ----------------------------------------------------------------------------


class BackendOpts(_Open):
    enabled: bool = False


class VcOpts(_Section):
    primary: str = "rvc"
    fallback: str = "seedvc"
    rvc: dict[str, Any] = Field(default_factory=lambda: {"index_rate": 0.5, "protect": 0.33, "f0_method": "rmvpe"})
    seedvc: dict[str, Any] = Field(default_factory=lambda: {"convert_style": False, "diffusion_steps": 30})


class SynthSection(_Section):
    n_candidates: int = 4
    n_candidates_short: int = 6
    routes: dict[str, bool] = Field(default_factory=lambda: {"A": True, "B": True, "C": False})
    backends: dict[str, BackendOpts] = Field(default_factory=dict)
    vc: VcOpts = Field(default_factory=VcOpts)
    keep_original_for_interjections: bool = True
    use_short_text_if_ratio_above: float = 1.25
    interjection_max_syllables: int = 2
    high_arousal: float = 0.75
    seed_base: int = 1234
    # Dub speaker lines that have no original audio too (route P: performer voice only, no VC;
    # narration is never dubbed). CLI default off; the desktop app turns it on.
    voice_unvoiced: bool = False


class Gates(_Section):
    utmos_min: float = 3.0
    cer_max: float = 0.10
    dur_ratio: tuple[float, float] = (0.8, 1.25)
    spk_sim_min: float | Literal["calibrated"] = "calibrated"
    spk_sim_default: float = 0.25  # used when calibration is unavailable


class ScoreSection(_Section):
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "utmos": 0.25, "pq": 0.10, "spk_sim": 0.20, "emo_sim": 0.20,
            "cer": 0.15, "dur": 0.05, "prosody": 0.05,
        }
    )
    gates: Gates = Field(default_factory=Gates)
    mos: str = "utmosv2"
    mos_fallback: str = "utmos22"
    aesthetics: str = "audiobox"
    speaker: str = "ecapa"
    speaker_extra: str = ""
    cer_asr: str = "gigaam_v3"
    cer_asr_fallback: str = "whisper_ru"
    emotion: str = "emotion2vec_plus_large"
    adv: str = "audeering_adv"
    regen_rounds: int = 2
    # interjections get a softer CER gate
    interjection_cer_max: float = 0.5


class StretchOpts(_Section):
    enabled: bool = True
    target_window: tuple[float, float] = (0.9, 1.1)
    max_stretch: float = 0.15
    engine: Literal["rubberband", "atempo"] = "rubberband"


class LoudnessOpts(_Section):
    target: str | float = "character_median"
    true_peak_dbtp: float = -1.0
    default_lufs: float = -20.0


class MasterSection(_Section):
    stretch: StretchOpts = Field(default_factory=StretchOpts)
    loudness: LoudnessOpts = Field(default_factory=LoudnessOpts)
    match_silence_profile: bool = True
    # what to do with lines whose selection status is "review"
    include_review_lines: Literal["skip", "provisional"] = "skip"


class PackSection(_Section):
    kirikiri: dict[str, Any] = Field(default_factory=lambda: {"mode": "patch_xp3", "patch_name": "", "subdir": ""})
    renpy: dict[str, Any] = Field(default_factory=lambda: {"mode": "loose"})
    nscripter: dict[str, Any] = Field(default_factory=lambda: {"mode": "loose"})
    unity: dict[str, Any] = Field(default_factory=lambda: {"mode": "unitypy_script"})
    other: dict[str, Any] = Field(default_factory=dict)


class ReviewSection(_Section):
    host: str = "127.0.0.1"
    port: int = 7861
    batch_accept_threshold: float = 0.75


class ReportSection(_Section):
    compare_routes: bool = True
    markdown: bool = True


class PipelineConfig(_Section):
    profile: str = "default"
    extends: str | None = None
    # Only the test profile may set this: a backend whose env/weights are missing
    # is replaced by its mock and the substitution is logged and reported.
    fallback_to_mock: bool = False
    project: ProjectSection = Field(default_factory=ProjectSection)
    license_policy: Literal["research", "commercial"] = "research"
    gpu: GpuSection = Field(default_factory=GpuSection)
    tools_file: str = "configs/tools.yaml"
    tools: dict[str, str] = Field(default_factory=dict)
    extract: ExtractSection = Field(default_factory=ExtractSection)
    parse: ParseSection = Field(default_factory=ParseSection)
    link: LinkSection = Field(default_factory=LinkSection)
    analyze: AnalyzeSection = Field(default_factory=AnalyzeSection)
    translate: TranslateSection = Field(default_factory=TranslateSection)
    text_emotion: TextEmotionSection = Field(default_factory=TextEmotionSection)
    voicebank: VoicebankSection = Field(default_factory=VoicebankSection)
    synth: SynthSection = Field(default_factory=SynthSection)
    score: ScoreSection = Field(default_factory=ScoreSection)
    master: MasterSection = Field(default_factory=MasterSection)
    pack: PackSection = Field(default_factory=PackSection)
    review: ReviewSection = Field(default_factory=ReviewSection)
    report: ReportSection = Field(default_factory=ReportSection)

    # path of the file this config was loaded from (not part of the schema)
    source_path: str | None = Field(default=None, exclude=True)

    @field_validator("synth")
    @classmethod
    def _routes_known(cls, v: SynthSection) -> SynthSection:
        bad = set(v.routes) - {"A", "B", "C", "P"}
        if bad:
            raise ValueError(f"unknown synth routes: {sorted(bad)} (allowed: A, B, C, P)")
        return v

    def section_hash(self, *names: str) -> str:
        """Stable hash of the given config sections (for stage cache keys)."""
        dump = self.model_dump(mode="json", by_alias=True)
        return hash_obj({n: dump.get(n) for n in names})

    def enabled_backends(self) -> list[str]:
        return [name for name, opts in self.synth.backends.items() if opts.enabled]

    def base_dir(self) -> Path:
        """Directory that relative config paths (glossary, tools_file) resolve against.

        Config files live in ``<repo>/configs``; paths like ``configs/glossary.yaml``
        are relative to the repo root, i.e. the parent of the configs dir.
        """
        if self.source_path:
            p = Path(self.source_path).resolve().parent
            return p.parent if p.name == "configs" else p
        return Path.cwd()

    def resolve(self, rel: str | Path) -> Path:
        p = Path(rel)
        if p.is_absolute():
            return p
        cand = self.base_dir() / p
        if cand.exists():
            return cand
        return Path.cwd() / p


# ----------------------------------------------------------------------------
# project.yaml
# ----------------------------------------------------------------------------


class VoiceCodec(_Open):
    container: str = "ogg"
    codec: str = "vorbis"
    sr: int = 44100
    ch: int = 1
    bitrate_k: int = 96


class ProjectYaml(_Open):
    game_dir: str
    engine: str = "generic"
    engine_version: str = ""
    src_lang: str = "auto"
    # written by parse when src_lang was detected: {lang, confidence, letters, lines, scripts}; absent for a user-set code
    src_lang_detected: dict | None = None
    audio_lang: str = ""
    tgt_lang: str = "ru"
    voice_codec: VoiceCodec = Field(default_factory=VoiceCodec)
    encrypted: bool = False
    drm: list[str] = Field(default_factory=list)
    archives: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    notes: str = ""


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_raw(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    seen = seen or set()
    path = path.resolve()
    if path in seen:
        raise ValueError(f"config extends cycle at {path}")
    seen.add(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    parent = data.get("extends")
    if parent:
        base = _load_raw((path.parent / parent), seen)
        data = deep_merge(base, data)
    return data


def load_config(path: str | Path | None, overrides: dict[str, Any] | None = None) -> PipelineConfig:
    """Load a pipeline config (with ``extends``) and apply dotted-key overrides."""
    if path is None:
        data: dict[str, Any] = {}
        src = None
    else:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"config not found: {p}")
        data = _load_raw(p)
        src = str(p.resolve())
    for key, val in (overrides or {}).items():
        cur = data
        parts = key.split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = val
    cfg = PipelineConfig.model_validate(data)
    cfg.source_path = src
    if cfg.fallback_to_mock and cfg.profile != "test":
        raise ValueError("fallback_to_mock is only allowed with profile: test")
    return cfg


def load_project_yaml(path: Path) -> ProjectYaml:
    return ProjectYaml.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def save_project_yaml(path: Path, proj: ProjectYaml) -> None:
    from vndub.manifest.io import write_yaml

    write_yaml(path, proj.model_dump(mode="json"))

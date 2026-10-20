"""Layout of the ``work/`` and ``out/`` directories.

Every file a stage reads or writes is addressed through :class:`WorkPaths` so the
layout is defined in exactly one place. Paths stored inside manifests are
relative to the work dir (posix separators), so a work dir can be moved.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkPaths:
    work: Path
    out: Path

    @classmethod
    def create(cls, work: Path, out: Path) -> "WorkPaths":
        wp = cls(work.resolve(), out.resolve())
        wp.work.mkdir(parents=True, exist_ok=True)
        return wp

    # --- helpers -----------------------------------------------------------
    def rel(self, p: Path | str) -> str:
        """Path relative to the work dir, posix style (as stored in manifests)."""
        p = Path(p)
        try:
            return p.resolve().relative_to(self.work).as_posix()
        except ValueError:
            return p.as_posix()

    def abs(self, rel: str | Path) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.work / p

    # --- top level ---------------------------------------------------------
    @property
    def project_yaml(self) -> Path:
        return self.work / "project.yaml"

    @property
    def stage_status(self) -> Path:
        return self.work / "stage_status.json"

    @property
    def logs(self) -> Path:
        return self.work / "logs"

    def log(self, stage: str) -> Path:
        return self.logs / f"{stage}.log"

    @property
    def cache(self) -> Path:
        return self.work / "cache"

    @property
    def index_db(self) -> Path:
        return self.work / "index.sqlite"

    # --- extract -----------------------------------------------------------
    @property
    def raw(self) -> Path:
        return self.work / "raw"

    @property
    def raw_scripts(self) -> Path:
        return self.raw / "scripts"

    @property
    def raw_voice(self) -> Path:
        return self.raw / "voice"

    @property
    def audio_src(self) -> Path:
        return self.work / "audio" / "src"

    @property
    def audio_src_stereo(self) -> Path:
        return self.work / "audio" / "src_stereo"

    @property
    def audio_src16(self) -> Path:
        return self.work / "audio" / "src16"

    @property
    def audio_index(self) -> Path:
        return self.work / "audio_index.jsonl"

    # --- parse / link ------------------------------------------------------
    @property
    def manifest_dir(self) -> Path:
        return self.work / "manifest"

    @property
    def lines(self) -> Path:
        return self.manifest_dir / "lines.jsonl"

    @property
    def characters(self) -> Path:
        return self.manifest_dir / "characters.yaml"

    @property
    def link(self) -> Path:
        return self.work / "link.jsonl"

    @property
    def review_queue(self) -> Path:
        return self.work / "review_queue.jsonl"

    @property
    def spk_dir(self) -> Path:
        return self.work / "analysis" / "spk"

    def spk_emb(self, audio_id: str) -> Path:
        return self.spk_dir / f"{audio_id}.npy"

    # --- analyze -----------------------------------------------------------
    @property
    def analysis(self) -> Path:
        return self.work / "analysis"

    def passport(self, line_id: str) -> Path:
        return self.analysis / f"{line_id}.json"

    def e2v_emb(self, line_id: str) -> Path:
        return self.analysis / "emb" / f"{line_id}.npy"

    def feat(self, line_id: str, kind: str) -> Path:
        return self.analysis / "feat" / f"{line_id}.{kind}.npy"

    # --- translate ---------------------------------------------------------
    @property
    def translation_dir(self) -> Path:
        return self.work / "translation"

    def translation(self, lang: str) -> Path:
        return self.translation_dir / f"lines.{lang}.jsonl"

    # --- voicebank ---------------------------------------------------------
    @property
    def voicebank(self) -> Path:
        return self.work / "voicebank"

    def vb(self, speaker_id: str) -> Path:
        return self.voicebank / speaker_id

    def vb_refs(self, speaker_id: str) -> Path:
        return self.vb(speaker_id) / "refs"

    def vb_refs_json(self, speaker_id: str) -> Path:
        return self.vb(speaker_id) / "refs.json"

    def vb_performer(self, speaker_id: str) -> Path:
        return self.vb(speaker_id) / "performer.yaml"

    def vb_rvc(self, speaker_id: str) -> Path:
        return self.vb(speaker_id) / "rvc"

    def vb_lora(self, speaker_id: str) -> Path:
        return self.vb(speaker_id) / "lora"

    def vb_synthetic(self, speaker_id: str) -> Path:
        return self.vb(speaker_id) / "synthetic"

    @property
    def spk_calibration(self) -> Path:
        return self.voicebank / "spk_calibration.json"

    # --- synth / score -----------------------------------------------------
    @property
    def synth(self) -> Path:
        return self.work / "synth"

    def synth_dir(self, line_id: str) -> Path:
        return self.synth / line_id

    def candidate(self, line_id: str, backend: str, route: str, k: int) -> Path:
        return self.synth_dir(line_id) / f"{backend}_{route}_{k}.wav"

    @property
    def scores_dir(self) -> Path:
        return self.work / "scores"

    def scores(self, line_id: str) -> Path:
        return self.scores_dir / f"{line_id}.jsonl"

    @property
    def selection(self) -> Path:
        return self.work / "selection.jsonl"

    @property
    def overrides(self) -> Path:
        return self.work / "overrides.jsonl"

    # --- master / pack / report -------------------------------------------
    @property
    def final(self) -> Path:
        return self.work / "final"

    def final_wav(self, line_id: str) -> Path:
        return self.final / f"{line_id}.wav"

    @property
    def final_encoded(self) -> Path:
        return self.final / "encoded"

    @property
    def final_index(self) -> Path:
        return self.work / "final_index.jsonl"

    @property
    def report_dir(self) -> Path:
        return self.work / "report"

    def out_engine(self, engine: str) -> Path:
        return self.out / engine

"""Config loading (extends, overrides, validation) and the CLI surface."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from vndub.cli import app
from vndub.config import PipelineConfig, deep_merge, load_config
from vndub.stages.common import STAGES

REPO = Path(__file__).resolve().parent.parent


def test_default_and_test_profiles_load() -> None:
    cfg = load_config(REPO / "configs" / "pipeline.yaml")
    assert cfg.profile == "default" and not cfg.fallback_to_mock
    assert cfg.gpu.vram_budget_gb == 7.5
    assert set(cfg.enabled_backends()) == {"qwen3_tts", "cosyvoice3", "chatterbox_ml"}
    assert cfg.score.weights["utmos"] == 0.25
    t = load_config(REPO / "configs" / "pipeline.test.yaml")
    assert t.profile == "test" and t.fallback_to_mock
    assert t.link.asr == "whisper_tiny"
    # extends keeps base values the child does not override
    assert t.score.weights == cfg.score.weights
    assert t.synth.backends["qwen3_tts"].enabled is False
    assert t.synth.backends["mock_tts"].enabled is True


def test_every_configured_backend_exists() -> None:
    from vndub.backends import registry

    for name in ("pipeline.yaml", "pipeline.test.yaml"):
        cfg = load_config(REPO / "configs" / name)
        names = [cfg.link.asr, cfg.link.speaker_embedder, cfg.analyze.e2v, cfg.analyze.adv, cfg.analyze.events, cfg.analyze.aligner, cfg.analyze.f0, cfg.analyze.vad,
                 cfg.translate.llm.backend, cfg.voicebank.performer.backend, cfg.voicebank.rvc.backend, cfg.synth.vc.primary, cfg.score.mos, cfg.score.speaker, cfg.score.cer_asr, cfg.score.emotion, cfg.score.adv]
        names += [n for n in (cfg.link.asr_fallback, cfg.analyze.anime_ser, cfg.analyze.aligner_fallback, cfg.synth.vc.fallback, cfg.score.mos_fallback, cfg.score.aesthetics, cfg.score.cer_asr_fallback) if n]
        names += list(cfg.synth.backends)
        for n in names:
            registry.get(n)  # raises KeyError on typos


def test_overrides_and_validation(tmp_path: Path) -> None:
    cfg = load_config(REPO / "configs" / "pipeline.yaml", {"synth.n_candidates": 6, "score.gates.cer_max": 0.2})
    assert cfg.synth.n_candidates == 6 and cfg.score.gates.cer_max == 0.2
    bad = tmp_path / "bad.yaml"
    bad.write_text("extends: " + str(REPO / "configs" / "pipeline.yaml").replace("\\", "/") + "\nlink:\n  asr_typo: x\n", encoding="utf-8")
    with pytest.raises(Exception, match="asr_typo"):
        load_config(bad)
    mock_default = tmp_path / "m.yaml"
    mock_default.write_text("fallback_to_mock: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="only allowed with profile: test"):
        load_config(mock_default)
    routes = tmp_path / "r.yaml"
    routes.write_text("synth:\n  routes: {D: true}\n", encoding="utf-8")
    with pytest.raises(Exception, match="unknown synth routes"):
        load_config(routes)


def test_extends_cycle(tmp_path: Path) -> None:
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text("extends: b.yaml\n", encoding="utf-8")
    b.write_text("extends: a.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cycle"):
        load_config(a)


def test_deep_merge() -> None:
    assert deep_merge({"a": {"b": 1, "c": 2}, "x": [1]}, {"a": {"c": 3}, "x": [2]}) == {"a": {"b": 1, "c": 3}, "x": [2]}


def test_section_hash_changes_only_with_section() -> None:
    a = PipelineConfig()
    b = load_config(None, {"synth.n_candidates": 9})
    assert a.section_hash("link") == b.section_hash("link")
    assert a.section_hash("synth") != b.section_hash("synth")


def test_cli_help_lists_all_stages() -> None:
    r = CliRunner().invoke(app, ["--help"])
    assert r.exit_code == 0, r.output
    for s in STAGES + ["run", "review", "models", "tools"]:
        assert s in r.output, s


def test_cli_models_list_and_tools_check() -> None:
    r = CliRunner().invoke(app, ["models", "list", "--role", "tts"])
    assert r.exit_code == 0, r.output
    assert "qwen3_tts" in r.output and "chatterbox_ml" in r.output
    r = CliRunner().invoke(app, ["tools", "check", "--config", str(REPO / "configs" / "pipeline.yaml")])
    assert "ffmpeg" in r.output

"""The GGUF TTS specs (fish_s2, moss_tts, zonos2) agree with the CLI adapter that drives them (offline).

The hub side (do the repos and files exist?) is ``tests/test_registry_repos_network.py``.
"""

from __future__ import annotations

import fnmatch
import string
from pathlib import Path

import pytest

from vndub import tools
from vndub.backends import registry

GGUF_TTS = ("fish_s2", "moss_tts", "zonos2")


def _gguf_names(name: str) -> list[str]:
    """File names (no directory) the spec's ``allow`` patterns stand for; the patterns of these specs are exact paths."""
    (w,) = registry.get(name).weights
    assert w.kind == "hf_file" and w.allow and not any(c in p for p in w.allow for c in "*?[")
    return [p.rsplit("/", 1)[-1] for p in w.allow]


@pytest.mark.parametrize("name", GGUF_TTS)
def test_model_glob_selects_the_downloaded_main_file(name: str) -> None:
    spec = registry.get(name)
    names = _gguf_names(name)
    glob = spec.options["model_glob"]
    hits = [n for n in names if n.endswith(".gguf") and fnmatch.fnmatchcase(n, glob)]
    assert len(hits) == 1, (glob, names)  # exactly one file is the model; dac.gguf / spk-encoder.gguf are companions
    assert spec.weights[0].repo and spec.weights[0].size_gb > 0


@pytest.mark.parametrize("name", GGUF_TTS)
def test_tool_is_a_known_tool(name: str) -> None:
    assert registry.get(name).options["tool"] in tools.TOOL_INFO


@pytest.mark.parametrize("name", ("fish_s2", "moss_tts"))
def test_args_template_builds_an_audiocpp_command(name: str) -> None:
    from adapters.gguf import cli_impl

    spec = registry.get(name)
    cls = {"fish_s2": cli_impl.FishS2CLI, "moss_tts": cli_impl.MossTTSCLI}[name]
    used = {f for a in spec.options["args"] for _lit, f, _spec, _conv in string.Formatter().parse(a) if f}
    inst = object.__new__(cls)  # no env, no files: only the command construction
    inst.options = {**spec.options, "binary": "audiocpp_cli"}
    inst.binary, inst.model_dir, inst.model_file = Path("audiocpp_cli"), Path("models/x"), Path("models/x/m.gguf")
    item = {"text": "hello", "timbre_ref": "ref.wav", "timbre_ref_text": "ref words", "out": "o.wav", "seed": 7, "target_duration": 3.2}
    values = inst._values(item, Path("t.txt"))
    assert used <= set(values), used - set(values)  # every placeholder is one the adapter fills
    cmd = inst.command(values)
    assert cmd[1:5] == ["--task", "clon", "--family", {"fish_s2": "fish_audio", "moss_tts": "moss_tts_v15"}[name]]
    assert cmd[cmd.index("--out") + 1] == "o.wav" and cmd[cmd.index("--voice-ref") + 1] == "ref.wav"
    assert cmd[cmd.index("--backend") + 1] == "best"  # audio.cpp defaults to CPU
    assert not any("{" in a for a in cmd)
    if name == "moss_tts":
        assert "duration_frames=40" in cmd  # 3.2 s x 12.5 frames per second

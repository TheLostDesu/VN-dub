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


# -- Zonos2: the real zonos2-cli command line (zonos2.cpp src/main.cpp) ---------------------------------------------

# every option zonos2-cli's argument loop accepts (anything else prints the usage and exits 1)
ZONOS2_CLI_FLAGS = {
    "--gpu", "--cpu", "--tts", "--clone", "--speaker", "--spk-encoder", "--spk", "--dac", "--seed", "--greedy", "--max",
    "--emotion", "--emotion-valence", "--emotion-arousal", "--emotion-strength", "--emotion-cfg-scale", "--emotion-dir",
    "--speaking-rate", "--inaccurate", "--noisy-bg", "--quality", "--save-speaker",
}
# the first version's guesses, none of which exist in the CLI
ZONOS2_GUESSED_FLAGS = {"--model", "--text", "--valence", "--arousal", "--out"}


def _zonos2(tmp_path: Path, **options):
    from adapters.gguf import cli_impl

    spec = registry.get("zonos2")
    inst = object.__new__(cli_impl.Zonos2CLI)  # no env, no files: only the command construction
    inst.options = {**spec.options, "binary": str(tmp_path / "zonos2-cli.exe"), **options}
    inst.binary, inst.model_dir, inst.model_file = tmp_path / "zonos2-cli.exe", tmp_path / "ZONOS2-GGUF", tmp_path / "ZONOS2-GGUF" / "zonos2-q4_k.gguf"
    return inst


def _zonos2_item(**over):
    item = {"text": "Привет!", "timbre_ref": "ref.wav", "out": "o.wav", "seed": 7, "knobs": {}}
    item.update(over)
    return item


def _zonos2_cmd(inst, item) -> list[str]:
    return inst.command(inst._values(item, Path("t.txt")))


def test_zonos2_spec_has_args_and_they_equal_the_adapter_default() -> None:
    from adapters.gguf import cli_impl

    args = registry.get("zonos2").options["args"]
    assert args == cli_impl.Zonos2CLI.default_args
    used = {f for a in args for _lit, f, _spec, _conv in string.Formatter().parse(a) if f}
    assert {"model", "text", "out", "ref", "seed"} <= used


def test_zonos2_command_is_the_real_cli(tmp_path: Path) -> None:
    inst = _zonos2(tmp_path)
    cmd = _zonos2_cmd(inst, _zonos2_item())
    # zonos2-cli <model.gguf> --tts "<text>" <out.wav> --clone ref.wav --seed N --gpu
    assert cmd[0].endswith("zonos2-cli.exe") and cmd[1].endswith("zonos2-q4_k.gguf")
    assert cmd[2:5] == ["--tts", "Привет!", "o.wav"]
    assert cmd[cmd.index("--clone") + 1] == "ref.wav" and cmd[cmd.index("--seed") + 1] == "7" and cmd[-1] == "--gpu"
    flags = {a for a in cmd if a.startswith("--")}
    assert flags <= ZONOS2_CLI_FLAGS and not flags & ZONOS2_GUESSED_FLAGS
    assert not any("{" in a for a in cmd)
    # neutral style: no emotion flag at all (and no flag is left without its value)
    assert not {"--emotion", "--emotion-valence", "--emotion-arousal", "--emotion-strength"} & set(cmd)


def test_zonos2_cpu_option_and_binary_dir_emotion_directions(tmp_path: Path) -> None:
    (tmp_path / "emotion_directions").mkdir()
    cmd = _zonos2_cmd(_zonos2(tmp_path, gpu=False), _zonos2_item())
    assert cmd[-1] == "--cpu" and "--gpu" not in cmd
    assert cmd[cmd.index("--emotion-dir") + 1] == str(tmp_path / "emotion_directions")  # the CLI default is relative to the cwd
    no_dir = _zonos2_cmd(_zonos2(tmp_path / "nowhere"), _zonos2_item())
    assert "--emotion-dir" not in no_dir  # absent folder: flag left out, the CLI reports it if an emotion was asked for


def test_zonos2_emotion_knobs_map_to_the_cli_flags(tmp_path: Path) -> None:
    from vndub.text import style_prompt

    inst = _zonos2(tmp_path)
    # knobs of vndub.text.style_prompt.to_knobs: probabilities in _ZONOS_ORDER, valence / arousal / strength
    probs = {"happy": 0.6, "angry": 0.1, "neutral": 0.3}
    vec = [probs.get(k, 0.0) for k in style_prompt._ZONOS_ORDER]
    knobs = {"emotion": vec, "valence": 0.85, "arousal": 0.3, "emotion_strength": 1.1, "speaking_rate": 14.0}
    cmd = _zonos2_cmd(inst, _zonos2_item(knobs=knobs))
    assert cmd[cmd.index("--emotion") + 1] == "happy=0.600,angry=0.100"  # only the directions the CLI ships; neutral has none
    assert cmd[cmd.index("--emotion-valence") + 1] == "0.700"  # 0..1 around 0.5 -> signed axis
    assert cmd[cmd.index("--emotion-arousal") + 1] == "-0.400"
    assert cmd[cmd.index("--emotion-strength") + 1] == "1.100"
    assert "--speaking-rate" not in cmd  # a bucket index in the CLI, not a rate
    assert {a for a in cmd if a.startswith("--")} <= ZONOS2_CLI_FLAGS


def test_zonos2_emotion_order_is_the_style_prompt_order() -> None:
    from adapters.gguf import cli_impl
    from vndub.text import style_prompt

    assert list(cli_impl.ZONOS_EMOTION_ORDER) == style_prompt._ZONOS_ORDER


def test_zonos2_without_a_reference_asks_for_no_emotion(tmp_path: Path) -> None:
    """The CLI refuses emotion control without --speaker / --clone, so a reference-less item is plain synthesis."""
    cmd = _zonos2_cmd(_zonos2(tmp_path), _zonos2_item(timbre_ref=None, knobs={"emotion": {"happy": 1.0}, "valence": 0.9}))
    assert "--clone" not in cmd and "--emotion" not in cmd and "--emotion-valence" not in cmd and "--emotion-strength" not in cmd


def test_zonos2_hands_the_configured_ffmpeg_to_the_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZONOS2_FFMPEG", raising=False)
    assert _zonos2(tmp_path).child_env() is None
    env = _zonos2(tmp_path, ffmpeg="C:/tools/ffmpeg.exe").child_env()
    assert env is not None and env["ZONOS2_FFMPEG"] == "C:/tools/ffmpeg.exe"
    monkeypatch.setenv("ZONOS2_FFMPEG", "mine")
    assert _zonos2(tmp_path, ffmpeg="C:/tools/ffmpeg.exe").child_env() is None  # the user's own variable wins


# -- an empty value takes its flag with it ------------------------------------------------------------------------------


def _fish(args):
    from adapters.gguf import cli_impl

    inst = object.__new__(cli_impl.FishS2CLI)
    inst.options = {"args": args}
    inst.binary, inst.model_dir, inst.model_file = Path("audiocpp_cli"), Path("m"), Path("m/f.gguf")
    return inst


def test_empty_value_drops_its_flag_not_the_next_one() -> None:
    spec = registry.get("fish_s2")
    inst = _fish(spec.options["args"])
    item = {"text": "hi", "timbre_ref": "ref.wav", "timbre_ref_text": "", "out": "o.wav", "seed": 5}  # no reference transcript
    cmd = inst.command(inst._values(item, Path("t.txt")))
    assert "--reference-text" not in cmd and cmd[cmd.index("--seed") + 1] == "5"  # it used to swallow --seed
    assert cmd[cmd.index("--voice-ref") + 1] == "ref.wav"
    item["timbre_ref"] = ""
    cmd = inst.command(inst._values(item, Path("t.txt")))
    assert "--voice-ref" not in cmd and "--reference-text" not in cmd and cmd[-2:] == ["--out", "o.wav"]


def test_command_keeps_flags_with_literal_values_and_bare_flags() -> None:
    inst = _fish(["--task", "clon", "--flag", "{ref}", "--bare", "--x", "{ref_text}", "--y", "{text}"])
    cmd = inst.command({"ref": "", "ref_text": "", "text": "t"})
    assert cmd[1:] == ["--task", "clon", "--bare", "--y", "t"]
    assert _fish(["{ref}", "--k", "v"]).command({"ref": ""})[1:] == ["--k", "v"]  # a lone empty placeholder is still dropped


# -- the shipped binary names ----------------------------------------------------------------------------------------------


def test_default_executables_are_the_shipped_binaries(monkeypatch: pytest.MonkeyPatch) -> None:
    assert tools.TOOL_INFO["audiocpp"].exe == "audiocpp_cli" and tools.TOOL_INFO["zonos2_cli"].exe == "zonos2-cli"
    found = {"audiocpp_cli": "C:/t/audiocpp_cli.exe", "zonos2-cli": "C:/t/zonos2-cli.exe"}
    monkeypatch.setattr(tools.shutil, "which", lambda name: found.get(name))
    t = tools.Tools()
    assert t.resolve("audiocpp") == Path("C:/t/audiocpp_cli.exe") and t.resolve("zonos2_cli") == Path("C:/t/zonos2-cli.exe")
    found.clear()
    found["zonos2"] = "C:/t/zonos2.exe"  # the old default name still works as an alternative
    assert t.resolve("zonos2_cli") == Path("C:/t/zonos2.exe") and t.resolve("audiocpp") is None


# -- the child never inherits the RPC stdin ------------------------------------------------------------------------------


def test_cli_child_gets_devnull_stdin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    from adapters.gguf import cli_impl

    seen: dict = {}

    def fake_run(cmd, **kw):
        seen.update(kw)
        Path(cmd[cmd.index("--out") + 1]).write_bytes(b"")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(cli_impl.subprocess, "run", fake_run)
    monkeypatch.setattr(cli_impl, "load_audio", lambda p, sr: ([0.0], 44100))
    inst = _fish(["--out", "{out}"])
    inst.generate({"text": "hi", "out": str(tmp_path / "o.wav")})
    assert seen["stdin"] == subprocess.DEVNULL


def test_applio_job_gets_devnull_stdin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    from adapters.rvc import applio_impl

    seen: dict = {}

    def fake_run(cmd, **kw):
        seen.update(kw)
        return subprocess.CompletedProcess(cmd, 0, '{"ok": true}\n', None)

    monkeypatch.setattr(applio_impl.subprocess, "run", fake_run)
    res = applio_impl.run_job(tmp_path, {"op": "x"}, None, tmp_path / "job.log")
    assert res["ok"] is True and seen["stdin"] == subprocess.DEVNULL

"""llama-server command line and the all-layers-on-the-GPU check (no binary, no GPU, no network)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from vndub.backends.base import BackendUnavailable
from vndub.backends.llm import llamacpp as lc

HELP_WITH_FIT = "  -ngl, --gpu-layers N ...\n  --fit [on|off] ...\n  -np, --parallel N ..."
HELP_OLD = "  -ngl, --n-gpu-layers N\n  -np, --parallel N"
BIN = Path("C:/tools/llama-server.exe")
GGUF = Path("C:/models/x.gguf")


def _opt(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def test_build_args_asks_for_every_layer_and_switches_fit_off() -> None:
    args = lc.build_args(BIN, GGUF, 8089, {"ctx": 16384, "n_gpu_layers": -1}, HELP_WITH_FIT)
    assert _opt(args, "-ngl") == lc.ALL_LAYERS
    assert _opt(args, "--fit") == "off"
    assert _opt(args, "-np") == "1"
    assert _opt(args, "-c") == "16384"
    assert args[0] == str(BIN) and _opt(args, "-m") == str(GGUF)


def test_build_args_old_build_without_fit_flag() -> None:
    args = lc.build_args(BIN, GGUF, 8089, {}, HELP_OLD)
    assert "--fit" not in args
    assert _opt(args, "-ngl") == lc.ALL_LAYERS


def test_build_args_explicit_layers_are_the_users_choice() -> None:
    args = lc.build_args(BIN, GGUF, 8089, {"n_gpu_layers": 30}, HELP_WITH_FIT)
    assert _opt(args, "-ngl") == "30"
    assert _opt(args, "--fit") == "off"


def test_build_args_parallel_slots_scale_the_context() -> None:
    args = lc.build_args(BIN, GGUF, 8089, {"ctx": 8192, "parallel": 2, "reasoning_budget": 0, "extra_args": ["-fa", "on"]}, "")
    assert _opt(args, "-c") == "16384"
    assert _opt(args, "-np") == "2"
    assert _opt(args, "--reasoning-budget") == "0"
    assert args[-2:] == ["-fa", "on"]


def test_parse_offload_takes_the_last_line() -> None:
    log = "load_tensors: offloaded 10/49 layers to GPU\nx\nload_tensors: offloaded 49/49 layers to GPU\n"
    assert lc.parse_offload(log) == (49, 49)
    assert lc.parse_offload("nothing here") is None


@pytest.mark.parametrize(
    ("requested", "want_gpu", "offload", "bad"),
    [
        (-1, True, (49, 49), False),
        (-1, True, (30, 49), True),
        (-1, True, (0, 49), True),
        (-1, True, None, False),  # no line: another build or no log, nothing proven
        (30, True, (30, 49), False),  # the user's own partial offload
        (-1, False, (0, 49), False),  # CPU run allowed by the registry
    ],
)
def test_offload_problem(requested: int, want_gpu: bool, offload: tuple[int, int] | None, bad: bool) -> None:
    assert (lc.offload_problem(requested, want_gpu, offload) is not None) is bad


class _Proc:
    def __init__(self) -> None:
        self.ended = False

    def terminate(self) -> None:
        self.ended = True

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        self.ended = True

    def poll(self) -> int | None:
        return None


def _backend(tmp_path: Path, device: str, options: dict[str, Any] | None = None) -> tuple[lc.LlamaCppBackend, _Proc]:
    b = lc.LlamaCppBackend(lc.SPECS[0], options or {}, {"device": device, "log_dir": tmp_path})
    b.gguf = GGUF
    b.port = 8089
    proc = _Proc()
    b.proc = proc  # type: ignore[assignment]
    return b, proc


def test_started_rejects_a_partial_offload(tmp_path: Path) -> None:
    log = tmp_path / "llama_server.log"
    log.write_text("old run: offloaded 49/49 layers to GPU\n", encoding="utf-8")
    start = log.stat().st_size
    with open(log, "a", encoding="utf-8") as f:
        f.write("load_tensors: offloaded 31/49 layers to GPU\n")
    b, proc = _backend(tmp_path, "cuda:0")
    with pytest.raises(BackendUnavailable, match="31/49"):
        b._started(["x", "-m", "y"], log, start)
    assert proc.ended and b.proc is None and not b.loaded


def test_started_reports_the_full_offload(tmp_path: Path) -> None:
    log = tmp_path / "llama_server.log"
    log.write_text("load_tensors: offloaded 49/49 layers to GPU\n", encoding="utf-8")
    b, proc = _backend(tmp_path, "cuda:0")
    info = b._started([str(BIN), "-m", str(GGUF)], log, 0)
    assert info["gpu_layers"] == "49/49" and info["device"] == "cuda" and info["args"] == ["-m", str(GGUF)]
    assert b.loaded and not proc.ended
    assert b.load().get("gpu_layers") == "49/49"  # a second load() of the running server repeats the facts


def test_started_on_cpu_or_with_users_layer_count_does_not_judge(tmp_path: Path) -> None:
    log = tmp_path / "llama_server.log"
    log.write_text("load_tensors: offloaded 0/49 layers to GPU\n", encoding="utf-8")
    b, _ = _backend(tmp_path, "cpu")
    assert b._started(["x"], log, 0)["device"] == "cpu"
    b2, _ = _backend(tmp_path, "cuda:0", {"n_gpu_layers": 0})
    assert b2._started(["x"], log, 0)["gpu_layers"] == "0/49"


def test_builtin_llm_estimates_follow_ctx_and_parallel() -> None:
    from vndub.backends.base import estimate_vram_gb

    gemma, qwen, hy = lc.SPECS
    assert estimate_vram_gb(gemma) == pytest.approx(7.5, abs=0.1)  # never on the default 7.5 - 0.5 budget
    assert estimate_vram_gb(qwen) == pytest.approx(6.3, abs=0.1)
    assert estimate_vram_gb(hy) == pytest.approx(6.0, abs=0.1)
    assert estimate_vram_gb(hy, {"ctx": 16384}) == pytest.approx(7.0, abs=0.1)
    assert estimate_vram_gb(qwen, {"parallel": 2}) > estimate_vram_gb(qwen)
    assert estimate_vram_gb(gemma) == pytest.approx(gemma.vram_estimate_gb, abs=0.1)
    assert estimate_vram_gb(qwen) == pytest.approx(qwen.vram_estimate_gb, abs=0.1)

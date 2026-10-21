"""ModelRegistry with a fake CUDA: one-resident-heavy rule, budget checks, fallbacks, policy."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from vndub.backends import registry
from vndub.backends.base import BackendSpec, BackendUnavailable
from vndub.backends.weights import MARKER, local_dir
from vndub.gpu import GB, ModelRegistry, NoCuda, VramBudgetExceeded


class FakeCuda:
    def __init__(self, total_gb: float = 8.0, other_gb: float = 0.3) -> None:
        self.total = int(total_gb * GB)
        self.used = int(other_gb * GB)  # desktop / other processes
        self.empty_calls = 0

    def available(self) -> bool:
        return True

    def mem_get_info(self) -> tuple[int, int]:
        return self.total - self.used, self.total

    def empty_cache(self) -> None:
        self.empty_calls += 1


class FakeBackend:
    def __init__(self, spec: BackendSpec, options: dict[str, Any], device: str, cuda: FakeCuda | None, log: list) -> None:
        self.spec, self.options, self.device, self.cuda, self.log = spec, options, device, cuda, log
        self.loaded = False

    def load(self) -> dict[str, Any]:
        self.loaded = True
        if self.cuda and self.device.startswith("cuda"):
            self.cuda.used += int(self.spec.vram_estimate_gb * GB)
        self.log.append(("load", self.spec.name, self.device))
        return {"device": self.device}

    def unload(self) -> None:
        if self.loaded and self.cuda and self.device.startswith("cuda"):
            self.cuda.used -= int(self.spec.vram_estimate_gb * GB)
        self.loaded = False
        self.log.append(("unload", self.spec.name, self.device))

    def call(self, op: str, items=None, **kw):
        return []

    def selftest(self) -> dict[str, Any]:
        return {"ok": True}

    def peak_vram(self) -> float | None:
        return self.spec.vram_estimate_gb if self.device.startswith("cuda") else None


def spec(name: str, vram: float = 3.0, heavy: bool = True, cpu_ok: bool = True) -> BackendSpec:
    return BackendSpec(name=name, role="tts", env="fake", adapter=name, license="MIT", commercial_ok=True, vram_estimate_gb=vram, heavy=heavy, cpu_ok=cpu_ok)


@pytest.fixture
def fake_specs(monkeypatch: pytest.MonkeyPatch) -> dict[str, BackendSpec]:
    specs = {s.name: s for s in [spec("big_a", 4.5), spec("big_b", 3.8), spec("huge", 7.9), spec("small", 0.4, heavy=False), spec("gpu_only", 3.0, cpu_ok=False)]}
    real = registry.all_specs()
    monkeypatch.setattr(registry, "_SPECS", {**real, **specs})
    return specs


def make_reg(tmp_path: Path, cuda: Any, **kw: Any) -> tuple[ModelRegistry, list]:
    log: list = []
    reg = ModelRegistry(
        cuda=cuda,
        check_usable=kw.pop("check_usable", False),
        factory=lambda s, o, d: FakeBackend(s, o, d, cuda if isinstance(cuda, FakeCuda) else None, log),
        models_dir=tmp_path / "models",
        log_dir=tmp_path / "logs",
        timings_path=tmp_path / "gpu_events.jsonl",
        **kw,
    )
    return reg, log


def test_one_resident_heavy_model(tmp_path: Path, fake_specs) -> None:
    cuda = FakeCuda()
    reg, log = make_reg(tmp_path, cuda)
    a = reg.acquire("big_a")
    assert a.device == "cuda:0" and reg.resident == ["big_a"]
    b = reg.acquire("big_b")
    assert reg.resident == ["big_b"]
    assert not a.loaded and b.loaded
    assert log[:3] == [("load", "big_a", "cuda:0"), ("unload", "big_a", "cuda:0"), ("load", "big_b", "cuda:0")]
    assert cuda.empty_calls >= 1
    # light (CPU) backends coexist with the heavy one
    s = reg.acquire("small")
    assert s.device == "cpu" and set(reg.resident) == {"big_b", "small"}
    reg.close()
    assert reg.resident == [] and cuda.used == int(0.3 * GB)


def test_reacquire_same_model_does_not_reload(tmp_path: Path, fake_specs) -> None:
    reg, log = make_reg(tmp_path, FakeCuda())
    a1 = reg.acquire("big_a")
    a2 = reg.acquire("big_a")
    assert a1 is a2 and [e for e in log if e[0] == "load"] == [("load", "big_a", "cuda:0")]
    a3 = reg.acquire("big_a", {"seed": 1})  # different options -> new instance
    assert a3 is not a1 and not a1.loaded
    reg.close()


def test_budget_refusal_is_clear(tmp_path: Path, fake_specs) -> None:
    reg, _ = make_reg(tmp_path, FakeCuda(), budget_gb=7.5, safety_margin_gb=0.5)
    with pytest.raises(VramBudgetExceeded) as ei:
        reg.acquire("huge")
    msg = str(ei.value)
    assert "huge" in msg and "budget 7.5" in msg and "7.9" in msg
    assert reg.resident == []


def test_free_memory_refusal(tmp_path: Path, fake_specs) -> None:
    cuda = FakeCuda(total_gb=8.0, other_gb=5.0)  # another app holds 5 GB
    reg, _ = make_reg(tmp_path, cuda)
    with pytest.raises(VramBudgetExceeded, match=r"only 3\.00 GB free"):
        reg.acquire("big_a")
    cuda.used = int(0.3 * GB)
    assert reg.acquire("big_a").loaded
    reg.close()


def test_release_and_events(tmp_path: Path, fake_specs) -> None:
    reg, _ = make_reg(tmp_path, FakeCuda())
    reg.acquire("big_a")
    reg.acquire("small")
    reg.release("big_a")
    assert reg.resident == ["small"]
    reg.release()
    assert reg.resident == []
    events = [json.loads(l) for l in (tmp_path / "gpu_events.jsonl").read_text(encoding="utf-8").splitlines()]
    kinds = [(e["event"], e["model"]) for e in events]
    assert ("load", "big_a") in kinds and ("unload", "big_a") in kinds
    unload = next(e for e in events if e["event"] == "unload" and e["model"] == "big_a")
    assert unload["peak_vram_gb"] == 4.5


def test_cpu_fallback_rules(tmp_path: Path, fake_specs) -> None:
    reg, _ = make_reg(tmp_path, NoCuda(), allow_cpu_fallback=False)
    with pytest.raises(BackendUnavailable, match="CUDA not available"):
        reg.acquire("big_a")
    reg2, _ = make_reg(tmp_path, NoCuda(), allow_cpu_fallback=True)
    assert reg2.acquire("big_a").device == "cpu"
    with pytest.raises(BackendUnavailable, match="CUDA not available"):
        reg2.acquire("gpu_only")
    assert reg2.acquire("small").device == "cpu"
    reg2.close()


# ---------------------------------------------------------------------------
# resolution against the real registry (usability: env, weights, license)
# ---------------------------------------------------------------------------


def _fake_env(root: Path, env: str) -> None:
    scripts = root / ".venvs" / env / ("Scripts" if sys.platform.startswith("win") else "bin")
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / ("python.exe" if sys.platform.startswith("win") else "python")).write_bytes(b"")
    (root / ".venvs" / env / "pyvenv.cfg").write_text("home = x\n", encoding="utf-8")


def _fake_weights(models: Path, name: str) -> None:
    s = registry.get(name)
    for w in s.weights:
        d = local_dir(models, w, s.name)
        d.mkdir(parents=True, exist_ok=True)
        (d / MARKER).write_text("test", encoding="utf-8")


def test_fallback_chain_to_usable_fallback(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _fake_env(root, "asr")
    _fake_weights(tmp_path / "models", "whisper_large_v3_turbo")  # anime_whisper weights absent
    reg, _ = make_reg(tmp_path, NoCuda(), check_usable=True, root=root, allow_cpu_fallback=True)
    s, sub = reg.resolve("anime_whisper")
    assert s.name == "whisper_large_v3_turbo"
    assert sub is not None and "anime_whisper: weights missing" in sub.reason and "vndub models pull anime_whisper" in sub.reason


def test_unusable_without_mock_lists_all_reasons(tmp_path: Path) -> None:
    reg, _ = make_reg(tmp_path, NoCuda(), check_usable=True, root=tmp_path / "empty_root")
    with pytest.raises(BackendUnavailable) as ei:
        reg.resolve("anime_whisper")
    msg = str(ei.value)
    assert "anime_whisper: env 'asr' not created" in msg and "whisper_large_v3_turbo: env 'asr' not created" in msg


def test_test_profile_substitutes_mock(tmp_path: Path) -> None:
    reg, _ = make_reg(tmp_path, NoCuda(), check_usable=True, root=tmp_path / "empty_root", fallback_to_mock=True)
    s, sub = reg.resolve("anime_whisper")
    assert s.name == "mock_asr" and sub is not None and sub.used == "mock_asr"
    reg.acquire("anime_whisper")
    assert reg.substitutions and reg.substitutions[0].requested == "anime_whisper"


def test_commercial_policy_filters_noncommercial(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _fake_env(root, "hf")
    _fake_weights(tmp_path / "models", "audeering_adv")
    reg, _ = make_reg(tmp_path, NoCuda(), check_usable=True, root=root, license_policy="commercial", allow_cpu_fallback=True)
    with pytest.raises(BackendUnavailable, match="license_policy=commercial"):
        reg.resolve("audeering_adv")
    research, _ = make_reg(tmp_path, NoCuda(), check_usable=True, root=root, license_policy="research", allow_cpu_fallback=True)
    assert research.resolve("audeering_adv")[0].name == "audeering_adv"


def test_heavy_needs_cuda_or_cpu_fallback_during_resolution(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _fake_env(root, "hf")
    _fake_weights(tmp_path / "models", "voxcpm2")  # cpu_ok=False
    reg, _ = make_reg(tmp_path, NoCuda(), check_usable=True, root=root, allow_cpu_fallback=True, fallback_to_mock=True)
    s, sub = reg.resolve("voxcpm2")
    assert s.name == "mock_tts" and "adapter needs a GPU" in sub.reason
    gpu, _ = make_reg(tmp_path, FakeCuda(), check_usable=True, root=root)
    assert gpu.resolve("voxcpm2")[0].name == "voxcpm2"

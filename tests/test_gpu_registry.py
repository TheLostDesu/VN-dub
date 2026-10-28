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
    kw.setdefault("vram_retry_s", 0.0)  # the second look at the free memory without the 3 s wait
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
    _fake_env(root, "main")
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
    assert "anime_whisper: env 'main' not created" in msg and "whisper_large_v3_turbo: env 'main' not created" in msg


def test_test_profile_substitutes_mock(tmp_path: Path) -> None:
    reg, _ = make_reg(tmp_path, NoCuda(), check_usable=True, root=tmp_path / "empty_root", fallback_to_mock=True)
    s, sub = reg.resolve("anime_whisper")
    assert s.name == "mock_asr" and sub is not None and sub.used == "mock_asr"
    reg.acquire("anime_whisper")
    assert reg.substitutions and reg.substitutions[0].requested == "anime_whisper"


def test_commercial_policy_filters_noncommercial(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _fake_env(root, "main")
    _fake_weights(tmp_path / "models", "audeering_adv")
    reg, _ = make_reg(tmp_path, NoCuda(), check_usable=True, root=root, license_policy="commercial", allow_cpu_fallback=True)
    with pytest.raises(BackendUnavailable, match="license_policy=commercial"):
        reg.resolve("audeering_adv")
    research, _ = make_reg(tmp_path, NoCuda(), check_usable=True, root=root, license_policy="research", allow_cpu_fallback=True)
    assert research.resolve("audeering_adv")[0].name == "audeering_adv"


def test_heavy_needs_cuda_or_cpu_fallback_during_resolution(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _fake_env(root, "main")
    _fake_weights(tmp_path / "models", "voxcpm2")  # cpu_ok=False
    reg, _ = make_reg(tmp_path, NoCuda(), check_usable=True, root=root, allow_cpu_fallback=True, fallback_to_mock=True)
    s, sub = reg.resolve("voxcpm2")
    assert s.name == "mock_tts" and "adapter needs a GPU" in sub.reason
    gpu, _ = make_reg(tmp_path, FakeCuda(), check_usable=True, root=root)
    assert gpu.resolve("voxcpm2")[0].name == "voxcpm2"


# -- never from system RAM (2026-10-22) ---------------------------------------


def test_dynamic_vram_estimate_grows_with_ctx_and_parallel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vndub.backends.base import estimate_vram_gb

    llm = BackendSpec(
        name="fake_llm", role="llm", env="fake", adapter="x", license="MIT", commercial_ok=True, vram_estimate_gb=6.0,
        options={"ctx": 8192, "vram": {"weights_gib": 4.0, "kv_bytes_per_token": 131072, "overhead_gib": 1.0}},
    )
    monkeypatch.setattr(registry, "_SPECS", {**registry.all_specs(), "fake_llm": llm})
    assert estimate_vram_gb(llm) == 6.0  # 4 + 1 (8K tokens x 128 KiB) + 1
    assert estimate_vram_gb(llm, {"ctx": 16384}) == 7.0
    assert estimate_vram_gb(llm, {"parallel": 2}) == 7.0
    reg, _log = make_reg(tmp_path, FakeCuda(total_gb=8.0, other_gb=0.3), budget_gb=7.5)
    reg.acquire("fake_llm")
    ev = [json.loads(l) for l in (tmp_path / "gpu_events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert ev[-1]["vram_estimate_gb"] == 6.0
    with pytest.raises(VramBudgetExceeded):
        reg.acquire("fake_llm", {"parallel": 3})  # 4 + 3 + 1 = 8 GiB > 7.5 - 0.5


class CpuTorchBackend(FakeBackend):
    def load(self) -> dict[str, Any]:
        self.loaded = True
        return {"device": "cpu"}  # the env's torch has no CUDA: the adapter fell back silently


def test_adapter_that_lands_on_cpu_is_refused_unless_allowed(tmp_path: Path, fake_specs) -> None:
    log: list = []

    def make(allow: bool) -> ModelRegistry:
        return ModelRegistry(
            cuda=FakeCuda(), check_usable=False, factory=lambda s, o, d: CpuTorchBackend(s, o, d, None, log),
            models_dir=tmp_path / "m", log_dir=tmp_path / "l", allow_cpu_fallback=allow,
        )

    with pytest.raises(BackendUnavailable, match="no CUDA"):
        make(False).acquire("big_a")
    assert ("unload", "big_a", "cuda:0") in log  # the CPU-bound process was ended
    reg = make(True)
    b = reg.acquire("big_a")  # cpu_ok and the profile allows it: a warning, stays resident
    assert reg.resident == ["big_a"] and b.loaded
    with pytest.raises(BackendUnavailable):
        make(True).acquire("gpu_only")  # cpu_ok=False: never


def test_stage_level_substitution_is_recorded(tmp_path: Path, fake_specs) -> None:
    reg, _log = make_reg(tmp_path, FakeCuda())
    reg.acquire("big_b", substituted_for="huge", reason="huge: needs ~7.9 GB > budget")
    assert [(s.requested, s.used) for s in reg.substitutions] == [("huge", "big_b")]
    ev = [json.loads(l) for l in (tmp_path / "gpu_events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert ev[-1]["substituted_for"] == "huge" and ev[-1]["device_actual"] == "cuda:0"


class FailingLoadBackend(FakeBackend):
    def load(self) -> dict[str, Any]:
        raise RuntimeError("out of memory while loading")


def test_substitution_is_not_recorded_when_the_fallback_fails_to_load(tmp_path: Path, fake_specs) -> None:
    """A fallback that never came up was not used: the report, gpu_events and the app must not announce ``huge -> big_b``."""
    log: list = []
    reg = ModelRegistry(
        cuda=FakeCuda(), check_usable=False, factory=lambda s, o, d: FailingLoadBackend(s, o, d, None, log),
        models_dir=tmp_path / "m", log_dir=tmp_path / "l", vram_retry_s=0.0,
    )
    with pytest.raises(RuntimeError, match="out of memory"):
        reg.acquire("big_b", substituted_for="huge", reason="huge: needs ~7.9 GB > budget")
    assert reg.substitutions == [] and reg.resident == []
    assert ("unload", "big_b", "cuda:0") in log  # the failed load was cleaned up


def test_substitution_is_not_recorded_when_the_fallback_is_refused_for_vram(tmp_path: Path, fake_specs) -> None:
    reg, _log = make_reg(tmp_path, FakeCuda(total_gb=8.0, other_gb=5.0))  # another app holds 5 GB: big_b (3.8) does not fit
    with pytest.raises(VramBudgetExceeded):
        reg.acquire("big_b", substituted_for="huge", reason="huge: needs ~7.9 GB > budget")
    assert reg.substitutions == []
    # the next, successful use of the fallback is the one that gets announced
    reg2, _ = make_reg(tmp_path, FakeCuda())
    reg2.acquire("big_b", substituted_for="huge", reason="huge: needs ~7.9 GB > budget")
    assert [(s.requested, s.used) for s in reg2.substitutions] == [("huge", "big_b")]


# -- would_fit: the answer acquire will give, without loading (2026-10-22, round 8) ----------------------------


def test_would_fit_answers_without_loading(tmp_path: Path, fake_specs) -> None:
    reg, log = make_reg(tmp_path, FakeCuda(total_gb=8.0, other_gb=0.3), budget_gb=7.5, safety_margin_gb=0.5)
    s, why = reg.would_fit("big_a")
    assert s.name == "big_a" and why is None
    s, why = reg.would_fit("huge")  # 7.9 GB against a 7.5 - 0.5 budget: the estimate says no
    assert s.name == "huge" and why is not None and "budget 7.5" in why and "huge" in why
    # nothing was loaded, nothing recorded, nothing touched
    assert log == [] and reg.heavy is None and reg.light == {} and reg.resident == [] and reg.substitutions == []
    assert not (tmp_path / "gpu_events.jsonl").exists()
    # and it is the message acquire raises
    with pytest.raises(VramBudgetExceeded) as ei:
        reg.acquire("huge")
    assert str(ei.value) == why
    reg.close()


def test_would_fit_checks_the_live_free_memory(tmp_path: Path, fake_specs) -> None:
    cuda = FakeCuda(total_gb=8.0, other_gb=5.0)  # another application holds 5 GB
    reg, log = make_reg(tmp_path, cuda, vram_retry_s=0.0)
    s, why = reg.would_fit("big_a")
    assert s.name == "big_a" and why is not None and "only 3.00 GB free" in why and log == []
    cuda.used = int(0.3 * GB)
    assert reg.would_fit("big_a") == (s, None)


def test_would_fit_credits_the_resident_heavy_model_acquire_would_unload(tmp_path: Path, fake_specs) -> None:
    cuda = FakeCuda(total_gb=8.0, other_gb=0.3)
    reg, log = make_reg(tmp_path, cuda, vram_retry_s=0.0)
    first = reg.acquire("big_a")  # 4.5 GB resident: only 3.2 GB are free now, big_b wants 3.8
    assert cuda.mem_get_info()[0] / GB < 3.8 + 0.5
    s, why = reg.would_fit("big_b")
    assert why is None  # acquire unloads big_a first, so it loads
    assert reg.resident == ["big_a"] and first.loaded  # ... but asking changed nothing
    assert reg.would_fit("big_a") == (registry.get("big_a"), None)  # the resident one needs no check at all
    assert reg.acquire("big_b").loaded and reg.resident == ["big_b"]
    reg.close()


def test_would_fit_cpu_and_light_rules(tmp_path: Path, fake_specs) -> None:
    reg, _ = make_reg(tmp_path, NoCuda(), allow_cpu_fallback=False)
    s, why = reg.would_fit("big_a")
    assert why is not None and "CUDA not available" in why
    assert reg.would_fit("small")[1] is None  # a light backend runs on the CPU by design
    reg2, _ = make_reg(tmp_path, NoCuda(), allow_cpu_fallback=True)
    assert reg2.would_fit("big_a")[1] is None
    assert "CUDA not available" in (reg2.would_fit("gpu_only")[1] or "")  # cpu_ok=False: never


def test_would_fit_reports_an_unusable_backend_instead_of_raising(tmp_path: Path) -> None:
    reg, _ = make_reg(tmp_path, NoCuda(), check_usable=True, root=tmp_path / "empty_root")
    s, why = reg.would_fit("anime_whisper")
    assert s.name == "anime_whisper" and why is not None and "env 'main' not created" in why
    root = tmp_path / "root"
    _fake_env(root, "main")
    _fake_weights(tmp_path / "models", "audeering_adv")
    commercial, _ = make_reg(tmp_path, NoCuda(), check_usable=True, root=root, license_policy="commercial", allow_cpu_fallback=True)
    assert "license_policy=commercial" in (commercial.would_fit("audeering_adv")[1] or "")
    research, _ = make_reg(tmp_path, NoCuda(), check_usable=True, root=root, license_policy="research", allow_cpu_fallback=True)
    assert research.would_fit("audeering_adv")[1] is None


# -- the second look at the live free memory ----------------------------------------------------------------


class GrowingCuda(FakeCuda):
    """Free memory as the driver reports it right after an unload: low at first, the real number a moment later."""

    def __init__(self, readings_gb: list[float], total_gb: float = 8.0) -> None:
        super().__init__(total_gb=total_gb, other_gb=0.0)
        self.readings = list(readings_gb)
        self.reads = 0

    def mem_get_info(self) -> tuple[int, int]:
        free = self.readings[min(self.reads, len(self.readings) - 1)]
        self.reads += 1
        return int(free * GB), self.total


def test_acquire_looks_again_once_when_the_live_reading_is_low(tmp_path: Path, fake_specs) -> None:
    cuda = GrowingCuda([3.0, 7.5])  # big_a needs 4.5: the first reading is the VRAM that is still being handed back
    reg, _ = make_reg(tmp_path, cuda, vram_retry_s=0.0)
    assert reg.acquire("big_a").loaded
    assert cuda.reads == 2 and cuda.empty_calls >= 1  # the probe's cached reading was dropped before the second look
    assert "looking again" in (tmp_path / "logs" / "gpu.log").read_text(encoding="utf-8")
    reg.close()


def test_acquire_gives_up_after_the_second_look(tmp_path: Path, fake_specs) -> None:
    cuda = GrowingCuda([3.0, 3.1])
    reg, log = make_reg(tmp_path, cuda, vram_retry_s=0.0)
    with pytest.raises(VramBudgetExceeded, match=r"only 3\.10 GB free"):
        reg.acquire("big_a")
    assert cuda.reads == 2 and log == []


def test_the_static_estimate_is_not_retried(tmp_path: Path, fake_specs, monkeypatch: pytest.MonkeyPatch) -> None:
    from vndub import gpu

    slept: list[float] = []
    monkeypatch.setattr(gpu.time, "sleep", lambda s: slept.append(s))
    cuda = GrowingCuda([8.0])
    reg, _ = make_reg(tmp_path, cuda, budget_gb=7.5, vram_retry_s=3.0)
    with pytest.raises(VramBudgetExceeded):
        reg.acquire("huge")  # 7.9 > 7.5 - 0.5 whatever the GPU is doing
    assert cuda.reads == 0 and slept == []
    low = GrowingCuda([3.0, 7.5])
    reg2, _ = make_reg(tmp_path, low, vram_retry_s=3.0)
    reg2.acquire("big_a")
    assert slept == [3.0]  # the configured wait, once
    reg2.close()


def test_retry_default_and_from_config(tmp_path: Path) -> None:
    from vndub.config import load_config
    from vndub.gpu import from_config
    from vndub.paths import WorkPaths
    from vndub.tools import Tools

    assert ModelRegistry().vram_retry_s == 3.0
    cfg = load_config(Path(__file__).resolve().parent.parent / "configs" / "pipeline.test.yaml", {})
    paths = WorkPaths.create(tmp_path / "work", tmp_path / "out")
    reg = from_config(cfg, paths, Tools.load(cfg.resolve(cfg.tools_file), cfg.tools), cuda=NoCuda())
    assert reg.vram_retry_s == 3.0


# -- abort_all reaches a backend that is still loading (2026-10-28, release 1.0.0) --------------------------------------


class BlockingLoadBackend(FakeBackend):
    """A load that waits (a real model takes minutes) until ``abort`` kills it, like RemoteBackend does."""

    def __init__(self, *a: Any) -> None:
        super().__init__(*a)
        import threading

        self.killed = threading.Event()

    def load(self) -> dict[str, Any]:
        self.log.append(("loading", self.spec.name))
        if not self.killed.wait(timeout=10):
            raise AssertionError("abort never came")
        raise RuntimeError("adapter was aborted (job cancelled) during 'load'")

    def abort(self) -> None:
        self.log.append(("abort", self.spec.name))
        self.killed.set()


def test_abort_all_reaches_a_backend_that_is_still_loading(tmp_path: Path, fake_specs) -> None:
    """Cancel during a model load used to wait for the whole load (up to the 1800 s RPC timeout): the backend was not
    resident yet, so abort_all did not know about it."""
    import threading

    log: list = []
    reg = ModelRegistry(
        cuda=FakeCuda(), check_usable=False, factory=lambda s, o, d: BlockingLoadBackend(s, o, d, None, log),
        models_dir=tmp_path / "m", log_dir=tmp_path / "l", vram_retry_s=0.0,
    )
    errors: list[BaseException] = []

    def job() -> None:
        try:
            reg.acquire("big_a")
        except BaseException as e:  # noqa: BLE001 - the test records whatever the cancelled load raised
            errors.append(e)

    t = threading.Thread(target=job)
    t.start()
    for _ in range(200):  # until the job thread is inside load()
        if ("loading", "big_a") in log:
            break
        threading.Event().wait(0.01)
    assert reg._loading is not None and reg.resident == []
    reg.abort_all()
    t.join(timeout=5)
    assert not t.is_alive()
    assert ("abort", "big_a") in log and ("unload", "big_a", "cuda:0") in log
    assert len(errors) == 1 and "aborted" in str(errors[0])
    assert reg._loading is None and reg.resident == [] and reg.substitutions == []
    reg.abort_all()  # nothing loading, nothing resident: a no-op

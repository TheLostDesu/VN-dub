"""Static consistency of the backend registry, adapter tables and env files (no heavy imports)."""

from __future__ import annotations

import ast
import importlib
import re
import sys
from pathlib import Path

import pytest

from vndub.backends import registry
from vndub.backends.base import ROLES
from vndub.backends.registry import inject_tools

REPO = Path(__file__).resolve().parent.parent
SPECS = registry.all_specs()
REMOTE = [s for s in SPECS.values() if s.env is not None]
ENVS = sorted({s.env for s in REMOTE})
WEIGHT_KINDS = {"hf", "hf_file", "url", "env", "builtin", "modelscope"}


def _table(env: str) -> dict[str, str]:
    path = REPO / "adapters" / env / "table.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "TABLE" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{path} has no TABLE literal")


def test_names_roles_licenses() -> None:
    assert len(SPECS) == len({s.name for s in SPECS.values()})
    for s in SPECS.values():
        assert s.role in ROLES, s.name
        assert s.license.strip(), s.name
        assert isinstance(s.commercial_ok, bool), s.name
        assert s.vram_estimate_gb >= 0, s.name
        assert set(s.profiles) <= {"default", "test", "optional"}, s.name
        assert s.adapter, s.name


def test_fallbacks_and_mocks_exist_and_match_roles() -> None:
    for s in SPECS.values():
        for ref in (s.fallback, s.mock):
            if ref:
                assert ref in SPECS, f"{s.name} -> {ref}"
                assert SPECS[ref].role == s.role, f"{s.name} -> {ref} role mismatch"
        if s.mock:
            assert SPECS[s.mock].env is None, f"mock {s.mock} must run in-process"


def test_fallback_chains_terminate() -> None:
    for s in SPECS.values():
        seen, cur = set(), s.name
        while cur:
            assert cur not in seen, f"fallback cycle at {cur}"
            seen.add(cur)
            cur = SPECS[cur].fallback


@pytest.mark.parametrize("env", ENVS)
def test_env_has_adapter_entrypoint_table_and_requirements(env: str) -> None:
    d = REPO / "adapters" / env
    assert (d / "main.py").exists() and (d / "table.py").exists() and (d / "__init__.py").exists()
    assert (REPO / "envs" / f"{env}.txt").exists()
    table = _table(env)
    for key, target in table.items():
        mod, _, cls = target.partition(":")
        f = REPO / Path(*mod.split(".")).with_suffix(".py")
        assert f.exists(), f"{env}:{key} -> missing module file {f}"
        assert re.search(rf"^class {cls}\b", f.read_text(encoding="utf-8"), re.M), f"{env}:{key} -> class {cls} not in {f}"


def test_every_remote_spec_has_an_adapter() -> None:
    tables = {env: _table(env) for env in ENVS}
    for s in REMOTE:
        assert s.adapter in tables[s.env], f"{s.name}: adapter '{s.adapter}' missing from adapters/{s.env}/table.py"


def test_local_specs_are_importable() -> None:
    for s in SPECS.values():
        if s.env is None:
            assert s.local_impl, s.name
            cls = registry.load_class(s.local_impl)
            assert hasattr(cls, "call"), s.name


def test_weight_specs() -> None:
    for s in SPECS.values():
        for w in s.weights:
            assert w.kind in WEIGHT_KINDS, f"{s.name}: {w.kind}"
            if w.kind in ("hf", "hf_file"):
                assert "/" in w.repo, s.name
            if w.kind == "env":
                assert w.post and w.subdir, f"{s.name}: env weights need post + subdir"
        if s.env is not None and s.weights == ():
            pytest.fail(f"remote backend {s.name} declares no weights (use WeightSpec('builtin'))")


def test_tts_specs_declare_caps() -> None:
    for s in SPECS.values():
        if s.role == "tts":
            assert "sr" in s.caps, s.name
            assert s.caps.get("stress_format", "plain") in ("plain", "plus"), s.name


STDLIB = set(getattr(sys, "stdlib_module_names", ())) | {"__future__"}


@pytest.mark.parametrize("path", [REPO / "adapters" / "common" / "rpc.py", *[REPO / "adapters" / e / "table.py" for e in ENVS], *[REPO / "adapters" / e / "main.py" for e in ENVS]])
def test_adapter_entry_modules_import_only_stdlib_at_top(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module]
        for n in names:
            top = n.split(".")[0]
            assert top in STDLIB or top == "adapters", f"{path.name} imports {n} at module level"


def test_adapter_tables_import_without_heavy_deps() -> None:
    for env in ENVS:
        mod = importlib.import_module(f"adapters.{env}.table")
        assert isinstance(mod.TABLE, dict) and mod.TABLE


def test_inject_tools() -> None:
    class Tools:
        paths = {"applio_dir": str(REPO)}

        def resolve(self, name: str):
            return REPO / "pyproject.toml" if name == "audiocpp" else None

    fish = SPECS["fish_s2"]
    assert inject_tools(fish, {}, Tools())["binary"].endswith("pyproject.toml")
    rvc = SPECS["rvc"]
    assert inject_tools(rvc, {}, Tools())["applio_dir"] == str(REPO)
    assert inject_tools(rvc, {}, None) == {}


def test_rpc_cli_params_prefers_named_backend(tmp_path: Path) -> None:
    from adapters.common.rpc import _cli_params

    p = _cli_params("whisper_tiny", "main", tmp_path, "cpu")
    assert p["name"] == "whisper_tiny" and p["model"] == "faster_whisper"
    p2 = _cli_params("faster_whisper", "main", tmp_path, "cpu")
    assert p2["model"] == "faster_whisper"


def test_models_selftest_reports_missing_pieces_cleanly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vndub import models
    from vndub.config import load_config

    monkeypatch.setenv("VNDUB_MODELS_DIR", str(tmp_path / "models"))
    cfg = load_config(REPO / "configs" / "pipeline.yaml")
    monkeypatch.setattr("vndub.backends.runner.env_exists", lambda env, root=None: True)
    res = models.selftest(cfg, "qwen3_tts")
    assert res["ok"] is False and res["error"] == "weights missing: run `vndub models pull qwen3_tts`"
    monkeypatch.setattr("vndub.backends.runner.env_exists", lambda env, root=None: False)
    res = models.selftest(cfg, "voxcpm2")
    assert res["ok"] is False and "env 'main' not created" in res["error"] and "make_envs" in res["hint"]
    assert models.selftest(cfg, "no_such_backend")["ok"] is False

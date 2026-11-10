"""Round 18: the adapters' import-level probe (no weights) and NaN-free responses.

The three late failures of 0.7.1 / 1.0.1 / 1.0.2 were vendor imports that broke after minutes of weight loading; the probe
runs the same imports and shims in seconds. A real env is used when this machine has it (``.venvs/funasr`` /
``.venvs/chatterbox``, CPU torch); everything else runs a stand-in implementation module in a child process.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from adapters.common import rpc
from adapters.common.rpc import ModelImpl, json_safe

REPO = Path(__file__).resolve().parent.parent

IMPL = '''
import math
from adapters.common.rpc import ModelImpl


class NanImpl(ModelImpl):
    probe_imports = ("json",)

    def op_score(self, item, **_):
        return {"mos": float("nan"), "peak": float("inf"), "track": [1.0, float("-inf")], "ok": 1.5}


class BrokenImpl(ModelImpl):
    probe_imports = ("module_that_is_not_installed_r18",)


class HidingImpl(ModelImpl):
    @classmethod
    def probe_checks(cls, params):
        raise TypeError("'NoneType' object is not callable")
'''

TABLE = "{'nan': 'r18impl:NanImpl', 'broken': 'r18impl:BrokenImpl', 'hiding': 'r18impl:HidingImpl'}"


def _strict(line: str) -> Any:
    def refuse(name: str) -> Any:
        raise ValueError(f"non-JSON constant {name}")

    return json.loads(line, parse_constant=refuse)


def _child(tmp_path: Path, code: str, *args: str, stdin: str = "") -> subprocess.CompletedProcess[str]:
    (tmp_path / "r18impl.py").write_text(IMPL, encoding="utf-8")
    prog = f"import sys; sys.path[:0] = [{str(REPO)!r}, {str(tmp_path)!r}]\nfrom adapters.common.rpc import lazy_table, main, serve\n{code}"
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, "-c", prog, *args], input=stdin, capture_output=True, text=True, encoding="utf-8", timeout=120, env=env)


def test_json_safe_replaces_every_non_finite_float() -> None:
    np = pytest.importorskip("numpy")
    data = {"a": float("nan"), "b": [1.0, float("inf"), {"c": -math.inf}], "d": (np.float32("inf"), np.float64(2.5)), "e": "NaN", "f": 3}
    out = json_safe(data)
    assert out == {"a": None, "b": [1.0, None, {"c": None}], "d": [None, 2.5], "e": "NaN", "f": 3}
    _strict(json.dumps(out))


def test_a_nan_result_reaches_the_core_as_null_through_a_real_adapter_process(tmp_path: Path) -> None:
    reqs = [
        {"jsonrpc": "2.0", "id": 1, "method": "load", "params": {"model": "nan", "name": "nan"}},
        {"jsonrpc": "2.0", "id": 2, "method": "run", "params": {"op": "score", "items": [{"id": "l1"}]}},
        {"jsonrpc": "2.0", "id": 3, "method": "probe", "params": {"model": "nan"}},
        {"jsonrpc": "2.0", "id": 4, "method": "probe", "params": {"model": "broken"}},
        {"jsonrpc": "2.0", "id": 5, "method": "shutdown"},
    ]
    r = _child(tmp_path, "serve(lazy_table(" + TABLE + "), 'testenv')", stdin="".join(json.dumps(q) + "\n" for q in reqs))
    lines = [l for l in r.stdout.splitlines() if l.strip()]
    msgs = {m["id"]: m for m in map(_strict, lines)}  # every line is strict JSON: no NaN / Infinity words
    assert msgs[2]["result"] == [{"id": "l1", "mos": None, "peak": None, "track": [1.0, None], "ok": 1.5}]
    assert msgs[3]["result"]["ok"] is True and msgs[3]["result"]["imports"] == ["json"]
    err = msgs[4]["error"]
    assert "import failed in env 'testenv'" in err["message"] and "module_that_is_not_installed_r18" in err["message"]
    hint = err["data"]["hint"]
    assert "Diagnostics" in hint and "scripts/make_envs.ps1 -Env testenv" in hint  # the app's way first, the command line kept


def test_the_probe_command_line_reports_each_adapter_and_fails_on_any_error(tmp_path: Path) -> None:
    r = _child(tmp_path, "main(lazy_table(" + TABLE + "), 'testenv')", "--probe")
    assert r.returncode == 1, r.stderr
    res = _strict(r.stdout)
    assert res["nan"]["ok"] is True
    assert res["broken"]["ok"] is False and "ModuleNotFoundError" in res["broken"]["error"]
    assert res["hiding"] == {"ok": False, "s": res["hiding"]["s"], "error": "TypeError: 'NoneType' object is not callable"}
    ok = _child(tmp_path, "main(lazy_table(" + TABLE + "), 'testenv')", "--probe", "nan")
    assert ok.returncode == 0 and set(_strict(ok.stdout)) == {"nan"}


def test_probe_runs_setup_imports_and_checks_in_that_order(monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []

    class Impl(ModelImpl):
        probe_imports = ("json",)

        @classmethod
        def probe_setup(cls, params: dict[str, Any]) -> None:
            order.append("setup")

        @classmethod
        def probe_checks(cls, params: dict[str, Any]) -> dict[str, Any] | None:
            order.append("checks")
            return {"extra": 1}

    real = rpc.importlib.import_module
    monkeypatch.setattr(rpc.importlib, "import_module", lambda m: (order.append(f"import {m}"), real(m))[1])
    res = Impl.probe({})
    assert order == ["setup", "import json", "checks"] and res["ok"] and res["extra"] == 1


def test_every_adapter_table_entry_has_a_probe() -> None:
    """The env tables load each implementation lazily; every one must answer ``probe`` (the default imports nothing)."""
    import importlib

    for env in ("main", "funasr", "chatterbox", "rvc", "seedvc", "f5", "gguf", "higgs"):
        table = importlib.import_module(f"adapters.{env}.table").TABLE
        for key, path in table.items():
            mod, _, name = path.partition(":")
            try:
                cls = getattr(importlib.import_module(mod), name)
            except ImportError:  # an implementation module that needs its env's packages at import time
                continue
            assert callable(getattr(cls, "probe", None)), (env, key)


def _venv_python(env: str) -> Path:
    return REPO / ".venvs" / env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


BLOCK_PKG_RESOURCES = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "pkg_resources" or name.startswith("pkg_resources."):
            raise ModuleNotFoundError("No module named 'pkg_resources'", name=name)
sys.meta_path.insert(0, Block())
"""


@pytest.mark.parametrize("env,key", [("chatterbox", "chatterbox_ml"), ("funasr", "cosyvoice3")])
def test_the_real_probe_passes_in_a_built_env(env: str, key: str) -> None:
    py = _venv_python(env)
    if not py.exists():
        pytest.skip(f".venvs/{env} is not built on this machine")
    r = subprocess.run([str(py), "-B", "-m", f"adapters.{env}.main", "--probe", key], cwd=str(REPO), capture_output=True, text=True, encoding="utf-8", timeout=600, env={**os.environ, "VNDUB_ENVS_DIR": str(REPO / ".venvs")})
    assert r.returncode == 0, r.stdout + r.stderr[-2000:]
    assert _strict(r.stdout)[key]["ok"] is True


def test_the_real_cosyvoice_probe_catches_the_round_13_env() -> None:
    """With pkg_resources unavailable and the stand-in disabled, the probe fails in the import the real load failed in."""
    py = _venv_python("funasr")
    if not py.exists():
        pytest.skip(".venvs/funasr is not built on this machine")
    code = BLOCK_PKG_RESOURCES + (
        "import adapters.funasr.cosyvoice_impl as m\n"
        "m.ensure_pkg_resources = lambda: False\n"
        "from adapters.common.rpc import run_probes, lazy_table\n"
        "import json\n"
        "print(json.dumps(run_probes(lazy_table({'cosyvoice3': 'adapters.funasr.cosyvoice_impl:CosyVoice3TTS'}), 'funasr', ['cosyvoice3'], __import__('pathlib').Path('.'))))\n"
    )
    r = subprocess.run([str(py), "-B", "-c", code], cwd=str(REPO), capture_output=True, text=True, encoding="utf-8", timeout=600, env={**os.environ, "VNDUB_ENVS_DIR": str(REPO / ".venvs")})
    res = json.loads(r.stdout.strip().splitlines()[-1])["cosyvoice3"]
    assert res["ok"] is False and "pkg_resources" in res["error"], res

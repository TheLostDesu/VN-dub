"""adapters/common/yamlcompat.py: ruamel.yaml 0.19 broke the old ``Loader`` classes HyperPyYAML loads
model configs with (``'Loader' object has no attribute 'max_depth'``); the shim gives them the attribute."""

from __future__ import annotations

import sys
import types

import pytest

from adapters.common.yamlcompat import ruamel_old_api_compat


def _fake_ruamel(monkeypatch, loader_module: types.ModuleType) -> None:
    ruamel = types.ModuleType("ruamel")
    ruamel_yaml = types.ModuleType("ruamel.yaml")
    ruamel.yaml = ruamel_yaml  # type: ignore[attr-defined]
    ruamel_yaml.loader = loader_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ruamel", ruamel)
    monkeypatch.setitem(sys.modules, "ruamel.yaml", ruamel_yaml)
    monkeypatch.setitem(sys.modules, "ruamel.yaml.loader", loader_module)


def test_old_loaders_get_max_depth_once(monkeypatch):
    mod = types.ModuleType("ruamel.yaml.loader")

    class Loader:  # ruamel.yaml 0.19: no max_depth on the old classes
        pass

    class RoundTripLoader:
        pass

    class SafeLoader:  # a class that has it (a future ruamel.yaml) is left alone
        max_depth = 7

    mod.Loader, mod.RoundTripLoader, mod.SafeLoader = Loader, RoundTripLoader, SafeLoader  # type: ignore[attr-defined]
    _fake_ruamel(monkeypatch, mod)

    assert ruamel_old_api_compat() is True
    assert Loader.max_depth == 0 and RoundTripLoader.max_depth == 0  # 0 = no limit, the YAML() default
    assert SafeLoader.max_depth == 7
    assert ruamel_old_api_compat() is False  # idempotent: nothing left to patch


def test_without_ruamel_is_a_noop(monkeypatch):
    monkeypatch.setitem(sys.modules, "ruamel", None)  # import raises ImportError
    assert ruamel_old_api_compat() is False


def test_real_hyperpyyaml_loads_nested_document():
    """In an env with the real libraries (funasr, main): a HyperPyYAML document parses after the shim."""
    ruamel_yaml = pytest.importorskip("ruamel.yaml")
    hyperpyyaml = pytest.importorskip("hyperpyyaml")
    ruamel_old_api_compat()
    doc = "a: {b: [1, 2, {c: 3}]}\nc: !ref <a[b]>\nd: !new:collections.OrderedDict\n    x: 1\ne: !tuple (1, 2)\n"
    h = hyperpyyaml.load_hyperpyyaml(doc)
    assert h["c"] == [1, 2, {"c": 3}] and dict(h["d"]) == {"x": 1} and h["e"] == (1, 2), ruamel_yaml.__version__

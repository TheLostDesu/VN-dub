"""adapters/common/pkg_resources_compat.py: a stand-in ``pkg_resources`` for an env whose setuptools has none.

uv venvs carry no setuptools and setuptools 82 removed ``pkg_resources``; lightning 2.2 (CosyVoice's Matcha-TTS submodule
imports it) runs ``__import__("pkg_resources").declare_namespace(...)`` at import, so CosyVoice 3 never loaded
(``ModuleNotFoundError: No module named 'pkg_resources'`` from ``cosyvoice.flow.flow_matching``). The stand-in is registered
only when the real module is missing and covers what those libraries call.
"""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
import types
from pathlib import Path

import pytest

from adapters.common import pkg_resources_compat as compat

REPO = Path(__file__).resolve().parent.parent
FUNASR_PY = REPO / ".venvs" / "funasr" / ("Scripts/python.exe" if sys.platform.startswith("win") else "bin/python")


def test_registers_the_stand_in_only_when_the_real_module_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pkg_resources", None)  # not imported yet
    monkeypatch.setattr(compat, "_find_spec", lambda name: None)  # and not installed
    assert compat.ensure_pkg_resources() is True
    import pkg_resources

    assert pkg_resources is compat
    assert compat.ensure_pkg_resources() is False  # idempotent


def test_a_present_module_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    real = types.ModuleType("pkg_resources")
    monkeypatch.setitem(sys.modules, "pkg_resources", real)
    assert compat.ensure_pkg_resources() is False
    assert sys.modules["pkg_resources"] is real
    # installed but not imported yet: the real one is imported later, nothing is registered
    monkeypatch.setitem(sys.modules, "pkg_resources", None)
    monkeypatch.setattr(compat, "_find_spec", lambda name: object())
    assert compat.ensure_pkg_resources() is False
    assert sys.modules["pkg_resources"] is None


def test_what_lightning_pyworld_and_jieba_call() -> None:
    assert compat.declare_namespace("lightning.fabric") is None  # lightning: a no-op on a regular package
    d = compat.get_distribution("pytest")  # pyworld: get_distribution(name).version
    assert d.version == importlib.metadata.version("pytest") and d.project_name.lower() == "pytest" and d.key == "pytest"
    assert d.parsed_version == compat.parse_version(d.version)
    with pytest.raises(compat.DistributionNotFound):
        compat.get_distribution("vndub-no-such-distribution")
    assert issubclass(compat.DistributionNotFound, ImportError)  # callers catch ImportError, as with the real one
    assert compat.parse_version("1.10.0") > compat.parse_version("1.9")
    names = [ep.name for ep in compat.iter_entry_points("console_scripts")]
    assert "pytest" in names and [ep.name for ep in compat.iter_entry_points("console_scripts", "pytest")] == ["pytest"]
    # jieba: resources relative to the folder of the named module (a plain module names its package folder)
    assert compat.resource_exists("vndub.config", "util.py") and compat.resource_isdir("vndub", "stages")
    assert compat.resource_exists("vndub", "stages\\synth.py")  # os.path.join form on Windows
    assert not compat.resource_exists("vndub", "no-such-file.yaml")
    with compat.resource_stream("vndub.config", "util.py") as fh:
        assert b"def repo_root" in fh.read()
    assert b"__version__" in compat.resource_string("vndub", "__init__.py")
    assert "synth.py" in compat.resource_listdir("vndub", "stages")
    assert Path(compat.resource_filename("vndub", "stages")).is_dir()


def test_anything_else_names_the_way_out() -> None:
    with pytest.raises(AttributeError, match="setuptools 80"):
        compat.working_set  # noqa: B018
    with pytest.raises(AttributeError):
        compat.__wrapped__  # noqa: B018  # dunder probes (inspect, copy) stay plain AttributeErrors


@pytest.mark.slow
@pytest.mark.skipif(not FUNASR_PY.exists(), reason="no funasr env on this machine")
def test_lightning_and_matcha_import_with_the_stand_in() -> None:
    """The real lightning 2.2 (and Matcha-TTS when the CosyVoice checkout is there) import over the stand-in."""
    matcha = REPO / ".venvs" / "funasr" / "src" / "CosyVoice" / "third_party" / "Matcha-TTS"
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])\n"
        "from adapters.common import pkg_resources_compat as c; sys.modules['pkg_resources'] = c\n"
        "import lightning; print('lightning', lightning.__version__)\n"
        "import os\n"
        "if os.path.isdir(sys.argv[2]):\n"
        "    import matcha.utils; print('matcha ok')\n"
    )
    r = subprocess.run([str(FUNASR_PY), "-I", "-c", code, str(REPO), str(matcha)], capture_output=True, text=True, timeout=280)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "lightning 2." in r.stdout
    if matcha.is_dir():
        assert "matcha ok" in r.stdout

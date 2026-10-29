"""A stand-in for ``pkg_resources`` in an environment whose setuptools no longer ships it.

uv creates virtual environments without setuptools, and setuptools 82 (February 2026) removed ``pkg_resources``
altogether; an environment built either way has no ``pkg_resources``. Some model repositories still import it at
module level: ``lightning`` 2.2 runs ``__import__("pkg_resources").declare_namespace(__name__)`` in
``lightning.fabric`` and ``lightning.pytorch``, and Matcha-TTS (the CosyVoice submodule) imports lightning from
``matcha.utils``, so ``cosyvoice.flow.flow_matching`` cannot be imported and CosyVoice 3 never loads
(``ModuleNotFoundError: No module named 'pkg_resources'`` while HyperPyYAML builds the model). ``pyworld``
(``get_distribution("pyworld").version``) and jieba (``resource_stream``) are other users.

:func:`ensure_pkg_resources` registers this module as ``pkg_resources`` when the real one cannot be imported. It
covers the calls those libraries make, on top of ``importlib.metadata`` / ``packaging``: ``declare_namespace``
(a no-op: regular packages need none), ``get_distribution(name).version`` and ``DistributionNotFound``,
``parse_version``, ``iter_entry_points`` and the ``resource_*`` helpers (relative to the folder of the named
module, as pkg_resources' ``DefaultProvider`` resolves them). Any other attribute raises ``AttributeError``
naming it, with the way out (setuptools 80.x in the env).

The env pins (``envs/funasr+cosyvoice.txt``) now ask for setuptools 80.10.2, the last line that ships
``pkg_resources``, so a freshly built environment uses the real module and this one stays unused.
"""

from __future__ import annotations

import importlib
import importlib.metadata as _metadata
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Iterator

_find_spec = importlib.util.find_spec  # tests replace it


def ensure_pkg_resources() -> bool:
    """Make ``import pkg_resources`` succeed: the real module when the env has one, else this stand-in.

    Returns True when the stand-in was registered. Call it before importing the model package.
    """
    if sys.modules.get("pkg_resources") is not None:
        return False
    try:
        if _find_spec("pkg_resources") is not None:
            return False
    except (ImportError, ValueError):  # a broken half-installed setuptools: the stand-in is still better than nothing
        pass
    sys.modules["pkg_resources"] = sys.modules[__name__]
    return True


# ----------------------------------------------------------------------------
# distributions
# ----------------------------------------------------------------------------


class ResolutionError(Exception):
    """Base of the lookup errors (as in pkg_resources)."""


class DistributionNotFound(ResolutionError, ImportError):
    """``get_distribution`` of a project that is not installed (an ImportError too, as in pkg_resources)."""


class Distribution:
    """The part of ``pkg_resources.Distribution`` libraries read: ``version``, ``project_name``, ``key``, ``location``."""

    def __init__(self, project_name: str, version: str, location: str) -> None:
        self.project_name = project_name
        self.key = project_name.lower()
        self.version = version
        self.location = location

    @property
    def parsed_version(self) -> Any:
        return parse_version(self.version)

    def __repr__(self) -> str:
        return f"{self.project_name} {self.version}"

    def __str__(self) -> str:
        return f"{self.project_name} {self.version}"


def get_distribution(name: Any) -> Distribution:
    """``importlib.metadata`` lookup of ``name`` (a project name or anything with ``project_name``)."""
    project = str(getattr(name, "project_name", name)).strip()
    try:
        dist = _metadata.distribution(project)
    except _metadata.PackageNotFoundError as e:
        raise DistributionNotFound(f"The '{project}' distribution was not found and is required by the application") from e
    meta_name = dist.metadata["Name"] if dist.metadata else None
    try:
        location = str(dist.locate_file(""))
    except Exception:  # an exotic finder: the location is informational
        location = ""
    return Distribution(str(meta_name or project), dist.version, location)


def parse_version(v: Any) -> Any:
    """``packaging.version.Version`` (what pkg_resources returns since setuptools 66); a tuple when packaging is absent."""
    try:
        from packaging.version import Version
    except ImportError:
        return tuple(int(x) if x.isdigit() else x for x in str(v).replace("-", ".").split("."))
    return Version(str(v))


def iter_entry_points(group: str, name: str | None = None) -> Iterator[Any]:
    """The entry points of ``group`` (every one, or the one called ``name``); each has ``.name`` and ``.load()``."""
    eps = _metadata.entry_points()
    sel = eps.select(group=group) if hasattr(eps, "select") else eps.get(group, [])  # type: ignore[union-attr]
    for ep in sel:
        if name is None or ep.name == name:
            yield ep


def declare_namespace(name: str) -> None:
    """A no-op: ``lightning`` calls it on regular packages, which need no namespace registration."""
    return None


# ----------------------------------------------------------------------------
# resources: relative to the folder that holds the named module (pkg_resources' DefaultProvider)
# ----------------------------------------------------------------------------


def _resource_path(package_or_requirement: Any, resource_name: str) -> Path:
    mod_name = package_or_requirement if isinstance(package_or_requirement, str) else getattr(package_or_requirement, "__name__", str(package_or_requirement))
    module = importlib.import_module(mod_name)
    base = Path(getattr(module, "__file__", "") or "").parent
    rel = str(resource_name).replace("\\", "/")
    return base.joinpath(*[part for part in rel.split("/") if part and part != "."])


def resource_filename(package_or_requirement: Any, resource_name: str) -> str:
    return str(_resource_path(package_or_requirement, resource_name))


def resource_exists(package_or_requirement: Any, resource_name: str) -> bool:
    return _resource_path(package_or_requirement, resource_name).exists()


def resource_isdir(package_or_requirement: Any, resource_name: str) -> bool:
    return _resource_path(package_or_requirement, resource_name).is_dir()


def resource_listdir(package_or_requirement: Any, resource_name: str) -> list[str]:
    return os.listdir(_resource_path(package_or_requirement, resource_name))


def resource_stream(package_or_requirement: Any, resource_name: str):
    return open(_resource_path(package_or_requirement, resource_name), "rb")


def resource_string(package_or_requirement: Any, resource_name: str) -> bytes:
    return _resource_path(package_or_requirement, resource_name).read_bytes()


def __getattr__(attr: str) -> Any:
    if attr.startswith("__"):
        raise AttributeError(attr)
    raise AttributeError(
        f"pkg_resources.{attr} is not provided by this stand-in (the environment's setuptools has no pkg_resources): "
        "install setuptools 80.x into the environment (envs/funasr+cosyvoice.txt pins it) to get the real module"
    )

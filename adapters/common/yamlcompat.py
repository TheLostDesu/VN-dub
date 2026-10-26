"""Compatibility shims for the YAML libraries the model repositories load their configs with."""

from __future__ import annotations

OLD_LOADERS = ("BaseLoader", "SafeLoader", "Loader", "RoundTripLoader")


def ruamel_old_api_compat() -> bool:
    """Make ruamel.yaml >= 0.19 usable through its old ``Loader`` classes.

    HyperPyYAML (CosyVoice, speechbrain) parses hyperparameter files with
    ``yaml.load(stream, Loader=ruamel.yaml.Loader)``. ruamel.yaml 0.19 added a nesting limit that
    its composer reads from ``self.loader.max_depth``: the new-API ``YAML()`` object sets it, the old
    ``Loader`` classes never do, so every document fails with
    ``AttributeError: 'Loader' object has no attribute 'max_depth'`` before the model is even built.

    The env pins ask for ruamel.yaml 0.18 (HyperPyYAML 1.2.3 requires ``< 0.19``); an environment
    built before that pin keeps 0.19.1, and this gives its loader classes the attribute
    (0 = no limit, the ``YAML()`` default). Call it before importing the model package.

    Returns True when a class was patched. A missing ruamel.yaml is not an error.
    """
    try:
        from ruamel.yaml import loader as ruamel_loader
    except ImportError:
        return False
    patched = False
    for name in OLD_LOADERS:
        cls = getattr(ruamel_loader, name, None)
        if cls is not None and not hasattr(cls, "max_depth"):
            cls.max_depth = 0
            patched = True
    return patched

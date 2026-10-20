"""Engine plugins. ``get_engine(name)`` -> engine instance; ``ENGINES`` lists all names."""

from __future__ import annotations

import importlib

from vndub.engines.base import BaseEngine

# engine name -> module:class
ENGINES: dict[str, str] = {
    "renpy": "vndub.engines.renpy:RenpyEngine",
    "kirikiri": "vndub.engines.kirikiri:KirikiriEngine",
    "nscripter": "vndub.engines.nscripter:NScripterEngine",
    "siglus": "vndub.engines.siglus:SiglusEngine",
    "bgi": "vndub.engines.bgi:BgiEngine",
    "catsystem2": "vndub.engines.catsystem2:CatSystem2Engine",
    "artemis": "vndub.engines.artemis:ArtemisEngine",
    "yuris": "vndub.engines.yuris:YurisEngine",
    "majiro": "vndub.engines.majiro:MajiroEngine",
    "unity": "vndub.engines.unity:UnityEngine",
    "tyrano": "vndub.engines.tyrano:TyranoEngine",
    "generic": "vndub.engines.generic_prefix:GenericPrefixEngine",
}


def get_engine(name: str) -> BaseEngine:
    if name not in ENGINES:
        raise KeyError(f"unknown engine {name!r}; known: {', '.join(ENGINES)}")
    mod, _, cls = ENGINES[name].partition(":")
    return getattr(importlib.import_module(mod), cls)()


def all_engines() -> list[BaseEngine]:
    return [get_engine(n) for n in ENGINES]

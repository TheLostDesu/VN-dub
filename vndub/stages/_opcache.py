"""Per-item result cache for model ops (used by link / analyze / translate).

:class:`StageCache` stores only keys; model outputs (transcripts, emotion
probabilities, translations ...) are kept next to it in
``work/cache/results/<name>.json`` so a rerun can reuse them. An item is
fresh when its key (backend + options + input content hash) is unchanged and
its declared output files still exist. ``--force`` clears the keys.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from vndub.manifest.io import read_json, write_json

if TYPE_CHECKING:
    from vndub.stages.common import StageContext


class OpCache:
    def __init__(self, ctx: "StageContext", name: str, version: str = "1") -> None:
        self.cache = ctx.cache(name, version)
        self.path = ctx.paths.cache / "results" / f"{name}.json"
        self.values: dict[str, Any] = {} if ctx.force else (read_json(self.path, {}) or {})
        self._dirty = 0
        self._lock = threading.RLock()  # translate drives several llama-server slots from threads

    def key(self, *inputs: Any, files: Iterable[str | Path] = ()) -> str:
        with self._lock:
            return self.cache.key(*inputs, files=files)

    def get(self, item_id: str, key: str, outputs: Iterable[str | Path] = ()) -> Any:
        with self._lock:
            if item_id in self.values and self.cache.fresh(item_id, key, outputs):
                return self.values[item_id]
            return None

    def put(self, item_id: str, key: str, value: Any) -> None:
        with self._lock:
            self.values[item_id] = value
            self.cache.put(item_id, key)
            self._dirty += 1
            if self._dirty >= 50:
                self.save()

    def save(self) -> None:
        with self._lock:
            if self._dirty:
                write_json(self.path, self.values, indent=False)
                self._dirty = 0
            self.cache.save()

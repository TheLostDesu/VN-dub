"""Content-hash stage cache.

Each stage keeps ``work/cache/<stage>.json``: ``item_id -> key`` where
``key = sha256(stage version, relevant config hash, item inputs)``. An item is
recomputed only when its key changed or one of its declared outputs is missing,
which makes every stage resumable (a crash loses at most the items in flight)
and idempotent (rerunning with the same inputs does nothing).

File inputs are hashed by content; a memo keyed by (size, mtime) avoids
re-reading unchanged multi-GB voice banks on every run.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Iterable

from vndub.manifest.io import read_json, write_json
from vndub.util import hash_file, hash_obj


class FileHasher:
    def __init__(self, memo_path: Path) -> None:
        self.memo_path = memo_path
        self._memo: dict[str, list[Any]] = read_json(memo_path, {}) or {}
        self._dirty = False
        self._lock = threading.Lock()

    def __call__(self, path: str | Path) -> str:
        p = Path(path)
        try:
            st = p.stat()
        except FileNotFoundError:
            return "missing"
        k = str(p.resolve())
        with self._lock:
            m = self._memo.get(k)
            if m and m[0] == st.st_size and m[1] == st.st_mtime_ns:
                return m[2]
        h = hash_file(p)
        with self._lock:
            self._memo[k] = [st.st_size, st.st_mtime_ns, h]
            self._dirty = True
        return h

    def save(self) -> None:
        with self._lock:
            if self._dirty:
                write_json(self.memo_path, self._memo, indent=False)
                self._dirty = False


class StageCache:
    def __init__(self, cache_dir: Path, stage: str, version: str, config_hash: str, hasher: FileHasher | None = None) -> None:
        self.path = cache_dir / f"{stage}.json"
        self.stage = stage
        self.version = version
        self.config_hash = config_hash
        self.hasher = hasher or FileHasher(cache_dir / "filehash.json")
        self._items: dict[str, str] = read_json(self.path, {}) or {}
        self._dirty = 0
        self._lock = threading.Lock()

    def key(self, *inputs: Any, files: Iterable[str | Path] = ()) -> str:
        """Cache key from JSON-able ``inputs`` plus content hashes of ``files``."""
        fh = [self.hasher(f) for f in files]
        return hash_obj([self.stage, self.version, self.config_hash, list(inputs), fh])

    def fresh(self, item_id: str, key: str, outputs: Iterable[str | Path] = ()) -> bool:
        with self._lock:
            if self._items.get(item_id) != key:
                return False
        return all(Path(o).exists() for o in outputs)

    def put(self, item_id: str, key: str) -> None:
        with self._lock:
            self._items[item_id] = key
            self._dirty += 1
            dirty = self._dirty
        if dirty >= 50:  # checkpoint regularly so a crash loses little
            self.save()

    def drop(self, item_id: str) -> None:
        with self._lock:
            if self._items.pop(item_id, None) is not None:
                self._dirty += 1

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._dirty += 1

    def __contains__(self, item_id: str) -> bool:
        return item_id in self._items

    def save(self) -> None:
        with self._lock:
            if self._dirty:
                write_json(self.path, self._items, indent=False)
                self._dirty = 0
        self.hasher.save()

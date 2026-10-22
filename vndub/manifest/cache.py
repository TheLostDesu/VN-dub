"""Content-hash stage cache.

Each stage keeps ``work/cache/<stage>.json``: ``item_id -> key`` where
``key = sha256(stage version, relevant config hash, item inputs)``. An item is
recomputed only when its key changed or one of its declared outputs is missing,
which makes every stage resumable (a crash loses at most the items in flight)
and idempotent (rerunning with the same inputs does nothing).

File inputs are hashed by content; a memo keyed by (size, mtime) avoids
re-reading unchanged multi-GB voice banks on every run.

Durability: the item map is written every ``CHECKPOINT_ITEMS`` new entries *and* when the last write
is older than ``CHECKPOINT_S`` seconds, so a hard kill (power loss, a force-closed app) loses seconds
of work, not the 49 entries below the count threshold; ``save()`` writes nothing when nothing changed.

Config upgrades: ``rekey()`` switches a cache to a new config hash and remembers the old one, so
entries stored under the previous hash still count as fresh (and are migrated to the new key on
the first lookup) -- used when a config section stops being part of the hash (the ``*_dir`` paths
of ``project``) and an existing project must not be regenerated for it.
"""

from __future__ import annotations

import threading
import time
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


CHECKPOINT_ITEMS = 50  # write the item map after this many new entries ...
CHECKPOINT_S = 10.0  # ... or when the last write is older than this (seconds)


class StageCache:
    def __init__(
        self, cache_dir: Path, stage: str, version: str, config_hash: str, hasher: FileHasher | None = None, legacy_config_hashes: Iterable[str] = (),
    ) -> None:
        self.path = cache_dir / f"{stage}.json"
        self.stage = stage
        self.version = version
        self.config_hash = config_hash
        self.legacy_config_hashes: list[str] = [h for h in legacy_config_hashes if h and h != config_hash]
        self.hasher = hasher or FileHasher(cache_dir / "filehash.json")
        self._items: dict[str, str] = read_json(self.path, {}) or {}
        self._dirty = 0
        self._lock = threading.Lock()
        self._clock = time.monotonic  # replaced by tests
        self._last_save = self._clock()
        self._legacy: dict[str, list[str]] = {}  # new key -> the keys the same inputs had under the legacy config hashes

    def rekey(self, config_hash: str) -> None:
        """Use ``config_hash`` from now on and keep accepting entries stored under the current one."""
        if config_hash == self.config_hash:
            return
        self.legacy_config_hashes = [self.config_hash, *[h for h in self.legacy_config_hashes if h != config_hash]]
        self.config_hash = config_hash

    def key(self, *inputs: Any, files: Iterable[str | Path] = ()) -> str:
        """Cache key from JSON-able ``inputs`` plus content hashes of ``files``."""
        fh = [self.hasher(f) for f in files]
        key = hash_obj([self.stage, self.version, self.config_hash, list(inputs), fh])
        if self.legacy_config_hashes:
            olds = [hash_obj([self.stage, self.version, h, list(inputs), fh]) for h in self.legacy_config_hashes]
            with self._lock:
                self._legacy[key] = olds
        return key

    def fresh(self, item_id: str, key: str, outputs: Iterable[str | Path] = ()) -> bool:
        """Is ``item_id`` stored under ``key`` (or, after :meth:`rekey`, under the legacy key of the same inputs)
        with all ``outputs`` present? A legacy entry is migrated to ``key``."""
        migrate = False
        with self._lock:
            cur = self._items.get(item_id)
            if cur != key:
                if cur is None or cur not in self._legacy.get(key, ()):
                    return False
                migrate = True
        if not all(Path(o).exists() for o in outputs):
            return False
        if migrate:
            with self._lock:
                self._items[item_id] = key
                self._dirty += 1
        return True

    def put(self, item_id: str, key: str) -> None:
        with self._lock:
            self._items[item_id] = key
            self._dirty += 1
            due = self._dirty >= CHECKPOINT_ITEMS or (self._clock() - self._last_save) >= CHECKPOINT_S
        if due:  # checkpoint regularly so a crash loses little
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
            self._last_save = self._clock()
        self.hasher.save()

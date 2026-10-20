"""Small shared helpers: hashing, slugs, timestamps, logging."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import orjson


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _default(o: Any) -> Any:
    if isinstance(o, Path):
        return o.as_posix()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if hasattr(o, "model_dump"):
        return o.model_dump(mode="json")
    raise TypeError(f"cannot hash {type(o)!r}")


def hash_obj(obj: Any) -> str:
    """Stable sha256 of a JSON-able object (sorted keys)."""
    return hash_bytes(orjson.dumps(obj, option=orjson.OPT_SORT_KEYS | orjson.OPT_SERIALIZE_NUMPY, default=_default))


def hash_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def file_fingerprint(path: str | Path) -> str:
    """Cheap fingerprint (size + mtime). Use hash_file when content identity matters."""
    st = os.stat(path)
    return f"{st.st_size}:{st.st_mtime_ns}"


_SLUG_RE = re.compile(r"[^0-9a-zA-Z]+")


def slugify(name: str, fallback: str = "x") -> str:
    """ASCII slug for ids. Non-latin names fall back to a short hash.

    ``寧々`` -> ``u_<hash6>``; ``Nene Ayachi`` -> ``nene_ayachi``.
    """
    norm = unicodedata.normalize("NFKC", name).strip()
    ascii_part = _SLUG_RE.sub("_", norm.encode("ascii", "ignore").decode()).strip("_").lower()
    if ascii_part and len(ascii_part) >= max(1, len(norm) // 2):
        return ascii_part
    if not norm:
        return fallback
    return "u_" + hashlib.sha1(norm.encode("utf-8")).hexdigest()[:6]


def chunked(items: list[Any], n: int) -> Iterable[list[Any]]:
    for i in range(0, len(items), max(1, n)):
        yield items[i : i + n]


class Timer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0


_LOG_FMT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def get_logger(name: str, log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Logger that writes to ``log_file`` (if given) and to stderr (warnings and up)."""
    logger = logging.getLogger(f"vndub.{name}")
    logger.setLevel(level)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_LOG_FMT))
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    sh.setLevel(logging.WARNING)
    logger.addHandler(sh)
    return logger


def is_windows() -> bool:
    return sys.platform.startswith("win")


def repo_root() -> Path:
    """Root of the vndub checkout (parent of the ``vndub`` package)."""
    return Path(__file__).resolve().parent.parent

"""Resumable downloads with sha256 verification, shared by the tool installer and the model manager.

* ``fetch`` downloads ``http(s)://`` or ``file://`` URLs into ``<dest>.part`` and renames on success.
  An interrupted download (connection reset, cancelled job, app closed) resumes from the ``.part`` file
  with an HTTP ``Range`` request; a server that ignores ``Range`` makes it start over.
* The checksum is computed over the whole finished file, so a resumed download is verified the same way
  as a fresh one. A mismatch deletes the file and raises ``AppError("checksum_mismatch")``.
* Unreachable server / 404 / DNS failure -> ``AppError("download_unreachable")`` with params
  ``{url, filename, target_dir}`` and action ``show_manual``: the UI shows the manual download card.
* No telemetry, no redirects to anything but what the server answers; nothing is sent besides the request.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from vndub_app.errors import AppError

log = logging.getLogger("vndub_app.downloads")

CHUNK = 1 << 20
USER_AGENT = "VNDub-installer/1.0 (+local desktop app)"

ProgressCb = Callable[[int, int], None]  # (bytes_done, bytes_total or 0 when unknown)


@dataclass
class FetchResult:
    path: Path
    size: int
    sha256: str | None  # hex digest when it was computed (always when ``expected_sha256`` was given)
    verified: bool      # True when a sha256 or a size was actually checked against an expectation
    resumed_from: int = 0
    skipped: bool = False  # the file was already there and valid


class _Transient(Exception):
    """A failure worth retrying (reset, timeout, 5xx)."""


def sha256_file(path: Path, progress: ProgressCb | None = None, chunk: int = CHUNK) -> str:
    h = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            done += len(b)
            if progress:
                progress(done, total)
    return h.hexdigest()


def _unreachable(url: str, filename: str, target_dir: Path | str | None, exc: BaseException | str) -> AppError:
    return AppError(
        "download_unreachable",
        params={"url": url, "filename": filename, "target_dir": str(target_dir) if target_dir else ""},
        action="show_manual",
        details=f"{type(exc).__name__}: {exc}" if isinstance(exc, BaseException) else str(exc),
        status=502,
    )


def _mismatch(filename: str, expected: str, actual: str) -> AppError:
    return AppError(
        "checksum_mismatch",
        params={"filename": filename, "expected": expected, "actual": actual},
        action="retry",
        details=f"sha256 of {filename}: expected {expected}, got {actual}",
        status=422,
    )


def _read_meta(meta: Path) -> dict:
    try:
        return json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _open_http(url: str, offset: int, timeout: float, extra: dict[str, str] | None = None):
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity", **(extra or {})}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    req = urllib.request.Request(url, headers=headers)
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code == 416:
            return e  # range not satisfiable: handled by the caller (HTTPError is a response too)
        if e.code in (408, 425, 429) or e.code >= 500:
            raise _Transient(f"HTTP {e.code}") from e
        raise


def _parse_content_range(value: str | None) -> tuple[int | None, int | None]:
    """``bytes 100-199/1000`` -> (100, 1000); ``bytes */1000`` -> (None, 1000)."""
    if not value or not value.lower().startswith("bytes "):
        return None, None
    spec = value[6:].strip()
    rng, _, total = spec.partition("/")
    start = None
    if rng and rng != "*":
        try:
            start = int(rng.split("-", 1)[0])
        except ValueError:
            start = None
    try:
        tot = int(total) if total and total != "*" else None
    except ValueError:
        tot = None
    return start, tot


def fetch(
    url: str,
    dest: Path,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    progress: ProgressCb | None = None,
    checkpoint: Callable[[], None] | None = None,
    target_dir: Path | str | None = None,
    timeout: float = 30.0,
    retries: int = 3,
    chunk: int = CHUNK,
    trust_existing: bool = True,
    identity: str | None = None,
    headers: dict[str, str] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> FetchResult:
    """Download ``url`` to ``dest`` (resumable). See the module docstring for the error contract.

    ``identity`` names *what* is downloaded when the URL is not stable (a Hugging Face file has a new pre-signed CDN
    link every time it is resolved): a ``.part`` file is continued only when it was started with the same identity
    (default: the URL itself). ``headers`` are extra request headers (e.g. a Hugging Face token for a gated file).
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    filename = dest.name
    target_dir = target_dir if target_dir is not None else dest.parent
    exp_sha = (expected_sha256 or "").lower() or None

    # a valid file that is already there (e.g. the user saved it manually into the target folder)
    if dest.exists():
        ok, digest = _existing_ok(dest, exp_sha, expected_size, trust_existing, progress)
        if ok:
            return FetchResult(dest, dest.stat().st_size, digest, bool(exp_sha or expected_size), skipped=True)
        dest.unlink()

    part = dest.with_name(dest.name + ".part")
    meta = dest.with_name(dest.name + ".part.json")
    resumed_from = 0
    last_exc: BaseException | None = None
    parsed = urllib.parse.urlparse(url)
    ident = identity or url
    if parsed.scheme not in ("http", "https", "file"):
        raise _unreachable(url, filename, target_dir, f"unsupported URL scheme {parsed.scheme!r}")

    started: dict[str, int] = {}  # offset the first response body started at (an attempt may fail mid-way)
    for attempt in range(retries + 1):
        if checkpoint:
            checkpoint()
        offset = part.stat().st_size if part.exists() else 0
        if offset and _read_meta(meta).get("url") not in (None, ident):
            part.unlink()
            offset = 0
        try:
            total, start = _stream(url, part, offset, expected_size, progress, checkpoint, timeout, chunk, parsed, ident, headers, started)
            resumed_from = started.get("start", start)
            size = part.stat().st_size
            if total and size != total:
                raise _Transient(f"short read: {size} of {total} bytes")
            if expected_size and size != expected_size:
                # the server delivered a complete but different file
                part.unlink(missing_ok=True)
                meta.unlink(missing_ok=True)
                raise _mismatch(filename, f"size:{expected_size}", f"size:{size}")
            last_exc = None
            break
        except _Transient as e:
            last_exc = e
        except urllib.error.HTTPError as e:
            raise _unreachable(url, filename, target_dir, e) from e
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as e:
            last_exc = e
        except OSError as e:
            if isinstance(e, (FileNotFoundError, PermissionError)) and parsed.scheme == "file":
                raise _unreachable(url, filename, target_dir, e) from e
            last_exc = e
        if attempt < retries:
            log.info("download %s: attempt %d failed (%s: %s), retrying", filename, attempt + 1, type(last_exc).__name__, last_exc)
            sleep(min(8.0, 0.5 * (2**attempt)))
    if last_exc is not None:
        raise _unreachable(url, filename, target_dir, last_exc) from last_exc

    digest = None
    verified = bool(expected_size)
    if exp_sha:
        digest = sha256_file(part)
        if digest != exp_sha:
            part.unlink(missing_ok=True)
            meta.unlink(missing_ok=True)
            raise _mismatch(filename, exp_sha, digest)
        verified = True
    os.replace(part, dest)
    meta.unlink(missing_ok=True)
    return FetchResult(dest, dest.stat().st_size, digest, verified, resumed_from=resumed_from)


def _existing_ok(dest: Path, sha: str | None, size: int | None, trust: bool, progress: ProgressCb | None) -> tuple[bool, str | None]:
    if sha:
        digest = sha256_file(dest, progress)
        return digest == sha, digest
    if size:
        return dest.stat().st_size == size, None
    return trust, None


def _stream(
    url: str,
    part: Path,
    offset: int,
    expected_size: int | None,
    progress: ProgressCb | None,
    checkpoint: Callable[[], None] | None,
    timeout: float,
    chunk: int,
    parsed: urllib.parse.ParseResult,
    ident: str,
    headers: dict[str, str] | None = None,
    started: dict[str, int] | None = None,
) -> tuple[int, int]:
    """Append the (remaining) body to ``part``.

    Returns ``(full size the server announced or 0, offset the body actually started at)``; the first call
    that gets a body also stores that offset in ``started["start"]``.
    """
    started = started if started is not None else {}
    meta = part.with_name(part.name[: -len(".part")] + ".part.json")
    if parsed.scheme == "file":
        src = Path(urllib.request.url2pathname(parsed.path))
        if parsed.netloc and os.name == "nt":  # file://C:/x or file://host/share
            src = Path(urllib.request.url2pathname(f"//{parsed.netloc}{parsed.path}"))
        total = src.stat().st_size
        if offset > total:
            part.unlink(missing_ok=True)
            offset = 0
        _write_meta(meta, ident, total)
        started.setdefault("start", offset)
        with open(src, "rb") as r, open(part, "ab" if offset else "wb") as w:
            r.seek(offset)
            done = offset
            _pump(r, w, done, total, progress, checkpoint, chunk)
        return total, offset

    resp = _open_http(url, offset, timeout, headers)
    try:
        status = getattr(resp, "status", None) or getattr(resp, "code", None)
        total = 0
        if status == 416:
            _, tot = _parse_content_range(resp.headers.get("Content-Range"))
            if tot is not None and offset == tot:
                return tot, offset  # the part file was already complete
            part.unlink(missing_ok=True)
            raise _Transient("range not satisfiable; restarting")
        if status == 206:
            start, tot = _parse_content_range(resp.headers.get("Content-Range"))
            if start is not None and start != offset:
                part.unlink(missing_ok=True)
                raise _Transient("server resumed at a different offset; restarting")
            total = tot or (offset + int(resp.headers.get("Content-Length") or 0))
            mode = "ab"
        else:
            if offset:
                offset = 0  # the server ignored Range: start over
            total = int(resp.headers.get("Content-Length") or 0) or (expected_size or 0)
            mode = "wb"
        _write_meta(meta, ident, total)
        started.setdefault("start", offset)
        with open(part, mode) as w:
            _pump(resp, w, offset, total, progress, checkpoint, chunk)
        return total, offset
    finally:
        try:
            resp.close()
        except Exception:
            pass


def _write_meta(meta: Path, url: str, total: int) -> None:
    try:
        meta.write_text(json.dumps({"url": url, "total": total}), encoding="utf-8")
    except OSError:
        pass


def _pump(src, dst, done: int, total: int, progress: ProgressCb | None, checkpoint: Callable[[], None] | None, chunk: int) -> None:
    if progress:
        progress(done, total)
    while True:
        b = src.read(chunk)
        if not b:
            break
        dst.write(b)
        done += len(b)
        if progress:
            progress(done, total)
        if checkpoint:
            checkpoint()


def free_space_gb(path: Path) -> float:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / 1e9

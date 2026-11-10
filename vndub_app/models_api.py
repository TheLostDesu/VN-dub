"""Model manager backend: table, presets, resumable downloads and self-tests.

``GET  /api/models``           ``{rows, presets, preset_info, disk}`` (table of ``vndub models status``)
``POST /api/models/download``  ``{names}`` -> Job (per-file progress in ``job.result.files``, resumable)
``POST /api/models/check``     ``{names}`` -> Job (``vndub.models.selftest``; peak VRAM stored for the table)

The router carries full ``/api/...`` paths; include it without a prefix.

Row: ``{name, role, env, size_gb, disk_gb, vram_gb, license, commercial_ok, present, usable,
reason, reason_params, reason_details, presets: [min|recommended|all], selftest}``. ``reason`` is a code
(``license | platform | env_missing | weights_missing`` or ``""`` when usable); the UI shows
``models.reason.<code>``; ``reason_details`` is the technical text. Names/purposes are looked up as
``models.<name>.name`` / ``models.<name>.purpose``. ``selftest`` is ``{ok, peak_vram_gb, at, device,
code, details}`` or ``null``.

Presets: ``recommended`` = the ``default`` profile of the registry; ``all`` = every real backend (default
and optional); ``min`` = :data:`PRESET_MIN` (one TTS family for casting and synthesis, one translation LLM,
the cheapest analysis and scoring models), see docs/decisions/O.md.
"""

from __future__ import annotations

import fnmatch
import io
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Protocol

from fastapi import APIRouter
from pydantic import BaseModel

from vndub.backends import registry
from vndub.backends.base import BackendSpec, BackendUnavailable, WeightSpec
from vndub.backends.weights import dir_size_gb, is_present, local_dir
from vndub_app import appdirs, downloads
from vndub_app.errors import AppError
from vndub_app.jobs import JobContext

log = logging.getLogger("vndub_app.models")

router = APIRouter()

# one TTS family (Base for cloning/synthesis + VoiceDesign for casting), the smallest translation LLM, the
# cheapest speaker-emotion model for analysis, and the scoring models (MOS, speaker similarity, CER ASR)
PRESET_MIN = (
    "qwen3_tts",
    "qwen3_tts_voicedesign",
    "hy_mt15_7b",
    "emotion2vec_plus_base",
    "utmos22",
    "ecapa",
    "gigaam_v3",
)
PRESETS = ("min", "recommended", "all")
_TEST_ONLY = ("test",)
_REASON_PREFIXES = (
    ("license ", "license"),
    ("not supported on", "platform"),
    ("env '", "env_missing"),
    ("weights missing", "weights_missing"),
)


# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------


def build_cfg():
    """A ``PipelineConfig`` for model operations: app profile + data-folder models dir + installed tool paths."""
    from vndub.config import load_config

    from vndub_app import installer

    s = appdirs.load_settings()
    overrides: dict[str, Any] = {
        "license_policy": s.license_policy,
        "project.models_dir": str(appdirs.effective_models_dir(s)),
        "tools_file": str(appdirs.resource("configs", "tools.yaml")),
        "tools": installer.tool_paths(),
        "gpu.vram_budget_gb": float(s.gpu_budget_gb),
    }
    return load_config(appdirs.base_config_path(s.profile), overrides)


_ENV_LOCK = threading.Lock()


class _models_env:
    """``VNDUB_MODELS_DIR`` = the effective models dir while ``vndub.models`` runs (it prefers the env var)."""

    def __enter__(self) -> Path:
        _ENV_LOCK.acquire()
        self.prev = os.environ.get("VNDUB_MODELS_DIR")
        d = appdirs.effective_models_dir()
        d.mkdir(parents=True, exist_ok=True)
        os.environ["VNDUB_MODELS_DIR"] = str(d)
        return d

    def __exit__(self, *exc: Any) -> None:
        if self.prev is None:
            os.environ.pop("VNDUB_MODELS_DIR", None)
        else:
            os.environ["VNDUB_MODELS_DIR"] = self.prev
        _ENV_LOCK.release()


# ----------------------------------------------------------------------------
# table
# ----------------------------------------------------------------------------


def _shown(spec: BackendSpec, profile: str) -> bool:
    if spec.name.startswith("mock_") or spec.name in registry.custom_names():  # custom models have their own list
        return False
    if tuple(spec.profiles) == _TEST_ONLY:
        return profile in ("test", "apptest")
    return True


def shown_specs(profile: str | None = None) -> list[BackendSpec]:
    profile = profile or appdirs.load_settings().profile
    specs = [s for s in registry.all_specs().values() if _shown(s, profile)]
    return sorted(specs, key=lambda s: (s.role, s.name))


def preset_names(profile: str | None = None) -> dict[str, list[str]]:
    shown = shown_specs(profile)
    names = {s.name for s in shown}
    return {
        "min": [n for n in PRESET_MIN if n in names],
        "recommended": [s.name for s in shown if "default" in s.profiles],
        "all": [s.name for s in shown],
    }


def expected_size_gb(spec: BackendSpec) -> float:
    return round(sum(w.size_gb for w in spec.weights if w.kind != "builtin"), 2)


def _unique_weights(specs: list[BackendSpec], models_dir: Path) -> list[tuple[BackendSpec, WeightSpec]]:
    seen: set[str] = set()
    out = []
    for s in specs:
        for w in s.weights:
            if w.kind == "builtin":
                continue
            key = str(local_dir(models_dir, w, s.name))
            if key not in seen:
                seen.add(key)
                out.append((s, w))
    return out


def missing_gb(names: list[str], models_dir: Path) -> float:
    specs = [registry.get(n) for n in names]
    return round(sum(w.size_gb for s, w in _unique_weights(specs, models_dir) if not is_present(models_dir, w, s.name)), 2)


def _reason(text: str, hint: str) -> tuple[str, dict[str, Any]]:
    import re

    for prefix, code in _REASON_PREFIXES:
        if text.startswith(prefix):
            if code == "env_missing":
                m = re.match(r"env '([^']+)'", text)
                return code, {"env": m.group(1) if m else ""}
            if code == "weights_missing":
                m = re.search(r"\(([^)]*)\)", text)
                return code, {"what": m.group(1) if m else ""}
            return code, {}
    return "unknown", {}


# -- stored self-test results ---------------------------------------------------


def selftests_file() -> Path:
    return appdirs.data_dir() / "model_selftests.json"


def read_selftests() -> dict[str, dict[str, Any]]:
    return appdirs.read_json(selftests_file(), {}) or {}


def store_selftest(name: str, row: dict[str, Any]) -> None:
    data = read_selftests()
    data[name] = row
    appdirs.write_json_atomic(selftests_file(), data)


def selftest_row(name: str, res: dict[str, Any]) -> dict[str, Any]:
    """Normalize a ``vndub.models.selftest`` result: codes instead of English messages."""
    code = None
    if not res.get("ok"):
        err = str(res.get("error", ""))
        if "weights missing" in err:
            code = "weights_missing"
        elif "not created" in err:
            code = "env_missing"
        elif err.startswith("license"):
            code = "license"
        else:
            code = "check_failed"
    return {
        "ok": bool(res.get("ok")),
        "peak_vram_gb": res.get("peak_vram_gb"),
        "device": res.get("device"),
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "code": code,
        "details": None if res.get("ok") else (str(res.get("error", "")) + (f" ({res['hint']})" if res.get("hint") else "")),
    }


def list_models() -> dict[str, Any]:
    s = appdirs.load_settings()
    models_dir = appdirs.effective_models_dir(s)
    pres = preset_names(s.profile)
    members: dict[str, list[str]] = {}
    for p, names in pres.items():
        for n in names:
            members.setdefault(n, []).append(p)
    tests = read_selftests()
    rows = []
    for spec in shown_specs(s.profile):
        real = [w for w in spec.weights if w.kind != "builtin"]
        present = all(is_present(models_dir, w, spec.name) for w in spec.weights)
        disk = round(sum(dir_size_gb(local_dir(models_dir, w, spec.name)) for w in real), 2)
        u = registry.usable(spec, policy=s.license_policy, models_dir=models_dir)
        code, rparams = ("", {}) if u.ok else _reason(u.reason, u.hint)
        rows.append(
            {
                "name": spec.name,
                "role": spec.role,
                "env": spec.env or "core",
                "size_gb": expected_size_gb(spec),
                "disk_gb": disk,
                "vram_gb": round(spec.vram_estimate_gb, 1),
                "license": spec.license,
                "commercial_ok": spec.commercial_ok,
                "present": present,
                "usable": bool(u.ok),
                "reason": code,
                "reason_params": rparams,
                "reason_details": "" if u.ok else (u.reason + (f" -> {u.hint}" if u.hint else "")),
                "presets": members.get(spec.name, []),
                "selftest": tests.get(spec.name),
            }
        )
    info = {
        p: {"count": len(names), "size_gb": round(sum(expected_size_gb(registry.get(n)) for n in names), 2), "missing_gb": missing_gb(names, models_dir)}
        for p, names in pres.items()
    }
    return {
        "rows": rows,
        "presets": pres,
        "preset_info": info,
        "disk": {"free_gb": round(downloads.free_space_gb(models_dir), 1), "data_dir": str(appdirs.data_dir()), "models_dir": str(models_dir)},
    }


def summary() -> dict[str, Any]:
    """Counts for Doctor: how much of the default profile and of the minimum preset is present."""
    s = appdirs.load_settings()
    models_dir = appdirs.effective_models_dir(s)
    pres = preset_names(s.profile)

    def count(names: list[str]) -> tuple[int, int]:
        have = sum(1 for n in names if all(is_present(models_dir, w, n) for w in registry.get(n).weights))
        return have, len(names)

    d_have, d_tot = count(pres["recommended"])
    m_have, m_tot = count(pres["min"])
    return {
        "default_present": d_have,
        "default_total": d_tot,
        "default_missing_gb": missing_gb(pres["recommended"], models_dir),
        "min_present": m_have,
        "min_total": m_tot,
    }


# ----------------------------------------------------------------------------
# downloads
# ----------------------------------------------------------------------------


class HubClient(Protocol):
    def list_files(self, repo: str, revision: str | None, allow: tuple[str, ...]) -> list[tuple[str, int]]: ...

    def fetch(self, repo: str, filename: str, revision: str | None, local: Path, progress: Callable[[int, int], None]) -> None: ...


class HfHub:
    """Hugging Face through ``huggingface_hub`` (listing, auth, redirects) with our own resumable byte transfer.

    ``huggingface_hub`` 2.x does not resume a half-downloaded file in a later call (its temporary file has a
    unique name and is deleted on failure), and with ``hf_xet`` an exception raised by a progress callback is
    swallowed, so a cancelled job kept downloading the current file to the end. Therefore :meth:`fetch` only asks
    the hub for the file's pre-signed download link, size and checksum, and streams it with
    :func:`vndub_app.downloads.fetch`: ``<file>.part`` + HTTP ``Range`` resume, cancel within a second,
    size and (for LFS files) sha256 verification. Tokens (``HF_TOKEN``, ``hf auth login``) are honoured by the
    hub call; the signed link needs no token.
    """

    def list_files(self, repo: str, revision: str | None, allow: tuple[str, ...]) -> list[tuple[str, int]]:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo, revision=revision, files_metadata=True)
        files = [(str(f.rfilename), int(f.size or 0)) for f in (info.siblings or [])]
        if allow:
            files = [(n, sz) for n, sz in files if any(fnmatch.fnmatch(n, pat) for pat in allow)]
        return files

    def resolve(self, repo: str, filename: str, revision: str | None) -> tuple[str, int, str | None]:
        """``(download URL, size, sha256 or None)`` of one file; the URL is a short-lived CDN link for LFS files."""
        from huggingface_hub import get_hf_file_metadata, hf_hub_url

        meta = get_hf_file_metadata(hf_hub_url(repo, filename, revision=revision))
        etag = str(meta.etag or "").strip('"')
        sha = etag.lower() if re.fullmatch(r"[0-9a-fA-F]{64}", etag) else None  # LFS etag = sha256; git blobs have sha1
        return str(meta.location), int(meta.size or 0), sha

    def fetch(self, repo: str, filename: str, revision: str | None, local: Path, progress: Callable[[int, int], None]) -> None:
        dst = safe_join(local, filename)
        try:
            url, size, sha = self.resolve(repo, filename, revision)
        except ImportError:  # an old huggingface_hub without the metadata API: its own downloader, no byte resume
            return self._fetch_with_hub(repo, filename, revision, local, progress)
        try:
            ident = f"hf:{repo}@{revision or 'main'}:{filename}:{size}:{sha or ''}"  # the signed link changes on every resolve
            downloads.fetch(url, dst, expected_size=size or None, expected_sha256=sha, progress=progress, target_dir=dst.parent, trust_existing=False, identity=ident, headers=_hub_auth(url))
        except AppError as e:
            if e.code == "download_unreachable":  # the signed CDN link is neither useful nor nice in an error message
                raise ConnectionError(e.details or "download failed") from e
            raise

    def _fetch_with_hub(self, repo: str, filename: str, revision: str | None, local: Path, progress: Callable[[int, int], None]) -> None:
        from huggingface_hub import hf_hub_download

        bar = _progress_class(progress)
        try:
            hf_hub_download(repo_id=repo, filename=filename, revision=revision, local_dir=str(local), tqdm_class=bar)
        except TypeError:  # an old huggingface_hub without tqdm_class: no byte progress, same download
            hf_hub_download(repo_id=repo, filename=filename, revision=revision, local_dir=str(local))


def _hub_auth(url: str) -> dict[str, str]:
    """``Authorization`` for a file that is served by the hub itself (small non-LFS files of gated or private repos);
    pre-signed CDN links need none, and the token is never sent to another host."""
    from urllib.parse import urlsplit

    try:
        from huggingface_hub import constants, get_token

        if (urlsplit(url).hostname or "").lower() != (urlsplit(constants.ENDPOINT).hostname or "").lower():
            return {}
        token = get_token()
    except Exception:
        return {}
    return {"Authorization": f"Bearer {token}"} if token else {}


def safe_join(base: Path, rel: str) -> Path:
    """``base / rel`` for a file name taken from a repo listing; refuses ``..`` and absolute names."""
    p = (Path(base) / rel).resolve()
    try:
        p.relative_to(Path(base).resolve())
    except ValueError:
        raise AppError("bad_request", params={"field": "files", "value": rel}, details=f"unsafe file name {rel!r}", status=422) from None
    return p


def _progress_class(cb: Callable[[int, int], None]):
    from tqdm import tqdm

    class _Progress(tqdm):  # type: ignore[misc]
        def __init__(self, *a: Any, **kw: Any) -> None:
            kw["disable"] = False  # hub passes disable=None (= off without a TTY); we need the counts
            kw["file"] = io.StringIO()
            super().__init__(*a, **kw)
            self._vn_done = int(kw.get("initial") or 0)
            cb(self._vn_done, int(self.total or 0))

        def update(self, n: float | None = 1) -> bool | None:
            self._vn_done += int(n or 0)
            cb(self._vn_done, int(self.total or 0))
            return super().update(n)

    return _Progress


_HUB: HubClient | None = None


def set_hub(client: HubClient | None) -> None:
    """Tests: replace the Hugging Face client."""
    global _HUB
    _HUB = client


def hub() -> HubClient:
    return _HUB or HfHub()


class _Rows:
    def __init__(self, ctx: JobContext) -> None:
        self.ctx = ctx
        self.rows: list[dict[str, Any]] = []
        self._last = 0.0

    def add(self, backend: str, weight: str, filename: str, total: int) -> dict[str, Any]:
        label = f"{weight.rstrip('/').rsplit('/', 1)[-1]}/{filename}" if weight else filename
        row = {"id": f"{backend}:{weight}:{filename}", "name": label, "backend": backend, "weight": weight, "filename": filename, "total": total, "done": 0, "state": "pending", "error": None}
        self.rows.append(row)
        return row

    def set(self, row: dict[str, Any], force: bool = False, **kv: Any) -> None:
        row.update(kv)
        now = time.monotonic()
        if force or now - self._last >= 0.3:
            self._last = now
            self.flush()

    def flush(self) -> None:
        self.ctx.set_result(files=[dict(r) for r in self.rows])
        total = sum(r["total"] for r in self.rows)
        done = sum(min(r["done"], r["total"]) for r in self.rows if r["total"])
        if total:
            # job-level counter in megabytes (bytes would be unreadable); exact bytes are in `files`
            cur = next((r for r in self.rows if r["state"] == "downloading"), None)
            t_mb = max(1, round(total / 1e6))
            self.ctx.progress((t_mb if done >= total else min(t_mb, int(done / 1e6))), t_mb, cur["filename"] if cur else None)
        else:
            self.ctx.checkpoint()


class _Hooks:
    """``vndub.models.PullHooks`` with progress rows and resume (the hub client / downloads.fetch)."""

    def __init__(self, backend: str, ctx: JobContext, rows: _Rows) -> None:
        self.backend, self.ctx, self.rows = backend, ctx, rows

    def hf(self, w: WeightSpec, d: Path, files: list[tuple[str, int]] | None = None) -> None:
        """Download the files of ``w`` into ``d``. ``files`` = a listing the caller already has (name, size)."""
        if files is None:
            files = hub().list_files(w.repo, w.revision, tuple(w.allow))
        mine = [self.rows.add(self.backend, w.repo, name, size) for name, size in files]
        self.rows.flush()
        d.mkdir(parents=True, exist_ok=True)
        for (name, size), row in zip(files, mine):
            self.ctx.checkpoint()
            have = d / name
            if size > 0 and have.is_file() and have.stat().st_size == size:  # arrived in an earlier, interrupted run
                self.rows.set(row, force=True, state="done", done=size, skipped=True)
                continue
            self.rows.set(row, force=True, state="downloading")

            def cb(done: int, total: int, _row: dict[str, Any] = row) -> None:
                if total and (not _row["total"] or total > _row["total"]):
                    _row["total"] = total
                self.rows.set(_row, done=done)

            hub().fetch(w.repo, name, w.revision, d, cb)
            self.rows.set(row, force=True, state="done", done=row["total"])

    def url(self, w: WeightSpec, dst: Path) -> None:
        row = self.rows.add(self.backend, w.url, dst.name, int(w.size_gb * 1e9))
        self.rows.set(row, force=True, state="downloading")

        def cb(done: int, total: int) -> None:
            if total:
                row["total"] = total
            self.rows.set(row, done=done)

        downloads.fetch(w.url, dst, progress=cb, checkpoint=self.ctx.checkpoint, target_dir=dst.parent, trust_existing=False)
        self.rows.set(row, force=True, state="done", done=row["total"])


def hub_error_kind(e: BaseException) -> str | None:
    """``"gated"`` (licence not accepted / token missing), ``"not_found"`` (repo, revision or file) or ``None``."""
    names = {c.__name__ for c in type(e).__mro__}
    status = getattr(getattr(e, "response", None), "status_code", None)
    if "GatedRepoError" in names or status == 403:
        return "gated"
    if names & {"RepositoryNotFoundError", "RevisionNotFoundError", "EntryNotFoundError", "RemoteEntryNotFoundError"} or status == 404:
        return "not_found"
    if status == 401:
        return "gated"
    return None


def _download_error(spec: BackendSpec, w: WeightSpec, e: BaseException, models_dir: Path) -> AppError:
    from vndub.backends.runner import AdapterError

    if isinstance(e, AppError):
        return e
    if isinstance(e, BackendUnavailable):
        import re

        m = re.search(r"env '([^']+)'", str(e))
        return AppError("env_missing", params={"env": m.group(1) if m else spec.env or "", "model": spec.name}, action="open_tools", details=str(e))
    if isinstance(e, AdapterError):
        return AppError("models_download_failed", params={"model": spec.name}, action="retry", details=str(e))
    name = type(e).__name__
    if hub_error_kind(e) == "gated":
        return AppError("hf_gated", params={"page_url": f"https://huggingface.co/{w.repo}" if w.repo else "", "model": spec.name}, action="show_manual", details=f"{name}: {e}", status=403)
    if isinstance(e, PermissionError):
        return AppError("permission_denied", params={"path": str(models_dir)}, action="open_data_dir", details=f"{name}: {e}")
    if downloads.is_disk_full(e):  # ENOSPC / Windows 112
        return AppError("disk_full", params={"need_gb": 0, "free_gb": round(downloads.free_space_gb(models_dir), 1), "path": str(models_dir)}, action="open_data_dir", details=f"{name}: {e}")
    network = isinstance(e, (OSError, ConnectionError, TimeoutError)) or any(t in name for t in ("Http", "Connect", "Timeout", "Hub", "Entry", "Repository", "Gated", "Offline", "Request"))
    if network:
        url = f"https://huggingface.co/{w.repo}" if w.repo else (w.url or "")
        return AppError(
            "download_unreachable",
            params={"url": url, "filename": w.repo or w.filename or spec.name, "target_dir": str(local_dir(models_dir, w, spec.name)), "model": spec.name},
            action="show_manual",
            details=f"{name}: {e}",
            status=502,
        )
    return AppError("models_download_failed", params={"model": spec.name}, action="retry", details=f"{name}: {e}")


def download_models(names: list[str], ctx: JobContext) -> dict[str, Any]:
    """Job body for ``POST /api/models/download``."""
    from vndub import models as pipeline_models

    specs = []
    for n in dict.fromkeys(names):
        try:
            specs.append(registry.get(n))
        except KeyError:
            raise AppError("bad_request", params={"field": "names", "value": n}, details=f"unknown backend {n!r}", status=422) from None
    models_dir = appdirs.effective_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    todo = [(s, w) for s, w in _unique_weights(specs, models_dir) if not is_present(models_dir, w, s.name)]
    need = sum(w.size_gb for _s, w in todo) * 1.05
    free = downloads.free_space_gb(models_dir)
    if todo and free < need:
        raise AppError("disk_full", params={"need_gb": round(need, 1), "free_gb": round(free, 1), "path": str(models_dir)}, action="open_data_dir")
    rows = _Rows(ctx)
    done_names: list[str] = []
    failed: list[dict[str, Any]] = []
    first: AppError | None = None
    ctx.message("models.msg.downloading", count=len(todo))
    for spec, w in todo:
        ctx.checkpoint()
        ctx.message("models.msg.model", model=spec.name)
        hooks = _Hooks(spec.name, ctx, rows)
        n_before = len(rows.rows)
        try:
            pipeline_models.pull_weight(spec, w, models_dir, hooks=hooks)  # type: ignore[arg-type]
            if len(rows.rows) == n_before:  # env/post-only weights have no file rows
                r = rows.add(spec.name, w.repo or w.url or w.subdir or w.kind, w.post or w.subdir or w.kind, 0)
                rows.set(r, force=True, state="done")
            done_names.append(spec.name)
        except Exception as e:
            from vndub_app.jobs import JobCancelled

            if isinstance(e, JobCancelled):
                raise
            err = _download_error(spec, w, e, models_dir)
            for r in rows.rows[n_before:]:
                if r["state"] != "done":
                    rows.set(r, force=True, state="failed", error=err.code)
            failed.append({"name": spec.name, "code": err.code, "params": err.params})
            first = first or err
            ctx.log(f"ERROR {spec.name}: {err.code} {err.details.splitlines()[0] if err.details else ''}")
    rows.flush()
    ctx.set_result(files=[dict(r) for r in rows.rows], downloaded=done_names, failed=failed)
    if first is not None:
        raise first
    return {"downloaded": done_names}


def check_models(names: list[str], ctx: JobContext) -> dict[str, Any]:
    """Job body for ``POST /api/models/check``: self-test each backend, remember peak VRAM."""
    from vndub import models as pipeline_models

    for n in names:
        try:
            registry.get(n)
        except KeyError:
            raise AppError("bad_request", params={"field": "names", "value": n}, details=f"unknown backend {n!r}", status=422) from None
    results = []
    cfg = build_cfg()
    for i, n in enumerate(names):
        ctx.checkpoint()
        ctx.message("models.msg.checking", model=n)
        ctx.progress(i, len(names), n)
        with _models_env():
            res = pipeline_models.selftest(cfg, n)
        row = selftest_row(n, res)
        store_selftest(n, row)
        results.append({"name": n, **row})
        ctx.set_result(rows=list(results))
    ctx.progress(len(names), len(names), None)
    return {"rows": results}


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------


class NamesBody(BaseModel):
    names: list[str]


def _check_names(names: list[str]) -> None:
    if not names:
        raise AppError("bad_request", params={"field": "names"}, details="names must not be empty", status=422)
    for n in names:
        try:
            registry.get(n)
        except KeyError:
            raise AppError("bad_request", params={"field": "names", "value": n}, details=f"unknown backend {n!r}", status=422) from None


@router.get("/api/models")
def api_models() -> dict[str, Any]:
    return list_models()


@router.post("/api/models/download")
def api_models_download(body: NamesBody) -> dict[str, Any]:
    from vndub_app import jobs

    _check_names(body.names)
    names = list(body.names)
    job = jobs.runner().submit("models_download", lambda ctx: download_models(names, ctx), params={"names": names}, gpu=False)
    return job.info()


@router.post("/api/models/check")
def api_models_check(body: NamesBody) -> dict[str, Any]:
    from vndub_app import jobs

    _check_names(body.names)
    names = list(body.names)
    job = jobs.runner().submit("models_check", lambda ctx: check_models(names, ctx), params={"names": names}, gpu=True)
    return job.info()

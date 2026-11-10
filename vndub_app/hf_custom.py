"""Models by Hugging Face link: inspect a repo, download chosen files, keep a list of what was added.

``POST /api/models/hf/inspect``   ``{url}`` -> repo facts: files, sizes, licence, gated, detected ``kind``, suggested files
``POST /api/models/hf/add``       ``{repo, revision, files, kind, label?}`` -> Job ``models_hf_add`` (per-file rows in ``result.files``)
``GET  /api/models/custom``       what was added (with ``present``)
``DELETE /api/models/custom/{id}``  remove the files and the entry (refused while a job uses it)
``GET  /api/models/hf/recommended``  the editable list ``app/recommended_models.json``

Added models live in ``<models_dir>/custom/<owner>__<repo>/`` and are listed in ``<data>/custom_models.json``;
the file is also what ``vndub.backends.registry`` reads (env ``VNDUB_CUSTOM_MODELS``, set by :func:`sync_env`) so a
``gguf_llm`` entry becomes a translation backend the pipeline can use (Settings -> "Translation model").

Kinds (see :func:`detect_kind`): ``builtin`` (the repo is one the app already lists), ``gguf_llm`` (a translation LLM for
llama.cpp), ``rvc`` (a character voice model: stored and listed; the pipeline does not attach it to a character by
itself), ``files`` (anything else: downloaded and listed, not used by the pipeline).

Network access goes through two injectable clients, like ``models_api.set_hub``: :func:`set_info_client` (repo
metadata, :class:`HfInfo`) and ``models_api.set_hub`` (file listing and bytes). Tests replace both.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlsplit

from fastapi import APIRouter
from pydantic import BaseModel

from vndub.backends import registry
from vndub.backends.base import WeightSpec
from vndub.backends.weights import MARKER, dir_size_gb
from vndub.util import now_iso
from vndub_app import appdirs, downloads, models_api
from vndub_app.errors import AppError
from vndub_app.jobs import JobCancelled, JobContext

log = logging.getLogger("vndub_app.hf_custom")

router = APIRouter()

KINDS = ("gguf_llm", "rvc", "files")
HF_HOSTS = frozenset({"huggingface.co", "www.huggingface.co", "hf.co", "www.hf.co"})
# first path segments of huggingface.co that are not model owners
RESERVED_OWNERS = frozenset(
    {"datasets", "spaces", "models", "docs", "api", "blog", "settings", "login", "join", "papers", "collections", "organizations", "new", "pricing", "tasks", "chat", "posts", "buckets", "learn", "docs"}
)
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_HOST_LIKE = re.compile(r"^[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)+(:\d+)?$")
RVC_MAX_BYTES = 1_000_000_000
BIG_MODEL_BYTES = 10_000_000_000  # a translation GGUF bigger than this will not fit an 8 GB card
PREFERRED_QUANTS = ("Q4_K_M", "IQ4_XS", "Q4_K_S", "Q5_K_M")
_LOW_QUANT = re.compile(r"(?<![A-Za-z0-9])(?:IQ1|IQ2|TQ1|TQ2|Q2)", re.I)
_SHARD = re.compile(r"^(?P<stem>.+)-(?P<i>\d{5})-of-(?P<n>\d{5})\.gguf$", re.I)
# pipeline tags of repos that hold GGUF files but are not text models
_NON_TEXT_TAGS = frozenset({"text-to-speech", "automatic-speech-recognition", "audio-to-audio", "text-to-image", "text-to-video", "image-to-image", "feature-extraction", "audio-classification", "text-to-audio"})
_LOCK = threading.RLock()


# ----------------------------------------------------------------------------
# link parsing
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedUrl:
    repo: str  # "owner/name"
    revision: str | None = None
    path: str | None = None  # file or folder from a /blob/ /resolve/ /tree/ link


def _bad_url(text: str, why: str = "") -> AppError:
    return AppError("hf_bad_url", params={"url": text[:200]}, details=why or "not a Hugging Face model link", status=422)


def _mirror_host() -> str:
    """Host of the mirror set in Settings (``hf_endpoint``), ``""`` when none: a link copied from the mirror's page works too."""
    try:
        ep = (getattr(appdirs.load_settings(), "hf_endpoint", "") or "").strip()
        return (urlsplit(ep).hostname or "").lower() if ep else ""
    except Exception:
        return ""


def parse_hf_url(text: str) -> ParsedUrl:
    """``https://huggingface.co/<owner>/<repo>[/tree|blob|resolve/<rev>[/<path>]]``, ``hf.co/...``, no scheme, ``owner/repo``."""
    raw = (text or "").strip().strip("\"'<>").strip()
    if not raw or len(raw) > 2000 or any(c.isspace() for c in raw):
        raise _bad_url(raw, "empty or contains spaces")
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", raw):
        url = raw
    elif _HOST_LIKE.match(raw.split("/", 1)[0].split("?", 1)[0]):  # "huggingface.co/..." or some other host
        url = "https://" + raw
    else:  # "owner/repo[/...]"
        url = "https://huggingface.co/" + raw.lstrip("/")
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        raise _bad_url(raw, "malformed URL") from None
    if parts.scheme.lower() not in ("http", "https"):
        raise _bad_url(raw, f"unsupported scheme {parts.scheme!r}")
    if "@" in parts.netloc or (port not in (None, 80, 443)):
        raise _bad_url(raw, "credentials or a port in the URL")
    host = (parts.hostname or "").lower()
    if host not in HF_HOSTS and host != _mirror_host():
        raise _bad_url(raw, f"host {host!r} is not huggingface.co")
    segs = [unquote(s) for s in parts.path.split("/") if s]
    if len(segs) < 2 or segs[0].lower() in RESERVED_OWNERS:
        raise _bad_url(raw, "expected huggingface.co/<owner>/<model>")
    owner, name = segs[0], segs[1]
    if name.lower().endswith(".git"):
        name = name[:-4]
    if not (_NAME_RE.match(owner) and _NAME_RE.match(name)) or ".." in owner + name:
        raise _bad_url(raw, "invalid owner or model name")
    revision: str | None = None
    path: str | None = None
    rest = segs[2:]
    if rest and rest[0] in ("tree", "blob", "resolve", "raw", "blame", "edit", "commit") and len(rest) > 1:
        revision = rest[1]
        if len(revision) > 200 or any(ord(c) < 32 for c in revision) or ".." in revision.split("/"):
            raise _bad_url(raw, "invalid revision")
        if rest[0] != "commit" and len(rest) > 2:
            sub = rest[2:]
            if ".." not in sub:
                path = "/".join(sub)
    return ParsedUrl(f"{owner}/{name}", revision, path)


# ----------------------------------------------------------------------------
# repo metadata (injectable)
# ----------------------------------------------------------------------------


@dataclass
class RepoInfo:
    repo: str
    sha: str = ""
    files: list[tuple[str, int]] = field(default_factory=list)  # (name, size in bytes)
    license: str | None = None
    gated: str | bool | None = None  # False | "auto" | "manual"
    private: bool = False
    pipeline_tag: str | None = None
    tags: list[str] = field(default_factory=list)


class InfoClient(Protocol):
    def info(self, repo: str, revision: str | None) -> RepoInfo: ...


class HfInfo:
    """``HfApi().model_info(files_metadata=True)``; raises the hub's own exceptions (mapped by :func:`hub_app_error`)."""

    def info(self, repo: str, revision: str | None) -> RepoInfo:
        from huggingface_hub import HfApi

        i = HfApi().model_info(repo, revision=revision, files_metadata=True)
        card = i.card_data.to_dict() if getattr(i, "card_data", None) is not None else {}
        lic = card.get("license")
        if lic == "other" and card.get("license_name"):
            lic = str(card["license_name"])
        if not lic:
            lic = next((t.split(":", 1)[1] for t in (i.tags or []) if t.startswith("license:")), None)
        return RepoInfo(
            repo=str(i.id or repo),
            sha=str(i.sha or ""),
            files=[(str(s.rfilename), int(s.size or 0)) for s in (i.siblings or [])],
            license=str(lic) if lic else None,
            gated=i.gated if i.gated else False,
            private=bool(i.private),
            pipeline_tag=i.pipeline_tag,
            tags=[str(t) for t in (i.tags or [])],
        )


_INFO: InfoClient | None = None


def set_info_client(client: InfoClient | None) -> None:
    """Tests: replace the repo-metadata client."""
    global _INFO
    _INFO = client


def info_client() -> InfoClient:
    return _INFO or HfInfo()


def hub_app_error(e: BaseException, repo: str, *, filename: str = "", target_dir: str = "") -> AppError:
    """A hub / network exception as the app's error: ``hf_gated``, ``hf_not_found``, ``download_unreachable`` ..."""
    if isinstance(e, AppError):
        return e
    page = f"https://huggingface.co/{repo}"
    name = type(e).__name__
    kind = models_api.hub_error_kind(e)
    if kind == "gated":
        return AppError("hf_gated", params={"repo": repo, "page_url": page}, action="none", details=f"{name}: {e}", status=403)
    if kind == "not_found":
        return AppError("hf_not_found", params={"repo": repo, "page_url": page}, action="none", details=f"{name}: {e}", status=404)
    if isinstance(e, PermissionError):
        return AppError("permission_denied", params={"path": target_dir or str(appdirs.effective_models_dir())}, action="open_data_dir", details=f"{name}: {e}")
    if downloads.is_disk_full(e):  # ENOSPC / Windows 112
        d = target_dir or str(appdirs.effective_models_dir())
        return AppError("disk_full", params={"need_gb": 0, "free_gb": round(downloads.free_space_gb(Path(d)), 1), "path": d}, action="open_data_dir", details=f"{name}: {e}")
    network = isinstance(e, (OSError, ConnectionError, TimeoutError)) or any(t in name for t in ("Http", "Connect", "Timeout", "Hub", "Offline", "Request", "Network"))
    if network:
        return AppError(
            "download_unreachable",
            params={"url": page, "filename": filename or repo, "target_dir": target_dir},
            action="show_manual",
            details=f"{name}: {e}",
            status=502,
        )
    return AppError("models_download_failed", params={"model": repo}, action="retry", details=f"{name}: {e}")


# ----------------------------------------------------------------------------
# kind detection and file choice
# ----------------------------------------------------------------------------


@dataclass
class GgufSet:
    """One model file, or all shards (``-0000N-of-0000M``) of one: they must be downloaded together."""

    key: str
    files: list[str]
    size: int
    complete: bool


def gguf_sets(files: list[tuple[str, int]]) -> list[GgufSet]:
    groups: dict[str, dict[str, Any]] = {}
    for name, size in files:
        if not name.lower().endswith(".gguf") or "mmproj" in name.lower():  # a vision projector is not a language model
            continue
        m = _SHARD.match(name)
        if m:
            key, idx, n = f"{m['stem']}|{m['n']}", int(m["i"]), int(m["n"])
        else:
            key, idx, n = name, 1, 1
        g = groups.setdefault(key, {"n": n, "parts": {}})
        g["parts"][idx] = (name, size)
    out = []
    for key, g in groups.items():
        idxs = sorted(g["parts"])
        parts = [g["parts"][i] for i in idxs]
        out.append(GgufSet(key, [p[0] for p in parts], sum(p[1] for p in parts), idxs == list(range(1, g["n"] + 1))))
    return sorted(out, key=lambda s: s.files[0])


def choose_gguf_set(sets: list[GgufSet]) -> GgufSet | None:
    """One sensible quantization: Q4_K_M, then IQ4_XS / Q4_K_S / Q5_K_M, else the smallest single file (not a 1-2 bit
    one), else the first complete shard set. Incomplete shard sets are never chosen."""
    sets = [s for s in sets if s.complete]
    if not sets:
        return None
    for q in PREFERRED_QUANTS:
        pat = re.compile(rf"(?<![A-Za-z0-9]){q}(?![A-Za-z0-9])", re.I)
        hits = [s for s in sets if pat.search(s.files[0])]
        if hits:
            return min(hits, key=lambda s: (s.size, s.files[0]))
    usable = [s for s in sets if not _LOW_QUANT.search(s.files[0])] or sets
    single = [s for s in usable if len(s.files) == 1]
    if single:
        return min(single, key=lambda s: (s.size, s.files[0]))
    return min(usable, key=lambda s: s.files[0])


def suggest_gguf(files: list[tuple[str, int]], focus: str | None = None) -> list[str]:
    sets = gguf_sets(files)
    if focus:
        exact = next((s for s in sets if focus in s.files), None)
        if exact is not None:
            return list(exact.files)
        inside = [s for s in sets if s.files[0].startswith(focus.rstrip("/") + "/")]
        if inside:
            sets = inside
    pick = choose_gguf_set(sets)
    return list(pick.files) if pick else []


def suggest_rvc(repo: str, files: list[tuple[str, int]]) -> list[str]:
    """The ``.pth`` named like the repo (else the first), plus the ``.index`` that shares its stem (else the only one)."""
    pths = sorted(n for n, _ in files if n.lower().endswith(".pth"))
    if not pths:
        return []
    repo_name = repo.rsplit("/", 1)[-1].lower()
    pick = next((p for p in pths if Path(p).stem.lower() in repo_name or repo_name in Path(p).stem.lower()), pths[0])
    idx = sorted(n for n, _ in files if n.lower().endswith(".index"))
    stem = Path(pick).stem.lower()
    index = next((i for i in idx if stem in Path(i).stem.lower()), idx[0] if len(idx) == 1 else None)
    return [pick] + ([index] if index else [])


def builtin_for(repo: str) -> str | None:
    """Name of the built-in backend whose weights come from exactly this repo (first by role, name)."""
    low = repo.lower()
    for spec in models_api.shown_specs():
        if any(w.repo.lower() == low for w in spec.weights if w.repo):
            return spec.name
    return None


def detect_kind(repo: str, files: list[tuple[str, int]], tags: list[str] | None = None) -> tuple[str, list[str]]:
    """``(kind, suggested file names)`` for a repo listing."""
    tags = [t.lower() for t in (tags or [])]
    if builtin_for(repo):
        return "builtin", []
    if any(n.lower().endswith(".gguf") and "mmproj" not in n.lower() for n, _ in files):
        return "gguf_llm", suggest_gguf(files)
    sizes = dict(files)
    rvc = suggest_rvc(repo, files)
    if rvc and (sum(sizes.get(n, 0) for n in rvc) < RVC_MAX_BYTES or "rvc" in tags):
        return "rvc", rvc
    total = sum(s for _, s in files)
    return "files", [n for n, _ in files] if 0 < total < RVC_MAX_BYTES else []


def _warnings(info: RepoInfo, kind: str, suggested: list[tuple[str, int]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if info.gated:
        out.append({"code": "gated", "params": {}})
    lic = (info.license or "").lower()
    if not lic:
        out.append({"code": "unknown_license", "params": {}})
    elif kind == "gguf_llm" and lic not in registry._COMMERCIAL_LICENSES:  # noqa: SLF001
        out.append({"code": "license_check", "params": {"license": info.license}})
    if kind == "gguf_llm":
        size = sum(s for _, s in suggested)
        if size > BIG_MODEL_BYTES:
            out.append({"code": "too_big", "params": {"size_gb": round(size / 1e9, 1)}})
        if (info.pipeline_tag or "") in _NON_TEXT_TAGS:
            out.append({"code": "not_text_model", "params": {"tag": info.pipeline_tag}})
    return out


# ----------------------------------------------------------------------------
# inspect
# ----------------------------------------------------------------------------


def inspect_url(url: str) -> dict[str, Any]:
    parsed = parse_hf_url(url)
    sync_env()
    try:
        info = info_client().info(parsed.repo, parsed.revision)
    except Exception as e:
        raise hub_app_error(e, parsed.repo) from e
    files = sorted(info.files)
    sizes = dict(files)
    kind, suggested = detect_kind(info.repo or parsed.repo, files, info.tags)
    if kind == "gguf_llm":
        suggested = suggest_gguf(files, parsed.path) or suggested
    elif parsed.path and parsed.path in sizes and kind in ("files", "rvc") and parsed.path not in suggested:
        suggested = [*suggested, parsed.path] if kind == "rvc" else [parsed.path]
    free = downloads.free_space_gb(appdirs.effective_models_dir())
    return {
        "repo": info.repo or parsed.repo,
        "revision": parsed.revision or "main",
        "sha": info.sha,
        "page_url": f"https://huggingface.co/{info.repo or parsed.repo}",
        "files": [{"name": n, "size": s} for n, s in files],
        "total_size": sum(sizes.values()),
        "license": info.license,
        "gated": bool(info.gated),
        "private": info.private,
        "kind": kind,
        "suggested": suggested,
        "suggested_size": sum(sizes.get(n, 0) for n in suggested),
        "builtin": builtin_for(info.repo or parsed.repo),
        "warnings": _warnings(info, kind, [(n, sizes.get(n, 0)) for n in suggested]),
        "free_gb": round(free, 1),
        "focus": parsed.path,
    }


# ----------------------------------------------------------------------------
# the custom list (custom_models.json)
# ----------------------------------------------------------------------------


def custom_file() -> Path:
    return appdirs.data_dir() / "custom_models.json"


def sync_env() -> Path:
    """Point the backend registry at this app's list (``VNDUB_CUSTOM_MODELS``); cheap, call it before using the registry."""
    f = custom_file()
    os.environ[registry.CUSTOM_ENV] = str(f)
    return f


def read_entries() -> list[dict[str, Any]]:
    return registry.read_custom_entries(custom_file())


def _write_entries(entries: list[dict[str, Any]]) -> None:
    appdirs.write_json_atomic(custom_file(), {"version": 1, "models": entries})
    sync_env()


def slug_of(repo: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", repo.lower()).strip("_") or "model"


def dir_name(repo: str) -> str:
    owner, _, name = repo.partition("/")
    return re.sub(r"[^A-Za-z0-9._-]", "_", f"{owner}__{name}")


def entry_id(repo: str) -> str:
    return f"hf_{slug_of(repo)}"


def unique_id(repo: str, entries: list[dict[str, Any]]) -> str:
    cid = entry_id(repo)
    other = next((e for e in entries if e.get("id") == cid and str(e.get("repo", "")).lower() != repo.lower()), None)
    return cid if other is None else f"{cid}_{hashlib.sha1(repo.lower().encode()).hexdigest()[:6]}"


def find_entry(cid: str) -> dict[str, Any] | None:
    return next((e for e in read_entries() if e.get("id") == cid), None)


def entry_dir(entry: dict[str, Any], models_dir: Path | None = None) -> Path:
    return Path(models_dir or appdirs.effective_models_dir()) / str(entry.get("dir") or "")


def list_custom() -> list[dict[str, Any]]:
    sync_env()
    s = appdirs.load_settings()
    models_dir = appdirs.effective_models_dir(s)
    out = []
    for e in read_entries():
        d = entry_dir(e, models_dir)
        spec = None
        if e.get("kind") == "gguf_llm":
            try:
                spec = registry.get(str(e.get("id")))
            except KeyError:
                spec = None
        usable = None
        if spec is not None:
            usable = bool(registry.usable(spec, policy=s.license_policy, models_dir=models_dir).ok)
        out.append(
            {
                **e,
                "present": (d / MARKER).exists(),
                "disk_gb": round(dir_size_gb(d), 2),
                "usable": usable,
                "in_use": getattr(s, "translation_llm", "") == e.get("id"),
            }
        )
    return out


# ----------------------------------------------------------------------------
# add (job)
# ----------------------------------------------------------------------------


def clean_files(files: list[str]) -> list[str]:
    out: list[str] = []
    for f in files:
        name = str(f).replace("\\", "/").strip()
        parts = name.split("/")
        if not name or ".." in parts or ":" in parts[0] or any(p in ("", ".") for p in parts):  # absolute, drive, empty or dotted parts
            raise AppError("bad_request", params={"field": "files", "value": str(f)[:200]}, details="invalid file name", status=422)
        if name not in out:
            out.append(name)
    return out


def add_model(repo: str, revision: str | None, files: list[str], kind: str, label: str | None, ctx: JobContext) -> dict[str, Any]:
    """Job body for ``POST /api/models/hf/add``."""
    sync_env()
    models_dir = appdirs.effective_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    rel = f"custom/{dir_name(repo)}"
    dest = models_dir / rel
    cid = unique_id(repo, read_entries())
    ctx.message("models.msg.hf_add", repo=repo)
    try:
        listing = dict(models_api.hub().list_files(repo, revision, ()))
    except Exception as e:
        raise hub_app_error(e, repo, target_dir=str(dest)) from e
    missing = [f for f in files if f not in listing]
    if missing:
        raise AppError("bad_request", params={"field": "files", "value": missing[0]}, details=f"not in {repo}: {missing}", status=422)
    chosen = [(f, listing[f]) for f in files]
    todo = sum(sz for f, sz in chosen if not ((dest / f).is_file() and (dest / f).stat().st_size == sz))
    free = downloads.free_space_gb(models_dir)
    need = todo * 1.05 / 1e9
    if todo and free < need:
        raise AppError("disk_full", params={"need_gb": round(need, 1), "free_gb": round(free, 1), "path": str(models_dir)}, action="open_data_dir")
    rows = models_api._Rows(ctx)  # noqa: SLF001
    hooks = models_api._Hooks(cid, ctx, rows)  # noqa: SLF001
    weight = WeightSpec("hf_file", repo=repo, revision=revision, subdir=rel, size_gb=round(sum(s for _, s in chosen) / 1e9, 3))
    try:
        hooks.hf(weight, dest, chosen)
    except JobCancelled:
        raise  # the partial files stay: the next start resumes them
    except Exception as e:
        cur = next((r for r in rows.rows if r["state"] == "downloading"), None)
        err = hub_app_error(e, repo, filename=cur["filename"] if cur else "", target_dir=str(dest))
        for r in rows.rows:
            if r["state"] != "done":
                rows.set(r, force=True, state="failed", error=err.code)
        raise err from e
    rows.flush()
    lic: str | None = None
    try:
        lic = info_client().info(repo, revision).license
    except Exception:  # the licence is informative; the download is done
        pass
    with _LOCK:
        entries = read_entries()
        cid = unique_id(repo, entries)
        old = next((e for e in entries if e.get("id") == cid), None)
        (dest / MARKER).write_text(f"{now_iso()} hf {repo}@{revision or 'main'}\n", encoding="utf-8")
        if old is not None:  # same repo again (another quantization): drop the files that are no longer wanted
            for f in old.get("files", []):
                if f not in files:
                    try:
                        (dest / f).unlink()
                    except OSError:
                        pass
        entry = {
            "id": cid,
            "repo": repo,
            "revision": revision or "main",
            "files": list(files),
            "kind": kind,
            "dir": rel,
            "size_gb": round(sum(s for _, s in chosen) / 1e9, 3),
            "license": lic if lic is not None else (old or {}).get("license"),
            "label": (label or "").strip() or (old or {}).get("label") or repo.rsplit("/", 1)[-1],
            "added_at": now_iso(),
        }
        _write_entries([e for e in entries if e.get("id") != cid] + [entry])
    ctx.set_result(files=[dict(r) for r in rows.rows], id=cid, entry=entry)
    return {"id": cid, "entry": entry}


# ----------------------------------------------------------------------------
# delete
# ----------------------------------------------------------------------------


def _in_use(cid: str) -> bool:
    from vndub_app import jobs

    s = appdirs.load_settings()
    for j in jobs.runner().list(active_only=True):
        if j.kind == "models_hf_add" and j.params.get("id") == cid:
            return True
        if getattr(s, "translation_llm", "") == cid and (j.gpu or j.project):
            return True  # a pipeline job may be translating with it
    return False


def delete_custom(cid: str) -> dict[str, Any]:
    sync_env()
    entry = find_entry(cid)
    if entry is None:
        raise AppError("bad_request", params={"field": "id", "value": cid}, details="no such custom model", status=404)
    if _in_use(cid):
        raise AppError("bad_request", params={"field": "id", "value": cid, "reason": "in_use"}, details="a job uses this model", status=409)
    models_dir = appdirs.effective_models_dir()
    d = entry_dir(entry, models_dir).resolve()
    root = (models_dir / "custom").resolve()
    try:
        d.relative_to(root)
    except ValueError:  # never delete anything outside <models>/custom
        raise AppError("bad_request", params={"field": "id", "value": cid}, details=f"{d} is not inside {root}", status=422) from None
    if d != root and d.exists():
        try:
            shutil.rmtree(d)
        except OSError as e:  # a file is open in another program
            raise AppError("permission_denied", params={"path": str(d)}, action="open_data_dir", details=f"{type(e).__name__}: {e}") from e
    with _LOCK:
        _write_entries([e for e in read_entries() if e.get("id") != cid])
    if getattr(appdirs.load_settings(), "translation_llm", "") == cid:
        appdirs.update_settings(lambda s: setattr(s, "translation_llm", "") if s.translation_llm == cid else None)
        _regenerate_project_config()
    return {"ok": True}


def _regenerate_project_config() -> None:
    from vndub_app import session

    slug = session.current_slug()
    if slug:
        try:
            appdirs.write_project_config(appdirs.project_handle(slug))
        except AppError:
            pass


# ----------------------------------------------------------------------------
# recommended links
# ----------------------------------------------------------------------------


def recommended_file() -> Path:
    return appdirs.resource("app", "recommended_models.json")


def recommended() -> dict[str, Any]:
    data = appdirs.read_json(recommended_file(), {}) or {}
    items = []
    for it in data.get("items", []) if isinstance(data, dict) else []:
        if isinstance(it, dict) and it.get("repo"):
            items.append({**it, "url": f"https://huggingface.co/{it['repo']}"})
    return {"items": items, "checked": data.get("checked") if isinstance(data, dict) else None}


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------


class InspectBody(BaseModel):
    url: str


class AddBody(BaseModel):
    repo: str
    revision: str | None = None
    files: list[str]
    kind: str
    label: str | None = None


@router.post("/api/models/hf/inspect")
def api_inspect(body: InspectBody) -> dict[str, Any]:
    return inspect_url(body.url)


@router.get("/api/models/hf/recommended")
def api_recommended() -> dict[str, Any]:
    return recommended()


@router.post("/api/models/hf/add")
def api_add(body: AddBody) -> dict[str, Any]:
    from vndub_app import jobs

    sync_env()
    repo = parse_hf_url(body.repo).repo
    if body.kind not in KINDS:
        raise AppError("bad_request", params={"field": "kind", "value": body.kind}, status=422)
    files = clean_files(body.files)
    if not files:
        raise AppError("bad_request", params={"field": "files"}, details="no files chosen", status=422)
    if body.kind == "gguf_llm" and not any(f.lower().endswith(".gguf") for f in files):
        raise AppError("bad_request", params={"field": "files"}, details="gguf_llm needs at least one .gguf file", status=422)
    revision = (body.revision or "").strip() or None
    cid = unique_id(repo, read_entries())
    for j in jobs.runner().list(active_only=True):
        if j.kind == "models_hf_add" and j.params.get("repo") == repo:
            raise AppError("job_busy", params={"job_id": j.id, "kind": j.kind}, status=409)
    label = body.label
    job = jobs.runner().submit(
        "models_hf_add",
        lambda ctx: add_model(repo, revision, files, body.kind, label, ctx),
        params={"id": cid, "repo": repo, "revision": revision, "files": files, "kind": body.kind},
        gpu=False,
    )
    return job.info()


@router.get("/api/models/custom")
def api_custom() -> list[dict[str, Any]]:
    return list_custom()


@router.delete("/api/models/custom/{cid}")
def api_custom_delete(cid: str) -> dict[str, Any]:
    return delete_custom(cid)

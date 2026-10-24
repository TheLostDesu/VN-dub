"""Step 7: put the voice patch into the game, with backup and rollback. Never claims success for manual engines.

``GET  /api/install/plan``         what will be written (+ ``INSTALL.md``, ``manifest_patch.json`` summary with the patch ``coverage``, backups)
``GET  /api/patch/coverage``      ``{coverage}``: dubbed / installed / provisional / left out / narration counts of the built patch
``POST /api/install/apply``        ``{backup: true}`` -> Job; refuses with ``install_manual_only`` for manual engines
``POST /api/install/rollback``     ``{backup_id}`` -> Job (restores replaced files, removes files the patch added)
``POST /api/install/zip``          -> Job, result ``{path}`` (patch folder + INSTALL.md for a friend, no originals)
``POST /api/install/open-patch``   open ``out/<engine>/`` in the file manager

All endpoints work on the current project (``vndub_app.session.current_handle()``) or ``?slug=``; the router carries
full ``/api/...`` paths (include it without a prefix).

What is written (``out/<engine>/`` laid over the game folder, same relative paths):

======================  =====================================================================
Ren'Py, NScripter,      loose files: automatic
TyranoScript
KiriKiri                a patch ``.xp3`` (or ``unencrypted/``) the game loads: automatic when the game is
                        not encrypted or ``version.dll`` (KirikiriTools) already sits next to the exe
Siglus, Artemis,        archive has to be rebuilt with an outside tool: manual
CatSystem2, YU-RIS,
Majiro, Unity, BGI,
generic
======================  =====================================================================

Not copied automatically for any engine: ``INSTALL.md``, ``manifest_patch.json``, temporary ``_*`` folders and
the ``unvoiced/`` folder (voices for lines the game never voiced: a manual step, see the plan ``notes``).
Backups live in ``<data>/backups/<slug>/<timestamp>/`` (``files/`` + ``manifest.json``); the manifest is written
before the first game file is touched so an interrupted install can still be rolled back.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from vndub_app import appdirs, downloads, openers
from vndub_app.errors import AppError
from vndub_app.jobs import JobContext

log = logging.getLogger("vndub_app.patch_install")

router = APIRouter()

METADATA = ("INSTALL.md", "manifest_patch.json")
MANUAL_DIRS = ("unvoiced",)
AUTO_ENGINES = ("renpy", "nscripter", "tyrano", "kirikiri")
REPAIR_ENGINES = ("siglus", "artemis", "catsystem2", "yuris", "majiro")
PLAN_FILE_LIMIT = 300
STORED_EXTS = (".ogg", ".opus", ".mp3", ".flac", ".m4a", ".xp3", ".zip", ".wav")  # audio / archives: no point deflating (wav: kept fast)
MANIFEST = "manifest.json"


# ----------------------------------------------------------------------------
# project, engine, patch folder
# ----------------------------------------------------------------------------


def _project(slug: str | None) -> appdirs.ProjectHandle:
    """``?slug=`` project, else the app's current project (``vndub_app.session``)."""
    if slug:
        return appdirs.project_handle(slug)
    from vndub_app import session

    return session.current_handle()


def _project_info(h: appdirs.ProjectHandle) -> Any:
    from vndub.config import load_project_yaml

    f = h.work / "project.yaml"
    if not f.exists():
        raise AppError("nothing_to_install", params={"reason": "not_detected"}, action="none", status=409, details=f"{f} not found: run the import first")
    return load_project_yaml(f)


def _engine_out(engine: str) -> str:
    from vndub.engines import ENGINES, get_engine

    if engine in ENGINES:
        return get_engine(engine).out_name
    return engine


def patch_dir_of(h: appdirs.ProjectHandle, engine: str) -> Path:
    return h.out / _engine_out(engine)


def patch_files(patch_dir: Path) -> list[Path]:
    """Files the automatic install writes (everything except metadata, ``_*`` temp folders and ``unvoiced/``)."""
    if not patch_dir.is_dir():
        return []
    out: list[Path] = []
    for root, dirs, files in os.walk(patch_dir):
        rel_root = Path(root).relative_to(patch_dir)
        if rel_root == Path("."):
            dirs[:] = sorted(d for d in dirs if not d.startswith(("_", ".")) and d not in MANUAL_DIRS)
        for f in sorted(files):
            if rel_root == Path(".") and f in METADATA:
                continue
            if f.startswith(".") or f.endswith(".vndub-tmp"):
                continue
            out.append(Path(root) / f)
    return out


def _unvoiced_dir(patch_dir: Path) -> Path | None:
    for d in MANUAL_DIRS:
        p = patch_dir / d
        if p.is_dir() and any(p.rglob("*")):
            return p
    return None


# ----------------------------------------------------------------------------
# classification and plan
# ----------------------------------------------------------------------------


def _version_dll_helper(game_dir: Path) -> dict[str, Any]:
    path: str | None = None
    try:
        from vndub_app import installer

        e = installer.get_entry("xp3pack")
        p = installer._find_file(installer.target_dir(e), "version.dll")
        path = str(p) if p else None
    except Exception:
        path = None
    return {"name": "version.dll", "tool_path": path, "in_game": (game_dir / "version.dll").exists()}


def _target_base(engine: str, game_dir: Path) -> tuple[Path, str | None]:
    """Directory the patch folder is laid over, or a reason key when the game folder does not look right."""
    if engine == "renpy":
        return game_dir, (None if (game_dir / "game").is_dir() else "install.manual.layout_mismatch")
    if engine == "tyrano":
        for base in (game_dir, game_dir / "resources" / "app"):
            if (base / "data").is_dir():
                return base, None
        return game_dir, "install.manual.tyrano_packed"
    return game_dir, None


def classify(engine: str, project: Any, game_dir: Path, patch_dir: Path, files: list[Path]) -> dict[str, Any]:
    """``auto_possible`` / ``manual_reason_key`` / ``base`` / ``helpers`` for one engine."""
    res: dict[str, Any] = {"auto_possible": False, "manual_reason_key": None, "base": game_dir, "helpers": []}
    if not game_dir.is_dir():
        res["manual_reason_key"] = "install.manual.game_missing"
        return res
    if not files:
        res["manual_reason_key"] = "install.manual.no_patch"
        return res
    if engine == "unity":
        res["manual_reason_key"] = "install.manual.unity"
    elif engine in REPAIR_ENGINES:
        res["manual_reason_key"] = "install.manual.repack"
    elif engine not in AUTO_ENGINES:
        res["manual_reason_key"] = "install.manual.loose_unverified"
    else:
        base, bad = _target_base(engine, game_dir)
        res["base"] = base
        if bad:
            res["manual_reason_key"] = bad
        elif engine == "kirikiri":
            has_dll = (game_dir / "version.dll").exists()
            res["helpers"] = [_version_dll_helper(game_dir)]
            encrypted = bool(getattr(project, "encrypted", False))
            xp3 = [f for f in files if f.suffix.lower() == ".xp3" and f.parent == patch_dir]
            unenc = [f for f in files if f.relative_to(patch_dir).parts[0] == "unencrypted"]
            if unenc and not has_dll:
                res["manual_reason_key"] = "install.manual.encrypted_kirikiri"  # the folder is only read through version.dll
            elif encrypted and not has_dll:
                res["manual_reason_key"] = "install.manual.encrypted_kirikiri"
            elif not xp3 and not unenc:
                res["manual_reason_key"] = "install.manual.loose_unverified"
            else:
                res["auto_possible"] = True
        else:
            res["auto_possible"] = True
    return res


def _safe_target(base: Path, rel: Path) -> Path:
    dst = base / rel
    try:
        dst.resolve().relative_to(base.resolve())
    except ValueError:
        raise AppError("permission_denied", params={"path": str(dst)}, action="none", details="target escapes the game folder", status=400) from None
    return dst


def _rel_target(game_dir: Path, base: Path, rel: Path) -> str:
    return PurePosixPath((base / rel).relative_to(game_dir).as_posix()).as_posix()


COVERAGE_COUNTS = ("dub_lines", "installed", "provisional", "provisional_review", "provisional_regen", "left_out", "narration")


def _count(v: Any) -> int:
    try:
        return max(0, int(v or 0))
    except (TypeError, ValueError):
        return 0


def public_coverage(cov: Any) -> dict[str, Any] | None:
    """The ``coverage`` of ``manifest_patch.json`` (``vndub.stages.pack.patch_coverage``) as the UI gets it: the counts, the
    ``left_out_reasons`` with underscores instead of spaces (``no winner`` -> ``no_winner``: i18n keys ``coverage.reason.<reason>``)
    and no per-line list (``left_out_lines`` stays in the manifest). ``None`` for a patch built before the coverage existed."""
    if not isinstance(cov, dict):
        return None
    out: dict[str, Any] = {k: _count(cov.get(k)) for k in COVERAGE_COUNTS}
    reasons = cov.get("left_out_reasons")
    out["left_out_reasons"] = {str(k).replace(" ", "_"): _count(v) for k, v in reasons.items() if _count(v)} if isinstance(reasons, dict) else {}
    out["include_review_lines"] = str(cov.get("include_review_lines") or "")
    return out


def read_manifest(path: Path | str | None) -> dict[str, Any] | None:
    """``manifest_patch.json`` (the file, or the patch folder that holds it), ``None`` when it is missing or unreadable."""
    if not path:
        return None
    f = Path(path)
    if f.is_dir():
        f = f / "manifest_patch.json"
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def coverage_of(path: Path | str | None) -> dict[str, Any] | None:
    """What the patch covers (:func:`public_coverage`) read from its manifest."""
    data = read_manifest(path)
    return public_coverage(data.get("coverage")) if data else None


def _manifest_summary(patch_dir: Path) -> dict[str, Any] | None:
    data = read_manifest(patch_dir)
    if data is None:
        return None
    backends: dict[str, int] = {}
    routes: dict[str, int] = {}
    for e in data.get("files", []):
        backends[str(e.get("backend") or "-")] = backends.get(str(e.get("backend") or "-"), 0) + 1
        routes[str(e.get("route") or "-")] = routes.get(str(e.get("route") or "-"), 0) + 1
    return {"count": len(data.get("files", [])), "engine": data.get("engine"), "backends": backends, "routes": routes, "coverage": public_coverage(data.get("coverage"))}


def backups_root(slug: str) -> Path:
    return appdirs.backups_dir() / slug


def list_backups(slug: str) -> list[dict[str, Any]]:
    root = backups_root(slug)
    out = []
    if root.is_dir():
        for d in sorted(root.iterdir(), reverse=True):
            m = appdirs.read_json(d / MANIFEST, None)
            if not m or m.get("state") == "rolled_back":  # nothing left to roll back
                continue
            out.append(
                {
                    "id": d.name,
                    "created": m.get("created"),
                    "files": len(m.get("files", [])),
                    "replaced": sum(1 for f in m.get("files", []) if f.get("existed")),
                    "backup": bool(m.get("backup")),
                    "state": m.get("state"),
                    "engine": m.get("engine"),
                }
            )
    return out


def build_plan(slug: str | None = None) -> dict[str, Any]:
    h = _project(slug)
    proj = _project_info(h)
    engine = proj.engine
    game_dir = Path(proj.game_dir) if proj.game_dir else h.game_dir
    pdir = patch_dir_of(h, engine)
    files = patch_files(pdir)
    cls = classify(engine, proj, game_dir, pdir, files)
    base: Path = cls["base"]
    items = []
    total = 0
    replaces = 0
    for f in files:
        rel = f.relative_to(pdir)
        dst = base / rel
        exists = dst.exists()
        size = f.stat().st_size
        total += size
        replaces += 1 if exists else 0
        items.append({"target": _rel_target(game_dir, base, rel) if game_dir.is_dir() else rel.as_posix(), "source": rel.as_posix(), "exists": exists, "size": size})
    items.sort(key=lambda i: (not i["exists"], i["target"]))
    notes: list[dict[str, Any]] = []
    unv = _unvoiced_dir(pdir)
    if unv is not None:
        notes.append({"key": "install.note.unvoiced_manual", "params": {"path": str(unv)}})
    if getattr(proj, "encrypted", False) and engine == "kirikiri":
        notes.append({"key": "install.note.encrypted", "params": {}})
    md = pdir / "INSTALL.md"
    return {
        "slug": h.slug,
        "engine": engine,
        "game_dir": str(game_dir),
        "game_dir_ok": game_dir.is_dir(),
        "patch_dir": str(pdir),
        "auto_possible": bool(cls["auto_possible"]),
        "manual_reason_key": cls["manual_reason_key"],
        "install_md": md.read_text(encoding="utf-8", errors="replace") if md.exists() else "",
        "files": items[:PLAN_FILE_LIMIT],
        "files_truncated": len(items) > PLAN_FILE_LIMIT,
        "file_count": len(items),
        "replace_count": replaces,
        "add_count": len(items) - replaces,
        "total_mb": round(total / 1e6, 2),
        "backups": list_backups(h.slug),
        "manifest": _manifest_summary(pdir),
        "notes": notes,
        "helpers": cls["helpers"],
    }


# ----------------------------------------------------------------------------
# apply / rollback
# ----------------------------------------------------------------------------


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _new_backup_dir(slug: str) -> tuple[str, Path]:
    root = backups_root(slug)
    root.mkdir(parents=True, exist_ok=True)
    base = time.strftime("%Y%m%d-%H%M%S")
    bid, i = base, 1
    while (root / bid).exists():
        i += 1
        bid = f"{base}-{i}"
    d = root / bid
    d.mkdir()
    return bid, d


def _copy_atomic(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".vndub-tmp")
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _missing_dirs(game_dir: Path, dst: Path) -> list[Path]:
    out = []
    p = dst.parent
    while p != game_dir and game_dir in p.parents and not p.exists():
        out.append(p)
        p = p.parent
    return out


def apply_patch(slug: str | None, backup: bool, ctx: JobContext) -> dict[str, Any]:
    """Job body for ``POST /api/install/apply``."""
    plan = build_plan(slug)
    if not plan["auto_possible"]:
        raise _manual_only(plan)
    h = _project(slug)
    proj = _project_info(h)
    game_dir = Path(plan["game_dir"])
    pdir = Path(plan["patch_dir"])
    files = patch_files(pdir)
    if not files:
        raise AppError("nothing_to_install", params={"engine": plan["engine"]}, status=409)
    base, _ = _target_base(proj.engine, game_dir)
    todo = [(f, _safe_target(base, f.relative_to(pdir))) for f in files]

    # room for the backup (data drive) and for the growth of the game folder
    replaced_bytes = sum(d.stat().st_size for _s, d in todo if d.exists())
    growth = sum(max(0, s.stat().st_size - (d.stat().st_size if d.exists() else 0)) for s, d in todo)
    if backup and downloads.free_space_gb(appdirs.backups_dir()) * 1e9 < replaced_bytes * 1.05:
        raise AppError("disk_full", params={"need_gb": round(replaced_bytes * 1.05 / 1e9, 1), "free_gb": round(downloads.free_space_gb(appdirs.backups_dir()), 1), "path": str(appdirs.backups_dir())}, action="open_data_dir")
    if downloads.free_space_gb(game_dir) * 1e9 < growth * 1.02:
        raise AppError("disk_full", params={"need_gb": round(growth * 1.02 / 1e9, 1), "free_gb": round(downloads.free_space_gb(game_dir), 1), "path": str(game_dir)}, action="none")

    bid, bdir = _new_backup_dir(h.slug)
    manifest: dict[str, Any] = {
        "id": bid, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "engine": proj.engine, "game_dir": str(game_dir), "base": str(base),
        "backup": bool(backup), "state": "installing", "files": [], "created_dirs": [],
    }
    created_dirs: set[Path] = set()
    ctx.message("install.msg.backup" if backup else "install.msg.copy", count=len(todo))
    n = len(todo)
    for i, (src, dst) in enumerate(todo):
        ctx.progress(i, n * 2 if backup else n, dst.name)
        existed = dst.exists()
        entry: dict[str, Any] = {"target": dst.relative_to(game_dir).as_posix(), "existed": existed, "backup_file": None}
        if existed and backup:
            bf = bdir / "files" / dst.relative_to(game_dir)
            bf.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dst, bf)
            entry["backup_file"] = bf.relative_to(bdir / "files").as_posix()
            entry["sha256_before"] = _sha256(dst)
        created_dirs.update(_missing_dirs(game_dir, dst))
        manifest["files"].append(entry)
    manifest["created_dirs"] = sorted((p.relative_to(game_dir).as_posix() for p in created_dirs), key=len)
    appdirs.write_json_atomic(bdir / MANIFEST, manifest)  # before the first game file is touched

    ctx.message("install.msg.copy", count=n)
    written = 0
    try:
        for i, (src, dst) in enumerate(todo):
            ctx.progress(n + i if backup else i, n * 2 if backup else n, dst.name)
            _copy_atomic(src, dst)
            written += 1
    except BaseException as e:
        log.warning("install failed after %d/%d files (%s): rolling back", written, n, e)
        ctx.log(f"ERROR install failed after {written}/{n} files: {e}; rolling back")
        try:
            _restore(manifest, bdir, game_dir)
            manifest["state"] = "rolled_back"
            appdirs.write_json_atomic(bdir / MANIFEST, manifest)
        except Exception as e2:  # pragma: no cover - best effort
            ctx.log(f"ERROR rollback after failed install also failed: {e2}")
        if isinstance(e, PermissionError):
            raise AppError("permission_denied", params={"path": str(getattr(e, "filename", "") or game_dir)}, action="show_manual", details=f"{type(e).__name__}: {e}") from e
        raise
    manifest["state"] = "installed"
    manifest["installed"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    appdirs.write_json_atomic(bdir / MANIFEST, manifest)
    replaced = sum(1 for f in manifest["files"] if f["existed"])
    ctx.progress(n * 2 if backup else n, n * 2 if backup else n, None)
    return {"backup_id": bid, "backup": bool(backup), "files": n, "replaced": replaced, "added": n - replaced, "game_dir": str(game_dir)}


def _manual_only(plan: dict[str, Any]) -> AppError:
    """The error for a plan that cannot be applied automatically (never a success, never a job)."""
    reason = plan.get("manual_reason_key")
    if reason == "install.manual.no_patch":
        return AppError("nothing_to_install", params={"engine": plan["engine"]}, action="none", status=409)
    if reason == "install.manual.game_missing":
        return AppError("game_dir_invalid", params={"path": plan["game_dir"]}, action="choose_game_dir", status=409)
    return AppError(
        "install_manual_only",
        params={"engine": plan["engine"], "reason_key": plan["manual_reason_key"], "patch_dir": plan["patch_dir"]},
        action="show_manual",
        details=f"engine {plan['engine']}: {plan['manual_reason_key']}",
        status=409,
    )


def _restore(manifest: dict[str, Any], bdir: Path, game_dir: Path) -> dict[str, int]:
    restored = removed = unrecoverable = 0
    for f in manifest.get("files", []):
        dst = game_dir / f["target"]
        if f.get("existed"):
            bf = f.get("backup_file")
            if bf and (bdir / "files" / bf).is_file():
                _copy_atomic(bdir / "files" / bf, dst)
                restored += 1
            else:
                unrecoverable += 1  # replaced without a backup: the original cannot come back
        else:
            try:
                if dst.exists():
                    dst.unlink()
                    removed += 1
            except OSError as e:
                log.warning("cannot remove %s: %s", dst, e)
    for d in sorted(manifest.get("created_dirs", []), key=len, reverse=True):
        p = game_dir / d
        try:
            if p.is_dir() and not any(p.iterdir()):
                p.rmdir()
        except OSError:
            pass
    return {"restored": restored, "removed": removed, "unrecoverable": unrecoverable}


def rollback_patch(slug: str | None, backup_id: str, ctx: JobContext) -> dict[str, Any]:
    """Job body for ``POST /api/install/rollback``."""
    h = _project(slug)
    if not backup_id or "/" in backup_id or "\\" in backup_id or backup_id.startswith("."):
        raise AppError("bad_request", params={"field": "backup_id", "value": backup_id}, status=422)
    bdir = backups_root(h.slug) / backup_id
    manifest = appdirs.read_json(bdir / MANIFEST, None)
    if not manifest:
        raise AppError("file_not_found", params={"path": str(bdir / MANIFEST)}, action="none", details="backup not found", status=404)
    game_dir = Path(manifest["game_dir"])
    if not game_dir.is_dir():
        raise AppError("game_dir_invalid", params={"path": str(game_dir)}, action="choose_game_dir", status=409)
    if manifest.get("state") == "rolled_back":
        raise AppError("nothing_to_install", params={"reason": "already_rolled_back", "backup_id": backup_id}, status=409)
    ctx.message("install.msg.rollback", count=len(manifest.get("files", [])))
    ctx.progress(0, 1, None)
    res = _restore(manifest, bdir, game_dir)
    manifest["state"] = "rolled_back"
    manifest["rolled_back"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    appdirs.write_json_atomic(bdir / MANIFEST, manifest)
    ctx.progress(1, 1, None)
    return {"backup_id": backup_id, **res}


# ----------------------------------------------------------------------------
# zip for a friend, open folder
# ----------------------------------------------------------------------------


def build_zip(slug: str | None, ctx: JobContext) -> dict[str, Any]:
    """Job body for ``POST /api/install/zip``: the patch folder + INSTALL.md (no originals, no local manifest)."""
    h = _project(slug)
    proj = _project_info(h)
    pdir = patch_dir_of(h, proj.engine)
    if not pdir.is_dir():
        raise AppError("nothing_to_install", params={"engine": proj.engine}, status=409)
    members: list[tuple[Path, str]] = []
    for root, dirs, files in os.walk(pdir):
        rel_root = Path(root).relative_to(pdir)
        if rel_root == Path("."):
            dirs[:] = sorted(d for d in dirs if not d.startswith(("_", ".")))
        for f in sorted(files):
            if f.startswith(".") or f.endswith(".vndub-tmp") or (rel_root == Path(".") and f == "manifest_patch.json"):
                continue
            members.append((Path(root) / f, (rel_root / f).as_posix()))
    if not members:
        raise AppError("nothing_to_install", params={"engine": proj.engine}, status=409)
    h.out.mkdir(parents=True, exist_ok=True)
    out = h.out / f"{h.slug}-{_engine_out(proj.engine)}-voice-patch.zip"
    tmp = out.with_name(out.name + ".tmp")
    ctx.message("install.msg.zip", count=len(members))
    total = sum(s.stat().st_size for s, _ in members)
    done = 0
    with zipfile.ZipFile(tmp, "w", allowZip64=True) as z:
        for src, arc in members:
            ctx.checkpoint()
            comp = zipfile.ZIP_STORED if src.suffix.lower() in STORED_EXTS else zipfile.ZIP_DEFLATED
            z.write(src, f"vndub-voice-patch/{arc}", compress_type=comp)
            done += src.stat().st_size
            ctx.progress(done, max(total, 1), arc)
    os.replace(tmp, out)
    return {"path": str(out), "files": len(members), "size_mb": round(out.stat().st_size / 1e6, 2)}


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------


class ApplyBody(BaseModel):
    backup: bool = True


class RollbackBody(BaseModel):
    backup_id: str


@router.get("/api/install/plan")
def api_plan(slug: str | None = None) -> dict[str, Any]:
    return build_plan(slug)


@router.get("/api/patch/coverage")
def api_coverage(slug: str | None = None) -> dict[str, Any]:
    """``{coverage}``: what the built patch covers (:func:`public_coverage`), ``null`` while there is no patch (or one built
    before the coverage existed). The Run screen asks for it when the app is opened on a finished run (no job to read it from)."""
    h = _project(slug)
    try:
        engine = _project_info(h).engine
    except AppError:
        return {"coverage": None}
    return {"coverage": coverage_of(patch_dir_of(h, engine))}


@router.post("/api/install/apply")
def api_apply(body: ApplyBody | None = None, slug: str | None = None) -> dict[str, Any]:
    from vndub_app import jobs

    body = body or ApplyBody()
    plan = build_plan(slug)
    if not plan["auto_possible"]:
        raise _manual_only(plan)  # never a job, never a success
    s = plan["slug"]
    job = jobs.runner().submit("install_apply", lambda ctx: apply_patch(s, body.backup, ctx), params={"backup": body.backup}, project=s, gpu=False)
    return job.info()


@router.post("/api/install/rollback")
def api_rollback(body: RollbackBody, slug: str | None = None) -> dict[str, Any]:
    from vndub_app import jobs

    h = _project(slug)
    job = jobs.runner().submit("install_rollback", lambda ctx: rollback_patch(h.slug, body.backup_id, ctx), params={"backup_id": body.backup_id}, project=h.slug, gpu=False)
    return job.info()


@router.post("/api/install/zip")
def api_zip(slug: str | None = None) -> dict[str, Any]:
    from vndub_app import jobs

    h = _project(slug)
    job = jobs.runner().submit("install_zip", lambda ctx: build_zip(h.slug, ctx), project=h.slug, gpu=False)
    return job.info()


@router.post("/api/install/open-patch")
def api_open_patch(slug: str | None = None) -> dict[str, Any]:
    h = _project(slug)
    proj = _project_info(h)
    pdir = patch_dir_of(h, proj.engine)
    if not pdir.is_dir():
        raise AppError("nothing_to_install", params={"engine": proj.engine}, status=409)
    openers.open_path(pdir)
    return {"ok": True, "path": str(pdir)}

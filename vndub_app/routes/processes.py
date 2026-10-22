"""Backend processes as the process ledger sees them (``vndub_app.winproc``): list them, terminate them.

``GET  /api/system/processes``       ``{job_object: {installed, error}, processes: [{pid, ppid, exe, name, env, role, ours, orphan}]}``
``POST /api/system/processes/kill``  ``{scope: "orphans" | "all"}`` (default ``orphans``) -> ``{killed, pids}``

The ledger lists every running process whose image file lies under the environments folder (with uv's ``_uv``), the tools
folder or, in the packaged app, the bundled ``bin`` / ``git`` folders: the backend adapters (``role`` ``adapter``, ``env`` =
the environment), ``llama-server`` (``llama_server``), other tools (``tool``). ``ours`` = a descendant of this app, ``orphan`` =
nobody who started it is still running (a leftover of a crashed or killed run: it keeps VRAM and file locks).

``scope: "orphans"`` terminates the orphans only. ``scope: "all"`` terminates everything listed, the processes of a running job
included (the user asked for it: the job then fails or restarts its adapter); the interface disables that button while a job
is active. Windows only; elsewhere the list is empty.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel

from vndub_app import installer, winproc

router = APIRouter(prefix="/api/system")


class KillBody(BaseModel):
    scope: Literal["orphans", "all"] = "orphans"


@router.get("/processes")
def list_processes() -> dict[str, Any]:
    procs = winproc.list_processes(installer.process_roots())
    return {"job_object": winproc.job_object_status(), "processes": [p.public() for p in procs]}


@router.post("/processes/kill")
def kill_processes(body: KillBody | None = None) -> dict[str, Any]:
    scope = body.scope if body is not None else "orphans"
    roots = installer.process_roots()
    if scope == "all":
        killed = winproc.kill_all_under(roots, reason="terminated from the interface (all)")
    else:
        killed = winproc.kill_orphans(roots, reason="terminated from the interface (orphans)")
    return {"killed": len(killed), "pids": [p.pid for p in killed]}

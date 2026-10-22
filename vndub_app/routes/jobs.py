"""Jobs routes (API.md section 4)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from vndub_app import jobs

router = APIRouter(prefix="/api")


@router.get("/jobs")
def list_jobs(project: str | None = None, active: int = Query(0)) -> list[dict[str, Any]]:
    return [j.info() for j in jobs.runner().list(project or None, active_only=bool(active))]


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return jobs.runner().get(job_id).info(with_log=True)


@router.get("/jobs/{job_id}/log")
def get_job_log(job_id: str) -> dict[str, Any]:
    """The whole log of a job: ``{path, lines}`` from its log file (newest 5 MB), else the in-memory tail (``path`` null)."""
    return jobs.runner().read_log(job_id)


@router.post("/jobs/{job_id}/pause")
def pause(job_id: str) -> dict[str, Any]:
    return jobs.runner().pause(job_id).info()


@router.post("/jobs/{job_id}/resume")
def resume(job_id: str) -> dict[str, Any]:
    return jobs.runner().resume(job_id).info()


@router.post("/jobs/{job_id}/cancel")
def cancel(job_id: str) -> dict[str, Any]:
    return jobs.runner().cancel(job_id).info()

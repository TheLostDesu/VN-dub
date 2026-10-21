"""Fakes for the Hugging Face link tests: repo metadata (``hf_custom.set_info_client``) and file bytes (``models_api.set_hub``).

``FakeInfo`` / ``FakeHub`` raise exceptions that carry the *names* of the real ``huggingface_hub`` errors
(``RepositoryNotFoundError``, ``GatedRepoError``), which is all ``models_api.hub_error_kind`` looks at.
"""

from __future__ import annotations

import fnmatch
import time
from pathlib import Path
from typing import Any, Callable

from vndub_app.hf_custom import RepoInfo


class RepositoryNotFoundError(Exception):
    pass


class GatedRepoError(RepositoryNotFoundError):
    pass


def repo_info(
    repo: str,
    files: dict[str, int | bytes],
    *,
    license: str | None = "apache-2.0",
    gated: str | bool = False,
    tags: tuple[str, ...] = (),
    pipeline_tag: str | None = "text-generation",
    sha: str = "0123456789abcdef0123456789abcdef01234567",
) -> RepoInfo:
    sized = [(n, v if isinstance(v, int) else len(v)) for n, v in files.items()]
    return RepoInfo(repo=repo, sha=sha, files=sized, license=license, gated=gated, pipeline_tag=pipeline_tag, tags=list(tags))


class FakeInfo:
    """``hf_custom.InfoClient``: ``infos`` maps a repo id to a :class:`RepoInfo`; ``errors`` raise instead."""

    def __init__(self, infos: dict[str, RepoInfo] | None = None, errors: dict[str, Exception] | None = None) -> None:
        self.infos = dict(infos or {})
        self.errors = dict(errors or {})
        self.asked: list[tuple[str, str | None]] = []

    def info(self, repo: str, revision: str | None) -> RepoInfo:
        self.asked.append((repo, revision))
        if repo in self.errors:
            raise self.errors[repo]
        if repo not in self.infos:
            raise RepositoryNotFoundError(f"Repository Not Found: {repo}")
        return self.infos[repo]


class FakeHub:
    """``models_api.HubClient`` over ``repos = {repo: {filename: bytes}}``.

    ``fail_on={"repo/file": exc}`` raises once for that file; ``slow=True`` sleeps between the progress calls so the
    job's 0.3 s progress throttle lets a cancel request through; ``on_fetch(repo, filename)`` runs before the bytes.
    """

    def __init__(
        self,
        repos: dict[str, dict[str, bytes]],
        *,
        fail_on: dict[str, Exception] | None = None,
        slow: bool = False,
        on_fetch: Callable[[str, str], None] | None = None,
    ) -> None:
        self.repos = repos
        self.fail_on = dict(fail_on or {})
        self.slow = slow
        self.on_fetch = on_fetch
        self.fetched: list[tuple[str, str]] = []

    def list_files(self, repo: str, revision: str | None, allow: tuple[str, ...]) -> list[tuple[str, int]]:
        if repo not in self.repos:
            raise OSError(f"Connection refused: {repo}")
        return [(n, len(b)) for n, b in self.repos[repo].items() if not allow or any(fnmatch.fnmatch(n, p) for p in allow)]

    def fetch(self, repo: str, filename: str, revision: str | None, local: Path, progress: Callable[[int, int], None]) -> None:
        key = f"{repo}/{filename}"
        if self.on_fetch:
            self.on_fetch(repo, filename)
        if key in self.fail_on:
            raise self.fail_on.pop(key)
        data = self.repos[repo][filename]
        out = local / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        steps = 4
        progress(0, len(data))
        for i in range(1, steps + 1):
            if self.slow:
                time.sleep(0.4)
            progress(len(data) * i // steps, len(data))
        out.write_bytes(data)
        self.fetched.append((repo, filename))


def gguf(n: int = 64) -> bytes:
    """A few bytes that look like a GGUF file."""
    return (b"GGUF" + bytes(range(256)) * ((n // 256) + 1))[:n]


def ok_flags(job: dict[str, Any]) -> str:
    return f"{job['state']} {job.get('error')}"

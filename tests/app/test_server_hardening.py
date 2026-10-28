"""Release 1.0 fixes in ``server.py``: one handler for ``app.log`` (rollover loses nothing) and the pictures of the in-app help."""

from __future__ import annotations

import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import pytest

from vndub_app import appdirs, server

NAMES = ("vndub_app", "vndub", "uvicorn.error")
REPO = Path(__file__).resolve().parents[2]


def marked(name: str) -> list[logging.Handler]:
    return [h for h in logging.getLogger(name).handlers if getattr(h, server._FILE_HANDLER_MARK, False)]  # noqa: SLF001


# ---- app.log: one handler, a rollover that loses nothing -----------------------------------------------------------------------


def test_the_three_loggers_share_one_handler(app_env: Path) -> None:
    path = server.setup_logging(logging.INFO)
    handlers = [marked(n) for n in NAMES]
    assert all(len(h) == 1 for h in handlers) and handlers[0][0] is handlers[1][0] is handlers[2][0]
    shared = handlers[0][0]
    assert isinstance(shared, RotatingFileHandler) and Path(shared.baseFilename) == path
    assert server.setup_logging(logging.DEBUG) == path  # again: the same handler with another level, no second one
    assert all(marked(n) == [shared] for n in NAMES) and shared.level == logging.DEBUG
    assert all(logging.getLogger(n).level == logging.DEBUG for n in NAMES)
    server.close_logging()
    assert all(marked(n) == [] for n in NAMES) and shared.stream is None


def test_handlers_of_the_old_layout_are_merged_into_one(app_env: Path) -> None:
    """A handler per logger on the same file (what an earlier version did) is replaced, so a rollover can rename the file."""
    path = appdirs.logs_dir() / "app.log"
    old = []
    for n in NAMES:
        h = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
        setattr(h, server._FILE_HANDLER_MARK, True)  # noqa: SLF001
        logging.getLogger(n).addHandler(h)
        old.append(h)
    server.setup_logging(logging.INFO)
    kept = [marked(n) for n in NAMES]
    assert all(len(k) == 1 for k in kept) and kept[0][0] is kept[1][0] is kept[2][0] is old[0]
    assert old[1].stream is None and old[2].stream is None  # the spares are closed: they no longer hold app.log open


def test_a_new_data_folder_gets_its_own_handler_on_all_three_loggers(app_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server.setup_logging(logging.INFO)
    first = marked("vndub_app")[0]
    other = tmp_path / "other-data"
    other.mkdir()
    monkeypatch.setenv("VNDUB_DATA", str(other))
    path = server.setup_logging(logging.INFO)
    assert path == other / "logs" / "app.log"
    second = [marked(n) for n in NAMES]
    assert all(len(s) == 1 and s[0] is second[0][0] and s[0] is not first for s in second)
    assert first.stream is None  # the old folder's log file is released


def test_a_rollover_with_all_three_loggers_writing_loses_no_record(app_env: Path) -> None:
    """With a handler per logger the rename of app.log failed (the other handlers kept the file open): records were dropped."""
    path = server.setup_logging(logging.INFO)
    shared = marked("vndub_app")[0]
    assert isinstance(shared, RotatingFileHandler)
    shared.maxBytes = 3_000  # tiny, so that the test rolls over many times
    shared.backupCount = 50
    problems: list[Any] = []
    shared.handleError = lambda record: problems.append(record.getMessage())  # type: ignore[method-assign]
    total = 300
    for i in range(total):
        logging.getLogger(NAMES[i % 3]).info("record %04d %s", i, "x" * 40)
    shared.flush()
    server.close_logging()
    files = sorted(path.parent.glob("app.log*"))
    assert len(files) > 5, files  # it did roll over
    seen = [int(m.group(1)) for f in files for m in re.finditer(r"record (\d{4}) ", f.read_text(encoding="utf-8"))]
    assert problems == []
    assert sorted(seen) == list(range(total))  # every record is on disk, once


# ---- the pictures of the in-app help ---------------------------------------------------------------------------------------------


def test_the_help_pictures_are_served_where_the_page_asks_for_them(client: Any) -> None:
    """``docs/APP.md`` writes ``![..](img/app/00-doctor.png)``; the page is ``/`` (HashRouter), so the browser asks for ``/img/app/00-doctor.png``."""
    src = REPO / "docs" / "img" / "app" / "00-doctor.png"
    r = client.get("/img/app/00-doctor.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png" and r.content == src.read_bytes()
    md = (REPO / "docs" / "APP.md").read_text(encoding="utf-8")
    pictures = re.findall(r"\]\((img/[^)\s]+)\)", md)
    assert len(pictures) >= 10
    for rel in pictures:  # every picture the guide shows
        got = client.get("/" + rel)
        assert got.status_code == 200 and got.headers["content-type"].startswith("image/"), rel


@pytest.mark.parametrize(
    "path",
    [
        "/img/app/missing.png",  # not there
        "/img/%2e%2e/APP.md",  # up and out of the pictures folder
        "/img/app/%2e%2e/%2e%2e/APP.md",
        "/img/app/..%2f..%2fAPP.md",
        "/img/app/..%5c..%5cAPP.md",
        "/img/app/%2e%2e%5c%2e%2e%5cAPP.md",
        "/img//app/00-doctor.png",  # an empty segment
        "/img/C:/Windows/win.ini",  # a drive
        "/img/app",  # a folder
        "/img/app/",
        "/img/",
        "/img/app/00-doctor.png.txt",  # not a picture type
        "/img/../README.md",
    ],
)
def test_nothing_but_pictures_below_docs_img_is_served(client: Any, path: str) -> None:
    r = client.get(path)
    # either the picture route refuses (404) or, for a path the client normalised away from /img, the SPA answers: never the file
    assert r.status_code == 404 or "text/html" in r.headers.get("content-type", ""), (path, r.status_code)
    assert not r.headers.get("content-type", "").startswith(("image/", "text/markdown", "text/plain"))


def test_svg_and_other_types_are_not_served_even_when_the_file_exists(client: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    img = tmp_path / "docs" / "img"
    img.mkdir(parents=True)
    (img / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (img / "evil.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'><script>alert(1)</script></svg>", encoding="utf-8")
    (img / "notes.md").write_text("# secret", encoding="utf-8")
    monkeypatch.setattr(appdirs, "resource", lambda *parts: tmp_path.joinpath(*parts))
    assert client.get("/img/a.png").status_code == 200
    assert client.get("/img/evil.svg").status_code == 404
    assert client.get("/img/notes.md").status_code == 404


def test_other_paths_still_get_the_app_and_the_api_is_untouched(client: Any) -> None:
    r = client.get("/some/page")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert client.get("/api/health").json()["ok"] is True
    assert client.get("/api/help").status_code == 200

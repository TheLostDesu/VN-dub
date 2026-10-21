"""The built interface is in the repository and matches its sources (``vndub_app/webbuild.py``)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from vndub_app import webbuild


def test_committed_build_is_fresh() -> None:
    st = webbuild.status()
    assert st["built"], "vndub_app/web/dist is missing: run `npm run build` in vndub_app/web"
    assert st["stamp"] and st["stamp"]["files"] > 20
    assert st["fresh"], "the interface sources changed after the last build: run `npm run build` in vndub_app/web and keep dist/"


def _mini(tmp_path: Path) -> Path:
    web = tmp_path / "vndub_app" / "web"
    (web / "src").mkdir(parents=True)
    (web / "src" / "main.tsx").write_text("console.log(1);\n", encoding="utf-8")
    (web / "package.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "app" / "i18n").mkdir(parents=True)
    (tmp_path / "app" / "i18n" / "ru.json").write_text("{}\n", encoding="utf-8")
    (web / "dist").mkdir()
    (web / "dist" / "index.html").write_text("<html></html>", encoding="utf-8")
    return web


def test_hash_ignores_line_endings_and_sees_edits(tmp_path: Path) -> None:
    web = _mini(tmp_path)
    h = webbuild.sources_hash(web)
    (web / "src" / "main.tsx").write_bytes(b"console.log(1);\r\n")
    assert webbuild.sources_hash(web) == h, "CRLF checkouts hash the same"
    (web / "src" / "main.tsx").write_text("console.log(2);\n", encoding="utf-8")
    assert webbuild.sources_hash(web) != h
    (tmp_path / "app" / "i18n" / "ru.json").write_text('{"a": "b"}\n', encoding="utf-8")
    assert webbuild.sources_hash(web) != h, "the i18n tables are bundled: they count"


def test_status_and_ensure_built(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    web = _mini(tmp_path)
    assert webbuild.status(web)["fresh"] is False  # no stamp yet
    calls: list[Path] = []

    def fake_build(w: Path, npm: str | None = None) -> None:
        calls.append(w)
        (w / "dist" / webbuild.STAMP).write_text('{"sources_sha256": "%s", "files": 3}' % webbuild.sources_hash(w), encoding="utf-8")

    monkeypatch.setattr(webbuild, "find_npm", lambda: "npm")
    monkeypatch.setattr(webbuild, "build", fake_build)
    assert webbuild.ensure_built(web)["fresh"] is True and calls == [web]
    assert webbuild.ensure_built(web)["fresh"] is True and calls == [web], "a fresh build is not rebuilt"
    (web / "src" / "main.tsx").write_text("console.log(3);\n", encoding="utf-8")
    monkeypatch.setattr(webbuild, "find_npm", lambda: None)
    st = webbuild.ensure_built(web)
    assert st["fresh"] is False and st["built"] and len(calls) == 1, "no Node.js: the old build is served"
    shutil.rmtree(web / "src")
    assert webbuild.status(web)["fresh"] is None, "no sources (packaged exe): nothing to compare"


@pytest.mark.skipif(webbuild.find_npm() is None, reason="Node.js is not installed")
def test_stamp_script_matches_python(tmp_path: Path) -> None:
    import subprocess

    web = _mini(tmp_path)
    (web / "scripts").mkdir()
    shutil.copyfile(webbuild.WEB / "scripts" / "stamp.mjs", web / "scripts" / "stamp.mjs")
    (web / "src" / "sub").mkdir()
    (web / "src" / "sub" / "a.ts").write_bytes("export const ы = 1;\r\n".encode("utf-8"))
    node = Path(webbuild.find_npm()).with_name("node.exe" if Path(webbuild.find_npm()).suffix else "node")
    subprocess.run([str(node), "scripts/stamp.mjs"], cwd=web, check=True, capture_output=True)
    assert webbuild.read_stamp(web)["sources_sha256"] == webbuild.sources_hash(web)

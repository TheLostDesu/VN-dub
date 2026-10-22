"""Installer: resumable downloads (local HTTP server), sha256, archives, tools.local.yaml, envs."""

from __future__ import annotations

import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Iterator

import pytest
import yaml
from o_helpers import RangeServer, client_for, drop_runner, free_port, fresh_runner, make_env, sha, tool_entry, wait_job, write_manifest, zip_bytes

from vndub_app import appdirs, doctor, downloads, installer
from vndub_app.errors import AppError

FAKE_EXE = b"MZ\x90\x00 fake rubberband " + bytes(range(256)) * 40
ARCHIVE = zip_bytes({"rb-1.0/rubberband.exe": FAKE_EXE, "rb-1.0/README.txt": b"hello"})
BIG = os.urandom(300_000) + b"tail"


@pytest.fixture
def o_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    data = make_env(monkeypatch, tmp_path)
    fresh_runner()
    # install_envs / ensure_pytools first make sure of a managed Python (uv python install): not in these tests
    monkeypatch.setattr(installer, "ensure_python", lambda version, ctx: Path("python"))
    yield data
    installer.set_manifest_override(None)
    doctor.set_probes(None)
    drop_runner()


def no_sleep(_s: float) -> None:
    return None


def install(names: list[str]) -> dict[str, Any]:
    from vndub_app import jobs

    job = jobs.runner().submit("tools_install", lambda ctx: installer.install_tools(names, ctx), gpu=False)
    return wait_job(job.id)


def use_manifest(tmp_path: Path, *entries: dict[str, Any]) -> None:
    installer.set_manifest_override(write_manifest(tmp_path / "manifest.json", list(entries)))


def rubberband_entry(url: str, **kw: Any) -> dict[str, Any]:
    return tool_entry("rubberband", url, "rb.zip", ARCHIVE, provides={"rubberband": "rb-1.0/rubberband.exe"}, **kw)


# ---- downloads: resume, checksum, unreachable ---------------------------------------------------------------


def test_fetch_downloads_verifies_and_renames(tmp_path: Path) -> None:
    with RangeServer({"big.bin": BIG}) as srv:
        seen: list[tuple[int, int]] = []
        res = downloads.fetch(srv.url("big.bin"), tmp_path / "big.bin", expected_sha256=sha(BIG), expected_size=len(BIG), progress=lambda d, t: seen.append((d, t)), chunk=64 * 1024)
    assert (tmp_path / "big.bin").read_bytes() == BIG
    assert res.verified and res.sha256 == sha(BIG) and res.size == len(BIG) and not res.skipped
    assert not (tmp_path / "big.bin.part").exists() and not (tmp_path / "big.bin.part.json").exists()
    assert seen[-1] == (len(BIG), len(BIG)) and len(seen) > 3 and all(b >= a for (a, _), (b, _) in zip(seen, seen[1:]))


def test_fetch_resumes_from_a_part_file_with_a_range_request(tmp_path: Path) -> None:
    half = len(BIG) // 2
    with RangeServer({"big.bin": BIG}) as srv:
        dest = tmp_path / "big.bin"
        (tmp_path / "big.bin.part").write_bytes(BIG[:half])
        (tmp_path / "big.bin.part.json").write_text(json.dumps({"url": srv.url("big.bin"), "total": len(BIG)}), encoding="utf-8")
        res = downloads.fetch(srv.url("big.bin"), dest, expected_sha256=sha(BIG), chunk=32 * 1024)
        assert srv.requests == [("big.bin", f"bytes={half}-")]
    assert res.resumed_from == half and dest.read_bytes() == BIG and res.verified


def test_fetch_survives_a_dropped_connection_by_resuming(tmp_path: Path) -> None:
    cut = 100_000
    with RangeServer({"big.bin": BIG}, drop_first={"big.bin": cut}) as srv:
        res = downloads.fetch(srv.url("big.bin"), tmp_path / "big.bin", expected_sha256=sha(BIG), chunk=16 * 1024, sleep=no_sleep)
        assert len(srv.requests) == 2 and srv.requests[0][1] is None
        assert srv.requests[1][1] == f"bytes={cut}-"
    assert (tmp_path / "big.bin").read_bytes() == BIG and res.verified


def test_fetch_restarts_when_the_server_ignores_range(tmp_path: Path) -> None:
    half = 5000
    with RangeServer({"big.bin": BIG}, ignore_range=True) as srv:
        (tmp_path / "big.bin.part").write_bytes(BIG[:half])
        res = downloads.fetch(srv.url("big.bin"), tmp_path / "big.bin", expected_sha256=sha(BIG))
    assert (tmp_path / "big.bin").read_bytes() == BIG and res.resumed_from == 0


def test_fetch_checksum_mismatch_deletes_the_file(tmp_path: Path) -> None:
    with RangeServer({"big.bin": BIG}) as srv, pytest.raises(AppError) as ei:
        downloads.fetch(srv.url("big.bin"), tmp_path / "big.bin", expected_sha256="0" * 64)
    e = ei.value
    assert e.code == "checksum_mismatch" and e.action == "retry"
    assert e.params["filename"] == "big.bin" and e.params["expected"] == "0" * 64 and e.params["actual"] == sha(BIG)
    assert list(tmp_path.iterdir()) == []


def test_fetch_wrong_size_is_a_checksum_error_too(tmp_path: Path) -> None:
    with RangeServer({"big.bin": BIG}) as srv, pytest.raises(AppError) as ei:
        downloads.fetch(srv.url("big.bin"), tmp_path / "big.bin", expected_size=len(BIG) + 1)
    assert ei.value.code == "checksum_mismatch"


@pytest.mark.parametrize("how", ["closed_port", "404", "503", "scheme", "dns"])
def test_fetch_unreachable_is_a_manual_download_card(tmp_path: Path, how: str) -> None:
    target = tmp_path / "t"
    srv_ctx = RangeServer({"x.bin": b"1"}, status={"gone.bin": 503})
    with srv_ctx as srv:
        url = {
            "closed_port": f"http://127.0.0.1:{free_port()}/x.bin",
            "404": srv.url("nothing.bin"),
            "503": srv.url("gone.bin"),
            "scheme": "ftp://example.invalid/x.bin",
            "dns": "http://no-such-host.invalid/x.bin",
        }[how]
        with pytest.raises(AppError) as ei:
            downloads.fetch(url, target / "x.bin", target_dir=target, retries=1, sleep=no_sleep, timeout=3.0)
    e = ei.value
    assert e.code == "download_unreachable" and e.action == "show_manual"
    assert e.params == {"url": url, "filename": "x.bin", "target_dir": str(target)}
    assert not (target / "x.bin").exists()


def test_fetch_file_url_and_existing_file(tmp_path: Path) -> None:
    src = tmp_path / "src.bin"
    src.write_bytes(BIG)
    dest = tmp_path / "out" / "copy.bin"
    res = downloads.fetch(src.as_uri(), dest, expected_sha256=sha(BIG))
    assert dest.read_bytes() == BIG and res.verified
    again = downloads.fetch("http://127.0.0.1:1/never-used", dest, expected_sha256=sha(BIG))  # already valid: no request
    assert again.skipped and again.verified
    dest.write_bytes(b"corrupt")
    fixed = downloads.fetch(src.as_uri(), dest, expected_sha256=sha(BIG))  # invalid: replaced
    assert dest.read_bytes() == BIG and not fixed.skipped


def test_fetch_honours_cancel_and_keeps_the_part_file(tmp_path: Path) -> None:
    from vndub_app.jobs import JobCancelled

    calls = {"n": 0}

    def checkpoint() -> None:
        calls["n"] += 1
        if calls["n"] > 3:
            raise JobCancelled("cancelled by user")

    with RangeServer({"big.bin": BIG}) as srv:
        with pytest.raises(JobCancelled):
            downloads.fetch(srv.url("big.bin"), tmp_path / "big.bin", checkpoint=checkpoint, chunk=8 * 1024)
        part = tmp_path / "big.bin.part"
        kept = part.stat().st_size
        assert part.exists() and 0 < kept < len(BIG)
        res = downloads.fetch(srv.url("big.bin"), tmp_path / "big.bin", expected_sha256=sha(BIG))
        assert srv.requests[-1] == ("big.bin", f"bytes={kept}-")
    assert (tmp_path / "big.bin").read_bytes() == BIG and res.resumed_from == kept


# ---- tool installation jobs ------------------------------------------------------------------------------------


def test_install_tool_end_to_end(o_env: Path, tmp_path: Path) -> None:
    with RangeServer({"rb.zip": ARCHIVE}) as srv:
        use_manifest(tmp_path, rubberband_entry(srv.url("rb.zip")))
        c = client_for(installer)
        before = {t["name"]: t for t in c.get("/api/tools").json()}["rubberband"]
        assert before["present"] is False and before["manual"] is False and before["verified"] is True
        assert before["filename"] == "rb.zip" and before["sha256"] == sha(ARCHIVE) and before["required_for"] == ["*"]
        job = c.post("/api/tools/install", json={"names": ["rubberband"]}).json()
        assert job["kind"] == "tools_install" and job["gpu"] is False
        done = wait_job(job)
    assert done["state"] == "done", done.get("error")
    exe = o_env / "tools" / "rubberband" / "1.0" / "rb-1.0" / "rubberband.exe"
    assert exe.read_bytes() == FAKE_EXE and (exe.parent / "README.txt").exists()
    assert not (exe.parent.parent / "rb.zip").exists()  # the archive is removed after extraction
    cfg = yaml.safe_load((o_env / "configs" / "tools.local.yaml").read_text(encoding="utf-8"))
    assert Path(cfg["tools"]["rubberband"]) == exe
    files = done["result"]["files"]
    assert len(files) == 1 and files[0]["state"] == "done" and files[0]["verified"] is True and files[0]["done"] == files[0]["total"] == len(ARCHIVE)
    assert done["result"]["installed"] == ["rubberband"] and done["result"]["unverified"] == []
    after = {t["name"]: t for t in c.get("/api/tools").json()}["rubberband"]
    assert after["present"] is True and after["path"] == str(exe)
    # the generated project config gets the new path
    from vndub_app.appdirs import create_project

    g = tmp_path / "g"
    g.mkdir()
    h = create_project(g)
    assert yaml.safe_load(h.config_file.read_text(encoding="utf-8"))["tools"]["rubberband"] == str(exe)


def test_second_install_is_a_no_op(o_env: Path, tmp_path: Path) -> None:
    with RangeServer({"rb.zip": ARCHIVE}) as srv:
        use_manifest(tmp_path, rubberband_entry(srv.url("rb.zip")))
        assert install(["rubberband"])["state"] == "done"
        n = len(srv.requests)
        again = install(["rubberband"])
        assert again["state"] == "done" and len(srv.requests) == n
        assert again["result"]["files"][0]["skipped"] is True


def test_install_checksum_mismatch_fails_with_checksum_mismatch(o_env: Path, tmp_path: Path) -> None:
    e = rubberband_entry("http://127.0.0.1:1/x")
    with RangeServer({"rb.zip": ARCHIVE + b"tampered"}) as srv:
        e["url"] = srv.url("rb.zip")
        e["size"] = None
        e["verify"] = "sha256"  # the pinned sha256 is that of the genuine archive
        use_manifest(tmp_path, e)
        done = install(["rubberband"])
    assert done["state"] == "failed" and done["error"]["code"] == "checksum_mismatch"
    assert done["error"]["params"]["expected"] == sha(ARCHIVE) and done["error"]["params"]["actual"] == sha(ARCHIVE + b"tampered")
    assert not (o_env / "tools" / "rubberband" / "1.0" / "rb.zip").exists()
    assert installer.read_local_tools() == {}
    assert done["result"]["files"][0]["state"] == "failed"


def test_install_unreachable_url_gives_the_manual_card_params(o_env: Path, tmp_path: Path) -> None:
    url = f"http://127.0.0.1:{free_port()}/rb.zip"
    use_manifest(tmp_path, rubberband_entry(url))
    done = install(["rubberband"])
    err = done["error"]
    assert done["state"] == "failed" and err["code"] == "download_unreachable" and err["action"] == "show_manual"
    assert err["params"]["url"] == url and err["params"]["filename"] == "rb.zip"
    assert err["params"]["target_dir"] == str(o_env / "tools" / "rubberband" / "1.0")


def test_manual_download_into_the_target_folder_completes_the_install(o_env: Path, tmp_path: Path) -> None:
    url = f"http://127.0.0.1:{free_port()}/rb.zip"
    use_manifest(tmp_path, rubberband_entry(url))
    first = install(["rubberband"])
    target = Path(first["error"]["params"]["target_dir"])
    (target / "rb.zip").write_bytes(ARCHIVE)  # the user saved the file where the card said
    done = install(["rubberband"])
    assert done["state"] == "done", done.get("error")
    assert (target / "rb-1.0" / "rubberband.exe").read_bytes() == FAKE_EXE


def test_a_wrong_file_in_the_target_folder_is_not_trusted(o_env: Path, tmp_path: Path) -> None:
    url = f"http://127.0.0.1:{free_port()}/rb.zip"
    use_manifest(tmp_path, rubberband_entry(url))
    target = o_env / "tools" / "rubberband" / "1.0"
    target.mkdir(parents=True)
    (target / "rb.zip").write_bytes(b"not the pinned archive")
    done = install(["rubberband"])
    assert done["state"] == "failed" and done["error"]["code"] == "download_unreachable"  # it tried to fetch the real one


def test_tools_placed_by_hand_are_discovered(o_env: Path, tmp_path: Path) -> None:
    use_manifest(tmp_path, rubberband_entry("http://127.0.0.1:1/x"))
    exe = o_env / "tools" / "rubberband" / "1.0" / "anywhere" / "deep" / "rubberband.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"MZ")
    row = {t["name"]: t for t in installer.list_tools()}["rubberband"]
    assert row["present"] is True and Path(row["path"]) == exe
    assert installer.tools_object().resolve("rubberband") == exe


def test_manual_entry_placed_by_hand_counts_as_installed(o_env: Path, tmp_path: Path) -> None:
    e = tool_entry("bgi_extractor", "https://example.com/page", "bgi-extractor.exe", b"", provides={"bgi_extractor": "bgi-extractor.exe"}, archive="none", manual=True, engines=["bgi"], size=None, sha256=None, verify="none", unverified_reason="source only")
    use_manifest(tmp_path, e)
    first = install(["bgi_extractor"])
    assert first["state"] == "failed" and first["error"]["code"] == "download_unreachable" and first["error"]["params"]["reason"] == "manual"
    assert {t["name"]: t for t in installer.list_tools()}["bgi_extractor"]["manual"] is True
    (Path(first["error"]["params"]["target_dir"]) / "bgi-extractor.exe").write_bytes(b"MZ built by me")
    done = install(["bgi_extractor"])
    assert done["state"] == "done" and Path(installer.read_local_tools()["bgi_extractor"]).read_bytes() == b"MZ built by me"


def test_single_file_tool_with_unverified_size_only_entry(o_env: Path, tmp_path: Path) -> None:
    data = b"MZ single exe"
    with RangeServer({"Xp3Pack.exe": data}) as srv:
        e = tool_entry("xp3pack", srv.url("Xp3Pack.exe"), "Xp3Pack.exe", data, provides={"xp3pack": "Xp3Pack.exe"}, archive="none", engines=["kirikiri"], sha256=None, verify="size")
        use_manifest(tmp_path, e)
        done = install(["xp3pack"])
    assert done["state"] == "done", done.get("error")
    assert (o_env / "tools" / "xp3pack" / "1.0" / "Xp3Pack.exe").read_bytes() == data
    assert done["result"]["unverified"] == ["xp3pack"]  # size-checked only: reported, never silent


def test_extra_files_land_in_the_same_folder(o_env: Path, tmp_path: Path) -> None:
    dll = b"MZ dll"
    cudart = zip_bytes({"cudart64_12.dll": b"cudart", "cublas64_12.dll": b"cublas"})
    main = zip_bytes({"llama-server.exe": b"MZ server"})
    with RangeServer({"main.zip": main, "cudart.zip": cudart, "version.dll": dll}) as srv:
        e = tool_entry("llama_server", srv.url("main.zip"), "main.zip", main, provides={"llama_server": "llama-server.exe"})
        e["extra"] = [
            {"url": srv.url("cudart.zip"), "filename": "cudart.zip", "archive": "zip", "size": len(cudart), "sha256": sha(cudart), "verify": "sha256"},
            {"url": srv.url("version.dll"), "filename": "version.dll", "archive": "none", "size": len(dll), "sha256": sha(dll), "verify": "sha256"},
        ]
        use_manifest(tmp_path, e)
        done = install(["llama_server"])
    assert done["state"] == "done", done.get("error")
    d = o_env / "tools" / "llama_server" / "1.0"
    assert {p.name for p in d.iterdir() if p.is_file()} >= {"llama-server.exe", "cudart64_12.dll", "cublas64_12.dll", "version.dll"}
    assert len(done["result"]["files"]) == 3


def test_partial_failure_installs_the_rest_and_reports_the_first_error(o_env: Path, tmp_path: Path) -> None:
    other = zip_bytes({"v/vgmstream-cli.exe": b"MZ vgm"})
    with RangeServer({"rb.zip": ARCHIVE, "vgm.zip": other}, status={"vgm.zip": 404}) as srv:
        good = rubberband_entry(srv.url("rb.zip"))
        bad = tool_entry("vgmstream_cli", srv.url("vgm.zip"), "vgm.zip", other, provides={"vgmstream_cli": "v/vgmstream-cli.exe"})
        use_manifest(tmp_path, bad, good)
        done = install(["vgmstream_cli", "rubberband"])
    assert done["state"] == "failed" and done["error"]["code"] == "download_unreachable"
    assert done["result"]["installed"] == ["rubberband"] and [f["name"] for f in done["result"]["failed"]] == ["vgmstream_cli"]
    assert "rubberband" in installer.read_local_tools()


def test_zip_slip_is_refused(o_env: Path, tmp_path: Path) -> None:
    evil = zip_bytes({"../../evil.exe": b"MZ evil", "ok/rubberband.exe": b"MZ"})
    with RangeServer({"rb.zip": evil}) as srv:
        use_manifest(tmp_path, tool_entry("rubberband", srv.url("rb.zip"), "rb.zip", evil, provides={"rubberband": "ok/rubberband.exe"}))
        done = install(["rubberband"])
    assert done["state"] == "failed" and done["error"]["code"] == "tool_install_failed" and done["error"]["params"]["reason"] == "unsafe_archive"
    assert not list(o_env.rglob("evil.exe")) and not (tmp_path / "evil.exe").exists()


def test_bad_archive_and_missing_executable(o_env: Path, tmp_path: Path) -> None:
    junk = b"this is not a zip"
    with RangeServer({"rb.zip": junk}) as srv:
        use_manifest(tmp_path, tool_entry("rubberband", srv.url("rb.zip"), "rb.zip", junk, provides={"rubberband": "rubberband.exe"}))
        done = install(["rubberband"])
    assert done["error"]["code"] == "tool_install_failed" and done["error"]["params"]["reason"] == "bad_archive"
    empty = zip_bytes({"readme.txt": b"nothing useful"})
    with RangeServer({"rb.zip": empty}) as srv:
        use_manifest(tmp_path, tool_entry("rubberband", srv.url("rb.zip"), "rb.zip", empty, provides={"rubberband": "rubberband.exe"}))
        done = install(["rubberband"])
    assert done["error"]["code"] == "tool_install_failed" and done["error"]["params"]["reason"] == "executable_missing"
    assert installer.read_local_tools() == {}


def test_install_refuses_when_the_disk_is_too_small(o_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    use_manifest(tmp_path, rubberband_entry("http://127.0.0.1:1/x"))
    monkeypatch.setattr(downloads, "free_space_gb", lambda p: 0.0)
    done = install(["rubberband"])
    assert done["error"]["code"] == "disk_full" and done["error"]["params"]["free_gb"] == 0.0


def test_endpoint_validation(o_env: Path, tmp_path: Path) -> None:
    use_manifest(tmp_path, rubberband_entry("http://127.0.0.1:1/x"))
    c = client_for(installer)
    r = c.post("/api/tools/install", json={"names": ["nope"]})
    assert r.status_code == 422 and r.json()["error"]["code"] == "bad_request"
    assert c.post("/api/tools/install", json={"names": []}).status_code == 422
    assert c.post("/api/envs/install", json={"envs": ["nope"]}).status_code == 422
    assert c.post("/api/envs/install", json={"envs": ["main"], "torch": "tpu"}).status_code == 422


def test_wrapper_launcher_runs_the_script_with_the_tools_python(o_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = b"import sys\nprint('unrpyc ran', sys.argv[1:])\n"
    arc = zip_bytes({"unrpyc-9/unrpyc.py": script})
    py = Path(sys.executable)
    monkeypatch.setattr(installer, "ensure_pytools", lambda ctx: py)
    with RangeServer({"u.zip": arc}) as srv:
        e = tool_entry("unrpyc", srv.url("u.zip"), "u.zip", arc, provides={"unrpyc": "unrpyc"}, engines=["renpy"], needs_pytools=True, wrapper={"name": "unrpyc", "script": "unrpyc-9/unrpyc.py", "python": "pytools"})
        use_manifest(tmp_path, e)
        done = install(["unrpyc"])
    assert done["state"] == "done", done.get("error")
    launcher = Path(installer.read_local_tools()["unrpyc"])
    assert launcher.name == ("unrpyc.cmd" if os.name == "nt" else "unrpyc")
    import subprocess

    out = subprocess.run([str(launcher), "a.rpyc"], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0 and "unrpyc ran ['a.rpyc']" in out.stdout


def test_pip_entry_uses_uv_with_hashes_and_no_deps(o_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    venv = installer.pytools_dir()
    script = installer._venv_script(venv, "unrpa")
    py = installer._venv_python(venv)
    calls: list[tuple[list[str], str]] = []

    def fake_stream(cmd: list[str], ctx: Any, **kw: Any) -> tuple[int, list[str]]:
        req = Path(cmd[cmd.index("-r") + 1])
        calls.append((cmd, req.read_text(encoding="utf-8")))
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_bytes(b"MZ")
        return 0, []

    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_bytes(b"MZ")
    (venv / "pyvenv.cfg").write_text("home = x\n", encoding="utf-8")
    monkeypatch.setattr(installer, "run_stream", fake_stream)
    monkeypatch.setattr(installer, "find_uv", lambda: Path("uv"))
    e = {
        "name": "unrpa", "version": "2.3.0", "kind": "pip", "platforms": ["win32", "linux", "darwin"], "engines": ["renpy"], "required": False,
        "pip": {"requirement": "unrpa==2.3.0", "hashes": ["sha256:" + "a" * 64, "sha256:" + "b" * 64], "script": "unrpa"},
        "size": None, "sha256": None, "verify": "sha256", "provides": {"unrpa": "unrpa"}, "extra": [],
    }
    use_manifest(tmp_path, e)
    done = install(["unrpa"])
    assert done["state"] == "done", done.get("error")
    cmd, req = calls[0]
    assert cmd[:3] == ["uv", "pip", "install"] and "--no-deps" in cmd and "--require-hashes" in cmd and cmd[cmd.index("--python") + 1] == str(py)
    assert req.strip() == "unrpa==2.3.0 --hash=sha256:" + "a" * 64 + " --hash=sha256:" + "b" * 64
    assert Path(installer.read_local_tools()["unrpa"]) == script
    assert done["result"]["unverified"] == []  # hashes verify pip installs


def test_pip_failure_is_mapped(o_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    venv = installer.pytools_dir()
    py = installer._venv_python(venv)
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_bytes(b"MZ")
    (venv / "pyvenv.cfg").write_text("home = x\n", encoding="utf-8")
    monkeypatch.setattr(installer, "find_uv", lambda: Path("uv"))
    e = {
        "name": "unrpa", "version": "2.3.0", "kind": "pip", "platforms": ["win32", "linux", "darwin"], "engines": ["renpy"], "required": False,
        "pip": {"requirement": "unrpa==2.3.0", "hashes": ["sha256:" + "a" * 64], "script": "unrpa"},
        "size": None, "sha256": None, "verify": "sha256", "provides": {"unrpa": "unrpa"}, "extra": [],
    }
    use_manifest(tmp_path, e)
    monkeypatch.setattr(installer, "run_stream", lambda cmd, ctx, **kw: (2, ["error: Failed to fetch: https://pypi.org/simple/unrpa/", "dns error: no such host"]))
    done = install(["unrpa"])
    assert done["error"]["code"] == "download_unreachable" and done["error"]["action"] == "show_manual"
    monkeypatch.setattr(installer, "run_stream", lambda cmd, ctx, **kw: (2, ["error: THESE PACKAGES DO NOT MATCH THE HASHES from the requirements file"]))
    done = install(["unrpa"])
    assert done["error"]["code"] == "checksum_mismatch"
    monkeypatch.setattr(installer, "run_stream", lambda cmd, ctx, **kw: (1, ["error: something odd"]))
    done = install(["unrpa"])
    assert done["error"]["code"] == "tool_install_failed"


# ---- environments ---------------------------------------------------------------------------------------------------


def test_env_listing_matches_the_requirement_files(o_env: Path) -> None:
    names = installer.env_names()
    assert "main" in names and "funasr" in names and "llm" not in names  # llm.txt installs nothing
    assert not {"asr", "hf", "score"} & set(names)  # merged into main (one torch 2.8 env) on 2026-10-22
    assert all("+" not in n and "." not in n for n in names)
    c = client_for(installer)
    rows = {r["name"]: r for r in c.get("/api/envs").json()}
    assert set(rows) == set(names) and all(r["present"] is False and r["torch"] is None for r in rows.values())
    assert rows["rvc"]["needs_git"] is True and rows["main"]["needs_git"] is False  # main clones only in its optional extras
    assert {"qwen3_tts", "anime_whisper", "utmos22"} <= set(rows["main"]["backends"])  # tts + asr + scoring share the env


def fake_make_envs(tmp_path: Path, *, fail: bool = False):
    """A stand-in for scripts/make_envs.*: prints the same '== env X' headers and creates the env folders."""
    script = tmp_path / "fake_make_envs.py"
    script.write_text(
        "import sys, os, json\nfrom pathlib import Path\n"
        "envs_dir, torch, fail, names = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == '1', sys.argv[4:]\n"
        "for n in names:\n"
        "    print(f'== env {n} ({torch.upper()})', flush=True)\n"
        "    print('   resolving...', flush=True)\n"
        "    if fail:\n"
        "        print('error: Failed to fetch https://download.pytorch.org/whl/cpu: dns error', flush=True); sys.exit(1)\n"
        "    py = envs_dir / n / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')\n"
        "    py.parent.mkdir(parents=True, exist_ok=True); py.write_bytes(b'')\n"
        "    (envs_dir / n / 'pyvenv.cfg').write_text('home = x')\n"
        "    (envs_dir / n / 'vndub_env.json').write_text(json.dumps({'env': n, 'cpu': torch == 'cpu'}))\n"
        "    print(f'{n}: ok', flush=True)\n",
        encoding="utf-8",
    )

    def build(envs: list[str], torch: str, envs_path: Path, python: Path | None = None) -> tuple[list[str], dict[str, str]]:
        return [sys.executable, str(script), str(envs_path), torch, "1" if fail else "0", *envs], {}

    return build


def run_env_job(envs: list[str], torch: str) -> dict[str, Any]:
    from vndub_app import jobs

    job = jobs.runner().submit("envs_install", lambda ctx: installer.install_envs(envs, torch, ctx), gpu=False)
    return wait_job(job.id)


def test_env_install_streams_the_script_output_and_reports_per_env(o_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer, "build_make_envs_command", fake_make_envs(tmp_path))
    monkeypatch.setattr(installer, "find_uv", lambda: Path("uv"))
    done = run_env_job(["main", "funasr"], "cpu")
    assert done["state"] == "done", done.get("error")
    assert done["result"]["envs"] == ["main", "funasr"] and done["result"]["torch"] == "cpu"
    assert [f["state"] for f in done["result"]["files"]] == ["done", "done"]
    log = "\n".join(done["log_tail"])
    assert "== env main (CPU)" in log and "== env funasr (CPU)" in log and "funasr: ok" in log
    assert done["done"] == 2 and done["total"] == 2  # envs are the progress unit
    rows = {r["name"]: r for r in installer.list_envs()}
    assert rows["main"]["present"] and rows["main"]["torch"] == "cpu" and rows["funasr"]["present"] and not rows["chatterbox"]["present"]


def test_env_install_auto_torch_follows_doctor(o_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer, "build_make_envs_command", fake_make_envs(tmp_path))
    monkeypatch.setattr(installer, "find_uv", lambda: Path("uv"))
    doctor.set_probes(doctor.Probes(gpu=lambda: None))
    assert run_env_job(["main"], "auto")["result"]["torch"] == "cpu"
    doctor.set_probes(doctor.Probes(gpu=lambda: doctor.GpuInfo("NVIDIA X", 8.0, "572.16", "12.8")))
    done = run_env_job(["funasr"], "auto")
    assert done["result"]["torch"] == "cuda" and installer.list_envs()[installer.env_names().index("funasr")]["torch"] == "cuda"
    assert run_env_job(["chatterbox"], "cpu")["result"]["torch"] == "cpu"  # explicit choice wins


def test_env_install_failure_is_an_error_with_the_log(o_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer, "find_uv", lambda: Path("uv"))
    monkeypatch.setattr(installer, "build_make_envs_command", fake_make_envs(tmp_path, fail=True))
    done = run_env_job(["main"], "cpu")
    assert done["state"] == "failed" and done["error"]["code"] == "download_unreachable"  # dns error in the output
    assert [f["state"] for f in done["result"]["files"]] == ["failed"]
    monkeypatch.setattr(installer, "run_stream", lambda cmd, ctx, **kw: (1, ["error: build failed for numpy"]))
    done = run_env_job(["main"], "cpu")
    assert done["error"]["code"] == "env_install_failed" and done["error"]["params"]["exit_code"] == 1 and "numpy" in done["error"]["details"]


def test_env_install_needs_git_for_checkout_envs(o_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    real_which = shutil.which
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: None if name == "git" else real_which(name, *a, **k))
    monkeypatch.setattr(installer, "find_uv", lambda: Path("uv"))
    done = run_env_job(["rvc", "main"], "cpu")
    assert done["state"] == "failed" and done["error"]["code"] == "git_missing" and done["error"]["params"]["envs"] == ["rvc"]
    assert done["error"]["action"] == "show_manual"


def test_env_install_needs_uv(o_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer, "find_uv", lambda: None)
    done = run_env_job(["main"], "cpu")
    assert done["error"]["code"] == "tool_missing" and done["error"]["params"]["tool"] == "uv"


def test_make_envs_command_line(o_env: Path) -> None:
    cmd, extra = installer.build_make_envs_command(["main", "funasr"], "cpu", Path("E:/envs"))
    if os.name == "nt":
        assert cmd[0].lower().endswith("powershell.exe") or cmd[0].lower().endswith("powershell")
        assert "-File" in cmd and cmd[cmd.index("-File") + 1].endswith("make_envs.ps1")
        assert cmd[cmd.index("-Env") + 1] == "main,funasr" and cmd[cmd.index("-EnvsDir") + 1] == str(Path("E:/envs")) and "-Cpu" in cmd
        assert "-Cpu" not in installer.build_make_envs_command(["main"], "cuda", Path("E:/envs"))[0]
    else:
        assert cmd[0] == "bash" and cmd[-2:] == ["main", "funasr"] and extra == {"ENVS_DIR": str(Path("E:/envs")), "CPU": "1"}


def test_make_envs_script_accepts_the_envs_dir_parameter() -> None:
    """The scripts take the envs dir (the app keeps its envs in <data>/envs) and still default to <repo>/.venvs."""
    root = Path(__file__).resolve().parents[2] / "scripts"
    ps = (root / "make_envs.ps1").read_text(encoding="utf-8")
    assert "[string]$EnvsDir" in ps and "VNDUB_ENVS_DIR" in ps and '".venvs"' in ps and "[switch]$Cpu" in ps
    sh = (root / "make_envs.sh").read_text(encoding="utf-8")
    assert "ENVS_DIR" in sh and "VNDUB_ENVS_DIR" in sh and "/.venvs" in sh and "CPU" in sh


def test_runner_and_adapters_honour_vndub_envs_dir(o_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from adapters.common import rpc
    from vndub.backends import runner

    envs = o_env.parent / "envs"
    assert runner.envs_root() == envs
    assert runner.env_python("main") == envs / "main" / ("Scripts/python.exe" if sys.platform.startswith("win") else "bin/python")
    explicit = o_env.parent / "checkout"
    assert runner.envs_root(explicit) == explicit / ".venvs"  # an explicit root always means <root>/.venvs
    assert rpc.env_src_dir("rvc", "Applio") == envs / "rvc" / "src" / "Applio"
    monkeypatch.delenv("VNDUB_ENVS_DIR")
    assert runner.envs_root() == Path(runner.__file__).resolve().parents[2] / ".venvs"
    assert rpc.env_src_dir("rvc", "Applio").parts[-4:] == (".venvs", "rvc", "src", "Applio")


def test_adapter_processes_start_the_env_the_installer_created(o_env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The interpreter of an adapter process is looked up where env_exists() and the installer look (regression:
    it was always <repo>/.venvs, so every env the app installed into <data>/envs was "not created")."""
    from vndub.backends import registry, runner

    envs = o_env.parent / "envs"
    exe = "Scripts/python.exe" if sys.platform.startswith("win") else "bin/python"
    assert runner.AdapterProcess("main", "x").python == envs / "main" / exe
    assert runner.AdapterProcess("main", "x").root == runner.repo_root()  # adapters still run from the checkout
    explicit = tmp_path / "checkout"
    assert runner.AdapterProcess("main", "x", root=explicit).python == explicit / ".venvs" / "main" / exe
    seen: dict[str, Any] = {}

    def factory(env: str, name: str, **kw: Any) -> Any:
        seen.update(kw)
        return runner.AdapterProcess(env, name, **kw)

    spec = next(s for s in registry.all_specs().values() if s.env == "main")
    b = runner.RemoteBackend(spec, {}, process_factory=factory)
    assert seen["root"] is None and b.proc.python == envs / "main" / exe


def test_adapters_root_is_a_folder_of_its_own_when_frozen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Packaged app: adapters run from <bundle>/backend, not from the bundle folder whose CPython DLLs and
    extension modules (_ssl.pyd, _ctypes.pyd ...) an env interpreter would load ahead of its own."""
    from vndub.backends import runner

    assert runner.adapters_root() == runner.repo_root()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert runner.adapters_root() == tmp_path / "backend"
    spec = (Path(__file__).resolve().parents[2] / "packaging" / "vndub_app.spec").read_text(encoding="utf-8")
    assert 'tree(ROOT / "adapters", "backend/adapters")' in spec


def test_configure_environment(o_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert installer.configure_environment() == o_env.parent / "envs"  # an explicit setting wins
    monkeypatch.delenv("VNDUB_ENVS_DIR")
    chosen = installer.configure_environment()
    assert os.environ["VNDUB_ENVS_DIR"] == str(chosen)
    assert chosen in (appdirs.envs_dir(), installer.repo_envs_dir())
    monkeypatch.delenv("VNDUB_ENVS_DIR")
    assert installer.envs_dir() == chosen and os.environ["VNDUB_ENVS_DIR"] == str(chosen)  # the getter exports it as well


def test_run_stream_cancel_kills_the_process(o_env: Path, tmp_path: Path) -> None:
    from vndub_app import jobs

    marker = tmp_path / "started"
    code = f"import time, pathlib\nprint('go', flush=True)\npathlib.Path(r'{marker}').write_text('x')\n[print('tick', flush=True) or time.sleep(0.05) for _ in range(2000)]\n"

    def body(ctx: Any) -> None:
        installer.run_stream([sys.executable, "-c", code], ctx)

    r = jobs.runner()
    job = r.submit("envs_install", body, gpu=False)
    t0 = time.monotonic()
    while not marker.exists() and time.monotonic() - t0 < 20:
        time.sleep(0.05)
    r.cancel(job.id)
    done = wait_job(job.id, timeout=20)
    assert done["state"] == "cancelled" and time.monotonic() - t0 < 20
    assert any("tick" in l for l in done["log_tail"])


def test_tools_python_writes_utf8_to_pipes(tmp_path: Path) -> None:
    """Japanese file names / text in a tool's output must not crash it under a cp1251 / cp932 console code page."""
    import subprocess
    import venv

    env_dir = tmp_path / "pytools"
    venv.EnvBuilder(with_pip=False, symlinks=False).create(env_dir)
    py = installer._venv_python(env_dir)
    f = installer.ensure_utf8_output(py)
    assert f is not None and f.name == "sitecustomize.py" and f.parent.name == "site-packages"
    assert installer.ensure_utf8_output(py) == f  # idempotent
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONUTF8", "PYTHONIOENCODING")}
    r = subprocess.run([str(py), "-c", "print('ファイル名 日本語'); import sys; print('ошибка', file=sys.stderr)"], capture_output=True, env=env, timeout=60)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    assert r.stdout.decode("utf-8").strip() == "ファイル名 日本語" and r.stderr.decode("utf-8").strip() == "ошибка"

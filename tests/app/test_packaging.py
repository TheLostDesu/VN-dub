"""Packaging files: PyInstaller spec, Inno Setup script, build script (static checks; the build itself is `packaging/build_app.ps1`)."""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "packaging"


def _iss_text() -> str:
    return (PKG / "VNDub.iss").read_bytes().decode("utf-8-sig")


def test_spec_is_valid_python_and_bundles_what_the_app_needs() -> None:
    text = (PKG / "vndub_app.spec").read_text(encoding="utf-8")
    ast.parse(text)
    for needed in ('"configs"', '"app"', '"docs"', '"adapters"', '"envs"', "make_envs.ps1", "make_envs.sh", "vndub_app/web/dist", "vndub/report/templates", '"bin"', "uv.exe", "README.md"):
        assert needed in text, needed
    assert "console=False" in text and "COLLECT(" in text and 'name="VNDub"' in text
    assert 'collect_submodules("vndub")' in text and 'collect_submodules("vndub_app")' in text and "huggingface_hub" in text
    # heavy ML packages stay out of the light core; what the pipeline imports stays in
    excludes = text.split("excludes = [", 1)[1].split("]", 1)[0]
    for out in ("torch", "gradio", "tensorflow", "onnxruntime", "transformers"):
        assert f'"{out}"' in excludes, out
    for keep in ("numpy", "scipy", "soundfile", "librosa", "pydantic", "yaml", "orjson", "uvicorn", "fastapi", "webview"):
        assert f'"{keep}"' not in excludes, keep


def test_every_resource_the_app_looks_up_is_in_the_bundle_layout() -> None:
    """appdirs.resource(...) targets of the shell's self-check exist in the repo and are covered by the spec."""
    from vndub_app import shell

    text = (PKG / "vndub_app.spec").read_text(encoding="utf-8")
    for parts in shell.REQUIRED_RESOURCES:
        assert (ROOT.joinpath(*parts)).exists(), parts
        assert f'"{parts[0]}"' in text, parts


def test_iss_is_a_complete_inno_script() -> None:
    raw = (PKG / "VNDub.iss").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf"), "UTF-8 BOM: the script has Cyrillic custom messages"
    text = raw.decode("utf-8-sig")
    for section in ("[Setup]", "[Languages]", "[CustomMessages]", "[Tasks]", "[Files]", "[Icons]", "[Run]", "[Code]", "[InstallDelete]"):
        assert section in text, section
    assert "PrivilegesRequired=lowest" in text and r"DefaultDirName={localappdata}\Programs\{#AppName}" in text
    assert 'Name: "russian"; MessagesFile: "compiler:Languages\\Russian.isl"' in text and 'Name: "english"; MessagesFile: "compiler:Default.isl"' in text
    assert "OutputBaseFilename=VNDub-Setup" in text and "OutputDir=..\\installer" in text and r'Source: "..\dist\VNDub\*"' in text
    assert "{autoprograms}" in text and "{autodesktop}" in text and "Flags: unchecked" in text  # Start menu + optional desktop shortcut
    # uninstall keeps the data folder unless the user says so
    for needed in ("InitializeUninstall", "CurUninstallStepChanged", "usUninstall", "usPostUninstall", "DelTree", "DELETEDATA", "location.json", "VNDUB_DATA"):
        assert needed in text, needed
    # every Pascal begin has an end (a rough balance check; the real check is the compile test below)
    code = text.split("[Code]", 1)[1]
    begins = len(re.findall(r"\bbegin\b", code))
    ends = len(re.findall(r"\bend\b", code))
    assert begins == ends, (begins, ends)
    for msg in ("DeleteDataInstruction", "DeleteDataText", "DeleteDataKeep", "DeleteDataDelete"):
        assert f"russian.{msg}=" in text and f"english.{msg}=" in text


def test_iss_installer_contract() -> None:
    """Per-user install, stable AppId, UTF-8 messages, and an uninstall that keeps the data unless told otherwise."""
    text = _iss_text()
    # one AppId forever: upgrades install over the old build and share one uninstall entry
    assert "AppId={{6F2D8E54-9C3B-4A71-B0E5-3D7A1C95F8A2}" in text
    # per-user: no UAC, program in %LOCALAPPDATA%\Programs\VNDub, no switch to an all-users install
    assert "PrivilegesRequired=lowest" in text and "PrivilegesRequiredOverridesAllowed" not in text
    assert r"DefaultDirName={localappdata}\Programs\{#AppName}" in text
    # 64-bit app; "x64" alone is deprecated since Inno Setup 6.3 (a compile warning)
    assert "ArchitecturesAllowed=x64compatible" in text and "ArchitecturesInstallIn64BitMode=x64compatible" in text
    assert re.search(r"^#if Ver < EncodeVer\(6, 3, 0\)", text, re.M), "the script needs 6.3+ (UTF8Decode, x64compatible)"
    # the data question: Yes (= keep) is the first button and the answer of a suppressed box; only "No" deletes
    assert "{param:DELETEDATA|0}" in text and "UninstallSilent" in text
    assert "SuppressibleTaskDialogMsgBox(" in text and "MB_YESNO" in text and "IDYES) = IDNO" in text
    keep, delete = text.index("Labels[0] := CustomMessage('DeleteDataKeep')"), text.index("Labels[1] := CustomMessage('DeleteDataDelete')")
    assert keep < delete
    # data removal never takes the whole folder (the user may have pointed it at a folder with other files)
    code = text.split("[Code]", 1)[1]
    assert not re.search(r"DelTree\(\s*(DataDir|Dir)\s*,", code), "DelTree of the whole data folder"
    assert "RemoveDir(Dir)" in code
    # a running VNDub is closed before the uninstall removes its files (Setup does that itself via CloseApplications);
    # only the copy that runs from the install folder, never another VNDub.exe
    assert "CloseApplications=yes" in text and "StopRunningApp();" in text
    assert "CloseMainWindow" in text and "ExecutablePath -eq $Exe" in text and "taskkill" not in text.lower()
    # the runtime is replaced on upgrade and removed on uninstall
    assert re.search(r'^\[InstallDelete\]\s*\n(?:;.*\n)*Type: filesandordirs; Name: "\{app\}\\_internal"', text, re.M)
    assert re.search(r'^\[UninstallDelete\]\s*\n(?:;.*\n)*Type: filesandordirs; Name: "\{app\}\\_internal"', text, re.M)
    # the Russian messages are real UTF-8 text with the BOM
    assert "Удалить данные" in text and "Оставить данные" in text


def test_iss_data_list_covers_what_the_app_writes() -> None:
    """The uninstaller removes a fixed list of data entries: a new top-level folder/file of the app must be added to it."""
    code = _iss_text().split("[Code]", 1)[1]
    listed = set(re.findall(r"(?:DelTree|DeleteFile)\(Dir \+ '\\([A-Za-z0-9_.]+)'", code))
    assert {"models", "projects", "backups", "settings.json"} <= listed, listed
    written: set[str] = set()
    for path in (ROOT / "vndub_app").rglob("*.py"):
        src = path.read_text(encoding="utf-8")
        written.update(re.findall(r"(?<![\w.])sub\(\"([A-Za-z0-9_.]+)\"", src))  # appdirs.sub("logs") and its in-module calls
        written.update(re.findall(r"appdirs\.sub\(\"([A-Za-z0-9_.]+)\"", src))
        written.update(re.findall(r"data_dir(?:\(\))?\s*/\s*\"([A-Za-z0-9_.]+)\"", src))
        written.update(re.findall(r"<data>/([A-Za-z0-9_.]+)", src))  # the layout the modules document
    assert {"models", "projects", "logs", "settings.json", "app.lock", "jobs.json"} <= written, written  # the scan itself works
    missing = written - listed
    assert not missing, f"packaging/VNDub.iss RemoveAppData does not remove {sorted(missing)}"


def test_build_script_has_every_step_and_degrades_gracefully() -> None:
    text = (PKG / "build_app.ps1").read_text(encoding="utf-8")
    text.encode("ascii")  # Windows PowerShell 5.1 reads BOM-less files as ANSI
    for needed in ("npm", "pyinstaller", "--selfcheck", "--selfcheck-out", "ISCC.exe", "Inno Setup", "-SkipSpa", "-SkipInstaller", "VNDub-Setup.exe", "uv.exe", "_staging"):
        assert needed in text, needed
    assert "Warn " in text and "is not installed" in text  # missing Node / Inno Setup: a warning, not a failure


def test_build_script_installer_only_switch() -> None:
    """-InstallerOnly packs the existing dist\\VNDub: no SPA, PyInstaller or self-check, but the exe must exist."""
    text = (PKG / "build_app.ps1").read_text(encoding="utf-8")
    assert "[switch]$InstallerOnly" in text and "[string]$Iscc" in text
    only, full = text.split("if ($InstallerOnly) {", 1)[1].split("} else {", 1)
    assert r"dist\VNDub\VNDub.exe not found" in only and "Test-Path $exe" in only
    for step in ("npm", "pyinstaller", "--selfcheck", "Copy-Item"):
        assert step not in only, step
        assert step in full, step
    # the per-user Inno Setup folder and the registry entry are searched; the version comes from pyproject.toml
    assert r"Programs\Inno Setup 6\ISCC.exe" in text and "Inno Setup 6_is1" in text
    assert "pyproject.toml" in text and "/DAppVersion=$ver" in text
    # a missing Inno Setup is an error for -InstallerOnly, a warning otherwise
    assert re.search(r"if \(\$InstallerOnly\) \{ throw \$msg \}\s*\n\s*Warn \$msg", text)
    # the installer is the one just compiled, not a stale file
    assert "LastWriteTime -lt $tc" in text


def test_installer_version_is_the_app_version() -> None:
    import tomllib

    import vndub_app

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["version"] == vndub_app.__version__
    default = re.search(r'#define AppVersion "([^"]+)"', _iss_text())
    assert default and default.group(1) == project["version"], "the .iss fallback version is stale"


@pytest.mark.skipif(not sys.platform.startswith("win") or shutil.which("powershell") is None, reason="needs Windows PowerShell")
def test_build_script_rejects_contradicting_switches() -> None:
    r = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(PKG / "build_app.ps1"), "-InstallerOnly", "-SkipInstaller"],
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode != 0
    assert "exclude each other" in r.stdout + r.stderr


def _find_iscc() -> str | None:
    cands = [os.environ.get("ISCC", ""), shutil.which("iscc") or ""]
    for var, sub in (("LOCALAPPDATA", "Programs"), ("ProgramFiles(x86)", ""), ("ProgramFiles", "")):
        base = os.environ.get(var)
        if base:
            cands.append(str(Path(base, sub, "Inno Setup 6", "ISCC.exe")))
    return next((c for c in cands if c and Path(c).is_file()), None)


@pytest.mark.skipif(
    os.environ.get("VNDUB_BUILD_INSTALLER_TEST") != "1" or _find_iscc() is None,
    reason="set VNDUB_BUILD_INSTALLER_TEST=1 and install Inno Setup 6 (ISCC.exe) to compile the installer",
)
def test_installer_script_compiles(tmp_path: Path) -> None:
    """Compile packaging/VNDub.iss for real (a tiny payload instead of dist\\VNDub, so it takes seconds): no errors, no warnings."""
    iscc = _find_iscc()
    assert iscc
    payload = tmp_path / "payload"
    (payload / "_internal").mkdir(parents=True)
    (payload / "VNDub.exe").write_bytes(b"MZ")
    (payload / "_internal" / "a.txt").write_text("x", encoding="utf-8")
    text = _iss_text()
    files_line = 'Source: "..\\dist\\VNDub\\*"'
    icon_line = "SetupIconFile=VNDub.ico"
    assert files_line in text and icon_line in text
    text = text.replace(files_line, f'Source: "{payload}\\*"').replace(icon_line, f"SetupIconFile={PKG / 'VNDub.ico'}")
    script = tmp_path / "VNDub.iss"
    script.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))  # keep the BOM: the Russian messages depend on it
    out = tmp_path / "out"
    r = subprocess.run([iscc, f"/O{out}", "/DAppVersion=9.9.9", str(script)], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240)
    log = r.stdout + r.stderr
    assert r.returncode == 0, log[-3000:]
    assert "Successful compile" in log
    warnings = [line for line in log.splitlines() if line.lstrip().lower().startswith("warning")]
    assert not warnings, warnings
    setup = out / "VNDub-Setup.exe"
    assert setup.is_file() and setup.stat().st_size > 1_000_000


@pytest.mark.skipif(not sys.platform.startswith("win") or shutil.which("powershell") is None, reason="needs Windows PowerShell")
def test_build_script_parses() -> None:
    cmd = (
        "$e=$null;$t=$null;[void][System.Management.Automation.Language.Parser]::ParseFile("
        f"'{PKG / 'build_app.ps1'}',[ref]$t,[ref]$e);if($e.Count){{$e|ForEach-Object{{$_.Message}};exit 1}}"
    )
    r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr


def test_icon_and_entry_point_exist() -> None:
    assert (PKG / "VNDub.ico").read_bytes()[:4] == b"\x00\x00\x01\x00"
    entry = (PKG / "entry.py").read_text(encoding="utf-8")
    assert "freeze_support" in entry and "vndub_app.shell" in entry


def test_pin_script_is_importable_and_checks_the_manifest() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("pin_tools_manifest", ROOT / "scripts" / "pin_tools_manifest.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    assert callable(mod.main)

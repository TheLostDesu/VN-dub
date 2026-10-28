"""Packaging files: PyInstaller spec, Inno Setup script, build script (static checks; the build itself is `packaging/build_app.ps1`)."""

from __future__ import annotations

import ast
import json
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


def _section(text: str, name: str) -> str:
    m = re.search(rf"^\[{name}\]\s*\n(.*?)(?=^\[|\Z)", text, re.M | re.S)
    assert m, name
    return m.group(1)


def test_iss_carries_git_and_the_redistributable_and_removes_them_with_the_app() -> None:
    text = _iss_text()
    # both folders come from dist\VNDub (build_app.ps1 adds them): the one recursive Source picks them up
    files = _section(text, "Files")
    assert r'Source: "..\dist\VNDub\*"; DestDir: "{app}"' in files and "recursesubdirs" in files and "createallsubdirs" in files
    assert "git" in files and "redist" in files  # named in the comment; nothing excludes them
    assert "Excludes" not in files
    # compiling without them is an error, not an installer on which no model starts
    assert r'FileExists(SourcePath + "..\dist\VNDub\git\cmd\git.exe")' in text
    assert r'FileExists(SourcePath + "..\dist\VNDub\redist\vc_redist.x64.exe")' in text
    # an upgrade replaces them, an uninstall removes them (the redistributable stays on disk in between for the app to run again)
    for section in ("InstallDelete", "UninstallDelete"):
        body = _section(text, section)
        for folder in ("git", "redist"):
            assert f'Type: filesandordirs; Name: "{{app}}\\{folder}"' in body, (section, folder)
    # the system-wide runtime is never uninstalled, and the file is not deleted after the install
    assert "/uninstall" not in text.lower() and "vc_redist" not in text.split("procedure CurUninstallStepChanged", 1)[1]
    assert "deleteafterinstall" not in text.lower()


def test_iss_carries_the_bundled_bgi_extractor() -> None:
    """bin\\bgi-extractor (compiled by build_app.ps1) ships like Git: required to compile, replaced on upgrade, removed on uninstall."""
    text = _iss_text()
    assert r'FileExists(SourcePath + "..\dist\VNDub\bin\bgi-extractor\bgi-extractor.exe")' in text
    for section in ("InstallDelete", "UninstallDelete"):
        assert 'Type: filesandordirs; Name: "{app}\\bin"' in _section(text, section), section
    # the app looks where the manifest says, and that folder is one the data folder never uses: a data folder the
    # user put inside {app} must survive an upgrade
    manifest = json.loads((ROOT / "app" / "tools_manifest.json").read_text(encoding="utf-8"))
    bundled = [rel for e in manifest["tools"] for rel in (e.get("bundled") or {}).values()]
    assert bundled and all(rel.startswith("bin/") for rel in bundled)
    data_names = set(re.findall(r"(?:DelTree|DeleteFile)\(Dir \+ '\\([A-Za-z0-9_.]+)'", text.split("[Code]", 1)[1]))
    assert "tools" in data_names and "bin" not in data_names


def test_iss_runs_the_redistributable_only_when_the_runtime_is_missing_or_old() -> None:
    text = _iss_text()
    code = text.split("[Code]", 1)[1]
    # the check: HKLM, 64-bit view, VS 14.0 Runtimes\x64; Installed = 1 and Major.Minor >= 14.40
    assert "HKLM64" in code and r"VCRuntimeKey = 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64'" in code
    assert "VCMinMajor = 14;" in code and "VCMinMinor = 40;" in code
    assert "(Installed = 1) and ((Major > VCMinMajor) or ((Major = VCMinMajor) and (Minor >= VCMinMinor)))" in code
    for value in ("'Installed'", "'Major'", "'Minor'"):
        assert f"VCRuntimeKey, {value}" in code
    body = code.split("procedure InstallVCRuntime();", 1)[1].split("procedure CurStepChanged", 1)[0]
    # the run is behind the check: the missing-or-old branch is the only one that gets to Exec
    check = body.index("if not VCRuntimeMissingOrOld() then")
    skipped = body.index("vc_redist.x64.exe is skipped")
    run = body.index("Exec(Redist, '/install /quiet /norestart', '', SW_HIDE, ewWaitUntilTerminated, Code)")
    assert check < skipped < run and "Exit;" in body[check:run]
    assert "ExpandConstant('{app}\\redist\\vc_redist.x64.exe')" in body and "FileExists(Redist)" in body
    # success = 0, 1638 (a newer one is installed), 3010 (restart advised); anything else is logged and the setup goes on
    assert "(Code = 0) or (Code = 1638) or (Code = 3010)" in body
    for word in ("RaiseException", "Abort", "WizardForm.Close", "Halt", "ExitSetupMsgBox"):
        assert word not in body, word
    assert body.count("Log(") >= 6 and "IntToStr(Code)" in body  # the decision and the exit code are in the install log
    # it runs after the files are in place, with a status message in both languages
    assert re.search(r"procedure CurStepChanged\(CurStep: TSetupStep\);\s*begin.*?\n\s*if CurStep = ssPostInstall then\s*InstallVCRuntime\(\);", code, re.S)
    assert "WizardForm.StatusLabel.Caption := CustomMessage('VCRedistInstalling')" in body and "not WizardSilent" in body
    for lang in ("russian", "english"):
        for msg in ("VCRedistInstalling", "VCRedistFailed"):
            assert f"{lang}.{msg}=" in text, (lang, msg)
    assert "Visual C++" in text and "администратора" in text and "administrator" in text
    # per-user setup stays: the redistributable elevates itself, Setup does not
    assert "PrivilegesRequired=lowest" in text and "PrivilegesRequiredOverridesAllowed" not in text
    # the decision is the one of the Doctor row (same key, same minimum)
    from vndub_app import doctor

    assert doctor.VC_REG_KEY in code and f"VCMinMinor = {doctor.VC_MIN[1]};" in code


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


def _build_script() -> str:
    return (PKG / "build_app.ps1").read_text(encoding="utf-8")


def test_build_script_bundles_the_vc_redistributable_with_a_signature_check() -> None:
    text = _build_script()
    # the permalink serves the latest build, so no hash is pinned: the Microsoft signature is the check
    assert "https://aka.ms/vs/17/release/vc_redist.x64.exe" in text
    assert "Get-AuthenticodeSignature" in text and '$sig.Status -ne "Valid"' in text and "CN=Microsoft Corporation" in text
    check = text.split("function Get-SignatureProblem", 1)[1].split("function Get-VcRedist", 1)[0]
    assert "not by Microsoft Corporation" in check and "return $null" in check
    # a rejected download is deleted and fails the build; the copy in dist\VNDub is checked again
    fetch = text.split("function Get-VcRedist", 1)[1].split("function Test-MinGitZip", 1)[0]
    assert "Remove-Item -Force $new; throw" in fetch and "Get-SignatureProblem $file" in fetch
    assert "Get-SignatureProblem $redistDst" in text
    # cached in the git-ignored staging folder (-RefreshRedist downloads again); the file version is recorded
    assert 'Join-Path $staging "redist"' in text and "[switch]$RefreshRedist" in text and "Test-Path $file" in fetch
    assert "VersionInfo.FileVersion" in text and 'Join-Path $dist "redist"' in text
    assert "packaging/_staging/" in (ROOT / ".gitignore").read_text(encoding="utf-8")


def test_build_script_pins_mingit_by_url_size_and_sha256() -> None:
    text = _build_script()
    block = text.split("$MinGit = @{", 1)[1].split("}", 1)[0]
    url = re.search(r'Url\s*=\s*"([^"]+)"', block)
    assert url and url.group(1).startswith("https://github.com/git-for-windows/git/releases/download/v")
    assert url.group(1).endswith("-64-bit.zip") and "busybox" not in url.group(1) and "MinGit-" in url.group(1)
    version = re.search(r'Version\s*=\s*"([\d.]+)"', block)
    git_version = re.search(r'GitVersion\s*=\s*"([\w.]+)"', block)
    assert version and git_version and git_version.group(1).startswith(version.group(1).rsplit(".", 1)[0])
    assert f"MinGit-{version.group(1)}-64-bit.zip" in url.group(1)
    sha = re.search(r'Sha256\s*=\s*"([0-9a-f]{64})"', block)
    size = re.search(r"Size\s*=\s*(\d{7,})", block)
    assert sha and size
    # verified after every download and for the cached copy; a mismatch is an error
    assert "Get-FileHash $file -Algorithm SHA256" in text and "$MinGit.Sha256" in text and "$MinGit.Size" in text
    assert "does not match the pinned size" in text
    # extracted to dist\VNDub\git, the version it reports is checked, the GPL notice must be there
    assert 'Join-Path $dist "git"' in text and r'"cmd\git.exe"' in text and "$MinGit.GitVersion" in text
    assert "LICENSE.txt" in text and "ExtractToDirectory" in text
    # if an earlier build left the zip in the staging folder, it is still the pinned one
    cached = ROOT / "packaging" / "_staging" / "mingit" / url.group(1).rsplit("/", 1)[1]
    if cached.is_file():
        import hashlib

        assert cached.stat().st_size == int(size.group(1))
        assert hashlib.sha256(cached.read_bytes()).hexdigest() == sha.group(1)


def test_build_script_adds_the_runtime_after_pyinstaller_and_also_for_installer_only() -> None:
    text = _build_script()
    only, full = text.split("if ($InstallerOnly) {", 1)[1].split("} else {", 1)
    assert "Add-BundledRuntime" in only  # the folder may predate this change: fill what is missing, keep what is there
    assert full.index('Run "pyinstaller"') < full.index("Add-BundledRuntime") < full.index("--selfcheck")  # PyInstaller recreates dist\VNDub
    assert text.index("function Add-BundledRuntime") < text.index("Add-BundledRuntime\n")  # defined before the first call
    summary = text.split('Step "summary"', 1)[1]
    assert "VC++ redist" in summary and "git        :" in summary


def test_build_script_compiles_bgi_extractor_from_a_pinned_commit() -> None:
    text = _build_script()
    block = text.split("$BgiExtractor = @{", 1)[1].split("\n}", 1)[0]
    assert 'Repo    = "https://github.com/mirinnano/bgi-extractor.git"' in block
    assert re.search(r'Commit\s*=\s*"[0-9a-f]{40}"', block)
    # fetched by commit id, then it must be an unmodified checkout of that commit (cached in the git-ignored staging folder)
    assert '"fetch", "-q", "--depth", "1", $BgiExtractor.Repo, $BgiExtractor.Commit' in text
    assert '"rev-parse", "HEAD"' in text and '"status", "--porcelain"' in text and 'Join-Path $staging "bgi-extractor\\src"' in text
    # MSVC via vcvars64.bat: static C runtime, UTF-8 sources, the UTF-8 code page manifest embedded; the MIT notice ships along
    flags = set(re.search(r'ClFlags\s*=\s*"([^"]+)"', block).group(1).split())
    assert {"/MT", "/utf-8", "/std:c++17", "/EHsc", "/O2"} <= flags
    assert "vswhere.exe" in text and "Microsoft.VisualStudio.Component.VC.Tools.x86.x64" in text and "vcvars64.bat" in text
    assert "/MANIFEST:EMBED /MANIFESTINPUT:" in text and '"LICENSE.txt"' in text
    # a change of commit, flags, sources or manifest compiles again
    stamp = text.split("function Get-BgiStamp", 1)[1].split("function Build-BgiExtractor", 1)[0]
    for part in ("$BgiExtractor.Commit", "$BgiExtractor.ClFlags", "$BgiExtractor.Sources", "bgi-extractor.manifest"):
        assert part in stamp, part
    # every build checks the copy in dist: it starts, and it opens an archive in a folder with Japanese characters
    check = text.split("function Get-BgiExtractorProblem", 1)[1].split("function Add-BundledTools", 1)[0]
    assert "--help" in check and "0x30C6" in check and "BURIKO ARC20" in check and r"\(0 entries\)" in check
    assert "Get-BgiExtractorProblem $exe" in text and 'Join-Path $Root "dist\\VNDub\\bin\\bgi-extractor"' in text
    # after MinGit (it can fetch the source), before the self-check (which looks for it), also for -InstallerOnly
    only, full = text.split("if ($InstallerOnly) {", 1)[1].split("} else {", 1)
    assert only.index("Add-BundledRuntime") < only.index("Add-BundledTools")
    assert full.index("Add-BundledRuntime") < full.index("Add-BundledTools") < full.index("--selfcheck")
    assert "bgi-extractor" in text.split('Step "summary"', 1)[1]


def test_bgi_extractor_manifest_sets_the_utf8_code_page() -> None:
    import xml.etree.ElementTree as ET

    root = ET.parse(PKG / "bgi-extractor.manifest").getroot()
    ns = "{http://schemas.microsoft.com/SMI/2019/WindowsSettings}"
    assert [el.text for el in root.iter(f"{ns}activeCodePage")] == ["UTF-8"]


@pytest.mark.skipif(not sys.platform.startswith("win") or shutil.which("powershell") is None, reason="needs Windows PowerShell")
def test_signature_check_accepts_microsoft_and_rejects_everything_else(tmp_path: Path) -> None:
    """Run the build script's own Get-SignatureProblem on files with and without a (Microsoft Corporation) signature."""
    func = re.search(r"function Get-SignatureProblem.*?\n\}\n", _build_script(), re.S)
    assert func
    system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    windows_signed = system32 / "notepad.exe"  # signed by "Microsoft Windows": a Microsoft certificate, but not Microsoft Corporation
    unsigned = tmp_path / "plain.exe"
    unsigned.write_bytes(b"MZ not signed")
    staged = PKG / "_staging" / "redist" / "vc_redist.x64.exe"  # present after one build; signed by Microsoft Corporation
    files = [windows_signed, unsigned] + ([staged] if staged.is_file() else [])
    script = tmp_path / "check.ps1"
    script.write_text(
        func.group(0)
        + "\n$ErrorActionPreference = 'Stop'\n"
        + "foreach ($f in $args) { if (Test-Path $f) { $p = Get-SignatureProblem $f; Write-Output ('[' + (Split-Path -Leaf $f) + '] ' + $(if ($p) { $p } else { 'OK' })) } }\n",
        encoding="utf-8",
    )
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), *map(str, files)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "[plain.exe] signature status is " in r.stdout  # not Valid: refused
    if windows_signed.is_file():
        assert "[notepad.exe] OK" not in r.stdout  # only "Microsoft Corporation" passes
    if staged.is_file():
        assert "[vc_redist.x64.exe] OK" in r.stdout


def test_installer_version_is_the_app_version() -> None:
    import tomllib

    import vndub
    import vndub_app

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["version"] == vndub_app.__version__
    assert project["version"] == vndub.__version__, "vndub/__init__.py is stale: `vndub --version` prints it"
    default = re.search(r'#define AppVersion "([^"]+)"', _iss_text())
    assert default and default.group(1) == project["version"], "the .iss fallback version is stale"
    web = ROOT / "vndub_app" / "web"
    package = json.loads((web / "package.json").read_text(encoding="utf-8"))
    assert package["version"] == project["version"], "vndub_app/web/package.json is stale"
    lock = json.loads((web / "package-lock.json").read_text(encoding="utf-8"))
    assert lock["version"] == project["version"] and lock["packages"][""]["version"] == project["version"], "vndub_app/web/package-lock.json is stale"


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
    # the same layout as the repository: <root>/packaging/VNDub.iss next to <root>/dist/VNDub (with the bundled runtime)
    pkg = tmp_path / "packaging"
    payload = tmp_path / "dist" / "VNDub"
    (payload / "_internal").mkdir(parents=True)
    (payload / "git" / "cmd").mkdir(parents=True)
    (payload / "redist").mkdir()
    (payload / "bin" / "bgi-extractor").mkdir(parents=True)
    pkg.mkdir()
    (payload / "VNDub.exe").write_bytes(b"MZ")
    (payload / "_internal" / "a.txt").write_text("x", encoding="utf-8")
    (payload / "git" / "cmd" / "git.exe").write_bytes(b"MZ")
    (payload / "redist" / "vc_redist.x64.exe").write_bytes(b"MZ")
    (payload / "bin" / "bgi-extractor" / "bgi-extractor.exe").write_bytes(b"MZ")
    shutil.copy(PKG / "VNDub.ico", pkg / "VNDub.ico")
    script = pkg / "VNDub.iss"
    script.write_bytes(b"\xef\xbb\xbf" + _iss_text().encode("utf-8"))  # keep the BOM: the Russian messages depend on it
    out = tmp_path / "out"
    r = subprocess.run([iscc, f"/O{out}", "/DAppVersion=9.9.9", str(script)], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240)
    log = r.stdout + r.stderr
    assert r.returncode == 0, log[-3000:]
    assert "Successful compile" in log
    warnings = [line for line in log.splitlines() if line.lstrip().lower().startswith("warning")]
    assert not warnings, warnings
    setup = out / "VNDub-Setup.exe"
    assert setup.is_file() and setup.stat().st_size > 1_000_000
    # a payload without Git (or without the redistributable, or bgi-extractor) is refused: no installer on which clean machines break
    (payload / "bin" / "bgi-extractor" / "bgi-extractor.exe").unlink()
    r = subprocess.run([iscc, f"/O{out}", "/DAppVersion=9.9.9", str(script)], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240)
    assert r.returncode != 0 and "bgi-extractor.exe is missing" in r.stdout + r.stderr
    (payload / "git" / "cmd" / "git.exe").unlink()
    r = subprocess.run([iscc, f"/O{out}", "/DAppVersion=9.9.9", str(script)], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240)
    assert r.returncode != 0 and "git.exe is missing" in r.stdout + r.stderr


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

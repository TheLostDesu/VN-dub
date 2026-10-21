<#
.SYNOPSIS
  Build the VNDub desktop app: SPA (when Node is available) -> PyInstaller one-folder build
  dist\VNDub\VNDub.exe -> self-check of the exe -> Inno Setup installer (when Inno Setup is installed).

.EXAMPLE
  packaging\build_app.ps1                  # everything that can run on this machine
  packaging\build_app.ps1 -SkipSpa         # reuse an existing vndub_app\web\dist
  packaging\build_app.ps1 -Clean           # drop build\ and dist\VNDub first
  packaging\build_app.ps1 -SkipInstaller   # never try Inno Setup
  packaging\build_app.ps1 -InstallerOnly   # only installer\VNDub-Setup.exe, from the existing dist\VNDub
                                           # (no SPA, no PyInstaller, no self-check, no uv)

.NOTES
  Needs uv (bundled into the app as bin\uv.exe; found via -Uv, PATH or ~\.local\bin) and the dev group
  (pyinstaller, pillow) of the project environment: `uv sync` once.
  A missing Node.js or Inno Setup is a warning, not an error: the SPA is then the "not built" page / the
  installer step is skipped. The lead rebuilds after the SPA lands. With -InstallerOnly a missing Inno
  Setup is an error (that is all the run is for).
  Inno Setup 6.3+ (ISCC.exe) is found via -Iscc, PATH, the registry or the usual folders, the per-user
  one (%LOCALAPPDATA%\Programs\Inno Setup 6) included:
  winget install --id JRSoftware.InnoSetup -e --scope user
  The installer version is the one in pyproject.toml ([project] version).
#>
param(
    [switch]$SkipSpa,
    [switch]$SkipInstaller,
    [switch]$SkipSelfcheck,
    [switch]$InstallerOnly,
    [switch]$Clean,
    [string]$Uv = "",
    [string]$Iscc = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$t0 = Get-Date
if ($InstallerOnly -and $SkipInstaller) { throw "-InstallerOnly and -SkipInstaller exclude each other" }

function Step([string]$msg) { Write-Host ""; Write-Host "== $msg" -ForegroundColor Cyan }
function Warn([string]$msg) { Write-Host "WARNING: $msg" -ForegroundColor Yellow }
function Run([string]$what, [scriptblock]$block) {
    # native tools write progress to stderr; judge success by the exit code only (Windows PowerShell 5.1)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $block 2>&1 | ForEach-Object { "$_" } } finally { $ErrorActionPreference = $prev }
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE)" }
}
function Get-AppVersion {
    # [project] version of pyproject.toml (the installer, the exe and the Python package carry the same one)
    $inProject = $false
    foreach ($line in Get-Content (Join-Path $Root "pyproject.toml")) {
        if ($line -match '^\s*\[(.+)\]\s*$') { $inProject = ($Matches[1] -eq "project"); continue }
        if ($inProject -and $line -match '^\s*version\s*=\s*"([^"]+)"') { return $Matches[1] }
    }
    throw "no [project] version found in pyproject.toml"
}
function Find-Iscc([string]$explicit) {
    $cands = @($explicit, (Get-Command iscc -ErrorAction SilentlyContinue | ForEach-Object { $_.Source }))
    foreach ($key in @(
            "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1",
            "HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1",
            "HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1")) {
        $loc = (Get-ItemProperty $key -ErrorAction SilentlyContinue).InstallLocation
        if ($loc) { $cands += (Join-Path $loc "ISCC.exe") }
    }
    $cands += (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe")
    $cands += (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe")
    $cands += (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe")
    foreach ($c in $cands) { if ($c -and (Test-Path $c)) { return (Resolve-Path $c).Path } }
    return $null
}

$exe = Join-Path $Root "dist\VNDub\VNDub.exe"
$spaBuilt = $false
$sizeMb = $null
$buildSeconds = $null
$selfOk = $null

if ($InstallerOnly) {
    # -- the existing build -----------------------------------------------------------------------------
    Step "existing build (dist\VNDub)"
    if (-not (Test-Path $exe)) { throw "dist\VNDub\VNDub.exe not found: run packaging\build_app.ps1 first (-InstallerOnly only packs an existing build)" }
    $spaBuilt = (Test-Path (Join-Path $Root "dist\VNDub\_internal\vndub_app\web\dist\index.html")) -or (Test-Path (Join-Path $Root "dist\VNDub\vndub_app\web\dist\index.html"))
    if (-not $spaBuilt) { Warn "the existing dist\VNDub has no SPA (vndub_app\web\dist\index.html): the installed app would show the 'SPA not built' page" }
    $sizeMb = [math]::Round((Get-ChildItem (Join-Path $Root "dist\VNDub") -Recurse -File | Measure-Object Length -Sum).Sum / 1MB)
    Write-Host "using $exe (built $((Get-Item $exe).LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')), folder size $sizeMb MB)"
    Write-Host "SPA, PyInstaller and self-check skipped (-InstallerOnly)"
} else {
    # -- uv ----------------------------------------------------------------------------------------------
    Step "uv"
    $uvExe = $null
    foreach ($c in @($Uv, (Get-Command uv -ErrorAction SilentlyContinue | ForEach-Object { $_.Source }), (Join-Path $env:USERPROFILE ".local\bin\uv.exe"))) {
        if ($c -and (Test-Path $c)) { $uvExe = (Resolve-Path $c).Path; break }
    }
    if (-not $uvExe) { throw "uv not found (pass -Uv <path>, put it on PATH, or install it: https://docs.astral.sh/uv/)" }
    Write-Host "using $uvExe"

    # -- SPA ---------------------------------------------------------------------------------------------
    Step "SPA (vndub_app\web)"
    $web = Join-Path $Root "vndub_app\web"
    if ($SkipSpa) {
        Write-Host "skipped (-SkipSpa)"
        $spaBuilt = Test-Path (Join-Path $web "dist\index.html")
    } elseif (-not (Test-Path (Join-Path $web "package.json"))) {
        Warn "vndub_app\web\package.json not found: the SPA is not part of this build (the app shows a 'not built' page)"
    } elseif (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
        Warn "Node.js (npm) is not installed: the SPA is NOT built. Install Node 20+ and run this script again. Using an existing vndub_app\web\dist if there is one."
        $spaBuilt = Test-Path (Join-Path $web "dist\index.html")
    } else {
        Push-Location $web
        try {
            if (Test-Path "package-lock.json") { Run "npm ci" { npm ci } } else { Run "npm install" { npm install } }
            Run "npm run build" { npm run build }
        } finally { Pop-Location }
        $spaBuilt = Test-Path (Join-Path $web "dist\index.html")
    }
    if ($spaBuilt) { Write-Host "SPA: vndub_app\web\dist present" } else { Warn "no vndub_app\web\dist: the bundled app will show the 'SPA not built' page" }

    # -- staging (uv.exe, icon) --------------------------------------------------------------------------
    Step "staging"
    $staging = Join-Path $Root "packaging\_staging"
    New-Item -ItemType Directory -Force $staging | Out-Null
    Copy-Item -Force $uvExe (Join-Path $staging ([System.IO.Path]::GetFileName($uvExe)))
    if (-not (Test-Path (Join-Path $Root "packaging\VNDub.ico"))) {
        Run "make_icon" { & $uvExe run python packaging\make_icon.py }
    }

    # -- PyInstaller -------------------------------------------------------------------------------------
    Step "PyInstaller"
    if ($Clean) {
        Remove-Item -Recurse -Force (Join-Path $Root "build\pyinstaller"), (Join-Path $Root "dist\VNDub") -ErrorAction SilentlyContinue
    }
    $tb = Get-Date
    Run "pyinstaller" { & $uvExe run pyinstaller packaging\vndub_app.spec --noconfirm --distpath dist --workpath build\pyinstaller }
    if (-not (Test-Path $exe)) { throw "dist\VNDub\VNDub.exe was not produced" }
    $buildSeconds = [math]::Round(((Get-Date) - $tb).TotalSeconds)
    $sizeMb = [math]::Round((Get-ChildItem (Join-Path $Root "dist\VNDub") -Recurse -File | Measure-Object Length -Sum).Sum / 1MB)
    Write-Host "dist\VNDub\VNDub.exe built in ${buildSeconds}s, folder size $sizeMb MB"

    # -- self-check of the packaged exe ------------------------------------------------------------------
    if ($SkipSelfcheck) {
        Step "self-check"; Write-Host "skipped (-SkipSelfcheck)"
    } else {
        Step "VNDub.exe --selfcheck"
        $out = Join-Path $env:TEMP ("vndub-selfcheck-" + [guid]::NewGuid().ToString("N").Substring(0, 8) + ".json")
        # a windowed exe has no console: it writes the JSON line to the file named here
        $p = Start-Process -FilePath $exe -ArgumentList @("--selfcheck", "--selfcheck-out", "`"$out`"") -Wait -PassThru -WindowStyle Hidden
        if (-not (Test-Path $out)) { throw "self-check wrote no result file (exit code $($p.ExitCode))" }
        $line = Get-Content -Raw -Encoding UTF8 $out
        Write-Host $line.Trim()
        $res = $line | ConvertFrom-Json
        $selfOk = [bool]$res.ok
        foreach ($w in @($res.warnings)) { if ($w) { Warn $w } }
        Remove-Item -Force $out -ErrorAction SilentlyContinue
        if (-not $selfOk -or $p.ExitCode -ne 0) { throw "self-check failed: $(@($res.problems) -join '; ')" }
        Write-Host "self-check OK" -ForegroundColor Green
    }
}

# -- Inno Setup --------------------------------------------------------------------------------------
Step "installer (Inno Setup)"
$installer = $null
$installerSeconds = $null
$iscc = $null
if (-not $SkipInstaller) { $iscc = Find-Iscc $Iscc }
if ($SkipInstaller) {
    Write-Host "skipped (-SkipInstaller)"
} elseif (-not $iscc) {
    $msg = "Inno Setup 6 (ISCC.exe) is not installed: installer\VNDub-Setup.exe was NOT built. Install it (winget install --id JRSoftware.InnoSetup -e --scope user, or https://jrsoftware.org/isdl.php) and run this script again (-InstallerOnly packs the existing dist\VNDub)."
    if ($InstallerOnly) { throw $msg }
    Warn $msg
} else {
    $ver = Get-AppVersion
    Write-Host "using $iscc, installer version $ver"
    $installer = Join-Path $Root "installer\VNDub-Setup.exe"
    $tc = Get-Date
    # ISCC prints one "Compressing:" line per file (thousands) and one "Parsing" line per directive: show the
    # rest (version, warnings in yellow, errors, result)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { $isccOut = @(& $iscc "/DAppVersion=$ver" (Join-Path $Root "packaging\VNDub.iss") 2>&1 | ForEach-Object { "$_" }) } finally { $ErrorActionPreference = $prev }
    $isccExit = $LASTEXITCODE
    foreach ($l in $isccOut) {
        if ($l -match '^\s*(Compressing|Parsing|Reading|Updating|Verification)') { continue }
        if ($l -match '^\s*Warning') { Warn $l.Trim() } elseif ($l.Trim()) { Write-Host $l }
    }
    if ($isccExit -ne 0) { throw "iscc failed (exit $isccExit)" }
    if (-not (Test-Path $installer) -or (Get-Item $installer).LastWriteTime -lt $tc) { throw "iscc did not produce installer\VNDub-Setup.exe" }
    $installerSeconds = [math]::Round(((Get-Date) - $tc).TotalSeconds)
    Write-Host ("installer: {0} ({1} MB, compiled in {2}s, sha256 {3})" -f $installer, [math]::Round((Get-Item $installer).Length / 1MB), $installerSeconds, (Get-FileHash $installer -Algorithm SHA256).Hash.ToLower())
}

# -- summary -----------------------------------------------------------------------------------------
Step "summary"
$total = [math]::Round(((Get-Date) - $t0).TotalSeconds)
Write-Host "exe        : $exe"
Write-Host "folder size: $sizeMb MB"
Write-Host "SPA        : $(if ($spaBuilt) { 'bundled' } else { 'NOT bundled' })"
Write-Host "self-check : $(if ($null -eq $selfOk) { 'skipped' } elseif ($selfOk) { 'ok' } else { 'FAILED' })"
Write-Host "installer  : $(if ($installer -and (Test-Path $installer)) { $installer } else { 'not built' })"
Write-Host "time       : ${total}s$(if ($buildSeconds) { " (PyInstaller ${buildSeconds}s)" })$(if ($installerSeconds) { " (installer ${installerSeconds}s)" })"
exit 0

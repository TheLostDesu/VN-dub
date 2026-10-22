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
  packaging\build_app.ps1 -RefreshRedist   # download vc_redist.x64.exe again (the cached copy is the "latest" of its day)

.NOTES
  Needs uv (bundled into the app as bin\uv.exe; found via -Uv, PATH or ~\.local\bin) and the dev group
  (pyinstaller, pillow) of the project environment: `uv sync` once.
  The installer carries what the downloaded models need on a clean Windows (both are added to dist\VNDub after
  PyInstaller, also by -InstallerOnly when they are not there yet; downloads are cached in packaging\_staging,
  so the first build needs the network, later ones do not):
    redist\vc_redist.x64.exe  Microsoft Visual C++ 2015-2022 Redistributable (x64) from https://aka.ms/vs/17/release/vc_redist.x64.exe.
                              The permalink always serves the latest build, so no hash is pinned: the Authenticode
                              signature must be Valid and made by Microsoft Corporation. The installer runs it only
                              when the runtime on the machine is missing or older than 14.40.
    git\                      MinGit (Git for Windows, portable, 64-bit), pinned below with URL, size and sha256
                              (GPL-2.0; LICENSE.txt ships inside). git\cmd\git.exe is what the app puts on PATH
                              when the user has no Git of his own.
    bin\bgi-extractor\        bgi-extractor.exe (BGI/Ethornell archives -> dialogue JSON, MIT; LICENSE.txt next to it).
                              Upstream publishes source only: the pinned commit is fetched with git and compiled
                              with MSVC once, then cached in packaging\_staging\bgi-extractor. Compiling needs the
                              Visual Studio 2022 Build Tools with the C++ workload:
                              winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
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
    [switch]$RefreshRedist,
    [string]$Uv = "",
    [string]$Iscc = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$t0 = Get-Date
if ($InstallerOnly -and $SkipInstaller) { throw "-InstallerOnly and -SkipInstaller exclude each other" }

# -- what the installer carries besides the app (see .NOTES) ----------------------------------------------
$staging = Join-Path $Root "packaging\_staging"
$VcRedistUrl = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
# MinGit: release v2.56.0.windows.2 of 2026-10-05 (the newest one that was at least two weeks old when it was pinned).
# Size and sha256 are the ones GitHub shows for the release asset (and the downloaded file was checked against them).
# Moving to a newer release means changing these five values together.
$MinGit = @{
    Version    = "2.56.0.2"
    GitVersion = "2.56.0.windows.2"   # what `git --version` prints after "git version "
    File       = "MinGit-2.56.0.2-64-bit.zip"
    Url        = "https://github.com/git-for-windows/git/releases/download/v2.56.0.windows.2/MinGit-2.56.0.2-64-bit.zip"
    Size       = 39806486
    Sha256     = "da35e72aa21c005a5a0d298cfbae110bc1609a815730ea0dde84b01a1b3cd3be"
}
# bgi-extractor (https://github.com/mirinnano/bgi-extractor, MIT): no release binaries upstream, so the build compiles
# this commit (2026-10-12; its source was read in full when it was pinned: standard C++17, file I/O only). /MT links the
# C runtime statically (no vcruntime140.dll needed). packaging\bgi-extractor.manifest is embedded for the UTF-8 code
# page: the tool takes paths as char*, and without it a Japanese character in a game path is lost on a Russian or
# English Windows. Moving to another commit means changing Commit and Version here and "version" of bgi_extractor in
# app\tools_manifest.json together (a test checks).
$BgiExtractor = @{
    Repo    = "https://github.com/mirinnano/bgi-extractor.git"
    Commit  = "68b260331fae9c91f9aa5262bdb9d2b741bc007b"
    Version = "1.0.0+68b2603"   # upstream's CMake project version + short commit
    ClFlags = "/nologo /std:c++17 /O2 /EHsc /utf-8 /W4 /MT /DNDEBUG"
    Sources = @("main.cpp", "arc.cpp", "dsc.cpp", "parser.cpp", "decompressor.cpp")
}
$Bundled = @{}   # filled by Add-BundledRuntime and Add-BundledTools, printed in the summary

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
function Save-Download([string]$url, [string]$dest) {
    # a half-downloaded file never gets the real name
    $prevProgress = $ProgressPreference
    $ProgressPreference = "SilentlyContinue"   # Windows PowerShell 5.1 is very slow with the progress bar on
    $part = "$dest.part"
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $part -MaximumRedirection 10
        Move-Item -Force $part $dest
    } finally {
        $ProgressPreference = $prevProgress
        Remove-Item -Force $part -ErrorAction SilentlyContinue
    }
}
function Get-SignatureProblem([string]$file) {
    # $null when the file carries a valid Authenticode signature made by Microsoft Corporation, else the reason
    $sig = Get-AuthenticodeSignature -FilePath $file
    if ($sig.Status -ne "Valid") { return "signature status is $($sig.Status)" }
    $subject = "$($sig.SignerCertificate.Subject)"
    if ($subject -notmatch '(^|, )CN=Microsoft Corporation(,|$)') { return "signed by '$subject', not by Microsoft Corporation" }
    return $null
}
function Get-VcRedist {
    # packaging\_staging\redist\vc_redist.x64.exe, downloaded once; the signature is checked every time it is used
    $dir = Join-Path $staging "redist"
    New-Item -ItemType Directory -Force $dir | Out-Null
    $file = Join-Path $dir "vc_redist.x64.exe"
    $fetch = $RefreshRedist -or -not (Test-Path $file)
    if (-not $fetch) {
        $problem = Get-SignatureProblem $file
        if ($problem) { Warn "cached vc_redist.x64.exe rejected ($problem): downloading it again"; $fetch = $true }
    }
    if ($fetch) {
        Write-Host "downloading $VcRedistUrl"
        $new = "$file.new"
        Save-Download $VcRedistUrl $new
        $problem = Get-SignatureProblem $new
        if ($problem) { Remove-Item -Force $new; throw "the downloaded vc_redist.x64.exe is rejected: $problem" }
        Move-Item -Force $new $file
    }
    return $file
}
function Test-MinGitZip([string]$file) {
    if ((Get-Item $file).Length -ne $MinGit.Size) { return $false }
    return ((Get-FileHash $file -Algorithm SHA256).Hash.ToLower() -eq $MinGit.Sha256)
}
function Get-MinGitZip {
    # packaging\_staging\mingit\<zip>, downloaded once and checked against the pinned size and sha256 every time
    $dir = Join-Path $staging "mingit"
    New-Item -ItemType Directory -Force $dir | Out-Null
    $zip = Join-Path $dir $MinGit.File
    if ((Test-Path $zip) -and -not (Test-MinGitZip $zip)) {
        Warn "cached $($MinGit.File) does not match the pinned size/sha256: downloading it again"
        Remove-Item -Force $zip
    }
    if (-not (Test-Path $zip)) {
        Write-Host "downloading $($MinGit.Url)"
        $new = "$zip.new"
        Save-Download $MinGit.Url $new
        if (-not (Test-MinGitZip $new)) { Remove-Item -Force $new; throw "the downloaded $($MinGit.File) does not match the pinned size $($MinGit.Size) / sha256 $($MinGit.Sha256)" }
        Move-Item -Force $new $zip
    }
    return $zip
}
function Get-GitVersion([string]$gitExe) {
    # "2.56.0.windows.2" from `git --version`, $null when git.exe does not run
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { $out = @(& $gitExe --version 2>&1 | ForEach-Object { "$_" }) } catch { return $null } finally { $ErrorActionPreference = $prev }
    if ($LASTEXITCODE -ne 0) { return $null }
    if (($out -join " ") -match 'git version (\S+)') { return $Matches[1] }
    return $null
}
function Add-BundledRuntime {
    # dist\VNDub\redist\vc_redist.x64.exe and dist\VNDub\git\ (the installer's "dist\VNDub\*" source picks both up).
    # Idempotent: what is already there (right version) is kept, so -InstallerOnly needs no network when the
    # folder was prepared by an earlier build; PyInstaller recreates dist\VNDub, so a full build always refills it.
    $dist = Join-Path $Root "dist\VNDub"
    $redistSrc = Get-VcRedist
    $redistDir = Join-Path $dist "redist"
    $redistDst = Join-Path $redistDir "vc_redist.x64.exe"
    New-Item -ItemType Directory -Force $redistDir | Out-Null
    Copy-Item -Force $redistSrc $redistDst
    $problem = Get-SignatureProblem $redistDst
    if ($problem) { throw "redist\vc_redist.x64.exe in dist\VNDub is rejected: $problem" }
    $sig = Get-AuthenticodeSignature -FilePath $redistDst
    $Bundled.RedistVersion = (Get-Item $redistDst).VersionInfo.FileVersion
    $Bundled.RedistSize = (Get-Item $redistDst).Length
    $Bundled.RedistSha256 = (Get-FileHash $redistDst -Algorithm SHA256).Hash.ToLower()
    $Bundled.RedistSigner = ($sig.SignerCertificate.Subject -split ",")[0]
    Write-Host ("redist\vc_redist.x64.exe: file version {0}, {1:N1} MB, sha256 {2}, signature Valid ({3})" -f $Bundled.RedistVersion, ($Bundled.RedistSize / 1MB), $Bundled.RedistSha256, $Bundled.RedistSigner)

    $gitDir = Join-Path $dist "git"
    $gitExe = Join-Path $gitDir "cmd\git.exe"
    $have = if (Test-Path $gitExe) { Get-GitVersion $gitExe } else { $null }
    if ($have -eq $MinGit.GitVersion) {
        Write-Host "git: MinGit $($MinGit.Version) is already in dist\VNDub\git"
    } else {
        $zip = Get-MinGitZip
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $tmp = "$gitDir.tmp"
        Remove-Item -Recurse -Force $gitDir, $tmp -ErrorAction SilentlyContinue
        Write-Host "extracting $($MinGit.File) to dist\VNDub\git"
        [System.IO.Compression.ZipFile]::ExtractToDirectory($zip, $tmp)
        Move-Item $tmp $gitDir
    }
    $ver = Get-GitVersion $gitExe
    if ($ver -ne $MinGit.GitVersion) { throw "dist\VNDub\git\cmd\git.exe --version says '$ver', expected '$($MinGit.GitVersion)'" }
    # GPL-2.0: the licence text travels with the binaries
    if (-not (Test-Path (Join-Path $gitDir "LICENSE.txt"))) { throw "dist\VNDub\git\LICENSE.txt is missing: MinGit's licence notice must ship with it" }
    $Bundled.GitVersion = $ver
    $Bundled.GitMB = [math]::Round((Get-ChildItem $gitDir -Recurse -File | Measure-Object Length -Sum).Sum / 1MB)
    Write-Host ("git: dist\VNDub\git\cmd\git.exe --version -> git version {0} ({1} MB on disk, LICENSE.txt present; zip {2} bytes, sha256 {3})" -f $ver, $Bundled.GitMB, $MinGit.Size, $MinGit.Sha256)
}
function Find-VcVars {
    # vcvars64.bat of the newest Visual Studio (or Build Tools) that has the x64 C++ compiler, $null when there is none
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { return $null }
    $found = @(& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find "VC\Auxiliary\Build\vcvars64.bat") | Select-Object -First 1
    if ($found -and (Test-Path $found)) { return "$found" }
    return $null
}
function Find-GitExe {
    # git on PATH, else the MinGit that Add-BundledRuntime put into dist\VNDub\git
    $onPath = Get-Command git -ErrorAction SilentlyContinue | ForEach-Object { $_.Source } | Select-Object -First 1
    if ($onPath) { return $onPath }
    $mingit = Join-Path $Root "dist\VNDub\git\cmd\git.exe"
    if (Test-Path $mingit) { return $mingit }
    return $null
}
function Invoke-Git([string]$git, [string[]]$argv) {
    # the output lines of `git <argv>`; throws when git fails
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { $out = @(& $git @argv 2>&1 | ForEach-Object { "$_" }) } finally { $ErrorActionPreference = $prev }
    if ($LASTEXITCODE -ne 0) { throw "git $($argv -join ' ') failed (exit $LASTEXITCODE): $($out -join ' ')" }
    return $out
}
function Test-BgiSource([string]$git, [string]$src) {
    # $true when $src is an unmodified checkout of the pinned commit
    if (-not (Test-Path (Join-Path $src ".git"))) { return $false }
    try {
        $head = @(Invoke-Git $git @("-C", $src, "rev-parse", "HEAD"))[0]
        $changed = @(Invoke-Git $git @("-C", $src, "status", "--porcelain", "--untracked-files=no"))
    } catch { return $false }
    return ($head -eq $BgiExtractor.Commit) -and ($changed.Count -eq 0)
}
function Get-BgiExtractorSource([string]$git) {
    # packaging\_staging\bgi-extractor\src: the pinned commit, fetched once (git checks the content against the commit id)
    $src = Join-Path $staging "bgi-extractor\src"
    if (Test-BgiSource $git $src) { return $src }
    $tmp = "$src.tmp"
    Remove-Item -Recurse -Force $src, $tmp -ErrorAction SilentlyContinue
    Write-Host "fetching $($BgiExtractor.Repo) at $($BgiExtractor.Commit)"
    Invoke-Git $git @("init", "-q", $tmp) | Out-Null
    Invoke-Git $git @("-C", $tmp, "fetch", "-q", "--depth", "1", $BgiExtractor.Repo, $BgiExtractor.Commit) | Out-Null
    Invoke-Git $git @("-C", $tmp, "checkout", "-q", "FETCH_HEAD") | Out-Null
    Move-Item $tmp $src
    if (-not (Test-BgiSource $git $src)) { throw "$src is not a clean checkout of bgi-extractor commit $($BgiExtractor.Commit)" }
    return $src
}
function Get-BgiStamp {
    # what the cached build was made from: a change of any part compiles again
    $manifest = (Get-FileHash (Join-Path $PSScriptRoot "bgi-extractor.manifest") -Algorithm SHA256).Hash.ToLower()
    return "commit=$($BgiExtractor.Commit); flags=$($BgiExtractor.ClFlags); sources=$($BgiExtractor.Sources -join ','); manifest=$manifest"
}
function Build-BgiExtractor([string]$src, [string]$dest) {
    # $dest\bgi-extractor.exe (cl + link with the embedded manifest) and $dest\LICENSE.txt
    $vcvars = Find-VcVars
    if (-not $vcvars) {
        throw "no Visual C++ compiler found: bgi-extractor is compiled from source (once, then cached in packaging\_staging\bgi-extractor). Install the Visual Studio 2022 Build Tools with the C++ workload: winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override `"--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended`""
    }
    Write-Host "compiling bgi-extractor with $vcvars"
    $obj = Join-Path $staging "bgi-extractor\obj"
    Remove-Item -Recurse -Force $obj -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force $obj | Out-Null
    $files = ($BgiExtractor.Sources | ForEach-Object { "`"$(Join-Path $src "src\$_")`"" }) -join " "
    $manifest = Join-Path $PSScriptRoot "bgi-extractor.manifest"
    $vsInstaller = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer"   # vcvars64.bat calls vswhere from PATH
    $line = "set `"PATH=$vsInstaller;%PATH%`" && call `"$vcvars`" >nul && cd /d `"$obj`" && cl $($BgiExtractor.ClFlags) /I`"$src\src`" $files /Fe:bgi-extractor.exe /link /MANIFEST:EMBED /MANIFESTINPUT:`"$manifest`""
    Run "cl (bgi-extractor)" { cmd /d /c $line }
    $exe = Join-Path $obj "bgi-extractor.exe"
    if (-not (Test-Path $exe)) { throw "cl did not produce bgi-extractor.exe" }
    New-Item -ItemType Directory -Force $dest | Out-Null
    Copy-Item -Force $exe (Join-Path $dest "bgi-extractor.exe")
    # MIT: the licence text travels with the binary
    Copy-Item -Force (Join-Path $src "LICENSE") (Join-Path $dest "LICENSE.txt")
    Remove-Item -Recurse -Force $obj -ErrorAction SilentlyContinue
}
function Get-BgiExtractorProblem([string]$exe) {
    # $null when the exe starts and opens an archive whose path has Japanese characters (which needs the UTF-8 code page
    # of the embedded manifest: without it the archive is "not found"), else the reason
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $help = @(& $exe --help 2>&1 | ForEach-Object { "$_" })
        if ($LASTEXITCODE -ne 0 -or ($help -join "`n") -notmatch "bgi-extractor") { return "--help failed (exit $LASTEXITCODE)" }
        $jp = -join [char[]](0x30C6, 0x30B9, 0x30C8)   # katakana "tesuto": in no ANSI code page of a Russian or English Windows
        $dir = Join-Path $env:TEMP ("vndub-bgi-" + [guid]::NewGuid().ToString("N").Substring(0, 8) + "-" + $jp)
        New-Item -ItemType Directory -Force $dir | Out-Null
        try {
            $arc = Join-Path $dir "$jp.arc"
            # an empty BURIKO ARC20 archive: the magic and an entry count of 0
            [System.IO.File]::WriteAllBytes($arc, [byte[]]([System.Text.Encoding]::ASCII.GetBytes("BURIKO ARC20") + @(0, 0, 0, 0)))
            $out = @(& $exe -l $arc 2>&1 | ForEach-Object { "$_" })
            if ($LASTEXITCODE -ne 0 -or ($out -join "`n") -notmatch "\(0 entries\)") {
                return "it cannot open an archive whose path has Japanese characters (exit $LASTEXITCODE`: $($out -join ' ')); is packaging\bgi-extractor.manifest embedded?"
            }
        } finally { Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue }
    } finally { $ErrorActionPreference = $prev }
    return $null
}
function Add-BundledTools {
    # dist\VNDub\bin\bgi-extractor\ (bgi-extractor.exe, LICENSE.txt) from the cached build; compiled when it is missing
    # or was made from other inputs (Get-BgiStamp). Needs git (MinGit from Add-BundledRuntime will do) the first time.
    $cache = Join-Path $staging "bgi-extractor\bin"
    $stampFile = Join-Path $cache "build-stamp.txt"
    $stamp = Get-BgiStamp
    $cached = (Test-Path (Join-Path $cache "bgi-extractor.exe")) -and (Test-Path (Join-Path $cache "LICENSE.txt")) -and
        (Test-Path $stampFile) -and ((Get-Content -Raw $stampFile).Trim() -eq $stamp)
    if ($cached) {
        Write-Host "bgi-extractor: the cached build of commit $($BgiExtractor.Commit) is current"
    } else {
        $git = Find-GitExe
        if (-not $git) { throw "git not found: it is needed once to fetch the bgi-extractor source" }
        $src = Get-BgiExtractorSource $git
        Remove-Item -Recurse -Force $cache -ErrorAction SilentlyContinue
        Build-BgiExtractor $src $cache
        Set-Content -Encoding ASCII $stampFile $stamp
    }
    $dst = Join-Path $Root "dist\VNDub\bin\bgi-extractor"
    Remove-Item -Recurse -Force $dst -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force $dst | Out-Null
    Copy-Item -Force (Join-Path $cache "bgi-extractor.exe"), (Join-Path $cache "LICENSE.txt") $dst
    $exe = Join-Path $dst "bgi-extractor.exe"
    $problem = Get-BgiExtractorProblem $exe
    if ($problem) { throw "bin\bgi-extractor\bgi-extractor.exe in dist\VNDub is rejected: $problem" }
    $Bundled.BgiVersion = $BgiExtractor.Version
    $Bundled.BgiSha256 = (Get-FileHash $exe -Algorithm SHA256).Hash.ToLower()
    Write-Host ("bgi-extractor: dist\VNDub\bin\bgi-extractor\bgi-extractor.exe {0} (commit {1}), {2:N0} KB, sha256 {3}; opens Unicode paths, LICENSE.txt present" -f $BgiExtractor.Version, $BgiExtractor.Commit, ((Get-Item $exe).Length / 1KB), $Bundled.BgiSha256)
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
    Step "bundled runtime (Visual C++ Redistributable, MinGit)"
    Add-BundledRuntime
    Step "bundled tools (bgi-extractor)"
    Add-BundledTools
    $sizeMb = [math]::Round((Get-ChildItem (Join-Path $Root "dist\VNDub") -Recurse -File | Measure-Object Length -Sum).Sum / 1MB)
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

    # -- VC++ redistributable, MinGit, bgi-extractor (PyInstaller recreated dist\VNDub, so they are added every time) ---
    Step "bundled runtime (Visual C++ Redistributable, MinGit)"
    Add-BundledRuntime
    Step "bundled tools (bgi-extractor)"
    Add-BundledTools
    $sizeMb = [math]::Round((Get-ChildItem (Join-Path $Root "dist\VNDub") -Recurse -File | Measure-Object Length -Sum).Sum / 1MB)
    Write-Host "folder size with the bundled runtime and tools: $sizeMb MB"

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
Write-Host "VC++ redist: $(if ($Bundled.RedistVersion) { "$($Bundled.RedistVersion), signed by $($Bundled.RedistSigner), sha256 $($Bundled.RedistSha256)" } else { 'NOT bundled' })"
Write-Host "git        : $(if ($Bundled.GitVersion) { "MinGit $($Bundled.GitVersion), $($Bundled.GitMB) MB on disk" } else { 'NOT bundled' })"
Write-Host "tools      : $(if ($Bundled.BgiVersion) { "bgi-extractor $($Bundled.BgiVersion), sha256 $($Bundled.BgiSha256)" } else { 'NOT bundled' })"
Write-Host "self-check : $(if ($null -eq $selfOk) { 'skipped' } elseif ($selfOk) { 'ok' } else { 'FAILED' })"
Write-Host "installer  : $(if ($installer -and (Test-Path $installer)) { $installer } else { 'not built' })"
Write-Host "time       : ${total}s$(if ($buildSeconds) { " (PyInstaller ${buildSeconds}s)" })$(if ($installerSeconds) { " (installer ${installerSeconds}s)" })"
exit 0

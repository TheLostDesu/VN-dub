<#
.SYNOPSIS
  Create backend virtual environments .venvs\<env> from envs\<env>.txt (uv, Python 3.11).

.EXAMPLE
  scripts\make_envs.ps1 -Env main
  scripts\make_envs.ps1 -Env main,funasr -Cpu
  scripts\make_envs.ps1 -Env all -SkipExtras
  scripts\make_envs.ps1 -Env main -EnvsDir "$env:LOCALAPPDATA\VNDub\envs"   # desktop app layout

.NOTES
  Envs are created under -EnvsDir (default: $env:VNDUB_ENVS_DIR, else <repo>\.venvs). The core finds
  them through the same variable (vndub.backends.runner.envs_root); -Cpu installs CPU torch wheels,
  without it the CUDA wheels of each env file (# @cuda) are used.
  -Python is a version (default 3.11) or the path of an interpreter; the desktop app passes the CPython it installed
  with uv (a real folder, not uv's minor-version junction, which Windows may refuse to follow).
  Directives inside envs\<env>.txt (comments, see envs\README.md):
    # @torch torch==X torchaudio==Y     installed first from the PyTorch index
    # @cuda cu128                       CUDA tag of that index (ignored with -Cpu)
    # @git <name> <url> <ref> [--recursive]   clone into .venvs\<env>\src\<name>
    # @req-from <name> <file>           install <file> from that checkout
    # @nodeps <requirement>             install with --no-deps
  envs\<env>+<extra>.txt are optional extras (failures are reported, not fatal, unless -Strict).
  Every env built gets <env>\vndub_env.json (adapters\common\envrecord.py): requirement hash, app version, git HEADs,
  installed torch kind, failed extras. Import-level check without weights: <env>\Scripts\python.exe -m adapters.<env>.main --probe
#>
param(
    [string[]]$Env = @("main"),
    [switch]$Cpu,
    [switch]$SkipExtras,
    [switch]$Strict,
    [switch]$Recreate,
    [string]$Python = "3.11",
    [string]$EnvsDir = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$ReqDir = Join-Path $Root "envs"
if (-not $EnvsDir) { $EnvsDir = $env:VNDUB_ENVS_DIR }
if (-not $EnvsDir) { $EnvsDir = Join-Path $Root ".venvs" }
$VenvsRoot = $EnvsDir
# build-time pins for sdist-only packages (see envs/build.constraints.txt)
$BuildConstraints = Join-Path $ReqDir "build.constraints.txt"
# the build record <env>\vndub_env.json (hash of the requirement files, app version, git HEADs, installed torch kind):
# written by this stdlib-only helper with the env's own python, read back by the app (vndub_app/installer.py env_states)
$RecordPy = Join-Path $Root "adapters\common\envrecord.py"
if (-not (Test-Path $RecordPy)) { $RecordPy = Join-Path $Root "backend\adapters\common\envrecord.py" }  # packaged app layout

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv not found on PATH (install: pip install uv, or https://docs.astral.sh/uv/)"
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Warning "git not found: envs that clone repositories (funasr+cosyvoice, rvc, seedvc, main+voxcpm, main+utmosv2) will fail"
}

$Env = @($Env | ForEach-Object { $_ -split "," } | Where-Object { $_ })
if ($Env -contains "all") {
    $Env = Get-ChildItem $ReqDir -Filter *.txt | Where-Object { $_.BaseName -notmatch "[+.]" } | ForEach-Object { $_.BaseName }
}

function Invoke-Checked([string]$What, [scriptblock]$Block) {
    # Native tools (uv, git) write progress to stderr; in Windows PowerShell 5.1 that becomes
    # a terminating NativeCommandError under "Stop", so judge success by the exit code only.
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Block 2>&1 | ForEach-Object { "$_" }
    } finally {
        $ErrorActionPreference = $prev
    }
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit $LASTEXITCODE)" }
}

function Get-Directives([string]$File) {
    $d = @{ torch = ""; cuda = "cu128"; git = @(); reqfrom = @(); nodeps = @() }
    foreach ($line in Get-Content -Encoding UTF8 $File) {
        if ($line -match "^\s*#\s*@torch\s+(.+)$") { $d.torch = $Matches[1].Trim() }
        elseif ($line -match "^\s*#\s*@cuda\s+(\S+)") { $d.cuda = $Matches[1].Trim() }
        elseif ($line -match "^\s*#\s*@git\s+(\S+)\s+(\S+)\s+(\S+)(\s+--recursive)?") {
            $d.git += , @{ name = $Matches[1]; url = $Matches[2]; ref = $Matches[3]; recursive = [bool]$Matches[4] }
        }
        elseif ($line -match "^\s*#\s*@req-from\s+(\S+)\s+(\S+)") { $d.reqfrom += , @{ name = $Matches[1]; file = $Matches[2] } }
        elseif ($line -match "^\s*#\s*@nodeps\s+(.+)$") { $d.nodeps += $Matches[1].Trim() }
    }
    return $d
}

function Install-ReqFile([string]$EnvName, [string]$Py, [string]$File, [string]$Constraints) {
    $d = Get-Directives $File
    $src = Join-Path $VenvsRoot "$EnvName\src"
    foreach ($g in $d.git) {
        $dst = Join-Path $src $g.name
        if (-not (Test-Path $dst)) {
            New-Item -ItemType Directory -Force $src | Out-Null
            if ($g.ref -match "^[0-9a-f]{40}$") {
                Invoke-Checked "git clone $($g.url)" { git clone $g.url $dst }
                Invoke-Checked "git checkout $($g.ref)" { git -C $dst checkout $g.ref }
                if ($g.recursive) { Invoke-Checked "git submodules" { git -C $dst submodule update --init --recursive } }
            } else {
                if ($g.recursive) { Invoke-Checked "git clone $($g.url)" { git clone --depth 1 --branch $g.ref --recurse-submodules $g.url $dst } }
                else { Invoke-Checked "git clone $($g.url)" { git clone --depth 1 --branch $g.ref $g.url $dst } }
            }
        } elseif ($g.ref -match "^[0-9a-f]{40}$") {
            # an existing checkout used to be kept as it was: a new pinned commit in envs\*.txt never reached a rebuilt env
            $headFile = Join-Path $dst ".git\HEAD"
            $head = if (Test-Path $headFile) { (Get-Content -Raw $headFile).Trim() } else { "" }
            if ($head -ne $g.ref) {
                Write-Host "   $($g.name): $head -> $($g.ref)"
                Invoke-Checked "git fetch $($g.url)" { git -C $dst fetch origin $g.ref }
                Invoke-Checked "git checkout $($g.ref)" { git -C $dst checkout --force $g.ref }
                if ($g.recursive) { Invoke-Checked "git submodules" { git -C $dst submodule update --init --recursive } }
            }
        }
    }
    $uvArgs = @("pip", "install", "--python", $Py, "-r", $File, "--build-constraint", $BuildConstraints)
    if ($Constraints) { $uvArgs += @("-c", $Constraints) }
    Invoke-Checked "uv pip install -r $File" { & uv @uvArgs }
    foreach ($r in $d.reqfrom) {
        $f = Join-Path (Join-Path $src $r.name) $r.file
        $a2 = @("pip", "install", "--python", $Py, "-r", $f, "--build-constraint", $BuildConstraints)
        if ($Constraints) { $a2 += @("-c", $Constraints) }
        Invoke-Checked "uv pip install -r $f" { & uv @a2 }
    }
    foreach ($n in $d.nodeps) {
        Invoke-Checked "uv pip install --no-deps $n" { & uv pip install --python $Py --no-deps --build-constraint $BuildConstraints $n }
    }
}

$summary = @()
foreach ($e in $Env) {
    $t0 = Get-Date
    $req = Join-Path $ReqDir "$e.txt"
    if (-not (Test-Path $req)) { throw "no requirements file $req" }
    if (-not (Get-Content -Encoding UTF8 $req | Where-Object { $_ -match "^\s*[^#\s]" -or $_ -match "^\s*#\s*@" })) {
        $summary += "$e : nothing to install (see $req)"
        continue
    }
    $venv = Join-Path $VenvsRoot $e
    if ($Recreate -and (Test-Path $venv)) { Remove-Item -Recurse -Force $venv }
    # no record while the build runs: a failed or cancelled build leaves an env the app reports as incomplete
    foreach ($f in @("vndub_env.json", "vndub_probe.json")) { Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $venv $f) }
    $py = Join-Path $venv "Scripts\python.exe"
    Write-Host "== env $e ($(if ($Cpu) { 'CPU' } else { 'CUDA' }))" -ForegroundColor Cyan
    if (-not (Test-Path (Join-Path $venv "pyvenv.cfg"))) {
        # a half-created venv (interrupted run) has python.exe but no pyvenv.cfg
        if (Test-Path $venv) { Remove-Item -Recurse -Force $venv }
        Invoke-Checked "uv venv $venv" { uv venv $venv --python $Python }
    }

    $d = Get-Directives $req
    $constraints = ""
    if ($d.torch) {
        $index = if ($Cpu) { "https://download.pytorch.org/whl/cpu" } else { "https://download.pytorch.org/whl/$($d.cuda)" }
        $pkgs = $d.torch -split "\s+"
        # torch==2.8.0 is satisfied by an installed 2.8.0+cpu (PEP 440 ignores the local tag): asking for CUDA over a
        # CPU build changed nothing. When the installed kind differs from the requested one, reinstall the torch packages.
        $want = if ($Cpu) { "cpu" } else { "cuda" }
        $have = ""
        try { $have = "$(& $py $RecordPy torch-kind $venv | Select-Object -Last 1)".Trim() } catch { $have = "" }
        $reinstall = @()
        if ($have -and $have -ne "none" -and $have -ne $want) {
            Write-Host "   torch is a $have build, $want requested: reinstalling $($d.torch)"
            foreach ($p in $pkgs) { $reinstall += @("--reinstall-package", ($p -split "[=<>!~]")[0]) }
        }
        Invoke-Checked "torch install" { & uv pip install --python $py --index-url $index @reinstall @pkgs }
        $constraints = Join-Path $venv "torch-constraints.txt"
        Set-Content -Encoding ASCII -Path $constraints -Value $pkgs
    }
    Install-ReqFile $e $py $req $constraints

    $extrasFailed = @()
    if (-not $SkipExtras) {
        foreach ($x in Get-ChildItem $ReqDir -Filter "$e+*.txt") {
            try {
                Write-Host "   extra $($x.Name)" -ForegroundColor Cyan
                Install-ReqFile $e $py $x.FullName $constraints
            } catch {
                if ($Strict) { throw }
                Write-Warning "extra $($x.Name) failed: $_"
                $extrasFailed += $x.Name
            }
        }
    }
    $cpuFlag = if ($Cpu) { "1" } else { "0" }
    Invoke-Checked "build record" { & $py $RecordPy write --venv $venv --env $e --req-dir $ReqDir --cpu $cpuFlag "--torch=$($d.torch)" "--extras-failed=$($extrasFailed -join ',')" }  # name=value: PowerShell 5.1 drops an empty "" argument
    $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    $summary += "$e : ok in $mins min" + $(if ($extrasFailed.Count) { " (extras failed: $($extrasFailed -join ', '))" } else { "" })
}
Write-Host ""
$summary | ForEach-Object { Write-Host $_ -ForegroundColor Green }
Write-Host "selftest: $VenvsRoot\<env>\Scripts\python.exe -m adapters.<env>.main --selftest   (or: uv run vndub models selftest <backend>)"
exit 0

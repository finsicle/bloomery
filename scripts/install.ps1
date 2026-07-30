# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
<#
.SYNOPSIS
    Install bloomery on Windows.

.DESCRIPTION
    Core is installed first on purpose. It is a few megabytes, so `bloomery
    doctor` runs within seconds and tells you what this machine can do before you
    commit to a multi-gigabyte PyTorch download.

    Note that WSL2 is the supported path on Windows, not native PowerShell.
    Distributed training and per-job memory limits are both more reliable there.
    This script works natively, but expect rough edges.

.PARAMETER WithTraining
    Also install PyTorch and the training stack.

.EXAMPLE
    .\scripts\install.ps1
    .\scripts\install.ps1 -WithTraining
#>

[CmdletBinding()]
param(
    [switch]$WithTraining
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$PythonVersion = '3.12'

function Write-Info { param($Message) Write-Host "==> $Message" -ForegroundColor Cyan }
function Write-Warn { param($Message) Write-Host "==> $Message" -ForegroundColor Yellow }
function Stop-WithError { param($Message) Write-Host "==> $Message" -ForegroundColor Red; exit 1 }

# Run from the repository root regardless of where this was invoked.
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
Set-Location $RepoRoot

if (-not (Test-Path 'pyproject.toml')) {
    Stop-WithError "pyproject.toml not found in $RepoRoot"
}

Write-Info 'checking for WSL2'
if ($env:WSL_DISTRO_NAME) {
    Write-Info "running inside WSL ($env:WSL_DISTRO_NAME) — use install.sh instead"
    exit 1
}
Write-Warn 'native Windows is best-effort; WSL2 is the supported path'

# --------------------------------------------------------------------------- #
# uv
# --------------------------------------------------------------------------- #
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Info 'installing uv'
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression

    # The installer edits the persisted user PATH, which does not affect the
    # running session.
    $UvBin = Join-Path $env:USERPROFILE '.local\bin'
    if (Test-Path (Join-Path $UvBin 'uv.exe')) {
        $env:PATH = "$UvBin;$env:PATH"
    }
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Stop-WithError 'uv installed but not on PATH; open a new terminal and re-run'
    }
}
Write-Info "uv $((uv --version) -split ' ' | Select-Object -Index 1)"

# --------------------------------------------------------------------------- #
# virtualenv
# --------------------------------------------------------------------------- #
if (-not (Test-Path '.venv')) {
    Write-Info "creating .venv on python $PythonVersion"
    uv venv --python $PythonVersion
} else {
    Write-Info 'reusing existing .venv'
}

$Bloomery = Join-Path $RepoRoot '.venv\Scripts\bloomery.exe'

# --------------------------------------------------------------------------- #
# core install, then immediately report
# --------------------------------------------------------------------------- #
Write-Info 'installing bloomery core'
uv pip install --quiet -e .

Write-Info 'probing hardware'
Write-Host ''
& $Bloomery doctor
$DoctorStatus = $LASTEXITCODE
Write-Host ''

if ($DoctorStatus -ne 0) {
    Write-Warn 'doctor reported a blocking problem — see the notes above'
}

# --------------------------------------------------------------------------- #
# training stack
# --------------------------------------------------------------------------- #
if ($WithTraining) {
    Write-Info 'installing PyTorch and trainers (several GB, this takes a while)'
    # --torch-backend=auto inspects the CUDA driver, AMD GPU version and Intel
    # GPU presence and resolves the matching wheel index. It is only available
    # on `uv pip`, which is why this is not a `uv sync`.
    uv pip install --torch-backend=auto -e ".[train]"
    Write-Info 'verifying torch can see the hardware'
    Write-Host ''
    & $Bloomery doctor
} else {
    Write-Host @'
Core is installed. To add PyTorch and the training stack:

  uv pip install --torch-backend=auto -e ".[train]"

or re-run this script with -WithTraining.
'@
}

Write-Host ''
Write-Info 'activate with: .venv\Scripts\Activate.ps1'

# Build dist/agent.exe (single file, console app) with PyInstaller.
# Usage (from anywhere): powershell -ExecutionPolicy Bypass -File scripts/build.ps1 [-DistPath <dir>]
param(
    # Output folder for agent.exe (default: <repo>/dist). Useful when distgent.exe is running and locked.
    [string]$DistPath = ""
)
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
if (-not $DistPath) { $DistPath = Join-Path $root "dist" }
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Virtual env not found at $python. Create it and run: pip install -r requirements.txt"
}

Push-Location $root
try {
    # Data paths are absolute because PyInstaller resolves them relative to --specpath.
    & $python -m PyInstaller `
        --noconfirm --clean `
        --onefile --console `
        --name agent `
        --paths $root `
        --specpath (Join-Path $root "build") `
        --distpath $DistPath `
        --workpath (Join-Path $root "build") `
        --add-data "$root\agent\models.yaml;agent" `
        --add-data "$root\agent\assets;agent\assets" `
        (Join-Path $root "agent\cli.py")
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }
    Write-Host "Built: $(Join-Path $DistPath 'agent.exe')"
}
finally {
    Pop-Location
}

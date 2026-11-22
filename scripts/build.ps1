# Build dist/agent.exe (single file, console app) with PyInstaller.
# Usage (from anywhere): powershell -ExecutionPolicy Bypass -File scripts/build.ps1 [-DistPath <dir>]
#
# Without -DistPath the target is <repo>/dist. If dist\agent.exe is locked (a window or a
# VS Code terminal is still running it), the build goes to <repo>/dist-new instead and asks
# you to close the window. The next build that finds dist\agent.exe unlocked builds into
# dist again and removes dist-new, so dist always ends up with the newest version.
param(
    # Explicit output folder for agent.exe; skips the dist / dist-new logic.
    [string]$DistPath = ""
)
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Virtual env not found at $python. Create it and run: pip install -r requirements.txt"
}

function Test-FileLocked([string]$Path) {
    if (-not (Test-Path $Path)) { return $false }
    try {
        $stream = [System.IO.File]::Open($Path, 'Open', 'ReadWrite', 'None')
        $stream.Close()
        return $false
    }
    catch {
        return $true
    }
}

$dist = Join-Path $root "dist"
$distNew = Join-Path $root "dist-new"
$fallback = $false
if (-not $DistPath) {
    if (Test-FileLocked (Join-Path $dist "agent.exe")) {
        $DistPath = $distNew
        $fallback = $true
    }
    else {
        $DistPath = $dist
    }
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
        --collect-data pptx `
        --collect-data docx `
        --collect-all playwright `
        --hidden-import agent.mcp_servers.source_reader `
        (Join-Path $root "agent\cli.py")
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }
    Write-Host "Built: $(Join-Path $DistPath 'agent.exe')"

    if ($fallback) {
        Write-Host ""
        Write-Host "NOTE: dist\agent.exe is locked: a window or a VS Code terminal is still running it." -ForegroundColor Yellow
        Write-Host "      The new version was built to dist-new\agent.exe instead." -ForegroundColor Yellow
        Write-Host "      Close every agent.exe window (in VS Code: type /quit in the terminal or close it)," -ForegroundColor Yellow
        Write-Host "      then run this script again to replace dist with the new version." -ForegroundColor Yellow
        Write-Host "      Do not keep working from dist-new: its data\ folder is removed on the next build." -ForegroundColor Yellow
    }
    elseif ($DistPath -eq $dist -and (Test-Path $distNew)) {
        Remove-Item -Recurse -Force $distNew
        Write-Host "dist now has the newest version; removed dist-new."
    }
}
finally {
    Pop-Location
}

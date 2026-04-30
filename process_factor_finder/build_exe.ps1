$ErrorActionPreference = "Stop"

Write-Host "== Process Factor Finder one-file EXE build =="
Write-Host "Project: $PSScriptRoot"

Set-Location $PSScriptRoot

Write-Host "`n[1/3] Checking PyInstaller..."
python -m pip show pyinstaller | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "PyInstaller is not installed. Installing now..."
    python -m pip install pyinstaller
}

Write-Host "`n[2/3] Building ProcessFactorFinder.exe..."
python -m PyInstaller --clean --noconfirm ProcessFactorFinder.spec

Write-Host "`n[3/3] Done."
Write-Host "EXE path: $PSScriptRoot\dist\ProcessFactorFinder.exe"
Write-Host "Run it by double-clicking, or with:"
Write-Host "  .\dist\ProcessFactorFinder.exe"

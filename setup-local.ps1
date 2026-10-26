$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$localPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $localPython)) {
    py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 is required. Install it and retry.' }
}
if (Test-Path -LiteralPath 'artifacts/torch-2.11.0+cu128-cp311-cp311-win_amd64.whl') {
    & $localPython -m pip install --no-index --find-links artifacts --find-links artifacts/wheels -r requirements-local.lock
} else {
    & $localPython -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements-local.lock
}
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $localPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency verification failed.' }

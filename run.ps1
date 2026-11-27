$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $projectPython)) {
        py -3.11 -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 is required.' }
        & $projectPython -m pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    }
    & $projectPython -m flask --app app --no-debug run --host 127.0.0.1 --port 5000
} finally {
    Pop-Location
}

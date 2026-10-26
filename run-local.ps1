param([int]$Port = 8501)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& "$PSScriptRoot\.venv\Scripts\python.exe" -m streamlit run scripts/app.py --server.address 127.0.0.1 --server.port $Port --server.headless true --browser.gatherUsageStats false

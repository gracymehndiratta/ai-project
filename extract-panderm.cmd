@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe -u scripts\extract_panderm.py %*

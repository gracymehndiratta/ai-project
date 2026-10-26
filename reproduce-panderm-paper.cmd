@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe -u scripts\fetch_panderm_paper.py
if errorlevel 1 exit /b %errorlevel%
.venv\Scripts\python.exe -u scripts\run_panderm_paper_reference.py
if errorlevel 1 exit /b %errorlevel%
.venv\Scripts\python.exe scripts\report_panderm_paper.py

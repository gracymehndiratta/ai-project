@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe -u scripts\train_panderm_capacity.py
if errorlevel 1 exit /b %errorlevel%
.venv\Scripts\python.exe scripts\report_panderm_capacity.py

@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
.venv\Scripts\python.exe -u scripts\train_specialists.py %*
if errorlevel 1 goto end
.venv\Scripts\python.exe scripts\report_specialists.py
:end
pause

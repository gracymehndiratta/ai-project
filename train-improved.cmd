@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
.venv\Scripts\python.exe -u scripts\train_improved.py %*
pause

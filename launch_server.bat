@echo off
setlocal

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo Missing virtual environment at "%~dp0venv".
    pause
    exit /b 1
)

set MONGODB_URI=mongodb://127.0.0.1:27017
set MONGODB_DB=telemetry_db

echo Launching backend from: %cd%
start http://127.0.0.1:8000/admin/users

"venv\Scripts\python.exe" run_server.py

pause

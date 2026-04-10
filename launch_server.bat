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
echo Ensuring public TCP port 8000 mapping...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0ensure_public_port_8000.ps1"
if errorlevel 1 (
    echo Warning: could not refresh the UPnP public port mapping.
    echo The backend will still start for local/LAN use.
)
start http://127.0.0.1:8000/admin/login

"venv\Scripts\python.exe" run_server.py

pause


@echo off
setlocal

set "APP_DIR=%~dp0"
set "SCRIPT=%APP_DIR%start_mvp.ps1"

if not exist "%SCRIPT%" (
  echo start_mvp.ps1 not found:
  echo %SCRIPT%
  pause
  exit /b 1
)

start "TRIBE Review MVP" powershell -ExecutionPolicy Bypass -File "%SCRIPT%"
exit /b 0

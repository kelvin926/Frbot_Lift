@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=R:\Frbot_Lift"

cd /d "%PROJECT_ROOT%"
title Frbot Lift Visible Runner

powershell.exe -NoLogo -ExecutionPolicy Bypass -File "%SCRIPT_DIR%run_python.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"

echo.
echo [Frbot Lift] Process exited with code %EXIT_CODE%.
exit /b %EXIT_CODE%

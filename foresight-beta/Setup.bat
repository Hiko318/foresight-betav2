@echo off
REM Foresight Setup Wizard Launcher
REM This launches the PowerShell-based setup wizard

title Foresight Setup

echo.
echo ================================================
echo    Foresight Setup Wizard
echo ================================================
echo.
echo Starting setup wizard...
echo.

REM Launch PowerShell setup wizard with execution policy bypass
powershell.exe -ExecutionPolicy Bypass -File "%~dp0Setup-Wizard.ps1"

REM Check if PowerShell script ran successfully
if %ERRORLEVEL% EQU 0 (
    echo.
    echo Setup completed successfully!
    echo.
) else (
    echo.
    echo Setup encountered an error.
    echo Please try running as administrator or check the logs.
    echo.
    pause
)

exit /b %ERRORLEVEL%
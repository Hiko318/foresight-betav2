@echo off
echo Starting Argus...
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Check if we're in the installation directory or source directory
if exist "package.json" (
    echo Found package.json, starting from current directory...
    npm start
) else (
    echo Package.json not found in current directory.
    echo Please run this script from the Argus installation folder.
    echo.
    pause
    exit /b 1
)

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo Application exited with an error.
    echo Make sure all dependencies are installed.
    pause
)
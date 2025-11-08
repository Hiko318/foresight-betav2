@echo off
title Foresight - Phone Mirror & Detection
echo.
echo   ███████╗ ██████╗ ██████╗ ███████╗███████╗██╗ ██████╗ ██╗  ██╗████████╗
echo   ██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔════╝██║██╔════╝ ██║  ██║╚══██╔══╝
echo   █████╗  ██║   ██║██████╔╝█████╗  ███████╗██║██║  ███╗███████║   ██║   
echo   ██╔══╝  ██║   ██║██╔══██╗██╔══╝  ╚════██║██║██║   ██║██╔══██║   ██║   
echo   ██║     ╚██████╔╝██║  ██║███████╗███████║██║╚██████╔╝██║  ██║   ██║   
echo   ╚═╝      ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   
echo.
echo Starting Foresight Application...
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Check and install dependencies
echo Checking dependencies...

REM Check for Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo Node.js not found. Installing Node.js...
    echo Please download and install Node.js from https://nodejs.org/
    echo After installation, restart this script.
    pause
    exit /b 1
) else (
    echo ✓ Node.js found
)

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found. Installing Python...
    echo Please download and install Python from https://python.org/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
) else (
    echo ✓ Python found
)

REM Check for scrcpy (optional for phone mirroring)
scrcpy --version >nul 2>&1
if errorlevel 1 (
    echo Warning: scrcpy not found. Phone mirroring will not work.
    echo To enable phone mirroring, install scrcpy from:
    echo https://github.com/Genymobile/scrcpy/releases
) else (
    echo ✓ scrcpy found
)

REM Activate Python virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    echo Activating Python virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo Warning: Python virtual environment not found. Some features may not work properly.
)

REM Install Node.js dependencies if needed
if exist "package.json" (
    echo Found package.json, checking Node.js dependencies...
    if not exist "node_modules" (
        echo Installing Node.js dependencies...
        npm install
        if errorlevel 1 (
            echo Failed to install Node.js dependencies.
            pause
            exit /b 1
        )
    ) else (
        echo ✓ Node.js dependencies found
    )
) else (
    echo Package.json not found in current directory.
    echo Please run this script from the Foresight installation folder.
    echo.
    pause
    exit /b 1
)

REM Install Python dependencies if needed
if exist "requirements.txt" (
    echo Checking Python dependencies...
    pip install -r requirements.txt --quiet
    if errorlevel 1 (
        echo Warning: Some Python dependencies may not have installed correctly.
        echo This may affect YOLO detection functionality.
    ) else (
        echo ✓ Python dependencies installed/updated
    )
)

echo.
echo Starting Foresight Application...
npm start

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo An error occurred while starting the application.
    pause
)
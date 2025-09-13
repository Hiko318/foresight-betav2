@echo off
setlocal enabledelayedexpansion
color 0A
title Argus Installer

echo ========================================
echo    ARGUS INSTALLER
echo ========================================
echo.
echo This installer will:
echo 1. Check for required software
echo 2. Install missing dependencies
echo 3. Set up the application
echo 4. Launch Argus
echo.
pause
cls

REM Change to script directory
cd /d "%~dp0"

echo ========================================
echo    CHECKING SYSTEM REQUIREMENTS
echo ========================================
echo.

REM Check for Node.js
echo [1/3] Checking Node.js...
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js not found!
    echo.
    echo Please install Node.js from: https://nodejs.org
    echo Download the LTS version and run the installer.
    echo.
    echo After installing Node.js, run this installer again.
    pause
    exit /b 1
) else (
    for /f "tokens=*" %%i in ('node --version') do set NODE_VERSION=%%i
    echo ✅ Node.js found: !NODE_VERSION!
)

REM Check for Python
echo [2/3] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found!
    echo.
    echo Please install Python from: https://python.org
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    echo After installing Python, run this installer again.
    pause
    exit /b 1
) else (
    for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
    echo ✅ Python found: !PYTHON_VERSION!
)

REM Check for scrcpy
echo [3/3] Checking scrcpy...
scrcpy --version >nul 2>&1
if errorlevel 1 (
    echo ⚠️  scrcpy not found!
    echo.
    echo scrcpy is required for phone mirroring functionality.
    echo You can download it from: https://github.com/Genymobile/scrcpy
    echo.
    echo The app will still work without scrcpy, but phone mirroring will be disabled.
    echo.
    set /p CONTINUE="Continue without scrcpy? (y/n): "
    if /i "!CONTINUE!" neq "y" (
        echo Installation cancelled.
        pause
        exit /b 1
    )
    echo ⚠️  Continuing without scrcpy...
) else (
    for /f "tokens=*" %%i in ('scrcpy --version 2^>^&1 ^| findstr "scrcpy"') do set SCRCPY_VERSION=%%i
    echo ✅ scrcpy found: !SCRCPY_VERSION!
)

echo.
echo ========================================
echo    INSTALLING DEPENDENCIES
echo ========================================
echo.

REM Install Node.js dependencies
echo [1/3] Installing Node.js dependencies...
if not exist "node_modules" (
    echo Installing npm packages...
    npm install
    if errorlevel 1 (
        echo ❌ Failed to install Node.js dependencies!
        pause
        exit /b 1
    )
    echo ✅ Node.js dependencies installed successfully!
) else (
    echo ✅ Node.js dependencies already installed!
)

REM Create Python virtual environment
echo [2/3] Setting up Python environment...
if not exist ".venv" (
    echo Creating Python virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ❌ Failed to create Python virtual environment!
        pause
        exit /b 1
    )
    echo ✅ Python virtual environment created!
) else (
    echo ✅ Python virtual environment already exists!
)

REM Install Python dependencies
echo [3/3] Installing Python dependencies...
echo Activating virtual environment and installing packages...
call .venv\Scripts\activate.bat
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install Python dependencies!
    pause
    exit /b 1
)
echo ✅ Python dependencies installed successfully!

echo.
echo ========================================
echo    INSTALLATION COMPLETE!
echo ========================================
echo.
echo ✅ All dependencies installed successfully!
echo ✅ Argus is ready to use!
echo.
echo Starting Argus...
echo.

REM Launch the application
timeout /t 3 /nobreak >nul
npm start

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo ❌ Application exited with an error.
    echo.
    echo You can try running it manually with: npm start
    echo Or use the start-argus.bat file for quick access.
    pause
)
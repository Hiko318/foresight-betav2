@echo off
setlocal EnableDelayedExpansion
title Foresight Launcher (Auto Setup)

echo.
echo =============================================================
echo   Foresight - Auto Setup and Launch
echo =============================================================
echo.

REM Change to the script directory
cd /d "%~dp0"

REM --------------------------------------------------------------
REM Helper: ensure winget available
REM --------------------------------------------------------------
where winget >nul 2>&1
if errorlevel 1 (
  set HAS_WINGET=0
  echo Winget not found. Automatic installation will be limited.
) else (
  set HAS_WINGET=1
)

REM --------------------------------------------------------------
REM Ensure Node.js
REM --------------------------------------------------------------
node --version >nul 2>&1
if errorlevel 1 (
  echo Node.js not found.
  if "%HAS_WINGET%"=="1" (
    echo Installing Node.js LTS via winget...
    winget install -e --id OpenJS.NodeJS.LTS --accept-package-agreements --accept-source-agreements
  ) else (
    echo Please install Node.js from https://nodejs.org/ and rerun.
  )
)
REM Try to add default Node path for current session
if exist "%ProgramFiles%\nodejs\node.exe" set "PATH=%ProgramFiles%\nodejs;%PATH%"
if exist "%LocalAppData%\Programs\nodejs\node.exe" set "PATH=%LocalAppData%\Programs\nodejs;%PATH%"

node --version >nul 2>&1
if errorlevel 1 (
  echo Node.js is still not available. Exiting.
  pause & exit /b 1
)
echo ✓ Node.js ready

REM --------------------------------------------------------------
REM Ensure Python (and Windows launcher py)
REM --------------------------------------------------------------
py -3 --version >nul 2>&1
if errorlevel 1 (
  echo Python launcher not found.
  if "%HAS_WINGET%"=="1" (
    echo Installing Python 3 via winget...
    winget install -e --id Python.Python.3 --accept-package-agreements --accept-source-agreements
  ) else (
    echo Please install Python 3 from https://python.org/ and rerun.
  )
)
REM Try common Python paths
REM Add common Python install paths to the current session PATH
for /d %%P in ("%LocalAppData%\Programs\Python\Python3*") do (
  if exist "%%~fP\python.exe" set "PATH=%%~fP;%%~fP\Scripts;%PATH%"
)
for /d %%P in ("%ProgramFiles%\Python3*") do (
  if exist "%%~fP\python.exe" set "PATH=%%~fP;%%~fP\Scripts;%PATH%"
)
for /d %%P in ("%ProgramFiles(x86)%\Python3*") do (
  if exist "%%~fP\python.exe" set "PATH=%%~fP;%%~fP\Scripts;%PATH%"
)

py -3 --version >nul 2>&1 || (
  echo Python still not available. Exiting.
  pause & exit /b 1
)
echo ✓ Python ready

REM --------------------------------------------------------------
REM Optional: Ensure Git (not required to run app)
REM --------------------------------------------------------------
git --version >nul 2>&1
if errorlevel 1 (
  if "%HAS_WINGET%"=="1" (
    echo Installing Git via winget...
    winget install -e --id Git.Git --accept-package-agreements --accept-source-agreements
  ) else (
    echo Git not found ^(optional^).
  )
)

REM --------------------------------------------------------------
REM Optional: Ensure scrcpy for phone mirroring
REM --------------------------------------------------------------
scrcpy --version >nul 2>&1
if errorlevel 1 (
  if "%HAS_WINGET%"=="1" (
    echo Installing scrcpy via winget...
    winget install -e --id Genymobile.scrcpy --accept-package-agreements --accept-source-agreements
  ) else (
    echo scrcpy not found. Phone mirroring will be disabled.
  )
) else (
  echo ✓ scrcpy ready
)

REM --------------------------------------------------------------
REM Python dependencies (prefer venv)
REM --------------------------------------------------------------
set VENV_DIR=.venv
if not exist "%VENV_DIR%\Scripts\activate.bat" (
  echo Creating Python virtual environment...
  py -3 -m venv "%VENV_DIR%"
)
if exist "%VENV_DIR%\Scripts\activate.bat" (
  echo Activating virtual environment and installing requirements...
  call "%VENV_DIR%\Scripts\activate.bat"
  py -3 -m pip install --upgrade pip >nul 2>&1
  if exist "requirements.txt" (
    py -3 -m pip install -r requirements.txt
  )
) else (
  echo Virtual environment not available, installing requirements globally...
  if exist "requirements.txt" (
    py -3 -m pip install -r requirements.txt
  )
)

REM --------------------------------------------------------------
REM Node dependencies
REM --------------------------------------------------------------
if exist package-lock.json (
  echo Installing Node dependencies ^(npm ci^)...
  call npm ci
  if errorlevel 1 (
    echo npm ci failed, falling back to npm install...
    call npm install
  )
) else (
  echo Installing Node dependencies ^(npm install^)...
  call npm install
)

REM --------------------------------------------------------------
REM Generate icon if missing
REM --------------------------------------------------------------
if not exist "assets\icon.ico" (
  echo Generating application icon...
  npm run gen:ico
)

echo.
echo Launching Foresight...
echo.
npm start
if errorlevel 1 (
  echo.
  echo An error occurred while starting Foresight.
  pause
)

endlocal
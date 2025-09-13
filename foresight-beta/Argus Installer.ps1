# Argus Complete Installer
# This script will check for dependencies, install them, and launch the app

# Set console properties
$Host.UI.RawUI.WindowTitle = "Argus Installer"
$Host.UI.RawUI.BackgroundColor = "Black"
$Host.UI.RawUI.ForegroundColor = "Green"
Clear-Host

# Set location to script directory
Set-Location $PSScriptRoot

function Write-Header {
    param([string]$Title)
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "    $Title" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Error {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor Red
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠️  $Message" -ForegroundColor Yellow
}

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ️  $Message" -ForegroundColor Cyan
}

try {
    Write-Header "ARGUS INSTALLER"
    
    Write-Host "This installer will:" -ForegroundColor White
    Write-Host "1. Check for required software" -ForegroundColor Gray
    Write-Host "2. Install missing dependencies" -ForegroundColor Gray
    Write-Host "3. Set up the application" -ForegroundColor Gray
    Write-Host "4. Launch Argus" -ForegroundColor Gray
    Write-Host ""
    
    $continue = Read-Host "Press Enter to continue or 'q' to quit"
    if ($continue -eq 'q') {
        Write-Host "Installation cancelled." -ForegroundColor Yellow
        exit 0
    }
    
    Clear-Host
    Write-Header "CHECKING SYSTEM REQUIREMENTS"
    
    # Check Node.js
    Write-Host "[1/3] Checking Node.js..." -ForegroundColor Yellow
    try {
        $nodeVersion = node --version 2>$null
        if ($nodeVersion) {
            Write-Success "Node.js found: $nodeVersion"
        } else {
            throw "Node.js not found"
        }
    } catch {
        Write-Error "Node.js not found!"
        Write-Host ""
        Write-Info "Please install Node.js from: https://nodejs.org"
        Write-Info "Download the LTS version and run the installer."
        Write-Host ""
        Write-Host "After installing Node.js, run this installer again." -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }
    
    # Check Python
    Write-Host "[2/3] Checking Python..." -ForegroundColor Yellow
    try {
        $pythonVersion = python --version 2>$null
        if ($pythonVersion) {
            Write-Success "Python found: $pythonVersion"
        } else {
            throw "Python not found"
        }
    } catch {
        Write-Error "Python not found!"
        Write-Host ""
        Write-Info "Please install Python from: https://python.org"
        Write-Info "Make sure to check 'Add Python to PATH' during installation."
        Write-Host ""
        Write-Host "After installing Python, run this installer again." -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }
    
    # Check scrcpy
    Write-Host "[3/3] Checking scrcpy..." -ForegroundColor Yellow
    try {
        $scrcpyVersion = scrcpy --version 2>$null
        if ($scrcpyVersion) {
            Write-Success "scrcpy found: $($scrcpyVersion -split '\n' | Select-Object -First 1)"
        } else {
            throw "scrcpy not found"
        }
    } catch {
        Write-Warning "scrcpy not found!"
        Write-Host ""
        Write-Info "scrcpy is required for phone mirroring functionality."
        Write-Info "You can download it from: https://github.com/Genymobile/scrcpy"
        Write-Host ""
        Write-Host "The app will still work without scrcpy, but phone mirroring will be disabled." -ForegroundColor Gray
        Write-Host ""
        $continue = Read-Host "Continue without scrcpy? (y/n)"
        if ($continue.ToLower() -ne 'y') {
            Write-Host "Installation cancelled." -ForegroundColor Yellow
            exit 1
        }
        Write-Warning "Continuing without scrcpy..."
    }
    
    Write-Host ""
    Write-Header "INSTALLING DEPENDENCIES"
    
    # Install Node.js dependencies
    Write-Host "[1/3] Installing Node.js dependencies..." -ForegroundColor Yellow
    if (-not (Test-Path "node_modules")) {
        Write-Host "Installing npm packages..." -ForegroundColor Gray
        npm install
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install Node.js dependencies"
        }
        Write-Success "Node.js dependencies installed successfully!"
    } else {
        Write-Success "Node.js dependencies already installed!"
    }
    
    # Create Python virtual environment
    Write-Host "[2/3] Setting up Python environment..." -ForegroundColor Yellow
    if (-not (Test-Path ".venv")) {
        Write-Host "Creating Python virtual environment..." -ForegroundColor Gray
        python -m venv .venv
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create Python virtual environment"
        }
        Write-Success "Python virtual environment created!"
    } else {
        Write-Success "Python virtual environment already exists!"
    }
    
    # Install Python dependencies
    Write-Host "[3/3] Installing Python dependencies..." -ForegroundColor Yellow
    Write-Host "Activating virtual environment and installing packages..." -ForegroundColor Gray
    & ".venv\Scripts\Activate.ps1"
    pip install -r requirements.txt --quiet
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install Python dependencies"
    }
    Write-Success "Python dependencies installed successfully!"
    
    Write-Host ""
    Write-Header "INSTALLATION COMPLETE!"
    
    Write-Success "All dependencies installed successfully!"
    Write-Success "Argus is ready to use!"
    Write-Host ""
    Write-Host "Starting Argus..." -ForegroundColor Green
    Write-Host ""
    
    # Launch the application
    Start-Sleep -Seconds 2
    npm start
    
} catch {
    Write-Host ""
    Write-Error "Installation failed: $_"
    Write-Host ""
    Write-Info "You can try running the app manually with: npm start"
    Write-Info "Or use the start-argus.bat file for quick access."
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# Keep window open if there's an error
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Error "Application exited with an error."
    Write-Host ""
    Write-Info "You can try running it manually with: npm start"
    Write-Info "Or use the start-argus.bat file for quick access."
    Read-Host "Press Enter to exit"
}
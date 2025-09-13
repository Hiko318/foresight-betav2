# Argus Launcher
Write-Host "Starting Argus..." -ForegroundColor Green
Write-Host ""

# Set location to script directory
Set-Location $PSScriptRoot

try {
    # Check if node_modules exists
    if (-not (Test-Path "node_modules")) {
        Write-Host "Installing Node.js dependencies..." -ForegroundColor Yellow
        npm install
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install Node.js dependencies"
        }
    }

    # Check if Python virtual environment exists
    if (-not (Test-Path ".venv")) {
        Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
        python -m venv .venv
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create Python virtual environment"
        }
    }

    # Activate virtual environment and install Python dependencies
    Write-Host "Checking Python dependencies..." -ForegroundColor Yellow
    & ".venv\Scripts\Activate.ps1"
    pip install -r requirements.txt --quiet
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install Python dependencies"
    }

    # Start the application
    Write-Host "Starting Argus application..." -ForegroundColor Green
    Write-Host ""
    npm start

} catch {
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host "Press any key to exit..." -ForegroundColor Yellow
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# Keep window open if there's an error
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Application exited with an error." -ForegroundColor Red
    Write-Host "Press any key to exit..." -ForegroundColor Yellow
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}
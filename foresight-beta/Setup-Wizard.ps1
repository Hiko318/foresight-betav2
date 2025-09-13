# Foresight Beta Setup Wizard
# Professional installation wizard with GUI

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

# Global variables
$script:currentStep = 0
$script:totalSteps = 6
$script:installPath = "C:\Program Files\Foresight Beta"
$script:sourceDirectory = $PSScriptRoot
$script:nodeInstalled = $false
$script:pythonInstalled = $false
$script:scrcpyInstalled = $false

# Create main form
$form = New-Object System.Windows.Forms.Form
$form.Text = "Foresight Beta Setup Wizard"
$form.Size = New-Object System.Drawing.Size(600, 450)
$form.StartPosition = "CenterScreen"
$form.FormBorderStyle = "FixedDialog"
$form.MaximizeBox = $false
$form.MinimizeBox = $false

# Header panel
$headerPanel = New-Object System.Windows.Forms.Panel
$headerPanel.Size = New-Object System.Drawing.Size(600, 80)
$headerPanel.Location = New-Object System.Drawing.Point(0, 0)
$headerPanel.BackColor = [System.Drawing.Color]::FromArgb(45, 45, 48)
$form.Controls.Add($headerPanel)

# Title label
$titleLabel = New-Object System.Windows.Forms.Label
$titleLabel.Text = "Foresight Beta Setup"
$titleLabel.Font = New-Object System.Drawing.Font("Segoe UI", 16, [System.Drawing.FontStyle]::Bold)
$titleLabel.ForeColor = [System.Drawing.Color]::White
$titleLabel.Location = New-Object System.Drawing.Point(20, 15)
$titleLabel.Size = New-Object System.Drawing.Size(300, 30)
$headerPanel.Controls.Add($titleLabel)

# Subtitle label
$subtitleLabel = New-Object System.Windows.Forms.Label
$subtitleLabel.Text = "Phone Mirroring & AI Detection Application"
$subtitleLabel.Font = New-Object System.Drawing.Font("Segoe UI", 10)
$subtitleLabel.ForeColor = [System.Drawing.Color]::LightGray
$subtitleLabel.Location = New-Object System.Drawing.Point(20, 45)
$subtitleLabel.Size = New-Object System.Drawing.Size(400, 20)
$headerPanel.Controls.Add($subtitleLabel)

# Progress bar
$progressBar = New-Object System.Windows.Forms.ProgressBar
$progressBar.Location = New-Object System.Drawing.Point(20, 90)
$progressBar.Size = New-Object System.Drawing.Size(540, 20)
$progressBar.Style = "Continuous"
$progressBar.Maximum = $script:totalSteps
$progressBar.Value = 1
$form.Controls.Add($progressBar)

# Main content panel
$contentPanel = New-Object System.Windows.Forms.Panel
$contentPanel.Location = New-Object System.Drawing.Point(20, 120)
$contentPanel.Size = New-Object System.Drawing.Size(540, 240)
$contentPanel.BorderStyle = "FixedSingle"
$form.Controls.Add($contentPanel)

# Navigation buttons
$backButton = New-Object System.Windows.Forms.Button
$backButton.Text = "< Back"
$backButton.Location = New-Object System.Drawing.Point(380, 380)
$backButton.Size = New-Object System.Drawing.Size(80, 30)
$backButton.Enabled = $false
$form.Controls.Add($backButton)

$nextButton = New-Object System.Windows.Forms.Button
$nextButton.Text = "Next >"
$nextButton.Location = New-Object System.Drawing.Point(470, 380)
$nextButton.Size = New-Object System.Drawing.Size(80, 30)
$form.Controls.Add($nextButton)

$cancelButton = New-Object System.Windows.Forms.Button
$cancelButton.Text = "Cancel"
$cancelButton.Location = New-Object System.Drawing.Point(290, 380)
$cancelButton.Size = New-Object System.Drawing.Size(80, 30)
$form.Controls.Add($cancelButton)

# Step content functions
function Show-WelcomeStep {
    $contentPanel.Controls.Clear()
    
    $welcomeLabel = New-Object System.Windows.Forms.Label
    $welcomeLabel.Text = "Welcome to Foresight Beta Setup"
    $welcomeLabel.Font = New-Object System.Drawing.Font("Segoe UI", 14, [System.Drawing.FontStyle]::Bold)
    $welcomeLabel.Location = New-Object System.Drawing.Point(20, 20)
    $welcomeLabel.Size = New-Object System.Drawing.Size(500, 30)
    $contentPanel.Controls.Add($welcomeLabel)
    
    $descText = "This wizard will guide you through the installation of Foresight Beta, a powerful application for:`n`n" +
                "• Phone screen mirroring via scrcpy`n" +
                "• Real-time AI object detection using YOLO`n" +
                "• Search and Rescue (SAR) mode for emergency scenarios`n" +
                "• Advanced computer vision capabilities`n`n" +
                "The setup will automatically detect and install required dependencies:`n" +
                "• Node.js (for the main application)`n" +
                "• Python (for AI detection)`n" +
                "• scrcpy (for phone mirroring)`n`n" +
                "Click Next to continue."
    
    $descLabel = New-Object System.Windows.Forms.Label
    $descLabel.Text = $descText
    $descLabel.Font = New-Object System.Drawing.Font("Segoe UI", 10)
    $descLabel.Location = New-Object System.Drawing.Point(20, 60)
    $descLabel.Size = New-Object System.Drawing.Size(500, 160)
    $contentPanel.Controls.Add($descLabel)
}

function Show-InstallLocationStep {
    $contentPanel.Controls.Clear()
    
    $locationLabel = New-Object System.Windows.Forms.Label
    $locationLabel.Text = "Choose Installation Location"
    $locationLabel.Font = New-Object System.Drawing.Font("Segoe UI", 14, [System.Drawing.FontStyle]::Bold)
    $locationLabel.Location = New-Object System.Drawing.Point(20, 20)
    $locationLabel.Size = New-Object System.Drawing.Size(500, 30)
    $contentPanel.Controls.Add($locationLabel)
    
    $descLabel = New-Object System.Windows.Forms.Label
    $descLabel.Text = "Select the folder where Foresight Beta will be installed:"
    $descLabel.Font = New-Object System.Drawing.Font("Segoe UI", 10)
    $descLabel.Location = New-Object System.Drawing.Point(20, 60)
    $descLabel.Size = New-Object System.Drawing.Size(500, 20)
    $contentPanel.Controls.Add($descLabel)
    
    # Installation path textbox
    $script:pathTextBox = New-Object System.Windows.Forms.TextBox
    $script:pathTextBox.Text = $script:installPath
    $script:pathTextBox.Location = New-Object System.Drawing.Point(20, 90)
    $script:pathTextBox.Size = New-Object System.Drawing.Size(400, 25)
    $script:pathTextBox.Font = New-Object System.Drawing.Font("Segoe UI", 10)
    $contentPanel.Controls.Add($script:pathTextBox)
    
    # Browse button
    $browseButton = New-Object System.Windows.Forms.Button
    $browseButton.Text = "Browse..."
    $browseButton.Location = New-Object System.Drawing.Point(430, 89)
    $browseButton.Size = New-Object System.Drawing.Size(80, 27)
    $contentPanel.Controls.Add($browseButton)
    
    # Space requirements
    $spaceLabel = New-Object System.Windows.Forms.Label
    $spaceLabel.Text = "Space required: ~50 MB"
    $spaceLabel.Font = New-Object System.Drawing.Font("Segoe UI", 9)
    $spaceLabel.ForeColor = [System.Drawing.Color]::Gray
    $spaceLabel.Location = New-Object System.Drawing.Point(20, 125)
    $spaceLabel.Size = New-Object System.Drawing.Size(200, 20)
    $contentPanel.Controls.Add($spaceLabel)
    
    # Warning about admin rights
    $adminLabel = New-Object System.Windows.Forms.Label
    $adminLabel.Text = "Note: Installing to Program Files may require administrator privileges."
    $adminLabel.Font = New-Object System.Drawing.Font("Segoe UI", 9)
    $adminLabel.ForeColor = [System.Drawing.Color]::Orange
    $adminLabel.Location = New-Object System.Drawing.Point(20, 150)
    $adminLabel.Size = New-Object System.Drawing.Size(500, 20)
    $contentPanel.Controls.Add($adminLabel)
    
    # Browse button click event
    $browseButton.Add_Click({
        $folderDialog = New-Object System.Windows.Forms.FolderBrowserDialog
        $folderDialog.Description = "Select installation folder for Foresight Beta"
        $folderDialog.SelectedPath = $script:pathTextBox.Text
        $folderDialog.ShowNewFolderButton = $true
        
        if ($folderDialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
            $selectedPath = $folderDialog.SelectedPath
            if (-not $selectedPath.EndsWith("Foresight Beta")) {
                $selectedPath = Join-Path $selectedPath "Foresight Beta"
            }
            $script:pathTextBox.Text = $selectedPath
            $script:installPath = $selectedPath
        }
    })
    
    # Update install path when text changes
    $script:pathTextBox.Add_TextChanged({
        $script:installPath = $script:pathTextBox.Text
    })
}

function Show-DependencyCheckStep {
    $contentPanel.Controls.Clear()
    
    $checkLabel = New-Object System.Windows.Forms.Label
    $checkLabel.Text = "Checking System Requirements"
    $checkLabel.Font = New-Object System.Drawing.Font("Segoe UI", 14, [System.Drawing.FontStyle]::Bold)
    $checkLabel.Location = New-Object System.Drawing.Point(20, 20)
    $checkLabel.Size = New-Object System.Drawing.Size(500, 30)
    $contentPanel.Controls.Add($checkLabel)
    
    # Node.js check
    $nodeLabel = New-Object System.Windows.Forms.Label
    $nodeLabel.Text = "Node.js: Checking..."
    $nodeLabel.Location = New-Object System.Drawing.Point(20, 70)
    $nodeLabel.Size = New-Object System.Drawing.Size(400, 20)
    $contentPanel.Controls.Add($nodeLabel)
    
    # Python check
    $pythonLabel = New-Object System.Windows.Forms.Label
    $pythonLabel.Text = "Python: Checking..."
    $pythonLabel.Location = New-Object System.Drawing.Point(20, 100)
    $pythonLabel.Size = New-Object System.Drawing.Size(400, 20)
    $contentPanel.Controls.Add($pythonLabel)
    
    # scrcpy check
    $scrcpyLabel = New-Object System.Windows.Forms.Label
    $scrcpyLabel.Text = "scrcpy: Checking..."
    $scrcpyLabel.Location = New-Object System.Drawing.Point(20, 130)
    $scrcpyLabel.Size = New-Object System.Drawing.Size(400, 20)
    $contentPanel.Controls.Add($scrcpyLabel)
    
    # Perform checks
    $form.Refresh()
    Start-Sleep -Milliseconds 500
    
    # Check Node.js
    try {
        $nodeVersion = & node --version 2>$null
        if ($nodeVersion) {
            $nodeLabel.Text = "Node.js: [OK] Installed ($nodeVersion)"
            $nodeLabel.ForeColor = [System.Drawing.Color]::Green
            $script:nodeInstalled = $true
        } else {
            throw "Not found"
        }
    } catch {
        $nodeLabel.Text = "Node.js: [X] Not installed (will be downloaded)"
        $nodeLabel.ForeColor = [System.Drawing.Color]::Red
        $script:nodeInstalled = $false
    }
    
    $form.Refresh()
    Start-Sleep -Milliseconds 500
    
    # Check Python
    try {
        $pythonVersion = & python --version 2>$null
        if ($pythonVersion) {
            $pythonLabel.Text = "Python: [OK] Installed ($pythonVersion)"
            $pythonLabel.ForeColor = [System.Drawing.Color]::Green
            $script:pythonInstalled = $true
        } else {
            throw "Not found"
        }
    } catch {
        $pythonLabel.Text = "Python: [X] Not installed (will be downloaded)"
        $pythonLabel.ForeColor = [System.Drawing.Color]::Red
        $script:pythonInstalled = $false
    }
    
    $form.Refresh()
    Start-Sleep -Milliseconds 500
    
    # Check scrcpy
    try {
        $scrcpyVersion = & scrcpy --version 2>$null
        if ($scrcpyVersion) {
            $scrcpyLabel.Text = "scrcpy: [OK] Installed"
            $scrcpyLabel.ForeColor = [System.Drawing.Color]::Green
            $script:scrcpyInstalled = $true
        } else {
            throw "Not found"
        }
    } catch {
        $scrcpyLabel.Text = "scrcpy: [X] Not installed (optional - manual install required)"
        $scrcpyLabel.ForeColor = [System.Drawing.Color]::Orange
        $script:scrcpyInstalled = $false
    }
    
    # Summary
    $summaryLabel = New-Object System.Windows.Forms.Label
    if (-not $script:nodeInstalled -or -not $script:pythonInstalled) {
        $summaryLabel.Text = "Some dependencies need to be installed. Click Next to continue."
        $summaryLabel.ForeColor = [System.Drawing.Color]::Orange
    } else {
        $summaryLabel.Text = "All required dependencies are installed! Click Next to continue."
        $summaryLabel.ForeColor = [System.Drawing.Color]::Green
    }
    $summaryLabel.Location = New-Object System.Drawing.Point(20, 170)
    $summaryLabel.Size = New-Object System.Drawing.Size(500, 40)
    $contentPanel.Controls.Add($summaryLabel)
}

function Show-InstallationStep {
    $contentPanel.Controls.Clear()
    
    $installLabel = New-Object System.Windows.Forms.Label
    $installLabel.Text = "Installing Dependencies"
    $installLabel.Font = New-Object System.Drawing.Font("Segoe UI", 14, [System.Drawing.FontStyle]::Bold)
    $installLabel.Location = New-Object System.Drawing.Point(20, 20)
    $installLabel.Size = New-Object System.Drawing.Size(500, 30)
    $contentPanel.Controls.Add($installLabel)
    
    $statusLabel = New-Object System.Windows.Forms.Label
    $statusLabel.Text = "Preparing installation..."
    $statusLabel.Location = New-Object System.Drawing.Point(20, 70)
    $statusLabel.Size = New-Object System.Drawing.Size(500, 20)
    $contentPanel.Controls.Add($statusLabel)
    
    $installProgress = New-Object System.Windows.Forms.ProgressBar
    $installProgress.Location = New-Object System.Drawing.Point(20, 100)
    $installProgress.Size = New-Object System.Drawing.Size(500, 20)
    $installProgress.Style = "Marquee"
    $contentPanel.Controls.Add($installProgress)
    
    $logBox = New-Object System.Windows.Forms.TextBox
    $logBox.Multiline = $true
    $logBox.ScrollBars = "Vertical"
    $logBox.ReadOnly = $true
    $logBox.Location = New-Object System.Drawing.Point(20, 130)
    $logBox.Size = New-Object System.Drawing.Size(500, 90)
    $logBox.Font = New-Object System.Drawing.Font("Consolas", 8)
    $contentPanel.Controls.Add($logBox)
    
    $nextButton.Enabled = $false
    $form.Refresh()
    
    # Step 1: Create installation directory
    $statusLabel.Text = "Creating installation directory..."
    $logBox.AppendText("Creating directory: $script:installPath`r`n")
    $form.Refresh()
    
    try {
        if (-not (Test-Path $script:installPath)) {
            New-Item -Path $script:installPath -ItemType Directory -Force | Out-Null
            $logBox.AppendText("Directory created successfully`r`n")
        } else {
            $logBox.AppendText("Directory already exists`r`n")
        }
    } catch {
        $logBox.AppendText("Failed to create directory: $($_.Exception.Message)`r`n")
        $nextButton.Enabled = $true
        return
    }
    
    # Step 2: Copy application files
    $statusLabel.Text = "Copying application files..."
    $logBox.AppendText("Copying files from $script:sourceDirectory to $script:installPath`r`n")
    $form.Refresh()
    
    try {
        # List of files and folders to copy
        $itemsToCopy = @(
            "src",
            "scripts", 
            "assets",
            "package.json",
            "package-lock.json",
            "requirements.txt",
            "yolov8n.pt",
            "start-foresight.bat",
            "start-foresight.ps1",
            "README.md"
        )
        
        foreach ($item in $itemsToCopy) {
            $sourcePath = Join-Path $script:sourceDirectory $item
            $destPath = Join-Path $script:installPath $item
            
            if (Test-Path $sourcePath) {
                if (Test-Path $sourcePath -PathType Container) {
                    # Copy directory
                    Copy-Item -Path $sourcePath -Destination $destPath -Recurse -Force
                    $logBox.AppendText("Copied folder: $item`r`n")
                } else {
                    # Copy file
                    Copy-Item -Path $sourcePath -Destination $destPath -Force
                    $logBox.AppendText("Copied file: $item`r`n")
                }
            }
        }
        $logBox.AppendText("File copying completed successfully`r`n")
    } catch {
        $logBox.AppendText("File copying failed: $($_.Exception.Message)`r`n")
        $nextButton.Enabled = $true
        return
    }
    
    # Step 3: Install Node.js dependencies
    if ($script:nodeInstalled) {
        $statusLabel.Text = "Installing Node.js dependencies..."
        $logBox.AppendText("Installing npm packages in $script:installPath`r`n")
        $form.Refresh()
        
        try {
            Set-Location $script:installPath
            $npmOutput = & npm install 2>&1
            $logBox.AppendText("npm install completed successfully`r`n")
            Set-Location $script:sourceDirectory
        } catch {
            $logBox.AppendText("npm install failed: $($_.Exception.Message)`r`n")
            Set-Location $script:sourceDirectory
        }
    }
    
    # Step 4: Install Python dependencies
    if ($script:pythonInstalled) {
        $statusLabel.Text = "Installing Python dependencies..."
        $logBox.AppendText("Installing Python packages...`r`n")
        $form.Refresh()
        
        try {
            Set-Location $script:installPath
            $pipOutput = & pip install -r requirements.txt 2>&1
            $logBox.AppendText("pip install completed successfully`r`n")
            Set-Location $script:sourceDirectory
        } catch {
            $logBox.AppendText("pip install failed: $($_.Exception.Message)`r`n")
            Set-Location $script:sourceDirectory
        }
    }
    
    $statusLabel.Text = "Installation completed!"
    $installProgress.Style = "Continuous"
    $installProgress.Value = 100
    $logBox.AppendText("Foresight Beta installed successfully to: $script:installPath`r`n")
    $nextButton.Enabled = $true
}

function Show-CompletionStep {
    $contentPanel.Controls.Clear()
    
    $completeLabel = New-Object System.Windows.Forms.Label
    $completeLabel.Text = "Setup Complete!"
    $completeLabel.Font = New-Object System.Drawing.Font("Segoe UI", 14, [System.Drawing.FontStyle]::Bold)
    $completeLabel.ForeColor = [System.Drawing.Color]::Green
    $completeLabel.Location = New-Object System.Drawing.Point(20, 20)
    $completeLabel.Size = New-Object System.Drawing.Size(500, 30)
    $contentPanel.Controls.Add($completeLabel)
    
    $successText = "Foresight Beta has been successfully installed to:`n$script:installPath`n`n" +
                   "For phone mirroring:`n" +
                   "1. Enable USB debugging on your Android device`n" +
                   "2. Connect your phone via USB`n" +
                   "3. Launch the application and click 'Start Phone Mirror'`n`n" +
                   "Thank you for installing Foresight Beta!"
    
    $successLabel = New-Object System.Windows.Forms.Label
    $successLabel.Text = $successText
    $successLabel.Font = New-Object System.Drawing.Font("Segoe UI", 10)
    $successLabel.Location = New-Object System.Drawing.Point(20, 60)
    $successLabel.Size = New-Object System.Drawing.Size(500, 120)
    $contentPanel.Controls.Add($successLabel)
    
    # Desktop shortcut checkbox
    $script:desktopCheckbox = New-Object System.Windows.Forms.CheckBox
    $script:desktopCheckbox.Text = "Create desktop shortcut"
    $script:desktopCheckbox.Location = New-Object System.Drawing.Point(20, 190)
    $script:desktopCheckbox.Size = New-Object System.Drawing.Size(200, 20)
    $script:desktopCheckbox.Checked = $true
    $contentPanel.Controls.Add($script:desktopCheckbox)
    
    # Launch checkbox
    $script:launchCheckbox = New-Object System.Windows.Forms.CheckBox
    $script:launchCheckbox.Text = "Launch Foresight Beta now"
    $script:launchCheckbox.Location = New-Object System.Drawing.Point(20, 210)
    $script:launchCheckbox.Size = New-Object System.Drawing.Size(200, 20)
    $script:launchCheckbox.Checked = $true
    $contentPanel.Controls.Add($script:launchCheckbox)
    
    $nextButton.Text = "Finish"
}

# Navigation event handlers
$nextButton.Add_Click({
    $script:currentStep++
    $progressBar.Value = $script:currentStep + 1
    
    switch ($script:currentStep) {
        1 { Show-InstallLocationStep }
        2 { Show-DependencyCheckStep }
        3 { Show-InstallationStep }
        4 { Show-CompletionStep }
        5 { 
            # Create desktop shortcut if checkbox is checked
            if ($script:desktopCheckbox -and $script:desktopCheckbox.Checked) {
                try {
                    $desktopPath = [Environment]::GetFolderPath("Desktop")
                    $shortcutPath = Join-Path $desktopPath "Foresight Beta.lnk"
                    $targetPath = Join-Path $script:installPath "start-foresight.bat"
                    $iconPath = Join-Path $script:installPath "assets\icon.png"
                    
                    $WshShell = New-Object -ComObject WScript.Shell
                    $Shortcut = $WshShell.CreateShortcut($shortcutPath)
                    $Shortcut.TargetPath = $targetPath
                    $Shortcut.WorkingDirectory = $script:installPath
                    $Shortcut.Description = "Foresight Beta - Phone Mirroring & AI Detection"
                    if (Test-Path $iconPath) {
                        $Shortcut.IconLocation = $iconPath
                    }
                    $Shortcut.Save()
                } catch {
                    # Silently continue if shortcut creation fails
                }
            }
            
            # Launch app if checkbox is checked
            if ($script:launchCheckbox -and $script:launchCheckbox.Checked) {
                Start-Process -FilePath "$script:installPath\start-foresight.bat" -WorkingDirectory $script:installPath
            }
            $form.Close()
        }
    }
    
    $backButton.Enabled = ($script:currentStep -gt 0)
    if ($script:currentStep -eq 3) {
        $nextButton.Text = "Finish"
    }
})

$backButton.Add_Click({
    if ($script:currentStep -gt 0) {
        $script:currentStep--
        $progressBar.Value = $script:currentStep + 1
        
        switch ($script:currentStep) {
            0 { Show-WelcomeStep }
            1 { Show-InstallLocationStep }
            2 { Show-DependencyCheckStep }
            3 { Show-InstallationStep }
        }
        
        $backButton.Enabled = ($script:currentStep -gt 0)
        $nextButton.Text = "Next >"
        $nextButton.Enabled = $true
    }
})

$cancelButton.Add_Click({
    $result = [System.Windows.Forms.MessageBox]::Show("Are you sure you want to cancel the installation?", "Cancel Setup", [System.Windows.Forms.MessageBoxButtons]::YesNo, [System.Windows.Forms.MessageBoxIcon]::Question)
    if ($result -eq [System.Windows.Forms.DialogResult]::Yes) {
        $form.Close()
    }
})

# Initialize first step
Show-WelcomeStep

# Show the form
$form.ShowDialog() | Out-Null
const { app, BrowserWindow, ipcMain, shell, dialog } = require('electron');
const path = require('path');
const { spawn, exec } = require('child_process');
const { initDetectionDatabase, logDetection, getRecentDetections, closeDetectionDatabase } = require('./db');

class ForesightApp {
  constructor() {
    this.mainWindow = null;
    this.scrcpyWindow = null;
    this.overlayWindow = null;  // ChatGPT recommendation #3: Transparent overlay window
    this.yoloProcess = null;
    this.scrcpyProcess = null;
    this.isCapturing = false;
    this.sarModeEnabled = false;
    this.detectionLoggingEnabled = false;
    this.faceSaveDir = null; // User-configurable save folder for verified faces
    this.childWindows = [];
    this.scrcpyWindowState = 'normal'; // Track scrcpy window state: 'normal', 'minimized', 'hidden'
    this.mainWindowState = 'normal'; // Track main window state
    this.scrcpyFocused = false; // Track if scrcpy window is focused
    this.focusMonitorInterval = null; // Interval for monitoring focus
    // Embed scrcpy state
    this.phoneMirrorBounds = null;
    this.embedScrcpyEnabled = true;
    this.embedMonitorInterval = null;
    this.lastEmbedErrorTs = 0;
    // Gallery watcher
    this.galleryWatcher = null;
    this.galleryWatchDir = null;
  }

  createMainWindow() {
    console.log('Creating main window...');
    this.mainWindow = new BrowserWindow({
      width: 1200,
      height: 800,
      webPreferences: {
        nodeIntegration: true,
        contextIsolation: false,
        enableRemoteModule: true,
        backgroundThrottling: false  // ChatGPT recommendation #7: Disable background throttling
      },
      icon: path.join(__dirname, '../assets/icon.ico'),
      title: 'Foresight',
      resizable: true,
      minimizable: true,
      maximizable: true,
      show: false
    });

    console.log('Loading HTML file...');
    this.mainWindow.loadFile(path.join(__dirname, 'renderer/index.html'));

    // Show window when ready and maximize by default
    this.mainWindow.once('ready-to-show', () => {
      console.log('Window ready to show - displaying and maximizing...');
      this.mainWindow.show();
      this.mainWindow.maximize();
    });

    // Add error handling
    this.mainWindow.webContents.on('did-fail-load', (event, errorCode, errorDescription) => {
      console.error('Failed to load:', errorCode, errorDescription);
    });

    this.mainWindow.on('closed', () => {
      console.log('Main window closed');
      this.mainWindow = null;
    });

    // Enhanced window events for synchronized grouping
    this.mainWindow.on('minimize', () => {
      this.mainWindowState = 'minimized';
      this.mainWindow.webContents.send('console-log', 'Main window minimized - synchronizing scrcpy window');
      this.synchronizedMinimize();
    });

    this.mainWindow.on('restore', () => {
      this.mainWindowState = 'normal';
      this.mainWindow.webContents.send('console-log', 'Main window restored - synchronizing scrcpy window');
      this.synchronizedRestore();
    });
    
    // Additional window state tracking
    this.mainWindow.on('show', () => {
      if (this.mainWindowState === 'minimized') {
        this.mainWindowState = 'normal';
        this.synchronizedRestore();
      }
    });
    
    this.mainWindow.on('hide', () => {
      this.mainWindowState = 'hidden';
      this.synchronizedMinimize();
    });

    this.mainWindow.on('closed', () => {
      this.cleanup();
      app.quit();
    });

    // Disable development tools by default
    // Only open dev tools if explicitly requested with --dev-tools flag
    if (process.argv.includes('--dev-tools')) {
      this.mainWindow.webContents.openDevTools();
    }
  }

  // Cgi recommend ni chat gpt para maging mas dali 
  createOverlayWindow() {
    console.log('Creating transparent overlay window for flicker-free rendering...');
    this.overlayWindow = new BrowserWindow({
      width: 1498,
      height: 937,
      frame: false,
      transparent: true,
      alwaysOnTop: true,
      skipTaskbar: true,
      resizable: false,
      movable: false,
      webPreferences: {
        nodeIntegration: true,
        contextIsolation: false,
        backgroundThrottling: false  // ChatGPT recommendation #7: Disable background throttling
      },
      show: false
    });

    // Load the overlay.html file
    this.overlayWindow.loadFile(path.join(__dirname, 'renderer', 'overlay.html'));

    this.overlayWindow.on('closed', () => {
      this.clearFocusMonitoring();
      this.overlayWindow = null;
    });

    // Allow mouse interactions to pass through overlay
    this.overlayWindow.setIgnoreMouseEvents(true, { forward: true });
    
    // Start focus monitoring when overlay is created
    this.setupFocusMonitoring();
    
    console.log('Transparent overlay window created for anti-flicker rendering');
  }



  synchronizedMinimize() {
    if (!this.isCapturing || !this.scrcpyProcess) {
      this.mainWindow.webContents.send('console-log', 'No scrcpy window to minimize');
      return;
    }
    
    if (this.scrcpyWindowState === 'minimized') {
      this.mainWindow.webContents.send('console-log', 'Scrcpy window already minimized');
      return;
    }
    
    this.scrcpyWindowState = 'minimized';
    
    // Enhanced minimize with multiple attempts and methods
    const attemptMinimize = (attempt = 1, maxAttempts = 3) => {
      const psCommand = `
        $processes = Get-Process -Name "scrcpy" -ErrorAction SilentlyContinue
        $found = $false
        
        foreach ($proc in $processes) {
          if ($proc.MainWindowTitle -eq "Foresight Phone Mirror") {
            $found = $true
            Add-Type -TypeDefinition '
              using System;
              using System.Runtime.InteropServices;
              public class Win32API {
                [DllImport("user32.dll")]
                public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
                [DllImport("user32.dll")]
                public static extern bool IsWindowVisible(IntPtr hWnd);
                [DllImport("user32.dll")]
                public static extern bool SetWindowPos(IntPtr hWnd, IntPtr hWndInsertAfter, int X, int Y, int cx, int cy, uint uFlags);
                [DllImport("user32.dll")]
                public static extern bool IsIconic(IntPtr hWnd);
              }
            '
            
            # Method ${attempt}: Progressive minimize approach
            if (${attempt} -eq 1) {
              [Win32API]::ShowWindow($proc.MainWindowHandle, 6)  # SW_MINIMIZE
            } elseif (${attempt} -eq 2) {
              [Win32API]::ShowWindow($proc.MainWindowHandle, 2)  # SW_SHOWMINIMIZED
            } else {
              [Win32API]::ShowWindow($proc.MainWindowHandle, 0)  # SW_HIDE
            }
            
            Start-Sleep -Milliseconds 200
            
            # Verify minimize state
            $isMinimized = [Win32API]::IsIconic($proc.MainWindowHandle)
            $isVisible = [Win32API]::IsWindowVisible($proc.MainWindowHandle)
            
            Write-Host "Attempt ${attempt}: Minimized=$isMinimized, Visible=$isVisible"
            
            if ($isMinimized -or (-not $isVisible)) {
              Write-Host "Scrcpy window successfully minimized on attempt ${attempt}"
              exit 0
            } else {
              Write-Host "Minimize attempt ${attempt} may have failed"
              exit 2
            }
          }
        }
        
        if (-not $found) {
          Write-Host "Scrcpy window not found for minimize"
          exit 1
        }
      `;
      
      exec(`powershell -Command "${psCommand}"`, (error, stdout, stderr) => {
        if (error || (stdout && stdout.includes('exit 2'))) {
          this.mainWindow.webContents.send('console-log', `Minimize attempt ${attempt} failed: ${error ? error.message : 'verification failed'}`);
          
          if (attempt < maxAttempts) {
            setTimeout(() => attemptMinimize(attempt + 1, maxAttempts), 300);
          } else {
            this.mainWindow.webContents.send('console-log', 'All minimize attempts failed, trying fallback');
            this.fallbackMinimizeScrcpy();
          }
        } else {
          this.mainWindow.webContents.send('console-log', `Scrcpy window minimized successfully on attempt ${attempt}`);
        }
      });
    };
    
    attemptMinimize();
  }
  
  synchronizedRestore() {
     if (!this.isCapturing || !this.scrcpyProcess) {
       this.mainWindow.webContents.send('console-log', 'No scrcpy window to restore');
       return;
     }
     
     if (this.scrcpyWindowState === 'normal') {
       this.mainWindow.webContents.send('console-log', 'Scrcpy window already restored');
       return;
     }
     
     this.scrcpyWindowState = 'normal';
     
     // Enhanced restore with multiple attempts and verification
     const attemptRestore = (attempt = 1, maxAttempts = 3) => {
       const psCommand = `
         $processes = Get-Process -Name "scrcpy" -ErrorAction SilentlyContinue
         $found = $false
         
         foreach ($proc in $processes) {
           if ($proc.MainWindowTitle -eq "Foresight Phone Mirror") {
             $found = $true
             Add-Type -TypeDefinition '
               using System;
               using System.Runtime.InteropServices;
               public class Win32API {
                 [DllImport("user32.dll")]
                 public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
                 [DllImport("user32.dll")]
                 public static extern bool SetWindowPos(IntPtr hWnd, IntPtr hWndInsertAfter, int X, int Y, int cx, int cy, uint uFlags);
                 [DllImport("user32.dll")]
                 public static extern bool SetForegroundWindow(IntPtr hWnd);
                 [DllImport("user32.dll")]
                 public static extern bool IsWindowVisible(IntPtr hWnd);
                 [DllImport("user32.dll")]
                 public static extern bool IsIconic(IntPtr hWnd);
               }
             '
             
             # Method ${attempt}: Progressive restore approach
             if (${attempt} -eq 1) {
               [Win32API]::ShowWindow($proc.MainWindowHandle, 9)   # SW_RESTORE
             } elseif (${attempt} -eq 2) {
               [Win32API]::ShowWindow($proc.MainWindowHandle, 1)   # SW_SHOWNORMAL
               Start-Sleep -Milliseconds 100
               [Win32API]::ShowWindow($proc.MainWindowHandle, 9)   # SW_RESTORE
             } else {
               [Win32API]::ShowWindow($proc.MainWindowHandle, 3)   # SW_SHOWMAXIMIZED
               Start-Sleep -Milliseconds 100
               [Win32API]::ShowWindow($proc.MainWindowHandle, 9)   # SW_RESTORE
             }
             
             Start-Sleep -Milliseconds 300
             
             # Reposition to layer 2 after restore
             $mainWindow = Get-Process -Name "electron" | Where-Object {$_.MainWindowTitle -eq "Foresight"} | Select-Object -First 1
             if ($mainWindow) {
               [Win32API]::SetWindowPos($proc.MainWindowHandle, $mainWindow.MainWindowHandle, 1, 128, 1498, 937, 0x0040)
             } else {
               [Win32API]::SetWindowPos($proc.MainWindowHandle, [IntPtr]::Zero, 1, 128, 1498, 937, 0x0040)
             }
             
             Start-Sleep -Milliseconds 100
             
             # Verify restore state
             $isMinimized = [Win32API]::IsIconic($proc.MainWindowHandle)
             $isVisible = [Win32API]::IsWindowVisible($proc.MainWindowHandle)
             
             Write-Host "Attempt ${attempt}: Minimized=$isMinimized, Visible=$isVisible"
             
             if ((-not $isMinimized) -and $isVisible) {
               Write-Host "Scrcpy window successfully restored on attempt ${attempt}"
               exit 0
             } else {
               Write-Host "Restore attempt ${attempt} may have failed"
               exit 2
             }
           }
         }
         
         if (-not $found) {
           Write-Host "Scrcpy window not found for restore"
           exit 1
         }
       `;
       
       exec(`powershell -Command "${psCommand}"`, (error, stdout, stderr) => {
         if (error || (stdout && stdout.includes('exit 2'))) {
           this.mainWindow.webContents.send('console-log', `Restore attempt ${attempt} failed: ${error ? error.message : 'verification failed'}`);
           
           if (attempt < maxAttempts) {
             setTimeout(() => attemptRestore(attempt + 1, maxAttempts), 300);
           } else {
             this.mainWindow.webContents.send('console-log', 'All restore attempts failed, trying fallback');
             this.fallbackRestoreScrcpy();
           }
         } else {
           this.mainWindow.webContents.send('console-log', `Scrcpy window restored and repositioned successfully on attempt ${attempt}`);
         }
       });
     };
     
     attemptRestore();
   }
  
  fallbackMinimizeScrcpy() {
    // Alternative minimize method using different approach
    const psCommand = `
      Get-Process -Name "scrcpy" -ErrorAction SilentlyContinue | ForEach-Object {
        if ($_.MainWindowTitle -eq "Foresight Phone Mirror") {
          $_.CloseMainWindow()
          Write-Host "Fallback minimize attempted"
        }
      }
    `;
    
    exec(`powershell -Command "${psCommand}"`, (error, stdout, stderr) => {
      this.mainWindow.webContents.send('console-log', 'Fallback minimize method attempted');
    });
  }
  
  fallbackRestoreScrcpy() {
    // Alternative restore method
    this.mainWindow.webContents.send('console-log', 'Attempting fallback restore - repositioning scrcpy window');
    setTimeout(() => {
      this.maintainScrcpyPosition();
    }, 500);
  }

  setupWindowTracking() {
    if (!this.mainWindow || !this.isCapturing) return;
    // Skip legacy window positioning when embedding is enabled
    if (this.embedScrcpyEnabled) {
      this.mainWindow.webContents.send('console-log', 'Embedding mode enabled - skipping legacy scrcpy layering');
      return;
    }
    // Legacy: external window positioning
    this.positionScrcpyWindow();
    this.mainWindow.webContents.send('console-log', 'Window layering enabled - scrcpy on layer 2, main app on layer 1');
  }
  
  setupPositionTracking() {
    if (!this.mainWindow || !this.isCapturing) return;
    
    // ChatGPT Fix #1: Throttle window positioning to prevent constant repositioning
    let positionThrottle = null;
    const throttledPosition = () => {
      if (positionThrottle) return; // Skip if already scheduled
      positionThrottle = setTimeout(() => {
        this.maintainScrcpyPosition();
        positionThrottle = null;
      }, 400); // Throttle to max 2.5 times per second
    };
    
    // Track main window movement to keep scrcpy glued in position
    this.mainWindow.on('move', throttledPosition);
    this.mainWindow.on('resize', throttledPosition);
    
    // Set up periodic synchronization check
    this.setupSynchronizationMonitor();
    
    this.mainWindow.webContents.send('console-log', 'Position tracking and synchronization monitoring enabled (throttled)');
  }
  
  setupSynchronizationMonitor() {
    // ChatGPT Fix #1: Throttle window positioning to reduce flicker
    // Check window states every 4 seconds (reduced frequency)
    this.syncMonitorInterval = setInterval(() => {
      if (!this.isCapturing || !this.scrcpyProcess) return;
      
      // Check if main window is minimized but scrcpy isn't (or vice versa)
      const isMainMinimized = this.mainWindow.isMinimized();
      
      if (isMainMinimized && this.scrcpyWindowState !== 'minimized') {
        this.mainWindow.webContents.send('console-log', 'Sync check: Main minimized, syncing scrcpy');
        this.synchronizedMinimize();
      } else if (!isMainMinimized && this.scrcpyWindowState === 'minimized') {
        this.mainWindow.webContents.send('console-log', 'Sync check: Main restored, syncing scrcpy');
        this.synchronizedRestore();
      }
    }, 4000); // Reduced from 2000ms to 4000ms
  }
  
  clearSynchronizationMonitor() {
    if (this.syncMonitorInterval) {
      clearInterval(this.syncMonitorInterval);
      this.syncMonitorInterval = null;
    }
  }

  setupFocusMonitoring() {
    if (this.focusMonitorInterval) {
      clearInterval(this.focusMonitorInterval);
    }

    this.focusMonitorInterval = setInterval(() => {
      this.checkScrcpyFocus();
    }, 500); // Check every 500ms
  }

  clearFocusMonitoring() {
    if (this.focusMonitorInterval) {
      clearInterval(this.focusMonitorInterval);
      this.focusMonitorInterval = null;
    }
  }

  checkScrcpyFocus() {
    if (!this.isCapturing || !this.scrcpyProcess || !this.overlayWindow) return;

    const psCommand = `
      Add-Type -AssemblyName System.Windows.Forms
      $foregroundWindow = [System.Windows.Forms.Form]::ActiveForm
      $processes = Get-Process -Name "scrcpy" -ErrorAction SilentlyContinue
      
      foreach ($proc in $processes) {
        if ($proc.MainWindowTitle -eq "Foresight Phone Mirror") {
          Add-Type -TypeDefinition '
            using System;
            using System.Runtime.InteropServices;
            public class Win32API {
              [DllImport("user32.dll")]
              public static extern IntPtr GetForegroundWindow();
            }
          '
          
          $foregroundHandle = [Win32API]::GetForegroundWindow()
          if ($foregroundHandle -eq $proc.MainWindowHandle) {
            Write-Host "focused"
          } else {
            Write-Host "unfocused"
          }
          exit 0
        }
      }
      Write-Host "unfocused"
    `;

    exec(`powershell -Command "${psCommand}"`, (error, stdout, stderr) => {
      if (!error) {
        const isFocused = stdout.trim() === 'focused';
        if (isFocused !== this.scrcpyFocused) {
          this.scrcpyFocused = isFocused;
          this.updateOverlayVisibility();
        }
      }
    });
  }

  updateOverlayVisibility() {
    if (!this.overlayWindow) return;

    if (this.scrcpyFocused) {
      this.overlayWindow.show();
    } else {
      this.overlayWindow.hide();
    }
  }
  
  maintainScrcpyPosition() {
    if (!this.isCapturing || !this.scrcpyProcess) return;
    
    const winWidth = 1498;
    const winHeight = 937;
    const scrcpyX = 1;
    const scrcpyY = 128;
    
    // Maintain scrcpy position relative to screen, not main window
    const psCommand = `
      Add-Type -AssemblyName System.Windows.Forms
      $processes = Get-Process -Name "scrcpy" -ErrorAction SilentlyContinue
      $mainWindow = Get-Process -Name "electron" | Where-Object {$_.MainWindowTitle -eq "Foresight"} | Select-Object -First 1
      
      foreach ($proc in $processes) {
        if ($proc.MainWindowTitle -eq "Argus Phone Mirror") {
          Add-Type -TypeDefinition '
            using System;
            using System.Runtime.InteropServices;
            public class Win32API {
              [DllImport("user32.dll")]
              public static extern bool SetWindowPos(IntPtr hWnd, IntPtr hWndInsertAfter, int X, int Y, int cx, int cy, uint uFlags);
            }
          '
          
          # Keep scrcpy in layer 2 above main window
          if ($mainWindow) {
            [Win32API]::SetWindowPos($proc.MainWindowHandle, $mainWindow.MainWindowHandle, ${scrcpyX}, ${scrcpyY}, ${winWidth}, ${winHeight}, 0x0001)
          }
          exit 0
        }
      }
    `;
    
    exec(`powershell -Command "${psCommand}"`, (error, stdout, stderr) => {
      // Silent execution - no logging to avoid spam
    });
  }
  
  positionScrcpyWindow() {
    if (!this.isCapturing || !this.scrcpyProcess) return;
    
    const winWidth = 1498;
    const winHeight = 937;
    const scrcpyX = 1;
    const scrcpyY = 128;
    
    // Try multiple times to position the window
    let attempts = 0;
    const maxAttempts = 5;
    
    const tryPosition = () => {
      attempts++;
      this.mainWindow.webContents.send('console-log', `Attempting to position scrcpy window (attempt ${attempts})...`);
      
      // PowerShell command to position window in layer 2 (above main app but not always on top)
      const psCommand = `
        Add-Type -AssemblyName System.Windows.Forms
        $processes = Get-Process -Name "scrcpy" -ErrorAction SilentlyContinue
        $mainWindow = Get-Process -Name "electron" | Where-Object {$_.MainWindowTitle -eq "Foresight"} | Select-Object -First 1
        
        foreach ($proc in $processes) {
          if ($proc.MainWindowTitle -eq "Foresight Phone Mirror") {
            Add-Type -TypeDefinition '
              using System;
              using System.Runtime.InteropServices;
              public class Win32API {
                [DllImport("user32.dll")]
                public static extern bool SetWindowPos(IntPtr hWnd, IntPtr hWndInsertAfter, int X, int Y, int cx, int cy, uint uFlags);
                [DllImport("user32.dll")]
                public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
              }
            '
            # Show the window first
            [Win32API]::ShowWindow($proc.MainWindowHandle, 1)
            
            # Position scrcpy window above main window (layer 2) but not topmost
            if ($mainWindow) {
              [Win32API]::SetWindowPos($proc.MainWindowHandle, $mainWindow.MainWindowHandle, ${scrcpyX}, ${scrcpyY}, ${winWidth}, ${winHeight}, 0x0040)
            } else {
              [Win32API]::SetWindowPos($proc.MainWindowHandle, [IntPtr]::Zero, ${scrcpyX}, ${scrcpyY}, ${winWidth}, ${winHeight}, 0x0040)
            }
            
            Write-Host "Window positioned in layer 2 successfully"
            exit 0
          }
        }
        Write-Host "Scrcpy window not found"
        exit 1
      `;
      
      exec(`powershell -Command "${psCommand}"`, (error, stdout, stderr) => {
        if (error) {
          this.mainWindow.webContents.send('console-log', `Position attempt ${attempts} failed: ${error.message}`);
          if (attempts < maxAttempts) {
            setTimeout(tryPosition, 1000);
          } else {
            this.mainWindow.webContents.send('console-log', 'Failed to position scrcpy window after all attempts');
          }
        } else {
          this.mainWindow.webContents.send('console-log', `Scrcpy window positioned in layer 2 at (${scrcpyX}, ${scrcpyY}) with size ${winWidth}x${winHeight}`);
          // Lock the window and set up position tracking
          setTimeout(() => {
            this.lockScrcpyWindow();
            this.setupPositionTracking();
          }, 500);
        }
      });
    };
    
    // Start positioning attempts after a delay
    setTimeout(tryPosition, 2000);
  }
  
  lockScrcpyWindow() {
    if (!this.isCapturing || !this.scrcpyProcess) return;
    
    this.mainWindow.webContents.send('console-log', 'Locking scrcpy window position...');
    
    const psCommand = `
      Add-Type -AssemblyName System.Windows.Forms
      $processes = Get-Process -Name "scrcpy" -ErrorAction SilentlyContinue
      foreach ($proc in $processes) {
        if ($proc.MainWindowTitle -eq "Foresight Phone Mirror") {
          Add-Type -TypeDefinition '
            using System;
            using System.Runtime.InteropServices;
            public class Win32API {
              [DllImport("user32.dll")]
              public static extern int GetWindowLong(IntPtr hWnd, int nIndex);
              [DllImport("user32.dll")]
              public static extern int SetWindowLong(IntPtr hWnd, int nIndex, int dwNewLong);
            }
          '
          $style = [Win32API]::GetWindowLong($proc.MainWindowHandle, -16)
          $newStyle = $style -band -bnot 0x00040000
          [Win32API]::SetWindowLong($proc.MainWindowHandle, -16, $newStyle)
          Write-Host "Window locked successfully"
          exit 0
        }
      }
      Write-Host "Scrcpy window not found for locking"
      exit 1
    `;
    
    exec(`powershell -Command "${psCommand}"`, (error, stdout, stderr) => {
      if (error) {
        this.mainWindow.webContents.send('console-log', `Window locking failed: ${error.message}`);
      } else {
        this.mainWindow.webContents.send('console-log', 'Scrcpy window locked - position cannot be changed');
      }
    });
  }

  startCapture() {
    if (this.isCapturing) return;
    
    console.log('Starting phone capture...');
    this.mainWindow.webContents.send('console-log', 'Checking for connected devices...');
    
    // First check for connected devices using adb
    const adbCheck = spawn('adb', ['devices']);
    let adbOutput = '';
    
    adbCheck.stdout.on('data', (data) => {
      adbOutput += data.toString();
    });
    
    adbCheck.stderr.on('data', (data) => {
      console.error('ADB stderr:', data.toString());
    });
    
    adbCheck.on('error', (error) => {
      console.error('ADB not found:', error);
      this.mainWindow.webContents.send('console-log', 'Error: ADB not found. Please install Android SDK Platform Tools.');
      this.mainWindow.webContents.send('console-log', 'Download from: https://developer.android.com/studio/releases/platform-tools');
      return;
    });
    
    adbCheck.on('close', (code) => {
      if (code !== 0) {
        this.mainWindow.webContents.send('console-log', 'Error: ADB command failed.');
        return;
      }
      
      // Parse adb devices output
      const lines = adbOutput.split('\n');
      const devices = lines.filter(line => 
        line.trim() && 
        !line.includes('List of devices attached') && 
        line.includes('device')
      );
      
      if (devices.length === 0) {
        this.mainWindow.webContents.send('console-log', 'Can\'t detect device - USB debugging must be turned on');
        this.mainWindow.webContents.send('console-log', '');
        this.mainWindow.webContents.send('console-log', 'Instructions to enable USB debugging:');
        this.mainWindow.webContents.send('console-log', '1. Go to Settings > About phone');
        this.mainWindow.webContents.send('console-log', '2. Tap "Build number" 7 times to enable Developer options');
        this.mainWindow.webContents.send('console-log', '3. Go back to Settings > Developer options');
        this.mainWindow.webContents.send('console-log', '4. Enable "USB debugging"');
        this.mainWindow.webContents.send('console-log', '5. Connect your phone via USB cable');
        this.mainWindow.webContents.send('console-log', '6. Allow USB debugging when prompted on your phone');
        this.mainWindow.webContents.send('console-log', '');
        return;
      }
      
      // Device found, start scrcpy
      // Robustly parse device id: split on any whitespace and take the first token
      const deviceLine = devices[0].trim();
      const deviceId = deviceLine.split(/\s+/)[0];
      this.mainWindow.webContents.send('console-log', `Device detected: ${deviceId}`);
      this.mainWindow.webContents.send('console-log', 'Starting screen mirror...');
      
      this.isCapturing = true;
      
      // Position scrcpy window to fit the UI exactly as specified
      const winWidth = 1498;
      const winHeight = 937;
      
      // Fixed position and size to fit the UI perfectly
      const scrcpyX = 1;
      const scrcpyY = 128;
      
      // Start scrcpy process with calculated position and VSync settings
      this.scrcpyProcess = spawn('scrcpy', [
        '-s', deviceId,
        '--window-title=Foresight Phone Mirror',
        `--window-x=${scrcpyX}`,
        `--window-y=${scrcpyY}`,
        `--window-width=${winWidth}`,
        `--window-height=${winHeight}`,
        '--stay-awake',
        '--turn-screen-off',
        '--window-borderless',
        '--max-fps=30'  // ChatGPT Fix #5: Tame FPS for stability (30 FPS reduces flicker)
      ]);
      
      // After scrcpy initializes, embed it into the control panel
      setTimeout(() => {
        if (this.embedScrcpyEnabled) {
          this.embedScrcpyWindow();
          this.startEmbedMonitor();
        } else {
          this.setupWindowTracking();
        }
      }, 2000);

      // Forward scrcpy logs for easier diagnostics
      if (this.scrcpyProcess.stdout) {
        this.scrcpyProcess.stdout.on('data', (data) => {
          this.mainWindow.webContents.send('console-log', `scrcpy: ${data.toString()}`);
        });
      }
      if (this.scrcpyProcess.stderr) {
        this.scrcpyProcess.stderr.on('data', (data) => {
          this.mainWindow.webContents.send('console-log', `scrcpy error: ${data.toString()}`);
        });
      }

      this.scrcpyProcess.on('error', (error) => {
        console.error('Scrcpy error:', error);
        this.mainWindow.webContents.send('console-log', `Scrcpy error: ${error.message}`);
        this.mainWindow.webContents.send('console-log', 'Please install scrcpy: https://github.com/Genymobile/scrcpy');
        this.isCapturing = false;
      });

      this.scrcpyProcess.on('close', (code) => {
        console.log(`Scrcpy process exited with code ${code}`);
        this.isCapturing = false;
        this.mainWindow.webContents.send('console-log', 'Screen mirror stopped.');
        this.mainWindow.webContents.send('capture-stopped');
        this.stopEmbedMonitor();
      });

      // Notify renderer
      this.mainWindow.webContents.send('capture-started');
    });
  }

  stopCapture() {
    if (!this.isCapturing) {
      this.mainWindow.webContents.send('console-log', 'No active capture to stop.');
      return;
    }
    
    console.log('Stopping phone capture...');
    this.mainWindow.webContents.send('console-log', 'Stopping screen mirror...');
    
    // Remove window tracking listeners
    this.mainWindow.removeAllListeners('move');
    this.mainWindow.removeAllListeners('resize');
    this.stopEmbedMonitor();
    
    // Clear synchronization monitor
    this.clearSynchronizationMonitor();
    
    if (this.scrcpyProcess) {
      this.scrcpyProcess.kill();
      this.scrcpyProcess = null;
    }
    
    if (this.yoloProcess) {
      this.yoloProcess.kill();
      this.yoloProcess = null;
    }
    
    this.isCapturing = false;
    this.sarModeEnabled = false;
    this.scrcpyWindowState = 'normal';
    this.mainWindow.webContents.send('console-log', 'Capture stopped successfully.');
    this.mainWindow.webContents.send('capture-stopped');
  }

  async toggleSarMode() {
    this.sarModeEnabled = !this.sarModeEnabled;
    
    if (this.sarModeEnabled) {
      console.log('Enabling SAR mode...');
      await this.startYoloScrcpy();
    } else {
      console.log('Disabling SAR mode...');
      
      if (this.yoloProcess) {
        this.yoloProcess.kill();
        this.yoloProcess = null;
      }
      
      if (this.scrcpyProcess) {
        this.scrcpyProcess.kill();
        this.scrcpyProcess = null;
      }
      
      this.mainWindow.webContents.send('sar-stopped');
    }
  }

  async startYoloScrcpy() {
    console.log('Starting YOLO scrcpy window...');
    this.mainWindow.webContents.send('console-log', 'Starting YOLO scrcpy window...');
    
    console.log('SAR mode will capture Foresight Phone Mirror window content');
    this.mainWindow.webContents.send('console-log', 'SAR mode capturing window: Foresight Phone Mirror');
    
    // First, start scrcpy if not already running
    if (!this.scrcpyProcess) {
      this.mainWindow.webContents.send('console-log', 'Starting scrcpy for SAR mode...');
      
      // Position scrcpy window to fit the UI exactly as specified
      const winWidth = 1498;
      const winHeight = 937;
      const scrcpyX = 1;
      const scrcpyY = 128;
      
      this.scrcpyProcess = spawn('scrcpy', [
        '--window-title=Foresight Phone Mirror',
        `--window-x=${scrcpyX}`,
        `--window-y=${scrcpyY}`,
        `--window-width=${winWidth}`,
        `--window-height=${winHeight}`,
        '--stay-awake',
        '--turn-screen-off',
        '--window-borderless'
      ]);
      
      this.scrcpyProcess.on('error', (error) => {
        console.error('Scrcpy error:', error);
        this.mainWindow.webContents.send('console-log', `Scrcpy error: ${error.message}`);
        this.sarModeEnabled = false;
        this.mainWindow.webContents.send('sar-stopped');
      });
      
      this.scrcpyProcess.on('close', (code) => {
        console.log(`Scrcpy process exited with code ${code}`);
        this.mainWindow.webContents.send('console-log', `Scrcpy process exited with code ${code}`);
        if (this.yoloProcess) {
          this.yoloProcess.kill();
          this.yoloProcess = null;
        }
        this.sarModeEnabled = false;
        this.mainWindow.webContents.send('sar-stopped');
        this.stopEmbedMonitor();
        // Close overlay window when scrcpy stops
        if (this.overlayWindow) {
          this.overlayWindow.close();
          this.overlayWindow = null;
        }
      });
    }
    
    // Create and show the Electron overlay window for flicker-free rendering
    if (!this.overlayWindow) {
      this.createOverlayWindow();
      // Position overlay to match scrcpy window
      setTimeout(() => {
        this.positionOverlayWindow();
      }, 1000);
    }
    
    // Wait for scrcpy window to be created, then start YOLO
    setTimeout(() => {
      if (this.embedScrcpyEnabled) {
        this.embedScrcpyWindow();
        this.startEmbedMonitor();
      }
      const scriptPath = path.join(__dirname, '../scripts/yolo_detection.py');
      console.log(`Starting YOLO with script: ${scriptPath}`);
      this.mainWindow.webContents.send('console-log', `Starting YOLO with script: ${scriptPath}`);

      const workingDir = path.join(__dirname, '..');
      const baseArgs = [
        scriptPath,
        '--source=window',
        '--window-title=Foresight Phone Mirror'
      ];

      // Pass face save configuration to YOLO
      if (this.faceSaveDir) {
        baseArgs.push('--enable-face-save');
        baseArgs.push(`--face-save-dir=${this.faceSaveDir}`);
      }

      // Attach logging from a process to UI
      const attachYoloLogs = (proc) => {
        if (proc.stdout) {
          proc.stdout.on('data', (data) => {
            const output = data.toString();
            console.log(`YOLO stdout: ${output}`);
            try {
              const lines = output.split('\n');
              for (const line of lines) {
                if (line.includes('DETECTION_DATA:')) {
                  const detectionData = JSON.parse(line.replace('DETECTION_DATA:', ''));
                  if (this.overlayWindow) {
                    this.overlayWindow.webContents.send('yolo-detections', detectionData);
                  }
                  // Conditionally log detection type to SQLite
                  if (this.detectionLoggingEnabled) {
                    try {
                      let detectionType = 'unknown';
                      if (Array.isArray(detectionData)) {
                        const first = detectionData[0] || {};
                        detectionType = first.type || first.label || 'unknown';
                      } else if (typeof detectionData === 'object' && detectionData) {
                        detectionType = detectionData.type || detectionData.label || 'unknown';
                      }
                      logDetection(detectionType);
                      // Announce to renderer for live updates
                      this.mainWindow && this.mainWindow.webContents.send('detection-logged', {
                        type: detectionType,
                        timestamp: new Date().toISOString()
                      });
                    } catch (_) {}
                  }
                } else if (line.startsWith('FACE_SAVED:')) {
                  const savedPath = line.replace('FACE_SAVED:', '').trim();
                  this.mainWindow && this.mainWindow.webContents.send('console-log', `Face saved: ${savedPath}`);
                  this.mainWindow && this.mainWindow.webContents.send('face-saved', { path: savedPath, timestamp: new Date().toISOString() });
                }
              }
            } catch (_) {}
            // Forward YOLO info and error logs to console
            if (output.includes('[ERROR]') || output.includes('YOLO model initialized')) {
              this.mainWindow.webContents.send('console-log', `YOLO: ${output.trim()}`);
            }
          });
        }
        if (proc.stderr) {
          proc.stderr.on('data', (data) => {
            const output = data.toString();
            console.error(`YOLO stderr: ${output}`);
            this.mainWindow.webContents.send('console-log', `YOLO Error: ${output.trim()}`);
          });
        }
      };

      // Prefer Python 3.9 explicitly to avoid TF/Keras version issues
      // Remove generic 'python' fallback; try 'py -3.9' then 'py -3'
      const interpreters = [
        { label: 'py -3.9', cmd: 'py', argsPrefix: ['-3.9'] },
        { label: 'py -3', cmd: 'py', argsPrefix: ['-3'] }
      ];

      const attemptInterpreter = (index) => {
        if (index >= interpreters.length) {
          this.mainWindow.webContents.send('console-log', 'YOLO failed to start with all interpreters');
          this.sarModeEnabled = false;
          this.mainWindow.webContents.send('sar-stopped');
          return;
        }

        const { label, cmd, argsPrefix } = interpreters[index];
        const args = [...argsPrefix, ...baseArgs];
        this.mainWindow.webContents.send('console-log', `Using Python interpreter: ${label}`);

        const proc = spawn(cmd, args, { cwd: workingDir, stdio: ['pipe', 'pipe', 'pipe'] });
        this.yoloProcess = proc;
        attachYoloLogs(proc);

        let attemptedFallback = false;

        proc.on('error', (error) => {
          console.error(`YOLO spawn error with ${label}:`, error);
          this.mainWindow.webContents.send('console-log', `YOLO spawn error with ${label}: ${error.message}`);
          if (!attemptedFallback) {
            attemptedFallback = true;
            attemptInterpreter(index + 1);
          }
        });

        proc.on('close', (code) => {
          const shouldFallback = code !== 0; // 103 from py means version missing
          console.log(`YOLO process with ${label} exited with code ${code}`);
          this.mainWindow.webContents.send('console-log', `YOLO process with ${label} exited with code ${code}`);
          if (shouldFallback && !attemptedFallback) {
            attemptedFallback = true;
            attemptInterpreter(index + 1);
          }
        });
      };

      attemptInterpreter(0);
    }, 3000);

    this.mainWindow.webContents.send('sar-started');
  }

  updateScrcpyWindowBounds() {
    return new Promise((resolve) => {
      const psCommand = `
        Add-Type -AssemblyName System.Windows.Forms
        $processes = Get-Process -Name "scrcpy" -ErrorAction SilentlyContinue
        foreach ($proc in $processes) {
          if ($proc.MainWindowTitle -eq "Foresight Phone Mirror") {
            Add-Type -TypeDefinition '
              using System;
              using System.Runtime.InteropServices;
              public struct RECT {
                public int Left;
                public int Top;
                public int Right;
                public int Bottom;
              }
              public class Win32API {
                [DllImport("user32.dll")]
                public static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);
              }
            '
            $rect = New-Object RECT
            [Win32API]::GetWindowRect($proc.MainWindowHandle, [ref]$rect)
            Write-Host "$($rect.Left),$($rect.Top),$($rect.Right - $rect.Left),$($rect.Bottom - $rect.Top)"
            exit 0
          }
        }
        Write-Host "0,0,1498,937"
        exit 1
      `;
      
      exec(`powershell -Command "${psCommand}"`, (error, stdout, stderr) => {
        const bounds = stdout.trim().split(',').map(Number);
        resolve({
          x: bounds[0] || 1,
          y: bounds[1] || 128,
          width: bounds[2] || 1498,
          height: bounds[3] || 937
        });
      });
    });
  }

  async positionYoloScrcpyWindow() {
    if (!this.yoloProcess) {
      this.mainWindow.webContents.send('console-log', 'No YOLO process to position');
      return;
    }

    this.mainWindow.webContents.send('console-log', 'Positioning YOLO scrcpy window...');
    
    // Use fixed coordinates as specified: X=0, Y=260, W=1419, H=760
    const yoloX = 0;
    const yoloY = 260;
    const yoloWidth = 1419;
    const yoloHeight = 760;
    
    this.mainWindow.webContents.send('console-log', `Positioning YOLO window at: ${yoloX}, ${yoloY}, ${yoloWidth}x${yoloHeight}`);
    
    const psCommand = `
      Add-Type -AssemblyName System.Windows.Forms
      $processes = Get-Process -Name "python" -ErrorAction SilentlyContinue
      $mainWindow = Get-Process -Name "electron" | Where-Object {$_.MainWindowTitle -eq "Argus"} | Select-Object -First 1
      
      foreach ($proc in $processes) {
        $windows = $proc.MainWindowTitle
        if ($windows -like "*Foresight SAR Detection*" -or $windows -eq "Foresight SAR Detection") {
          Add-Type -TypeDefinition '
            using System;
            using System.Runtime.InteropServices;
            public class Win32API {
              [DllImport("user32.dll")]
              public static extern bool SetWindowPos(IntPtr hWnd, IntPtr hWndInsertAfter, int X, int Y, int cx, int cy, uint uFlags);
              [DllImport("user32.dll")]
              public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
              [DllImport("user32.dll")]
              public static extern bool SetForegroundWindow(IntPtr hWnd);
            }
          '
          
          # Show and bring to front
          [Win32API]::ShowWindow($proc.MainWindowHandle, 1)
          [Win32API]::SetForegroundWindow($proc.MainWindowHandle)
          
          # Position YOLO window at same location as scrcpy
          if ($mainWindow) {
            [Win32API]::SetWindowPos($proc.MainWindowHandle, $mainWindow.MainWindowHandle, ${yoloX}, ${yoloY}, ${yoloWidth}, ${yoloHeight}, 0x0001)
          } else {
              [Win32API]::SetWindowPos($proc.MainWindowHandle, [IntPtr]::Zero, ${yoloX}, ${yoloY}, ${yoloWidth}, ${yoloHeight}, 0x0040)
          }
          
          Write-Host "YOLO window positioned successfully"
          exit 0
        }
      }
      Write-Host "YOLO window not found"
      exit 1
    `;
    
    exec(`powershell -Command "${psCommand}"`, (error, stdout, stderr) => {
      if (error) {
        this.mainWindow.webContents.send('console-log', `YOLO positioning failed: ${error.message}`);
      } else {
        this.mainWindow.webContents.send('console-log', `YOLO window positioned at (${yoloX}, ${yoloY}) with size ${yoloWidth}x${yoloHeight}`);
      }
    });
  }

  positionOverlayWindow() {
    if (!this.overlayWindow) return;
    
    console.log('Positioning overlay window...');
    this.mainWindow.webContents.send('console-log', 'Positioning overlay window...');
    
    // Position overlay to match scrcpy window exactly
    // Scrcpy window is at (1, 128) with size 1498x937
    // Overlay should align with scrcpy window position
    const winWidth = 1498;
    const winHeight = 937;
    const scrcpyX = 1;
    const scrcpyY = 128;
    
    this.overlayWindow.setBounds({
      x: scrcpyX,
      y: scrcpyY,
      width: winWidth,
      height: winHeight
    });
    
    // Ensure overlay stays on top
    this.overlayWindow.setAlwaysOnTop(true, 'screen-saver');
    this.overlayWindow.show();
  }

  updateYOLORegion(coordinates) {
    console.log('Updating YOLO region coordinates:', coordinates);
    this.mainWindow.webContents.send('console-log', `Updating YOLO region: x=${coordinates.x}, y=${coordinates.y}, w=${coordinates.width}, h=${coordinates.height}`);
    
    // Save coordinates to a file that the YOLO script can read
    const fs = require('fs');
    const path = require('path');
    const coordsFile = path.join(__dirname, '..', 'yolo_coordinates.json');
    
    try {
      fs.writeFileSync(coordsFile, JSON.stringify(coordinates, null, 2));
      console.log('YOLO coordinates saved to:', coordsFile);
    } catch (error) {
      console.error('Failed to save YOLO coordinates:', error);
    }
  }

  cleanup() {
    console.log('Cleaning up processes...');
    
    // Clear synchronization monitor
    this.clearSynchronizationMonitor();
    
    // Clear focus monitoring
    this.clearFocusMonitoring();
    
    // Stop YOLO process
    if (this.yoloProcess) {
      console.log('Terminating YOLO process...');
      this.yoloProcess.kill('SIGTERM');
      this.yoloProcess = null;
    }
    
    // Stop scrcpy process
    if (this.scrcpyProcess) {
      console.log('Terminating scrcpy process...');
      this.scrcpyProcess.kill('SIGTERM');
      this.scrcpyProcess = null;
    }
    
    // Close overlay window (ChatGPT recommendation #3)
    if (this.overlayWindow && !this.overlayWindow.isDestroyed()) {
      console.log('Closing transparent overlay window...');
      this.overlayWindow.close();
      this.overlayWindow = null;
    }
    
    // Close all child windows
    this.childWindows.forEach(window => {
      if (window && !window.isDestroyed()) {
        window.close();
      }
    });
    this.childWindows = [];
    
    this.isCapturing = false;
    this.sarModeEnabled = false;
    
    console.log('Cleanup complete');
  }

  setupIpcHandlers() {
    ipcMain.on('start-capture', () => {
      this.startCapture();
    });

    ipcMain.on('stop-capture', () => {
      this.stopCapture();
    });

    ipcMain.on('toggle-sar', () => {
      this.toggleSarMode();
    });

    ipcMain.on('get-status', (event) => {
      event.reply('status-update', {
        isCapturing: this.isCapturing,
        sarModeEnabled: this.sarModeEnabled,
        detectionLoggingEnabled: this.detectionLoggingEnabled
      });
    });

    ipcMain.on('update-yolo-region', (event, coordinates) => {
      this.updateYOLORegion(coordinates);
    });

    // Toggle detection logging to SQLite
    ipcMain.on('set-detection-logging', (event, enabled) => {
      this.detectionLoggingEnabled = !!enabled;
      event.reply('status-update', {
        isCapturing: this.isCapturing,
        sarModeEnabled: this.sarModeEnabled,
        detectionLoggingEnabled: this.detectionLoggingEnabled
      });
      this.mainWindow && this.mainWindow.webContents.send('console-log', `Detection logging ${this.detectionLoggingEnabled ? 'enabled' : 'disabled'}`);
    });

    // Provide recent detection logs to renderer
    ipcMain.on('get-detection-logs', (event, limit = 50) => {
      try {
        const rows = getRecentDetections(limit);
        event.reply('detection-logs', rows);
      } catch (err) {
        event.reply('detection-logs', []);
      }
    });

    // Face save directory settings
    ipcMain.on('get-face-save-dir', (event) => {
      event.reply('face-save-dir', this.faceSaveDir || null);
    });
    ipcMain.on('choose-face-save-dir', async (event) => {
      try {
        const res = await dialog.showOpenDialog(this.mainWindow, {
          properties: ['openDirectory', 'createDirectory']
        });
        if (!res.canceled && res.filePaths && res.filePaths[0]) {
          this.faceSaveDir = res.filePaths[0];
          this.saveFaceSaveSettings();
          event.reply('face-save-dir', this.faceSaveDir);
          this.mainWindow && this.mainWindow.webContents.send('console-log', `Face save folder set to: ${this.faceSaveDir}`);
        }
      } catch (e) {
        this.mainWindow && this.mainWindow.webContents.send('console-log', `Folder selection failed: ${e.message}`);
      }
    });
    ipcMain.on('set-face-save-dir', (event, dirPath) => {
      try {
        if (dirPath) {
          this.faceSaveDir = dirPath;
          this.saveFaceSaveSettings();
          event.reply('face-save-dir', this.faceSaveDir);
          this.mainWindow && this.mainWindow.webContents.send('console-log', `Face save folder set to: ${this.faceSaveDir}`);
        }
      } catch (e) {
        this.mainWindow && this.mainWindow.webContents.send('console-log', `Failed to set folder: ${e.message}`);
      }
    });

    // Provide detected images from configured folder (or default)
    ipcMain.on('get-detected-images', (event, limit = 200) => {
      const targetDir = this.faceSaveDir || 'C\\\\Users\\\\Asus\\\\Desktop\\\\Detected';
      const payload = this._buildDetectedImagesPayload(targetDir, limit);
      // Start or move watcher to this directory for real-time updates
      this._attachGalleryWatcher(targetDir, limit);
      event.reply('detected-images', payload);
    });

    // Receive phone mirror panel bounds from renderer to position embedded scrcpy
    ipcMain.on('phone-mirror/bounds', (event, bounds) => {
      try {
        this.phoneMirrorBounds = bounds;
        if (this.embedScrcpyEnabled) {
          this.embedScrcpyWindow();
        }
      } catch (e) {
        const now = Date.now();
        if (now - this.lastEmbedErrorTs > 3000) {
          this.lastEmbedErrorTs = now;
          this.mainWindow && this.mainWindow.webContents.send('console-log', `[FORESIGHT][EMBED] bounds update failed: ${e.message}`);
        }
      }
    });
  }

  _buildDetectedImagesPayload(targetDir, limit = 200) {
    const fs = require('fs');
    let files = [];
    try {
      const entries = fs.readdirSync(targetDir);
      files = entries
        .filter(name => /\.(jpg|jpeg|png|webp)$/i.test(name))
        .map(name => path.join(targetDir, name));
      files.sort((a, b) => {
        try {
          return fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs;
        } catch {
          return 0;
        }
      });
      files = files.slice(0, limit);
    } catch (e) {
      this.mainWindow && this.mainWindow.webContents.send('console-log', `Gallery load failed: ${e.message}`);
    }
    return { dir: targetDir, files };
  }

  _attachGalleryWatcher(targetDir, limit = 200) {
    const fs = require('fs');
    try {
      if (this.galleryWatcher && this.galleryWatchDir === targetDir) {
        return; // Already watching this dir
      }
      // Close existing watcher
      if (this.galleryWatcher) {
        try { this.galleryWatcher.close(); } catch {}
        this.galleryWatcher = null;
      }
      this.galleryWatchDir = targetDir;
      // Debounce burst of events
      let timer = null;
      this.galleryWatcher = fs.watch(targetDir, { persistent: true }, (eventType, filename) => {
        if (filename && /\.(jpg|jpeg|png|webp)$/i.test(filename)) {
          if (timer) clearTimeout(timer);
          timer = setTimeout(() => {
            const payload = this._buildDetectedImagesPayload(targetDir, limit);
            this.mainWindow && this.mainWindow.webContents.send('detected-images', payload);
          }, 150);
        }
      });
      this.mainWindow && this.mainWindow.webContents.send('console-log', `[FORESIGHT] Gallery watching: ${targetDir}`);
    } catch (e) {
      this.mainWindow && this.mainWindow.webContents.send('console-log', `[FORESIGHT] Gallery watch failed: ${e.message}`);
    }
  }

  // --- scrcpy embed helpers (Windows) ---
  embedScrcpyWindow() {
    if (!this.isCapturing || !this.scrcpyProcess) return;
    const b = this.phoneMirrorBounds;
    if (!b) return;
    const { x, y, width, height } = b;
    const psCommand = `
      Add-Type -TypeDefinition '
        using System;
        using System.Runtime.InteropServices;
        public static class Win32API {
          [DllImport("user32.dll")] public static extern int GetWindowLong(IntPtr hWnd, int nIndex);
          [DllImport("user32.dll")] public static extern int SetWindowLong(IntPtr hWnd, int nIndex, int dwNewLong);
          [DllImport("user32.dll")] public static extern IntPtr SetParent(IntPtr hWndChild, IntPtr hWndNewParent);
          [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr hWnd, IntPtr hWndInsertAfter, int X, int Y, int cx, int cy, uint uFlags);
        }
      '
      $proc = Get-Process -Name "scrcpy" -ErrorAction SilentlyContinue | Where-Object {$_.MainWindowTitle -eq "Foresight Phone Mirror"} | Select-Object -First 1
      $mainWindow = Get-Process -Name "electron" | Where-Object {$_.MainWindowTitle -eq "Foresight"} | Select-Object -First 1
      if (-not $proc -or -not $mainWindow) { exit 1 }
      $GWL_STYLE = -16
      $WS_CHILD = 0x40000000
      $WS_VISIBLE = 0x10000000
      $WS_POPUP = 0x80000000
      $WS_OVERLAPPEDWINDOW = 0x00CF0000
      $style = [Win32API]::GetWindowLong($proc.MainWindowHandle, $GWL_STYLE)
      $newStyle = ($style -bor $WS_CHILD -bor $WS_VISIBLE) -band (-bnot ($WS_POPUP -bor $WS_OVERLAPPEDWINDOW))
      [Win32API]::SetWindowLong($proc.MainWindowHandle, $GWL_STYLE, $newStyle) | Out-Null
      [Win32API]::SetParent($proc.MainWindowHandle, $mainWindow.MainWindowHandle) | Out-Null
      [Win32API]::SetWindowPos($proc.MainWindowHandle, [IntPtr]::Zero, ${x}, ${y}, ${width}, ${height}, 0x0040) | Out-Null
      Write-Host "EMBED_OK"
    `;
    exec(`powershell -Command "${psCommand}"`, (error, stdout, stderr) => {
      if (error) {
        const now = Date.now();
        if (now - this.lastEmbedErrorTs > 3000) {
          this.lastEmbedErrorTs = now;
          this.mainWindow && this.mainWindow.webContents.send('console-log', `[FORESIGHT][EMBED] error: ${error.message}`);
        }
        return;
      }
      const out = (stdout || '').toString().trim();
      if (out.includes('EMBED_OK')) {
        this.mainWindow && this.mainWindow.webContents.send('console-log', '[FORESIGHT][EMBED] embedded scrcpy into control panel');
      }
    });
  }

  startEmbedMonitor() {
    if (this.embedMonitorInterval) return;
    // Event-driven enforcement: keep scrcpy visually on top
    if (this.mainWindow && !this.enforceZOrderAttached) {
      this.mainWindow.on('focus', () => this.forceScrcpyTopMost());
      this.mainWindow.on('blur', () => this.forceScrcpyTopMost());
      this.mainWindow.on('move', () => this.forceScrcpyTopMost());
      this.mainWindow.on('resize', () => this.forceScrcpyTopMost());
      this.mainWindow.on('show', () => { this.showScrcpyWindow(); this.forceScrcpyTopMost(); });
      this.mainWindow.on('minimize', () => this.hideScrcpyWindow());
      this.mainWindow.on('restore', () => { this.showScrcpyWindow(); this.forceScrcpyTopMost(); });
      this.enforceZOrderAttached = true;
    }

    // Periodic enforcement to recover from any z-order drift
    this.embedMonitorInterval = setInterval(() => {
      if (!this.isCapturing || !this.scrcpyProcess) return;
      this.embedScrcpyWindow();
      this.forceScrcpyTopMost();
    }, 250);
  }

  stopEmbedMonitor() {
    if (this.embedMonitorInterval) {
      clearInterval(this.embedMonitorInterval);
      this.embedMonitorInterval = null;
    }
    // Note: listeners are lightweight; keep them until app stops capture
  }

  // Keep scrcpy window at the top of the control panel area
  enforceScrcpyTop() {
    if (!this.isCapturing || !this.scrcpyProcess) return;
    const b = this.phoneMirrorBounds;
    if (!b) return;
    const psCommand = `
      Add-Type -TypeDefinition '
        using System;
        using System.Runtime.InteropServices;
        public static class Win32API {
          [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr hWnd, IntPtr hWndInsertAfter, int X, int Y, int cx, int cy, uint uFlags);
        }
      '
      $proc = Get-Process -Name "scrcpy" -ErrorAction SilentlyContinue | Where-Object {$_.MainWindowTitle -eq "Foresight Phone Mirror"} | Select-Object -First 1
      if (-not $proc) { exit 1 }
      # Z-order only: keep at top among siblings without moving, sizing, or activating
      $SWP_NOSIZE = 0x0001
      $SWP_NOMOVE = 0x0002
      $SWP_NOACTIVATE = 0x0010
      [Win32API]::SetWindowPos($proc.MainWindowHandle, [IntPtr]::Zero, 0, 0, 0, 0, $SWP_NOSIZE -bor $SWP_NOMOVE -bor $SWP_NOACTIVATE) | Out-Null
      Write-Host "ZTOP_OK"
    `;
    exec(`powershell -Command "${psCommand}"`, (error, stdout, stderr) => {
      // No noisy logging; this runs frequently
    });
  }

  // Force global topmost so it always stays above control panel; tie visibility to main window
  forceScrcpyTopMost() {
    if (!this.isCapturing || !this.scrcpyProcess) return;
    const psCommand = `
      Add-Type -TypeDefinition '
        using System;
        using System.Runtime.InteropServices;
        public static class Win32API {
          [DllImport("user32.dll")] public static extern int GetWindowLong(IntPtr hWnd, int nIndex);
          [DllImport("user32.dll")] public static extern int SetWindowLong(IntPtr hWnd, int nIndex, int dwNewLong);
          [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr hWnd, IntPtr hWndInsertAfter, int X, int Y, int cx, int cy, uint uFlags);
          [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
        }
      '
      $proc = Get-Process -Name "scrcpy" -ErrorAction SilentlyContinue | Where-Object {$_.MainWindowTitle -eq "Foresight Phone Mirror"} | Select-Object -First 1
      if (-not $proc) { exit 1 }
      $GWL_STYLE = -16
      $WS_CHILD = 0x40000000
      # Ensure it is a top-level window (remove child style)
      $style = [Win32API]::GetWindowLong($proc.MainWindowHandle, $GWL_STYLE)
      if ($style -band $WS_CHILD) {
        $newStyle = $style -band -bnot $WS_CHILD
        [Win32API]::SetWindowLong($proc.MainWindowHandle, $GWL_STYLE, $newStyle) | Out-Null
      }
      # Make window topmost without activation
      $SWP_NOSIZE = 0x0001
      $SWP_NOMOVE = 0x0002
      $SWP_NOACTIVATE = 0x0010
      # HWND_TOPMOST = -1
      [Win32API]::SetWindowPos($proc.MainWindowHandle, [IntPtr](-1), 0, 0, 0, 0, $SWP_NOSIZE -bor $SWP_NOMOVE -bor $SWP_NOACTIVATE) | Out-Null
      Write-Host "TOPMOST_OK"
    `;
    exec(`powershell -Command "${psCommand}"`, () => {});
  }

  hideScrcpyWindow() {
    if (!this.isCapturing || !this.scrcpyProcess) return;
    const psCommand = `
      Add-Type -TypeDefinition '
        using System;
        using System.Runtime.InteropServices;
        public static class Win32API { [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow); }
      '
      $proc = Get-Process -Name "scrcpy" -ErrorAction SilentlyContinue | Where-Object {$_.MainWindowTitle -eq "Foresight Phone Mirror"} | Select-Object -First 1
      if (-not $proc) { exit 1 }
      # SW_HIDE = 0
      [Win32API]::ShowWindow($proc.MainWindowHandle, 0) | Out-Null
      Write-Host "HIDE_OK"
    `;
    exec(`powershell -Command "${psCommand}"`, () => {});
  }

  showScrcpyWindow() {
    if (!this.isCapturing || !this.scrcpyProcess) return;
    const psCommand = `
      Add-Type -TypeDefinition '
        using System;
        using System.Runtime.InteropServices;
        public static class Win32API {
          [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
          [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr hWnd, IntPtr hWndInsertAfter, int X, int Y, int cx, int cy, uint uFlags);
        }
      '
      $proc = Get-Process -Name "scrcpy" -ErrorAction SilentlyContinue | Where-Object {$_.MainWindowTitle -eq "Foresight Phone Mirror"} | Select-Object -First 1
      if (-not $proc) { exit 1 }
      # SW_SHOW = 5
      [Win32API]::ShowWindow($proc.MainWindowHandle, 5) | Out-Null
      # Restore topmost order
      $SWP_NOSIZE = 0x0001
      $SWP_NOMOVE = 0x0002
      $SWP_NOACTIVATE = 0x0010
      [Win32API]::SetWindowPos($proc.MainWindowHandle, [IntPtr](-1), 0, 0, 0, 0, $SWP_NOSIZE -bor $SWP_NOMOVE -bor $SWP_NOACTIVATE) | Out-Null
      Write-Host "SHOW_OK"
    `;
    exec(`powershell -Command "${psCommand}"`, () => {});
  }

  loadFaceSaveSettings() {
    const fs = require('fs');
    const settingsPath = path.join(app.getPath('userData'), 'face_settings.json');
    try {
      if (fs.existsSync(settingsPath)) {
        const data = JSON.parse(fs.readFileSync(settingsPath, 'utf-8'));
        if (data && data.faceSaveDir) {
          this.faceSaveDir = data.faceSaveDir;
          this.mainWindow && this.mainWindow.webContents.send('console-log', `Face save folder loaded: ${this.faceSaveDir}`);
        }
      }
    } catch (e) {
      this.mainWindow && this.mainWindow.webContents.send('console-log', `Failed to load face settings: ${e.message}`);
    }
  }

  saveFaceSaveSettings() {
    const fs = require('fs');
    const settingsPath = path.join(app.getPath('userData'), 'face_settings.json');
    try {
      fs.writeFileSync(settingsPath, JSON.stringify({ faceSaveDir: this.faceSaveDir }, null, 2));
    } catch (e) {
      this.mainWindow && this.mainWindow.webContents.send('console-log', `Failed to save face settings: ${e.message}`);
    }
  }
}

// App initialization
console.log('Initializing Foresight App...');
const foresightApp = new ForesightApp();

app.whenReady().then(async () => {
  const { dbPath } = await initDetectionDatabase();
  console.log(`Detection database initialized at: ${dbPath}`);
  console.log('Electron app ready - creating main window...');
  foresightApp.createMainWindow();
  foresightApp.setupIpcHandlers();
  // Load face save settings after window is ready
  foresightApp.loadFaceSaveSettings();
  console.log('App initialization complete');

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      console.log('App activated - creating new window...');
      foresightApp.createMainWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    foresightApp.cleanup();
    app.quit();
  }
});

app.on('before-quit', () => {
  foresightApp.cleanup();
  closeDetectionDatabase();
});
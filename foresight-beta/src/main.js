const { app, BrowserWindow, ipcMain, shell } = require('electron');
const path = require('path');
const { spawn, exec } = require('child_process');

class ArgusApp {
  constructor() {
    this.mainWindow = null;
    this.scrcpyWindow = null;
    this.overlayWindow = null;  // ChatGPT recommendation #3: Transparent overlay window
    this.yoloProcess = null;
    this.scrcpyProcess = null;
    this.isCapturing = false;
    this.sarModeEnabled = false;
    this.childWindows = [];
    this.scrcpyWindowState = 'normal'; // Track scrcpy window state: 'normal', 'minimized', 'hidden'
    this.mainWindowState = 'normal'; // Track main window state
    this.scrcpyFocused = false; // Track if scrcpy window is focused
    this.focusMonitorInterval = null; // Interval for monitoring focus
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
      icon: path.join(__dirname, '../assets/icon.png'),
      title: 'Argus',
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
          if ($proc.MainWindowTitle -eq "Argus Phone Mirror") {
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
           if ($proc.MainWindowTitle -eq "Argus Phone Mirror") {
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
             $mainWindow = Get-Process -Name "electron" | Where-Object {$_.MainWindowTitle -eq "Argus"} | Select-Object -First 1
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
        if ($_.MainWindowTitle -eq "Argus Phone Mirror") {
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
    
    // Set up window positioning with multiple attempts
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
        if ($proc.MainWindowTitle -eq "Argus Phone Mirror") {
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
      $mainWindow = Get-Process -Name "electron" | Where-Object {$_.MainWindowTitle -eq "Argus"} | Select-Object -First 1
      
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
        $mainWindow = Get-Process -Name "electron" | Where-Object {$_.MainWindowTitle -eq "Argus"} | Select-Object -First 1
        
        foreach ($proc in $processes) {
          if ($proc.MainWindowTitle -eq "Argus Phone Mirror") {
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
        if ($proc.MainWindowTitle -eq "Argus Phone Mirror") {
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
      this.mainWindow.webContents.send('console-log', `Device detected: ${devices[0].split('\t')[0]}`);
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
        '--window-title=Argus Phone Mirror',
        `--window-x=${scrcpyX}`,
        `--window-y=${scrcpyY}`,
        `--window-width=${winWidth}`,
        `--window-height=${winHeight}`,
        '--stay-awake',
        '--turn-screen-off',
        '--window-borderless',
        '--max-fps=30'  // ChatGPT Fix #5: Tame FPS for stability (30 FPS reduces flicker)
      ]);
      
      // Set up window tracking after a delay to ensure scrcpy window is created
      setTimeout(() => {
        this.setupWindowTracking();
      }, 2000);

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
    
    console.log('SAR mode will capture Argus Phone Mirror window content');
    this.mainWindow.webContents.send('console-log', 'SAR mode capturing window: Argus Phone Mirror');
    
    // First, start scrcpy if not already running
    if (!this.scrcpyProcess) {
      this.mainWindow.webContents.send('console-log', 'Starting scrcpy for SAR mode...');
      
      // Position scrcpy window to fit the UI exactly as specified
      const winWidth = 1498;
      const winHeight = 937;
      const scrcpyX = 1;
      const scrcpyY = 128;
      
      this.scrcpyProcess = spawn('scrcpy', [
        '--window-title=Argus Phone Mirror',
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
       const scriptPath = path.join(__dirname, '../scripts/yolo_detection.py');
       console.log(`Starting YOLO with script: ${scriptPath}`);
       this.mainWindow.webContents.send('console-log', `Starting YOLO with script: ${scriptPath}`);
       
       // Start YOLO without overlay flag - we'll use Electron overlay instead
       this.yoloProcess = spawn('python', [
         scriptPath,
         '--source=window',
         '--window-title=Argus Phone Mirror'
       ], {
         cwd: path.join(__dirname, '..'),
         stdio: ['pipe', 'pipe', 'pipe']
       });
       
       // Parse YOLO detection data and send to overlay
       this.yoloProcess.stdout.on('data', (data) => {
         const output = data.toString();
         console.log(`YOLO stdout: ${output}`);
         
         // Parse detection data from YOLO output
         try {
           const lines = output.split('\n');
           for (const line of lines) {
             if (line.includes('DETECTION_DATA:')) {
               const detectionData = JSON.parse(line.replace('DETECTION_DATA:', ''));
               // Send detection data to overlay window
               if (this.overlayWindow) {
                 this.overlayWindow.webContents.send('yolo-detections', detectionData);
               }
             }
           }
         } catch (e) {
           // Ignore parsing errors for non-JSON output
         }
         
         // Only send important messages to UI, not debug spam
         if (output.includes('[ERROR]') || output.includes('YOLO model initialized')) {
           this.mainWindow.webContents.send('console-log', `YOLO: ${output.trim()}`);
         }
       });
        
        // Log YOLO stderr for debugging (console only, not UI)
        this.yoloProcess.stderr.on('data', (data) => {
          const output = data.toString();
          console.error(`YOLO stderr: ${output}`);
          // Only send actual errors to UI
          this.mainWindow.webContents.send('console-log', `YOLO Error: ${output.trim()}`);
        });
       
       this.yoloProcess.on('error', (error) => {
         console.error('YOLO spawn error:', error);
         this.mainWindow.webContents.send('console-log', `YOLO spawn error: ${error.message}`);
         this.mainWindow.webContents.send('sar-error', error.message);
       });
       
       this.yoloProcess.on('close', (code) => {
         console.log(`YOLO process exited with code ${code}`);
         this.mainWindow.webContents.send('console-log', `YOLO process exited with code ${code}`);
         this.sarModeEnabled = false;
         this.mainWindow.webContents.send('sar-stopped');
       });
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
        if ($windows -like "*Argus SAR Detection*" -or $windows -eq "Argus SAR Detection") {
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
        sarModeEnabled: this.sarModeEnabled
      });
    });

    ipcMain.on('update-yolo-region', (event, coordinates) => {
      this.updateYOLORegion(coordinates);
    });
  }
}

// App initialization
console.log('Initializing Argus App...');
const argusApp = new ArgusApp();

app.whenReady().then(() => {
  console.log('Electron app ready - creating main window...');
  argusApp.createMainWindow();
  argusApp.setupIpcHandlers();
  console.log('App initialization complete');

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      console.log('App activated - creating new window...');
      argusApp.createMainWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    argusApp.cleanup();
    app.quit();
  }
});

app.on('before-quit', () => {
  argusApp.cleanup();
});
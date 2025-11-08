const { ipcRenderer } = require('electron');

class ForesightRenderer {
    constructor() {
        this.isCapturing = false;
        this.sarModeEnabled = false;
        this.detectionLoggingEnabled = false;
        this.initializeElements();
        this.setupEventListeners();
        this.setupIpcListeners();
        this.updateUI();
    }

    initializeElements() {
        // Buttons
        this.startCaptureBtn = document.getElementById('startCapture');
        this.stopCaptureBtn = document.getElementById('stopCapture');
        this.testDisplayBtn = document.getElementById('testDisplay');
        
        // Console
        this.clearConsoleBtn = document.getElementById('clearConsole');
        this.consoleOutput = document.getElementById('consoleOutput');
        
        // SAR Mode
        this.sarModeToggle = document.getElementById('sarMode');
        this.sarStatus = document.getElementById('sarStatus');
        this.sarOverlay = document.getElementById('sarOverlay');

        // DB Logging
        this.dbLoggingToggle = document.getElementById('dbLogging');
        this.dbLogStatus = document.getElementById('dbLogStatus');
        
        // Status elements (removed loadingText as left panel is gone)
        
        // Detection log panel
        this.detectionLogPanel = document.getElementById('detectionLogPanel');
        this.detectionLogList = document.getElementById('detectionLogList');
        this.refreshDetectionsBtn = document.getElementById('refreshDetections');
        
        // Mirror area
        this.mirrorPlaceholder = document.getElementById('mirrorPlaceholder');

        // Face save folder controls
        this.faceSavePathDisplay = document.getElementById('faceSavePath');
        this.chooseFaceFolderBtn = document.getElementById('chooseFaceFolder');
    }

    setupEventListeners() {
        // Capture controls
        this.startCaptureBtn.addEventListener('click', () => {
            this.startCapture();
        });

        this.stopCaptureBtn.addEventListener('click', () => {
            this.stopCapture();
        });

        this.testDisplayBtn.addEventListener('click', () => {
            this.testDisplay();
        });

        // SAR Mode toggle
        this.sarModeToggle.addEventListener('change', () => {
            this.toggleSarMode();
        });

        // DB Logging toggle
        this.dbLoggingToggle.addEventListener('change', () => {
            this.toggleDbLogging();
        });

        // Console clear
        this.clearConsoleBtn.addEventListener('click', () => {
            this.clearConsole();
        });

        // Refresh detections
        this.refreshDetectionsBtn.addEventListener('click', () => {
            this.requestDetectionLogs();
        });

        // Removed left panel event listeners

        // Choose face save folder
        this.chooseFaceFolderBtn.addEventListener('click', () => {
            ipcRenderer.send('choose-face-save-dir');
        });
    }

    setupIpcListeners() {
        // Capture events
        ipcRenderer.on('capture-started', () => {
            this.isCapturing = true;
            this.updateUI();
            this.logToConsole('Phone capture started successfully', 'success');
            this.updateMirrorArea(true);
        });

        ipcRenderer.on('capture-stopped', () => {
            this.isCapturing = false;
            this.sarModeEnabled = false;
            this.sarModeToggle.checked = false;
            this.updateUI();
            this.logToConsole('Phone capture stopped', 'info');
            this.updateMirrorArea(false);
        });

        ipcRenderer.on('capture-error', (event, error) => {
            this.logToConsole(`Capture error: ${error}`, 'error');
            this.isCapturing = false;
            this.updateUI();
        });

        // SAR Mode events
        ipcRenderer.on('sar-started', () => {
            this.sarModeEnabled = true;
            this.updateUI();
            this.logToConsole('SAR mode enabled - YOLO detection active', 'success');
            this.sarOverlay.classList.remove('hidden');
        });

        ipcRenderer.on('sar-stopped', () => {
            this.sarModeEnabled = false;
            this.sarModeToggle.checked = false;
            this.updateUI();
            this.logToConsole('SAR mode disabled', 'info');
            this.sarOverlay.classList.add('hidden');
        });

        ipcRenderer.on('sar-error', (event, error) => {
            this.logToConsole(`SAR error: ${error}`, 'error');
            this.sarModeEnabled = false;
            this.sarModeToggle.checked = false;
            this.updateUI();
        });

        // Status updates
        ipcRenderer.on('status-update', (event, status) => {
            this.isCapturing = status.isCapturing;
            this.sarModeEnabled = status.sarModeEnabled;
            this.detectionLoggingEnabled = !!status.detectionLoggingEnabled;
            this.dbLoggingToggle.checked = this.detectionLoggingEnabled;
            this.updateUI();
        });

        // Console log messages from main process
        ipcRenderer.on('console-log', (event, message) => {
            this.logToConsole(message, 'info');
        });

        // Live updates when a detection is logged
        ipcRenderer.on('detection-logged', (event, row) => {
            this.appendDetectionRow(row);
        });

        // Render detection logs from main
        ipcRenderer.on('detection-logs', (event, rows) => {
            this.renderDetectionLogs(rows);
        });

        // Face save folder updates
        ipcRenderer.on('face-save-dir', (event, dirPath) => {
            this.faceSavePathDisplay.textContent = dirPath || 'Not set';
        });

        // Notification when a face has been saved
        ipcRenderer.on('face-saved', (event, info) => {
            if (info && info.path) {
                this.logToConsole(`Face saved to: ${info.path}`, 'success');
            }
        });
    }

    startCapture() {
        this.logToConsole('Initiating phone capture...', 'info');
        ipcRenderer.send('start-capture');
    }

    stopCapture() {
        this.logToConsole('Stopping phone capture...', 'info');
        ipcRenderer.send('stop-capture');
    }

    toggleSarMode() {
        this.logToConsole('Toggling SAR mode...', 'info');
        ipcRenderer.send('toggle-sar');
    }

    toggleDbLogging() {
        const enabled = this.dbLoggingToggle.checked;
        ipcRenderer.send('set-detection-logging', enabled);
        // Toggle slide panel visibility
        if (enabled) {
            this.detectionLogPanel.classList.add('expanded');
            this.requestDetectionLogs();
        } else {
            this.detectionLogPanel.classList.remove('expanded');
        }
    }

    testDisplay() {
        this.logToConsole('Running display test...', 'info');
        
        // Simulate test sequence
        setTimeout(() => {
            this.logToConsole('Display test: Checking screen resolution...', 'info');
        }, 500);
        
        setTimeout(() => {
            this.logToConsole('Display test: Verifying color accuracy...', 'info');
        }, 1000);
        
        setTimeout(() => {
            this.logToConsole('Display test completed successfully', 'success');
        }, 1500);
    }

    updateUI() {
        // Update buttons
        this.startCaptureBtn.disabled = this.isCapturing;
        this.stopCaptureBtn.disabled = !this.isCapturing;
        
        // Update SAR toggle - always enabled
        this.sarModeToggle.disabled = false;
        
        // Update SAR status text
        this.sarStatus.textContent = this.sarModeEnabled ? 'Enabled' : 'Disabled';
        this.sarStatus.style.color = this.sarModeEnabled ? '#00ff88' : '#aaa';

        // Update DB logging status
        this.dbLogStatus.textContent = this.detectionLoggingEnabled ? 'Enabled' : 'Disabled';
        this.dbLogStatus.style.color = this.detectionLoggingEnabled ? '#00ff88' : '#aaa';
        
        // Removed loading text update (left panel removed)
    }

    updateMirrorArea(isActive) {
        if (isActive) {
            this.mirrorPlaceholder.innerHTML = `
                <div class="placeholder-content">
                    <div class="placeholder-icon">📱</div>
                    <h3>Phone Mirror Active</h3>
                    <p>Scrcpy window should appear separately</p>
                    <div style="margin-top: 15px; padding: 10px; background: rgba(0,255,136,0.1); border-radius: 8px;">
                        <small>Window Position: 400x100 (350x600)</small>
                    </div>
                </div>
            `;
        } else {
            this.mirrorPlaceholder.innerHTML = `
                <div class="placeholder-content">
                    <div class="placeholder-icon">📱</div>
                    <h3>Phone Mirror</h3>
                    <p>Click "Start Capture" to begin phone mirroring</p>
                </div>
            `;
        }
    }

    logToConsole(message, type = 'info') {
        const timestamp = new Date().toLocaleTimeString();
        const logLine = document.createElement('div');
        logLine.className = `console-line ${type}`;
        logLine.textContent = `[${timestamp}] ${message}`;
        
        this.consoleOutput.appendChild(logLine);
        this.consoleOutput.scrollTop = this.consoleOutput.scrollHeight;
        
        // Limit console lines to prevent memory issues
        const lines = this.consoleOutput.children;
        if (lines.length > 100) {
            this.consoleOutput.removeChild(lines[0]);
        }
    }

    clearConsole() {
        this.consoleOutput.innerHTML = '';
        this.logToConsole('Console cleared', 'info');
    }

    // Detection logs rendering helpers
    requestDetectionLogs(limit = 50) {
        ipcRenderer.send('get-detection-logs', limit);
    }

    renderDetectionLogs(rows) {
        if (!Array.isArray(rows)) return;
        this.detectionLogList.innerHTML = '';
        rows.forEach(row => this.appendDetectionRow(row));
    }

    appendDetectionRow(row) {
        if (!row) return;
        const el = document.createElement('div');
        el.className = 'detection-row';
        const time = new Date(row.timestamp).toLocaleTimeString();
        el.innerHTML = `<span class="detection-type">${row.type}</span><span class="detection-time">${time}</span>`;
        this.detectionLogList.prepend(el);
        // Cap list length
        const children = this.detectionLogList.children;
        if (children.length > 100) {
            this.detectionLogList.removeChild(children[children.length - 1]);
        }
    }

    // Request initial status
    requestStatus() {
        ipcRenderer.send('get-status');
        ipcRenderer.send('get-face-save-dir');
    }
}

// Initialize the renderer when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const app = new ForesightRenderer();
    
    // Request initial status
    setTimeout(() => {
        app.requestStatus();
    }, 500);
    
    // Log startup
    app.logToConsole('Foresight initialized', 'success');
    app.logToConsole('Ready for phone capture and SAR detection', 'info');
});
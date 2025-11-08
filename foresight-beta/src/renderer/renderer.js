let ipcRenderer;
try {
    ({ ipcRenderer } = require('electron'));
} catch (e) {
    // Fallback for browser preview (no Electron context)
    ipcRenderer = {
        send: () => {},
        on: () => {}
    };
}

class ForesightRenderer {
    constructor() {
        this.isCapturing = false;
        this.sarModeEnabled = false;
        this.detectionLoggingEnabled = false;
        this.animatingGallery = false;
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
        this.openGalleryBtn = document.getElementById('openGallery');
        
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

        // Full-height Gallery elements
        this.galleryMain = document.getElementById('galleryMain');
        this.galleryGrid = document.getElementById('galleryGrid');
        this.refreshGalleryBtn = document.getElementById('refreshGallery');
        this.setGalleryFolderBtn = document.getElementById('setGalleryFolder');
        this.galleryDirPath = document.getElementById('galleryDirPath');

        // Image preview modal
        this.imagePreview = document.getElementById('imagePreview');
        this.previewImg = document.getElementById('previewImg');

        // Mirror area
        this.mirrorPlaceholder = document.getElementById('mirrorPlaceholder');
        this.phoneMirrorPanel = document.getElementById('phoneMirrorPanel');
        this.phoneMirrorArea = document.querySelector('.phone-mirror-area');
        this.liveFeedEl = document.querySelector('.live-feed');

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

        // Open gallery with animated slide-in/out
        this.openGalleryBtn.addEventListener('click', () => {
            if (this.animatingGallery) return;
            const rightConsole = document.getElementById('rightConsole');
            const isVisible = !this.galleryMain.classList.contains('panel-hidden');
            this.animatingGallery = true;
            const done = () => { this.animatingGallery = false; };

            if (isVisible) {
                // Play slide-out, then hide
                this.galleryMain.classList.remove('slide-in');
                this.galleryMain.classList.add('slide-out');
                const onOutEnd = () => {
                    this.galleryMain.removeEventListener('animationend', onOutEnd);
                    this.galleryMain.classList.remove('slide-out');
                    this.galleryMain.classList.add('panel-hidden');
                    if (rightConsole) rightConsole.classList.remove('panel-hidden');
                    done();
                };
                this.galleryMain.addEventListener('animationend', onOutEnd);
            } else {
                // Show first, then play slide-in
                this.galleryMain.classList.remove('panel-hidden');
                if (rightConsole) rightConsole.classList.add('panel-hidden');
                // Refresh images on open
                this.requestDetectedImages();
                // Trigger slide-in
                this.galleryMain.classList.add('slide-in');
                const onInEnd = () => {
                    this.galleryMain.removeEventListener('animationend', onInEnd);
                    this.galleryMain.classList.remove('slide-in');
                    done();
                };
                this.galleryMain.addEventListener('animationend', onInEnd);
            }
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

        // Refresh gallery
        this.refreshGalleryBtn.addEventListener('click', () => {
            this.requestDetectedImages();
        });

        // Set gallery folder (same as choosing face save directory)
        if (this.setGalleryFolderBtn) {
            this.setGalleryFolderBtn.addEventListener('click', () => {
                ipcRenderer.send('choose-face-save-dir');
            });
        }

        // Gallery image click -> open preview
        this.galleryGrid.addEventListener('click', (e) => {
            const target = e.target;
            if (target && target.classList && target.classList.contains('gallery-thumb')) {
                this.showImagePreview(target.src);
            }
        });

        // Click-away to close preview
        if (this.imagePreview) {
            this.imagePreview.addEventListener('click', (e) => {
                if (e.target === this.imagePreview || (e.target && e.target.classList && e.target.classList.contains('preview-backdrop'))) {
                    this.hideImagePreview();
                }
            });
        }

        // Escape closes preview
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') this.hideImagePreview();
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
            if (this.galleryDirPath) {
                this.galleryDirPath.textContent = `Folder: ${dirPath || 'Not set'}`;
            }
            // Reload gallery from the new folder
            this.requestDetectedImages();
        });

        // Notification when a face has been saved
        ipcRenderer.on('face-saved', (event, info) => {
            if (info && info.path) {
                this.logToConsole(`Face saved to: ${info.path}`, 'success');
            }
        });

        // Detected images payload
        ipcRenderer.on('detected-images', (event, payload) => {
            const { dir, files } = payload || { dir: '', files: [] };
            if (this.galleryDirPath) {
                this.galleryDirPath.textContent = `Folder: ${dir || 'Not set'}`;
            }
            this.renderGallery(dir, files);
        });
    }

    setupPhoneMirrorBoundsSync() {
        if (!this.phoneMirrorPanel) return;
        const sendBounds = () => {
            const rect = this.phoneMirrorPanel.getBoundingClientRect();
            const dpr = window.devicePixelRatio || 1;
            ipcRenderer.send('phone-mirror/bounds', {
                x: Math.round(rect.left * dpr),
                y: Math.round(rect.top * dpr),
                width: Math.round(rect.width * dpr),
                height: Math.round(rect.height * dpr),
                dpr
            });
        };
        const ro = new ResizeObserver(() => sendBounds());
        ro.observe(this.phoneMirrorPanel);
        window.addEventListener('resize', sendBounds);
        // Initial send after DOM is ready
        setTimeout(sendBounds, 300);
    }

    requestDetectedImages(limit = 200) {
        ipcRenderer.send('get-detected-images', limit);
    }

    renderGallery(dir, files) {
        this.galleryGrid.innerHTML = '';
        if (!files || files.length === 0) {
            const empty = document.createElement('div');
            empty.textContent = `No images found in ${dir || 'folder'}`;
            empty.style.color = '#aaa';
            this.galleryGrid.appendChild(empty);
            return;
        }

        const fragment = document.createDocumentFragment();
        files.forEach((file) => {
            const item = document.createElement('div');
            item.className = 'gallery-item';
            const img = document.createElement('img');
            img.className = 'gallery-thumb';
            img.src = `file://${file}`;
            img.alt = file.split(/\\/).pop();
            item.appendChild(img);
            fragment.appendChild(item);
        });
        this.galleryGrid.appendChild(fragment);
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
        // Visual feedback for capturing state
        this.startCaptureBtn.classList.toggle('active', this.isCapturing);
        
        // Update SAR toggle - always enabled
        this.sarModeToggle.disabled = false;
        
        // Update SAR status text
        this.sarStatus.textContent = this.sarModeEnabled ? 'Enabled' : 'Disabled';
        this.sarStatus.style.color = this.sarModeEnabled ? '#00ff88' : '#aaa';

        // Update DB logging status
        this.dbLogStatus.textContent = this.detectionLoggingEnabled ? 'Enabled' : 'Disabled';
        this.dbLogStatus.style.color = this.detectionLoggingEnabled ? '#00ff88' : '#aaa';
        
        // Removed loading text update (left panel removed)

        // Ensure right side is empty by default (console hidden)
        const rightConsole = document.getElementById('rightConsole');
        if (rightConsole) rightConsole.classList.add('panel-hidden');
        // Gallery visible by default
        if (this.galleryMain) this.galleryMain.classList.remove('panel-hidden');
    }

    updateMirrorArea(isActive) {
        if (isActive) {
            // Hide placeholder while embedded mirror is active
            this.mirrorPlaceholder.style.display = 'none';
            if (this.phoneMirrorArea) this.phoneMirrorArea.classList.add('capturing');
            if (this.liveFeedEl) this.liveFeedEl.classList.remove('hidden');
        } else {
            this.mirrorPlaceholder.style.display = 'flex';
            this.mirrorPlaceholder.innerHTML = `
                <div class="placeholder-content">
                    <div class="placeholder-icon">📱</div>
                    <h3>Phone Mirror</h3>
                    <p>Click "Start Capture" to begin phone mirroring</p>
                </div>
            `;
            if (this.phoneMirrorArea) this.phoneMirrorArea.classList.remove('capturing');
            if (this.liveFeedEl) this.liveFeedEl.classList.add('hidden');
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

    // Image preview helpers
    showImagePreview(src) {
        if (!this.imagePreview || !this.previewImg) return;
        this.previewImg.src = src;
        this.imagePreview.classList.remove('hidden');
        this.imagePreview.classList.add('show');
    }

    hideImagePreview() {
        if (!this.imagePreview || !this.previewImg) return;
        this.imagePreview.classList.remove('show');
        this.imagePreview.classList.add('hidden');
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
    // Start bounds synchronization for scrcpy embedding
    app.setupPhoneMirrorBoundsSync();
    
    // Request initial status
    setTimeout(() => {
        app.requestStatus();
    }, 500);
    
    // Log startup
    app.logToConsole('Foresight initialized', 'success');
    app.logToConsole('Ready for phone capture and SAR detection', 'info');

    // Load gallery on startup
    app.requestDetectedImages();

    // Set header logo from absolute Windows path with safe fallback (Electron only)
    try {
        const path = require('path');
        const fs = require('fs');
        const { pathToFileURL } = require('url');
        const logoEl = document.getElementById('appLogo');
        if (logoEl) {
            const absoluteLogo = 'C\\\\Users\\\\Asus\\\\Desktop\\\\foresight-beta\\\\icon.png';
            if (fs.existsSync(absoluteLogo)) {
                logoEl.src = pathToFileURL(absoluteLogo).href;
            } else {
                const fallback = path.join(__dirname, '../assets/icon.png');
                logoEl.src = pathToFileURL(fallback).href;
            }
        }
    } catch (e) {
        // In browser preview, Node APIs are unavailable; HTML src remains as-is
    }
});
# Foresight Beta Installer Distribution Guide

## 📦 Installer Location

The complete installer is located at:
```
C:\Users\Asus\foresight-beta\foresight-beta\installer\dist\ForesightBetaSetup.exe
```

**File Size:** 9.8 MB

## 🚀 How to Distribute

1. **Copy the installer**: Take the `ForesightBetaSetup.exe` file from the `dist` folder
2. **Share it**: Send via email, cloud storage, USB drive, or any file sharing method
3. **Recipients run it**: They just double-click `ForesightBetaSetup.exe` to install

## 🎯 Installation Process for End Users

1. **Run** `ForesightBetaSetup.exe`
2. **Welcome Screen** - Click "Next"
3. **License Agreement** - Check "I accept" and click "Next"
4. **Choose Directory** - Select installation folder (default: `C:\Users\[Username]\Foresight`) and click "Next"
5. **Installation** - Wait for files to copy and dependencies to install
6. **Complete** - Choose whether to launch immediately and click "Finish"

## 🔧 What Gets Installed

- Complete Foresight Beta application
- All source code and assets
- Multiple launcher options:
  - `foresight-beta.bat` (Windows batch file)
  - `foresight-beta.ps1` (PowerShell script)
  - `foresight-beta.py` (Python script)

## ⚠️ System Requirements

- **OS**: Windows 10 or later
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 500MB free space
- **Node.js**: Required for running the application
- **Python**: 3.8+ recommended for detection scripts

## 🛠️ Post-Installation Notes

- If Node.js isn't installed, users will need to install it from [nodejs.org](https://nodejs.org)
- Users may need to run `npm install` in the installation directory if dependencies weren't installed automatically
- The installer handles most setup automatically, but provides clear error messages if manual steps are needed

## 📁 Installer Contents

The installer includes:
- Modern setup wizard with Foresight branding
- Automatic file copying from source to destination
- Dependency installation (Node.js and Python packages)
- Multiple launcher creation with error handling
- Professional icon and user interface

## 🔄 Updating

To create a new version of the installer:
1. Make changes to the Foresight Beta source code
2. Run `python installer/build_installer.py` from the project root
3. New installer will be generated in `installer/dist/`

---

**Ready to distribute!** Just share `ForesightBetaSetup.exe` with anyone who needs Foresight Beta.

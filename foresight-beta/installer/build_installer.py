#!/usr/bin/env python3
"""
Build script for Argus installer
Packages everything into a distributable installer executable
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def install_dependencies():
    """Install required dependencies for building"""
    dependencies = ['PyInstaller', 'Pillow', 'winshell', 'pywin32']
    
    for dep in dependencies:
        try:
            if dep == 'pywin32':
                __import__('win32com.client')
            else:
                __import__(dep.lower().replace('-', '_'))
        except ImportError:
            print(f"Installing {dep}...")
            subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)

def create_icon():
    """Create the application icon"""
    script_dir = Path(__file__).parent
    icon_script = script_dir / "create_icon.py"
    
    print("Creating application icon...")
    subprocess.run([sys.executable, str(icon_script)], check=True)

def build_installer():
    """Build the installer executable"""
    script_dir = Path(__file__).parent
    setup_script = script_dir / "setup.py"
    icon_path = script_dir / "assets" / "argus.ico"
    dist_dir = script_dir / "dist"
    
    # Create dist directory
    dist_dir.mkdir(exist_ok=True)
    
    # PyInstaller command for the installer
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--windowed",
        "--name", "ArgusSetup",
        "--distpath", str(dist_dir),
        "--workpath", str(script_dir / "build"),
        "--specpath", str(script_dir / "build")
    ]
    
    # Add icon if it exists
    if icon_path.exists():
        cmd.extend(["--icon", str(icon_path)])
    
    # Add data files (assets)
    assets_dir = script_dir / "assets"
    if assets_dir.exists():
        cmd.extend(["--add-data", f"{assets_dir};assets"])
    
    cmd.append(str(setup_script))
    
    print("Building installer executable...")
    subprocess.run(cmd, cwd=str(script_dir), check=True)
    
    # Clean up build files
    build_dir = script_dir / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    
    installer_path = dist_dir / "ArgusSetup.exe"
    if installer_path.exists():
        print(f"\nInstaller created successfully: {installer_path}")
        print(f"Size: {installer_path.stat().st_size / 1024 / 1024:.1f} MB")
        return installer_path
    else:
        raise Exception("Failed to create installer")

def create_readme():
    """Create a README file for the installer"""
    script_dir = Path(__file__).parent
    readme_content = """
# Argus Installer

## What is Argus?

Argus is an advanced object detection and tracking application featuring:

- **Ultra-high frequency YOLO detection** - Millisecond-level object detection
- **Real-time object tracking** - Continuous tracking with flicker-free rendering
- **SAR capabilities** - Search and Rescue mode for emergency situations
- **Modern interface** - Electron-based user interface
- **Cross-platform** - Works on Windows, macOS, and Linux

## Installation

1. Run `ArgusSetup.exe`
2. Follow the setup wizard (Next > Next > Next > Choose folder > Install)
3. Wait for installation to complete
4. Launch Argus from the installation folder

## System Requirements

- **Operating System**: Windows 10 or later
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 500MB free space
- **Node.js**: Will be required (installer will guide you)
- **Python**: 3.8 or later (for detection scripts)

## Usage

After installation, you can start Argus by:

- Double-clicking `argus.exe` in the installation folder
- Or running `start-argus.bat` from the installation folder

## Features

### Object Detection
- YOLOv8 neural network
- Real-time detection with optimized performance
- Multiple object classes supported

### Tracking
- Persistent object tracking across frames
- Smooth trajectory visualization
- Configurable tracking parameters

### Interface
- Clean, modern design
- Real-time video feed display
- Detection statistics and controls

## Support

For issues or questions about Argus, please check the documentation in the installation folder.

## License

Argus is provided "as is" for educational and research purposes.
"""
    
    readme_path = script_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"README created: {readme_path}")

def main():
    print("=== Argus Installer Build Script ===")
    print()
    
    try:
        # Install dependencies
        print("1. Installing dependencies...")
        install_dependencies()
        print("✓ Dependencies installed")
        print()
        
        # Create icon
        print("2. Creating application icon...")
        create_icon()
        print("✓ Icon created")
        print()
        
        # Create README
        print("3. Creating documentation...")
        create_readme()
        print("✓ Documentation created")
        print()
        
        # Build installer
        print("4. Building installer...")
        installer_path = build_installer()
        print("✓ Installer built successfully")
        print()
        
        print("=== Build Complete ===")
        print(f"Installer: {installer_path}")
        print()
        print("You can now distribute ArgusSetup.exe to install Argus on other systems.")
        
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
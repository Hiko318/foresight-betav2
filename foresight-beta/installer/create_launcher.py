#!/usr/bin/env python3
"""
Launcher creation script for Foresight Beta
Creates foresight-beta.exe that starts the application
"""

import os
import sys
import subprocess
from pathlib import Path

def create_launcher_script(install_path):
    """Create a Python launcher script"""
    launcher_content = f'''#!/usr/bin/env python3
"""
Foresight Beta Launcher
Starts the Foresight application
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Get the directory where this launcher is located
    launcher_dir = Path(__file__).parent.absolute()
    
    # Change to the application directory
    os.chdir(launcher_dir)
    
    try:
        # Start the application
        print("Starting Foresight Beta...")
        subprocess.run(["npm", "start"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error starting Foresight Beta: {{e}}")
        input("Press Enter to exit...")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: Node.js not found. Please install Node.js and try again.")
        input("Press Enter to exit...")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nForesight Beta stopped by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()
'''
    
    launcher_path = os.path.join(install_path, "foresight-beta-launcher.py")
    with open(launcher_path, 'w') as f:
        f.write(launcher_content)
    
    return launcher_path

def create_executable(launcher_script_path, install_path):
    """Create executable from Python script using PyInstaller"""
    try:
        # Check if PyInstaller is installed
        subprocess.run([sys.executable, "-m", "PyInstaller", "--version"], 
                      capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("PyInstaller not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], 
                      check=True)
    
    # Get paths
    script_dir = Path(__file__).parent
    icon_path = script_dir / "assets" / "foresight.ico"
    
    # PyInstaller command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--windowed",
        "--name", "foresight-beta",
        "--distpath", install_path,
        "--workpath", os.path.join(install_path, "build"),
        "--specpath", os.path.join(install_path, "build")
    ]
    
    # Add icon if it exists
    if icon_path.exists():
        cmd.extend(["--icon", str(icon_path)])
    
    cmd.append(launcher_script_path)
    
    print("Creating executable...")
    subprocess.run(cmd, cwd=install_path, check=True)
    
    # Clean up build files
    import shutil
    build_dir = os.path.join(install_path, "build")
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    
    # Remove the launcher script
    os.remove(launcher_script_path)
    
    exe_path = os.path.join(install_path, "foresight-beta.exe")
    if os.path.exists(exe_path):
        print(f"Executable created: {exe_path}")
        return exe_path
    else:
        raise Exception("Failed to create executable")

def create_batch_launcher(install_path):
    """Create a simple batch file launcher as fallback"""
    batch_content = f'''@echo off
cd /d "{install_path}"
echo Starting Foresight Beta...
npm start
pause
'''
    
    batch_path = os.path.join(install_path, "foresight-beta.bat")
    with open(batch_path, 'w') as f:
        f.write(batch_content)
    
    print(f"Batch launcher created: {batch_path}")
    return batch_path

def main():
    if len(sys.argv) != 2:
        print("Usage: python create_launcher.py <install_path>")
        sys.exit(1)
    
    install_path = sys.argv[1]
    
    try:
        # Create Python launcher script
        launcher_script = create_launcher_script(install_path)
        
        # Try to create executable
        try:
            create_executable(launcher_script, install_path)
        except Exception as e:
            print(f"Failed to create executable: {e}")
            print("Creating batch file launcher as fallback...")
            create_batch_launcher(install_path)
            
    except Exception as e:
        print(f"Error creating launcher: {e}")
        # Create batch file as ultimate fallback
        create_batch_launcher(install_path)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Standalone Installer Builder for Foresight Beta

This script creates a completely self-contained installer that embeds all
application files and dependencies, eliminating any possibility of recursive
installer launches.
"""

import os
import sys
import shutil
import zipfile
import base64
import json
import tempfile
from pathlib import Path

def get_project_root():
    """Get the project root directory"""
    current_dir = Path(__file__).parent
    return current_dir.parent

def create_application_archive():
    """Create a zip archive of all necessary application files"""
    project_root = get_project_root()
    
    print("Creating application archive...")
    
    # Files and directories to include
    include_items = [
        'src',
        'scripts', 
        'assets',
        'package.json',
        'package-lock.json',
        'requirements.txt',
        'README.md',
        'start-foresight.bat',
        'start-foresight.ps1'
    ]
    
    # Items to exclude completely
    exclude_items = {
        'installer',
        'Foresight Installer.bat',
        'Foresight Installer.ps1', 
        'Setup-Wizard.ps1',
        'Setup.bat',
        'ForesightBetaSetup.spec',
        '.git',
        '__pycache__',
        'node_modules',
        'dist',
        'build',
        '.gitignore',
        '.gitattributes'
    }
    
    # Create temporary zip file
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
        temp_zip_path = temp_zip.name
    
    with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for item in include_items:
            item_path = project_root / item
            
            if not item_path.exists():
                print(f"Warning: {item} not found, skipping...")
                continue
                
            print(f"Adding {item}...")
            
            if item_path.is_file():
                zipf.write(item_path, item)
            elif item_path.is_dir():
                # Add directory recursively, but exclude certain items
                for root, dirs, files in os.walk(item_path):
                    # Remove excluded directories from dirs list to prevent walking into them
                    dirs[:] = [d for d in dirs if d not in exclude_items]
                    
                    for file in files:
                        if file in exclude_items:
                            continue
                            
                        file_path = Path(root) / file
                        # Calculate relative path from project root
                        rel_path = file_path.relative_to(project_root)
                        zipf.write(file_path, str(rel_path))
    
    # Read the zip file and encode as base64
    with open(temp_zip_path, 'rb') as f:
        zip_data = f.read()
    
    os.unlink(temp_zip_path)
    
    encoded_data = base64.b64encode(zip_data).decode('utf-8')
    print(f"Application archive created ({len(zip_data)} bytes, {len(encoded_data)} base64 chars)")
    
    return encoded_data

def read_package_json():
    """Read and return package.json content"""
    project_root = get_project_root()
    package_json_path = project_root / 'package.json'
    
    if package_json_path.exists():
        with open(package_json_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        print("Warning: package.json not found")
        return None

def read_requirements_txt():
    """Read and return requirements.txt content"""
    project_root = get_project_root()
    requirements_path = project_root / 'requirements.txt'
    
    if requirements_path.exists():
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        print("Warning: requirements.txt not found")
        return None

def create_embedded_installer():
    """Create the standalone installer with embedded data"""
    print("Building standalone installer...")
    
    # Get embedded data
    app_data = create_application_archive()
    package_json = read_package_json()
    requirements = read_requirements_txt()
    
    # Read the standalone installer template
    installer_dir = Path(__file__).parent
    template_path = installer_dir / 'standalone_installer.py'
    
    with open(template_path, 'r', encoding='utf-8') as f:
        installer_code = f.read()
    
    # Replace the embedded data placeholders
    replacements = {
        'EMBEDDED_APP_DATA = None': f'EMBEDDED_APP_DATA = """{app_data}"""',
        'EMBEDDED_PACKAGE_JSON = None': f'EMBEDDED_PACKAGE_JSON = """{package_json or ""}"""' if package_json else 'EMBEDDED_PACKAGE_JSON = None',
        'EMBEDDED_REQUIREMENTS = None': f'EMBEDDED_REQUIREMENTS = """{requirements or ""}"""' if requirements else 'EMBEDDED_REQUIREMENTS = None'
    }
    
    for old, new in replacements.items():
        installer_code = installer_code.replace(old, new)
    
    # Write the final installer script
    output_path = installer_dir / 'foresight_standalone_installer.py'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(installer_code)
    
    print(f"Standalone installer script created: {output_path}")
    return output_path

def build_executable(installer_script_path):
    """Build the final executable using PyInstaller"""
    print("Building executable with PyInstaller...")
    
    installer_dir = Path(__file__).parent
    
    # PyInstaller command
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--onefile',
        '--windowed', 
        '--name', 'ForesightBetaStandaloneSetup',
        '--distpath', str(installer_dir / 'dist'),
        '--workpath', str(installer_dir / 'build'),
        '--specpath', str(installer_dir),
        str(installer_script_path)
    ]
    
    # Add icon if available
    icon_path = installer_dir / 'assets' / 'foresight.ico'
    if icon_path.exists():
        cmd.extend(['--icon', str(icon_path)])
    
    print(f"Running: {' '.join(cmd)}")
    
    import subprocess
    result = subprocess.run(cmd, cwd=installer_dir)
    
    if result.returncode == 0:
        exe_path = installer_dir / 'dist' / 'ForesightBetaStandaloneSetup.exe'
        if exe_path.exists():
            print(f"\n✅ Standalone installer built successfully!")
            print(f"📁 Location: {exe_path}")
            print(f"📊 Size: {exe_path.stat().st_size / 1024 / 1024:.1f} MB")
            print("\n🎉 This installer is completely self-contained and will not cause recursive launches!")
            return exe_path
        else:
            print("❌ Executable not found after build")
            return None
    else:
        print(f"❌ PyInstaller failed with exit code {result.returncode}")
        return None

def cleanup_temp_files():
    """Clean up temporary files created during build"""
    installer_dir = Path(__file__).parent
    
    # Keep the embedded installer script for verification
    temp_script = installer_dir / 'foresight_standalone_installer.py'
    if temp_script.exists():
        print(f"Keeping embedded installer script: {temp_script}")
    
    # Remove build directory
    build_dir = installer_dir / 'build'
    if build_dir.exists():
        shutil.rmtree(build_dir)
        print(f"Cleaned up: {build_dir}")
    
    # Remove spec file
    spec_file = installer_dir / 'ForesightBetaStandaloneSetup.spec'
    if spec_file.exists():
        spec_file.unlink()
        print(f"Cleaned up: {spec_file}")

def main():
    """Main build function"""
    print("🚀 Building Foresight Beta Standalone Installer")
    print("=" * 50)
    
    try:
        # Create the embedded installer script
        installer_script = create_embedded_installer()
        
        # Build the executable
        exe_path = build_executable(installer_script)
        
        if exe_path:
            print("\n" + "=" * 50)
            print("✅ BUILD SUCCESSFUL!")
            print(f"📦 Standalone installer: {exe_path}")
            print("\n🔧 This installer:")
            print("   • Contains all application files embedded")
            print("   • Creates fresh installation from scratch")
            print("   • Cannot cause recursive installer launches")
            print("   • Is completely self-contained")
        else:
            print("\n❌ BUILD FAILED!")
            return 1
            
    except Exception as e:
        print(f"\n❌ Build failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Clean up temporary files
        print("\n🧹 Cleaning up temporary files...")
        cleanup_temp_files()
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
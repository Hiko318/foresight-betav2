import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import shutil
import subprocess
import sys
import threading
import time
import base64
import zipfile
import tempfile
import json
from pathlib import Path

# Embedded application data will be inserted here by the build script
EMBEDDED_APP_DATA = None
EMBEDDED_PACKAGE_JSON = None
EMBEDDED_REQUIREMENTS = None

class StandaloneForesightInstaller:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Foresight Beta Setup")
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        self.root.configure(bg='white')
        
        # Try to set icon from embedded data
        try:
            self.setup_icon()
        except:
            pass
        
        self.current_step = 0
        self.install_path = tk.StringVar(value=os.path.join(os.path.expanduser("~"), "Desktop", "Foresight Beta"))
        self.installing = False
        
        self.setup_ui()
        self.show_welcome()
        
    def setup_icon(self):
        # Icon will be embedded as base64 data
        icon_data = """AAABAAEAEBAAAAEAIABoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAAAQAABILAAASCwAAAAAAAAAAAAD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///wAA"""
        
        if icon_data:
            try:
                import io
                icon_bytes = base64.b64decode(icon_data)
                with tempfile.NamedTemporaryFile(suffix='.ico', delete=False) as f:
                    f.write(icon_bytes)
                    self.root.iconbitmap(f.name)
                os.unlink(f.name)
            except:
                pass
        
    def setup_ui(self):
        # Main container
        main_frame = tk.Frame(self.root, bg='white', padx=30, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = tk.Label(main_frame, text="Foresight Beta Setup Wizard", 
                              font=('Arial', 18, 'bold'), bg='white', fg='#2c3e50')
        title_label.pack(pady=(0, 30))
        
        # Content area
        self.content_frame = tk.Frame(main_frame, bg='white')
        self.content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Button area
        button_frame = tk.Frame(main_frame, bg='white')
        button_frame.pack(fill=tk.X, pady=(30, 0))
        
        self.back_button = tk.Button(button_frame, text="← Back", 
                                    command=self.go_back, state=tk.DISABLED,
                                    font=('Arial', 10), padx=20)
        self.back_button.pack(side=tk.LEFT)
        
        self.cancel_button = tk.Button(button_frame, text="Cancel", 
                                      command=self.cancel_installation,
                                      font=('Arial', 10), padx=20)
        self.cancel_button.pack(side=tk.RIGHT, padx=(10, 0))
        
        self.next_button = tk.Button(button_frame, text="Next →", 
                                    command=self.go_next,
                                    font=('Arial', 10), padx=20, bg='#3498db', fg='white')
        self.next_button.pack(side=tk.RIGHT)
        
    def show_welcome(self):
        self.clear_content()
        
        welcome_label = tk.Label(self.content_frame, text="Welcome to Foresight Beta Setup!", 
                                font=('Arial', 16, 'bold'), bg='white', fg='#2c3e50')
        welcome_label.pack(pady=(20, 30))
        
        description = tk.Text(self.content_frame, height=12, wrap=tk.WORD, 
                             font=('Arial', 10), bg='white', relief=tk.FLAT, 
                             cursor="arrow", state=tk.DISABLED)
        description.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        desc_text = """This wizard will install Foresight Beta on your computer.

Foresight Beta is an advanced object detection and tracking system.

The installer will:
• Extract all application files to your chosen location
• Install Node.js dependencies (including Electron)
• Install Python dependencies
• Create desktop shortcuts

This is a completely standalone installer that contains all necessary files.

Click Next to continue."""
        
        description.config(state=tk.NORMAL)
        description.insert(tk.END, desc_text)
        description.config(state=tk.DISABLED)
        
        self.back_button.config(state=tk.DISABLED)
        self.next_button.config(text="Next →")
        
    def show_location_selection(self):
        self.clear_content()
        
        title_label = tk.Label(self.content_frame, text="Choose Installation Location", 
                              font=('Arial', 14, 'bold'), bg='white', fg='#2c3e50')
        title_label.pack(pady=(20, 30))
        
        info_label = tk.Label(self.content_frame, 
                             text="Select the folder where Foresight Beta will be installed:", 
                             font=('Arial', 10), bg='white')
        info_label.pack(pady=(0, 20))
        
        path_frame = tk.Frame(self.content_frame, bg='white')
        path_frame.pack(fill=tk.X, pady=(0, 20))
        
        path_entry = tk.Entry(path_frame, textvariable=self.install_path, 
                             font=('Arial', 10), width=50)
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        browse_button = tk.Button(path_frame, text="Browse...", 
                                 command=self.browse_location, font=('Arial', 10))
        browse_button.pack(side=tk.RIGHT)
        
        space_label = tk.Label(self.content_frame, 
                              text="Space required: ~200 MB", 
                              font=('Arial', 9), bg='white', fg='#7f8c8d')
        space_label.pack(pady=(20, 0))
        
        self.back_button.config(state=tk.NORMAL)
        self.next_button.config(text="Install")
        
    def browse_location(self):
        folder = filedialog.askdirectory(initialdir=self.install_path.get())
        if folder:
            self.install_path.set(os.path.join(folder, "Foresight Beta"))
            
    def show_installation(self):
        self.clear_content()
        self.installing = True
        
        title_label = tk.Label(self.content_frame, text="Installing Foresight Beta", 
                              font=('Arial', 14, 'bold'), bg='white', fg='#2c3e50')
        title_label.pack(pady=(20, 30))
        
        self.status_label = tk.Label(self.content_frame, text="Preparing installation...", 
                                    font=('Arial', 10), bg='white')
        self.status_label.pack(pady=(0, 20))
        
        self.progress = ttk.Progressbar(self.content_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(0, 20))
        self.progress.start()
        
        # Log area
        log_frame = tk.Frame(self.content_frame, bg='white')
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(log_frame, height=15, font=('Consolas', 9), 
                               bg='#f8f9fa', fg='#2c3e50')
        scrollbar = tk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.back_button.config(state=tk.DISABLED)
        self.next_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.DISABLED)
        
        # Start installation in separate thread
        threading.Thread(target=self.perform_installation, daemon=True).start()
        
    def perform_installation(self):
        try:
            install_dir = self.install_path.get()
            
            self.update_status("Creating installation directory...")
            self.log_message(f"Installing to: {install_dir}")
            
            # Remove existing directory if it exists
            if os.path.exists(install_dir):
                self.log_message("Removing existing installation...")
                shutil.rmtree(install_dir)
            
            # Create fresh installation directory
            os.makedirs(install_dir, exist_ok=True)
            self.log_message(f"✓ Created directory: {install_dir}")
            
            self.update_status("Extracting application files...")
            
            # Extract embedded application data
            if EMBEDDED_APP_DATA:
                self.log_message("Extracting embedded application files...")
                
                # Decode and extract the embedded zip data
                app_data = base64.b64decode(EMBEDDED_APP_DATA)
                
                with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
                    temp_zip.write(app_data)
                    temp_zip_path = temp_zip.name
                
                try:
                    with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                        zip_ref.extractall(install_dir)
                    self.log_message("✓ Application files extracted successfully")
                finally:
                    os.unlink(temp_zip_path)
            else:
                raise Exception("No embedded application data found")
            
            # Create package.json from embedded data
            if EMBEDDED_PACKAGE_JSON:
                self.log_message("Creating package.json...")
                package_json_path = os.path.join(install_dir, 'package.json')
                with open(package_json_path, 'w') as f:
                    f.write(EMBEDDED_PACKAGE_JSON)
                self.log_message("✓ package.json created")
            
            # Create requirements.txt from embedded data
            if EMBEDDED_REQUIREMENTS:
                self.log_message("Creating requirements.txt...")
                requirements_path = os.path.join(install_dir, 'requirements.txt')
                with open(requirements_path, 'w') as f:
                    f.write(EMBEDDED_REQUIREMENTS)
                self.log_message("✓ requirements.txt created")
            
            # Install Node.js dependencies
            self.update_status("Installing Node.js dependencies...")
            self.install_node_dependencies(install_dir)
            
            # Install Python dependencies
            self.update_status("Installing Python dependencies...")
            self.install_python_dependencies(install_dir)
            
            # Create shortcuts
            self.update_status("Creating shortcuts...")
            self.create_shortcuts(install_dir)
            
            self.update_status("Installation completed successfully!")
            self.log_message("\n🎉 Installation completed successfully!")
            self.log_message(f"Foresight Beta has been installed to: {install_dir}")
            
            self.root.after(0, self.show_completion)
            
        except Exception as e:
            self.log_message(f"\n❌ Installation failed: {str(e)}")
            self.root.after(0, lambda: self.show_error(str(e)))
            
    def install_node_dependencies(self, install_dir):
        self.log_message("Checking for Node.js...")
        
        try:
            result = subprocess.run(["npm", "--version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.log_message(f"✓ Found npm version: {result.stdout.strip()}")
                
                self.log_message("Running npm install...")
                result = subprocess.run(["npm", "install"], 
                                      cwd=install_dir, 
                                      capture_output=True, text=True, timeout=600,
                                      shell=True)
                
                if result.returncode == 0:
                    self.log_message("✓ Node.js dependencies installed successfully")
                else:
                    self.log_message(f"⚠ npm install had issues: {result.stderr[:200]}...")
                    
            else:
                self.log_message("✗ npm not found - Node.js dependencies not installed")
                
        except Exception as e:
            self.log_message(f"✗ Node.js installation failed: {str(e)}")
            
    def install_python_dependencies(self, install_dir):
        self.log_message("Installing Python packages...")
        
        try:
            requirements_path = os.path.join(install_dir, 'requirements.txt')
            if os.path.exists(requirements_path):
                result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                                      cwd=install_dir, 
                                      capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    self.log_message("✓ Python dependencies installed successfully")
                else:
                    self.log_message(f"⚠ Python install had warnings: {result.stderr[:100]}...")
            else:
                self.log_message("⚠ requirements.txt not found - skipping Python dependencies")
                
        except Exception as e:
            self.log_message(f"⚠ Python dependencies installation failed: {str(e)}")
            
    def create_shortcuts(self, install_dir):
        self.log_message("Creating desktop shortcut...")
        
        try:
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            shortcut_path = os.path.join(desktop, "Foresight Beta.bat")
            
            shortcut_content = f'''@echo off
cd /d "{install_dir}"
start "" "start-foresight.bat"
'''
            
            with open(shortcut_path, 'w') as f:
                f.write(shortcut_content)
            
            self.log_message(f"✓ Desktop shortcut created: {shortcut_path}")
            
        except Exception as e:
            self.log_message(f"⚠ Failed to create desktop shortcut: {str(e)}")
            
    def show_completion(self):
        self.clear_content()
        self.installing = False
        
        title_label = tk.Label(self.content_frame, text="Installation Complete!", 
                              font=('Arial', 16, 'bold'), bg='white', fg='#27ae60')
        title_label.pack(pady=(40, 30))
        
        success_text = f"""Foresight Beta has been successfully installed!

Installation location: {self.install_path.get()}

You can now:
• Use the desktop shortcut to launch Foresight Beta
• Navigate to the installation folder and run start-foresight.bat

Thank you for installing Foresight Beta!"""
        
        success_label = tk.Label(self.content_frame, text=success_text, 
                                font=('Arial', 11), bg='white', justify=tk.LEFT)
        success_label.pack(pady=(0, 40))
        
        self.progress.stop()
        self.next_button.config(text="Finish", state=tk.NORMAL)
        self.cancel_button.config(text="Close", state=tk.NORMAL)
        
    def show_error(self, error_msg):
        self.installing = False
        self.progress.stop()
        messagebox.showerror("Installation Error", 
                           f"Installation failed:\n\n{error_msg}\n\nPlease check the log for details.")
        self.cancel_button.config(state=tk.NORMAL, text="Close")
        
    def update_status(self, message):
        self.root.after(0, lambda: self.status_label.config(text=message))
        
    def log_message(self, message):
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        def update_log():
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, log_entry)
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
            
        self.root.after(0, update_log)
        
    def clear_content(self):
        for widget in self.content_frame.winfo_children():
            widget.destroy()
            
    def go_next(self):
        if self.current_step == 0:
            self.current_step = 1
            self.show_location_selection()
        elif self.current_step == 1:
            if not self.install_path.get().strip():
                messagebox.showerror("Error", "Please select an installation location.")
                return
            self.current_step = 2
            self.show_installation()
        elif self.current_step == 2:
            # Finish button
            self.root.quit()
            
    def go_back(self):
        if self.current_step == 1:
            self.current_step = 0
            self.show_welcome()
        elif self.current_step == 2 and not self.installing:
            self.current_step = 1
            self.show_location_selection()
            
    def cancel_installation(self):
        if self.installing:
            if messagebox.askyesno("Cancel Installation", 
                                 "Installation is in progress. Are you sure you want to cancel?"):
                self.root.quit()
        else:
            self.root.quit()
            
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = StandaloneForesightInstaller()
    app.run()
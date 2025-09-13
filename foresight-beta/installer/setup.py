import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import shutil
import subprocess
import sys
import threading
import time

class ArgusInstaller:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Argus Setup")
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        self.root.configure(bg='white')
        
        # Set icon if available
        try:
            icon_path = os.path.join(os.path.dirname(__file__), "assets", "argus.ico")
            if os.path.exists(icon_path):
                self.root.iconbitmap(icon_path)
        except:
            pass
        
        self.current_step = 0
        self.install_path = tk.StringVar(value=os.path.join(os.path.expanduser("~"), "Desktop", "Argus"))
        self.installing = False
        
        self.setup_ui()
        self.show_welcome()
        
    def setup_ui(self):
        # Main container
        main_frame = tk.Frame(self.root, bg='white', padx=30, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = tk.Label(main_frame, text="Argus Setup Wizard", 
                              font=('Arial', 18, 'bold'), bg='white', fg='#2c3e50')
        title_label.pack(pady=(0, 30))
        
        # Content area
        self.content_frame = tk.Frame(main_frame, bg='white')
        self.content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Button area
        button_frame = tk.Frame(main_frame, bg='white')
        button_frame.pack(fill=tk.X, pady=(30, 0))
        
        # Buttons
        self.back_btn = tk.Button(button_frame, text="← Back", 
                                 command=self.go_back, state=tk.DISABLED,
                                 font=('Arial', 10), width=12, height=2)
        self.back_btn.pack(side=tk.LEFT)
        
        self.cancel_btn = tk.Button(button_frame, text="Cancel", 
                                   command=self.cancel_install,
                                   font=('Arial', 10), width=12, height=2)
        self.cancel_btn.pack(side=tk.RIGHT)
        
        self.next_btn = tk.Button(button_frame, text="Next →", 
                                 command=self.go_next,
                                 font=('Arial', 10, 'bold'), width=12, height=2,
                                 bg='#3498db', fg='white')
        self.next_btn.pack(side=tk.RIGHT, padx=(0, 10))
        
    def clear_content(self):
        for widget in self.content_frame.winfo_children():
            widget.destroy()
            
    def show_welcome(self):
        self.clear_content()
        self.current_step = 0
        
        # Welcome message
        welcome_label = tk.Label(self.content_frame, text="Welcome to Argus Setup!", 
                                font=('Arial', 16, 'bold'), bg='white', fg='#2c3e50')
        welcome_label.pack(pady=(20, 20))
        
        # Description
        desc_text = """This wizard will install Argus on your computer.

Argus is an advanced object detection and tracking system.

The installer will:
• Copy all application files to your chosen location
• Install Node.js dependencies (including Electron)
• Install Python dependencies
• Create desktop shortcuts

Click Next to continue."""
        
        desc_label = tk.Label(self.content_frame, text=desc_text, 
                             font=('Arial', 11), bg='white', justify=tk.LEFT)
        desc_label.pack(pady=20)
        
        self.back_btn.config(state=tk.DISABLED)
        self.next_btn.config(text="Next →", command=self.show_location)
        
    def show_location(self):
        self.clear_content()
        self.current_step = 1
        
        # Title
        title_label = tk.Label(self.content_frame, text="Choose Installation Location", 
                              font=('Arial', 14, 'bold'), bg='white', fg='#2c3e50')
        title_label.pack(pady=(20, 20))
        
        # Description
        desc_label = tk.Label(self.content_frame, 
                             text="Select where you want to install Argus:", 
                             font=('Arial', 11), bg='white')
        desc_label.pack(pady=(0, 15))
        
        # Path selection
        path_frame = tk.Frame(self.content_frame, bg='white')
        path_frame.pack(fill=tk.X, pady=10)
        
        path_entry = tk.Entry(path_frame, textvariable=self.install_path, 
                             font=('Arial', 10), width=50)
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        browse_btn = tk.Button(path_frame, text="Browse...", 
                              command=self.browse_location,
                              font=('Arial', 10), width=10)
        browse_btn.pack(side=tk.RIGHT)
        
        # Space info
        space_label = tk.Label(self.content_frame, 
                              text="Space required: ~200 MB", 
                              font=('Arial', 9), bg='white', fg='#7f8c8d')
        space_label.pack(pady=(15, 0))
        
        self.back_btn.config(state=tk.NORMAL)
        self.next_btn.config(text="Install", command=self.start_installation)
        
    def browse_location(self):
        folder = filedialog.askdirectory(title="Select Installation Folder")
        if folder:
            self.install_path.set(os.path.join(folder, "Foresight Beta"))
            
    def start_installation(self):
        # Prevent multiple installations
        if self.installing:
            return
        self.installing = True
        
        self.clear_content()
        self.current_step = 2
        
        # Title
        title_label = tk.Label(self.content_frame, text="Installing Foresight Beta", 
                              font=('Arial', 14, 'bold'), bg='white', fg='#2c3e50')
        title_label.pack(pady=(20, 20))
        
        # Progress bar
        self.progress = ttk.Progressbar(self.content_frame, mode='indeterminate', length=400)
        self.progress.pack(pady=20)
        self.progress.start()
        
        # Status label
        self.status_label = tk.Label(self.content_frame, text="Preparing installation...", 
                                    font=('Arial', 11), bg='white', fg='#34495e')
        self.status_label.pack(pady=10)
        
        # Details text area
        self.details_text = tk.Text(self.content_frame, height=12, width=70, 
                                   font=('Consolas', 9), bg='#f8f9fa', 
                                   relief=tk.SUNKEN, bd=1)
        self.details_text.pack(pady=(20, 0), fill=tk.BOTH, expand=True)
        
        # Disable all buttons during installation
        self.back_btn.config(state=tk.DISABLED)
        self.next_btn.config(state=tk.DISABLED, text="Installing...")
        self.cancel_btn.config(state=tk.DISABLED)
        
        # Start installation in thread
        threading.Thread(target=self.install_files, daemon=True).start()
        
    def log_message(self, message):
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.details_text.insert(tk.END, log_entry)
        self.details_text.see(tk.END)
        self.root.update_idletasks()
        
    def update_status(self, message):
        self.status_label.config(text=message)
        self.root.update_idletasks()
        
    def install_files(self):
        try:
            install_dir = self.install_path.get()
            source_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            self.log_message("Starting installation...")
            self.update_status("Creating installation directory...")
            
            # Create installation directory
            os.makedirs(install_dir, exist_ok=True)
            self.log_message(f"Created directory: {install_dir}")
            
            self.update_status("Copying application files...")
            
            # Copy essential files and folders only (exclude installer files)
            items_to_copy = [
                'src', 'scripts', 'package.json', 'package-lock.json',
                'requirements.txt', 'README.md', 'start-argus.bat', 
                'start-argus.ps1'
            ]
            
            # Files and folders to explicitly exclude
            exclude_items = [
                'installer', 'Argus Installer.bat', 'Argus Installer.ps1',
                'Setup-Wizard.ps1', 'Setup.bat', 'ArgusSetup.spec',
                '.git', '__pycache__', 'node_modules', 'dist', 'build'
            ]
            
            # Copy main assets folder but exclude installer assets
            main_assets_src = os.path.join(source_dir, 'assets')
            main_assets_dst = os.path.join(install_dir, 'assets')
            if os.path.exists(main_assets_src):
                self.log_message("Copying assets folder...")
                shutil.copytree(main_assets_src, main_assets_dst, dirs_exist_ok=True)
                self.log_message("✓ Copied assets folder")
            
            for item in items_to_copy:
                # Skip if item is in exclusion list
                if item in exclude_items:
                    self.log_message(f"⚠ Skipped {item} (excluded)")
                    continue
                    
                src_path = os.path.join(source_dir, item)
                dst_path = os.path.join(install_dir, item)
                
                if os.path.exists(src_path):
                    self.log_message(f"Copying {item}...")
                    if os.path.isdir(src_path):
                        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src_path, dst_path)
                    self.log_message(f"✓ Copied {item}")
                else:
                    self.log_message(f"⚠ Skipped {item} (not found)")
            
            # Install Node.js dependencies
            self.update_status("Installing Node.js dependencies...")
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
                        self.log_message(f"⚠ npm install failed: {result.stderr[:200]}...")
                        
                else:
                    self.log_message("✗ npm not found")
                    
            except Exception as e:
                self.log_message(f"✗ Node.js installation failed: {str(e)}")
            
            # Install Python dependencies
            self.update_status("Installing Python dependencies...")
            self.log_message("Installing Python packages...")
            
            try:
                result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                                      cwd=install_dir, 
                                      capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    self.log_message("✓ Python dependencies installed successfully")
                else:
                    self.log_message(f"⚠ Python install had warnings: {result.stderr[:100]}...")
                    
            except Exception as e:
                self.log_message(f"⚠ Python dependencies installation failed: {str(e)}")
            
            # Create desktop shortcut
            self.update_status("Creating shortcuts...")
            self.log_message("Creating desktop shortcut...")
            
            try:
                desktop = os.path.join(os.path.expanduser("~"), "Desktop")
                shortcut_path = os.path.join(desktop, "Foresight Beta.bat")
                
                shortcut_content = f'''@echo off
cd /d "{install_dir}"
start "" "{os.path.join(install_dir, 'start-foresight.bat')}"
'''
                
                with open(shortcut_path, 'w') as f:
                    f.write(shortcut_content)
                    
                self.log_message("✓ Desktop shortcut created")
                
            except Exception as e:
                self.log_message(f"⚠ Could not create desktop shortcut: {str(e)}")
            
            self.update_status("Installation completed successfully!")
            self.log_message("")
            self.log_message("🎉 Installation completed successfully!")
            self.log_message(f"Foresight Beta is installed at: {install_dir}")
            self.log_message("You can now close this installer.")
            
            time.sleep(2)
            self.root.after(0, self.show_complete)
            
        except Exception as e:
            self.installing = False
            self.log_message(f"✗ Installation failed: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("Installation Error", 
                                                           f"Installation failed:\n{str(e)}"))
            self.root.after(0, lambda: self.next_btn.config(state=tk.NORMAL, text="Install"))
            self.root.after(0, lambda: self.back_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.cancel_btn.config(state=tk.NORMAL))
            
    def show_complete(self):
        self.installing = False
        
        self.clear_content()
        self.current_step = 3
        
        # Stop progress bar
        try:
            if hasattr(self, 'progress') and self.progress.winfo_exists():
                self.progress.stop()
        except:
            pass
        
        # Success message
        success_label = tk.Label(self.content_frame, text="Installation Complete!", 
                                font=('Arial', 16, 'bold'), bg='white', fg='#27ae60')
        success_label.pack(pady=(30, 20))
        
        # Info
        info_text = f"""Argus has been successfully installed!

Installation location:
{self.install_path.get()}

You can start Argus by:
• Double-clicking the desktop shortcut
• Running start-argus.bat in the installation folder

Thank you for installing Argus!"""
        
        info_label = tk.Label(self.content_frame, text=info_text, 
                             font=('Arial', 11), bg='white', justify=tk.CENTER)
        info_label.pack(pady=20)
        
        # Launch option
        self.launch_var = tk.BooleanVar(value=True)
        launch_check = tk.Checkbutton(self.content_frame, 
                                     text="Launch Argus now", 
                                     variable=self.launch_var,
                                     font=('Arial', 11), bg='white')
        launch_check.pack(pady=10)
        
        self.next_btn.config(text="Finish", command=self.finish_install, state=tk.NORMAL)
        
    def go_back(self):
        if self.current_step == 1:
            self.show_welcome()
        
    def go_next(self):
        pass  # Handled by specific button commands
        
    def cancel_install(self):
        if messagebox.askyesno("Cancel Installation", 
                              "Are you sure you want to cancel the installation?"):
            self.root.destroy()
            
    def finish_install(self):
        if self.launch_var.get():
            try:
                launcher_path = os.path.join(self.install_path.get(), "start-argus.bat")
                if os.path.exists(launcher_path):
                    subprocess.Popen([launcher_path], shell=True)
                    self.root.after(2000, self.root.destroy)
                else:
                    self.root.destroy()
            except Exception as e:
                messagebox.showwarning("Launch Error", 
                                     f"Could not launch Argus: {str(e)}")
                self.root.destroy()
        else:
            self.root.destroy()
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    installer = ArgusInstaller()
    installer.run()
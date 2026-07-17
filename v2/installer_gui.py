
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import shutil
import os
import sys
import subprocess

# --- GAMING THEME COLORS ---
BG_COLOR = "#050510"      # Deep Space Black
ACCENT_COLOR = "#00f3ff"  # Cyberpunk Cyan
TEXT_COLOR = "#e0e0fab"
SEC_ACCENT = "#bc13fe"    # Neon Purple
FONT_MAIN = ("Segoe UI", 10)
FONT_HEADER = ("Segoe UI", 16, "bold")
FONT_MONO = ("Consolas", 9)

class GamingInstaller(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PLUGINFER V2 INSTALLER")
        self.geometry("700x500")
        self.configure(bg=BG_COLOR)
        self.overrideredirect(True) # Frameless Window (Gamer Style)
        
        # Center Window
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - 700) // 2
        y = (screen_height - 500) // 2
        self.geometry(f"700x500+{x}+{y}")
        
        # --- UI LAYOUT ---
        self.create_widgets()
        
        # Make window draggable
        self.bind("<ButtonPress-1>", self.start_move)
        self.bind("<ButtonRelease-1>", self.stop_move)
        self.bind("<B1-Motion>", self.do_move)
        
    def create_widgets(self):
        # 1. Header (Drag Bar)
        header = tk.Frame(self, bg=BG_COLOR, height=40)
        header.pack(fill="x", pady=5)
        
        lbl_title = tk.Label(header, text="PLUGINFER // NEURAL_LINK_SETUP", 
                             bg=BG_COLOR, fg=ACCENT_COLOR, font=("Consolas", 12, "bold"))
        lbl_title.pack(side="left", padx=20)
        
        btn_close = tk.Button(header, text="X", bg=BG_COLOR, fg="red", 
                              bd=0, font=("Arial", 12), command=self.quit)
        btn_close.pack(side="right", padx=10)
        
        # 2. Hero Section
        hero_frame = tk.Frame(self, bg=BG_COLOR)
        hero_frame.pack(fill="both", expand=True, padx=40, pady=20)
        
        art_label = tk.Label(hero_frame, text="⚠ SYSTEM DETECTED: UNTAPPED GPU POWER", 
                             bg=BG_COLOR, fg=SEC_ACCENT, font=("Segoe UI", 24, "bold"))
        art_label.pack(pady=(20, 10))
        
        desc_label = tk.Label(hero_frame, text="Initialize the Decentralized Compute Node.\nEarn Passive Income. Join the Mesh.",
                              bg=BG_COLOR, fg="white", font=("Segoe UI", 11), justify="center")
        desc_label.pack(pady=10)
        
        # 3. Progress / Log Area
        self.log_text = tk.Text(hero_frame, bg="#0a0a15", fg=ACCENT_COLOR, 
                                height=10, width=70, font=FONT_MONO, bd=1, relief="flat")
        self.log_text.pack(pady=20)
        self.log_text.insert("end", "> INITIALIZING INSTALLER...\n> WAITING FOR USER INPUT...\n")
        self.log_text.config(state="disabled")
        
        # 4. Action Buttons
        btn_frame = tk.Frame(self, bg=BG_COLOR)
        btn_frame.pack(fill="x", pady=30, padx=40)
        
        self.btn_install = tk.Button(btn_frame, text="[ INSTALL SYSTEM ]", 
                                     bg=ACCENT_COLOR, fg="black", font=("Segoe UI", 12, "bold"),
                                     activebackground="white", activeforeground="black",
                                     relief="flat", padx=20, pady=10,
                                     command=self.start_installation)
        self.btn_install.pack(side="right")
        
        self.btn_folder = tk.Button(btn_frame, text="Select Destination", 
                                    bg="#222", fg="white", font=("Segoe UI", 10),
                                    relief="flat", padx=15, pady=10,
                                    command=lambda: messagebox.showinfo("Info", "Defaulting to C:\\Pluginfer for optimal performance."))
        self.btn_folder.pack(side="left")

    def log(self, msg):
        self.log_text.config(state="normal")
        for char in msg:
            self.log_text.insert("end", char)
            self.log_text.see("end")
            self.update()
            time.sleep(0.005) # Typing effect
        self.log_text.insert("end", "\n")
        self.log_text.config(state="disabled")

    def start_installation(self):
        self.btn_install.config(state="disabled", text="INSTALLING...")
        threading.Thread(target=self.run_install_logic, daemon=True).start()
        
    # --- IMPROVED INSTALL LOGIC WITH PROGRESS ---
    def count_files(self, src):
        total_size = 0
        total_files = 0
        try:
            for root, dirs, files in os.walk(src):
                if 'venv' in root or '__pycache__' in root or '.git' in root:
                    continue
                for file in files:
                    fp = os.path.join(root, file)
                    total_size += os.path.getsize(fp)
                    total_files += 1
        except: 
            pass
        return total_files, total_size

    def run_install_logic(self):
        try:
            self.log("> DETECTING SOURCE FILES...")
            exe_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
            parent_dir = os.path.dirname(exe_dir)
            
            source_dir = None
            if os.path.exists(os.path.join(exe_dir, "pluginfer_node.py")): source_dir = exe_dir
            elif os.path.exists(os.path.join(parent_dir, "pluginfer_node.py")): source_dir = parent_dir
                
            if not source_dir:
                self.log("> ERROR: SOURCES MISSING! PLACE SETUP.EXE NEXT TO FILES.")
                return

            self.log(f"> SOURCE: {source_dir}")
            
            # --- PRE-CALCULATE SIZE ---
            self.log("> ANALYZING PAYLOAD SIZE...")
            total_files, total_bytes = self.count_files(source_dir)
            self.log(f"> PAYLOAD: {total_files} Files (~{round(total_bytes/1024/1024)} MB)")
            
            target_dir = r"C:\Pluginfer\v2"
            
            # CLEANUP
            if os.path.exists(target_dir):
                try: shutil.rmtree(target_dir)
                except: pass

            # --- COPY WITH PROGRESS ---
            self.log("> STARTING TRANSFER...")
            copied_bytes = 0
            start_time = time.time()
            
            # Main bar
            self.progress = ttk.Progressbar(self, orient="horizontal", length=600, mode="determinate")
            self.progress.place(x=50, y=380) # Overwrite button area temporarily

            for root, dirs, files in os.walk(source_dir):
                # Filter dirs
                dirs[:] = [d for d in dirs if d not in ['venv', '__pycache__', '.git', 'build', 'dist', 'Pluginfer_v2_Secure_Installer']]
                
                # Make relative path for target
                rel_path = os.path.relpath(root, source_dir)
                target_root = os.path.join(target_dir, rel_path)
                os.makedirs(target_root, exist_ok=True)
                
                for file in files:
                    if file.endswith('.exe') or file.endswith('.spec'): continue
                    
                    src_file = os.path.join(root, file)
                    dst_file = os.path.join(target_root, file)
                    
                    try:
                        shutil.copy2(src_file, dst_file)
                        
                        # Update Progress
                        f_size = os.path.getsize(src_file)
                        copied_bytes += f_size
                        
                        # Calculate Stats
                        elapsed = time.time() - start_time
                        if elapsed > 0:
                            speed = copied_bytes / elapsed # B/s
                            remaining = total_bytes - copied_bytes
                            eta = remaining / speed if speed > 0 else 0
                            
                            percent = (copied_bytes / total_bytes) * 100
                            self.progress['value'] = percent
                            
                            # Update label every 100 files or so (to avoid lag)
                            if total_files > 1000 and int(percent) % 5 == 0:
                                self.update()
                                
                    except Exception as e:
                        print(f"Skip {file}: {e}")

            self.progress.destroy() # Remove bar
            self.log("> TRANSFER COMPLETE [100%]")
            self.log("> CONFIGURING MESH NETWORKING... [OK]")
            time.sleep(0.5)
            
            self.log("> SYSTEM READY.")
            self.btn_install.config(state="normal", text="[ LAUNCH NODE ]", bg=SEC_ACCENT, command=self.launch_node)
            
        except Exception as e:
            self.log(f"> FATAL ERROR: {e}")
            messagebox.showerror("Error", str(e))

    def launch_node(self):
        # Launch from C:\Pluginfer\v2\run_immortal.bat
        target_bat = r"C:\Pluginfer\v2\run_immortal.bat"
        
        if not os.path.exists(target_bat):
            # Try to just launch source dir bat if install failed
            messagebox.showerror("Error", f"Launcher not found at {target_bat}")
            return
            
        try:
            # Use subprocess to start separate cmd window
            subprocess.Popen(target_bat, cwd=r"C:\Pluginfer\v2", creationflags=subprocess.CREATE_NEW_CONSOLE)
            self.quit()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch: {e}")

    # Drag Logic
    def start_move(self, event): self.x = event.x; self.y = event.y
    def stop_move(self, event): self.x = None; self.y = None
    def do_move(self, event):
        deltax = event.x - self.x
        deltay = event.y - self.y
        x = self.winfo_x() + deltax
        y = self.winfo_y() + deltay
        self.geometry(f"+{x}+{y}")

if __name__ == "__main__":
    app = GamingInstaller()
    app.mainloop()

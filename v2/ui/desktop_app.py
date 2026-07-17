"""
Pluginfer Desktop App (GUI)
The premium control panel for the Pluginfer Mesh Network.
"""
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import logging
import queue
try:
    from PIL import Image, ImageTk, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logging.warning("PIL not found. Using text-only UI elements.")

# Core Imports
from core.mesh_controller import (
    MeshNetworkController, get_system_capabilities, 
    find_coordinator, connect_to_mesh
)
from core.system_doctor import SystemDoctor
from core.payments import StripeMockGateway
from core.updater import AutoUpdater
from core.game_detector import GameDetector

# Setup Logging
setup_logger = logging.getLogger()
setup_logger.setLevel(logging.INFO)

class ModernTheme:
    """Dark Mode Color Palette"""
    BG_DARK = "#0f0f23"
    BG_LIGHT = "#1a1a2e" 
    ACCENT = "#667eea"
    TEXT_MAIN = "#e0e0e0"
    TEXT_DIM = "#a0a0a0"
    SUCCESS = "#2ecc71"
    ERROR = "#e74c3c"

class PluginferApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pluginfer Desktop")
        self.root.geometry("900x600")
        self.root.configure(bg=ModernTheme.BG_DARK)
        
        # State
        self.node = None
        self.node = None
        self.doctor = None
        self.game_detector = None
        self.payment_gateway = StripeMockGateway()
        self.payment_gateway = StripeMockGateway()
        self.updater = AutoUpdater()
        self.is_running = False
        self.log_queue = queue.Queue()
        
        # Setup Styles
        self._setup_styles()
        
        # Layout
        self._create_header()
        self._create_main_content()
        self._create_footer()
        
        # Start Background Threads
        self._start_log_monitor()
        self._start_log_monitor()
        self._check_for_updates()
        self._check_gaming_status()
        
    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure("TFrame", background=ModernTheme.BG_DARK)
        style.configure("Card.TFrame", background=ModernTheme.BG_LIGHT, relief="flat")
        
        style.configure("TLabel", background=ModernTheme.BG_DARK, foreground=ModernTheme.TEXT_MAIN, font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 20, "bold"), foreground="white")
        style.configure("Stat.TLabel", font=("Segoe UI", 24, "bold"), foreground=ModernTheme.ACCENT)
        style.configure("Substat.TLabel", font=("Segoe UI", 10), foreground=ModernTheme.TEXT_DIM)
        
        style.configure("TButton", 
            background=ModernTheme.ACCENT, 
            foreground="white", 
            font=("Segoe UI", 10, "bold"),
            borderwidth=0,
            focuscolor=ModernTheme.ACCENT
        )
        style.map("TButton", background=[('active', '#5a6fd6')])
        
    def _create_header(self):
        header = ttk.Frame(self.root, padding="20")
        header.pack(fill="x")
        
        title = ttk.Label(header, text="🌐 Pluginfer Network", style="Header.TLabel")
        title.pack(side="left")
        
        self.status_indicator = tk.Canvas(header, width=15, height=15, bg=ModernTheme.BG_DARK, highlightthickness=0)
        self.status_indicator.pack(side="right", padx=10)
        self.status_indicator_circle = self.status_indicator.create_oval(2, 2, 13, 13, fill="gray")
        
        self.status_text = ttk.Label(header, text="OFFLINE", foreground="gray")
        self.status_text.pack(side="right")

    def _create_main_content(self):
        main = ttk.Frame(self.root, padding="20")
        main.pack(fill="both", expand=True)
        
        # Stats Row
        stats_frame = ttk.Frame(main)
        stats_frame.pack(fill="x", pady=(0, 20))
        
        self._create_stat_card(stats_frame, "Earnings", "$0.00", "Total Revenue").pack(side="left", fill="x", expand=True, padx=5)
        self._create_stat_card(stats_frame, "Tasks", "0", "Completed").pack(side="left", fill="x", expand=True, padx=5)
        self._create_stat_card(stats_frame, "Health", "100%", "System Status").pack(side="left", fill="x", expand=True, padx=5)
        
        # Controls & Logs
        content_split = ttk.Frame(main)
        content_split.pack(fill="both", expand=True)
        
        # Left: Controls
        controls = ttk.Frame(content_split, style="Card.TFrame", padding="15")
        controls.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        ttk.Label(controls, text="Node Controls", font=("Segoe UI", 12, "bold"), background=ModernTheme.BG_LIGHT).pack(anchor="w", mb=10)
        
        self.btn_start = ttk.Button(controls, text="START NODE", command=self.toggle_node)
        self.btn_start.pack(fill="x", pady=5)
        
        ttk.Button(controls, text="OPEN DASHBOARD", command=self.open_dashboard).pack(fill="x", pady=5)
        ttk.Button(controls, text="RUN SYSTEM DOCTOR", command=self.run_doctor_scan).pack(fill="x", pady=5)
        
        # Right: Logs
        logs_frame = ttk.Frame(content_split, style="Card.TFrame", padding="15")
        logs_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))
        
        ttk.Label(logs_frame, text="Activity Log", font=("Segoe UI", 12, "bold"), background=ModernTheme.BG_LIGHT).pack(anchor="w", mb=10)
        
        self.log_text = tk.Text(logs_frame, height=10, bg="#0a0a12", fg="#00ff00", font=("Consolas", 9), relief="flat")
        self.log_text.pack(fill="both", expand=True)
        
    def _create_stat_card(self, parent, title, value, subtitle):
        card = ttk.Frame(parent, style="Card.TFrame", padding="15")
        ttk.Label(card, text=title, background=ModernTheme.BG_LIGHT, foreground=ModernTheme.TEXT_DIM).pack(anchor="w")
        lbl_val = ttk.Label(card, text=value, background=ModernTheme.BG_LIGHT, style="Stat.TLabel")
        lbl_val.pack(anchor="w")
        ttk.Label(card, text=subtitle, background=ModernTheme.BG_LIGHT, style="Substat.TLabel").pack(anchor="w")
        return card

    def _create_footer(self):
        footer = ttk.Frame(self.root, padding="10")
        footer.pack(fill="x", side="bottom")
        
        self.lbl_version = ttk.Label(footer, text="v0.9.0", foreground="gray")
        self.lbl_version.pack(side="right")
        
        self.lbl_doctor_status = ttk.Label(footer, text="🩺 System Doctor: Ready", foreground=ModernTheme.SUCCESS)
        self.lbl_doctor_status.pack(side="left")

    # --- Actions ---
    
    def toggle_node(self):
        if not self.is_running:
            self.start_node()
        else:
            self.stop_node()
            
    def start_node(self):
        self.is_running = True
        self.btn_start.configure(text="STOP NODE")
        self._update_status(True)
        self.log("Starting Pluginfer Mesh Node...")
        
        # Init Backend as WORKER (Default for User App)
        self.node = MeshNetworkController(mode='worker', enable_discovery=True)
        self.node.start()
        
        # Init Doctor
        self.doctor = SystemDoctor(self.node)
        self.doctor.start()
        
        # Init Game Detector
        self.game_detector = GameDetector(self.node)
        self.game_detector.start()
        
        self.log(f"Node started! ID: {self.node.node_id[:8]}...")
        
        # Auto-Connect to Coordinator
        threading.Thread(target=self._auto_connect_to_mesh, daemon=True).start()

    def _auto_connect_to_mesh(self):
        self.log("📡 Searching for Coordinator...")
        coord_info = find_coordinator(timeout=3)
        
        if coord_info:
            host = coord_info['ip']
            port = coord_info['port']
            self.log(f"✅ Found Coordinator: {host}")
            
            # Register
            caps = get_system_capabilities()
            success = connect_to_mesh(host, port, caps)
            
            if success:
                self.log("🔗 Successfully joined Mesh Network!")
                self.root.after(0, lambda: self.status_text.configure(text="CONNECTED", foreground="#2ecc71"))
            else:
                self.log("❌ Failed to join mesh.")
        else:
            self.log("⚠️ No Coordinator found. Running in local mode.")
        
    def stop_node(self):
        self.is_running = False
        self.btn_start.configure(text="START NODE")
        self._update_status(False)
        self.log("Stopping node...")
        
        if self.doctor: self.doctor.stop()
        if self.doctor: self.doctor.stop()
        if self.game_detector: self.game_detector.stop()
        if self.node: self.node.stop()
        
    def run_doctor_scan(self):
        self.log("Running System Doctor Diagnostics...")
        if not self.doctor:
            self.doctor = SystemDoctor(None) # Temp doctor
            
        report = self.doctor.run_diagnostics()
        self.log(f"Diagnostics: {report}")
        
        issues = report.get('issues', [])
        if not issues:
            self.lbl_doctor_status.configure(text="🩺 System Doctor: All Systems Healthy", foreground=ModernTheme.SUCCESS)
            messagebox.showinfo("Health Check", "System is healthy! 🌟")
        else:
            self.lbl_doctor_status.configure(text=f"🩺 Issues Found: {len(issues)}", foreground=ModernTheme.ERROR)
            messagebox.showwarning("Health Check", f"Issues found: {', '.join(issues)}\nDoctor will attempt repair.")

    def open_dashboard(self):
        import webbrowser
        webbrowser.open("http://localhost:8000")
        self.log("Opening dashboard implemented in Web UI...")

    def _check_for_updates(self):
        def check():
            update = self.updater.check_for_updates()
            if update:
                self.root.after(0, lambda: messagebox.showinfo("Update Available", f"New version {update['version']} available!"))
        threading.Thread(target=check, daemon=True).start()

    def _check_gaming_status(self):
        """Monitor for gaming pauses"""
        if self.node and self.node.is_paused:
             self.status_indicator.itemconfig(self.status_indicator_circle, fill="#f1c40f") # Yellow
             self.status_text.configure(text="GAMING PAUSED", foreground="#f1c40f")
        elif self.is_running:
             # Only revert to online if not paused and IS running
             # This simple logic might flicker, but is sufficient for prototype
             # Better: store last state or check if text is NOT "ONLINE"
             current_text = self.status_text.cget("text")
             if current_text == "GAMING PAUSED":
                self._update_status(True)

        self.root.after(1000, self._check_gaming_status)

    def _update_status(self, online):
        color = ModernTheme.SUCCESS if online else "gray"
        self.status_indicator.itemconfig(self.status_indicator_circle, fill=color)
        self.status_text.configure(text="ONLINE" if online else "OFFLINE", foreground=color)
        
    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        self.log_text.see("end")

    def _start_log_monitor(self):
        # Placeholder if we wanted to pipe stdout to GUI
        pass

if __name__ == "__main__":
    root = tk.Tk()
    app = PluginferApp(root)
    root.mainloop()

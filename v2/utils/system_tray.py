"""
System Tray Integration for Pluginfer Node
Allows the application to run in the background with a system tray icon.
"""
import threading
import os
import sys
import webbrowser
from PIL import Image, ImageDraw
import pystray
from pystray import MenuItem as item

class SystemTrayApp:
    def __init__(self, node_controller, dashboard_url):
        self.node = node_controller
        self.dashboard_url = dashboard_url
        self.icon = None
        self.running = True

    def create_image(self):
        # Generate a simple icon if assets missing
        # Green circle with 'P'
        width = 64
        height = 64
        color1 = (0, 0, 0)
        color2 = (0, 255, 0)
        
        image = Image.new('RGB', (width, height), color1)
        dc = ImageDraw.Draw(image)
        dc.ellipse((8, 8, 56, 56), fill=color2)
        dc.text((20, 16), "P", fill=(0,0,0), font_size=40) # Simple fallback
        
        return image

    def on_dashboard(self):
        webbrowser.open(self.dashboard_url)

    def on_pause(self, icon, item):
        self.node.toggle_pause()
        # Update menu text (requires recreating icon menu usually, or dynamic check)
        # pystray dynamic menus are a bit complex, sticking to simple toggle logic for now.

    def on_exit(self, icon, item):
        self.running = False
        icon.stop()
        print("Stopping Node...")
        self.node.stop()
        os._exit(0) # Force kill

    def setup(self):
        image = self.create_image()
        
        # Determine Menu
        menu = (
            item('Open Dashboard', self.on_dashboard, default=True),
            item('Pause/Resume Computation', self.on_pause),
            item('Exit Pluginfer', self.on_exit)
        )

        self.icon = pystray.Icon("Pluginfer", image, "Pluginfer Node (Active)", menu)

    def run(self):
        self.setup()
        self.icon.run()

if __name__ == "__main__":
    # Test stub
    pass

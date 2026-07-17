"""
Resizes an image using PIL.
"""
from typing import Dict, Any, List
import logging
import sys
import os
from pathlib import Path

# Robust import strategy for both Dev and Frozen (PyInstaller) modes
# We need to ensure 'core' is importable.

# 1. Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 2. Add parent directory (project root) to path
parent_dir = str(Path(current_dir).parent.absolute())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 3. Add grandparent directory (just in case)
grandparent_dir = str(Path(current_dir).parent.parent.absolute())
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)

try:
    from core.plugin_base import PluginBase
except ImportError:
    # Fallback: Try finding core relative to this file location
    # This is often needed in PyInstaller temp dirs
    try:
        sys.path.append(os.path.join(current_dir, '..'))
        from core.plugin_base import PluginBase
    except ImportError:
         # Last resort: Try absolute path to MEIPASS if frozen
         if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
             sys.path.append(sys._MEIPASS)
             from core.plugin_base import PluginBase
         else:
             raise


logger = logging.getLogger(__name__)

class ImgResize(PluginBase):
    def config(self) -> Dict[str, Any]:
        return {
            "name": "img_resize",
            "version": "1.0.0",
            "description": "Resizes an image using PIL.",
            "category": "image",
            "tags": ['image'],
            "cost_per_exec": 0.005,
            "inputs": {'width': 'int', 'height': 'int', 'image_data': 'str'},
            "outputs": {'data': 'str'}
        }

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:

                        import base64
                        from io import BytesIO
                        from PIL import Image

                        # Decode Input
                        b64_data = input_data.get('image_data') or input_data.get('data')
                        if not b64_data:
                            raise ValueError("No image data provided")
            
                        # Handle potential header prefix (data:image/png;base64,...)
                        if ',' in b64_data:
                            b64_data = b64_data.split(',')[1]

                        img_bytes = base64.b64decode(b64_data)
                        img = Image.open(BytesIO(img_bytes))

                        # Process
                        width = int(input_data.get('width', 800))
                        height = int(input_data.get('height', 600))
                        resized_img = img.resize((width, height))

                        # Encode Output
                        buffered = BytesIO()
                        resized_img.save(buffered, format=img.format or "PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

                        return {
                            "status": "success", 
                            "message": f"Resized to {width}x{height}", 
                            "original_size": len(img_bytes),
                            "processed_size": len(buffered.getvalue()),
                            "data": img_str,
                            "format": img.format
                        }
         
        except Exception as e:
            logger.error(f"Plugin img_resize failed: {e}")
            return {"error": str(e)}

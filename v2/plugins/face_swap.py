"""
Face Swap Plugin.
Swaps faces between two images using OpenCV (Basic Overlay).
"""
from typing import Dict, Any, List
import logging
import sys
import os
import base64
from io import BytesIO
from pathlib import Path

# Robust import strategy
current_dir = os.path.dirname(os.path.abspath(__file__))
# ... (standard import logic) ...
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
parent_dir = str(Path(current_dir).parent.absolute())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
grandparent_dir = str(Path(current_dir).parent.parent.absolute())
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)

try:
    from core.plugin_base import PluginBase
except ImportError:
    try:
        sys.path.append(os.path.join(current_dir, '..'))
        from core.plugin_base import PluginBase
    except ImportError:
         if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
             sys.path.append(sys._MEIPASS)
             from core.plugin_base import PluginBase
         else:
             raise

logger = logging.getLogger(__name__)

class FaceSwap(PluginBase):
    def config(self) -> Dict[str, Any]:
        return {
            "name": "face_swap",
            "version": "1.0.0",
            "description": "Swaps face from Source to Target.",
            "category": "image",
            "tags": ['image', 'ai', 'face'],
            "cost_per_exec": 0.05,
            "inputs": {
                # We expect a list of 2 images OR two separate keys.
                # Standard UI sends 'image_data'. We might need to handle multipart.
                # For simplicity, let's assume 'source' and 'target' keys in input_data
                # OR if standard UI, it only sends ONE file. 
                # LIMITATION: Standard UI single-file upload doesn't support 2 files yet.
                # We will just swap the face within the SAME image (shuffle faces) if 1 image.
                'image_data': 'str' 
            },
            "outputs": {'data': 'str'}
        }

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            try:
                import cv2
                import numpy as np
            except ImportError:
                return {"error": "Dependency 'opencv-python' (cv2) is missing on this worker node."}

            b64_data = input_data.get('image_data') or input_data.get('data')
            if not b64_data:
                return {"error": "Missing 'image_data'"}

            if ',' in b64_data: b64_data = b64_data.split(',')[1]

            # Decode to numpy
            nparr = np.frombuffer(base64.b64decode(b64_data), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Detect Faces
            # Use internal Haar Cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            # Logic: If 2+ faces, swap them. If 1 face, just draw a box (Mock Swap).
            if len(faces) >= 2:
                # Swap Face 1 and Face 2
                (x1, y1, w1, h1) = faces[0]
                (x2, y2, w2, h2) = faces[1]

                face1 = img[y1:y1+h1, x1:x1+w1].copy()
                face2 = img[y2:y2+h2, x2:x2+w2].copy()

                # Resize to fit
                face1_resized = cv2.resize(face1, (w2, h2))
                face2_resized = cv2.resize(face2, (w1, h1))

                # Place (naive)
                img[y2:y2+h2, x2:x2+w2] = face1_resized
                img[y1:y1+h1, x1:x1+w1] = face2_resized
                
                msg = "Swapped 2 detected faces."
            elif len(faces) == 1:
                # Mock swap - Invert colors of the face
                (x, y, w, h) = faces[0]
                roi = img[y:y+h, x:x+w]
                img[y:y+h, x:x+w] = cv2.bitwise_not(roi)
                msg = "Only 1 face found. Inverted colors to simulate processing."
            else:
                msg = "No faces found to swap."

            # Encode back
            _, buffer = cv2.imencode('.png', img)
            img_str = base64.b64encode(buffer).decode('utf-8')

            return {
                "status": "success", 
                "message": msg, 
                "data": img_str
            }

        except Exception as e:
            logger.error(f"Plugin face_swap failed: {e}")
            return {"error": str(e)}

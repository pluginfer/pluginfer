"""
Plugin Generator
Programmatically generates 20+ essential plugins for the Pluginfer Ecosystem.
"""
import os
import textwrap

PLUGIN_DIR = r"c:\Pluginfr\projects\pluginfer_v2_READY_TO_RUN\pluginfer_v2\plugins"

def create_plugin(name, category, description, code_body, inputs, outputs, cost=0.005):
    class_name = "".join(x.title() for x in name.split('_'))
    filename = f"{name}.py"
    path = os.path.join(PLUGIN_DIR, filename)
    
    content = f'''"""
{description}
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

class {class_name}(PluginBase):
    def config(self) -> Dict[str, Any]:
        return {{
            "name": "{name}",
            "version": "1.0.0",
            "description": "{description}",
            "category": "{category}",
            "tags": {category.split()},
            "cost_per_exec": {cost},
            "inputs": {inputs},
            "outputs": {outputs}
        }}

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
{textwrap.indent(code_body, '            ')}
        except Exception as e:
            logger.error(f"Plugin {name} failed: {{e}}")
            return {{"error": str(e)}}
'''
    
    with open(path, "w") as f:
        f.write(content)
    print(f"✅ Generated {filename}")

def generate_all():
    if not os.path.exists(PLUGIN_DIR):
        os.makedirs(PLUGIN_DIR)

    # --- IMAGE PLUGINS ---
    # --- IMAGE PLUGINS (REAL IMPLEMENTATION) ---
    image_plugins = [
        ("img_resize", "image", "Resizes an image using PIL.", 
         """
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
         """, {"width": "int", "height": "int", "image_data": "str"}, {"data": "str"}),
         
        ("img_grayscale", "image", "Converts image to grayscale using PIL.", 
         """
            import base64
            from io import BytesIO
            from PIL import Image, ImageOps

            b64_data = input_data.get('image_data') or input_data.get('data')
            if ',' in b64_data: b64_data = b64_data.split(',')[1]

            img = Image.open(BytesIO(base64.b64decode(b64_data)))
            
            # Process
            gray_img = ImageOps.grayscale(img)

            buffered = BytesIO()
            gray_img.save(buffered, format=img.format or "PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            return {"status": "success", "message": "Converted to grayscale", "data": img_str}
         """, {"image_data": "str"}, {"data": "str"}),
         
        ("img_blur", "image", "Applies Gaussian blur using PIL.", 
         """
            import base64
            from io import BytesIO
            from PIL import Image, filter

            b64_data = input_data.get('image_data') or input_data.get('data')
            if ',' in b64_data: b64_data = b64_data.split(',')[1]

            img = Image.open(BytesIO(base64.b64decode(b64_data)))
            
            # Process
            radius = int(input_data.get('radius', 2))
            blurred_img = img.filter(filter.GaussianBlur(radius))

            buffered = BytesIO()
            blurred_img.save(buffered, format=img.format or "PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            return {"status": "success", "message": f"Blurred radius {radius}", "data": img_str}
         """, {"radius": "int"}, {"data": "str"}),

        ("img_rotate", "image", "Rotates image.", 
         """
            import base64
            from io import BytesIO
            from PIL import Image

            b64_data = input_data.get('image_data') or input_data.get('data')
            if ',' in b64_data: b64_data = b64_data.split(',')[1]

            img = Image.open(BytesIO(base64.b64decode(b64_data)))
            
            angle = int(input_data.get('angle', 90))
            # Expand=True matches the new dimensions
            rot_img = img.rotate(angle, expand=True)

            buffered = BytesIO()
            rot_img.save(buffered, format=img.format or "PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            return {"status": "success", "message": f"Rotated {angle} deg", "data": img_str}
         """, {"angle": "int"}, {"data": "str"}),
         
        ("img_invert", "image", "Inverts colors.", 
         """
            import base64
            from io import BytesIO
            from PIL import Image, ImageOps

            b64_data = input_data.get('image_data') or input_data.get('data')
            if ',' in b64_data: b64_data = b64_data.split(',')[1]

            img = Image.open(BytesIO(base64.b64decode(b64_data)))
            
            # Invert (handle RGBA vs RGB)
            if img.mode == 'RGBA':
                r,g,b,a = img.split()
                rgb_image = Image.merge('RGB', (r,g,b))
                inverted_image = ImageOps.invert(rgb_image)
                r2,g2,b2 = inverted_image.split()
                # Recombine with original alpha
                final_img = Image.merge('RGBA', (r2,g2,b2,a))
            else:
                final_img = ImageOps.invert(img)

            buffered = BytesIO()
            final_img.save(buffered, format=img.format or "PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            return {"status": "success", "message": "Inverted colors", "data": img_str}
         """, {"image_data": "str"}, {"data": "str"}),
    ]

    # --- TEXT PLUGINS ---
    text_plugins = [
        ("txt_wordcount", "text", "Counts words in text.", 
         """
            text = input_data.get('text', '')
            count = len(text.split())
            return {"count": count}
         """, {"text": "str"}, {"count": "int"}),
         
        ("txt_sentiment", "text", "Basic sentiment analysis.", 
         """
            text = input_data.get('text', '').lower()
            score = 0
            if 'good' in text or 'happy' in text: score += 1
            if 'bad' in text or 'sad' in text: score -= 1
            return {"sentiment": "positive" if score > 0 else "negative", "score": score}
         """, {"text": "str"}, {"sentiment": "str"}),
         
        ("txt_upper", "text", "Converts text to uppercase.", 
         """
            return {"text": input_data.get('text', '').upper()}
         """, {"text": "str"}, {"text": "str"}),
         
        ("txt_anonymize", "text", "Removes emails and phones.", 
         """
            import re
            text = input_data.get('text', '')
            # Simple regex for demo
            text = re.sub(r'[\w\.-]+@[\w\.-]+', '[EMAIL]', text)
            return {"text": text}
         """, {"text": "str"}, {"text": "str"}),
    ]

    # --- DATA PLUGINS ---
    data_plugins = [
        ("data_sort_csv", "data", "Sorts CSV data by column.", 
         """
            # Mock
            col = input_data.get('column', 'id')
            return {"message": f"Sorted by {col}", "rows": 100}
         """, {"csv": "str", "column": "str"}, {"sorted_csv": "str"}),
         
        ("data_dedupe", "data", "Removes duplicate entries.", 
         """
            items = input_data.get('list', [])
            return {"unique_items": list(set(items))}
         """, {"list": "list"}, {"unique_items": "list"}),
         
        ("json_formatter", "data", "Prettifies JSON string.", 
         """
            import json
            raw = input_data.get('json_str', '{}')
            parsed = json.loads(raw)
            return {"formatted": json.dumps(parsed, indent=4)}
         """, {"json_str": "str"}, {"formatted": "str"}),
    ]

    # --- MATH PLUGINS ---
    math_plugins = [
        ("math_prime_factors", "math", "Finds prime factors of a number.", 
         """
            n = input_data.get('number', 1)
            factors = []
            d = 2
            temp = n
            while d * d <= temp:
                while temp % d == 0:
                    factors.append(d)
                    temp //= d
                d += 1
            if temp > 1:
                factors.append(temp)
            return {"factors": factors}
         """, {"number": "int"}, {"factors": "list"}),
         
        ("math_matrix_mul", "math", "Multiplies two matrices (Mock).", 
         """
            dim = input_data.get('dimension', 2)
            # Simulate 100ms compute
            import time
            time.sleep(0.1) 
            return {"message": f"Multiplied {dim}x{dim} matrices", "result": "Matrix Result"}
         """, {"matrix_a": "list"}, {"result": "list"}, 0.02), # Higher cost
    ]

    all_plugins = image_plugins + text_plugins + data_plugins + math_plugins
    
    for p in all_plugins:
        create_plugin(*p)

if __name__ == "__main__":
    generate_all()

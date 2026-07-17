"""
PDF Compression Plugin.
Reduces file size of PDF documents.
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

class PdfCompress(PluginBase):
    def config(self) -> Dict[str, Any]:
        return {
            "name": "pdf_compress",
            "version": "1.0.0",
            "description": "Compresses PDF files.",
            "category": "document",
            "tags": ['pdf', 'compress'],
            "cost_per_exec": 0.01,
            "inputs": {'file_data': 'str'}, # Base64 PDF
            "outputs": {'data': 'str'}      # Base64 PDF
        }

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            try:
                from pypdf import PdfReader, PdfWriter
            except ImportError:
                return {"error": "Dependency 'pypdf' is missing on this worker node."}

            b64_data = input_data.get('file_data') or input_data.get('data')
            if not b64_data:
                return {"error": "Missing 'file_data'"}
            
            if ',' in b64_data:
                b64_data = b64_data.split(',')[1]

            # Decode
            pdf_bytes = base64.b64decode(b64_data)
            input_stream = BytesIO(pdf_bytes)
            
            # Compress
            reader = PdfReader(input_stream)
            writer = PdfWriter()
            
            for page in reader.pages:
                page.compress_content_streams()  # This provides some compression
                writer.add_page(page)
            
            # Optimize metadata/images could be added here
            
            output_stream = BytesIO()
            writer.write(output_stream)
            compressed_bytes = output_stream.getvalue()
            
            # Encode
            out_b64 = base64.b64encode(compressed_bytes).decode('utf-8')
            
            ratio = 100 * (1 - len(compressed_bytes) / len(pdf_bytes))

            return {
                "status": "success", 
                "message": f"PDF Compressed (Reduced by {ratio:.1f}%)", 
                "data": out_b64
            }

        except Exception as e:
            logger.error(f"Plugin pdf_compress failed: {e}")
            return {"error": str(e)}

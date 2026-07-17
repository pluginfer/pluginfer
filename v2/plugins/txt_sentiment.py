"""
Basic sentiment analysis.
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

class TxtSentiment(PluginBase):
    def config(self) -> Dict[str, Any]:
        return {
            "name": "txt_sentiment",
            "version": "1.0.0",
            "description": "Basic sentiment analysis.",
            "category": "text",
            "tags": ['text'],
            "cost_per_exec": 0.005,
            "inputs": {'text': 'str'},
            "outputs": {'sentiment': 'str'}
        }

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:

                        text = input_data.get('text', '').lower()
                        score = 0
                        if 'good' in text or 'happy' in text: score += 1
                        if 'bad' in text or 'sad' in text: score -= 1
                        return {"sentiment": "positive" if score > 0 else "negative", "score": score}
         
        except Exception as e:
            logger.error(f"Plugin txt_sentiment failed: {e}")
            return {"error": str(e)}

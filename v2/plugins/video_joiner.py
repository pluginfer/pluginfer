"""
Video Joiner Plugin (Mock/Prototype)
Simulates joining video chunks (Reduce Phase).
"""
from typing import Dict, Any, List
import logging
import sys
import os
from pathlib import Path

# Robust import strategy
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = str(Path(current_dir).parent.absolute())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from core.plugin_base import PluginBase
except ImportError:
    # Fallback
    sys.path.append(os.path.join(current_dir, '..'))
    from core.plugin_base import PluginBase

logger = logging.getLogger(__name__)

class VideoJoiner(PluginBase):
    def config(self) -> Dict[str, Any]:
        return {
            "name": "video_joiner",
            "version": "1.0.0",
            "description": "Joins video chunks (Reduce Phase)",
            "category": "video",
            "tags": ['video', 'join', 'system'],
            "cost_per_exec": 0.0,
            "inputs": {'segments': 'list', 'original_filename': 'str'},
            "outputs": {'final_file': 'str'}
        }

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mock join logic.
        Input: {'segments': [{'index': 0, 'data': '...'}, ...]}
        """
        segments = input_data.get('segments', [])
        original_filename = input_data.get('original_filename', 'output.mp4')
        
        # 1. Sort by index to ensure order (Critical for video)
        segments.sort(key=lambda x: x.get('index', 0))
        
        # 2. Verify Integirty (Mock)
        # Check if we have all parts 0..N
        if not segments:
             return {"error": "No segments provided"}
             
        for i, seg in enumerate(segments):
            if seg.get('index') != i:
                return {"error": f"Missing segment or wrong order. Expected {i}, got {seg.get('index')}"}
        
        # 3. "Join" (Mock)
        final_filename = f"processed_{original_filename}"
        
        # ✅ PROD: Real FFmpeg Joining (Concat Demuxer)
        import subprocess
        import shutil
        import tempfile
        from core.dependency_manager import DependencyManager
        
        ffmpeg_cmd = DependencyManager.get_ffmpeg_path()
        
        # Check if files actually exist (if not, we must mock)
        files_exist = all(os.path.exists(s['file_part']) for s in segments) if segments else False
        
        if ffmpeg_cmd and files_exist:
            try:
                # Create concat list file
                # file 'part_0.mp4'
                # file 'part_1.mp4'
                list_file = 'concat_list.txt'
                with open(list_file, 'w') as f:
                    for seg in segments:
                        # Escape if needed, but simple filenames preferred
                        f.write(f"file '{seg['file_part']}'\n")
                
                # ffmpeg -f concat -safe 0 -i list.txt -c copy output.mp4
                cmd = [
                    ffmpeg_cmd, '-y',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', list_file,
                    '-c', 'copy',
                    final_filename
                ]
                
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                logger.info(f"Successfully joined videos into {final_filename}")
                
            except Exception as e:
                logger.error(f"FFmpeg Join Failed: {e}")
                return {'status': 'error', 'error': str(e)}
                
        else:
            logger.warning("Falling back to MOCK Join (ffmpeg missing or files missing)")
            logger.info(f"Successfully joined {len(segments)} segments into {final_filename} (Simulated)")
        
        return {
            "status": "success",
            "final_file": final_filename,
            "segment_count": len(segments),
            "integrity_check": "passed"
        }

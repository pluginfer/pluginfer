"""
Video Splitter Plugin (Mock/Prototype)
Simulates splitting a large video file into smaller chunks for distributed processing.
"""
from typing import Dict, Any, List
import logging
import math
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

class VideoSplitter(PluginBase):
    def config(self) -> Dict[str, Any]:
        return {
            "name": "video_splitter",
            "version": "1.0.0",
            "description": "Splits video into chunks (Map Phase)",
            "category": "video",
            "tags": ['video', 'split', 'system'],
            "cost_per_exec": 0.0,
            "inputs": {'filename': 'str', 'duration': 'int', 'chunks': 'int'},
            "outputs": {'segments': 'list'}
        }

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mock split logic.
        Input: {'filename': 'movie.mp4', 'duration': 100, 'chunks': 4}
        """
        filename = input_data.get('filename', 'video.mp4')
        duration = input_data.get('duration', 60) # seconds
        
        # ✅ PRIVACY: Enforce small chunks (The Shredder)
        # Max chunk size = 2 seconds to ensure context-free obfuscation
        MAX_CHUNK_DURATION = 2
        min_chunks = math.ceil(duration / MAX_CHUNK_DURATION)
        
        requested_chunks = input_data.get('chunks', 5)
        num_chunks = max(requested_chunks, min_chunks)
        
        chunk_size = duration / num_chunks # Float division for precision
        segments = []
        
        logger.info(f"Splitting {filename} into {num_chunks} chunks (Max {MAX_CHUNK_DURATION}s) for obfuscation.")
        
        # ✅ PROD: Real FFmpeg Splitting
        import subprocess
        import shutil
        from core.dependency_manager import DependencyManager
        
        ffmpeg_cmd = DependencyManager.get_ffmpeg_path()
        use_mock = False
        
        if not ffmpeg_cmd:
            logger.warning("ffmpeg bundled/system binary not found. Falling back to MOCK logic.")
            use_mock = True
        elif not os.path.exists(filename) and filename.endswith('.mp4'):
             logger.warning(f"File {filename} not found. Falling back to MOCK logic.")
             use_mock = True

        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, duration)
            part_filename = f"{os.path.splitext(filename)[0]}_part_{i}.mp4"
            
            if not use_mock:
                try:
                    # ffmpeg -i input.mp4 -ss {start} -t {chunk_duration} -c copy part_{i}.mp4
                    # -y to overwrite
                    cmd = [
                        ffmpeg_cmd, '-y',
                        '-i', filename,
                        '-ss', str(start),
                        '-t', str(chunk_size),
                        '-c', 'copy', # Fast copy (no recode)
                        part_filename
                    ]
                    
                    # Run FFmpeg
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    logger.info(f"Generated {part_filename}")
                    
                except subprocess.CalledProcessError as e:
                    logger.error(f"FFmpeg failed for {part_filename}: {e}")
                    return {'status': 'error', 'error': str(e)}
            
            segments.append({
                'index': i,
                'file_part': part_filename,
                'start_time': start,
                'end_time': end,
                'status': 'ready_for_processing'
            })
            
        logger.info(f"Split {filename} ({duration}s) into {len(segments)} segments.")
        return {"segments": segments}

"""
Setup Dependencies
Automatically downloads and installs FFmpeg for Windows.
"""
import os
import sys
import urllib.request
import zipfile
import shutil
from pathlib import Path

# Config
FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
PROJECT_ROOT = Path(__file__).parent.absolute()
BIN_DIR = PROJECT_ROOT / "bin"
TEMP_ZIP = PROJECT_ROOT / "ffmpeg.zip"

def setup():
    print(f"[*] Setting up dependencies in {PROJECT_ROOT}...")
    
    # 1. Create bin directory
    if not BIN_DIR.exists():
        os.makedirs(BIN_DIR)
        print(f"    Created {BIN_DIR}")
    
    # Check if already installed
    if (BIN_DIR / "ffmpeg.exe").exists():
        print("    ✅ FFmpeg already installed in bin/")
        return

    # 2. Download
    print(f"[*] Downloading FFmpeg from {FFMPEG_URL}...")
    print("    This may take a minute (approx 80MB)...")
    try:
        urllib.request.urlretrieve(FFMPEG_URL, TEMP_ZIP)
        print("    ✅ Download complete.")
    except Exception as e:
        print(f"    ❌ Download Failed: {e}")
        return

    # 3. Extract
    print("[*] Extracting ffmpeg.exe...")
    try:
        with zipfile.ZipFile(TEMP_ZIP, 'r') as zip_ref:
            # Find the path to ffmpeg.exe inside the zip (it's usually in a subfolder)
            ffmpeg_path = None
            for name in zip_ref.namelist():
                if name.endswith("bin/ffmpeg.exe"):
                    ffmpeg_path = name
                    break
            
            if not ffmpeg_path:
                print("    ❌ Could not find ffmpeg.exe in zip archive")
                return
            
            # Extract just that file
            source = zip_ref.open(ffmpeg_path)
            target = open(BIN_DIR / "ffmpeg.exe", "wb")
            with source, target:
                shutil.copyfileobj(source, target)
                
            print(f"    ✅ Extracted to {BIN_DIR / 'ffmpeg.exe'}")
            
    except Exception as e:
         print(f"    ❌ Extraction Failed: {e}")
    finally:
        # 4. Cleanup
        if TEMP_ZIP.exists():
            os.remove(TEMP_ZIP)
            print("    Cleaned up temp files.")

if __name__ == "__main__":
    setup()

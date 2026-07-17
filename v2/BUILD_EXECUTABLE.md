# Building Executable Guide

## 📦 How to Create Executables from Pluginfer

This guide shows you how to convert Pluginfer Python scripts into standalone executables (.exe for Windows, binary for Linux/Mac).

---

## Why Build an Executable?

**Benefits:**
- ✅ No Python installation needed
- ✅ Single file to distribute
- ✅ Easier for non-technical users
- ✅ Includes all dependencies
- ✅ Professional distribution

**Drawbacks:**
- ❌ Larger file size (~50-100 MB)
- ❌ Platform-specific (Windows exe won't run on Linux)
- ❌ Slower startup (unpacks dependencies)
- ❌ Harder to debug

---

## Quick Start

### Method 1: Use Build Scripts (Easiest)

#### On Windows:
1. Extract the ZIP file
2. Double-click `build_windows.bat`
3. Wait for build to complete
4. Find executable in `dist/pluginfer.exe`

#### On Linux/Mac:
1. Extract the archive
2. Open terminal in the folder
3. Run: `./build_linux.sh`
4. Find executable in `dist/pluginfer`

### Method 2: Use Launcher (No Build Needed)

If you just want an easy way to run Pluginfer:

#### Windows:
```cmd
python launcher.py
```

#### Linux/Mac:
```bash
python3 launcher.py
```

This gives you an interactive menu without needing to build an executable!

---

## Detailed Instructions

### Windows Executable (.exe)

#### Prerequisites:
1. **Python 3.8+** installed
   - Download from: https://python.org
   - Make sure "Add to PATH" is checked during installation

2. **PyInstaller** (auto-installed by script)

#### Step-by-Step:

**Option A: Automated (Recommended)**
```cmd
# Run the build script
build_windows.bat

# Executable will be in: dist\pluginfer.exe
```

**Option B: Manual**
```cmd
# 1. Install PyInstaller
pip install pyinstaller

# 2. Build the executable
pyinstaller --onefile --name="pluginfer" pluginfer.py

# 3. Executable in: dist\pluginfer.exe
```

#### Advanced Build (with GUI):
```cmd
pyinstaller --onefile ^
    --name="pluginfer" ^
    --noconsole ^
    --icon=icon.ico ^
    --add-data="plugins;plugins" ^
    pluginfer.py
```

---

### Linux/Mac Executable

#### Prerequisites:
1. **Python 3.8+** installed
   ```bash
   # Check version
   python3 --version
   ```

2. **PyInstaller** (auto-installed by script)

#### Step-by-Step:

**Option A: Automated (Recommended)**
```bash
# Make script executable
chmod +x build_linux.sh

# Run build script
./build_linux.sh

# Executable will be in: dist/pluginfer
```

**Option B: Manual**
```bash
# 1. Install PyInstaller
pip3 install pyinstaller

# 2. Build the executable
pyinstaller --onefile --name="pluginfer" pluginfer.py

# 3. Executable in: dist/pluginfer

# 4. Make it executable
chmod +x dist/pluginfer
```

---

## What Gets Included?

When you build an executable, PyInstaller includes:

✅ **Python interpreter** (~30 MB)
✅ **All Python dependencies** (torch, numpy, etc.)
✅ **Pluginfer core modules**
✅ **Plugin files**
✅ **Configuration files**

**NOT Included:**
❌ PyTorch models (if you have large models, include them separately)
❌ User data (license files, logs, etc.)
❌ Documentation (add with `--add-data`)

---

## Build Options

### Single File vs. Directory

**Single File** (--onefile)
- One .exe file
- Slower startup (unpacks to temp)
- Easier to distribute
- Recommended for most users

```bash
pyinstaller --onefile pluginfer.py
```

**Directory** (default)
- Folder with many files
- Faster startup
- Harder to distribute
- Better for development

```bash
pyinstaller pluginfer.py
```

### Including Extra Files

Add plugins:
```bash
pyinstaller --onefile --add-data="plugins:plugins" pluginfer.py
```

Add documentation:
```bash
pyinstaller --onefile ^
    --add-data="plugins:plugins" ^
    --add-data="README.md:." ^
    pluginfer.py
```

### Hidden Imports

If you get import errors, add hidden imports:
```bash
pyinstaller --onefile ^
    --hidden-import=core.plugin_base ^
    --hidden-import=torch ^
    pluginfer.py
```

### Icon (Windows)

Add custom icon:
```bash
pyinstaller --onefile --icon=icon.ico pluginfer.py
```

---

## Troubleshooting

### Error: "PyInstaller not found"

**Solution:**
```bash
pip install pyinstaller
# or
pip3 install pyinstaller --user
```

### Error: "Module not found" when running exe

**Cause:** PyInstaller didn't detect all imports

**Solution:** Add hidden imports
```bash
pyinstaller --onefile ^
    --hidden-import=missing_module ^
    pluginfer.py
```

### Error: "Failed to execute script"

**Cause:** Missing data files

**Solution:** Include with --add-data
```bash
pyinstaller --onefile ^
    --add-data="plugins:plugins" ^
    pluginfer.py
```

### Executable is too large

**Solutions:**
1. Use `--exclude-module` to exclude unused modules
2. Use UPX compression:
   ```bash
   pyinstaller --onefile --upx-dir=/path/to/upx pluginfer.py
   ```
3. Remove unnecessary dependencies from requirements.txt

### Antivirus flags the executable

**Cause:** PyInstaller executables sometimes trigger false positives

**Solutions:**
1. Add exception in antivirus
2. Sign the executable (requires code signing certificate)
3. Submit to antivirus vendors as false positive

---

## Distribution

### Windows

**Option 1: Just the EXE**
1. Build with `build_windows.bat`
2. Distribute `dist/pluginfer.exe`
3. User needs `plugins/` folder in same directory

**Option 2: Complete Package**
The build script creates `pluginfer_standalone/` with:
- pluginfer.exe
- plugins/ folder
- README.md
- Documentation

Zip this folder and distribute.

### Linux/Mac

**Option 1: Binary**
1. Build with `build_linux.sh`
2. Distribute `dist/pluginfer`
3. User runs: `./pluginfer --test`

**Option 2: Complete Package**
The build script creates `pluginfer_standalone/` 

Create archive:
```bash
tar -czf pluginfer_linux.tar.gz pluginfer_standalone/
```

---

## Alternative: Docker Container

Instead of executable, you can distribute as Docker container:

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

CMD ["python", "pluginfer.py", "--test"]
```

Build:
```bash
docker build -t pluginfer .
docker run pluginfer
```

---

## Alternative: Portable Python

Create a portable Python distribution:

**Windows:**
1. Download Python embeddable package
2. Extract to folder
3. Add Pluginfer
4. Create run.bat:
   ```cmd
   @echo off
   python\python.exe pluginfer.py --test
   ```

**Linux:**
Use Python virtual environment:
```bash
python3 -m venv portable_env
source portable_env/bin/activate
pip install -r requirements.txt
# Distribute the whole portable_env folder
```

---

## Performance Tips

### Reduce Startup Time

1. **Use --onedir instead of --onefile**
   - Faster startup
   - Slightly larger distribution

2. **Exclude unused modules**
   ```bash
   pyinstaller --onefile ^
       --exclude-module matplotlib ^
       --exclude-module pandas ^
       pluginfer.py
   ```

3. **Use lazy imports in code**
   - Import only when needed
   - Not at module level

### Reduce Size

1. **Strip binaries** (Linux)
   ```bash
   strip dist/pluginfer
   ```

2. **Use UPX compression**
   ```bash
   pyinstaller --onefile --upx-dir=/usr/bin pluginfer.py
   ```

3. **Exclude test modules**
   ```bash
   pyinstaller --onefile ^
       --exclude-module tests ^
       pluginfer.py
   ```

---

## Best Practices

### 1. Version Your Builds

```bash
pyinstaller --onefile ^
    --name="pluginfer_v1.0.0" ^
    pluginfer.py
```

### 2. Include Version in Code

```python
# pluginfer.py
__version__ = "1.0.0"

if __name__ == "__main__":
    print(f"Pluginfer v{__version__}")
```

### 3. Test on Clean Machine

- Test executable on computer without Python
- Verify all dependencies included
- Check file paths work

### 4. Provide Multiple Formats

Distribute:
- Source code (ZIP)
- Windows executable
- Linux binary
- Mac binary
- Docker image

### 5. Document Requirements

Create README for executable:
- System requirements
- How to run
- Where to find plugins
- Troubleshooting

---

## Example: Complete Build Process

### Windows:
```cmd
# 1. Clean previous builds
rmdir /s /q build dist

# 2. Build executable
pyinstaller --onefile ^
    --name="pluginfer" ^
    --add-data="plugins;plugins" ^
    --add-data="core;core" ^
    --hidden-import=core ^
    pluginfer.py

# 3. Test
dist\pluginfer.exe --test

# 4. Create package
mkdir release
copy dist\pluginfer.exe release\
xcopy /E /I plugins release\plugins
copy README.md release\

# 5. Zip
powershell Compress-Archive -Path release -DestinationPath pluginfer_windows.zip
```

### Linux:
```bash
# 1. Clean previous builds
rm -rf build dist

# 2. Build executable
pyinstaller --onefile \
    --name="pluginfer" \
    --add-data="plugins:plugins" \
    --add-data="core:core" \
    --hidden-import=core \
    pluginfer.py

# 3. Test
./dist/pluginfer --test

# 4. Create package
mkdir release
cp dist/pluginfer release/
cp -r plugins release/
cp README.md release/

# 5. Archive
tar -czf pluginfer_linux.tar.gz release/
```

---

## Summary

**Easiest Method:**
1. Run `launcher.py` - No build needed!

**For Distribution:**
1. Use provided build scripts
2. Test on clean machine
3. Distribute with plugins folder
4. Include README

**For Development:**
1. Use Python directly
2. Build executable only for releases
3. Test thoroughly before distributing

---

## Support

**Build Issues:**
- Check PyInstaller version: `pyinstaller --version`
- Run with verbose: `pyinstaller --onefile --log-level DEBUG pluginfer.py`
- Check build.log in build folder

**Runtime Issues:**
- Run from terminal to see errors
- Check that plugins/ folder is present
- Verify all dependencies included

**Questions:**
- See PyInstaller docs: https://pyinstaller.org
- Pluginfer issues: GitHub Issues

---

**🚀 Ready to build? Run the build script for your platform!**

Windows: `build_windows.bat`
Linux/Mac: `./build_linux.sh`

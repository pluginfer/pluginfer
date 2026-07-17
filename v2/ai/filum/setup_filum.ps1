# Filum setup -- detects your CUDA / PyTorch situation and installs the right wheels.
# Run from PowerShell:    .\v2\ai\filum\setup_filum.ps1

$ErrorActionPreference = "Stop"
Write-Host "==== Filum environment setup ====" -ForegroundColor Cyan

# Detect Python.
$pythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $pythonExe) {
    Write-Host "ERROR: python not on PATH. Install Python 3.10+ first."
    exit 1
}
Write-Host "python: $pythonExe"

# Detect CUDA via nvidia-smi.
$cudaVersion = $null
try {
    $smi = & nvidia-smi 2>$null
    if ($smi) {
        $line = $smi | Select-String -Pattern "CUDA Version:" | Select-Object -First 1
        if ($line) {
            $cudaVersion = ($line -replace ".*CUDA Version:\s*([\d.]+).*", '$1').Trim()
            Write-Host "CUDA driver version: $cudaVersion"
        }
    }
} catch {
    Write-Host "no nvidia-smi -- assuming CPU-only"
}

# Pick the right torch wheel.
if ($cudaVersion) {
    $major = [int]($cudaVersion -split '\.')[0]
    $minor = [int]($cudaVersion -split '\.')[1]
    if (($major -gt 11) -or (($major -eq 11) -and ($minor -ge 8))) {
        Write-Host "CUDA >= 11.8 -- installing torch with CUDA 11.8" -ForegroundColor Green
        & python -m pip install torch --index-url "https://download.pytorch.org/whl/cu118"
    } elseif ($major -eq 11) {
        Write-Host "CUDA 11.x but < 11.8" -ForegroundColor Yellow
        Write-Host "  Your GTX 1650 driver supports CUDA $cudaVersion but recent PyTorch builds"
        Write-Host "  require >= 11.8. Two options:"
        Write-Host "    (A) UPDATE the GeForce driver -- recommended. Get the latest"
        Write-Host "        Studio or Game Ready driver from nvidia.com/Download/index.aspx"
        Write-Host "        (your GPU model: GeForce GTX 1650). New drivers expose CUDA 12.x"
        Write-Host "        and PyTorch 2.x will use them. After update, re-run this script."
        Write-Host "    (B) PYTORCH 1.x with CUDA 11.3 (older API, slower):"
        Write-Host "        python -m pip install torch==1.12.1+cu113 --index-url https://download.pytorch.org/whl/cu113"
        Write-Host ""
        Write-Host "Defaulting to CPU torch for now so you can run the demo."
        & python -m pip install torch
    } else {
        Write-Host "CUDA < 11 -- using CPU torch. Update the driver to enable GPU."
        & python -m pip install torch
    }
} else {
    Write-Host "no CUDA detected -- installing CPU torch"
    & python -m pip install torch
}

# Install runtime deps Filum actually uses.
Write-Host ""
Write-Host "==== installing Filum runtime deps ====" -ForegroundColor Cyan
& python -m pip install fastapi sse-starlette httpx pydantic cryptography reportlab pytest

# Optional teacher SDKs -- only if you'll do paid distillation.
Write-Host ""
Write-Host "==== teacher SDKs (optional) ====" -ForegroundColor Cyan
Write-Host "Install only the ones you'll actually use:"
Write-Host "  Anthropic:    pip install anthropic"
Write-Host "  Google:       pip install google-generativeai"
Write-Host "  OpenAI:       pip install openai"
Write-Host ""

# Verify.
Write-Host "==== verifying ====" -ForegroundColor Cyan
& python -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"
& python -c "from ai.filum.config import FilumConfig; c = FilumConfig(); p = c.estimate_param_count(); print('Filum target params: %.1fM' % p['total_M'])"
Write-Host ""
Write-Host "==== done ====" -ForegroundColor Green
Write-Host "Quick smoke test:"
Write-Host "    python -m ai.filum train --demo"
Write-Host "    (loss must decrease over 100 steps; ~3s on CPU)"

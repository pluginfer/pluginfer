# Pluginfer Windows installer — bootstraps a fresh Windows 10/11 box
# into a working Pluginfer node serving real model inference via
# Ollama for Windows.
#
# Run as Administrator from PowerShell:
#   iwr -useb https://get.pluginfer.network/install.ps1 | iex
#
# Or with options:
#   $env:PLUGINFER_SEED_HOST = "seed.pluginfer.network"
#   $env:PLUGINFER_MODEL = "qwen2.5:1.5b"
#   iwr -useb https://get.pluginfer.network/install.ps1 | iex
#
# The script:
#   1. Checks for / installs winget (Win 10 1809+) or downloads installers
#   2. Installs Python 3.12, Git, Ollama
#   3. Pulls the model via Ollama
#   4. Clones (or downloads) Pluginfer
#   5. Sets up venv + deps
#   6. Generates a per-node wallet
#   7. Writes config + creates a Windows Scheduled Task for auto-start
#   8. Verifies the real adapter resolved (NOT echo)

[CmdletBinding()]
param(
    [string]$SeedHost = $(if ($env:PLUGINFER_SEED_HOST) { $env:PLUGINFER_SEED_HOST } else { "127.0.0.1" }),
    [int]   $SeedPort = $(if ($env:PLUGINFER_SEED_PORT) { [int]$env:PLUGINFER_SEED_PORT } else { 9000 }),
    [int]   $NodePort = $(if ($env:PLUGINFER_NODE_PORT) { [int]$env:PLUGINFER_NODE_PORT } else { 8101 }),
    [int]   $OllamaPort = $(if ($env:PLUGINFER_OLLAMA_PORT) { [int]$env:PLUGINFER_OLLAMA_PORT } else { 11435 }),
    [string]$Model = $(if ($env:PLUGINFER_MODEL) { $env:PLUGINFER_MODEL } else { "qwen2.5:1.5b" }),
    [string]$PluginferVersion = $(if ($env:PLUGINFER_VERSION) { $env:PLUGINFER_VERSION } else { "main" }),
    [string]$PluginferRepo = "https://github.com/pluginfer/pluginfer.git",
    [string]$ReleaseUrl = "",
    [string]$InstallDir = "$env:USERPROFILE\.pluginfer",
    [switch]$NoForeground
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Msg)
    Write-Host "[pluginfer] $Msg" -ForegroundColor Cyan
}

function Test-Admin {
    $u = [Security.Principal.WindowsIdentity]::GetCurrent()
    $p = New-Object Security.Principal.WindowsPrincipal($u)
    return $p.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-Admin)) {
    Write-Warning "Not running as Administrator. Some package installs may fail."
}

# --- Step 1: winget / installers ------------------------------------
Write-Step "step 1/8: system dependencies"

function Install-IfMissing {
    param(
        [string]$Cmd,
        [string]$WingetId
    )
    if (Get-Command $Cmd -ErrorAction SilentlyContinue) {
        return
    }
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Step "installing $WingetId via winget"
        winget install --id $WingetId --silent --accept-package-agreements --accept-source-agreements
        $env:PATH = [Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + `
                    [Environment]::GetEnvironmentVariable("PATH", "User")
    } else {
        Write-Error "winget not available and $Cmd not installed. " +
                    "Install $Cmd manually or update to Windows 10 1809+."
    }
}

Install-IfMissing "python" "Python.Python.3.12"
Install-IfMissing "git"    "Git.Git"

# jq isn't bundled with Windows — use Python to parse JSON instead.

# --- Step 2: Ollama -------------------------------------------------
Write-Step "step 2/8: Ollama"
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install --id Ollama.Ollama --silent --accept-package-agreements --accept-source-agreements
    } else {
        $ollamaUrl = "https://ollama.com/download/OllamaSetup.exe"
        $ollamaTmp = "$env:TEMP\OllamaSetup.exe"
        Invoke-WebRequest -Uri $ollamaUrl -OutFile $ollamaTmp -UseBasicParsing
        Start-Process -FilePath $ollamaTmp -ArgumentList "/S" -Wait
    }
    $env:PATH = [Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + `
                [Environment]::GetEnvironmentVariable("PATH", "User")
}

# Set OLLAMA_HOST so we don't collide with Pluginfer devserver 11434.
[Environment]::SetEnvironmentVariable("OLLAMA_HOST", "0.0.0.0:$OllamaPort", "User")
$env:OLLAMA_HOST = "0.0.0.0:$OllamaPort"

# Restart Ollama if it's running, so it picks up the new host.
Get-Process -Name ollama -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1
Start-Process -FilePath "ollama" -ArgumentList "serve" `
              -RedirectStandardOutput "$env:TEMP\ollama.log" `
              -RedirectStandardError "$env:TEMP\ollama.err" `
              -WindowStyle Hidden

# Wait for Ollama API.
$elapsed = 0
while ($elapsed -lt 30) {
    try {
        $r = Invoke-WebRequest "http://127.0.0.1:$OllamaPort/api/tags" -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) { break }
    } catch { Start-Sleep -Seconds 1; $elapsed++ }
}
if ($elapsed -ge 30) {
    Write-Error "Ollama did not come up on port $OllamaPort within 30s"
}

# --- Step 3: pull the model ----------------------------------------
Write-Step "step 3/8: pull $Model (this can take a few minutes)"
$env:OLLAMA_HOST = "http://127.0.0.1:$OllamaPort"
ollama pull $Model

# --- Step 4: Pluginfer source --------------------------------------
Write-Step "step 4/8: Pluginfer source"
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
if ($ReleaseUrl) {
    $tarball = "$InstallDir\release.tar.gz"
    Invoke-WebRequest -Uri $ReleaseUrl -OutFile $tarball -UseBasicParsing
    tar -xzf $tarball -C $InstallDir --strip-components=1
    $RepoDir = $InstallDir
} else {
    $RepoDir = "$InstallDir\repo"
    if (-not (Test-Path "$RepoDir\.git")) {
        git clone $PluginferRepo $RepoDir
    }
    Push-Location $RepoDir
    git fetch --tags 2>$null
    git checkout $PluginferVersion 2>$null
    Pop-Location
}

# --- Step 5: venv + deps -------------------------------------------
Write-Step "step 5/8: Python virtualenv"
if (-not (Test-Path "$RepoDir\.venv")) {
    python -m venv "$RepoDir\.venv"
}
& "$RepoDir\.venv\Scripts\pip.exe" install --quiet --upgrade pip
& "$RepoDir\.venv\Scripts\pip.exe" install --quiet `
    -r "$RepoDir\v2\api\requirements-devserver.txt" `
    cryptography

# --- Step 6: wallet ------------------------------------------------
Write-Step "step 6/8: wallet"
$DataDir = "$env:USERPROFILE\.pluginfer"
New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
$PassphraseFile = "$DataDir\wallet.passphrase"
if (-not (Test-Path $PassphraseFile)) {
    # 32 random bytes hex-encoded — sufficient entropy for the wallet KDF.
    $bytes = New-Object byte[] 32
    [Security.Cryptography.RNGCryptoServiceProvider]::Create().GetBytes($bytes)
    -join ($bytes | ForEach-Object { "{0:x2}" -f $_ }) | Set-Content -NoNewline $PassphraseFile
}
$Passphrase = Get-Content $PassphraseFile -Raw

# --- Step 7: configure ---------------------------------------------
Write-Step "step 7/8: configure"
$NodeId = "$($env:COMPUTERNAME)-$(Get-Random -Maximum 0xFFFFFFFF | ForEach-Object { '{0:x}' -f $_ })"
$EnvLines = @(
    "PLUGINFER_SEED_HOST=$SeedHost",
    "PLUGINFER_SEED_PORT=$SeedPort",
    "PLUGINFER_NODE_PORT=$NodePort",
    "PLUGINFER_NODE_ID=$NodeId",
    "OLLAMA_HOST=http://127.0.0.1:$OllamaPort",
    "PLUGINFER_ALPHA_MODEL_ID=$Model",
    "PLUGINFER_WALLET_PASSPHRASE=$Passphrase",
    "PLUGINFER_JOBS_DB=$DataDir\jobs.db"
)
$EnvLines | Set-Content "$DataDir\auto_mesh.env"

foreach ($line in $EnvLines) {
    $parts = $line -split "=", 2
    Set-Item -Path "Env:$($parts[0])" -Value $parts[1]
}

# --- Step 8: boot + verify -----------------------------------------
Write-Step "step 8/8: starting auto_mesh"

$walletPath = "$DataDir\auto_mesh_wallet.pem"
$logPath = "$env:TEMP\pluginfer_node.log"

Push-Location "$RepoDir\v2"

$nodeProc = Start-Process -FilePath "$RepoDir\.venv\Scripts\python.exe" `
    -ArgumentList ("-m", "tools.auto_mesh",
                   "--seed-host", $SeedHost,
                   "--seed-port", $SeedPort,
                   "--node-port", $NodePort,
                   "--wallet-path", $walletPath) `
    -RedirectStandardOutput $logPath `
    -RedirectStandardError "$env:TEMP\pluginfer_node.err" `
    -WindowStyle Hidden -PassThru

# Wait for healthz.
$elapsed = 0
$up = $false
while ($elapsed -lt 30) {
    try {
        $r = Invoke-WebRequest "http://localhost:$NodePort/healthz" -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) { $up = $true; break }
    } catch { Start-Sleep -Seconds 1; $elapsed++ }
}
if (-not $up) {
    Stop-Process -Id $nodeProc.Id -Force -ErrorAction SilentlyContinue
    Write-Error "auto_mesh did not come up within 30s. Check $logPath"
}

# Verify real adapter resolved.
$hw = Invoke-WebRequest "http://localhost:$NodePort/v1/hardware" -UseBasicParsing | ConvertFrom-Json
if ($hw.runtime.name -ne "ollama" -or $hw.runtime.is_echo -eq $true) {
    Stop-Process -Id $nodeProc.Id -Force -ErrorAction SilentlyContinue
    Write-Host ""
    Write-Host "[pluginfer] ERROR: real adapter did NOT resolve." -ForegroundColor Red
    Write-Host "  Runtime: $($hw.runtime.name)"
    Write-Host "  is_echo: $($hw.runtime.is_echo)"
    Write-Host "  Tail of node log:"
    Get-Content $logPath -Tail 20
    Write-Host ""
    Write-Host "Fixes:"
    Write-Host "  1. Check Ollama:   Invoke-WebRequest http://127.0.0.1:$OllamaPort/api/tags"
    Write-Host "  2. Pull the model: ollama pull $Model"
    Write-Host "  3. Re-run this installer"
    exit 2
}

Write-Host ""
Write-Host "[pluginfer] OK — node up on http://localhost:$NodePort" -ForegroundColor Green
Write-Host "[pluginfer]      runtime: $($hw.runtime.name) ($Model)"
Write-Host "[pluginfer]      seed:    ${SeedHost}:$SeedPort"
Write-Host ""
Write-Host "Try it:"
Write-Host "  Invoke-WebRequest http://localhost:$NodePort/peers | Select-Object -ExpandProperty Content"
Write-Host '  Invoke-RestMethod -Method Post -ContentType "application/json" `'
Write-Host "      -Uri http://localhost:$NodePort/v1/chat/completions ``"
Write-Host '      -Body (@{messages=@(@{role="user";content="hi"});max_tokens=32;pluginfer_cost_ceiling_usd=0.05} | ConvertTo-Json)'
Write-Host ""
Write-Host "Stop:    Stop-Process -Id $($nodeProc.Id)"
Write-Host "Logs:    Get-Content $logPath -Wait"

if (-not $NoForeground) {
    Write-Host ""
    Write-Host "[pluginfer] supervisor running in foreground (PID $($nodeProc.Id))"
    Write-Host "[pluginfer] press Ctrl+C in this window to stop"
    Wait-Process -Id $nodeProc.Id
}

Pop-Location

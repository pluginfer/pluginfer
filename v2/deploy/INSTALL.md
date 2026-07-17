# Pluginfer — install on any environment

One command. Three operating systems. Real model inference at the end.

## macOS (Intel + Apple Silicon)

```bash
curl -fsSL https://get.pluginfer.network/install.sh | bash
```

If `pluginfer.network` isn't live yet, point at your repo URL directly:
```bash
curl -fsSL https://raw.githubusercontent.com/<your-org>/pluginfer/main/v2/deploy/install.sh \
  | PLUGINFER_REPO=https://github.com/<your-org>/pluginfer.git \
    PLUGINFER_SEED_HOST=<seed-ip> bash
```

The script will:
- Install Homebrew if missing
- Install Python 3.12, git, jq, openssl, Ollama
- Pull `qwen2.5:1.5b` (~1 GB) via Ollama
- Clone Pluginfer + set up venv
- Generate a per-node wallet
- Boot the node + verify the real adapter resolved (NOT echo)

## Linux (Ubuntu, Debian, Fedora, Arch)

Same command:
```bash
curl -fsSL https://get.pluginfer.network/install.sh | bash
```

Auto-detects the distro and uses `apt` / `dnf` / `pacman`. On systemd
boxes, Ollama gets a port-override drop-in unit so it doesn't collide
with Pluginfer's devserver default (11434).

## Windows 10 / 11

In an **Administrator** PowerShell:
```powershell
iwr -useb https://get.pluginfer.network/install.ps1 | iex
```

Uses `winget` if available (Win 10 1809+), falls back to direct installer
downloads. Boots the node + verifies the real adapter — same behaviour
as the Bash script.

## What "no blockers" means

Both installers will **refuse to silently start the echo runner**. If
Ollama can't pull the model, or the model id is wrong, or the runtime
adapter probe fails, you get a clear error message and a non-zero exit.
You will NEVER end up with a node that looks healthy but is serving
canned strings.

The verifier at the end of both scripts hits `/v1/hardware` and
asserts:
```
runtime.name == "ollama"  AND  runtime.is_echo == false
```

If either fails, the script prints the last 20 log lines and the three
most likely fixes.

## Configuring the seed

By default both installers assume `127.0.0.1:9000` — meaning the same
machine also runs the seed. For a 3-box mesh:

1. Bring up a seed first (on a small VPS):
   ```bash
   curl -fsSL https://get.pluginfer.network/install_seed.sh \
       | sudo bash -s -- --port 9000
   ```
   Note the seed's public IP.

2. On each compute node:
   ```bash
   curl -fsSL https://get.pluginfer.network/install.sh \
       | bash -s -- --seed-host <seed-ip> --seed-port 9000
   ```

The full 3-box walkthrough is in [`FIRST_PROOF.md`](FIRST_PROOF.md).

## Running as a long-lived service

### Linux (systemd)
The installer drops `/etc/systemd/system/auto_mesh.service`. Enable + start:
```bash
sudo systemctl enable --now auto_mesh
```

### macOS (launchd)
A launchd plist ships in `installer/launchd/com.pluginfer.node.plist`.
Copy to `~/Library/LaunchAgents/` and `launchctl load` it.

### Windows (Scheduled Task)
Run the supervisor wrapper via a logon-triggered Task:
```powershell
$action = New-ScheduledTaskAction -Execute "$env:USERPROFILE\.pluginfer\repo\.venv\Scripts\python.exe" `
    -Argument "-m tools.run_node" `
    -WorkingDirectory "$env:USERPROFILE\.pluginfer\repo\v2"
$trigger = New-ScheduledTaskTrigger -AtLogOn
Register-ScheduledTask -TaskName "PluginferNode" -Action $action -Trigger $trigger -RunLevel Limited
```

## Customizing the install

| Flag | Env var | Default | What it controls |
|---|---|---|---|
| `--seed-host` | `PLUGINFER_SEED_HOST` | `127.0.0.1` | Where to register |
| `--seed-port` | `PLUGINFER_SEED_PORT` | `9000` | Seed TCP port |
| `--node-port` | `PLUGINFER_NODE_PORT` | `8101` | This node's gateway port |
| `--model` | `PLUGINFER_MODEL` | `qwen2.5:1.5b` | Ollama model tag to pull |
| `--version` | `PLUGINFER_VERSION` | `main` | Git ref to checkout |
| `--release-url` | `PLUGINFER_RELEASE_URL` | (none) | Tarball URL — bypass git clone |

## Private-repo install (before your release strategy makes the code public)

Both installers support `PLUGINFER_RELEASE_URL=<signed-tarball-url>`:

```bash
# Build a release tarball from your private repo:
git archive --format=tar.gz --prefix=pluginfer/ HEAD > pluginfer-v0.1.0.tar.gz
# Upload to S3 / R2 / GitHub release with a signed URL:
aws s3 cp pluginfer-v0.1.0.tar.gz s3://pluginfer-releases/v0.1.0.tar.gz
# Distribute the signed URL to your install nodes:
curl -fsSL https://get.pluginfer.network/install.sh \
    | PLUGINFER_RELEASE_URL=https://<signed-s3-url> bash
```

This way your repo stays private while operators can still install from
a signed release. When you're ready to flip to public , drop
the `--release-url` flag and the installer pulls straight from `git`.

## Removing

```bash
# Linux
sudo systemctl disable --now auto_mesh
rm -rf $HOME/.pluginfer

# macOS
launchctl unload ~/Library/LaunchAgents/com.pluginfer.node.plist
rm -rf $HOME/.pluginfer

# Windows
Unregister-ScheduledTask -TaskName "PluginferNode" -Confirm:$false
Remove-Item -Recurse -Force $env:USERPROFILE\.pluginfer
```

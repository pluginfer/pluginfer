# Installing Pluginfer

> The lightweight, plug-and-play decentralized AI compute mesh.
>
> Earn from your idle GPU. Train AI for free.

---

## TL;DR

* **Windows:** double-click `Pluginfer-Setup.bat` (in this folder).
* **macOS / Linux:** run `bash Pluginfer-Setup.sh`.

That's it. The installer:

1. Detects your hardware (NVIDIA / AMD / Intel / Apple / CPU)
2. Generates your node identity (Ed25519 keypair, stored locally)
3. Configures the optimal tier (light / standard / max)
4. Creates a Start Menu shortcut (Windows) or a desktop entry (Linux)

After setup, double-click **Pluginfer** to open the GUI.

---

## What you need

* Python 3.10+ (auto-detected on PATH; install from python.org if missing)
* About 2 GB free disk space for dependencies + the genesis model
* Internet connection for the first install (deps + Filum-Genesis)

That's all. No admin rights are required for a per-user install.

---

## What the installer does

| Step | Action |
|---|---|
| 1 | Probes for `python` / `py` / `python3` on PATH |
| 2 | Runs `pip install` for `psutil`, `numpy`, `torch` |
| 3 | Calls `python -m ai.filum.auto_setup` — detects GPU, generates keypair |
| 4 | Creates a Start Menu / desktop shortcut |
| 5 | Prints "ready" |

The state directory is created at:

* **Windows:** `%APPDATA%\Pluginfer`
* **macOS:**   `~/Library/Application Support/Pluginfer`
* **Linux:**   `~/.config/pluginfer`

This folder holds:

* `node_key.bin` — your private Ed25519 seed (never share this)
* `node_pub.txt` — your public key (this is your node ID)
* `runtime_config.json` — auto-generated config

---

## What the GUI does

After install, double-click `Pluginfer` (or run `python -m
ai.filum.gui_launcher`). You see:

* A status row showing your detected GPU + tier
* A big green **START CONTRIBUTING** button
* Today's earnings + month-to-date counters
* An **Ask Filum** box — type any question, the AI answers from
  Pluginfer's own knowledge base

Click START CONTRIBUTING. Your idle GPU starts running training jobs
that other people submit. They pay; Pluginfer takes 5%; you earn the
rest. Earnings tick up every couple seconds.

---

## What the GUI doesn't do (yet)

* **Submit a custom training job from a folder** — coming in v0.2.
  For now the CLI is `python -m ai.filum train --adaptive`.
* **Build a real native .exe / .app** — this installer launches via
  Python. v0.2 ships PyInstaller artifacts so the user doesn't even
  need Python installed. See `installer/build_windows.spec`.

---

## Uninstall

* **Windows:** delete the `Pluginfer.lnk` shortcut + the
  `%APPDATA%\Pluginfer` folder. Optionally `pip uninstall torch`.
* **macOS:**   delete `~/Library/Application Support/Pluginfer`.
* **Linux:**   delete `~/.config/pluginfer` +
  `~/.local/share/applications/Pluginfer.desktop`.

Pluginfer never installs services without consent and creates no
files outside its state directory.

---

## Pre-1.0 disclaimer

This is a pre-launch installer. The mesh has not yet gone live with
external buyers. Earnings shown in the GUI are *estimated* until the
mesh is in steady state. Today's value of running Pluginfer is:

1. Free training of your own models via §E1 compute-as-currency
2. Contribution to the very-first mesh deployment
3. Cold-start attestation bonus when public mesh launches

For the production cash-payout flow, the mesh needs first paying
buyers — that's the next milestone.

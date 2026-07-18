# Pluginfer Quickstart (5 minutes)

Zero to a running node with a browser control panel, free testnet
credits, and a first signed job.

## 1. Install

Grab the latest release from
<https://github.com/pluginfer/pluginfer/releases/latest>:

- **Windows** — `Pluginfer-<version>-Setup.exe`. The installer is not
  yet Authenticode-signed, so SmartScreen will warn: "More info →
  Run anyway". Verify the download instead via the release's signed
  `manifest.json` (see [Signing setup](SIGNING_SETUP.md)). A Start
  Menu shortcut ("Pluginfer") is created.
- **Linux (Debian/Ubuntu)** — `pluginfer_<version>_amd64.deb`, then
  `sudo dpkg -i pluginfer_<version>_amd64.deb`.
- **macOS** — `Pluginfer-<version>.pkg` (unsigned for now — right-click
  → Open).
- **From source** — `git clone`, `cd pluginfer/v2`,
  `pip install -r requirements.txt`.

## 2. Run

```sh
pluginfer up          # from source: python pluginfer.py up
```

Zero config: the node picks a port, creates its wallet, detects a
local model runtime if you have one (Ollama etc. — honest echo mode
otherwise), and opens the **browser control panel** — status, earnings,
and buttons for everything below. `--share` additionally makes the
node reachable by the whole mesh through a free auto-tunnel.

## 3. Get free testnet credits

Click **Get free test credits** on the panel, or:

```sh
curl -X POST http://127.0.0.1:8100/v1/testnet/faucet \
  -H "Content-Type: application/json" -d '{"wallet_id": "me"}'
```

One-time starter balance per wallet. Testnet economics, stated
plainly: balances are real, persistent accounting but not redeemable
for cash; there is **no token, ever** — the ledger is plain USD.

## 4. Submit a job

From the panel ("Try it"), or from code:

```sh
pip install pluginfer
```

```python
from pluginfer import Pluginfer

with Pluginfer(base_url="http://localhost:8100") as p:
    job = p.jobs.submit(
        kind="llm.completion",
        payload={"prompt": "What is the capital of France?"},
        cost_ceiling_usd=0.05,
        latency_ceiling_ms=10_000,
    )
    final = p.jobs.wait_for(job.job_id, timeout_sec=30)
    print(final.state.state)
```

Every completed job carries an Ed25519-signed receipt. Want the
answer verified across independent nodes? Add `"quorum_n": 3` to the
payload — the result is majority-voted and only the agreeing majority
is paid ([guide](SETUP_GUIDES.md)).

## 5. Audit the money yourself

```sh
curl http://127.0.0.1:8100/v1/ledger/verify
```

Every balance is recomputed from its full entry history — tampering
or deletion is detected and blocks payouts automatically.

## Troubleshooting

| Symptom                        | Likely cause                                              |
| ------------------------------ | --------------------------------------------------------- |
| `auction: no provider matched` | Cost ceiling too low; raise `cost_ceiling_usd`.           |
| Faucet returns 409             | That wallet already got its one-time grant.               |
| Panel shows "local-only"       | Run with `--share` to become reachable by the mesh.       |
| SmartScreen blocks the .exe    | Expected while installers are unsigned — verify the signed `manifest.json` instead. |

Per-feature guides for everything else: [SETUP_GUIDES.md](SETUP_GUIDES.md).

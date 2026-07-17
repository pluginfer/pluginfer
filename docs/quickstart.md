# Pluginfer Quickstart (5 minutes)

This guide gets you from zero to a running Pluginfer node and a first
signed compute receipt on the chain.

> Tested fresh on Ubuntu 22.04, macOS 14, Windows 11 Pro.

## 1. Install

### Linux (Debian / Ubuntu)

```sh
curl -L https://github.com/pluginfer/pluginfer/releases/latest/download/pluginfer_amd64.deb -o pluginfer.deb
sudo dpkg -i pluginfer.deb
```

### macOS

Download the latest signed `.pkg` from
https://github.com/pluginfer/pluginfer/releases/latest, double-click,
follow the wizard.

### Windows

Download the latest `Pluginfer-*-Setup.exe`, run it. SmartScreen will
trust an EV-signed installer immediately.

## 2. Run

```sh
pluginfer start --role provider
```

You should see:

```
[pluginfer] Node started. Wallet: pf1abc...xyz
[pluginfer] Bootstrapping from seeds: 1 / 1 reachable
[pluginfer] Joined mesh. Peers: 1. Listening for jobs.
```

## 3. Submit a job (from another machine, or the same one)

```sh
pip install pluginfer
```

```python
from pluginfer import Pluginfer

with Pluginfer(api_key="pf_test_local",  # local node accepts the literal "pf_test_local"
               base_url="http://localhost:8100") as p:
    job = p.jobs.submit(
        kind="llm.completion",
        payload={"prompt": "What is the capital of France?", "max_tokens": 16},
        cost_ceiling_usd=0.05,
        latency_ceiling_ms=10_000,
    )
    print("submitted:", job.job_id)
    final = p.jobs.wait_for(job.job_id, timeout_sec=30)
    print("state:", final.state.state)
    print("result:", p.jobs.decode_result(p.jobs.result(job.job_id)))
```

## 4. Verify the receipt on the chain

```sh
pluginfer chain receipt <job_id>
```

Returns:

```
job_id: <id>
provider: pf1...
result_hash: <sha256>
provider_sig: <verified ✓>
on-chain block: 11,432
payment: 0.0003 PLG -> pf1...
```

## 5. (Provider) See your balance grow

```sh
pluginfer wallet balance
```

## Troubleshooting

| Symptom                           | Likely cause                                                      |
| --------------------------------- | ---------------------------------------------------------------- |
| `bootstrap: 0/0 reachable`        | No seed nodes configured. See `/etc/pluginfer/config.yml`.       |
| `auction: no provider matched`    | Cost ceiling too low; raise `cost_ceiling_usd`.                  |
| `result_hash mismatch`            | Provider returned tampered output -- refund eligible. Report it. |
| `429 rate_limited`                | API key throttled; wait `Retry-After` seconds.                   |
| Windows SmartScreen blocks `.exe` | Wait for EV cert reputation to build, or use the OV cert.        |

## Uninstall

```sh
sudo dpkg -r pluginfer    # Linux
sudo /Applications/Pluginfer.app/Uninstall.command   # macOS
# Windows: Settings -> Apps -> Pluginfer -> Uninstall
```

Wallets at `~/.config/pluginfer/wallet.pem` are NOT deleted by
default -- back them up if you uninstall on the user's only machine.

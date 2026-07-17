# Pluginfer first-proof — 3 boxes, ~90 minutes, ~$20

The goal: the artifact a VC will ask for. A real chat completion routed
through a real auction across 3 real machines running a real model,
returning a signed receipt that verifies under the gateway's public key.

## Topology

```
       ┌──────────────────────┐
       │   SEED   (CPU, $4/mo)│  ← Hetzner CX22, public IP
       │  seed.example.com    │
       └──────────────────────┘
              ▲       ▲
              │       │  TCP/9000 (REGISTER/PEERS)
              │       │
   ┌──────────┴───┐ ┌─┴─────────────┐
   │ NODE A (GPU) │ │ NODE B (GPU)  │
   │ RunPod 4090  │ │ Vast.ai 3090  │
   │ ~$0.40/hr    │ │ ~$0.25/hr     │
   │ Ollama+Qwen  │ │ Ollama+Qwen   │
   │ auto_mesh    │ │ auto_mesh     │
   └──────────────┘ └───────────────┘
```

Both compute nodes register with the seed; gossip propagates them to
each other; the auction on either node can route work to either.

## What you pay

| Item | Cost |
|---|---|
| Hetzner CX22 (seed) | €3.79/mo prorated to ~$0.01/hr |
| RunPod RTX 4090 spot | $0.34/hr |
| Vast.ai RTX 3090 spot | $0.20-0.30/hr |
| **Total for 2-hour proof** | **~$2** |

If you keep the seed running long-term and rent GPUs on-demand: $5/mo
baseline. The expensive boxes only run when you need to demo.

## Step-by-step (~90 min total)

### Step 1: Rent the seed VPS (10 min)

1. Hetzner Cloud → New Server → Ubuntu 24.04, CX22 (€3.79/mo).
2. Region: pick one close to your GPUs.
3. SSH in:
   ```bash
   ssh root@<seed-ip>
   curl -fsSL https://raw.githubusercontent.com/<your-fork>/pluginfer/main/v2/deploy/install_seed.sh \
       | sudo bash -s -- --pluginfer-version main
   ```
4. Output ends with the seed pubkey. **Save it.** Note the IP and port 9000.

Verify:
```bash
nc -zv <seed-ip> 9000     # should connect
```

### Step 2: Rent two GPU nodes (15 min)

**RunPod path (recommended for first proof — UI is simple):**
1. runpod.io → Secure Cloud → RTX 4090 → Ubuntu 22.04 template.
2. Spot pricing, ~$0.34/hr.
3. Choose "Connect" → "TCP Public IP" so the node has a reachable IP.
4. SSH in.

**Vast.ai path (cheaper):**
1. vast.ai → search RTX 3090 / 4090 → filter for "Direct" connection
   type so the box has a public IP (not behind their NAT proxy).
2. Spot bid at ~$0.20-0.30/hr.
3. Use the Jupyter or SSH access.

On each GPU node:
```bash
curl -fsSL https://raw.githubusercontent.com/<your-fork>/pluginfer/main/v2/deploy/install_node.sh \
    | sudo bash -s -- \
        --seed-host <seed-ip> \
        --seed-port 9000 \
        --node-port 8101 \
        --model qwen2.5:1.5b
```

The script's last lines must show:
```
[install_node] OK: node <ip>:8101 serving qwen2.5:1.5b via ollama
```

If you see `is_echo: true` or `runtime: alpha-echo`, the model didn't
load. Re-check `systemctl status ollama` + `ollama list`.

### Step 3: Verify the mesh formed (5 min)

From your laptop:
```bash
# Each node sees the other (and the seed).
curl http://<node-A>:8101/peers | jq
curl http://<node-B>:8101/peers | jq
```

Expected:
- `view_size: 1` (the other node)
- `auction_size: 2` (own flagship + cross-node)
- `registered_cross_nodes: [<other-node-pubkey>]`
- `runtime.name: "ollama"` and `runtime.is_echo: false`

If `view_size = 0` after 30 seconds:
- Check the seed is reachable from both nodes: `nc -zv <seed-ip> 9000`.
- Check the node's outbound `seed_register` succeeded: `journalctl -u auto_mesh -n 100 | grep seed_register`.

### Step 4: Submit the FIRST real job (5 min)

```bash
curl -X POST http://<node-A>:8101/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "qwen2.5:1.5b",
      "messages": [{"role": "user", "content": "Explain in two sentences why decentralized GPU mesh networks matter."}],
      "max_tokens": 200,
      "pluginfer_cost_ceiling_usd": 0.05,
      "pluginfer_latency_ceiling_ms": 30000
    }' | jq
```

Expected response shape:
- HTTP 200
- `choices[0].message.content` is a real Qwen-generated answer (NOT
  `"pluginfer-alpha: Explain in..."` — that would mean echo path).
- Response headers include `X-Pluginfer-Receipt-ID` and
  `X-Pluginfer-Price-USD`.

### Step 5: Verify the signed receipt (5 min)

```bash
# Pull the signed PNIS receipt for the job.
RECEIPT_ID=<value from X-Pluginfer-Receipt-ID header>
curl http://<node-A>:8101/v1/receipts/$RECEIPT_ID | jq

# Verify the signature.
python3 <<'PY'
import json, sys
sys.path.insert(0, '/opt/pluginfer/v2')
from core.ai_receipt import AIReceipt
d = json.load(open('/tmp/receipt.json'))
print("verify():", AIReceipt.from_dict(d).verify())
PY
```

Save the screenshot of:
- `curl http://<node-A>:8101/peers` showing both nodes.
- The chat-completion response with real text.
- `AIReceipt.verify() == True`.

**That's the artifact.** Three real machines, real model, real auction,
real signed receipt that anyone can independently verify.

## What can go wrong + the fix

| Symptom | Fix |
|---|---|
| `view_size: 0` after 30s | Seed not reachable from nodes. Check ufw allows 9000 outbound and the seed's IP is public. |
| `runtime.name: "alpha-echo"` | Ollama not running. `systemctl status ollama` + `ollama list`. |
| 502 Bad Gateway on /v1/chat/completions | Auction had no qualifying provider. Check tier caps: untrusted tier caps at $0.10. The example above uses 0.05 — OK. If you bump cost_ceiling above 0.10, set `PLUGINFER_UNTRUSTED_MAX_USD=10.0` in `/etc/pluginfer/auto_mesh.env` and `systemctl restart auto_mesh`. |
| Echo response despite Ollama running | Wrong model id. The env var `PLUGINFER_ALPHA_MODEL_ID` must match the `ollama pull` name (e.g. `qwen2.5:1.5b`, not `Qwen/Qwen2.5-1.5B-Instruct`). |
| Slow first request | Ollama loads the model into VRAM on first call. Subsequent requests are 10-100x faster. |
| Nodes can't reach each other directly (cloud NAT) | The mesh-native relay (`/relay/{peer_hash}` we shipped) handles this automatically. Check `auto_mesh` logs for `cross_node_path: relay` on completed jobs. |

## After the proof: what changes

Once you have the screenshot:

1. **Update the README** with the receipt hash and the live `/peers`
   output. Move from "tested in isolation" to "running in production".
2. **Cold-DM 10 startups** with the OpenAI shim demo:
   `OPENAI_BASE_URL=http://<your-gateway>/v1` → real Qwen output at
   $0.03/M tokens. Use the templates in `growth/cold_dm_template.md`.
3. **Publish the seed pubkey + IPs** in the bundled
   `data/seed_registry.json` so other operators can join your mesh.
4. **Run a tiny DiLoCo training run** on the same 2 nodes — proves
   the WAN-tolerant training math we already shipped, not just the
   inference path.

## What this proof does NOT prove (still honest)

- 100-node scale. We've only tested 3.
- Production SLA. One run is not 99.9% uptime.
- Tensor-parallel of 70B models over WAN. We don't claim to.
- A real customer paid you. Use the receipts as the pitch artifact for
  the FIRST real customer; that's the next step after this proof.

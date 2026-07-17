# Pluginfer WAN deployment

Quickstart for spinning up a 3-region public seed quorum + your
own auto-mesh node.

## Seed cluster (3 regions, ~15 minutes total)

1. Rent three small Linux VPSes — Hetzner CX22 at €3.79/mo each
   works fine. Pick EU + US + APAC for global RTT.

2. On each, run:
   ```bash
   curl -fsSL https://pluginfer.network/deploy/install_seed.sh \
        | sudo bash -s -- --pluginfer-version v0.1.0
   ```
   The script:
   - installs Python 3.12 + the Pluginfer venv,
   - creates a `pluginfer` system user,
   - generates a fresh seed wallet,
   - drops the systemd unit + starts it,
   - opens tcp/9000 in ufw,
   - prints the seed's public key.

3. Save each seed's public key + (host, port). You'll need them
   in step 4.

## Quorum-signed seed registry

Once the seeds are up, generate a signed registry the auto_mesh
client can trust:

```bash
python -m deploy.generate_seed_registry \
    --validator-key /etc/pluginfer/validator_a.pem \
    --validator-key /etc/pluginfer/validator_b.pem \
    --seed seed-eu.pluginfer.network:9000:./eu_seed_pub.pem \
    --seed seed-us.pluginfer.network:9000:./us_seed_pub.pem \
    --seed seed-sg.pluginfer.network:9000:./sg_seed_pub.pem \
    --min-signatures 2 \
    > v2/data/seed_registry.json
```

The validator keys are SEPARATE from the seed wallets. Their only
job is to sign the (host, port, pubkey_fp) tuples in the registry.
Two distinct organizations holding validator keys = "Pluginfer
itself + one independent auditor" = the minimum quorum.

Commit `seed_registry.json` to the repo (or publish it via a
signed HTTPS endpoint). The auto_mesh client at boot:

1. Loads the bundle.
2. Filters to records with ≥ min_signatures quorum_signatures.
3. Sorts by reachability probe.
4. Uses the first reachable record.

## auto_mesh node (a buyer's workstation OR a paid provider)

```bash
# 1. Install (Mac/Linux):
curl -fsSL https://pluginfer.network/install.sh | bash
# 2. Run:
python -m tools.run_node      # the supervisor — auto-restart on crash
```

Or as a systemd service on Linux:
```bash
sudo cp v2/deploy/auto_mesh.service /etc/systemd/system/
# Customize /etc/pluginfer/auto_mesh.env with your seed URL.
sudo systemctl enable --now auto_mesh
```

## Observability

```bash
# Seed health
journalctl -u seed_node -f

# Auto-mesh status (peers, auction size, runtime mode)
curl -s http://localhost:8101/peers | jq

# Hosted gateway wallet balance
curl -s -H "Authorization: Bearer $PLG_KEY" \
    http://localhost:8101/v1/wallets/$WALLET_ID | jq
```

## What this gets you

- Three operator-published seeds with quorum-signed records — no
  TOFU single-trust-anchor problem.
- Auto-mesh nodes can join from anywhere; gossip propagates the
  mesh without recontacting the seed after bootstrap.
- Systemd-managed supervised processes with crash-loop backoff.
- Firewall + hardening (NoNewPrivileges, ProtectSystem) on the
  seed host.

## What this DOESN'T give you

- Real EV cert for `pluginfer.network` (off-keyboard, $99/yr).
- Apple Developer signing (off-keyboard, $99/yr).
- 99.99% SLA — depends on real-world uptime measurements, not
  promised at code level.

Total compute cost to keep this running: 3 × €3.79 = ~$13/mo.

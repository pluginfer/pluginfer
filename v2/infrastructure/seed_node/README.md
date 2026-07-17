# Pluginfer Bootstrap Seed Node

Lightweight asyncio TCP server that registers Pluginfer peers and serves
peer-list responses. Designed to run as a single Linux container on a
$5/month VPS (Hetzner / DigitalOcean / Vultr).

The seed is a discovery convenience, NOT a control plane. It does not
hold funds, does not validate transactions, and does not gate joining
the mesh. A peer that registers with the seed gets discovered faster;
one that uses DHT bootstrap directly is treated identically.

## Wire protocol

Newline-delimited JSON over TCP, one message per connection.

### REGISTER (client -> server)

```json
{
  "op": "REGISTER",
  "pubkey_pem": "-----BEGIN PUBLIC KEY-----\n...\n-----END PUBLIC KEY-----",
  "ip": "1.2.3.4",
  "port": 8100,
  "node_version": "1.0.0",
  "timestamp": 1714780800,
  "signature": "<base64 ECDSA over `pubkey_pem|ip|port|node_version|timestamp`>"
}
```

Server response:

```json
{ "status": "ok", "ttl_seconds": 600, "peers": 42 }
```

Failure modes (status: error):

| code             | reason                                                  |
| ---------------- | ------------------------------------------------------- |
| `bad_request`    | missing or wrong-typed field                            |
| `bad_port`       | port not in [1, 65535]                                  |
| `stale_timestamp`| client clock more than 30s off from server clock        |
| `bad_signature`  | ECDSA verify of (pubkey, signed_bytes, signature) fails |
| `rate_limited`   | source IP exceeded 10 registrations/min                 |

### PEERS (client -> server)

```json
{ "op": "PEERS", "max": 50 }
```

Response:

```json
{
  "status": "ok",
  "peers": [
    { "ip": "5.6.7.8", "port": 8100, "pubkey_pem": "...", "node_version": "1.0.0" },
    ...
  ]
}
```

`max` is clamped to 50 server-side.

### PING (client -> server)

```json
{ "op": "PING" }
```

Response:

```json
{ "status": "ok", "peers": 42, "uptime_s": 12345, "version": "pluginfer-seed/1.0.0" }
```

## Running locally

```sh
# from v2/
python -m infrastructure.seed_node.seed_server --host 0.0.0.0 --port 9000
```

## Running in Docker

```sh
# Build (from repo root)
docker build \
  -t pluginfer-seed:1.0.0 \
  -f v2/infrastructure/seed_node/Dockerfile .

# Run (host port 9000 -> container port 9000)
docker run -d \
  --name pluginfer-seed \
  --restart=always \
  -p 9000:9000 \
  pluginfer-seed:1.0.0
```

## VPS deployment (Ubuntu 22.04)

```sh
ssh root@your.vps.ip
curl -fsSL \
  https://raw.githubusercontent.com/pluginfer/pluginfer/main/v2/infrastructure/seed_node/deploy.sh \
  -o deploy.sh && chmod +x deploy.sh && ./deploy.sh
```

`deploy.sh up` installs Docker if needed, clones the repo to
`/opt/pluginfer`, builds the image, and starts the container with
`--restart=always`. `deploy.sh down` tears it down.

After deployment, verify:

```sh
sudo docker ps                     # pluginfer-seed should be running
echo '{"op":"PING"}' | nc 127.0.0.1 9000
# {"status":"ok","peers":0,"uptime_s":12,"version":"pluginfer-seed/1.0.0"}
```

## Hardcoding seed addresses

After deploying, add the seed's public IP + the seed-server's wallet
pubkey to `BOOTSTRAP_SEEDS` in `v2/core/complete_mesh_controller.py`.
The seed itself does NOT need to publish its pubkey through
registration -- it is the trust anchor; client nodes pin its pubkey
at build time.

To generate a fresh seed wallet (run once on the seed VPS):

```sh
python -c "from core.tokenomics import Wallet
w = Wallet()
print('SEED_NODE_PUBKEY_HEX:')
print(w.public_key_pem)
w.save_to_file('/etc/pluginfer/seed_wallet.pem')"
```

Take the printed pubkey and paste into `BOOTSTRAP_SEEDS`. Keep
`seed_wallet.pem` ON THE SEED ONLY -- it never leaves the seed VPS.

## Resource usage

The seed is sized for thousands of peers without breaking a sweat.
Memory is bounded by `len(self.peers)` * ~500 bytes per record;
10,000 peers = ~5 MB. CPU is dominated by ECDSA verify on REGISTER;
~3 ms per verify on a single core means ~300 registrations/sec
sustained throughput.

The `docker-compose.yml` deploy.resources block caps the container at
128 MB / 0.25 CPU which is sufficient for any single seed; serious
deployments run 3+ seeds behind round-robin DNS for redundancy.

## Logs

`json-file` driver, max-size 10 MB, 5 files retained (50 MB total).
Each log line is structured JSON:

```json
{"ts": 1714780800.0, "event": "REGISTER", "pubkey_short": "...",
 "ip": "1.2.3.4", "port": 8100, "node_version": "1.0.0"}
```

Events: `REGISTER`, `EXPIRE`, `RATE_LIMIT`, `STALE_TS`, `BAD_SIG`.

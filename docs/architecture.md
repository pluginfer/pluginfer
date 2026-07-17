# Pluginfer architecture

Pluginfer is a peer-to-peer GPU compute marketplace with crypto-native
economics. This doc traces the full request path from a Python SDK
call to an on-chain settlement, so a new contributor can hold the
whole system in their head.

## Layers

```
+----------------------------------------------------------+
|  SDKs (Python, JS/TS, curl)                              |
+----------------------------------------------------------+
|  REST API  (/v1/jobs, /v1/wallet, /v1/auth, /metrics)    |
|  -- FastAPI + auth + rate-limit + request-id middleware  |
+----------------------------------------------------------+
|  JobsService  (auction -> execute -> chain settle)       |
+-----------+----------------+--------------+--------------+
| Auction   | Providers       | Chain        | Mesh         |
| (sealed   | (peer GPUs +    | (PoW + DAA + | (DHT +       |
|  bid)     |  cloud LLMs)    |  BFT slash + | STUN +       |
|           |                 |  ZK privacy) |  signed      |
|           |                 |              |  gossip)     |
+-----------+----------------+--------------+--------------+
|  PNIS (Pluginfer Neural Intelligence System)             |
|  -- 1.1B-param transformer for in-house decisions        |
+----------------------------------------------------------+
```

## Request flow

1. **Submit** -- SDK calls `POST /v1/jobs`. Auth: API key (sha256 in
   the backend) or wallet ECDSA challenge-response.
2. **Auction** -- `JobsService.submit` runs the auction in a worker
   thread; each registered provider returns a `Bid` (price, eta,
   quality, privacy grade). `Bid.violates` filters; `Bid.score`
   ranks (Pareto with novel 4-term form).
3. **Match** -- the winning bid + losing-bid disclosure are recorded.
4. **Execute** -- the winner's `provider.execute(job, bid)` runs;
   result bytes + `sha256(result)` + provider's wallet signature land
   in the JobResult.
5. **Settle** -- the requester recomputes sha256 and verifies; on
   match, payment is signed and submitted. The chain validates the
   tx (sig + nonce + fee + balance), mines a block, and the
   participants' balances update.
6. **Verify** -- ANY observer can fetch `/v1/jobs/{id}/result`,
   recompute the hash, and verify the provider signature against
   the on-chain pubkey.

## Chain

- **PoW + DAA** -- standard difficulty retargeting every 16 blocks.
- **BFT slash evidence** (W32, partial) -- ≥2/3 validators sign
  equivocation evidence; on-chain stake destruction.
- **Privacy** (W4) -- Pedersen commitments + Schnorr PoK + bit-OR
  proofs. ZK gradient provenance binds a worker's gradient to the
  committed (data, model) tuple.
- **Replay protection** -- per-sender monotonic nonce baked into
  tx_id pre-image.
- **Mempool tamper protection** -- `add_transaction` recomputes
  tx_id from current fields and refuses if it disagrees with the
  stored id (sec3.3 v2).
- **State index** -- balances + nonces are derived state; cached
  on apply-block, recomputed on reorg or load_chain.

## Mesh

- **DHT** (CP-2) -- Kademlia routing with signed records, LRU+TTL
  storage. Node IDs derived from sha256(pubkey)[:20].
- **NAT traversal** (CP-2 + post-CP-FINAL hole-punch / TURN):
  strategy chain DIRECT -> UPnP -> STUN -> seed-brokered hole-punch
  -> TURN relay. The seed at port 9000 runs TCP (REGISTER / PEERS)
  AND UDP (REGISTER_UDP / INTRODUCE / RELAY) so symmetric-NAT
  peers (~15-20% of consumer routers) survive on the relay while
  full/restricted/port-restricted-cone peers (~80%) get a direct
  P2P UDP path after one round-trip via the seed.
- **Hole-punch protocol** -- A INTRODUCEs to B; the seed sends
  PUNCH_INVITE to BOTH peers with the OTHER's external (ip, port)
  and a shared nonce; both fire a PUNCH_HELLO burst; the first
  packet on each side opens the NAT pinhole; subsequent traffic
  flows direct. ECDSA-signed INTRODUCE + src-mismatch filter
  prevents reflection attacks (an attacker who sniffs an
  INTRODUCE can't replay it from elsewhere).
- **TURN relay** -- when hole-punch fails, A RELAY_OPENs a session
  through the seed; the seed forwards every RELAY packet to B.
  Per-session bandwidth quota (50 MB default) caps DoS.
- **Bootstrap** -- `BOOTSTRAP_SEEDS` (filled post-deploy by ops) +
  cached `peers.json` fallback.
- **Gossip** (W24) -- ECDSA-signed envelopes; receivers dedup via
  (origin, id) and verify before re-flooding.

## Providers

Three concrete `Provider` impls:

- `MeshGPUProvider` -- a peer GPU on the LOCAL Pluginfer mesh. Bid
  pulls hardware capability + slack-curve pricing. Execute dispatches
  via `task_router` and signs the result hash with the wallet.
- `_CloudLLMProvider` -- cloud LLM API adapter (OpenAI, Anthropic,
  Gemini, Ollama). API keys read from OS keychain (never disk),
  failing closed if not configured.
- `RemoteProvider` -- a peer reachable only over the network. Wraps
  a `MeshConnector` channel; on `execute()` it serialises the job
  over the channel, awaits the peer's signed result, returns it.
  Indistinguishable from a local provider from the auction's
  perspective. The companion `JobServer` runs on the OTHER end and
  dispatches inbound JOB_REQUESTs to that node's local provider
  (typically a `MeshGPUProvider`).

The Auction is provider-agnostic; new types implement the ABC. The
`MeshConnector` provides the transport layer (NAT survival via
hole-punch, then TURN relay if needed) so a `RemoteProvider` works
across continents the same way it works over loopback.

## Build & install

- `v2/build/` -- per-platform installer pipeline (.deb / .exe /
  .pkg). ECDSA-signed manifest mirrors the runtime updater (W31 /
  CP-3).
- `core/updater.py` -- verifies the release manifest signature
  before downloading or running any artefact.

## Observability

- `core/metrics.py` -- Counter, Gauge, Histogram, Registry; renders
  Prometheus text format.
- `GET /metrics` -- exposes
  `pluginfer_jobs_total{status}`,
  `pluginfer_job_duration_seconds`,
  `pluginfer_auction_duration_seconds`,
  `pluginfer_peers_connected`,
  `pluginfer_chain_height`,
  `pluginfer_balance_plg`,
  `pluginfer_uptime_seconds`.
- `core/structured_logging.py` -- one JSON object per record, with
  per-field redaction of secrets (CWE-532 defence in depth).

## What's NOT in v1.0

These are tracked but require off-keyboard work:

1. **Real DHT seed servers in production** -- code ships, ops fills
   `BOOTSTRAP_SEEDS` post-deploy.
2. **Paid Authenticode + Apple Developer + EV cert** -- pipeline
   consumes them; pipeline runs UNSIGNED otherwise. See
   `docs/SIGNING_SETUP.md`.
3. **Full BFT slash-evidence protocol** (W32) -- partial; needs ≥⅔
   validator coordination.
4. **PNIS pretraining** at full 1.13B param scale -- needs A100/H100
   compute days.
5. **Third-party security audit** (Trail of Bits / Halborn).
6. **Token + jurisdiction regulatory clearance**.

# Pluginfer REST API reference (v1)

Base URL: your own node — `http://127.0.0.1:8100` by default
(`pluginfer up`). There is no hosted cloud API; every deployment is
self-hosted.
OpenAPI spec: `GET /openapi.json` · Interactive docs: `GET /docs`

> **Note on units:** endpoints marked *(legacy chain)* belong to the
> internal test-chain surface whose accounting unit is "PLG" — an
> internal ledger unit, **not** a public token (there is no token,
> ever). The live money surface is the **USD** ledger on the node:
> `/v1/testnet/faucet`, `/v1/ledger/*`, `/v1/stake/*`,
> `/v1/economics/*` — see [SETUP_GUIDES](SETUP_GUIDES.md).

## Auth

Pluginfer accepts two credential types on every authed endpoint:

| Header                     | Type                          |
| -------------------------- | ----------------------------- |
| `Authorization: Bearer ...`| API key (`pf_live_...`)       |
| `X-Pluginfer-Session: ...` | Session id from wallet login  |

API keys are issued out-of-band by the operator (or via the dashboard
UI) and stored as `sha256(key)` only.

For wallet auth:

```
POST /v1/auth/challenge   -> { nonce, expires_at_unix, audience }
POST /v1/auth/verify      -> { session_id }
```

The client signs `nonce|audience|expires_at_unix` with their wallet
ECDSA key and posts the signature for verification.

## Endpoints

### `GET /v1/version`
No auth. Returns `{version, git_sha, api}`.

### `GET /v1/status`
No auth. Returns `{status, version, git_sha, chain_height,
peers_connected, uptime_seconds}`.

### `POST /v1/jobs` (auth required)
Body:
```json
{
  "kind": "llm.completion",
  "payload": {"prompt": "Hello", "max_tokens": 32},
  "cost_ceiling_usd": 0.05,
  "latency_ceiling_ms": 10000,
  "privacy_class": "public",
  "quality_floor": 0.7,
  "webhook_url": null
}
```
Returns `JobInfo` with `job_id`, current `state`, etc.

### `GET /v1/jobs/{id}` (auth required)
Returns the latest `JobInfo`. 404 if unknown, 403 if not yours.

### `GET /v1/jobs/{id}/result` (auth required)
Returns `JobResult` with `result_b64`, `result_hash_hex`, and
`provider_signature_b64`. Verify the hash before trusting the result.

### `GET /v1/jobs/{id}/stream` (auth required)
Server-Sent Events stream of state transitions. Events:
`job.queued`, `job.matched`, `job.running`, `job.completed`,
`job.failed`, `job.timeout`, `job.cancelled`, plus periodic `ping`.

### `DELETE /v1/jobs/{id}` (auth required)
Cancels a job that hasn't reached a terminal state.

### `GET /v1/wallet/balance` (auth required) *(legacy chain)*
Returns `{address, balance_plg, pending_plg, chain_height}` — internal
test-chain units, not money; the USD ledger lives at
`/v1/ledger/wallets/{id}`.

### `GET /v1/providers` (auth required)
Returns the registered provider directory.

### `GET /metrics`
Prometheus text format. No auth (front with reverse-proxy if you
want to gate by IP / token).

## Error format

```json
{ "detail": "missing_or_invalid_credentials" }
```

Every response carries `X-Request-ID` (echoed if you sent one,
generated otherwise) for log correlation.

## Status codes

| Code | Meaning                                      |
| ---- | -------------------------------------------- |
| 200  | OK / no body change                          |
| 202  | Accepted (job submitted, queued)             |
| 401  | Missing or invalid credentials               |
| 403  | Authenticated but not your resource          |
| 404  | Job not found                                |
| 429  | Rate limited (`Retry-After` header set)      |
| 500  | Server error                                 |

## Rate limits

Default: 60 requests/sec sustained, 60-request burst, per API key
(or per-IP if no key). Configurable per deployment.

# Setup guides — every shipped feature, step by step

Each guide is self-contained: prerequisites, exact commands, and what
you should see. Everything here is exercised by the test suite or a
live proof — if a guide doesn't work, that's a bug; please open an
issue.

Common prerequisite for source installs:

```sh
git clone https://github.com/pluginfer/pluginfer
cd pluginfer/v2 && pip install -r requirements.txt
```

The Windows/macOS installers and the `.deb` on the
[releases page](https://github.com/pluginfer/pluginfer/releases) bundle
all of this; `pluginfer up` is the same command everywhere.

---

## 1. Run your own AI node (zero config)

```sh
python pluginfer.py up
```

What happens, with no settings needed from you: hardware is detected,
a wallet is created (encrypted at rest), a free port is picked, real
models bind via [Ollama](https://ollama.com) if installed (an honestly
labeled echo model otherwise), and your browser opens the **control
panel** — status, credits, earnings, a try-it box, and a plain-English
how-it-works. Any OpenAI client works against
`http://127.0.0.1:8100/v1`. The node self-supervises and restarts
itself; `Ctrl+C` stops it.

## 2. Share your compute with the mesh (and earn)

```sh
python pluginfer.py up --share
```

`--share` opens a free Cloudflare tunnel (no account, no card) and
advertises your node so other people's jobs can reach it — zero router
configuration. You'll see `You are LIVE on the internet: https://…`.
Sharing is always an explicit flag, never a silent default. Earnings
from jobs you serve appear on the control panel and in the ledger.

## 3. Private enterprise mesh (link your datacenters)

On **every** site, same key:

```sh
python pluginfer.py up --share --swarm-key "your-company-secret"
```

Every mesh surface now requires the key — nodes or clients without it
get `401`. Clients send header `X-Pluginfer-Swarm-Key: <key>`; the
control panel prompts for it once. Connect sites by pointing each node
at another's address (`PLUGINFER_GOSSIP_BOOTSTRAP_PEER=host:port`).
Jobs are auctioned across sites by price, latency, and privacy class
(`public` / `private` / `sensitive`), and signed receipts double as
internal chargeback records.
Scope: one shared symmetric key, TLS transport required (`--share`
tunnels are https; otherwise put a TLS proxy in front).

## 4. Signet — the spend gateway (budget caps + signed receipts)

```sh
python -m governance.demo_harness --serve
```

Point any OpenAI-compatible client's base URL at it. Set budgets per
envelope; when a budget is exhausted the gateway **fails closed** (the
call is refused, not billed). Every call lands in a signed,
hash-chained receipt — verify the chain with the public endpoint on
the dashboard. Savings shown are measured counterfactuals, never
projections.

## 5. Multi-LLM routing (any number of models, any providers)

In your price sheet, each model may carry its own provider:

```json
{
  "gpt-4o":  {"input_per_1m": 2.5, "output_per_1m": 10,
              "upstream": "https://api.openai.com", "api_key_env": "OPENAI_API_KEY"},
  "llama-70b": {"input_per_1m": 0.6, "output_per_1m": 0.8,
              "upstream": "https://api.groq.com/openai", "api_key_env": "GROQ_API_KEY"}
}
```

Then pick a routing mode:

- `PLUGINFER_GW_AUTOROUTE=save` — easy prompts (chat/summarize/extract)
  go to your cheapest model; hard prompts are never downgraded.
- `PLUGINFER_GW_AUTOROUTE=smart` — additionally upgrades code and
  long-context prompts to your most capable model (extra cost is
  recorded as a **negative** saving, never hidden).
- `PLUGINFER_GW_ROUTES=/path/rules.json` — full custom rules, first
  match wins.

Works with any LLM behind an OpenAI-compatible chat API. Native
non-OpenAI wire formats need an adapter (issue #1).

## 6. Free testnet credits (the faucet)

```sh
curl -X POST http://127.0.0.1:8100/v1/testnet/faucet \
  -H "Content-Type: application/json" -d '{"wallet_id": "me"}'
```

One-time starter balance (default $25 test-USD) per wallet — or click
**Get free test credits** on the control panel. Refused outright in
mainnet mode. There is no token, ever.

## 7. Audit the money ledger yourself

```sh
curl http://127.0.0.1:8100/v1/ledger/verify
```

Recomputes every balance from its full signed entry history. Any
mismatch (edit, corruption, deleted state file) is reported — and
blocks payouts automatically until resolved.

## 8. Python SDK

```python
import sys; sys.path.insert(0, "v2/sdk/python")
from pluginfer import Pluginfer

c = Pluginfer(base_url="http://127.0.0.1:8100")
job = c.jobs.submit(kind="compute.echo", payload={"x": 1},
                    cost_ceiling_usd=0.01, latency_ceiling_ms=5000)
print(c.jobs.wait_for(job.job_id, timeout_sec=30).state.state)  # completed
```

## 9. JavaScript / TypeScript SDK

```sh
cd v2/sdk/javascript && npm install && npm run build
```

```js
import { Pluginfer } from "@pluginfer/sdk";   // or require() — both ship
const c = new Pluginfer({ baseUrl: "http://127.0.0.1:8100" });
const job = await c.jobs.submit({ kind: "compute.echo", payload: { x: 1 } });
```

Errors map to typed classes (`AuthenticationError`, `JobNotFoundError`,
`RateLimitError` with `retryAfterSec`).

## 10. Bring your own supply (any OpenAI endpoint joins the auction)

Have Ollama, vLLM, or any OpenAI-compatible server running? The
`meshllm` bridge wraps it as one more bidder in your node's auction —
see `core/meshllm_provider.py` for the registration pattern. Your
node then arbitrates between local models, mesh peers, and that
endpoint per job.

## 11. Streaming that survives disconnects

Submit with `"stream": true` as usual. Each SSE chunk carries a delta
cursor; if the connection drops, reconnect with the last cursor and the
stream resumes from the next chunk — no lost tokens, no double-billed
tokens.

## 12. Ops surface

- `GET /metrics` — Prometheus text format, scrape-ready.
- Structured JSON logs (`core/structured_logging.py`).
- Receipts carry measured energy and carbon figures.

## 13. Verify a release like a stranger

```sh
curl -sLO https://github.com/pluginfer/pluginfer/releases/latest/download/manifest.json
# verify manifest_signature (ECDSA P-256) against v2/build/release_pubkey.pem
# — the sorted-JSON body without the signature field is what's signed.
```

Every installer's SHA-256 is in the manifest; the manifest is signed;
the public key lives in the repo. No trust in us required.

## 14. Re-run the WAN proof

```sh
python pluginfer.py up --share          # note the https tunnel host
# then: Actions → wan-proof → Run workflow → bootstrap = <host>:443
```

A GitHub runner becomes a buyer-only stranger and clears a signed job
on your machine across the open internet.

## 15. Bitcoin-anchor the Signet audit trail (opt-in)

Signatures prove outsiders didn't edit the receipt log — but the
gateway operator holds the signing key. Anchoring closes that last gap
by publishing the chain head where even the operator can't unpublish
it:

```sh
set PLUGINFER_GW_ANCHOR=ots           # that's the whole setup
python -m governance.gateway
```

Every hour (tune with `PLUGINFER_GW_ANCHOR_INTERVAL_S`), if the chain
head moved, it is submitted to public OpenTimestamps calendar servers,
which batch it into a Bitcoin transaction. Anchor on demand with
`POST /v1/audit/anchor/now` (admin key). List anchors and download the
standard `.ots` proof files:

```sh
curl http://127.0.0.1:8788/v1/audit/anchors
curl -O http://127.0.0.1:8788/v1/audit/anchors/<anchor_id>/proof/0
```

Verify as a third party, with zero trust in the gateway:

```sh
pip install opentimestamps-client
ots upgrade proof.ots          # completes once Bitcoin-attested (~hours)
ots verify -d <chain_head_sha256> proof.ots
```

then confirm the same head at `GET /v1/receipts/verify`. Honest scope:
a fresh proof is *pending* until the calendars batch into Bitcoin;
anchoring is fail-open (calendar outages are journaled, spend
enforcement is never blocked); only the 32-byte head leaves your
network — no spend data. Airgapped? Leave it off; nothing changes.

## 16. Judge-gated cascade (widen safe savings)

The cascade (guide 4) accepts a cheap model answer only when no hard
failure signal fires. A judge model adds a substance check on top:

```sh
set PLUGINFER_GW_CASCADES=C:\path\to\cascades.json
set PLUGINFER_GW_CASCADE_JUDGE=gpt-4o-mini      # must be in your price sheet
set PLUGINFER_GW_CASCADE_JUDGE_THRESHOLD=7      # accept at score >= 7 (0-10)
set PLUGINFER_GW_CASCADE_JUDGE_ON_ERROR=escalate  # judge down -> escalate (default)
python -m governance.gateway
```

Flow per request: cheap model answers → hard signals check → judge
scores the answer against the request → accept (settle at cheap +
judge price; saving recorded signed) or escalate to the target model
(cheap + judge cost surfaced as negative saving). The judge's verdict
rides on every receipt.

**Measure before you trust it.** A judge is a model judging a model.
Curate a golden set of real prompts + answers you've labelled, then:

```sh
curl -X POST http://127.0.0.1:8788/v1/cascade/judge/golden \
  -H "X-Admin-Key: <admin>" -H "Content-Type: application/json" \
  -d "{\"items\": [{\"prompt\": \"2+2?\", \"answer\": \"4\", \"label\": \"accept\"},
                   {\"prompt\": \"2+2?\", \"answer\": \"5\", \"label\": \"escalate\"}]}"
```

The report gives agreement rate, **false accepts** (judge passed a bad
answer — the dangerous direction) and false escalates (judge burned
money on a good answer). Tune the threshold until false accepts are
acceptable for YOUR traffic, then enable in production.

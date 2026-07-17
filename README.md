# Pluginfer

> **Your AI bill, back under control — enforced, attributed, and provable.**
> A fail-closed spend gateway for OpenAI/Anthropic-compatible APIs that
> makes budget overruns *impossible* instead of merely visible, cuts the
> bill with measured (never projected) savings, and signs every call
> into a tamper-evident audit chain. Fully on-premises, one base-URL
> change, zero mesh required. The decentralized compute mesh it can
> route into is the second act — and we're honest about its status.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Honest status](https://img.shields.io/badge/claims-audited-brightgreen.svg)](AUDIT.md)

![30-second demo: budget cap refuses a call with HTTP 402, tamper-evident audit chain catches an edited receipt](docs/assets/signet-demo.gif)

## The problem

73% of enterprises overrun their AI budgets. Token spend broke the
management model: it isn't rate-limited, isn't attributable, and isn't
forecastable — and your AI vendor can't fix that, because their revenue
*is* your token growth. Every existing tool observes spend after the
fact. This one refuses the call **before** the money leaves.

## See it in 60 seconds (no API key needed)

```sh
git clone https://github.com/pluginfer/pluginfer && cd pluginfer/v2
pip install fastapi uvicorn httpx
python -m governance.demo_harness --serve
# open http://127.0.0.1:8799/dashboard
```

The demo drives a realistic traffic mix through the real gateway
against a simulated upstream: watch envelopes fill, a team hit its cap
and get refused with HTTP 402, cache hits land at $0, and the signed
receipt chain verify itself. Then hit `/demo/tamper` and watch the
audit badge catch the edit.

## Signet — the AI spend gateway that signs its receipts

Point your apps at the gateway instead of `api.openai.com` /
`api.anthropic.com` — one base-URL change; every OpenAI-compatible SDK
already supports it. Your provider key stays server-side.

- **Fail-closed budget envelopes** — hierarchical caps
  (`acme/support/chatbot`) reserved *before* the upstream call.
  Exhausted envelope → HTTP 402 with a machine-readable reason. An
  overrun is structurally impossible, not a dashboard alert.
- **Governed streaming** — hold taken up-front, live SSE relay, hard
  mid-stream cutoff the moment the running cost reaches the hold.
- **Measured savings, never projections** — exact-match + semantic
  response cache (a hit costs $0 and the receipt records what the
  upstream *actually billed* last time), opt-in cheap-model cascade
  with a conservative scorer, opt-in prompt compression. Escalation
  overhead is shown in red, subtracted from net — never hidden.
- **Signed, hash-chained receipts** — Ed25519 by default; each receipt
  embeds the previous one's hash and survives restarts. Any edit to
  history is caught at the exact receipt, verifiable by a third party
  with the public key alone (`/v1/receipts/verify`, `/v1/audit/anchor`
  for external anchoring). We deliberately do **not** call this a
  blockchain — one gateway is one writer.
- **Attribution** — spend by envelope, by model, and by API-key
  fingerprint (raw keys never stored). "The $455M went *where*?"
  becomes a query.
- **Auth that fails closed** — client keys (pinnable to an envelope),
  read keys, admin keys; startup refuses to bind a public interface
  with no auth configured.
- **A dashboard humans can read** — Plain-English, Technical, and
  Logs-&-audit views. Self-contained HTML, airgap-safe, light + dark.

The `governance/` package is deliberately standalone: pure
stdlib + FastAPI, no torch, no mesh — verified at ~600 ms / ~400
modules to import. Deploy it on your premises; nothing leaves your
network except your own upstream calls.

## The mesh (second act — read [AUDIT.md](AUDIT.md) first)

The longer bet: a sealed-bid compute marketplace where peer GPUs and
private datacenters bid against centralized APIs, with quorum
verification (K-of-N agreement across independent nodes) as the
honest mitigation for untrusted compute. The engineering is real and
tested (~1,100 tests) — and it has not yet cleared a real-WAN,
two-strangers run. We publish [AUDIT.md](AUDIT.md) so nobody has to
take our word for which claims are proven, mitigated, or open.

Run a node:

```sh
cd v2 && pip install -r requirements.txt
python pluginfer.py up
# Point ANY OpenAI client at your own node:
export OPENAI_BASE_URL=http://127.0.0.1:8100/v1
```

`up` detects your hardware, binds local models (real ones via
[Ollama](https://ollama.com) if present; an honestly-tagged echo
otherwise), joins a mesh if a seed is reachable, and self-supervises.
`python pluginfer.py up --seed-host <seed-ip>` joins a specific mesh.

### Testnet economics — stated up-front

The mesh currently runs **testnet economics**, and every money endpoint
says so in its response. What that means, so it can never be quietly
walked back:

- Earnings and commissions accrue as **real, persistent, auditable
  accounting** (`/v1/ledger/wallets/{id}`, `/v1/ledger/treasury`) — but
  they are **not redeemable for cash** during testnet.
- Testnet balances are **preserved**. Whether and how they are
  recognized at mainnet will be announced *before* mainnet opens —
  never decided retroactively.
- Real deposits and payouts are **disabled at the endpoint level**
  during testnet (a mis-set payment key cannot charge anyone). The
  cash rails (exactly-once deposits, two-phase withdrawals) are built
  and tested in `core/payment_flows.py`; flipping to mainnet is an
  explicit operator act requiring a real payment gateway.
- Payouts will only ever come from **real buyer payments** — provider
  earnings are never subsidized from a treasury and there are no token
  emissions.
- **Trying the mesh is free during testnet.** There is nothing to buy:
  `POST /v1/testnet/faucet {"wallet_id": "you"}` grants a one-time
  starter balance (default $25 test-USD, once per wallet) so anyone
  can run real jobs through the real auction/escrow/commission
  machinery at zero cost. The faucet refuses outright in mainnet mode.
- **There is no token — at mainnet either.** The ledger is denominated
  in plain USD. When mainnet opens, buyers deposit real money through
  a payment processor and providers withdraw real money; onboarding is
  as exotic as topping up a cloud account. The bootstrap sequence is
  deliberately usage-first: prove the loop on testnet → free real
  demand via faucet credits → priced demand from Signet receipts →
  fiat deposits open last, when there's something real to pay for.

## Honesty policy

Every money- or trust-claim in this repo is either tested, labelled an
estimate, or listed as open in [AUDIT.md](AUDIT.md). Savings are
reported only as measured counterfactuals from receipts. If you catch
us over-claiming, open an issue — that's a bug with the same severity
as a crash.

## Contributing

Bug reports, security advisories, PRs welcome — see
[docs/SECURITY.md](docs/SECURITY.md#responsible-disclosure). Run the
suite before a PR:

```sh
cd v2 && python -m pytest tests/ -q
```

## License

Apache 2.0. See [LICENSE](LICENSE).

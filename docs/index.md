# Pluginfer

Pluginfer is a peer-to-peer GPU compute marketplace with crypto-native
economics. Two strangers on different home networks can share compute
by each downloading and running a signed installer; the chain settles
payment and a verifiable provenance ticket proves the result wasn't
tampered with.

- [Quickstart](quickstart.md) -- 5 minutes from zero to a signed
  receipt
- [Architecture](architecture.md) -- the full system design
- [API reference](api-reference.md) -- REST endpoints
- [Security](SECURITY.md) -- threat model + audit posture
- [Signing setup](SIGNING_SETUP.md) -- code-signing certs you need
  before launch
  slack-aware auction, provider auction)

## Why

Centralised LLM API providers run a vertical stack: their hardware,
their model, their margin. Pluginfer's auction puts peer GPUs running
open-weight models (in users' off-peak hours) against those APIs, on
identical cost / latency / privacy / quality constraints set by the
caller. As the mesh grows, routine workloads route to the cheapest
qualifying provider -- often a 4090 in someone's home, underbidding
centralised APIs by 5-10x.

This is the cost-dynamics shift Pluginfer is targeting.

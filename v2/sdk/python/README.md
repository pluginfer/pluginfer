# pluginfer

Python SDK for [Pluginfer](https://pluginfer.network) — the distributed
AI compute mesh.

## Install

```sh
pip install pluginfer
```

## Quickstart

```python
from pluginfer import Pluginfer

with Pluginfer(api_key="pf_live_...", base_url="https://api.pluginfer.network") as p:
    # Submit a job, then stream events until done.
    job = p.jobs.submit(
        kind="llm.completion",
        payload={"prompt": "Say hi.", "max_tokens": 32},
        cost_ceiling_usd=0.05,
        latency_ceiling_ms=10_000,
    )
    print(job.job_id, job.state.state)

    for event in p.jobs.stream(job.job_id):
        print(event["event"])

    result = p.jobs.result(job.job_id)
    print(p.jobs.decode_result(result))
```

## Wallet-signature auth (no API key)

```python
from pluginfer import Pluginfer
from pluginfer_node_sdk.wallet import Wallet  # any obj with .public_key_pem and .sign(bytes)

p = Pluginfer(base_url="https://api.pluginfer.network")
sid = p.auth.login_with_wallet(Wallet.load_or_create())
# sid is now stashed; subsequent SDK calls are authed.
```

## License

Apache 2.0.

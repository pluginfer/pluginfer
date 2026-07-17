# @pluginfer/sdk

JavaScript / TypeScript SDK for [Pluginfer](https://pluginfer.network) —
the distributed AI compute mesh.

## Install

```sh
npm install @pluginfer/sdk
```

## Usage

```ts
import { Pluginfer } from "@pluginfer/sdk";

const p = new Pluginfer({
  apiKey: "pf_live_...",
  baseUrl: "https://api.pluginfer.network",
});

const job = await p.jobs.submit({
  kind: "llm.completion",
  payload: { prompt: "Say hi.", max_tokens: 32 },
  cost_ceiling_usd: 0.05,
  latency_ceiling_ms: 10_000,
});

const result = await p.jobs.result(job.job_id);
console.log(result.result_b64);
```

## License

Apache 2.0.

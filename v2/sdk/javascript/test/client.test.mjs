// JS SDK contract tests — run against the BUILT artifact (dist/), so a
// broken build or wrong package entry point fails here, not on a user.
// Plain node:test + a stub HTTP server; no framework dependencies.
import assert from "node:assert/strict";
import http from "node:http";
import { after, before, test } from "node:test";

const { Pluginfer, AuthenticationError, JobNotFoundError, RateLimitError } =
  await import("../dist/index.js");

let server;
let base;
const seen = [];

before(async () => {
  server = http.createServer((req, res) => {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => {
      seen.push({ method: req.method, url: req.url,
                  auth: req.headers["authorization"],
                  session: req.headers["x-pluginfer-session"],
                  body: body ? JSON.parse(body) : null });
      const route = `${req.method} ${req.url}`;
      const reply = (code, obj, headers = {}) => {
        res.writeHead(code, { "Content-Type": "application/json",
                              "X-Request-ID": "req-1", ...headers });
        res.end(JSON.stringify(obj));
      };
      if (route === "POST /v1/jobs")
        return reply(200, { job_id: "j1", kind: "compute.echo",
                            state: { state: "queued" } });
      if (route === "GET /v1/jobs/j1")
        return reply(200, { job_id: "j1", state: { state: "completed" } });
      if (route === "GET /v1/jobs/missing") return reply(404, { error: "nope" });
      if (route === "GET /v1/status") return reply(200, { ok: true });
      if (route === "GET /v1/wallet/balance") return reply(401, { error: "auth" });
      if (route === "GET /v1/providers")
        return reply(429, { error: "slow down" }, { "Retry-After": "7" });
      reply(500, { error: "unexpected route " + route });
    });
  });
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  base = `http://127.0.0.1:${server.address().port}`;
});

after(() => server.close());

test("submit + get round-trip with auth and session headers", async () => {
  const c = new Pluginfer({ baseUrl: base, apiKey: "pf_test_key" });
  c.setSession("sess-9");
  const j = await c.jobs.submit({ kind: "compute.echo", payload: { x: 1 } });
  assert.equal(j.job_id, "j1");
  const got = await c.jobs.get("j1");
  assert.equal(got.state.state, "completed");
  const submitReq = seen.find((r) => r.url === "/v1/jobs");
  assert.equal(submitReq.auth, "Bearer pf_test_key");
  assert.equal(submitReq.session, "sess-9");
  assert.deepEqual(submitReq.body, { kind: "compute.echo", payload: { x: 1 } });
});

test("404 maps to JobNotFoundError with request id", async () => {
  const c = new Pluginfer({ baseUrl: base });
  await assert.rejects(() => c.jobs.get("missing"), (e) => {
    assert.ok(e instanceof JobNotFoundError);
    assert.equal(e.status, 404);
    assert.equal(e.requestId, "req-1");
    return true;
  });
});

test("401 maps to AuthenticationError", async () => {
  const c = new Pluginfer({ baseUrl: base });
  await assert.rejects(() => c.wallet.balance(),
    (e) => e instanceof AuthenticationError && e.status === 401);
});

test("429 maps to RateLimitError carrying Retry-After", async () => {
  const c = new Pluginfer({ baseUrl: base });
  await assert.rejects(() => c.providers.list(), (e) => {
    assert.ok(e instanceof RateLimitError);
    assert.equal(e.retryAfterSec, 7);
    return true;
  });
});

test("CommonJS entry point resolves too", async () => {
  // Guards the dist-cjs half of the dual package — a user on require()
  // must get the same client.
  const { createRequire } = await import("node:module");
  const require_ = createRequire(import.meta.url);
  const cjs = require_("../dist-cjs/index.js");
  assert.equal(typeof cjs.Pluginfer, "function");
});

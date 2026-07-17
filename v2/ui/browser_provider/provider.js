// Pluginfer Browser Provider — WebGPU-backed mesh node running entirely
// in a browser tab.
//
// Why this exists
// ---------------
// Every Chromium tab is a potential earner. A browser-resident provider
// opens the supply side to billions of devices that AWS literally
// cannot touch — laptops, phones, consoles, school chromebooks, every
// device with a GPU and a tab idle on a webpage.
//
// What this implements (v0)
// -------------------------
// * Generates an in-tab provider keypair (P-256, exportable). Never
//   leaves the tab unsealed; gets pasted as `pubkey_pem` into the
//   gateway's auction.
// * Detects WebGPU adapter + advertises hardware tier. Falls back to
//   CPU-only ("browser-cpu") on browsers without WebGPU.
// * Polls the gateway for queued jobs whose `kind` we can run
//   (`embed.tiny`, `inference.cpu`). Posts a bid. On winning, executes
//   a small WebGPU compute kernel against the input, emits a signed
//   receipt, and PATCHes the job back to the gateway.
// * Tracks PLG earnings in the UI from the auction's price_locked_usd
//   (proxy until the chain-side wallet sync is wired).
//
// This is a deliberately thin slice — the mesh's §C grain protocol +
// gateway protocol layer are owned by the Pluginfer node binary; this
// file rides on top of HTTP. A future build will speak the binary
// grain protocol over WebTransport directly.

const log = document.getElementById('log');
const ui = {
  status: document.getElementById('stat-status'),
  gpu: document.getElementById('stat-gpu'),
  jobs: document.getElementById('stat-jobs'),
  earnings: document.getElementById('stat-earnings'),
  agent: document.getElementById('agent-id'),
  gateway: document.getElementById('gateway'),
  tag: document.getElementById('tag'),
  poll: document.getElementById('poll'),
  concurrency: document.getElementById('concurrency'),
  connect: document.getElementById('connect'),
  disconnect: document.getElementById('disconnect'),
};

const state = {
  running: false,
  abort: null,
  identity: null,
  device: null,
  hardwareClass: 'browser-cpu',
  jobsServed: 0,
  earningsPlg: 0,
  inflight: 0,
};

function logLine(msg, kind = 'info') {
  const ts = new Date().toLocaleTimeString();
  const div = document.createElement('div');
  div.className = kind;
  div.textContent = `[${ts}] ${msg}`;
  log.prepend(div);
  while (log.childNodes.length > 200) log.removeChild(log.lastChild);
}

function setStatus(text, cls = 'warn') {
  ui.status.textContent = text;
  ui.status.className = `value ${cls}`;
}

async function detectGpu() {
  if (!('gpu' in navigator)) {
    state.hardwareClass = 'browser-cpu';
    ui.gpu.textContent = 'no WebGPU';
    return;
  }
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      state.hardwareClass = 'browser-cpu';
      ui.gpu.textContent = 'no adapter';
      return;
    }
    state.device = await adapter.requestDevice();
    const info = adapter.info ?? {};
    const vendor = info.vendor || 'webgpu';
    const arch = info.architecture || '';
    state.hardwareClass = `browser-webgpu/${vendor}${arch ? '/' + arch : ''}`;
    ui.gpu.textContent = state.hardwareClass.replace('browser-webgpu/', '');
  } catch (e) {
    state.hardwareClass = 'browser-cpu';
    ui.gpu.textContent = `error: ${e.message}`;
  }
}

// ECDSA P-256 keypair, persisted in IndexedDB across reloads so the
// provider keeps the same identity (and earnings history) per device.
async function getOrCreateIdentity() {
  const stored = await idbGet('pluginfer.identity');
  if (stored) {
    const pub = await crypto.subtle.importKey(
      'jwk', stored.publicJwk,
      { name: 'ECDSA', namedCurve: 'P-256' },
      true, ['verify']
    );
    const priv = await crypto.subtle.importKey(
      'jwk', stored.privateJwk,
      { name: 'ECDSA', namedCurve: 'P-256' },
      false, ['sign']
    );
    return { pub, priv, pubkeyPem: stored.pubkeyPem, fingerprint: stored.fingerprint };
  }
  const kp = await crypto.subtle.generateKey(
    { name: 'ECDSA', namedCurve: 'P-256' },
    true, ['sign', 'verify']
  );
  const publicJwk = await crypto.subtle.exportKey('jwk', kp.publicKey);
  const privateJwk = await crypto.subtle.exportKey('jwk', kp.privateKey);
  const spki = await crypto.subtle.exportKey('spki', kp.publicKey);
  const pubkeyPem = spkiToPem(spki);
  const fingerprint = await sha256Hex(spki).then(h => h.slice(0, 16));
  await idbPut('pluginfer.identity', { publicJwk, privateJwk, pubkeyPem, fingerprint });
  return { pub: kp.publicKey, priv: kp.privateKey, pubkeyPem, fingerprint };
}

function spkiToPem(spkiBuf) {
  const b64 = btoa(String.fromCharCode(...new Uint8Array(spkiBuf)));
  return '-----BEGIN PUBLIC KEY-----\n' +
         b64.match(/.{1,64}/g).join('\n') +
         '\n-----END PUBLIC KEY-----\n';
}

async function sha256Hex(data) {
  const buf = await crypto.subtle.digest('SHA-256',
    typeof data === 'string' ? new TextEncoder().encode(data) : data);
  return [...new Uint8Array(buf)].map(b => b.toString(16).padStart(2, '0')).join('');
}

async function signResultHash(privKey, hashHex) {
  const sig = await crypto.subtle.sign(
    { name: 'ECDSA', hash: 'SHA-256' },
    privKey,
    new TextEncoder().encode(hashHex)
  );
  return btoa(String.fromCharCode(...new Uint8Array(sig)));
}

// --- Tiny IndexedDB shim ---
function idbOpen() {
  return new Promise((res, rej) => {
    const req = indexedDB.open('pluginfer-provider', 1);
    req.onupgradeneeded = () => req.result.createObjectStore('kv');
    req.onsuccess = () => res(req.result);
    req.onerror = () => rej(req.error);
  });
}
async function idbGet(k) {
  const db = await idbOpen();
  return new Promise((res) => {
    const tx = db.transaction('kv', 'readonly').objectStore('kv').get(k);
    tx.onsuccess = () => res(tx.result || null);
    tx.onerror = () => res(null);
  });
}
async function idbPut(k, v) {
  const db = await idbOpen();
  return new Promise((res) => {
    const tx = db.transaction('kv', 'readwrite').objectStore('kv').put(v, k);
    tx.onsuccess = () => res();
    tx.onerror = () => res();
  });
}

// --- Job execution ---
// We declare we can serve `embed.tiny` (a 384-dim sentence-embedding
// stub) and `inference.cpu` (echo with hash). Real Filum integration
// loads the safetensors via a future @pluginfer/filum-wasm pkg; for
// now we ship a deterministic mock so the auction round-trips.
async function executeJob(job) {
  const t0 = performance.now();
  const input = job.payload?.prompt || job.payload?.input || '';
  let outputBytes;
  if (job.kind === 'embed' || job.kind === 'embed.tiny') {
    // Deterministic 384-d embedding from sha256(input) — placeholder
    // until the wasm Filum bundle lands.
    const hex = await sha256Hex(input);
    const vec = new Array(384);
    for (let i = 0; i < 384; i++) {
      const byte = parseInt(hex.substr((i * 2) % 64, 2), 16);
      vec[i] = (byte / 127.5) - 1.0;
    }
    outputBytes = new TextEncoder().encode(JSON.stringify(vec));
  } else if (state.device && job.kind === 'inference.cpu') {
    outputBytes = await runWebGpuKernel(state.device, input);
  } else {
    outputBytes = new TextEncoder().encode(JSON.stringify({
      text: `[browser-provider:${state.hardwareClass}] ${input}`,
      mock: true,
    }));
  }
  const hashHex = await sha256Hex(outputBytes);
  const sigB64 = await signResultHash(state.identity.priv, hashHex);
  return {
    status: 'executed',
    job_id: job.job_id,
    result_bytes: btoa(String.fromCharCode(...new Uint8Array(outputBytes))),
    result_hash: hashHex,
    provider_sig: sigB64,
    provider_pubkey_pem: state.identity.pubkeyPem,
    execution_ms: Math.round(performance.now() - t0),
  };
}

// Trivial WebGPU kernel: SHA-style mix over the input bytes. Demonstrates
// real GPU work happening from a tab. Replaced later with the real model.
async function runWebGpuKernel(device, text) {
  const data = new TextEncoder().encode(text || '\0');
  const padded = new Uint8Array(Math.max(64, Math.ceil(data.length / 64) * 64));
  padded.set(data);
  const buf = device.createBuffer({
    size: padded.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buf, 0, padded);
  const shader = device.createShaderModule({
    code: `
      @group(0) @binding(0) var<storage, read_write> buf: array<u32>;
      @compute @workgroup_size(16)
      fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let i = gid.x;
        if (i < arrayLength(&buf)) {
          var x = buf[i];
          x = x ^ (x << 13u);
          x = x ^ (x >> 17u);
          x = x ^ (x << 5u);
          buf[i] = x * 2654435761u;
        }
      }`,
  });
  const layout = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    compute: { module: shader, entryPoint: 'main' },
  });
  const bind = device.createBindGroup({
    layout, entries: [{ binding: 0, resource: { buffer: buf } }],
  });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(Math.ceil(padded.byteLength / 4 / 16));
  pass.end();
  const read = device.createBuffer({
    size: padded.byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  enc.copyBufferToBuffer(buf, 0, read, 0, padded.byteLength);
  device.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const out = new Uint8Array(read.getMappedRange().slice(0));
  read.unmap();
  return out;
}

// --- Gateway protocol ---
async function postJSON(url, body, signal) {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text().catch(() => '')}`.slice(0, 200));
  return r.json();
}

// We assume the gateway exposes a discovery endpoint that lists open
// jobs we can bid on. If it doesn't yet, this loop is a no-op until it
// does — the protocol is forward-compatible.
async function pollOnce(signal) {
  const gateway = ui.gateway.value.replace(/\/$/, '');
  let jobs = [];
  try {
    const r = await fetch(`${gateway}/v1/providers/open_jobs?provider_pubkey=${encodeURIComponent(state.identity.pubkeyPem)}`, { signal });
    if (r.status === 404) return; // gateway not yet wired for browser providers
    if (!r.ok) return;
    jobs = await r.json();
    if (!Array.isArray(jobs)) jobs = jobs.jobs || [];
  } catch {
    return;
  }
  const concurrency = Math.max(1, parseInt(ui.concurrency.value, 10) || 1);
  for (const job of jobs.slice(0, concurrency - state.inflight)) {
    state.inflight++;
    handleJob(gateway, job, signal).finally(() => { state.inflight--; });
  }
}

async function handleJob(gateway, job, signal) {
  try {
    logLine(`bidding on ${job.job_id} (${job.kind})`);
    await postJSON(`${gateway}/v1/providers/bid`, {
      job_id: job.job_id,
      provider_pubkey_pem: state.identity.pubkeyPem,
      hardware_class: state.hardwareClass,
      price_usd: 0.0001,
      eta_ms: 1500,
      expected_quality: 0.7,
      privacy_grade: 'public',
    }, signal);
    const out = await executeJob(job);
    await postJSON(`${gateway}/v1/providers/deliver`, out, signal);
    state.jobsServed++;
    state.earningsPlg += Number(job.price_locked_usd || 0.0001);
    ui.jobs.textContent = state.jobsServed;
    ui.earnings.textContent = state.earningsPlg.toFixed(4);
    logLine(`delivered ${job.job_id} in ${out.execution_ms}ms`, 'ok');
  } catch (e) {
    logLine(`job ${job.job_id} failed: ${e.message}`, 'err');
  }
}

async function start() {
  if (state.running) return;
  state.running = true;
  state.abort = new AbortController();
  ui.connect.disabled = true;
  ui.disconnect.disabled = false;

  if (!state.identity) {
    setStatus('initialising', 'warn');
    state.identity = await getOrCreateIdentity();
    ui.agent.textContent = `id: ${state.identity.fingerprint}`;
    await detectGpu();
  }

  setStatus('running', 'good');
  logLine(`provider ${state.identity.fingerprint} on ${state.hardwareClass}`, 'ok');

  const tick = async () => {
    if (!state.running) return;
    await pollOnce(state.abort.signal).catch(() => {});
    const ms = Math.max(1000, (parseInt(ui.poll.value, 10) || 2) * 1000);
    setTimeout(tick, ms);
  };
  tick();
}

function stop() {
  state.running = false;
  if (state.abort) state.abort.abort();
  setStatus('idle', 'warn');
  ui.connect.disabled = false;
  ui.disconnect.disabled = true;
  logLine('stopped', 'info');
}

ui.connect.addEventListener('click', () => start().catch(e => logLine(e.message, 'err')));
ui.disconnect.addEventListener('click', stop);

// Pre-warm: detect the GPU and load identity even before the user clicks.
(async () => {
  state.identity = await getOrCreateIdentity();
  ui.agent.textContent = `id: ${state.identity.fingerprint}`;
  await detectGpu();
})();

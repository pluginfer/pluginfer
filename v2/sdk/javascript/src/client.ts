import {
  Job,
  JobCreate,
  JobResult,
  Provider,
  Status,
  WalletBalance,
} from "./types";
import {
  AuthenticationError,
  JobNotFoundError,
  PluginferError,
  RateLimitError,
} from "./errors";

export interface PluginferOptions {
  apiKey?: string;
  baseUrl?: string;
  timeoutMs?: number;
  fetch?: typeof fetch;
}

const DEFAULT_BASE = "https://api.pluginfer.network";

export class Pluginfer {
  private baseUrl: string;
  private apiKey?: string;
  private session?: string;
  private timeoutMs: number;
  private _fetch: typeof fetch;

  constructor(opts: PluginferOptions = {}) {
    this.baseUrl = (opts.baseUrl ?? DEFAULT_BASE).replace(/\/+$/, "");
    this.apiKey = opts.apiKey;
    this.timeoutMs = opts.timeoutMs ?? 30_000;
    this._fetch = opts.fetch ?? globalThis.fetch;
    if (!this._fetch) {
      throw new Error("fetch is not available; pass one via opts.fetch on Node < 18");
    }
  }

  setSession(sessionId: string): void {
    this.session = sessionId;
  }

  private headers(extra: Record<string, string> = {}): Record<string, string> {
    const h: Record<string, string> = {
      "User-Agent": "pluginfer-js/1.0.0",
      "Content-Type": "application/json",
      ...extra,
    };
    if (this.apiKey) h["Authorization"] = `Bearer ${this.apiKey}`;
    if (this.session) h["X-Pluginfer-Session"] = this.session;
    return h;
  }

  private async req<T>(method: string, path: string, body?: unknown): Promise<T> {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), this.timeoutMs);
    try {
      const res = await this._fetch(`${this.baseUrl}${path}`, {
        method,
        headers: this.headers(),
        body: body !== undefined ? JSON.stringify(body) : undefined,
        signal: ctl.signal,
      });
      const reqId = res.headers.get("X-Request-ID") ?? undefined;
      let parsed: unknown;
      const txt = await res.text();
      try { parsed = txt ? JSON.parse(txt) : {}; } catch { parsed = txt; }

      if (res.status === 401) {
        throw new AuthenticationError("authentication_failed", { status: 401, requestId: reqId, body: parsed });
      }
      if (res.status === 404) {
        throw new JobNotFoundError("not_found", { status: 404, requestId: reqId, body: parsed });
      }
      if (res.status === 429) {
        const retryAfterSec = Number(res.headers.get("Retry-After") ?? 1);
        throw new RateLimitError("rate_limited", { retryAfterSec, status: 429, requestId: reqId, body: parsed });
      }
      if (res.status < 200 || res.status >= 300) {
        throw new PluginferError(`http_${res.status}`, { status: res.status, requestId: reqId, body: parsed });
      }
      return parsed as T;
    } finally {
      clearTimeout(t);
    }
  }

  // ----- jobs ----------------------------------------------------------
  jobs = {
    submit: (b: JobCreate): Promise<Job> => this.req("POST", "/v1/jobs", b),
    get: (id: string): Promise<Job> => this.req("GET", `/v1/jobs/${id}`),
    result: (id: string): Promise<JobResult> => this.req("GET", `/v1/jobs/${id}/result`),
    cancel: (id: string): Promise<{ job_id: string; cancelled: boolean; state: string }> =>
      this.req("DELETE", `/v1/jobs/${id}`),
  };

  // ----- wallet --------------------------------------------------------
  wallet = {
    balance: (): Promise<WalletBalance> => this.req("GET", "/v1/wallet/balance"),
  };

  // ----- providers -----------------------------------------------------
  providers = {
    list: (): Promise<Provider[]> => this.req("GET", "/v1/providers"),
  };

  // ----- status / version ---------------------------------------------
  status(): Promise<Status> { return this.req("GET", "/v1/status"); }
  version(): Promise<{ version: string; git_sha: string; api: string }> {
    return this.req("GET", "/v1/version");
  }

  // ----- auth ----------------------------------------------------------
  auth = {
    issueChallenge: (): Promise<{ nonce: string; issued_at_unix: number; expires_at_unix: number; audience: string }> =>
      this.req("POST", "/v1/auth/challenge", {}),
    verify: (b: { nonce: string; pubkey_pem: string; signature_b64: string }): Promise<{ session_id: string; session_header: string }> =>
      this.req("POST", "/v1/auth/verify", b),
  };
}

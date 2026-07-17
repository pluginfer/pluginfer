export class PluginferError extends Error {
  status?: number;
  requestId?: string;
  body?: unknown;

  constructor(message: string, opts: { status?: number; requestId?: string; body?: unknown } = {}) {
    super(message);
    this.name = "PluginferError";
    this.status = opts.status;
    this.requestId = opts.requestId;
    this.body = opts.body;
  }
}

export class AuthenticationError extends PluginferError {
  constructor(message: string, opts = {}) { super(message, opts); this.name = "AuthenticationError"; }
}

export class JobNotFoundError extends PluginferError {
  constructor(message: string, opts = {}) { super(message, opts); this.name = "JobNotFoundError"; }
}

export class RateLimitError extends PluginferError {
  retryAfterSec: number;
  constructor(message: string, opts: { retryAfterSec?: number; status?: number; requestId?: string; body?: unknown } = {}) {
    super(message, opts);
    this.name = "RateLimitError";
    this.retryAfterSec = opts.retryAfterSec ?? 1.0;
  }
}

// Wire-format types shared across the JS SDK.

export type JobStateName =
  | "queued"
  | "matched"
  | "running"
  | "completed"
  | "failed"
  | "timeout"
  | "cancelled";

export interface JobState {
  state: JobStateName;
  detail?: string | null;
}

export interface Job {
  job_id: string;
  kind: string;
  state: JobState;
  submitted_at_unix: number;
  matched_provider_pubkey?: string | null;
  price_locked_usd?: number | null;
  cost_ceiling_usd: number;
  privacy_class: string;
}

export interface JobResult {
  job_id: string;
  state: JobState;
  result_b64?: string | null;
  result_hash_hex?: string | null;
  provider_signature_b64?: string | null;
  execution_ms?: number | null;
}

export interface JobCreate {
  kind: string;
  payload?: Record<string, unknown>;
  cost_ceiling_usd?: number;
  latency_ceiling_ms?: number;
  privacy_class?: "public" | "internal" | "confidential";
  quality_floor?: number;
  webhook_url?: string | null;
}

export interface WalletBalance {
  address: string;
  balance_plg: number;
  pending_plg: number;
  chain_height: number;
}

export interface Provider {
  pubkey: string;
  kind: string;
  quality_score: number;
  region: string;
}

export interface Status {
  status: string;
  version: string;
  git_sha: string;
  chain_height: number;
  peers_connected: number;
  uptime_seconds: number;
}

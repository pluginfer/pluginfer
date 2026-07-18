"""Governance Gateway — Budget-as-Contract for TRADITIONAL AI setups
(§RFC-3 / HG13b, streaming HG13c, thrift stack HG13e).

The zero-migration enterprise product: an org points its apps at THIS
endpoint instead of api.openai.com / api.anthropic.com (one base-URL
change — the drop-in trick every OpenAI-compatible tool supports) and
every call is governed on the way through:

  * **Fail-closed budget envelopes** (governance.budget_ledger): the request
    is priced BEFORE it leaves for the upstream; an exhausted envelope
    gets HTTP 402 with an honest machine-readable reason — overruns
    become impossible, not dashboards. Envelope selection: the
    ``X-Budget-Envelope`` header; hierarchical caps all bind.
  * **Receipts per call**: envelope, model, real token usage from the
    upstream's own ``usage`` block, cost at the configured price sheet,
    request/response hashes — signed by the gateway wallet when one is
    attached. Chargeback = GET /v1/budget/report; savings =
    GET /v1/savings; human view = GET /dashboard.
  * **Quote-before-run**: POST /v1/quote prices a body without
    forwarding it.

The thrift stack — the part that makes calls CHEAPER, not just
governed. Savings are MEASURED counterfactuals, never projections:

  * **Response cache** (exact-match): a byte-identical repeat of a
    previously answered request is served from the gateway at $0
    upstream cost; ``saved_usd`` on the receipt is what the upstream
    actually billed for that same request last time. Deterministic
    requests only by default (temperature==0 / unset-sampling), because
    serving a cached answer to a sampling request changes semantics —
    ``cache_all=True`` is the operator's explicit opt-in to cache
    everything. This is NOT a semantic (similarity) cache; that needs
    an embedder and is a separate work item (HG13f) — exact-match is
    the honest zero-quality-risk tier and, per the traffic studies in
    research/01_token_thrift.md, exact repeats alone are a large share
    of production traffic.
  * **Cascade routing** (FrugalGPT-shaped, OPT-IN per model): a request
    for an expensive model is first tried on a configured cheaper
    model; a scorer decides accept/escalate. ``saved_usd`` = the price
    difference AT THE MEASURED USAGE when accepted; a rejected try is
    recorded as a NEGATIVE saving (the extra cheap-model spend is real
    money and the ledger never hides it). The built-in scorer is
    deliberately conservative (hard failure signals only — empty
    content, truncation, refusal-shaped or error-shaped output);
    nothing cascades unless the operator configured that model.

Streaming (HG13c): governed, not refused. The hold is taken up-front
from max_tokens; chunks pass through live; the gateway counts output
as it flows and HARD-CUTS the stream (SSE error event, then [DONE])
if the running estimate hits the held amount — so an unbounded stream
cannot outspend its envelope. Settlement uses the usage block most
APIs emit at stream end (OpenAI: ``stream_options.include_usage`` is
injected; Anthropic: message_start/message_delta usage); when the
upstream sends none, we settle at the estimate and the receipt says
``estimated: true``. Streamed responses can't carry a final cost
header (cost is unknown at header time) — the receipt id header is
pre-allocated and the receipt lands in /v1/receipts at stream end.

Honesty rails, unchanged: the price sheet is REQUIRED config (unpriced
model = 400 — we refuse to guess money numbers); upstream errors
release the hold; every number on /dashboard traces to a receipt.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import secrets
import threading
import time
import urllib.error
import urllib.request
from typing import (
    Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple,
)

from fastapi import FastAPI
from fastapi import Request as FastApiRequest
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

logger = logging.getLogger("pluginfer.governance_gateway")

DEFAULT_MAX_OUTPUT_TOKENS = 1024
DEFAULT_CACHE_TTL_S = 300.0
_PASSTHROUGH_HEADERS = (
    "authorization", "x-api-key", "anthropic-version", "openai-beta",
)

# (url, body_bytes, headers, timeout_s) -> (status_code, response_bytes)
HttpPost = Callable[[str, bytes, Dict[str, str], float], Tuple[int, bytes]]
# (url, body_bytes, headers, timeout_s) -> (status_code, iterator of raw
# SSE byte chunks). The iterator is consumed in a worker thread
# (Starlette runs sync generators via its threadpool).
HttpStream = Callable[[str, bytes, Dict[str, str], float],
                      Tuple[int, Iterable[bytes]]]


def _ua(headers: Dict[str, str]) -> Dict[str, str]:
    """Provider WAFs (Cloudflare et al.) reject Python's default
    User-Agent outright — seen live as an HTML 403 from Cerebras. Send
    an honest product UA unless the caller set their own."""
    if not any(k.lower() == "user-agent" for k in headers):
        headers = {**headers, "User-Agent": "pluginfer-signet/1.0"}
    return headers


def _default_http_post(url: str, body: bytes, headers: Dict[str, str],
                       timeout_s: float) -> Tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=_ua(headers),
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return int(r.status), r.read()
    except urllib.error.HTTPError as e:
        # Upstream 4xx/5xx bodies carry the provider's own error JSON —
        # surface them verbatim instead of collapsing to a bare 502.
        return int(e.code), e.read()


def _default_http_stream(url: str, body: bytes,
                         headers: Dict[str, str],
                         timeout_s: float
                         ) -> Tuple[int, Iterable[bytes]]:
    req = urllib.request.Request(url, data=body, headers=_ua(headers),
                                 method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=timeout_s)
    except urllib.error.HTTPError as e:
        return int(e.code), iter([e.read()])

    def _iter() -> Iterator[bytes]:
        try:
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    return
                yield chunk
        finally:
            resp.close()

    return int(resp.status), _iter()


def _estimate_input_tokens(body: Dict[str, Any]) -> int:
    """chars/4 heuristic over every text part — an ESTIMATE for the
    pre-flight hold only; settlement uses the upstream's real usage."""
    chars = 0
    for m in body.get("messages", []) or []:
        c = m.get("content")
        if isinstance(c, str):
            chars += len(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict):
                    chars += len(str(part.get("text", "")))
    sysprompt = body.get("system")
    if isinstance(sysprompt, str):
        chars += len(sysprompt)
    return max(1, chars // 4)


# ---------------------------------------------------------------------------
# Thrift: exact-match response cache
# ---------------------------------------------------------------------------

class ResponseCache:
    """Exact-match cache: canonical-JSON key over the WHOLE request
    body, so any difference (model, one character of prompt,
    max_tokens) is a different entry. Stores what the upstream billed,
    so a hit's saved_usd is a measured counterfactual, not a model."""

    def __init__(self, *, ttl_s: float = DEFAULT_CACHE_TTL_S,
                 cache_all: bool = False,
                 clock: Callable[[], float] = time.time):
        self.ttl_s = float(ttl_s)
        self.cache_all = cache_all
        self._clock = clock
        self._lock = threading.Lock()
        self._store: Dict[str, Tuple[float, Dict[str, Any], float]] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(body: Dict[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(body, sort_keys=True, default=str).encode()
        ).hexdigest()

    def cacheable(self, body: Dict[str, Any]) -> bool:
        if self.ttl_s <= 0 or body.get("stream"):
            return False
        if self.cache_all:
            return True
        # Deterministic requests only: sampling params absent or zeroed.
        temp = body.get("temperature")
        return temp == 0 or temp == 0.0

    def get(self, body: Dict[str, Any]
            ) -> Optional[Tuple[Dict[str, Any], float]]:
        if not self.cacheable(body):
            return None
        with self._lock:
            hit = self._store.get(self.key(body))
            if hit is None or self._clock() > hit[0]:
                self.misses += 1
                return None
            self.hits += 1
            return hit[1], hit[2]

    def put(self, body: Dict[str, Any], resp: Dict[str, Any],
            billed_usd: float) -> None:
        if not self.cacheable(body):
            return
        with self._lock:
            self._store[self.key(body)] = (
                self._clock() + self.ttl_s, resp, float(billed_usd))


# ---------------------------------------------------------------------------
# Thrift: conservative cascade scorer
# ---------------------------------------------------------------------------

_REFUSAL_PREFIXES = (
    "i can't", "i cannot", "i'm sorry", "i am sorry", "i'm unable",
    "i am unable", "as an ai", "i won't",
)


def _extract_text(resp: Dict[str, Any]) -> str:
    # OpenAI shape.
    choices = resp.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") or {}
        return str(msg.get("content") or "")
    # Anthropic shape.
    content = resp.get("content")
    if isinstance(content, list):
        return "".join(str(p.get("text", "")) for p in content
                       if isinstance(p, dict))
    return ""


def _finish_reason(resp: Dict[str, Any]) -> str:
    choices = resp.get("choices")
    if isinstance(choices, list) and choices:
        return str(choices[0].get("finish_reason") or "")
    return str(resp.get("stop_reason") or "")


def cascade_accept(resp: Dict[str, Any]) -> Tuple[bool, str]:
    """Deliberately conservative: only HARD failure signals reject a
    cheap-model answer. Anything subtler (factuality, style) is beyond
    an honest heuristic — operators wanting a judge model configure
    one when HG13g lands. Returns (accept, reason)."""
    text = _extract_text(resp).strip()
    if not text:
        return False, "empty_content"
    fr = _finish_reason(resp)
    if fr in ("length", "max_tokens"):
        return False, "truncated"
    if text.lower().startswith(_REFUSAL_PREFIXES):
        return False, "refusal_shaped"
    return True, "ok"


# ---------------------------------------------------------------------------
# The gateway
# ---------------------------------------------------------------------------

class GovernanceGateway:
    def __init__(
        self,
        *,
        budget,                       # governance.budget_ledger.BudgetLedger
        upstream_base: str,
        price_sheet: Dict[str, Dict[str, float]],
        upstream_api_key: Optional[str] = None,
        admin_key: Optional[str] = None,
        auth: Optional[Any] = None,             # governance.auth.AuthConfig
        wallet: Optional[Any] = None,           # legacy signer override
        signer: Optional[Any] = None,           # governance.signing.GatewaySigner
        http_post: Optional[HttpPost] = None,
        http_stream: Optional[HttpStream] = None,
        cache: Optional[ResponseCache] = None,
        # HG13f — optional similarity cache tier BEHIND the exact cache.
        semantic_cache: Optional[Any] = None,   # token_thrift.SemanticCache
        # HG13h — optional prompt compressor (all transforms opt-in).
        compressor: Optional[Any] = None,       # token_thrift.PromptCompressor
        # {"expensive-model": "cheaper-model"} — opt-in per model.
        cascades: Optional[Dict[str, str]] = None,
        # governance.router.ModelRouter — rule-driven model selection.
        router: Optional[Any] = None,
        # Extra upstream bases callable via X-Pluginfer-Upstream.
        upstream_allowlist: Optional[List[str]] = None,
        timeout_s: float = 120.0,
    ) -> None:
        self.budget = budget
        self.upstream_base = upstream_base.rstrip("/")
        # Multi-upstream mode: a client may pick a DIFFERENT upstream
        # per call via the X-Pluginfer-Upstream header, but ONLY from
        # this explicit allowlist — an open redirect on a gateway that
        # attaches server-side credentials would be an SSRF hole, so
        # anything not listed is refused with 403. Empty list = the
        # header is entirely disabled (single-upstream mode, default).
        self.upstream_allowlist = [u.rstrip("/")
                                   for u in (upstream_allowlist or [])]
        self.price_sheet = dict(price_sheet)
        self.upstream_api_key = upstream_api_key
        # Auth: build a config from a bare admin_key if that legacy arg
        # was passed; else use the supplied AuthConfig (or an empty,
        # open one for dev/test).
        from governance.auth import AuthConfig
        if auth is not None:
            self.auth = auth
        else:
            self.auth = AuthConfig(admin_key=admin_key)
        self.admin_key = admin_key      # kept for back-compat readers
        # Signing ON BY DEFAULT — the audit's fix. A wallet/signer may
        # be injected; otherwise mint a persisted GatewaySigner (Ed25519
        # when available, HMAC fallback) keyed to the budget state dir.
        if signer is not None:
            self.signer = signer
        elif wallet is not None:
            self.signer = wallet            # legacy .sign/.public_key_pem
        else:
            from governance.signing import GatewaySigner
            state_dir = getattr(self.budget, "_state_dir", None)
            self.signer = GatewaySigner.create(
                str(state_dir) if state_dir else None)
        self.wallet = self.signer           # back-compat alias
        self.http_post: HttpPost = http_post or _default_http_post
        self.http_stream: HttpStream = http_stream or _default_http_stream
        self.cache = cache if cache is not None else ResponseCache()
        self.semantic_cache = semantic_cache
        self.compressor = compressor
        self.cascades = dict(cascades or {})
        self.router = router
        self.timeout_s = timeout_s
        # Estimated (not measured) savings live in their OWN bucket so
        # the dashboard never mixes them with cache/cascade
        # counterfactuals. Compression removed input tokens that were
        # never billed → the saving is an estimate, labelled so.
        self._compression_saved_est_usd = 0.0
        self._compression_lock = threading.Lock()
        self._receipts: List[Dict[str, Any]] = []
        self._receipts_lock = threading.Lock()
        # Signed hash-chained audit log. Every receipt embeds the sha256
        # of the previous receipt's full record (genesis = 64 zeros) AND
        # is signed. A naive edit breaks the chain; rewriting the chain
        # requires the signing key. Ed25519 signatures are PUBLICLY
        # verifiable (an auditor needs only the public key). Full
        # insider-proofing additionally needs the chain HEAD anchored
        # externally — exposed via /v1/audit/anchor, an operator
        # deployment choice we document rather than assume. NOT called a
        # blockchain: one gateway is one writer, there is no consensus.
        self._chain_head = "0" * 64
        # The chain SURVIVES restarts: reload every persisted receipt so
        # verify_chain() covers the full history (not just this process)
        # and new receipts link onto the real head instead of restarting
        # at genesis, which would contradict the file. A break found in
        # loaded history is NOT repaired or rotated away — it stays
        # visible in verify_chain(), because silently starting a fresh
        # chain after tampering is exactly what an audit log must never do.
        self._load_persisted_receipts()

    def _load_persisted_receipts(self) -> None:
        state_dir = getattr(self.budget, "_state_dir", None)
        if state_dir is None:
            return
        try:
            with open(state_dir / "receipts.jsonl", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            return
        loaded = 0
        with self._receipts_lock:
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("skipping malformed persisted receipt "
                                   "line — verify_chain will show a break "
                                   "if history is affected")
                    continue
                self._receipts.append(rec)
                self._chain_head = self._receipt_sha(rec)
                loaded += 1
        if loaded:
            logger.info("reloaded %d persisted receipts; chain head %s...",
                        loaded, self._chain_head[:12])

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def _prices_for(self, model: str) -> Optional[Dict[str, float]]:
        return self.price_sheet.get(model) or self.price_sheet.get("*")

    def _cost_usd(self, model: str, in_tok: int, out_tok: int
                  ) -> Optional[float]:
        p = self._prices_for(model)
        if p is None:
            return None
        return (in_tok / 1e6 * float(p["input_per_1m"])
                + out_tok / 1e6 * float(p["output_per_1m"]))

    def _estimate(self, body: Dict[str, Any]) -> Tuple[str, int, int,
                                                       Optional[float]]:
        from governance import tokenizer
        model = str(body.get("model", ""))
        in_tok = tokenizer.count_request(body)
        out_tok = int(body.get("max_tokens")
                      or DEFAULT_MAX_OUTPUT_TOKENS)
        return model, in_tok, out_tok, self._cost_usd(model, in_tok,
                                                      out_tok)

    # ------------------------------------------------------------------
    # Receipts + savings
    # ------------------------------------------------------------------

    @staticmethod
    def _receipt_sha(rec: Dict[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(rec, sort_keys=True, default=str).encode()
        ).hexdigest()

    _SIG_FIELDS = ("gateway_signature", "gateway_signature_b64",
                   "gateway_pubkey_pem", "gateway_sig_alg")

    @classmethod
    def _signing_body(cls, rec: Dict[str, Any]) -> str:
        """Canonical bytes a receipt's signature covers: the whole
        record MINUS the signature fields themselves."""
        core = {k: v for k, v in rec.items() if k not in cls._SIG_FIELDS}
        return json.dumps(core, sort_keys=True, default=str)

    def _emit_receipt(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        with self._receipts_lock:
            rec = {"receipt_id": rec.pop("receipt_id", None)
                   or "rcpt-" + secrets.token_urlsafe(12),
                   "ts": time.time(),
                   "prev_sha256": self._chain_head, **rec}
            if self.signer is not None:
                try:
                    rec["gateway_signature"] = self.signer.sign(
                        self._signing_body(rec))
                    rec["gateway_sig_alg"] = getattr(
                        self.signer, "algorithm", "unknown")
                    rec["gateway_pubkey_pem"] = self.signer.public_key_pem
                except Exception as e:
                    logger.warning("receipt signing failed: %s", e)
            # The chain head covers the FULL record incl. signature, so
            # tampering with a signature also breaks the chain.
            self._receipts.append(rec)
            self._chain_head = self._receipt_sha(rec)
        state_dir = getattr(self.budget, "_state_dir", None)
        if state_dir is not None:
            try:
                with open(state_dir / "receipts.jsonl", "a",
                          encoding="utf-8") as f:
                    f.write(json.dumps(rec, sort_keys=True, default=str)
                            + "\n")
            except Exception as e:
                logger.warning("receipt write failed: %s", e)
        return rec

    def savings_summary(self) -> Dict[str, Any]:
        """Aggregated from receipts — every dollar here traces to a
        receipt with a measured counterfactual. Escalation overhead is
        SUBTRACTED, not hidden; net is what a CFO may quote."""
        with self._receipts_lock:
            receipts = list(self._receipts)
        cache_saved = cascade_saved = escalation_cost = 0.0
        semantic_saved = routing_saved = 0.0
        by_envelope: Dict[str, float] = {}
        for r in receipts:
            s = float(r.get("saved_usd", 0.0) or 0.0)
            if not s:
                continue
            kind = r.get("kind")
            if s > 0 and kind == "cache_hit":
                cache_saved += s
            elif s > 0 and kind == "semantic_cache_hit":
                semantic_saved += s
            elif s > 0 and kind == "routed":
                routing_saved += s
            elif s > 0:
                cascade_saved += s
            else:
                escalation_cost += -s
            env = str(r.get("envelope", "default"))
            by_envelope[env] = by_envelope.get(env, 0.0) + s
        with self._compression_lock:
            comp_est = round(self._compression_saved_est_usd, 8)
        sem_stats = ({"hits": self.semantic_cache.hits,
                      "misses": self.semantic_cache.misses,
                      "backend": self.semantic_cache.backend_name}
                     if self.semantic_cache is not None else None)
        return {
            "cache_hits": self.cache.hits,
            "cache_misses": self.cache.misses,
            "cache_saved_usd": round(cache_saved, 8),
            "semantic_cache": sem_stats,
            "semantic_saved_usd": round(semantic_saved, 8),
            "cascade_saved_usd": round(cascade_saved, 8),
            "routing_saved_usd": round(routing_saved, 8),
            "cascade_escalation_cost_usd": round(escalation_cost, 8),
            # Measured net — cache + semantic + cascade + routing −
            # escalation. Compression is ESTIMATED and shown separately,
            # NEVER folded into the measured net a CFO would quote.
            "net_saved_usd": round(
                cache_saved + semantic_saved + cascade_saved
                + routing_saved - escalation_cost, 8),
            "compression_saved_est_usd": comp_est,
            "by_envelope": {k: round(v, 8)
                            for k, v in sorted(by_envelope.items())},
            "note": "net_saved_usd = measured counterfactuals only; "
                    "compression_saved_est_usd is a separate ESTIMATE",
        }

    def signed_savings_report(self) -> Dict[str, Any]:
        """The shareable proof artifact: spend + measured savings +
        the receipt-chain head, signed by the gateway key. A third
        party verifies the signature with the PUBLIC key alone — this
        is the savings claim nobody can inflate, because inflating it
        breaks the signature or the chain it points at."""
        sav = self.savings_summary()
        spend = self.budget.report()
        with self._receipts_lock:
            n = len(self._receipts)
            head = self._chain_head
            first_ts = self._receipts[0].get("ts") if self._receipts else None
            last_ts = self._receipts[-1].get("ts") if self._receipts else None
        report = {
            "title": "Pluginfer Signet — signed savings report",
            "generated_unix": time.time(),
            "period": {"from_unix": first_ts, "to_unix": last_ts},
            "total_spend_usd": spend.get("total_spend_usd"),
            "savings": sav,
            "receipt_count": n,
            "chain_head_sha256": head,
            "methodology": (
                "net_saved_usd aggregates measured counterfactuals from "
                "individually signed receipts (cache hits record what the "
                "upstream actually billed for the identical request; "
                "cascade escalation overhead is subtracted). Compression "
                "figures are estimates and reported separately, never "
                "folded into the net."),
        }
        out: Dict[str, Any] = {"report": report}
        if self.signer is not None:
            body = json.dumps(report, sort_keys=True, default=str)
            out["signature"] = self.signer.sign(body)
            out["algorithm"] = getattr(self.signer, "algorithm", None)
            out["public_key_pem"] = getattr(self.signer,
                                            "public_key_pem", None)
            out["how_to_verify"] = (
                "signature covers the canonical JSON (sorted keys) of "
                "`report`; verify with public_key_pem, then confirm "
                "chain_head_sha256 against GET /v1/receipts/verify")
        return out

    def verify_chain(self) -> Dict[str, Any]:
        """Walk the audit log and report integrity on TWO axes:

          * chain continuity — each receipt's prev_sha256 must equal the
            sha of the actual previous record; a naive edit breaks it,
          * signature validity — each receipt's signature must verify
            against its signing body; editing ANY field (even without
            touching hashes) invalidates the signature.

        Reports the exact index where either check first fails. Full
        insider-proofing still needs external anchoring of chain_head
        (see /v1/audit/anchor) — stated, not assumed."""
        with self._receipts_lock:
            receipts = list(self._receipts)
        expected = "0" * 64
        sig_alg = getattr(self.signer, "algorithm", None)
        publicly_verifiable = sig_alg == "ed25519"
        for i, rec in enumerate(receipts):
            if rec.get("prev_sha256") != expected:
                return {"ok": False, "reason": "chain_break",
                        "broken_at_index": i,
                        "broken_receipt_id": rec.get("receipt_id"),
                        "receipts_checked": len(receipts)}
            sig = rec.get("gateway_signature")
            if sig is not None and self.signer is not None:
                if not self.signer.verify(self._signing_body(rec), sig):
                    return {"ok": False, "reason": "bad_signature",
                            "broken_at_index": i,
                            "broken_receipt_id": rec.get("receipt_id"),
                            "receipts_checked": len(receipts)}
            expected = self._receipt_sha(rec)
        return {"ok": True, "receipts_checked": len(receipts),
                "chain_head_sha256": expected,
                "signed": self.signer is not None,
                "signature_algorithm": sig_alg,
                "publicly_verifiable": publicly_verifiable,
                "note": ("Ed25519: any auditor can verify each receipt "
                         "with the public key alone."
                         if publicly_verifiable else
                         "HMAC integrity only — not independently "
                         "verifiable; deploy with cryptography for "
                         "Ed25519, and anchor chain_head externally.")}

    def spend_by_key(self) -> Dict[str, Any]:
        """Which API keys spent what — keys appear as fingerprints
        (sha256 prefix of the client's auth header), never raw."""
        with self._receipts_lock:
            receipts = list(self._receipts)
        by_key: Dict[str, Dict[str, Any]] = {}
        for r in receipts:
            fp = str(r.get("key_fingerprint", "anonymous"))
            slot = by_key.setdefault(fp, {"spend_usd": 0.0, "calls": 0,
                                          "envelopes": set()})
            slot["spend_usd"] += float(r.get("cost_usd", 0.0) or 0.0)
            slot["calls"] += 1
            slot["envelopes"].add(str(r.get("envelope", "default")))
        return {"by_key": {
            k: {"spend_usd": round(v["spend_usd"], 8),
                "calls": v["calls"],
                "envelopes": sorted(v["envelopes"])}
            for k, v in sorted(by_key.items())}}

    @staticmethod
    def _key_fingerprint(request: FastApiRequest) -> str:
        auth = (request.headers.get("authorization")
                or request.headers.get("x-api-key") or "")
        if not auth:
            return "anonymous"
        return hashlib.sha256(auth.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Upstream plumbing
    # ------------------------------------------------------------------

    def _upstream_headers(self, request: FastApiRequest
                          ) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        for h in _PASSTHROUGH_HEADERS:
            v = request.headers.get(h)
            if v:
                headers[h.title()] = v
        if self.upstream_api_key:
            # Enterprise mode: the REAL provider key lives only here.
            headers["Authorization"] = f"Bearer {self.upstream_api_key}"
            headers["X-Api-Key"] = self.upstream_api_key
        return headers

    def _post_upstream(self, path: str, body: Dict[str, Any],
                       headers: Dict[str, str],
                       base: Optional[str] = None
                       ) -> Tuple[int, bytes, Optional[Dict[str, Any]]]:
        req_bytes = json.dumps(body).encode("utf-8")
        status, resp_bytes = self.http_post(
            (base or self.upstream_base) + path, req_bytes, headers,
            self.timeout_s)
        try:
            return status, resp_bytes, json.loads(
                resp_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return status, resp_bytes, None

    def _resolve_upstream(self, request: FastApiRequest
                          ) -> Tuple[str, Optional[JSONResponse]]:
        """Per-call upstream selection via X-Pluginfer-Upstream,
        allowlist-enforced. Returns (base, None) or (default, 403)."""
        want = (request.headers.get("x-pluginfer-upstream") or "").rstrip("/")
        if not want or want == self.upstream_base:
            return self.upstream_base, None
        if want in self.upstream_allowlist:
            return want, None
        return self.upstream_base, JSONResponse(
            status_code=403, content={
                "error": f"upstream {want!r} is not in this gateway's "
                         f"allowlist", "allowed":
                         [self.upstream_base] + self.upstream_allowlist})

    @staticmethod
    def _usage_cost_inputs(resp: Dict[str, Any],
                           usage_keys: Tuple[str, str]
                           ) -> Tuple[Optional[int], Optional[int]]:
        usage = resp.get("usage") or {}
        i, o = usage.get(usage_keys[0]), usage.get(usage_keys[1])
        return ((int(i) if i is not None else None),
                (int(o) if o is not None else None))

    # ------------------------------------------------------------------
    # The governed forward (non-streaming)
    # ------------------------------------------------------------------

    async def govern_and_forward(
        self, request: FastApiRequest, upstream_path: str,
        usage_keys: Tuple[str, str],
    ):
        try:
            body = json.loads(await request.body() or b"{}")
        except json.JSONDecodeError:
            return JSONResponse(status_code=400, content={
                "error": "request body is not valid JSON"})
        # AUTH — the forwarding endpoint spends the org's PAID upstream
        # key, so it is the most important thing to guard. When any auth
        # is configured, a valid client key is required; the key may
        # pin its own envelope (a caller can't spend another team's).
        from governance.auth import bearer
        client_key = None
        if self.auth.enforced:
            client_key = self.auth.client_for(bearer(request.headers))
            if client_key is None:
                return JSONResponse(status_code=401, content={
                    "error": "a valid client API key is required "
                             "(Authorization: Bearer <key>)"})
        envelope = request.headers.get("x-budget-envelope", "default")
        if client_key is not None and client_key.envelope:
            # Pinned envelope wins — the key can only spend its own.
            envelope = client_key.envelope
        key_fp = self._key_fingerprint(request)
        eff_base, deny = self._resolve_upstream(request)
        if deny is not None:
            return deny
        # Model routing (Signet router): rule-driven best-model-per-
        # task selection across any number of plugged-in models. The
        # swap happens BEFORE pricing/caching so every downstream tier
        # sees the routed request; the receipt records requested vs
        # routed and the MEASURED saving at actual usage. Only models
        # on the price sheet are routable — we never route into a
        # model we can't price.
        route_info = None
        requested_model = str(body.get("model", ""))
        if self.router is not None and not body.get("stream"):
            target, rule_id, task = self.router.route(body, envelope)
            if target and target != requested_model \
                    and self._prices_for(target) is not None:
                body = dict(body, model=target)
                route_info = {"from": requested_model, "to": target,
                              "rule": rule_id, "task": task}
        # HG13h — prompt compression, on the SUBMITTING side before
        # pricing so the org pays for fewer input tokens. Every
        # transform is opt-in and itemised; its saving is ESTIMATED
        # (removed tokens were never billed) and kept in its own bucket.
        comp_report = {"applied": [], "tokens_removed_est": 0}
        if self.compressor is not None and self.compressor.enabled:
            body, comp_report = self.compressor.compress(body)
        model, in_tok, out_tok, est = self._estimate(body)
        if est is None:
            return JSONResponse(status_code=400, content={
                "error": f"model '{model}' is not in the gateway price "
                         f"sheet — refusing to forward a call we can't "
                         f"price. Configured: "
                         f"{sorted(self.price_sheet)}"})
        if comp_report["tokens_removed_est"]:
            p = self._prices_for(model)
            if p:
                saved_est = (comp_report["tokens_removed_est"] / 1e6
                             * float(p["input_per_1m"]))
                with self._compression_lock:
                    self._compression_saved_est_usd += saved_est
                comp_report["saved_est_usd"] = round(saved_est, 8)

        # Thrift rung 1 — cache. A hit costs the org $0 and needs no
        # budget hold at all; the receipt carries the measured
        # counterfactual (what the upstream billed for the identical
        # request when it was first answered).
        cached = self.cache.get(body)
        if cached is not None:
            resp_json, saved = cached
            receipt = self._emit_receipt({
                "kind": "cache_hit", "envelope": envelope,
                "key_fingerprint": key_fp,
                "model": model, "cost_usd": 0.0,
                "saved_usd": round(saved, 8),
                "request_sha256": ResponseCache.key(body),
            })
            return JSONResponse(status_code=200, content=resp_json,
                                headers={
                "X-Pluginfer-Cost-USD": "0.00000000",
                "X-Pluginfer-Cache": "hit",
                "X-Pluginfer-Saved-USD": f"{saved:.8f}",
                "X-Pluginfer-Envelope": envelope,
                "X-Pluginfer-Receipt-Id": receipt["receipt_id"],
            })

        # Thrift rung 1b — semantic (similarity) cache, BEHIND exact.
        # Catches near-duplicates (whitespace drift, injected
        # timestamps, reordered params) at an operator-set threshold.
        # Honestly labelled with its backend + the measured similarity.
        if self.semantic_cache is not None:
            sc = self.semantic_cache.get(body)
            if sc is not None:
                resp_json, saved, sim = sc
                receipt = self._emit_receipt({
                    "kind": "semantic_cache_hit", "envelope": envelope,
                    "key_fingerprint": key_fp, "model": model,
                    "cost_usd": 0.0, "saved_usd": round(saved, 8),
                    "similarity": round(sim, 4),
                    "backend": self.semantic_cache.backend_name,
                })
                return JSONResponse(status_code=200, content=resp_json,
                                    headers={
                    "X-Pluginfer-Cost-USD": "0.00000000",
                    "X-Pluginfer-Cache": f"semantic:{sim:.3f}",
                    "X-Pluginfer-Saved-USD": f"{saved:.8f}",
                    "X-Pluginfer-Envelope": envelope,
                    "X-Pluginfer-Receipt-Id": receipt["receipt_id"],
                })

        call_id = "gw-" + secrets.token_urlsafe(12)
        refusal = self.budget.reserve(
            call_id, envelope, est,
            meta={"kind": "gateway", "model": model})
        if refusal:
            return JSONResponse(status_code=402, content={
                "error": refusal,
                "envelope": envelope,
                "estimated_cost_usd": round(est, 6),
                "report_url": "/v1/budget/report",
            })

        headers = self._upstream_headers(request)
        if body.get("stream"):
            return self._governed_stream(
                request=request, upstream_path=upstream_path,
                usage_keys=usage_keys, body=body, headers=headers,
                envelope=envelope, model=model, call_id=call_id,
                hold_usd=est, in_tok_est=in_tok, key_fp=key_fp,
                upstream_base=eff_base)

        # Thrift rung 2 — cascade (opt-in per model): try the cheaper
        # configured model first; only hard failure signals escalate.
        cascade_info: Optional[Dict[str, Any]] = None
        cheap_model = self.cascades.get(model)
        cheap_cost = 0.0
        if cheap_model and self._prices_for(cheap_model) is not None:
            cheap_body = dict(body, model=cheap_model)
            try:
                status, resp_bytes, resp_json = await \
                    asyncio.get_running_loop().run_in_executor(
                        None, self._post_upstream, upstream_path,
                        cheap_body, headers, eff_base)
            except Exception:
                status, resp_bytes, resp_json = 0, b"", None
            if status == 200 and isinstance(resp_json, dict):
                ok, why = cascade_accept(resp_json)
                ri, ro = self._usage_cost_inputs(resp_json, usage_keys)
                cheap_actual = (
                    self._cost_usd(cheap_model, ri, ro)
                    if ri is not None and ro is not None else None)
                if ok and cheap_actual is not None:
                    # Accepted: settle at the CHEAP price; saving is
                    # the target-model price at the SAME measured usage.
                    target_cost = self._cost_usd(model, ri, ro) or 0.0
                    saved = max(0.0, target_cost - cheap_actual)
                    self.budget.settle(call_id, cheap_actual,
                                       meta={"model": cheap_model,
                                             "cascade": "accepted"})
                    self.cache.put(body, resp_json, cheap_actual)
                    if self.semantic_cache is not None:
                        self.semantic_cache.put(body, resp_json,
                                                cheap_actual)
                    receipt = self._emit_receipt({
                        "kind": "cascade_accept", "envelope": envelope,
                        "key_fingerprint": key_fp,
                        "model": cheap_model,
                        "requested_model": model,
                        "cost_usd": round(cheap_actual, 8),
                        "saved_usd": round(saved, 8),
                        "input_tokens": ri, "output_tokens": ro,
                        "response_sha256": hashlib.sha256(
                            resp_bytes).hexdigest(),
                    })
                    return JSONResponse(
                        status_code=200, content=resp_json, headers={
                            "X-Pluginfer-Cost-USD": f"{cheap_actual:.8f}",
                            "X-Pluginfer-Cascade":
                                f"accepted:{cheap_model}",
                            "X-Pluginfer-Saved-USD": f"{saved:.8f}",
                            "X-Pluginfer-Envelope": envelope,
                            "X-Pluginfer-Receipt-Id":
                                receipt["receipt_id"],
                        })
                # Escalating: the cheap attempt's cost is REAL spend —
                # tracked and later surfaced as negative saving.
                cheap_cost = cheap_actual or 0.0
                cascade_info = {"tried": cheap_model,
                                "escalated_because": why
                                if status == 200 else "cheap_error"}
            else:
                cascade_info = {"tried": cheap_model,
                                "escalated_because": "cheap_error"}

        try:
            status, resp_bytes, resp_json = await \
                asyncio.get_running_loop().run_in_executor(
                    None, self._post_upstream, upstream_path, body,
                    headers, eff_base)
        except Exception as e:
            self.budget.release(call_id)
            return JSONResponse(status_code=502, content={
                "error": f"upstream unreachable: "
                         f"{type(e).__name__}: {e}"})

        if status >= 400 or not isinstance(resp_json, dict):
            # Upstream refused or answered garbage — release the hold,
            # relay honestly. (Any cascade try cost is still recorded.)
            self.budget.release(call_id)
            if cheap_cost:
                self._emit_receipt({
                    "kind": "cascade_escalate", "envelope": envelope,
                    "key_fingerprint": key_fp,
                    "model": model, "cost_usd": round(cheap_cost, 8),
                    "saved_usd": round(-cheap_cost, 8),
                    "cascade": cascade_info})
            return JSONResponse(
                status_code=status if status >= 400 else 502,
                content=resp_json if isinstance(resp_json, dict) else
                {"error": "upstream returned a non-JSON body",
                 "upstream_status": status},
            )

        real_in, real_out = self._usage_cost_inputs(resp_json,
                                                    usage_keys)
        estimated = real_in is None or real_out is None
        if not estimated:
            actual = self._cost_usd(model, real_in, real_out)
            if actual is None:      # sheet changed mid-flight; be honest
                actual, estimated = est, True
        else:
            actual = est
        total = float(actual) + cheap_cost
        self.budget.settle(call_id, total,
                           meta={"model": model, "estimated": estimated})
        self.cache.put(body, resp_json, float(actual))
        if self.semantic_cache is not None:
            self.semantic_cache.put(body, resp_json, float(actual))

        # Routing saving — MEASURED: the requested model's price at the
        # ACTUAL usage minus what was paid. Only computed when real
        # usage arrived; estimates never mint savings.
        route_saved = 0.0
        if route_info is not None and not estimated:
            req_cost = self._cost_usd(route_info["from"], real_in,
                                      real_out)
            if req_cost is not None:
                route_saved = max(0.0, req_cost - float(actual))
        net_saved = round(route_saved - cheap_cost, 8)
        receipt = self._emit_receipt({
            "kind": ("cascade_escalate" if cascade_info
                     else "routed" if route_info else "forward"),
            "envelope": envelope,
            "key_fingerprint": key_fp,
            "model": model,
            "requested_model": (route_info["from"] if route_info
                                else None),
            "routing": route_info,
            "upstream": eff_base + upstream_path,
            "input_tokens": real_in if real_in is not None else in_tok,
            "output_tokens": (real_out if real_out is not None
                              else out_tok),
            "cost_usd": round(total, 8),
            "saved_usd": net_saved if (cheap_cost or route_saved) else 0.0,
            "compression": comp_report if comp_report["applied"] else None,
            "estimated": estimated,
            "cascade": cascade_info,
            "request_sha256": hashlib.sha256(
                json.dumps(body).encode()).hexdigest(),
            "response_sha256": hashlib.sha256(resp_bytes).hexdigest(),
        })
        headers = {
            "X-Pluginfer-Cost-USD": f"{total:.8f}",
            "X-Pluginfer-Envelope": envelope,
            "X-Pluginfer-Receipt-Id": receipt["receipt_id"],
        }
        if route_info is not None:
            headers["X-Pluginfer-Routed"] = \
                f"{route_info['from']}->{route_info['to']}"
            headers["X-Pluginfer-Saved-USD"] = f"{route_saved:.8f}"
        return JSONResponse(status_code=status, content=resp_json,
                            headers=headers)

    # ------------------------------------------------------------------
    # HG13c — governed streaming
    # ------------------------------------------------------------------

    def _governed_stream(self, *, request: FastApiRequest,
                         upstream_path: str,
                         usage_keys: Tuple[str, str],
                         body: Dict[str, Any],
                         headers: Dict[str, str], envelope: str,
                         model: str, call_id: str, hold_usd: float,
                         in_tok_est: int,
                         key_fp: str = "anonymous",
                         upstream_base: Optional[str] = None
                         ) -> StreamingResponse:
        eff_base = upstream_base or self.upstream_base
        # Ask OpenAI-compatible upstreams to emit usage at stream end.
        # Harmless if unknown: servers ignore unrecognised options or
        # error — an error is relayed and the hold released.
        if "stream_options" not in body and "messages" in body \
                and upstream_path.endswith("/chat/completions"):
            body = dict(body,
                        stream_options={"include_usage": True})
        receipt_id = "rcpt-" + secrets.token_urlsafe(12)
        gw = self

        def _gen() -> Iterator[bytes]:
            out_chars = 0
            usage_seen: Dict[str, int] = {}
            cutoff = False
            buffer = b""
            try:
                status, chunks = gw.http_stream(
                    eff_base + upstream_path,
                    json.dumps(body).encode("utf-8"), headers,
                    gw.timeout_s)
                if status >= 400:
                    gw.budget.release(call_id)
                    for c in chunks:
                        yield c
                    return
                for chunk in chunks:
                    buffer += chunk
                    # Parse complete SSE lines for usage + content size;
                    # relay bytes verbatim either way.
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        text = line.decode("utf-8", "replace").strip()
                        if not text.startswith("data:"):
                            continue
                        payload = text[5:].strip()
                        if not payload or payload == "[DONE]":
                            continue
                        try:
                            obj = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        u = obj.get("usage")
                        if isinstance(u, dict):
                            for k, v in u.items():
                                if isinstance(v, int):
                                    usage_seen[k] = v
                        # Content size, both shapes.
                        for ch in obj.get("choices") or []:
                            delta = (ch or {}).get("delta") or {}
                            out_chars += len(str(
                                delta.get("content") or ""))
                        d = obj.get("delta")
                        if isinstance(d, dict):
                            out_chars += len(str(d.get("text") or ""))
                    yield chunk
                    running = gw._cost_usd(
                        model, in_tok_est, max(1, out_chars // 4))
                    if running is not None and running >= hold_usd:
                        cutoff = True
                        yield (b'data: {"error": {"message": '
                               b'"pluginfer_budget_cutoff: stream '
                               b'reached its budget hold; envelope '
                               b'protected"}}\n\n')
                        yield b"data: [DONE]\n\n"
                        break
            except Exception as e:
                gw.budget.release(call_id)
                yield (b'data: {"error": {"message": "upstream '
                       b'stream failed: '
                       + str(e).encode("utf-8", "replace")[:200]
                       + b'"}}\n\n')
                return
            # Settle: upstream usage when it arrived, else estimate.
            real_in = usage_seen.get(usage_keys[0])
            real_out = usage_seen.get(usage_keys[1])
            estimated = real_in is None or real_out is None
            if cutoff:
                actual, estimated = hold_usd, True
            elif not estimated:
                actual = gw._cost_usd(model, real_in, real_out)
                if actual is None:
                    actual, estimated = hold_usd, True
            else:
                actual = gw._cost_usd(
                    model, in_tok_est, max(1, out_chars // 4)) \
                    or hold_usd
            gw.budget.settle(call_id, float(actual),
                             meta={"model": model, "streamed": True,
                                   "estimated": estimated})
            gw._emit_receipt({
                "receipt_id": receipt_id,
                "kind": "stream", "envelope": envelope,
                "key_fingerprint": key_fp,
                "model": model, "cost_usd": round(float(actual), 8),
                "estimated": estimated, "cutoff": cutoff,
                "input_tokens": real_in if real_in is not None
                else in_tok_est,
                "output_tokens": (real_out if real_out is not None
                                  else max(1, out_chars // 4)),
            })

        return StreamingResponse(_gen(),
                                 media_type="text/event-stream",
                                 headers={
            "X-Pluginfer-Envelope": envelope,
            "X-Pluginfer-Receipt-Id": receipt_id,
            "X-Pluginfer-Stream": "governed",
        })


def build_governance_gateway(
    *,
    budget,
    upstream_base: str,
    price_sheet: Dict[str, Dict[str, float]],
    upstream_api_key: Optional[str] = None,
    admin_key: Optional[str] = None,
    auth: Optional[Any] = None,
    wallet: Optional[Any] = None,
    signer: Optional[Any] = None,
    http_post: Optional[HttpPost] = None,
    http_stream: Optional[HttpStream] = None,
    cache: Optional[ResponseCache] = None,
    semantic_cache: Optional[Any] = None,
    compressor: Optional[Any] = None,
    cascades: Optional[Dict[str, str]] = None,
    router: Optional[Any] = None,
    upstream_allowlist: Optional[List[str]] = None,
    timeout_s: float = 120.0,
) -> FastAPI:
    gw = GovernanceGateway(
        budget=budget, upstream_base=upstream_base,
        price_sheet=price_sheet, upstream_api_key=upstream_api_key,
        admin_key=admin_key, auth=auth, wallet=wallet, signer=signer,
        http_post=http_post, http_stream=http_stream, cache=cache,
        semantic_cache=semantic_cache, compressor=compressor,
        cascades=cascades, router=router,
        upstream_allowlist=upstream_allowlist, timeout_s=timeout_s,
    )
    app = FastAPI(title="Pluginfer Signet — AI Spend Gateway")
    app.state.gateway = gw

    from governance.auth import bearer

    def _need_admin(request: FastApiRequest):
        # Admin token via X-Admin-Key or bearer. When no auth is
        # configured at all, the surface stays open (dev/test default).
        if not gw.auth.enforced:
            return None
        key = request.headers.get("x-admin-key") or bearer(request.headers)
        if not gw.auth.is_admin(key):
            return JSONResponse(status_code=401, content={
                "error": "admin credential required (X-Admin-Key)"})
        return None

    def _need_reader(request: FastApiRequest):
        if not gw.auth.enforced:
            return None
        key = request.headers.get("x-admin-key") or bearer(request.headers)
        if not gw.auth.is_reader(key):
            return JSONResponse(status_code=401, content={
                "error": "read credential required to view spend data"})
        return None

    @app.get("/healthz")
    async def healthz():
        # Deliberately unauthenticated + no sensitive data — for probes.
        return {"status": "ok", "upstream": gw.upstream_base,
                "models_priced": sorted(gw.price_sheet),
                "cascades": gw.cascades,
                "cache_ttl_s": gw.cache.ttl_s,
                "auth_enforced": gw.auth.enforced,
                "signing": getattr(gw.signer, "algorithm", None)}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: FastApiRequest):
        return await gw.govern_and_forward(
            request, "/v1/chat/completions",
            usage_keys=("prompt_tokens", "completion_tokens"))

    @app.post("/v1/messages")
    async def messages(request: FastApiRequest):
        return await gw.govern_and_forward(
            request, "/v1/messages",
            usage_keys=("input_tokens", "output_tokens"))

    @app.post("/v1/quote")
    async def quote(request: FastApiRequest):
        try:
            body = json.loads(await request.body() or b"{}")
        except json.JSONDecodeError:
            return JSONResponse(status_code=400, content={
                "error": "request body is not valid JSON"})
        model, in_tok, out_tok, est = gw._estimate(body)
        if est is None:
            return JSONResponse(status_code=400, content={
                "error": f"model '{model}' is not in the price sheet",
                "configured": sorted(gw.price_sheet)})
        from governance import tokenizer
        return {"model": model,
                "estimated_input_tokens": in_tok,
                "max_output_tokens": out_tok,
                "estimated_max_cost_usd": round(est, 6),
                "tokenizer": tokenizer.backend(),
                "note": "input tokens are an estimate; actual billing "
                        "settles at the upstream's usage block"}

    @app.get("/v1/budget/report")
    async def budget_report(request: FastApiRequest, prefix: str = "",
                            since_unix: float = 0.0):
        deny = _need_reader(request)
        if deny is not None:
            return deny
        return gw.budget.report(prefix=prefix, since_unix=since_unix)

    @app.get("/v1/savings")
    async def savings(request: FastApiRequest):
        deny = _need_reader(request)
        if deny is not None:
            return deny
        return gw.savings_summary()

    @app.get("/v1/savings/report")
    async def savings_report(request: FastApiRequest,
                             format: str = "json"):
        deny = _need_reader(request)
        if deny is not None:
            return deny
        data = gw.signed_savings_report()
        if format != "html":
            return data
        rep = data["report"]
        sav = rep["savings"]
        html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Pluginfer Signet — Signed Savings Report</title>
<style>body{{font:15px/1.6 system-ui;max-width:720px;margin:40px auto;
padding:0 16px;color:#111}}h1{{font-size:22px}}.big{{font-size:34px;
font-weight:700;color:#008300}}table{{border-collapse:collapse;
width:100%;font-size:14px}}td,th{{border-bottom:1px solid #ddd;
padding:6px 8px;text-align:left}}pre{{background:#f4f4f2;padding:12px;
border-radius:8px;overflow-x:auto;font-size:11px}}
.note{{color:#666;font-size:13px}}</style></head><body>
<h1>Signet · Signed Savings Report</h1>
<p class="big">${sav.get('net_saved_usd', 0)}</p>
<p>measured net savings across <b>{rep['receipt_count']}</b> signed
receipts (total spend ${rep.get('total_spend_usd')})</p>
<table><tr><th>Component</th><th>USD</th></tr>
<tr><td>Exact-cache savings (measured)</td>
<td>{sav.get('cache_saved_usd')}</td></tr>
<tr><td>Semantic-cache savings (measured)</td>
<td>{sav.get('semantic_saved_usd')}</td></tr>
<tr><td>Cascade savings (measured)</td>
<td>{sav.get('cascade_saved_usd')}</td></tr>
<tr><td>Cascade escalation overhead (subtracted)</td>
<td>-{sav.get('cascade_escalation_cost_usd')}</td></tr>
<tr><td>Compression (estimate — reported separately)</td>
<td>{sav.get('compression_saved_est_usd')}</td></tr></table>
<p class="note">{rep['methodology']}</p>
<h2 style="font-size:15px">Cryptographic proof</h2>
<p class="note">Signature ({data.get('algorithm')}) covers the
canonical JSON below. Verify with the public key, then confirm the
chain head via <code>/v1/receipts/verify</code>.</p>
<pre>{json.dumps(data, indent=2, default=str)}</pre>
</body></html>"""
        return HTMLResponse(html)

    @app.get("/v1/receipts")
    async def receipts(request: FastApiRequest, limit: int = 50):
        deny = _need_reader(request)
        if deny is not None:
            return deny
        with gw._receipts_lock:
            return {"receipts": gw._receipts[-max(1, min(limit, 1000)):]}

    @app.get("/v1/receipts/verify")
    async def receipts_verify():
        # Integrity check is public by design — anyone may confirm the
        # log is intact; it exposes no spend figures.
        return gw.verify_chain()

    @app.get("/v1/audit/anchor")
    async def audit_anchor():
        """The signed chain HEAD + public key, for EXTERNAL anchoring.
        Publish this periodically to an append-only third-party store
        (or public timestamp) and full insider-proofing follows: the
        operator can no longer rewrite history without contradicting a
        head they already published."""
        with gw._receipts_lock:
            head = gw._chain_head
            count = len(gw._receipts)
        sig = gw.signer.sign(head) if gw.signer is not None else None
        return {"chain_head_sha256": head, "receipt_count": count,
                "signature": sig,
                "algorithm": getattr(gw.signer, "algorithm", None),
                "public_key_pem": getattr(gw.signer, "public_key_pem",
                                          None),
                "how_to_use": "publish this record to an append-only "
                              "external store on a schedule; auditors "
                              "verify each receipt with public_key_pem "
                              "and confirm the head matches."}

    @app.get("/v1/spend/by_key")
    async def spend_by_key(request: FastApiRequest):
        deny = _need_reader(request)
        if deny is not None:
            return deny
        return gw.spend_by_key()

    @app.post("/v1/budget/envelopes")
    async def set_envelope(request: FastApiRequest):
        deny = _need_admin(request)
        if deny is not None:
            return deny
        try:
            body = json.loads(await request.body() or b"{}")
            env = gw.budget.set_envelope(
                str(body["path"]), float(body["cap_usd"]),
                str(body.get("period", "month")))
        except (KeyError, ValueError, TypeError,
                json.JSONDecodeError) as e:
            return JSONResponse(status_code=400,
                                content={"error": str(e)})
        return env.to_dict()

    @app.get("/v1/budget/envelopes")
    async def list_envelopes(request: FastApiRequest):
        deny = _need_reader(request)
        if deny is not None:
            return deny
        return {"envelopes": gw.budget.envelopes()}

    @app.post("/v1/keys")
    async def issue_key(request: FastApiRequest):
        deny = _need_admin(request)
        if deny is not None:
            return deny
        try:
            body = json.loads(await request.body() or b"{}")
        except json.JSONDecodeError:
            body = {}
        raw = gw.auth.issue_client_key(
            label=str(body.get("label", "")),
            envelope=body.get("envelope"))
        return {"api_key": raw, "note": "store it now — only the "
                "fingerprint is retained server-side"}

    @app.delete("/v1/keys/{key_or_fp}")
    async def revoke_key(key_or_fp: str, request: FastApiRequest):
        deny = _need_admin(request)
        if deny is not None:
            return deny
        return {"revoked": gw.auth.revoke_client_key(key_or_fp)}

    @app.get("/v1/keys")
    async def list_keys(request: FastApiRequest):
        deny = _need_admin(request)
        if deny is not None:
            return deny
        return {"keys": gw.auth.list_client_keys()}

    @app.get("/dashboard")
    async def dashboard(request: FastApiRequest):
        # The dashboard fetches spend endpoints client-side; those
        # enforce read auth themselves. The shell is fine to serve.
        return HTMLResponse(_DASHBOARD_HTML)

    return app


# The dashboard is a self-contained page (no CDNs — it must work on an
# airgapped enterprise network) that renders /v1/budget/report,
# /v1/savings and /v1/receipts. Injected at import time from the
# sibling file so the HTML is editable without touching Python.
_DASHBOARD_PATH = os.path.join(os.path.dirname(__file__),
                               "dashboard.html")
try:
    with open(_DASHBOARD_PATH, encoding="utf-8") as _f:
        _DASHBOARD_HTML = _f.read()
except OSError:
    _DASHBOARD_HTML = (
        "<h1>Dashboard file missing</h1><p>Expected at "
        "governance/dashboard.html. The JSON endpoints "
        "(/v1/budget/report, /v1/savings, /v1/receipts) are "
        "unaffected.</p>")


def main() -> int:
    """Run the gateway standalone:

        set PLUGINFER_GW_UPSTREAM=https://api.openai.com
        set PLUGINFER_GW_PRICES=C:\\path\\to\\prices.json
        python -m governance.gateway

    prices.json: {"gpt-4o": {"input_per_1m": 2.5, "output_per_1m": 10}}
    Optional env: PLUGINFER_GW_KEY (upstream key held server-side),
    PLUGINFER_GW_ADMIN_KEY, PLUGINFER_BUDGET_DIR, PLUGINFER_GW_PORT,
    PLUGINFER_GW_CACHE_TTL_S, PLUGINFER_GW_CACHE_ALL=1,
    PLUGINFER_GW_CASCADES (path to {"expensive": "cheap"} JSON),
    PLUGINFER_GW_SEMANTIC=1 (+ _SEMANTIC_THRESHOLD, default 0.97),
    PLUGINFER_GW_COMPRESS_DEDUP=1 / _COMPRESS_WS=1 /
    _COMPRESS_MAX_INPUT_TOKENS=N.
    """
    # host_guard is optional — the standalone install may not ship it.
    try:
        import host_guard
        host_guard.install("governance-gateway")
    except Exception:
        pass
    logging.basicConfig(level=os.environ.get("PLUGINFER_LOG_LEVEL",
                                             "INFO"))
    upstream = os.environ.get("PLUGINFER_GW_UPSTREAM", "")
    prices_path = os.environ.get("PLUGINFER_GW_PRICES", "")
    if not upstream or not prices_path:
        print("PLUGINFER_GW_UPSTREAM and PLUGINFER_GW_PRICES are "
              "required — see `python -m governance.gateway` "
              "docstring.")
        return 2
    with open(prices_path, encoding="utf-8") as f:
        price_sheet = json.load(f)
    cascades = None
    cascades_path = os.environ.get("PLUGINFER_GW_CASCADES", "")
    if cascades_path:
        with open(cascades_path, encoding="utf-8") as f:
            cascades = json.load(f)
    router = None
    routes_path = os.environ.get("PLUGINFER_GW_ROUTES", "")
    if routes_path:
        from governance.router import ModelRouter
        with open(routes_path, encoding="utf-8") as f:
            router = ModelRouter(json.load(f))
    elif os.environ.get("PLUGINFER_GW_AUTOROUTE", "").lower() == "save":
        # Zero-config cost saver: route simple tasks to the cheapest
        # priced model, never downgrading code/long-context. A no-op
        # (empty rules) when there is nothing to optimize.
        from governance.router import ModelRouter, auto_save_rules
        rules = auto_save_rules(price_sheet)
        if rules:
            router = ModelRouter(rules)
    from governance.budget_ledger import BudgetLedger
    budget = BudgetLedger(
        os.environ.get("PLUGINFER_BUDGET_DIR",
                       str(os.path.join(os.path.expanduser("~"),
                                        ".pluginfer", "budget"))),
        require_envelope=os.environ.get(
            "PLUGINFER_GW_REQUIRE_ENVELOPE", "0") == "1",
    )
    semantic_cache = compressor = None
    if os.environ.get("PLUGINFER_GW_SEMANTIC", "0") == "1":
        from governance.embedders import make_embedder
        from governance.token_thrift import SemanticCache
        # 'lexical' (default, zero-dep) or 'ollama' (true semantic via
        # a local embedding model — PLUGINFER_GW_EMBED_MODEL). An
        # unavailable ollama backend FAILS STARTUP rather than silently
        # downgrading to lexical: config must mean what it says.
        embed_fn, backend_name = make_embedder(
            os.environ.get("PLUGINFER_GW_SEMANTIC_BACKEND", "lexical"),
            model=os.environ.get("PLUGINFER_GW_EMBED_MODEL",
                                 "nomic-embed-text"),
            base_url=os.environ.get("PLUGINFER_GW_OLLAMA_URL",
                                    "http://127.0.0.1:11434"))
        semantic_cache = SemanticCache(
            threshold=float(os.environ.get(
                "PLUGINFER_GW_SEMANTIC_THRESHOLD", "0.97")),
            ttl_s=float(os.environ.get("PLUGINFER_GW_CACHE_TTL_S",
                                       str(DEFAULT_CACHE_TTL_S))),
            cache_all=os.environ.get("PLUGINFER_GW_CACHE_ALL",
                                     "0") == "1",
            embed_fn=embed_fn, backend_name=backend_name)
    if any(os.environ.get(k) for k in (
            "PLUGINFER_GW_COMPRESS_DEDUP", "PLUGINFER_GW_COMPRESS_WS",
            "PLUGINFER_GW_COMPRESS_MAX_INPUT_TOKENS")):
        from governance.token_thrift import PromptCompressor
        compressor = PromptCompressor(
            dedup_exact=os.environ.get(
                "PLUGINFER_GW_COMPRESS_DEDUP", "0") == "1",
            collapse_whitespace=os.environ.get(
                "PLUGINFER_GW_COMPRESS_WS", "0") == "1",
            max_input_tokens=int(os.environ.get(
                "PLUGINFER_GW_COMPRESS_MAX_INPUT_TOKENS", "0") or 0))
    from governance.auth import AuthConfig
    admin_key = os.environ.get("PLUGINFER_GW_ADMIN_KEY") or None
    read_keys = [k for k in os.environ.get(
        "PLUGINFER_GW_READ_KEYS", "").split(",") if k.strip()]
    auth = AuthConfig(admin_key=admin_key, read_keys=read_keys)
    host = os.environ.get("PLUGINFER_GW_HOST", "127.0.0.1")
    # Refuse to expose a PUBLIC interface with no auth — the audit's
    # security fix, enforced at startup rather than trusted to config.
    if host not in ("127.0.0.1", "localhost", "::1") and not auth.enforced:
        print(f"REFUSING to bind {host} with no auth configured. Set "
              "PLUGINFER_GW_ADMIN_KEY (and issue client keys), or bind "
              "127.0.0.1. This protects your upstream API key from "
              "anyone who can reach the port.")
        return 3
    app = build_governance_gateway(
        budget=budget, upstream_base=upstream, price_sheet=price_sheet,
        upstream_api_key=os.environ.get("PLUGINFER_GW_KEY") or None,
        auth=auth,
        cache=ResponseCache(
            ttl_s=float(os.environ.get("PLUGINFER_GW_CACHE_TTL_S",
                                       str(DEFAULT_CACHE_TTL_S))),
            cache_all=os.environ.get("PLUGINFER_GW_CACHE_ALL",
                                     "0") == "1"),
        semantic_cache=semantic_cache, compressor=compressor,
        cascades=cascades, router=router,
        upstream_allowlist=[u for u in os.environ.get(
            "PLUGINFER_GW_UPSTREAMS", "").split(",") if u.strip()],
    )
    import uvicorn
    from governance import BANNER
    print(BANNER)
    port = int(os.environ.get("PLUGINFER_GW_PORT", "8788"))
    uvicorn.run(app, host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

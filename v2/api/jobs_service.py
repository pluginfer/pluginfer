"""JobsService — bridges the REST API to the core auction + ledger.

The API layer never touches Auction / ComputeLedger / Provider directly;
it goes through this thin in-process service so:

  * we can swap a real broker in without rewriting routers,
  * the test suite can spin up a JobsService against an in-memory chain
    + mock providers without HTTP,
  * the SDK has a stable contract independent of the core's internal
    naming churn.

Every job lives in `jobs` (in-memory dict for v1; persistent SQLite is a
v1.1 follow-up). Submission runs the auction synchronously, then queues
the winner for execution on a background `asyncio.Task` so the HTTP
caller gets a fast 202-style response. Caller polls GET /v1/jobs/{id}
or subscribes to /v1/jobs/{id}/stream for SSE updates.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.metrics import auction_duration_seconds, job_duration_seconds, jobs_total
from core.providers import (
    Auction,
    AuctionResult,
    Bid,
    JobSpec,
    Provider,
)


@dataclass
class JobRecord:
    job_id: str
    kind: str
    payload: Dict[str, Any]
    cost_ceiling_usd: float
    latency_ceiling_ms: int
    privacy_class: str
    quality_floor: float
    submitted_at_unix: float
    requester_identity: str
    state: str = "queued"
    detail: Optional[str] = None
    # §RFC-3 — which budget envelope this job draws from. Defaults to
    # requester_identity so chargeback attribution exists even for
    # callers that never heard of budgets.
    budget_envelope: Optional[str] = None
    matched_provider_pubkey: Optional[str] = None
    price_locked_usd: Optional[float] = None
    auction_result: Optional[AuctionResult] = None
    result_b64: Optional[str] = None
    result_hash_hex: Optional[str] = None
    provider_signature_b64: Optional[str] = None
    execution_ms: Optional[float] = None
    completed_at_unix: Optional[float] = None
    # G7 — energy + carbon accounting captured around provider.execute().
    # The PNIS receipt's energy_mj / carbon_gco2e fields are stamped
    # from this. Stays None until the executor returns.
    energy_report: Optional[Any] = None
    # G8 — per-token streaming. JobsService allocates one delta queue
    # per record on submit_streaming(); the executor's `on_delta`
    # callback pushes incremental chunks through call_soon_threadsafe
    # so the SSE handler in the devserver can consume them inside the
    # asyncio loop. None on non-streaming jobs.
    delta_queue: Optional[asyncio.Queue] = field(default=None, repr=False)
    # G/SSE-resume: persistent delta cursor that survives reconnect.
    # The asyncio.Queue is consumed live; the cursor's ring buffer
    # is what replay-on-reconnect reads from. Both fill from the
    # same `on_delta` callback; the cursor records seq numbers.
    delta_cursor: Optional[Any] = field(default=None, repr=False)
    _watchers: List[asyncio.Queue] = field(default_factory=list, repr=False)

    def to_info(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "state": {"state": self.state, "detail": self.detail},
            "submitted_at_unix": self.submitted_at_unix,
            "matched_provider_pubkey": self.matched_provider_pubkey,
            "price_locked_usd": self.price_locked_usd,
            "cost_ceiling_usd": self.cost_ceiling_usd,
            "privacy_class": self.privacy_class,
            "budget_envelope": self.budget_envelope,
        }

    def to_result(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "state": {"state": self.state, "detail": self.detail},
            "result_b64": self.result_b64,
            "result_hash_hex": self.result_hash_hex,
            "provider_signature_b64": self.provider_signature_b64,
            "execution_ms": self.execution_ms,
        }


@dataclass
class JobsService:
    auction: Auction
    # In-process job dict — kept as a thin cache over the JobStore so
    # SSE watchers + asyncio.Queue references survive between calls.
    # Every state transition is mirrored to `store` via `_persist(rec)`
    # which writes through to the durable backend (SQLite in prod,
    # InMemory in tests). On boot, `restore_open_jobs()` re-hydrates
    # the cache from the store and demotes still-running jobs to
    # `interrupted` so buyers see a terminal state instead of waiting
    # on a process that no longer exists.
    jobs: Dict[str, JobRecord] = field(default_factory=dict)
    store: Optional[Any] = None
    # Off-chain buyer/provider ledger. When set, every job submission
    # locks the auction-cleared price from the buyer's wallet
    # pre-execution; success releases (1-commission) × locked to the
    # winning provider, commission to treasury; failure refunds 100%.
    # When None (the test default for unit tests that don't care
    # about money), the auction runs without any economic side-effects.
    ledger: Optional[Any] = None
    # PNIS-Receipt v1 JSON, keyed by job_id. Populated at terminal
    # state for `completed` jobs by the devserver / API layer when a
    # gateway_wallet is available (see `attest_receipt`).
    pnis_receipts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Gateway wallet used to sign PNIS receipts. Constructed on first
    # use so test fixtures don't pay the cost.
    gateway_wallet: Optional[Any] = None
    # §RFC-3 — Budget-as-Contract. When set (a governance.budget_ledger
    # .BudgetLedger), submit() reserves the job's cost ceiling against
    # the caller's envelope BEFORE the auction (fail-closed, zero
    # economic side effects on refusal) and terminal settlement records
    # the ACTUAL cleared price. None = ungoverned (the historical
    # behavior; every unit test that doesn't care about budgets).
    budget: Optional[Any] = None
    _executor: Optional["asyncio.AbstractEventLoop"] = None

    def __post_init__(self) -> None:
        if self.store is None:
            from api.job_store import InMemoryJobStore
            self.store = InMemoryJobStore()
        # Re-hydrate from disk. Survivors are still-open jobs at the
        # last shutdown; mark them interrupted and surface so monitors
        # can alert / re-submit.
        for rec in self.store.restore_open_jobs():
            self.jobs[rec.job_id] = rec

    def _persist(self, rec: JobRecord) -> None:
        try:
            self.store.put(rec)
        except Exception as e:
            # Never let a persistence failure crash the auction path —
            # log and continue with the in-memory copy.
            import logging
            logging.getLogger(__name__).warning(
                "JobsService persist failed for %s: %s", rec.job_id, e,
            )

    def register_provider(self, p: Provider) -> None:
        self.auction.register(p)

    def _ensure_gateway_wallet(self) -> Any:
        if self.gateway_wallet is None:
            from core.tokenomics import Wallet
            self.gateway_wallet = Wallet()
        return self.gateway_wallet

    def attest_receipt(self, rec: JobRecord) -> Optional[Dict[str, Any]]:
        """Build a §A1 PNIS receipt for a completed job, signed by the
        gateway wallet, and cache it under `rec.job_id`. Returns the
        receipt dict, or None for non-completed jobs.

        The gateway is the trust anchor: its wallet signs, and the
        receipt records WHICH upstream provider it dispatched to (via
        ``provider.id`` and ``provider_attestation`` side-band). A
        verifier checks both:
          * gateway signature over the receipt body (proves gateway
            saw this job and computed these hashes),
          * upstream signature over result_hash (proves the upstream
            provider produced the bytes, available when the provider
            attested its own result).

        Forward-compat for a multi-signer schema is a §A1.1 work item.
        """
        if rec.state != "completed":
            return None
        cached = self.pnis_receipts.get(rec.job_id)
        if cached is not None:
            return cached
        from core.ai_receipt import ProviderRef, make_receipt
        from decimal import Decimal
        import base64 as _b64
        wallet = self._ensure_gateway_wallet()
        upstream_id = rec.matched_provider_pubkey or "unknown"

        if isinstance(rec.payload, dict):
            raw_input = rec.payload.get("prompt") or rec.payload.get("input") or ""
            model_id = str(rec.payload.get("model", rec.kind))
        else:
            raw_input = ""
            model_id = rec.kind
        input_bytes = raw_input.encode("utf-8") if isinstance(raw_input, str) \
            else str(raw_input).encode("utf-8")
        try:
            output_bytes = (
                _b64.b64decode(rec.result_b64) if rec.result_b64 else b""
            )
        except (ValueError, TypeError):
            output_bytes = b""

        # G7: pull energy + carbon out of the captured report (if the
        # executor ran a meter — JobsService.submit always does, but
        # direct callers of attest_receipt() may not). Default to
        # Decimal("0") so make_receipt's signature math stays valid;
        # the energy_source field on the receipt makes it explicit
        # whether this is a real measurement or a placeholder.
        e_mj = Decimal("0")
        carbon_gco2e = Decimal("0")
        energy_extras: Dict[str, Any] = {}
        if rec.energy_report is not None:
            e_mj = Decimal(rec.energy_report.energy_mj)
            carbon_gco2e = Decimal(rec.energy_report.carbon_gco2e)
            energy_extras = rec.energy_report.to_receipt_fields()
        receipt = make_receipt(
            job_id=rec.job_id,
            model_id=model_id,
            model_hash=rec.result_hash_hex or "0" * 64,
            model_kind="llm" if rec.kind.startswith("llm") else "compute",
            provider=ProviderRef(
                id=upstream_id,
                pubkey=wallet.public_key_pem,
                kind="gateway-attested",
            ),
            input_bytes=input_bytes,
            output_bytes=output_bytes,
            cost_plg=Decimal(str(rec.price_locked_usd or 0.0)),
            cost_usd_estimate=Decimal(str(rec.price_locked_usd or 0.0)),
            energy_mj=e_mj,
            latency_ms=int(rec.execution_ms or 0),
            signer=wallet,
        )
        d = receipt.to_dict()
        d["provider_attestation"] = {
            "provider_id": upstream_id,
            "signature_b64": rec.provider_signature_b64,
            "result_hash_hex": rec.result_hash_hex,
        }
        # §RFC-3 — chargeback attribution rides the receipt side-band
        # (same convention as energy_disclosure: the §A1 signed body
        # stays shape-stable). "Where did the money go?" is now a
        # GROUP BY budget_attribution.envelope over signed receipts.
        d["budget_attribution"] = {
            "envelope": rec.budget_envelope or rec.requester_identity,
            "requester_identity": rec.requester_identity,
        }
        # Stamp the carbon + zone + intensity as a sidecar so the §A1
        # signed body stays unchanged shape-wise (energy_mj is in the
        # signed body; carbon + zone are side-band for compliance
        # readers — they verify against energy_mj × the intensity).
        if energy_extras:
            d["energy_disclosure"] = energy_extras
            d["energy_disclosure"]["carbon_gco2e"] = str(carbon_gco2e)
        self.pnis_receipts[rec.job_id] = d
        return d

    async def _push_event(self, rec: JobRecord, event: str) -> None:
        # Mirror every state transition to the durable store. This is
        # the single point where `_persist` is called — every code path
        # that mutates rec.state must go through _push_event so that
        # SQLite (or whatever backend) sees a consistent journal.
        self._persist(rec)
        for q in list(rec._watchers):
            try:
                q.put_nowait({"event": event, "job": rec.to_info(),
                              "result": rec.to_result()})
            except asyncio.QueueFull:
                pass

    async def submit(
        self,
        *,
        kind: str,
        payload: Dict[str, Any],
        cost_ceiling_usd: float,
        latency_ceiling_ms: int,
        privacy_class: str,
        quality_floor: float,
        requester_identity: str,
        streaming: bool = False,
        buyer_wallet_id: Optional[str] = None,
        budget_envelope: Optional[str] = None,
    ) -> JobRecord:
        """Submit a job. When `streaming=True`, the JobRecord is built
        with a delta queue so the SSE caller can consume per-token
        chunks pushed by the provider (G8). Backward-compatible:
        non-streaming callers never see the queue."""
        # Host-guard gate — BEFORE the auction and before any ledger
        # lock, so a rejection has zero economic side effects. Under
        # host memory pressure we terminally fail the job with an
        # honest detail string rather than queue work this machine
        # cannot take (queueing under pressure is how hosts hang).
        try:
            import host_guard
            _accepting = host_guard.should_accept_work()
        except ImportError:
            _accepting = True
        if not _accepting:
            rec = JobRecord(
                job_id=secrets.token_urlsafe(16),
                kind=kind,
                payload=payload,
                cost_ceiling_usd=cost_ceiling_usd,
                latency_ceiling_ms=latency_ceiling_ms,
                privacy_class=privacy_class,
                quality_floor=quality_floor,
                submitted_at_unix=time.time(),
                requester_identity=requester_identity,
                state="failed",
                detail="host_guard: host under memory pressure — job "
                       "rejected before auction to protect this machine",
            )
            self.jobs[rec.job_id] = rec
            self._persist(rec)
            return rec
        job_id = secrets.token_urlsafe(16)
        # §RFC-3 budget gate — same placement discipline as host_guard:
        # BEFORE the auction and before any ledger lock, so a refusal
        # has zero economic side effects. The hold is the job's cost
        # CEILING (worst case); settlement later records the actual
        # auction-cleared price, which is <= the ceiling by the Bid
        # constraint check.
        envelope = budget_envelope or requester_identity or "default"
        if self.budget is not None:
            refusal = self.budget.reserve(
                job_id, envelope, cost_ceiling_usd,
                meta={"kind": kind,
                      "requester": requester_identity},
            )
            if refusal:
                rec = JobRecord(
                    job_id=job_id,
                    kind=kind,
                    payload=payload,
                    cost_ceiling_usd=cost_ceiling_usd,
                    latency_ceiling_ms=latency_ceiling_ms,
                    privacy_class=privacy_class,
                    quality_floor=quality_floor,
                    submitted_at_unix=time.time(),
                    requester_identity=requester_identity,
                    budget_envelope=envelope,
                    state="failed",
                    detail=refusal,
                )
                self.jobs[rec.job_id] = rec
                self._persist(rec)
                return rec
        rec = JobRecord(
            job_id=job_id,
            kind=kind,
            payload=payload,
            cost_ceiling_usd=cost_ceiling_usd,
            latency_ceiling_ms=latency_ceiling_ms,
            privacy_class=privacy_class,
            quality_floor=quality_floor,
            submitted_at_unix=time.time(),
            requester_identity=requester_identity,
            budget_envelope=envelope,
            delta_queue=asyncio.Queue() if streaming else None,
        )
        if streaming:
            from core.delta_cursor import DeltaCursor
            rec.delta_cursor = DeltaCursor()
        self.jobs[job_id] = rec
        # Persist immediately so a crash between submit() and the first
        # state transition still leaves a recoverable record.
        self._persist(rec)
        await self._push_event(rec, "job.queued")

        spec = JobSpec(
            job_id=job_id,
            kind=kind,
            payload=payload,
            cost_ceiling_usd=cost_ceiling_usd,
            latency_ceiling_ms=latency_ceiling_ms,
            privacy_class=privacy_class,
            quality_floor=quality_floor,
        )

        # §A23 consortium path: when the job is large/explicit-shardable
        # (training, multi-hour batch inference, 70B+ models that don't
        # fit on a single GPU), we run N parallel providers and
        # aggregate their results. The auction returns an ordered list
        # of bids; execute_consortium dispatches each member with its
        # shard fraction. Falls back to single-winner for small/normal
        # jobs so latency-sensitive chat completions stay 1-RTT.
        from core.consortium_auction import (
            execute_consortium,
            job_default_sharding_mode,
            job_needs_consortium,
            select_consortium,
            select_scale_to_compute,
        )
        consortium_size = job_needs_consortium(spec)
        if consortium_size is not None:
            sharding_mode = job_default_sharding_mode(spec)
            required_score = float((payload or {}).get(
                "required_compute_score", 0.0,
            ))
            if required_score > 0:
                # Elastic-scale-to-compute path: auction picks the
                # cheapest set of nodes whose summed performance
                # scores meet `required_compute_score`. Multiple
                # GTX 1650s combine to cover an H100-sized job.
                consortium = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: select_scale_to_compute(
                        self.auction, spec,
                        required_compute_score=required_score,
                        sharding_mode=sharding_mode,
                    ),
                )
            else:
                consortium = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: select_consortium(
                        self.auction, spec,
                        target_size=consortium_size,
                        minimum_size=(1 if sharding_mode == "data-parallel"
                                      else consortium_size),
                        sharding_mode=sharding_mode,
                    ),
                )
            if consortium.size == 0:
                rec.state = "failed"
                rec.detail = "no_provider_matched_consortium"
                jobs_total.inc(labels={"status": "no_provider_matched"})
                await self._push_event(rec, "job.failed")
                return rec
            rec.state = "matched"
            rec.matched_provider_pubkey = (
                f"consortium:{consortium.size}:{consortium.sharding_mode}"
            )
            rec.price_locked_usd = sum(m.bid.price_usd for m in consortium.members)
            # Lock total consortium price up front. Refunded fully on
            # complete failure; split per-member on success (each
            # member's share net of commission).
            if self.ledger is not None and buyer_wallet_id:
                from decimal import Decimal
                from core.buyer_ledger import InsufficientFunds
                try:
                    self.ledger.lock_for_job(
                        buyer_wallet_id=buyer_wallet_id, job_id=rec.job_id,
                        amount_usd=Decimal(str(rec.price_locked_usd)),
                    )
                    rec._buyer_wallet_id = buyer_wallet_id
                except InsufficientFunds as e:
                    rec.state = "failed"
                    rec.detail = f"insufficient_funds: {e}"
                    await self._push_event(rec, "job.failed")
                    return rec
            await self._push_event(rec, "job.matched")
            asyncio.create_task(self._run_consortium(rec, spec, consortium))
            return rec

        # Auction is sync (we run it in a thread to keep the loop
        # responsive on a busy node).
        with auction_duration_seconds.time():
            result = await asyncio.get_running_loop().run_in_executor(
                None, self.auction.run, spec,
            )
        rec.auction_result = result
        if not result.is_won():
            rec.state = "failed"
            rec.detail = "no_provider_matched"
            self._budget_terminal(rec)
            jobs_total.inc(labels={"status": "no_provider_matched"})
            await self._push_event(rec, "job.failed")
            return rec
        winner: Bid = result.winner
        rec.state = "matched"
        rec.matched_provider_pubkey = winner.provider_id
        rec.price_locked_usd = float(winner.price_usd)
        # Economic invariant: lock the auction-cleared price from
        # the buyer's wallet BEFORE execution.
        #
        # CRITICAL UX rule: if the wallet doesn't have enough, the
        # job PAUSES (state=`paused_funding`) instead of failing.
        # The buyer gets a clear "top up to continue" status from
        # GET /v1/jobs/{id}; a subsequent credit + `resume_funding`
        # call dispatches the auction. Failing-and-dropping a
        # long-running job because the buyer ran $0.10 short of
        # their next checkpoint is the kind of thing buyers hate
        # — they lose hours of work for a recoverable accounting
        # condition. Pluginfer never does that.
        if self.ledger is not None and buyer_wallet_id:
            from decimal import Decimal
            from core.buyer_ledger import InsufficientFunds
            try:
                self.ledger.lock_for_job(
                    buyer_wallet_id=buyer_wallet_id, job_id=rec.job_id,
                    amount_usd=Decimal(str(winner.price_usd)),
                )
                rec._buyer_wallet_id = buyer_wallet_id
            except InsufficientFunds as e:
                rec.state = "paused_funding"
                rec.detail = (
                    f"wallet_underfunded: top up {buyer_wallet_id} "
                    f"and POST /v1/jobs/{rec.job_id}/resume — "
                    f"need {winner.price_usd} USD. ({e})"
                )
                rec._buyer_wallet_id = buyer_wallet_id
                rec._pending_winner = winner
                rec._pending_spec = spec
                await self._push_event(rec, "job.paused_funding")
                return rec
        await self._push_event(rec, "job.matched")

        # Schedule execution on the loop. Don't await — caller should
        # poll or subscribe to SSE.
        asyncio.create_task(self._run_job(rec, spec, winner))
        return rec

    async def _run_job(self, rec: JobRecord, spec: JobSpec, winner: Bid) -> None:
        # Wrap the WHOLE body so any exception transitions the job to
        # failed instead of leaving it stuck in "running" forever (which
        # an unhandled asyncio task exception will otherwise do --
        # CPython only logs the warning at GC time).
        try:
            rec.state = "running"
            await self._push_event(rec, "job.running")
            provider = next(
                (p for p in self.auction.providers if getattr(p, "provider_id", None) == winner.provider_id),
                None,
            )
            if provider is None:
                rec.state = "failed"
                rec.detail = "winner_not_in_provider_set"
                self._settle_terminal(rec, provider_wallet_id=winner.provider_id)
                await self._push_event(rec, "job.failed")
                return
            # G7: meter energy + carbon around the provider call. The
            # meter source flips to NVML when a GPU is present, falls
            # back to CPU-TDP otherwise. The hardware_class on the
            # provider (when set) selects the right TDP band.
            from core.energy import EnergyMeter
            hw_class = getattr(provider, "hardware_class", "unknown")
            region_zone = getattr(provider, "region_zone", None)
            meter = EnergyMeter(
                hardware_class=hw_class,
                gpu_index=getattr(provider, "gpu_index", 0),
                region_zone=region_zone,
            ).start()
            t0 = time.monotonic()

            # G8: thread an on_delta callback through to providers that
            # opt in. Provider.execute() may accept `on_delta=` as a
            # keyword; we only pass it when the signature advertises
            # support to avoid breaking the existing one-shot path.
            import inspect
            loop = asyncio.get_running_loop()
            def _on_delta(chunk: Any) -> None:
                """Called from the executor thread by streaming providers.
                Pushes the chunk back into the asyncio loop via
                call_soon_threadsafe — asyncio.Queue.put is NOT thread
                safe so we MUST hop loops. ALSO appends to the
                persistent DeltaCursor so reconnect-with-since-cursor
                can replay anything the buyer missed."""
                q = rec.delta_queue
                if q is None:
                    return
                payload = chunk if isinstance(chunk, dict) else {"text": str(chunk)}
                # Persistent cursor write — survives SSE disconnect.
                # Sequence numbers go straight onto the payload so
                # the SSE handler can stamp `id:` lines per RFC.
                if rec.delta_cursor is not None:
                    seq = rec.delta_cursor.append(payload)
                    payload = {**payload, "_seq": seq}
                try:
                    loop.call_soon_threadsafe(q.put_nowait, payload)
                except RuntimeError:
                    pass    # loop already closed (test teardown race)

            execute_kwargs: Dict[str, Any] = {}
            try:
                sig = inspect.signature(provider.execute)
                if "on_delta" in sig.parameters and rec.delta_queue is not None:
                    execute_kwargs["on_delta"] = _on_delta
            except (TypeError, ValueError):
                pass

            # Failover budget: how many alternate providers we'll try
            # if the winner dies mid-execution. Bounded so a dead-mesh
            # scenario doesn't grind forever — defaults to 2, override
            # via PLUGINFER_FAILOVER_RETRIES env. The buyer's escrow
            # stays locked across retries (same job_id, same price);
            # only the executing provider changes.
            failover_budget = int(os.environ.get(
                "PLUGINFER_FAILOVER_RETRIES", "2",
            ))
            attempted_providers = {winner.provider_id}
            current_provider = provider
            current_winner = winner
            out = None
            last_err: Any = None
            while True:
                try:
                    out = await loop.run_in_executor(
                        None,
                        lambda p=current_provider, w=current_winner:
                        p.execute(spec, w, **execute_kwargs),
                    )
                    break
                except Exception as e:
                    last_err = e
                    if failover_budget <= 0:
                        break
                    # Look for a fresh winner among the remaining
                    # providers (excluding the ones we've already
                    # tried). Re-run the bid+score path on a reduced
                    # set so the substitute is the *next best*, not
                    # just random.
                    from core.providers import AuctionResult
                    candidates = [
                        p for p in self.auction.providers
                        if getattr(p, "provider_id", None) not in attempted_providers
                    ]
                    if not candidates:
                        break
                    sub_auction_view = type(self.auction)()
                    for p in candidates:
                        sub_auction_view.register(p)
                    sub_result = sub_auction_view.run(spec)
                    if not sub_result.is_won():
                        break
                    failover_budget -= 1
                    current_winner = sub_result.winner
                    attempted_providers.add(current_winner.provider_id)
                    current_provider = next(
                        (p for p in candidates
                         if getattr(p, "provider_id", None) == current_winner.provider_id),
                        None,
                    )
                    if current_provider is None:
                        break
                    rec.matched_provider_pubkey = current_winner.provider_id
                    rec.detail = (
                        f"failover: switched to {current_winner.provider_id} "
                        f"after {last_err.__class__.__name__}"
                    )
                    await self._push_event(rec, "job.failover")
            if out is None:
                rec.state = "failed"
                rec.detail = (
                    f"failover_exhausted: last_err="
                    f"{type(last_err).__name__}: {last_err}"
                )
                rec.energy_report = meter.stop()
                if rec.delta_queue is not None:
                    rec.delta_queue.put_nowait({
                        "terminal": True, "error": rec.detail,
                    })
                self._settle_terminal(
                    rec, provider_wallet_id=current_winner.provider_id,
                )
                await self._push_event(rec, "job.failed")
                return
            # Pin the actually-executing provider on the record so
            # the receipt + ledger settle against the right wallet.
            winner = current_winner
            rec.execution_ms = (time.monotonic() - t0) * 1000.0
            rec.energy_report = meter.stop()

            if isinstance(out, dict):
                rec.result_hash_hex = out.get("result_hash") or out.get("result_hash_hex")
                # Provider dicts use any of three keys for the b64-encoded
                # result blob — normalise so the SDK / devserver always
                # see it on rec.result_b64.
                rec.result_b64 = (
                    out.get("result_bytes")
                    or out.get("result_bytes_b64")
                    or out.get("result_b64")
                )
                rec.provider_signature_b64 = out.get("provider_sig") or out.get("provider_signature_b64")
                status_str = out.get("status", "executed")
                if status_str in ("executed", "completed", "ok"):
                    rec.state = "completed"
                elif status_str == "timeout":
                    rec.state = "timeout"
                    rec.detail = out.get("reason")
                else:
                    rec.state = "failed"
                    rec.detail = out.get("reason", status_str)
            else:
                rec.state = "completed"
                rec.result_b64 = None  # provider returned non-dict; nothing to ship
        except Exception as e:
            rec.state = "failed"
            rec.detail = f"runtime_error: {type(e).__name__}: {e}"

        # Settle escrow according to the terminal state. Successful job
        # → split lock to (provider, treasury). Failed job → refund
        # buyer in full. Idempotent: a duplicate _run_job (e.g. via a
        # retry from failover) calls into release/refund with the same
        # job_id, which is a no-op on the ledger.
        self._settle_terminal(rec, provider_wallet_id=winner.provider_id)

        rec.completed_at_unix = time.time()
        # Total wall time: submit -> terminal state.
        total = rec.completed_at_unix - rec.submitted_at_unix
        job_duration_seconds.observe(total)
        jobs_total.inc(labels={"status": rec.state})
        # G8: drop the terminal sentinel so SSE consumers can exit
        # their await-loop cleanly. We do this unconditionally on the
        # delta_queue path, even if the provider didn't stream any
        # incremental deltas — the SSE handler in that case emits a
        # single full-result chunk and then the terminal marker.
        if rec.delta_queue is not None:
            try:
                rec.delta_queue.put_nowait({
                    "terminal": True,
                    "state": rec.state,
                    "result_b64": rec.result_b64,
                })
            except Exception:
                pass
        await self._push_event(
            rec, "job.completed" if rec.state == "completed" else f"job.{rec.state}",
        )

    async def dispute(
        self, job_id: str, *, reason: str = "buyer_disputed",
    ) -> Optional[JobRecord]:
        """Buyer disputes a completed job within the dispute window
        (set by `PLUGINFER_DISPUTE_WINDOW_S`, default 300s after
        completion). If the dispute is within window AND the job
        already settled, we REVERSE the settlement: pull the released
        amount back from the provider, refund the buyer, and surface
        the provider on the slashing queue for repeat-offender
        tracking.

        Idempotent on job_id. Returns the JobRecord at its new state
        (disputed_refunded), or None if the job doesn't exist."""
        rec = self.jobs.get(job_id)
        if rec is None:
            return None
        dispute_window_s = float(os.environ.get(
            "PLUGINFER_DISPUTE_WINDOW_S", "300",
        ))
        if rec.state not in ("completed", "completed_partial"):
            rec.detail = f"dispute_rejected_state={rec.state}"
            return rec
        if rec.completed_at_unix and (
            time.time() - rec.completed_at_unix > dispute_window_s
        ):
            rec.detail = "dispute_window_closed"
            return rec
        if self.ledger is not None:
            esc = self.ledger.escrow_for(rec.job_id)
            buyer_id = getattr(rec, "_buyer_wallet_id", None)
            if esc is not None and esc.state == "released" and buyer_id:
                # Pull back the provider's earnings + treasury's
                # commission, refund the buyer. This is the punitive
                # case: provider delivered something, buyer says it
                # was wrong, we side with the buyer and slash the
                # provider's pending balance.
                from decimal import Decimal
                from core.buyer_ledger import (
                    TREASURY_WALLET_ID, WalletEntry,
                )
                # Build the list of (provider_id, paid_amount) to claw
                # back from — same shape whether single-winner or
                # consortium so the loop is uniform.
                claw_targets: List[Any] = []
                if esc.consortium_members:
                    c_rate = float(os.environ.get(
                        "PLUGINFER_COMMISSION_RATE", "0.10",
                    ))
                    for pid, share in esc.consortium_members:
                        paid_share = share * (Decimal("1") - Decimal(str(c_rate)))
                        claw_targets.append((pid, paid_share))
                elif esc.provider_wallet_id:
                    claw_targets.append(
                        (esc.provider_wallet_id, esc.provider_amount or Decimal("0"))
                    )
                if claw_targets:
                    total_pulled = Decimal("0")
                    for provider_id, paid in claw_targets:
                        prov = self.ledger.get_or_create_wallet(provider_id)
                        pullback = min(prov.available_usd, paid)
                        prov.available_usd -= pullback
                        prov.entries.append(WalletEntry(
                            kind="debit", amount_usd=pullback,
                            timestamp_unix=time.time(), job_id=rec.job_id,
                            note="disputed_refund_pullback",
                        ))
                        total_pulled += pullback
                    # Treasury return.
                    treas = self.ledger.get_or_create_wallet(
                        TREASURY_WALLET_ID, role="treasury",
                    )
                    commission_total = esc.commission_amount or Decimal("0")
                    treas_pullback = min(treas.available_usd, commission_total)
                    treas.available_usd -= treas_pullback
                    treas.entries.append(WalletEntry(
                        kind="debit", amount_usd=treas_pullback,
                        timestamp_unix=time.time(), job_id=rec.job_id,
                        note="disputed_refund_commission_return",
                    ))
                    buyer = self.ledger.get_or_create_wallet(buyer_id)
                    refund = total_pulled + treas_pullback
                    buyer.available_usd += refund
                    buyer.entries.append(WalletEntry(
                        kind="refund", amount_usd=refund,
                        timestamp_unix=time.time(), job_id=rec.job_id,
                        note=f"buyer dispute: {reason}",
                    ))
                    esc.state = "disputed_refunded"
                    # Compute shortfall = locked − pulled. If providers
                    # had already spent some earnings, this is what the
                    # slashing pipeline owes the buyer.
                    locked_total = esc.locked_usd
                    if (total_pulled + treas_pullback) < locked_total:
                        rec.detail = (
                            f"disputed; shortfall="
                            f"{locked_total - total_pulled - treas_pullback}"
                        )
        rec.state = "disputed_refunded"
        await self._push_event(rec, "job.disputed")
        return rec

    async def resume_funding(self, job_id: str) -> Optional[JobRecord]:
        """Buyer topped up their wallet; re-attempt the lock and dispatch
        execution. Idempotent: if the job is no longer paused, returns
        the current state without side-effects.

        Returns the JobRecord at its new state, or None if no such job.
        """
        rec = self.jobs.get(job_id)
        if rec is None:
            return None
        if rec.state != "paused_funding":
            return rec
        winner = getattr(rec, "_pending_winner", None)
        spec = getattr(rec, "_pending_spec", None)
        buyer_id = getattr(rec, "_buyer_wallet_id", None)
        if winner is None or spec is None or buyer_id is None:
            rec.state = "failed"
            rec.detail = "paused_funding_metadata_missing"
            await self._push_event(rec, "job.failed")
            return rec
        if self.ledger is not None:
            from decimal import Decimal
            from core.buyer_ledger import InsufficientFunds
            try:
                self.ledger.lock_for_job(
                    buyer_wallet_id=buyer_id, job_id=rec.job_id,
                    amount_usd=Decimal(str(winner.price_usd)),
                )
            except InsufficientFunds as e:
                rec.detail = f"still_underfunded: {e}"
                await self._push_event(rec, "job.paused_funding")
                return rec
        rec.state = "matched"
        rec.detail = None
        await self._push_event(rec, "job.matched")
        asyncio.create_task(self._run_job(rec, spec, winner))
        return rec

    async def abandon(self, job_id: str, *, deliver_partial: bool = True) -> Optional[JobRecord]:
        """Buyer (or a sweeper) gave up on a paused job. We do NOT
        silently drop the work: any partial output collected so far
        is preserved on the JobRecord and made available via
        GET /v1/jobs/{id}. Successful consortium shards stay paid;
        the rest is refunded to the buyer.

        After a long grace period (default 24h, env-tunable via
        PLUGINFER_PAUSE_GRACE_S), a background sweep calls this
        automatically so wallets aren't billed forever for
        unresumable jobs."""
        rec = self.jobs.get(job_id)
        if rec is None:
            return None
        if rec.state not in ("paused_funding", "queued", "matched"):
            return rec
        # No lock was placed (paused_funding state). Nothing to refund.
        rec.state = (
            "abandoned_partial" if deliver_partial and rec.result_b64
            else "abandoned"
        )
        rec.detail = "buyer_abandoned_before_resume"
        rec.completed_at_unix = time.time()
        await self._push_event(rec, "job.abandoned")
        return rec

    def _settle_terminal(self, rec: JobRecord,
                         *, provider_wallet_id: str,
                         consortium_members: Optional[List[Any]] = None) -> None:
        """Drain the escrow at terminal state. Safe to call multiple
        times for the same job (the ledger is idempotent per job_id).
        For consortium jobs, callers pass `consortium_members` with
        the (provider_id, price_usd) tuples to split correctly."""
        # §RFC-3 — budget settlement rides the SAME terminal choke
        # point as escrow, and runs even with no money ledger attached
        # (governance without economics is the gateway's whole mode).
        self._budget_terminal(rec, consortium_members=consortium_members)
        if self.ledger is None:
            return
        buyer_id = getattr(rec, "_buyer_wallet_id", None)
        if not buyer_id:
            return
        from decimal import Decimal
        try:
            if rec.state in ("completed", "completed_partial"):
                if consortium_members:
                    self.ledger.split_release_to_consortium(
                        job_id=rec.job_id,
                        members=[
                            (str(pid), Decimal(str(price)))
                            for pid, price in consortium_members
                        ],
                    )
                else:
                    self.ledger.release_to_provider(
                        job_id=rec.job_id,
                        provider_wallet_id=provider_wallet_id,
                    )
            else:
                self.ledger.refund_to_buyer(job_id=rec.job_id)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "ledger settlement failed for %s: %s", rec.job_id, e,
            )

    def _budget_terminal(self, rec: JobRecord,
                         consortium_members: Optional[List[Any]] = None
                         ) -> None:
        """§RFC-3: settle (completed → actual cleared price) or release
        (any other terminal state) the budget hold. Idempotent — safe
        from both _settle_terminal and the no-provider early exit.
        Reservations also self-expire in the ledger, so a path this
        misses can never wedge an envelope permanently."""
        if self.budget is None:
            return
        try:
            if rec.state in ("completed", "completed_partial"):
                actual = rec.price_locked_usd
                if actual is None and consortium_members:
                    actual = sum(float(p) for _, p in consortium_members)
                self.budget.settle(
                    rec.job_id, float(actual or 0.0),
                    meta={"provider": rec.matched_provider_pubkey or ""},
                )
            else:
                self.budget.release(rec.job_id)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "budget settlement failed for %s: %s", rec.job_id, e,
            )

    def quote(
        self,
        *,
        kind: str,
        payload: Dict[str, Any],
        cost_ceiling_usd: float,
        latency_ceiling_ms: int = 30_000,
        privacy_class: str = "public",
        quality_floor: float = 0.7,
    ) -> Dict[str, Any]:
        """§RFC-3 quote-before-run: price a job WITHOUT executing it.

        Runs the exact same sealed-bid auction submit() would run —
        same providers, same constraint checks, same scoring — and
        returns the clearing price instead of dispatching the winner.
        No JobRecord, no escrow, no budget hold, no execution: the
        auction collects bids only (Auction.run is side-effect-free).

        The price is a market snapshot, not an escrowed lock — bids
        can move as providers load up, hence ttl_hint_s. No cloud API
        offers even this; the mesh gets it for free because pricing
        already happens before execution.
        """
        spec = JobSpec(
            job_id="quote-" + secrets.token_urlsafe(8),
            kind=kind,
            payload=payload,
            cost_ceiling_usd=cost_ceiling_usd,
            latency_ceiling_ms=latency_ceiling_ms,
            privacy_class=privacy_class,
            quality_floor=quality_floor,
        )
        result = self.auction.run(spec)
        if not result.is_won():
            return {
                "would_clear": False,
                "price_usd": None,
                "reasons": [
                    {"provider_id": r.get("provider_id", "?"),
                     "reason": str(r.get("reason", ""))}
                    for r in result.rejected
                ],
            }
        w = result.winner
        return {
            "would_clear": True,
            "price_usd": float(w.price_usd),
            "provider_id": w.provider_id,
            "eta_ms": int(w.eta_ms),
            "expected_quality": float(w.expected_quality),
            "competing_bids": len(result.bids),
            "ttl_hint_s": 60,
        }

    async def _run_consortium(self, rec: JobRecord, spec: JobSpec,
                              consortium: Any) -> None:
        """Run a sharded job across N providers concurrently. Each
        member's `execute()` runs in the default thread pool; the
        aggregate combines per `consortium.sharding_mode`. Tolerates
        partial failures: if at least one member completes, the job
        is `completed_partial` (carries the combined bytes from the
        survivors). If none complete, `failed`."""
        from core.consortium_auction import execute_consortium
        try:
            rec.state = "running"
            await self._push_event(rec, "job.running")
            t0 = time.monotonic()
            try:
                exec_result = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: execute_consortium(self.auction, consortium, spec),
                )
            except NotImplementedError as e:
                rec.state = "failed"
                rec.detail = f"consortium_mode_unsupported: {e}"
                await self._push_event(rec, "job.failed")
                return
            except Exception as e:
                rec.state = "failed"
                rec.detail = f"consortium_execute_error: {type(e).__name__}: {e}"
                await self._push_event(rec, "job.failed")
                return
            rec.execution_ms = (time.monotonic() - t0) * 1000.0
            successful = [
                m for m in exec_result.per_member
                if m.get("status") in ("executed", "completed", "ok")
            ]
            if not successful:
                rec.state = "failed"
                rec.detail = "all_consortium_members_failed"
                await self._push_event(rec, "job.failed")
                return
            import base64
            if exec_result.combined_result_bytes is not None:
                rec.result_b64 = base64.b64encode(
                    exec_result.combined_result_bytes,
                ).decode("ascii")
                rec.result_hash_hex = exec_result.combined_result_hash
            rec.state = (
                "completed" if not exec_result.failed_members
                else "completed_partial"
            )
            rec.detail = (
                f"consortium {len(successful)}/{consortium.size} ok, "
                f"mode={consortium.sharding_mode}"
            )
        except Exception as e:
            rec.state = "failed"
            rec.detail = f"runtime_error: {type(e).__name__}: {e}"

        # Split escrow across the consortium members proportionally
        # to each member's bid (so a higher-quality member earns more
        # for the same job, exactly as the auction priced it). Failed
        # members get nothing; their shares are auto-refunded to the
        # buyer's available balance by split_release_to_consortium.
        successful_members: List[Any] = []
        if rec.state in ("completed", "completed_partial"):
            successful_provider_ids = {
                e.get("provider_id") for e in exec_result.per_member
                if e.get("status") in ("executed", "completed", "ok")
            }
            for m in consortium.members:
                if m.provider_id in successful_provider_ids:
                    successful_members.append((m.provider_id, m.bid.price_usd))
        self._settle_terminal(
            rec, provider_wallet_id="",
            consortium_members=successful_members or None,
        )

        rec.completed_at_unix = time.time()
        total = rec.completed_at_unix - rec.submitted_at_unix
        job_duration_seconds.observe(total)
        jobs_total.inc(labels={"status": rec.state})
        await self._push_event(
            rec,
            "job.completed" if rec.state.startswith("completed")
            else f"job.{rec.state}",
        )

    def get(self, job_id: str) -> Optional[JobRecord]:
        return self.jobs.get(job_id)

    async def cancel(self, job_id: str) -> bool:
        rec = self.jobs.get(job_id)
        if rec is None:
            return False
        if rec.state in ("completed", "failed", "cancelled", "timeout"):
            return False
        rec.state = "cancelled"
        rec.detail = "cancelled_by_user"
        await self._push_event(rec, "job.cancelled")
        return True

    async def subscribe(self, job_id: str) -> asyncio.Queue:
        rec = self.jobs.get(job_id)
        if rec is None:
            raise KeyError(job_id)
        q: asyncio.Queue = asyncio.Queue(maxsize=128)
        rec._watchers.append(q)
        # Replay current state so the subscriber doesn't miss the
        # transition that already happened.
        await q.put({"event": f"job.{rec.state}", "job": rec.to_info(),
                     "result": rec.to_result()})
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue) -> None:
        rec = self.jobs.get(job_id)
        if rec is None:
            return
        try:
            rec._watchers.remove(q)
        except ValueError:
            pass

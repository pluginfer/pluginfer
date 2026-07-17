"""
Payments — fiat on/off-ramps
============================
Previous version shipped `StripeMockGateway` that did `time.sleep(1)`
then "Simulate success". Removing the mock — anything that uses it
in production silently passes payments that never happened.

For v3 the payment surfaces are:

    * **Native PLG** — handled by `tokenomics.py` and `compute_ledger.py`.
      Most network economics happens here; no fiat involved.

    * **Fiat onramp** — defer to a real PSP (Stripe, MoonPay, Ramp).
      Plug yours in by implementing the `PaymentGateway` interface
      below; configure via env var.

    * **Crypto onramp** — defer to native chain. Once PLG migrates to
      Solana (Phase 1), buyers acquire it on a DEX (Jupiter / Raydium).

This module deliberately ships only the abstract interface and a
real Stripe wrapper that REQUIRES configured credentials. No mocks,
no fake flows. If credentials aren't set, you get an exception with
a clear remediation message.
"""

from __future__ import annotations

import abc
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PaymentResult:
    success: bool
    transaction_id: Optional[str]
    error: Optional[str] = None
    idempotent_replay: bool = False  # True when a cached prior result was returned


class PaymentGatewayNotConfigured(RuntimeError):
    """Raised when a payment method is invoked without real credentials."""


class IdempotencyStore:
    """Process-local (caller_id, key) -> (result, expires_at) store.

    Production deployments back this with Redis or a chain-anchored
    ledger so that a charge issued from any node never double-bills,
    even across crashes. The in-memory implementation is correct for
    a single-process gateway; the interface is the same so swapping
    in a Redis-backed subclass is a one-method override.

    Key TTL defaults to 24 hours per Stripe's idempotency contract
    (https://stripe.com/docs/api/idempotent_requests). Long enough
    that retried payments after a crash collapse to the original
    charge; short enough that the keyspace doesn't grow unbounded.
    """

    DEFAULT_TTL_S = 24 * 60 * 60   # 24 hours

    def __init__(self, ttl_s: int = DEFAULT_TTL_S):
        self._ttl_s = ttl_s
        self._store: Dict[Tuple[str, str], Tuple[PaymentResult, float]] = {}
        self._lock = threading.RLock()

    def get(self, customer_id: str, key: str) -> Optional[PaymentResult]:
        with self._lock:
            self._evict_expired_locked()
            entry = self._store.get((customer_id, key))
            if entry is None:
                return None
            result, _ = entry
            # Mark the replay so callers can distinguish "real new charge"
            # from "cached prior charge" — important for receipt logs.
            return PaymentResult(
                success=result.success,
                transaction_id=result.transaction_id,
                error=result.error,
                idempotent_replay=True,
            )

    def put(self, customer_id: str, key: str, result: PaymentResult) -> None:
        with self._lock:
            self._store[(customer_id, key)] = (result, time.time() + self._ttl_s)

    def _evict_expired_locked(self) -> None:
        now = time.time()
        # Cheap O(n) sweep; for production with >100k keys swap in a
        # priority queue or back with Redis EXPIRE.
        expired = [k for k, (_, exp) in self._store.items() if exp <= now]
        for k in expired:
            del self._store[k]

    def __len__(self) -> int:
        with self._lock:
            self._evict_expired_locked()
            return len(self._store)


# Module-level default store — gateways without an injected store fall
# back to this so retried charges across short-lived gateway instances
# still collapse to the original transaction.
_DEFAULT_STORE = IdempotencyStore()


class PaymentGateway(abc.ABC):
    """Abstract interface. Implementations live in `core/payments_*.py`.

    All gateways MUST honor the optional ``idempotency_key`` parameter:
    a charge call with the same (customer_id, key) returns the original
    PaymentResult instead of creating a second charge. Required for
    safe retry on network blips, gateway timeouts, and node crashes.
    """

    def __init__(self, idempotency_store: Optional[IdempotencyStore] = None):
        self._idem = idempotency_store or _DEFAULT_STORE

    @abc.abstractmethod
    def charge(self, amount_usd: float, currency: str, customer_id: str,
               description: str = "",
               idempotency_key: Optional[str] = None) -> PaymentResult:
        ...

    @abc.abstractmethod
    def refund(self, transaction_id: str,
               amount_usd: Optional[float] = None,
               idempotency_key: Optional[str] = None) -> PaymentResult:
        ...


class StripeGateway(PaymentGateway):
    """
    Real Stripe gateway. Requires:
        pip install stripe
        env: PLUGINFER_STRIPE_SECRET_KEY=sk_live_...
    """

    def __init__(self, api_key: Optional[str] = None,
                 idempotency_store: Optional[IdempotencyStore] = None):
        super().__init__(idempotency_store=idempotency_store)
        api_key = api_key or os.environ.get("PLUGINFER_STRIPE_SECRET_KEY")
        if not api_key:
            raise PaymentGatewayNotConfigured(
                "Set PLUGINFER_STRIPE_SECRET_KEY to enable Stripe payments."
            )
        try:
            import stripe                               # type: ignore
        except ImportError as e:
            raise PaymentGatewayNotConfigured(
                "stripe-python is not installed. `pip install stripe`."
            ) from e
        stripe.api_key = api_key
        self._stripe = stripe

    def charge(self, amount_usd: float, currency: str, customer_id: str,
               description: str = "",
               idempotency_key: Optional[str] = None) -> PaymentResult:
        # Idempotency: if we've already processed this (customer, key),
        # return the cached result. Stripe ALSO de-duplicates on its
        # side via Idempotency-Key header, so even if our cache is
        # missed (e.g. crash after Stripe accepted but before we cached),
        # the upstream call collapses too. Belt + suspenders.
        if idempotency_key:
            cached = self._idem.get(customer_id, idempotency_key)
            if cached is not None:
                return cached
        try:
            kwargs = {
                "amount": int(round(amount_usd * 100)),
                "currency": currency.lower(),
                "customer": customer_id,
                "description": description,
                "automatic_payment_methods": {"enabled": True},
            }
            create_kwargs = {}
            if idempotency_key:
                create_kwargs["idempotency_key"] = idempotency_key
            intent = self._stripe.PaymentIntent.create(**kwargs, **create_kwargs)
            result = PaymentResult(True, intent["id"])
        except Exception as e:
            logger.exception("Stripe charge failed")
            result = PaymentResult(False, None, str(e))
        if idempotency_key:
            self._idem.put(customer_id, idempotency_key, result)
        return result

    def refund(self, transaction_id: str,
               amount_usd: Optional[float] = None,
               idempotency_key: Optional[str] = None) -> PaymentResult:
        # Refunds are de-duplicated by transaction_id as the natural
        # idempotency key when none is supplied — refunding the same
        # transaction twice should never double-credit the customer.
        effective_key = idempotency_key or f"refund:{transaction_id}"
        cached = self._idem.get(transaction_id, effective_key)
        if cached is not None:
            return cached
        try:
            kwargs = {"payment_intent": transaction_id}
            if amount_usd is not None:
                kwargs["amount"] = int(round(amount_usd * 100))
            create_kwargs = {"idempotency_key": effective_key}
            refund = self._stripe.Refund.create(**kwargs, **create_kwargs)
            result = PaymentResult(True, refund["id"])
        except Exception as e:
            logger.exception("Stripe refund failed")
            result = PaymentResult(False, None, str(e))
        self._idem.put(transaction_id, effective_key, result)
        return result


def get_default_gateway() -> Optional[PaymentGateway]:
    """Return a real gateway iff configured. Else None — never a mock."""
    if os.environ.get("PLUGINFER_STRIPE_SECRET_KEY"):
        try:
            return StripeGateway()
        except PaymentGatewayNotConfigured as e:
            logger.warning("Stripe configured but unusable: %s", e)
    return None

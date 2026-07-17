"""RETIRED in CP-1.

The original test imported `core.payments.StripeMockGateway`. Pre-W31
hardening that class was a mock; the audit found it could double-charge
on retries. Replaced with `core.payments.StripeGateway` (real Stripe
wrapper that refuses to operate without a configured key + idempotency
key) which raises `PaymentGatewayNotConfigured` rather than synthesise
a fake receipt.

The replacement payment surface is tested via:
  - tests/test_e2e_product.py::test_payment_settles_on_chain (signed
    on-chain payment via PLG; off-ramp to fiat is a separate concern)
  - core/payments.py docstring tests

Keeping this file so git history preserves the deletion intent.
"""

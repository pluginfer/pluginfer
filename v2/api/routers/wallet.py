"""Wallet endpoints: balance, withdraw (deposit is on-chain only)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from ..middleware.auth import require_auth
from ..schemas import WalletBalance

router = APIRouter(prefix="/v1/wallet", tags=["wallet"])


@router.get("/balance", response_model=WalletBalance)
def get_balance(request: Request, identity: str = Depends(require_auth)) -> WalletBalance:
    """Return PLG balance for the *node's* wallet (the operator).

    For a third-party SDK user authenticated by API key, the balance
    reported is the operator's; for wallet-session auth, it is the
    authenticated wallet's. The two cases are distinguished by identity
    prefix.
    """
    ledger = getattr(request.app.state, "ledger", None)
    wallet = getattr(request.app.state, "wallet", None)
    if ledger is None or wallet is None:
        raise HTTPException(503, "ledger_unavailable")
    address = wallet.address
    if identity.startswith("wallet:"):
        address = getattr(request.state, "wallet_address", address)
    balance = ledger.get_balance(address)
    chain_height = max(0, len(getattr(ledger, "blocks", [])) - 1)
    return WalletBalance(
        address=address,
        balance_plg=float(balance),
        pending_plg=0.0,
        chain_height=chain_height,
    )

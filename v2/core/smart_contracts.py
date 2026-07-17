"""
Smart Contract Engine (Mini-VM) — chain-derived storage edition.

W23 closes the structural gap: the previous version persisted contracts
to a local JSON file (`contracts.json`). On a multi-node mesh that
diverged silently — every node had its own truth, and chain reorgs
left storage stale.

Now:
  * **Deploy** emits a `DeployContractTx`. Address is deterministic:
    ``"C" + sha256(owner || code_hash || nonce)[:39]``. Owner identity
    is asserted by the ECDSA signature on the tx (validated by
    `ComputeLedger._validate_deploy_contract_tx`).
  * **Execute** emits an `ExecuteContractTx` with the function name
    and args. The VM runs the code in `SecureSandbox` to compute
    `new_storage`; that diff goes into the tx payload so other nodes
    re-applying the chain converge to the same storage without re-
    running the sandbox (deterministic by virtue of being committed).
  * **Storage** is the chain's view via `ComputeLedger.contract_state`.
    No local JSON. Reorgs invalidate the cache via the same path that
    rebuilds balances + nonces.

The local file mode still works for unit-test scenarios where no
ledger is present (back-compat). Any production deployment should
construct `SmartContractVM(ledger=...)` and rely on chain state.
"""
import hashlib
import logging
import json
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

import os


@dataclass
class DeployContractPayload:
    owner: str
    code: str
    code_hash: str
    address: str
    initial_storage: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecuteContractPayload:
    address: str
    function_name: str
    args: List[Any]
    new_storage: Optional[Dict[str, Any]] = None
    return_value: Any = None


def derive_contract_address(owner: str, code_hash: str, nonce: int) -> str:
    """Deterministic, collision-resistant contract address.
    Same scheme the ledger validator recomputes for the
    deploy-tx address-mismatch check."""
    return "C" + hashlib.sha256(
        f"{owner}|{code_hash}|{nonce}".encode("utf-8")
    ).hexdigest()[:39]


def hash_contract_code(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


class SmartContractVM:
    def __init__(self, ledger=None, storage_file="contracts.json"):
        """
        :param ledger: ComputeLedger — when provided, contract state is
            derived from chain transactions (W23). When absent, falls
            back to the legacy local-JSON mode (single-node tests only).
        :param storage_file: legacy fallback path; ignored when ledger
            is provided.
        """
        self.ledger = ledger
        self.storage_file = storage_file
        self.contracts: Dict[str, Dict] = {}
        if ledger is None:
            self.load_state()

    @property
    def _state(self) -> Dict[str, Dict]:
        """Chain-derived view when ledger is wired; local JSON otherwise."""
        if self.ledger is not None:
            return self.ledger.contract_state
        return self.contracts

    def load_state(self):
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r') as f:
                    self.contracts = json.load(f)
                logger.info(f"[VM] Loaded {len(self.contracts)} contracts")
            except Exception as e:
                logger.error(f"[VM] Load failed: {e}")

    def save_state(self):
        if self.ledger is not None:
            return  # chain-derived; no local persistence
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(self.contracts, f, indent=4)
        except Exception as e:
            logger.error(f"[VM] Save failed: {e}")

    def deploy_contract(self, owner: str, code: str,
                        initial_storage: Dict = None,
                        nonce: int = 0) -> str:
        """Deploy a new contract. Returns Contract Address.

        Address is deterministic in (owner, code_hash, nonce) so the
        same caller redeploying the same code with the same nonce gets
        the same address — a recipe for the chain validator to detect
        accidental redeploys.
        """
        code_hash = hash_contract_code(code)
        contract_addr = derive_contract_address(owner, code_hash, nonce)
        # Local-mode fallback: write straight to in-memory contracts.
        # Ledger-mode callers should build a DeployContractTx via
        # `build_deploy_contract_tx` and submit it through the chain;
        # the validator + _apply_contract_tx_to_cache do the rest.
        if self.ledger is None:
            self.contracts[contract_addr] = {
                'code': code, 'code_hash': code_hash,
                'storage': initial_storage or {}, 'owner': owner,
            }
            self.save_state()
        logger.info(f"[VM] Deployed Contract {contract_addr}")
        return contract_addr
        
    def execute(self, contract_addr: str, function_name: str, args: list = []) -> Any:
        """
        Execute a contract function via the AST + multiprocessing sandbox.

        Previous version used exec() in a worker thread, which has the same
        attack surface as the old dynamic_executor RCE: arbitrary contract
        code could touch host filesystem, network, etc. Routing through
        `core.secure_sandbox.SecureSandbox` enforces:
            * AST static analysis (rejects os/sys/subprocess/socket/...)
            * Process-level isolation (SIGKILL-able subprocess)
            * Hard timeout

        Storage is read-only inside the sandbox in this version. State
        mutation requires returning a `new_storage` dict via the contract's
        well-known protocol (assign to `result = (return_value, new_storage)`).
        That keeps state transitions auditable on chain.
        """
        from .secure_sandbox import SecureSandbox, SecurityViolation

        contract = self._state.get(contract_addr)
        if not contract:
            raise ValueError("Contract not found")

        # Wrap the user's contract code so the sandbox protocol invokes
        # the right function with `args`, passes `storage` and `msg`,
        # and assigns the result variable that SecureSandbox returns.
        #
        # The hardened sandbox (W22) bans `dir()` and any `__dunder__`
        # attribute access, so the previous "if name in dir():" /
        # ".__code__.co_varnames" pattern no longer works. Replaced
        # with a try/except TypeError that handles signatures both with
        # and without (storage, msg).
        wrapper = (
            f"{contract['code']}\n"
            f"_storage = {repr(contract['storage'])}\n"
            f"_msg = {{'sender': 'USER_SIM'}}\n"
            f"try:\n"
            f"    _ret = {function_name}(*args, storage=_storage, msg=_msg)\n"
            f"except TypeError:\n"
            f"    _ret = {function_name}(*args)\n"
            f"# Contracts may return (value, new_storage) to update state.\n"
            f"if isinstance(_ret, tuple) and len(_ret) == 2 and isinstance(_ret[1], dict):\n"
            f"    result = {{'ret': _ret[0], 'new_storage': _ret[1]}}\n"
            f"else:\n"
            f"    result = {{'ret': _ret, 'new_storage': None}}\n"
        )

        try:
            sandbox_result = SecureSandbox.run(wrapper, args=list(args), timeout=5.0)
        except SecurityViolation as sv:
            logger.error("[VM] %s rejected by sandbox: %s", contract_addr, sv)
            raise
        except TimeoutError:
            logger.error("[VM] Contract %s timed out", contract_addr)
            raise
        except Exception as e:
            logger.error("[VM] Execution failed: %s", e)
            raise

        ret_val = sandbox_result.get("ret") if isinstance(sandbox_result, dict) else sandbox_result
        new_storage = sandbox_result.get("new_storage") if isinstance(sandbox_result, dict) else None
        if isinstance(new_storage, dict):
            if self.ledger is None:
                # Local-mode: write straight to in-memory contracts.
                contract['storage'] = new_storage
                self.save_state()
            # Ledger-mode: callers should now build an
            # ExecuteContractTx with `new_storage` baked in via
            # `build_execute_contract_tx` and submit through the
            # chain. The validator + _apply_contract_tx_to_cache
            # commit the storage transition deterministically.
        return ret_val


# ---------------------------------------------------------------------------
# Chain-tx builders (W23). These produce dicts the chain validator
# accepts as deploy_contract / execute_contract transactions. The
# caller signs the resulting tx_id with their wallet and broadcasts.
# ---------------------------------------------------------------------------


def build_deploy_contract_tx(
    *, owner_wallet, code: str,
    initial_storage: Optional[Dict[str, Any]] = None,
    nonce: int = 0, fee: str = "0.001",
) -> Dict[str, Any]:
    """Construct a signed deploy_contract transaction. The owner_wallet
    must expose `address`, `public_key_pem` (PEM string), and a `sign(msg)`
    method that returns base64 ECDSA-SHA256 over `msg`."""
    from .tokenomics import Transaction
    code_hash = hash_contract_code(code)
    address = derive_contract_address(owner_wallet.address, code_hash, nonce)
    payload = {
        "owner": owner_wallet.address,
        "code": code,
        "code_hash": code_hash,
        "address": address,
        "initial_storage": dict(initial_storage or {}),
    }
    timestamp = time.time()
    tx_id = Transaction.calculate_tx_hash(
        owner_wallet.address, address, "0", fee,
        timestamp, "deploy_contract",
        owner_wallet.public_key_pem, nonce,
    )
    sig = owner_wallet.sign(tx_id)
    return {
        "type": "deploy_contract",
        "tx_id": tx_id,
        "sender": owner_wallet.address,
        "recipient": address,
        "amount": "0",
        "fee": fee,
        "timestamp": timestamp,
        "nonce": nonce,
        "sender_pub_key": owner_wallet.public_key_pem,
        "signature": sig,
        "payload": payload,
    }


def build_execute_contract_tx(
    *, caller_wallet, contract_addr: str,
    function_name: str, args: List[Any],
    new_storage: Optional[Dict[str, Any]] = None,
    return_value: Any = None,
    nonce: int = 0, fee: str = "0.001",
) -> Dict[str, Any]:
    """Construct a signed execute_contract transaction. The caller is
    responsible for running the function in `SmartContractVM.execute`
    first to obtain `new_storage`/`return_value`; that result is
    embedded in the payload so the chain converges deterministically
    without every replicating node re-running the sandbox.
    """
    from .tokenomics import Transaction
    payload = {
        "address": contract_addr,
        "function_name": function_name,
        "args": list(args),
        "new_storage": dict(new_storage) if isinstance(new_storage, dict) else None,
        "return_value": return_value,
    }
    timestamp = time.time()
    tx_id = Transaction.calculate_tx_hash(
        caller_wallet.address, contract_addr, "0", fee,
        timestamp, "execute_contract",
        caller_wallet.public_key_pem, nonce,
    )
    sig = caller_wallet.sign(tx_id)
    return {
        "type": "execute_contract",
        "tx_id": tx_id,
        "sender": caller_wallet.address,
        "recipient": contract_addr,
        "amount": "0",
        "fee": fee,
        "timestamp": timestamp,
        "nonce": nonce,
        "sender_pub_key": caller_wallet.public_key_pem,
        "signature": sig,
        "payload": payload,
    }


def build_slash_tx(
    *, slash_payload: Dict[str, Any], fee: str = "0",
    nonce: int = 0,
) -> Dict[str, Any]:
    """Wrap a slash_evidence.apply_slash() result into a chain
    transaction. Sender = 'BFT_SLASH_PROTOCOL' so validators cannot
    forge a slash from a regular wallet; the validator path enforces
    the ≥2/3 quorum check on the embedded evidence.
    """
    from .tokenomics import Transaction
    timestamp = time.time()
    tx_id = Transaction.calculate_tx_hash(
        "BFT_SLASH_PROTOCOL", slash_payload.get("offender_addr", ""),
        "0", fee, timestamp, "slash", "SYSTEM", nonce,
    )
    return {
        "type": "slash",
        "tx_id": tx_id,
        "sender": "BFT_SLASH_PROTOCOL",
        "recipient": slash_payload.get("offender_addr", ""),
        "amount": "0",
        "fee": fee,
        "timestamp": timestamp,
        "nonce": nonce,
        "sender_pub_key": "SYSTEM",
        "signature": "",
        "payload": slash_payload,
    }

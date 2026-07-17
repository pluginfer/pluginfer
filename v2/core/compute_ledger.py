"""
Compute Ledger (The "Local Blockchain")
Ensures tamper-proof task history and correct ordering of sharded results.
Now supports Full Blockchain Features: Merkle Trees, Transactions, and Difficulty.
"""
import hashlib
import json
import time
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from decimal import Decimal
from .tokenomics import Transaction, Wallet

logger = logging.getLogger(__name__)

class Block:
    def __init__(self, index: int, previous_hash: str, transactions: List[Dict], timestamp: float = None, difficulty: int = 1, nonce: int = 0):
        self.index = index
        self.timestamp = timestamp or time.time()
        self.transactions = transactions # List of Transaction dicts
        self.previous_hash = previous_hash
        self.difficulty = difficulty
        self.nonce = nonce
        self.merkle_root = self.calculate_merkle_root()
        self.hash = self.calculate_hash()

    def calculate_merkle_root(self) -> str:
        """Calculate Merkle Root of transactions"""
        if not self.transactions:
            return hashlib.sha256(b"").hexdigest()
            
        # Get list of tx hashes
        hashes = [tx.get('tx_id') for tx in self.transactions]
        
        if not hashes:
            return hashlib.sha256(b"").hexdigest()
            
        while len(hashes) > 1:
            if len(hashes) % 2 != 0:
                hashes.append(hashes[-1]) # Duplicate last if odd
            new_hashes = []
            for i in range(0, len(hashes), 2):
                concat = hashes[i] + hashes[i+1]
                new_hashes.append(hashlib.sha256(concat.encode()).hexdigest())
            hashes = new_hashes
            
        return hashes[0]

    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of the block header"""
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "merkle_root": self.merkle_root,
            "previous_hash": self.previous_hash,
            "difficulty": self.difficulty,
            "nonce": self.nonce
        }, sort_keys=True).encode()
        
        return hashlib.sha256(block_string).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "transactions": self.transactions,
            "previous_hash": self.previous_hash,
            "merkle_root": self.merkle_root,
            "difficulty": self.difficulty,
            "nonce": self.nonce,
            "hash": self.hash
        }

class ComputeLedger:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        # sec1.5 / sec3.3: cached account state.
        # _balances: per-address PLG balance (Decimal).
        # _nonces: per-sender max nonce seen in a confirmed transfer.
        # Both are derived from chain state and recomputed on reorg.
        self._balances: Dict[str, Decimal] = {}
        self._nonces: Dict[str, int] = {}
        self._create_genesis_block()
        
    def _create_genesis_block(self):
        """Create the first block in the chain"""
        genesis_tx = {"tx_id": "0", "sender": "GENESIS",
                      "recipient": self.node_id, "amount": "0",
                      "type": "genesis", "sender_pub_key": "GENESIS",
                      "nonce": 0}
        genesis_block = Block(0, "0", [genesis_tx], difficulty=1)
        self.chain.append(genesis_block)
        # Genesis adds nothing to balances; just initialise caches.
        logger.info(f"Genesis Block created. Hash: {genesis_block.hash}")

    # ---------- account-state index (sec1.5 + sec3.3) ----------
    def _apply_block_to_state(self, block: "Block") -> None:
        """Incrementally update cached balances + nonces from a block."""
        for tx in block.transactions or []:
            try:
                amt = Decimal(str(tx.get("amount", 0)))
            except Exception:
                continue
            sender = tx.get("sender")
            recipient = tx.get("recipient")
            tx_type = tx.get("type")
            try:
                fee = Decimal(str(tx.get("fee", "0")))
            except Exception:
                fee = Decimal("0")

            # Credit recipient.
            if recipient:
                self._balances[recipient] = (
                    self._balances.get(recipient, Decimal("0")) + amt
                )
            # Debit sender (skip system senders that mint).
            if sender and sender not in ("COINBASE", "NETWORK_FEES",
                                          "GENESIS", "SYSTEM"):
                self._balances[sender] = (
                    self._balances.get(sender, Decimal("0"))
                    - amt - fee
                )
            # Update nonce table for non-system transfers.
            if tx_type == "transfer" and sender:
                n = int(tx.get("nonce", 0))
                prev = self._nonces.get(sender, -1)
                if n > prev:
                    self._nonces[sender] = n

            # W23: chain-derived contract state. Apply incrementally so
            # the cache always tracks the live chain head without a
            # full replay.
            if tx_type in ("deploy_contract", "execute_contract"):
                # Lazy-init the cache; if first access already replayed
                # the chain it'll be present, otherwise this primes it.
                if not hasattr(self, "_contract_state_cache"):
                    self._contract_state_cache = {}
                    self._contract_state_dirty = False
                self._apply_contract_tx_to_cache(tx)

            # W32: maintain a chain-derived view of slashed validators
            # so other modules can ask "is this address still under an
            # unbonding lock?" without re-walking the chain.
            if tx_type == "slash":
                payload = tx.get("payload") or {}
                offender_addr = payload.get("offender_addr")
                if offender_addr:
                    self.add_to_blacklist(offender_addr)
                    sc = getattr(self, "_staking_contract", None)
                    if sc is not None and hasattr(sc, "_unbonding_locks"):
                        try:
                            sc._unbonding_locks[offender_addr] = int(
                                payload.get("unbonding_lock_until_block", 0)
                            )
                        except Exception:
                            pass

    def _recompute_state_from_chain(self) -> None:
        """Walk the entire chain and rebuild balances + nonces.
        Called on reorg or when caches are suspected stale."""
        self._balances.clear()
        self._nonces.clear()
        # W23: chain-derived contract state must also be rebuilt on reorg.
        self._invalidate_contract_cache()
        for block in self.chain:
            self._apply_block_to_state(block)

    def get_account_nonce(self, address: str) -> int:
        """Last confirmed nonce for `address`. -1 if none."""
        return self._nonces.get(address, -1)

    # ---- SQLite durability (gap #1) ----
    def attach_blockstore(self, blockstore) -> None:
        """Attach a BlockStore so every accepted block is durably
        persisted (append-only, WAL-journaled). Existing chain in
        memory is migrated to the store on attach. Gap #1 fix.
        """
        self._blockstore = blockstore
        try:
            for block in self.chain:
                try:
                    blockstore.append_block(block.to_dict())
                except Exception:
                    # Already persisted (UNIQUE constraint) — skip.
                    pass
        except Exception as e:
            logger.warning("attach_blockstore migration failed: %s", e)

    def _persist_block(self, block) -> None:
        bs = getattr(self, "_blockstore", None)
        if bs is None:
            return
        try:
            bs.append_block(block.to_dict())
        except Exception as e:
            logger.error("blockstore append failed (height=%d): %s",
                         getattr(block, "index", -1), e)

    def save_chain(self, filename: str = "ledger.json"):
        """Save blockchain to disk with automatic backup"""
        import shutil
        import os
        
        # [BACKUP] Create timestamped backup
        if os.path.exists(filename):
            timestamp = int(time.time())
            backup_name = f"{filename}.{timestamp}.bak"
            try:
                shutil.copy2(filename, backup_name)
                logger.info(f"Ledger Backup created: {backup_name}")
            except Exception as e:
                logger.error(f"Ledger Backup failed: {e}")

        try:
            chain_data = [block.to_dict() for block in self.chain]
            with open(filename, "w") as f:
                json.dump(chain_data, f, indent=4)
            logger.info(f"Ledger saved to {filename} (Height: {len(self.chain)})")
        except Exception as e:
            logger.error(f"Failed to save ledger: {e}")

    def load_chain(self, filename: str = "ledger.json"):
        """Load blockchain from disk"""
        try:
            with open(filename, "r") as f:
                chain_data = json.load(f)
            
            loaded_chain = []
            for b_data in chain_data:
                # Reconstruct Block objects
                block = Block(
                    index=b_data['index'],
                    previous_hash=b_data['previous_hash'],
                    transactions=b_data['transactions'],
                    timestamp=b_data['timestamp'],
                    difficulty=b_data.get('difficulty', 1),
                    nonce=b_data.get('nonce', 0)
                )
                loaded_chain.append(block)
                
            if loaded_chain:
                self.chain = loaded_chain
                # CP-5 fix: balances + nonces are derived state (sec1.5).
                # `load_chain` was previously replacing self.chain without
                # rebuilding the caches -- get_balance / get_account_nonce
                # would return stale (genesis-only) values until the next
                # block was mined or applied. Now we recompute end-to-end.
                self._recompute_state_from_chain()
                logger.info(f"Ledger loaded from {filename} (Height: {len(self.chain)})")
                return True
        except FileNotFoundError:
            logger.info("No existing ledger found. Starting fresh.")
            return False
        except Exception as e:
            logger.error(f"Failed to load ledger: {e}")
            return False

    def get_latest_block(self) -> Block:
        return self.chain[-1]
        
    def receive_remote_block(self, block_data: Dict[str, Any]) -> bool:
        """
        Receive a block from a peer. Implements **longest-work fork
        resolution** so a multi-node network can survive partitions
        and reconverge.

        Previous version rejected ANY block at an existing height,
        so two nodes that mined simultaneously would permanently
        refuse each other. That isn't a chain — that's two parallel
        databases.

        Algorithm:
            1. Validate the block (PoW target met, hash integrity).
            2. If sequential and links cleanly: append.
            3. If at an existing height: keep the chain with greater
               cumulative work; the other becomes an orphan.
            4. If we're behind by more than one block: request a sync.
        """
        try:
            new_block = Block(
                index=block_data['index'],
                previous_hash=block_data['previous_hash'],
                transactions=block_data['transactions'],
                timestamp=block_data['timestamp'],
                difficulty=block_data.get('difficulty', 1),
                nonce=block_data.get('nonce', 0),
            )

            # 1. PoW validity: hash must meet difficulty.
            target_prefix = "0" * int(new_block.difficulty)
            if not new_block.hash.startswith(target_prefix):
                logger.warning("[REJECT] Block #%d fails PoW target", new_block.index)
                return False
            # And hash must be reproducible from contents.
            if new_block.hash != new_block.calculate_hash():
                logger.warning("[REJECT] Block #%d hash inconsistent", new_block.index)
                return False
            new_block.work = 16 ** int(new_block.difficulty)

            # 1.5 PER-TX VALIDATION (closes sec1.4).
            # Previously the loop only checked PoW + hash; the txs
            # inside the block were trusted on faith. A peer could
            # broadcast a block whose internal txs were unsigned or
            # whose mint had sender_pub_key='hax' and we'd accept it
            # provided the PoW was real. Validate each tx now.
            if not self._validate_block_transactions(new_block):
                return False

            latest = self.get_latest_block()

            # 2. Cleanly extends our chain.
            if new_block.index == latest.index + 1 and new_block.previous_hash == latest.hash:
                self.chain.append(new_block)
                self._apply_block_to_state(new_block)
                self._persist_block(new_block)
                logger.info("[CONSENSUS] Extended chain to #%d (%s)",
                            new_block.index, new_block.hash[:12])
                return True

            # 3. Same-height fork — pick the chain with more cumulative work.
            if new_block.index <= latest.index:
                if new_block.index < len(self.chain):
                    existing = self.chain[new_block.index]
                    if existing.hash == new_block.hash:
                        return False                # duplicate
                # Compute cumulative work of our chain through that height.
                our_work = sum(getattr(b, "work", 16 ** int(b.difficulty))
                               for b in self.chain[: new_block.index + 1])
                # We don't have the rival's full chain here, only the head;
                # use a heuristic: greater difficulty at same height wins.
                # For full safety, the caller can request the rival chain
                # via the gossip protocol and re-apply this method.
                rival_advantage = (
                    int(new_block.difficulty) > int(self.chain[new_block.index].difficulty)
                    if new_block.index < len(self.chain) else True
                )
                if not rival_advantage:
                    logger.info("[FORK] Keeping our higher-work chain at #%d", new_block.index)
                    return False
                # Reorg: drop our tail at and beyond new_block.index, replace.
                logger.warning("[REORG] Replacing chain from #%d onward (rival has more work)",
                               new_block.index)
                self.chain = self.chain[: new_block.index]
                self.chain.append(new_block)
                # State caches are now stale from the truncation —
                # rebuild from scratch. Reorgs are rare so the O(N*M)
                # cost is acceptable.
                self._recompute_state_from_chain()
                # Reorg the durable store too: drop everything past
                # the new block's predecessor, then re-persist the new
                # head. Foreign-key cascade on `transactions` keeps
                # the tx index consistent.
                bs = getattr(self, "_blockstore", None)
                if bs is not None:
                    try:
                        bs.truncate_to(new_block.index - 1)
                    except Exception as e:
                        logger.warning("blockstore reorg-truncate failed: %s", e)
                self._persist_block(new_block)
                return True

            # 4. We're behind. Caller (mesh controller) should trigger a
            # bulk sync from a peer; we can't safely append a future block
            # without the bridge.
            logger.warning("[SYNC] Received block #%d, we're at #%d — request bulk sync",
                           new_block.index, latest.index)
            return False

        except Exception as e:
            logger.error("[BLOCK REJECT] %s", e)
            return False
    
    # Types of transaction that are produced ONLY by consensus / chain
    # internals (mining, fee distribution, slashing). External callers
    # must not be able to push these into the pending pool — that was
    # the §3.5 issue: anyone constructing a `mint` Transaction could
    # add it to pending and have it mined, bypassing the supply cap
    # discipline. Internal callers pass `_internal=True` to bypass.
    _SYSTEM_TX_TYPES = {"mint", "coinbase", "fee_reward", "genesis", "slash"}

    def add_transaction(self, tx: Transaction, _internal: bool = False) -> bool:
        """
        Add transaction to pending pool.

        Validation:
          1. `_SYSTEM_TX_TYPES` (mint / coinbase / fee_reward / genesis /
             slash) are rejected unless `_internal=True`. Only the
             chain's own mining / consensus path may insert them.
          2. `transfer` txs require a valid ECDSA signature over the
             tx_id by the claimed `sender_pub_key`.
          3. Min fee policy (W34): transfers below `MIN_TX_FEE` are
             rejected (free-tx spam protection).
        """
        if tx.type in self._SYSTEM_TX_TYPES and not _internal:
            logger.error(
                "[REJECTED TX] %s: type=%s is system-only "
                "(must come via internal consensus path).",
                tx.tx_id, tx.type,
            )
            return False

        if tx.type == 'transfer':
            # Defence-in-depth (sec3.3 v2): recompute the tx_id from the
            # current field values and refuse if it disagrees with the
            # tx's stored id. Without this, an attacker who mutates the
            # recipient AFTER signing can pass `Wallet.verify` because
            # the stored tx_id is stale -- the chain layer would still
            # reject the resulting block, but the mempool pollution +
            # local state divergence is unnecessary risk.
            from .tokenomics import Transaction as _Tx
            expected_id = _Tx.calculate_tx_hash(
                tx.sender, tx.recipient, tx.amount,
                getattr(tx, "fee", Decimal("0")),
                tx.timestamp, tx.type, tx.sender_pub_key,
                getattr(tx, "nonce", 0),
            )
            if expected_id != tx.tx_id:
                logger.error(
                    "[REJECTED TX] %s: tampered tx_id (recompute %s)",
                    tx.tx_id, expected_id,
                )
                return False
            if not Wallet.verify(tx.sender_pub_key, tx.tx_id, tx.signature):
                logger.error("[REJECTED TX] %s: Invalid Signature", tx.tx_id)
                return False
            try:
                fee = Decimal(str(getattr(tx, "fee", "0")))
            except Exception:
                fee = Decimal("0")
            if fee < self.MIN_TX_FEE:
                logger.warning(
                    "[REJECTED TX] %s: fee %s below MIN_TX_FEE %s "
                    "(free-tx spam protection).",
                    tx.tx_id, fee, self.MIN_TX_FEE,
                )
                return False

            # sec3.3: per-sender monotonic nonce → replay protection.
            # Account for nonces of other still-pending transfers from
            # the same sender so we can queue (n, n+1, n+2) without
            # the second one being mistakenly rejected.
            highest_pending = max(
                (int(getattr(p, "nonce", 0)) for p in self.pending_transactions
                 if getattr(p, "sender", None) == tx.sender
                 and getattr(p, "type", None) == "transfer"),
                default=self._nonces.get(tx.sender, -1),
            )
            confirmed = self._nonces.get(tx.sender, -1)
            min_required = max(highest_pending, confirmed) + 1
            if int(tx.nonce) < min_required:
                logger.warning(
                    "[REJECTED TX] %s: nonce %d < required %d for %s "
                    "(replay/reorder protection).",
                    tx.tx_id, int(tx.nonce), min_required, tx.sender[:12],
                )
                return False

        self.pending_transactions.append(tx)
        return True

    # Min-fee policy: rejects free transfers (mempool spam protection).
    # Set very small for v3.0-alpha (~= 0.1c at compute peg). Governance
    # can raise/lower this once §6 audit (W27) lands a real DAO.
    MIN_TX_FEE = Decimal("0.001")
        
    def _validate_block_transactions(self, block: "Block") -> bool:
        """
        Validate every transaction inside a remote block. Refuses the
        block as a whole if any single tx is malformed or unsigned.

        Rules:
          * `transfer` must have a valid ECDSA signature over the tx_id
            by the claimed `sender_pub_key`, AND fee >= MIN_TX_FEE.
          * `mint` / `coinbase` must declare sender='COINBASE' and
            sender_pub_key='SYSTEM', AND its amount must not exceed
            the per-block reward ceiling (block_reward at this height
            * MAX_DIFFICULTY_FACTOR).
          * `fee_reward` must declare sender='NETWORK_FEES' and
            sender_pub_key='SYSTEM', AND amount >= 0.
          * `genesis` is only legal at block index 0.
          * `slash` requires consensus authorisation (BFT signed
            evidence). Until that path is wired (W32), slash txs in
            remote blocks are rejected entirely.
          * Tx hash must reproduce from its declared fields (tamper).

        See TODO sec1.4 for the original missing-validation finding.
        """
        # Compute block reward for this height to bound mint amounts.
        # Defer-import: TokenMinter computes get_block_reward.
        from .tokenomics import TokenMinter, Wallet

        per_block_max_mint = (
            TokenMinter.GENESIS_REWARD
            / (Decimal("2.0") ** (block.index // TokenMinter.HALVING_INTERVAL))
            * TokenMinter.MAX_DIFFICULTY_FACTOR
        )

        # Per-sender max-nonce-seen WITHIN this block (sec3.3).
        # Used to reject within-block replays / reorderings.
        seen_nonce_in_block: Dict[str, int] = {}

        for tx in block.transactions or []:
            tx_type = tx.get("type")
            tx_id = tx.get("tx_id")

            # 1. Tx hash must reproduce from declared fields.
            try:
                from .tokenomics import Transaction
                expected_id = Transaction.calculate_tx_hash(
                    tx.get("sender"), tx.get("recipient"),
                    tx.get("amount"), tx.get("fee", "0.0"),
                    tx.get("timestamp"), tx_type, tx.get("sender_pub_key"),
                    int(tx.get("nonce", 0)),
                )
                if expected_id != tx_id:
                    logger.warning("[REJECT] block #%d tx %s tampered "
                                   "(declared id %s != recomputed %s)",
                                   block.index, (tx_id or '?')[:8],
                                   (tx_id or '?')[:8], expected_id[:8])
                    return False
            except Exception as e:
                logger.warning("[REJECT] block #%d tx hash recompute failed: %s",
                               block.index, e)
                return False

            # 2. Type-specific rules.
            if tx_type == "transfer":
                pk = tx.get("sender_pub_key")
                sig = tx.get("signature")
                if not pk or not sig:
                    logger.warning("[REJECT] block #%d transfer %s "
                                   "missing pubkey/signature",
                                   block.index, (tx_id or '?')[:8])
                    return False
                if not Wallet.verify(pk, tx_id, sig):
                    logger.warning("[REJECT] block #%d transfer %s "
                                   "bad signature",
                                   block.index, (tx_id or '?')[:8])
                    return False
                try:
                    fee = Decimal(str(tx.get("fee", "0")))
                except Exception:
                    fee = Decimal("0")
                if fee < self.MIN_TX_FEE:
                    logger.warning("[REJECT] block #%d transfer %s "
                                   "fee %s below MIN_TX_FEE",
                                   block.index, (tx_id or '?')[:8], fee)
                    return False

                # sec3.3 nonce monotonicity:
                #   * vs. confirmed chain state (per-sender max nonce)
                #   * vs. any prior tx from the same sender in THIS block
                sender = tx.get("sender")
                tx_nonce = int(tx.get("nonce", 0))
                last_confirmed = self._nonces.get(sender, -1)
                last_in_block = seen_nonce_in_block.get(sender, -1)
                min_required = max(last_confirmed, last_in_block) + 1
                if tx_nonce < min_required:
                    logger.warning(
                        "[REJECT] block #%d transfer %s nonce %d "
                        "< required %d for sender %s",
                        block.index, (tx_id or '?')[:8],
                        tx_nonce, min_required, sender[:12],
                    )
                    return False
                seen_nonce_in_block[sender] = tx_nonce

            elif tx_type in ("mint", "coinbase"):
                if tx.get("sender") != "COINBASE" \
                        or tx.get("sender_pub_key") != "SYSTEM":
                    logger.warning("[REJECT] block #%d %s %s has "
                                   "non-system sender markers",
                                   block.index, tx_type, (tx_id or '?')[:8])
                    return False
                try:
                    amt = Decimal(str(tx.get("amount", 0)))
                except Exception:
                    amt = Decimal("-1")
                if amt <= 0 or amt > per_block_max_mint:
                    logger.warning("[REJECT] block #%d mint %s amount "
                                   "%s exceeds per-block ceiling %s",
                                   block.index, (tx_id or '?')[:8],
                                   amt, per_block_max_mint)
                    return False

            elif tx_type == "fee_reward":
                if tx.get("sender") != "NETWORK_FEES" \
                        or tx.get("sender_pub_key") != "SYSTEM":
                    logger.warning("[REJECT] block #%d fee_reward %s "
                                   "has non-system sender markers",
                                   block.index, (tx_id or '?')[:8])
                    return False
                try:
                    amt = Decimal(str(tx.get("amount", 0)))
                except Exception:
                    amt = Decimal("-1")
                if amt < 0:
                    logger.warning("[REJECT] block #%d fee_reward %s "
                                   "negative amount", block.index,
                                   (tx_id or '?')[:8])
                    return False

            elif tx_type == "genesis":
                if block.index != 0:
                    logger.warning("[REJECT] block #%d contains genesis "
                                   "tx (only legal at index 0)", block.index)
                    return False

            elif tx_type == "slash":
                # W32: validate the embedded slash-evidence with a
                # ≥2/3-stake-weighted attestation quorum. Verified on
                # the receive side so a single malicious accuser cannot
                # forge a slash; verified on the apply side so the
                # accuser cannot lie about which validator signed what.
                if tx.get("sender") != "BFT_SLASH_PROTOCOL" \
                        or tx.get("sender_pub_key") != "SYSTEM":
                    logger.warning("[REJECT] block #%d slash tx %s has "
                                   "non-protocol sender markers",
                                   block.index, (tx_id or '?')[:8])
                    return False
                payload = tx.get("payload") or {}
                ok, reason = self._validate_slash_payload(payload)
                if not ok:
                    logger.warning("[REJECT] block #%d slash tx %s evidence "
                                   "rejected: %s", block.index,
                                   (tx_id or '?')[:8], reason)
                    return False

            elif tx_type == "deploy_contract":
                # W23: chain-derived smart-contract deploy. Validates
                # owner signature over the canonical body of the deploy.
                ok, reason = self._validate_deploy_contract_tx(tx)
                if not ok:
                    logger.warning("[REJECT] block #%d deploy_contract %s: %s",
                                   block.index, (tx_id or '?')[:8], reason)
                    return False

            elif tx_type == "execute_contract":
                # W23: chain-derived smart-contract execute. Validates
                # caller signature; the actual VM execution + storage
                # diff is applied during _ingest_block.
                ok, reason = self._validate_execute_contract_tx(tx)
                if not ok:
                    logger.warning("[REJECT] block #%d execute_contract %s: %s",
                                   block.index, (tx_id or '?')[:8], reason)
                    return False

            else:
                logger.warning("[REJECT] block #%d unknown tx type %r",
                               block.index, tx_type)
                return False

        return True

    # Network-wide difficulty target. Set low for the alpha mesh (block
    # every ~2-5s on a consumer CPU), bumped at consensus checkpoints
    # by the validator set as hashrate grows. Number of leading hex
    # zeros required in the block hash.
    DIFFICULTY_TARGET = 4         # hex zeros (~ 2^16 work)
    PROOF_MAX_NONCE = 2**32       # protects against pathological loops

    # sec3.6: difficulty retargeting (DAA).
    # Every RETARGET_INTERVAL blocks, scale difficulty by the ratio of
    # observed inter-block time to TARGET_BLOCK_TIME_S. The ratio is
    # clamped to [1/MAX_RETARGET_FACTOR, MAX_RETARGET_FACTOR] to bound
    # how aggressively any single retarget can move (Bitcoin uses 4x;
    # we use 2x because a smaller mesh has higher hashrate variance).
    # Difficulty is the integer "hex zeros" prefix length, so we
    # adjust it in steps of 1 (round-half-up). Floor at 1, ceiling at
    # 16 (above 16 the search space collapses below 2^256 / 2^64 and
    # PROOF_MAX_NONCE = 2^32 stops being enough).
    TARGET_BLOCK_TIME_S = 30        # ~30s per block
    RETARGET_INTERVAL = 16          # retarget every 16 blocks
    MAX_RETARGET_FACTOR = 2.0
    MIN_DIFFICULTY = 1
    MAX_DIFFICULTY = 16

    def get_current_difficulty(self) -> int:
        """
        Compute the difficulty for the next block, deriving it
        deterministically from on-chain inter-block timestamps over
        the last RETARGET_INTERVAL blocks. Pure function of chain
        state — every node arrives at the same answer.

        Behavior:
          * Genesis epoch (height < RETARGET_INTERVAL): use
            DIFFICULTY_TARGET as the bootstrap value.
          * Otherwise: compare observed window time to expected
            (RETARGET_INTERVAL * TARGET_BLOCK_TIME_S). If observed
            << expected (blocks too fast → hashrate up), bump
            difficulty +1; if >> expected (too slow → hashrate down),
            drop -1. The clamp at MAX_RETARGET_FACTOR is the
            single-retarget step bound, NOT a per-block change cap.
          * Difficulty is bounded to [MIN_DIFFICULTY, MAX_DIFFICULTY].
        """
        h = self.get_height()
        if h < self.RETARGET_INTERVAL:
            return self.DIFFICULTY_TARGET

        # Use last block's difficulty as the base.
        last = self.chain[-1]
        base = int(getattr(last, "difficulty", self.DIFFICULTY_TARGET))

        window_start = self.chain[h - self.RETARGET_INTERVAL]
        window_end = last
        observed = float(window_end.timestamp - window_start.timestamp)
        if observed <= 0:
            return base
        expected = float(self.RETARGET_INTERVAL * self.TARGET_BLOCK_TIME_S)
        ratio = observed / expected

        # Clamp the ratio to bound a single retarget step.
        if ratio > self.MAX_RETARGET_FACTOR:
            ratio = self.MAX_RETARGET_FACTOR
        elif ratio < (1.0 / self.MAX_RETARGET_FACTOR):
            ratio = 1.0 / self.MAX_RETARGET_FACTOR

        # ratio < 1 → blocks were too fast → harder difficulty.
        # ratio > 1 → blocks were too slow → easier difficulty.
        if ratio < 0.75:
            new_diff = base + 1
        elif ratio > 1.5:
            new_diff = base - 1
        else:
            new_diff = base

        return max(self.MIN_DIFFICULTY, min(self.MAX_DIFFICULTY, new_diff))

    def mine_block(self, miner_address: str,
                   difficulty: Optional[int] = None,
                   stop_event=None) -> Optional["Block"]:
        """
        Real Proof-of-Work block production.

        Earlier this method had the PoW loop commented out, so blocks
        were emitted instantly with no scarcity protection. Two-node
        networks immediately diverged because both nodes minted at
        zero cost. This version:

          * Iterates `nonce` until block hash starts with N hex zeros.
          * Aborts if a `stop_event` is set (lets a worker pre-empt
            mining when a higher-work block arrives from a peer).
          * Bounds the search to PROOF_MAX_NONCE; falls back to None
            if the difficulty is impossible (caller can lower).

        For production, swap to GPU-accelerated PoW or move chain
        consensus to a BFT validator set on Solana / Base.
        """
        if not self.pending_transactions:
            return None

        # sec3.6: when caller doesn't pin difficulty, derive it from
        # chain state via the retarget DAA.
        diff = int(difficulty if difficulty is not None
                   else self.get_current_difficulty())

        # Aggregate fees as before.
        total_fees = Decimal("0.0")
        for tx in self.pending_transactions:
            try:
                total_fees += Decimal(str(tx.fee))
            except Exception:
                pass
        tx_data = [tx.to_dict() for tx in self.pending_transactions]
        if total_fees > 0:
            fee_tx_obj = Transaction(
                sender="NETWORK_FEES", recipient=miner_address,
                amount=total_fees, type="fee_reward",
                sender_pub_key="SYSTEM",
            )
            tx_data.append(fee_tx_obj.to_dict())
            logger.info("[MINING] Collected %s PLG in fees", total_fees)

        new_block = Block(
            index=len(self.chain),
            previous_hash=self.get_latest_block().hash,
            transactions=tx_data,
            difficulty=diff,
        )
        target_prefix = "0" * diff
        nonce = 0
        t0 = time.time()
        while nonce < self.PROOF_MAX_NONCE:
            if stop_event is not None and stop_event.is_set():
                logger.info("[MINING] Pre-empted at nonce=%d (better block arrived).", nonce)
                return None
            new_block.nonce = nonce
            new_block.hash = new_block.calculate_hash()
            if new_block.hash.startswith(target_prefix):
                break
            nonce += 1
        else:
            logger.warning("[MINING] Difficulty %d exceeded nonce budget; giving up.", diff)
            return None

        # Cumulative work — used by `receive_remote_block` for fork resolution.
        # Each leading-zero hex digit is 16 expected hashes.
        block_work = 16 ** diff
        new_block.work = block_work

        self.chain.append(new_block)
        self.pending_transactions = []
        self._apply_block_to_state(new_block)
        self._persist_block(new_block)
        elapsed = time.time() - t0
        logger.info(
            "Block #%d mined: hash=%s diff=%d nonce=%d work=%d in %.2fs",
            new_block.index, new_block.hash[:16], diff, nonce, block_work, elapsed,
        )
        return new_block

    def get_balance(self, address: str) -> float:
        """
        Balance through the current tip. Fast path: serve from the
        `_balances` cache built incrementally on every block apply
        (sec1.5 — closes the O(blocks * txs) hot loop that was the
        inner cost of every payment / fee check / staking op).

        Falls through to the chain walk only if the cache is missing
        an entry — paranoia path, should never trigger after init.
        """
        cached = self._balances.get(address)
        if cached is not None:
            return float(cached)
        return self.get_balance_at(address, self.get_height())

    def get_balance_at(self, address: str, height: int) -> float:
        """
        Calculate balance by traversing the chain up to and including
        the block at `height`. Used by governance to snapshot voting
        power at proposal-creation time (W27): voters can't pump their
        balance after voting and re-vote, because the weight is fixed
        to a specific past height.

        For current-tip queries, prefer `get_balance` (cache-backed,
        O(1)).  This walk is O(blocks * txs) and is reserved for
        historical heights and post-reorg cache rebuilds.
        """
        balance = 0.0
        upper = min(height, len(self.chain) - 1)
        for i in range(upper + 1):
            block = self.chain[i]
            for tx in block.transactions:
                if tx.get('recipient') == address:
                    try:
                        balance += float(tx.get('amount', 0))
                    except Exception:
                        pass
                if tx.get('sender') == address:
                    try:
                        balance -= float(tx.get('amount', 0))
                    except Exception:
                        pass
                    # Fee is debited from sender too (sec3.4 enforced
                    # MIN_TX_FEE; without this deduction the walk
                    # overstates sender balance whenever sender !=
                    # miner of the same block, allowing a fee-funded
                    # double-spend gap between cache and walk).
                    # System senders (COINBASE / NETWORK_FEES /
                    # GENESIS / SYSTEM) carry a synthetic fee of 0.
                    if tx.get('sender') not in ("COINBASE",
                                                  "NETWORK_FEES",
                                                  "GENESIS", "SYSTEM"):
                        try:
                            balance -= float(tx.get('fee', 0))
                        except Exception:
                            pass
        return balance

    # ---------- staking / slashing glue (W21/W32 minimum) ----------
    # These methods previously didn't exist; arbiter.slash_node called
    # them and AttributeError'd. We now provide working implementations
    # that delegate to an attached StakingContract (set via
    # set_staking_contract) and persist a simple blacklist set on the
    # ledger.  The FULL slash protocol — slash-evidence txs signed by
    # 2/3 of the validator set, on-chain stake destruction, unbonding
    # period — is W32; this commit only closes the crash.

    def set_staking_contract(self, staking_contract) -> None:
        self._staking_contract = staking_contract

    def get_stake(self, address: str) -> Optional[Dict[str, Any]]:
        sc = getattr(self, "_staking_contract", None)
        if sc is None:
            return None
        try:
            return sc.get_staking_info(address)
        except Exception as e:
            logger.debug("get_stake: staking contract raised %s", e)
            return None

    @property
    def blacklist(self) -> set:
        if not hasattr(self, "_blacklist"):
            self._blacklist: set = set()
        return self._blacklist

    def add_to_blacklist(self, address: str) -> None:
        self.blacklist.add(address)
        logger.warning("[LEDGER] blacklisted address %s", address[:12])

    def is_blacklisted(self, address: str) -> bool:
        return address in self.blacklist

    def total_supply_at(self, height: int) -> float:
        """
        Sum of all mint/coinbase tx amounts in chain[:height+1].
        Used by governance quorum.
        """
        total = 0.0
        upper = min(height, len(self.chain) - 1)
        for i in range(upper + 1):
            block = self.chain[i]
            for tx in block.transactions:
                if tx.get('type') in ('mint', 'coinbase'):
                    try:
                        total += float(tx.get('amount', 0))
                    except Exception:
                        pass
        return total

    def verify_chain(self) -> bool:
        """
        Verify the integrity of the blockchain.
        Returns False if tampering is detected.
        """
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]

            # 1. Check if the stored hash matches
            if current_block.hash != current_block.calculate_hash():
                logger.error(f"[TAMPERING DETECTED] at Block #{current_block.index}: Invalid Hash")
                return False

            # 2. Check previous hash link
            if current_block.previous_hash != previous_block.hash:
                logger.error(f"[TAMPERING DETECTED] at Block #{current_block.index}: Broken Chain Link")
                return False
                
            # 3. Merkle Root Check
            if current_block.merkle_root != current_block.calculate_merkle_root():
                 logger.error(f"[TAMPERING DETECTED] at Block #{current_block.index}: Invalid Merkle Root")
                 return False
                 
            # 4. [SECURITY] Transaction Re-hashing & Signature Verification
            for tx in current_block.transactions:
                tx_id = tx.get('tx_id')
                
                # A. Re-calculate ID from content to ensure integrity
                # Need consistent reproduction of Transaction.calculate_hash() logic
                try:
                    re_hash = Transaction.calculate_tx_hash(
                        tx.get('sender'),
                        tx.get('recipient'),
                        # Note: In to_dict(), amount/fee are converted to strings.
                        # In Transaction.__init__, they are Decimals.
                        # Transaction.calculate_tx_hash expects inputs as they are used in f-string.
                        # str(Decimal("10.0")) is "10.0". tx.get('amount') is "10.0". 
                        # This should match.
                        tx.get('amount'),
                        tx.get('fee', '0.0'),
                        tx.get('timestamp'),
                        tx.get('type'),
                        tx.get('sender_pub_key')
                    )
                    
                    if re_hash != tx_id:
                        logger.error(f"[TAMPERING DETECTED] in Block #{current_block.index}: TX {tx_id[:8]} content modified! Expected {re_hash[:8]} got {tx_id[:8]}")
                        return False
                except Exception as e:
                    logger.error(f"Hashing Error in Block #{current_block.index}: {e}")
                    return False

                # B. Verify Signature (if transfer)
                tx_type = tx.get('type')
                if tx_type == 'transfer':
                    pub_key = tx.get('sender_pub_key')
                    sig = tx.get('signature')
                    
                    if not pub_key or not sig:
                        logger.error(f"[INVALID TX] in Block #{current_block.index}: Missing Auth Data")
                        return False
                        
                    if not Wallet.verify(pub_key, tx_id, sig):
                         logger.error(f"[FORGED TX DETECTED] in Block #{current_block.index}: Signature Mismatch")
                         return False

        logger.info("[SUCCESS] Blockchain Verified: Integrity Intact")
        return True

    def get_height(self) -> int:
        return len(self.chain)

    # ---------- W32 slash-evidence validation ----------
    def _validate_slash_payload(self, payload: Dict[str, Any]) -> Tuple[bool, str]:
        """Reconstruct + verify a SlashEvidence dict carried as the
        payload of a slash tx. Used by validate_block_transactions to
        gate inclusion. Returns (ok, reason).

        The full attestation-quorum check requires the validator stake
        snapshot at the offence height; we pull it from the attached
        staking contract. If the staking contract isn't wired, evidence
        without quorum verification is rejected — chain-wide slash
        without quorum proof is exactly the abuse vector W32 closes.
        """
        try:
            from .slash_evidence import (
                SlashEvidence, BlockHeaderProof, Attestation, verify_evidence,
            )
        except Exception as e:
            return False, f"slash_evidence_unavailable:{e}"
        try:
            block_a = BlockHeaderProof(
                header=payload["block_a"]["header"],
                validator_pubkey_pem=payload["offender_pubkey_pem"],
                signature_b64=payload["block_a"]["signature_b64"],
            )
            block_b = BlockHeaderProof(
                header=payload["block_b"]["header"],
                validator_pubkey_pem=payload["offender_pubkey_pem"],
                signature_b64=payload["block_b"]["signature_b64"],
            )
            attestations = [
                Attestation(
                    validator_pubkey_pem=a["validator_pubkey_pem"],
                    signature_b64=a["signature_b64"],
                )
                for a in payload.get("attestations", [])
            ]
            evidence = SlashEvidence(
                offender_pubkey_pem=payload["offender_pubkey_pem"],
                block_a=block_a, block_b=block_b,
                height=int(payload["height"]),
                attestations=attestations,
            )
        except Exception as e:
            return False, f"malformed_evidence:{e}"

        sc = getattr(self, "_staking_contract", None)
        validator_stakes: Dict[str, Decimal] = {}
        if sc is not None and hasattr(sc, "validator_stake_snapshot"):
            try:
                validator_stakes = sc.validator_stake_snapshot()
            except Exception as e:
                return False, f"stake_snapshot_failed:{e}"
        if not validator_stakes:
            return False, "no_validator_stakes_for_quorum"

        ok, reason = verify_evidence(
            evidence,
            validator_stakes=validator_stakes,
            require_attestation_quorum=True,
        )
        if not ok:
            return False, f"evidence_invalid:{reason}"
        return True, ""

    # ---------- W23 chain-derived smart-contract storage ----------
    @property
    def contract_state(self) -> Dict[str, Dict[str, Any]]:
        """Public read-only view of all deployed contracts as derived
        from the chain. Replays from genesis on first access; cached
        afterwards. Reorgs invalidate the cache via `_invalidate_contract_cache`.
        """
        if not getattr(self, "_contract_state_dirty", True):
            return self._contract_state_cache       # type: ignore[attr-defined]
        self._contract_state_cache: Dict[str, Dict[str, Any]] = {}
        for block in self.chain:
            for tx in block.transactions or []:
                self._apply_contract_tx_to_cache(tx)
        self._contract_state_dirty = False
        return self._contract_state_cache

    def _invalidate_contract_cache(self) -> None:
        self._contract_state_dirty = True

    def _apply_contract_tx_to_cache(self, tx: Dict[str, Any]) -> None:
        cache = self._contract_state_cache
        ttype = tx.get("type")
        payload = tx.get("payload") or {}
        if ttype == "deploy_contract":
            addr = payload.get("address")
            if not addr:
                return
            cache[addr] = {
                "owner": payload.get("owner"),
                "code": payload.get("code"),
                "code_hash": payload.get("code_hash"),
                "storage": dict(payload.get("initial_storage") or {}),
                "deploy_tx_id": tx.get("tx_id"),
            }
        elif ttype == "execute_contract":
            addr = payload.get("address")
            if addr in cache and isinstance(payload.get("new_storage"), dict):
                cache[addr]["storage"] = dict(payload["new_storage"])

    def _validate_deploy_contract_tx(self, tx: Dict[str, Any]) -> Tuple[bool, str]:
        from .tokenomics import Wallet
        payload = tx.get("payload") or {}
        owner = payload.get("owner")
        code = payload.get("code")
        code_hash = payload.get("code_hash")
        addr = payload.get("address")
        sig = tx.get("signature")
        pk = tx.get("sender_pub_key")
        if not all([owner, code, code_hash, addr, sig, pk]):
            return False, "missing_fields"
        # Owner = sender, recompute code_hash, recompute address.
        actual_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
        if actual_hash != code_hash:
            return False, "code_hash_mismatch"
        nonce = int(tx.get("nonce", 0))
        expected_addr = "C" + hashlib.sha256(
            f"{owner}|{code_hash}|{nonce}".encode("utf-8")
        ).hexdigest()[:39]
        if expected_addr != addr:
            return False, "address_mismatch"
        if not Wallet.verify(pk, tx.get("tx_id"), sig):
            return False, "owner_signature_invalid"
        return True, ""

    def _validate_execute_contract_tx(self, tx: Dict[str, Any]) -> Tuple[bool, str]:
        from .tokenomics import Wallet
        payload = tx.get("payload") or {}
        addr = payload.get("address")
        function_name = payload.get("function_name")
        args = payload.get("args")
        sig = tx.get("signature")
        pk = tx.get("sender_pub_key")
        if not all([addr, function_name]):
            return False, "missing_fields"
        if not isinstance(args, list):
            return False, "args_not_list"
        if not sig or not pk:
            return False, "missing_caller_auth"
        if not Wallet.verify(pk, tx.get("tx_id"), sig):
            return False, "caller_signature_invalid"
        return True, ""

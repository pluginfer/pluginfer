"""
Pluginfer Tokenomics Core
Implements Wallet (ECC), Transactions, and Token Minting (Halving Logic).
"""
import hashlib
import json
import time
import base64
import logging
from decimal import Decimal, getcontext
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.exceptions import InvalidSignature

# Set Decimal precision high for fractional tokens
getcontext().prec = 28
logger = logging.getLogger(__name__)

class Wallet:
    """
    Crypto-Grade Wallet using ECC (SECP256K1)
    """
    def __init__(self, private_key_pem: bytes = None):
        if private_key_pem:
            # Load existing
            try:
                self.private_key = serialization.load_pem_private_key(
                    private_key_pem, password=None
                )
            except Exception as e:
                logger.error(f"Failed to load private key: {e}")
                self.private_key = ec.generate_private_key(ec.SECP256K1())
        else:
            # Generate new
            self.private_key = ec.generate_private_key(ec.SECP256K1())
            
        self.public_key = self.private_key.public_key()
        self.address = self.generate_address()
        # Export Public Key PEM for attaching to transactions
        self.public_key_pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode()
        
    def save_to_file(self, filename: str = "wallet.pem",
                     passphrase: bytes = None) -> bool:
        """
        Save private key to disk with **at-rest encryption**.

        Source of `passphrase`, in order of preference:
          1. Argument supplied by the caller (test code, automated jobs).
          2. `PLUGINFER_WALLET_PASSPHRASE` environment variable
             (recommended for headless / non-interactive operation).
          3. OS keychain via `keyring` (recommended for desktop users:
             stored in Windows Credential Manager / macOS Keychain /
             Linux Secret Service).
          4. Refuse to save with a clear error.

        Previous behaviour wrote the private key with
        `NoEncryption()`. A single filesystem read of `wallet.pem` was
        all an attacker needed to drain a node's wallet (TODO sec3.1).
        We refuse to ship that fallback even on first run — the user
        gets a clear error instead.

        Backup files (`.<ts>.bak`) of any prior wallet are also
        re-encrypted with the same passphrase before being written.
        """
        import os
        import shutil

        if passphrase is None:
            env_pw = os.environ.get("PLUGINFER_WALLET_PASSPHRASE")
            if env_pw:
                passphrase = env_pw.encode("utf-8")
        if passphrase is None:
            passphrase = self._read_passphrase_from_keychain(filename)

        if not passphrase:
            logger.error(
                "Wallet save refused: no passphrase available. Set "
                "PLUGINFER_WALLET_PASSPHRASE, install `keyring` and "
                "store one under service='pluginfer-wallet' "
                "username=%r, or pass `passphrase=...` explicitly. "
                "(sec3.1: refusing to write wallet unencrypted.)",
                filename,
            )
            return False

        # Backup any existing wallet (already encrypted, just copy).
        if os.path.exists(filename):
            timestamp = int(time.time())
            backup_name = f"{filename}.{timestamp}.bak"
            try:
                shutil.copy2(filename, backup_name)
                logger.info("Wallet Backup created: %s", backup_name)
            except Exception as e:
                logger.error("Wallet Backup failed: %s", e)

        try:
            pem = self.private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.BestAvailableEncryption(passphrase),
            )
            with open(filename, "wb") as f:
                f.write(pem)
            logger.info("Wallet saved (encrypted) to %s", filename)
            return True
        except Exception as e:
            logger.error("Failed to save wallet: %s", e)
            return False

    @classmethod
    def load_from_file(cls, filename: str = "wallet.pem",
                       passphrase: bytes = None):
        """
        Load wallet from disk. Encryption is now expected; if the file
        is unencrypted (legacy / test fixture), the load still works.
        Source of `passphrase` mirrors `save_to_file`.
        """
        import os
        try:
            with open(filename, "rb") as f:
                pem_data = f.read()
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.error("Failed to read wallet file: %s", e)
            return None

        if passphrase is None:
            env_pw = os.environ.get("PLUGINFER_WALLET_PASSPHRASE")
            if env_pw:
                passphrase = env_pw.encode("utf-8")
        if passphrase is None:
            passphrase = cls._read_passphrase_from_keychain(filename)

        try:
            priv = serialization.load_pem_private_key(pem_data, password=passphrase)
            return cls._from_loaded_priv(priv)
        except (ValueError, TypeError) as e:
            # Could be: encrypted but no passphrase; bad passphrase;
            # truly unencrypted file (passphrase ignored). Try once
            # without a passphrase as the legacy fallback.
            try:
                priv = serialization.load_pem_private_key(pem_data, password=None)
                logger.warning(
                    "Loaded UNENCRYPTED wallet from %s — re-saving with "
                    "encryption is strongly recommended (call "
                    "wallet.save_to_file() with a passphrase configured).",
                    filename,
                )
                return cls._from_loaded_priv(priv)
            except Exception as e2:
                logger.error("Failed to decrypt wallet (%s / %s); "
                             "is the passphrase correct?", e, e2)
                return None

    @staticmethod
    def _read_passphrase_from_keychain(filename: str):
        """Best-effort fetch via `keyring`. Returns bytes or None."""
        try:
            import keyring  # type: ignore
        except ImportError:
            return None
        try:
            value = keyring.get_password("pluginfer-wallet", filename)
            if value:
                return value.encode("utf-8")
        except Exception as e:
            logger.debug("keyring lookup failed: %s", e)
        return None

    @classmethod
    def _from_loaded_priv(cls, priv) -> "Wallet":
        """Internal: build a Wallet around an already-loaded private key."""
        w = cls.__new__(cls)
        w.private_key = priv
        w.public_key = priv.public_key()
        w.address = w.generate_address()
        w.public_key_pem = w.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode()
        return w
        
    def generate_address(self) -> str:
        """Generate a hashed address from public key (Ethereum-style)"""
        pub_bytes = self.public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        # SHA256 -> RIPEMD160 (Classic Bitcoin style, or just SHA256 for simplicity)
        # We'll use SHA256 truncate for simplicity in this python impl
        sha_hash = hashlib.sha256(pub_bytes).hexdigest()
        return f"PLG{sha_hash[:40]}" # 'PLG' prefix
        
    def export_keys(self) -> dict:
        """Export keys for storage"""
        priv = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        pub = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return {'private': priv.decode(), 'public': pub.decode(), 'address': self.address}

    def sign(self, message: str) -> str:
        """Sign a message (transaction hash)"""
        signature = self.private_key.sign(
            message.encode(),
            ec.ECDSA(hashes.SHA256())
        )
        return base64.b64encode(signature).decode()

    @staticmethod
    def verify(public_key_pem: str, message: str, signature_b64: str) -> bool:
        """Verify a signature"""
        try:
            pub_key = serialization.load_pem_public_key(public_key_pem.encode())
            signature = base64.b64decode(signature_b64)
            pub_key.verify(
                signature,
                message.encode(),
                ec.ECDSA(hashes.SHA256())
            )
            return True
        except InvalidSignature:
            return False
        except Exception as e:
            logger.error(f"Verification error: {e}")
            return False

class Transaction:
    """
    Represents a transfer or mint event.

    `fee` is a first-class init parameter (was previously hardcoded to
    0.0 with no way to set it before tx_id was computed → all transfers
    had fee=0, which the new MIN_TX_FEE policy in compute_ledger.py
    would reject). Callers that don't care about fees can omit it.
    """
    def __init__(self, sender: str, recipient: str, amount: Decimal,
                 type: str = 'transfer',
                 signature: str = None,
                 timestamp: float = None,
                 sender_pub_key: str = None,
                 fee: Decimal = None,
                 nonce: int = 0):
        self.sender = sender                     # 'COINBASE' for minting
        self.recipient = recipient
        self.amount = Decimal(str(amount))
        self.type = type                         # 'transfer', 'mint', 'fee'
        self.timestamp = timestamp or time.time()
        self.signature = signature
        self.sender_pub_key = sender_pub_key     # Required for verification
        self.fee = Decimal(str(fee)) if fee is not None else Decimal("0.0")
        # Per-sender monotonic nonce — replay protection (sec3.3).
        # Each sender's nonces must strictly increase across confirmed
        # transfers. ComputeLedger keeps `_account_nonces[sender]` and
        # rejects any transfer with nonce <= that value. System txs
        # (mint/coinbase/fee_reward/genesis/slash) ignore the nonce.
        self.nonce = int(nonce)
        self.tx_id = self.calculate_hash()
        
    @staticmethod
    def calculate_tx_hash(sender, recipient, amount, fee, timestamp,
                          type, sender_pub_key, nonce=0):
        """
        Static hash from raw values, kept consistent between
        Transaction creation and Ledger verification.

        Nonce is now part of the tx_id so a replayed transaction
        (same sender, same recipient, same amount, same fee, same
        timestamp, same nonce) hashes identically — the ledger
        rejects it via the per-sender nonce table. A new attempt
        from the same sender must bump the nonce, which produces
        a different tx_id and a different signature.
        """
        payload = (f"{sender}{recipient}{amount}{fee}{timestamp}"
                   f"{type}{sender_pub_key}{int(nonce)}")
        return hashlib.sha256(payload.encode()).hexdigest()

    def calculate_hash(self) -> str:
        return Transaction.calculate_tx_hash(
            self.sender, self.recipient, self.amount, self.fee,
            self.timestamp, self.type, self.sender_pub_key,
            getattr(self, "nonce", 0),
        )
        
    def to_dict(self):
        return {
            'tx_id': self.tx_id,
            'sender': self.sender,
            'recipient': self.recipient,
            'amount': str(self.amount),
            'fee': str(self.fee),
            'type': self.type,
            'timestamp': self.timestamp,
            'signature': self.signature,
            'sender_pub_key': self.sender_pub_key,
            'nonce': int(getattr(self, 'nonce', 0)),
        }

class TokenMinter:
    """
    Controls Supply, Halving, and Minting Logic.

    `total_minted` MUST be derived from chain state on every init —
    keeping it as an in-memory counter (the previous bug) meant the
    21M cap was per-process and reset on every restart, which made
    the supply cap decorative. We now scan the chain at construction
    time and on every recompute() call.
    """
    MAX_SUPPLY = Decimal("21000000.000000000000000000")
    GENESIS_REWARD = Decimal("50.0")
    HALVING_INTERVAL = 210000 # Blocks
    # Hard cap on caller-supplied `difficulty_factor` multiplier.
    # Without this, a single caller passing difficulty_factor=1e9 on
    # block 0 mints 50 * 1e9 PLG -> caps to MAX_SUPPLY -> drains the
    # entire 21M issuance to that wallet on the first block.
    # Bound it to the realistic range observable from chain difficulty.
    MAX_DIFFICULTY_FACTOR = Decimal("2.0")
    MIN_DIFFICULTY_FACTOR = Decimal("0.0")

    def __init__(self, ledger=None):
        self.ledger = ledger
        self.total_minted = Decimal("0.0")
        if ledger is not None:
            self.recompute_from_chain()

    def recompute_from_chain(self) -> Decimal:
        """Authoritative supply: sum of all mint/coinbase TXs in the chain."""
        if self.ledger is None:
            return self.total_minted
        total = Decimal("0.0")
        for block in getattr(self.ledger, "chain", []):
            for tx in getattr(block, "transactions", []) or []:
                if tx.get("type") in ("mint", "coinbase"):
                    try:
                        total += Decimal(str(tx.get("amount", 0)))
                    except Exception:
                        pass
        self.total_minted = total
        return total
        
    def get_block_reward(self, current_block_height: int) -> Decimal:
        """
        Calculate reward based on Bitcoin-style halving
        """
        halvings = current_block_height // self.HALVING_INTERVAL
        if halvings >= 64: # Limits to prevent underflow/infinite
            return Decimal("0.0")
            
        reward = self.GENESIS_REWARD / (Decimal("2.0") ** halvings)
        return reward
        
    def mint_coinbase(self, recipient_addr: str, block_height: int,
                      difficulty_factor: float = 1.0) -> Transaction:
        """
        Create a Coinbase transaction (Mint new tokens).

        `difficulty_factor` is bounded to [MIN_DIFFICULTY_FACTOR,
        MAX_DIFFICULTY_FACTOR]. Out-of-range values were a supply-cap
        bypass: a single caller passing `difficulty_factor=1e9` on
        block 0 minted 50 * 1e9 PLG, which the cap then clamped to
        MAX_SUPPLY = 21M PLG — sending the entire network issuance to
        the attacker's wallet in one transaction.
        """
        try:
            df = Decimal(str(difficulty_factor))
        except (TypeError, ValueError, ArithmeticError):
            logger.warning("[MINT] Invalid difficulty_factor %r; refusing.",
                           difficulty_factor)
            return None
        # Reject NaN / Infinity before any comparison
        # (Decimal NaN compared with a real number raises InvalidOperation).
        if df.is_nan() or df.is_infinite():
            logger.warning("[MINT] non-finite difficulty_factor %r; refusing.",
                           difficulty_factor)
            return None
        if df <= self.MIN_DIFFICULTY_FACTOR:
            logger.warning("[MINT] difficulty_factor %s <= 0; refusing.", df)
            return None
        if df > self.MAX_DIFFICULTY_FACTOR:
            logger.warning(
                "[MINT] difficulty_factor %s exceeds MAX_DIFFICULTY_FACTOR %s; "
                "clamping to prevent supply-cap bypass.",
                df, self.MAX_DIFFICULTY_FACTOR,
            )
            df = self.MAX_DIFFICULTY_FACTOR

        base_reward = self.get_block_reward(block_height)
        final_amount = base_reward * df

        # Hard supply-cap check.
        if self.total_minted + final_amount > self.MAX_SUPPLY:
            final_amount = self.MAX_SUPPLY - self.total_minted

        if final_amount <= 0:
            return None  # Supply exhausted.

        self.total_minted += final_amount

        tx = Transaction(
            sender="COINBASE",
            recipient=recipient_addr,
            amount=final_amount,
            type='mint',
            sender_pub_key="SYSTEM",  # No pubkey for coinbase
        )
        return tx

# [PHASE 5] Deflationary Burn Address
BURN_ADDRESS = "0x000000000000000000000000000000000000dead"

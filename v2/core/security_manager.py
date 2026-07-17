"""
Security Manager
Handles authentication, rate limiting, and execution isolation.
"""
import time
import hashlib
import secrets
import logging
import os
import json
from typing import Dict, Any, Optional, List
from threading import Lock
from dataclasses import dataclass
from .ai_sentinel import AISentinel # ✅ NEW

logger = logging.getLogger(__name__)

@dataclass
class SecurityContext:
    """Context for a secure execution request"""
    token: str
    client_id: str
    permissions: List[str]
    created_at: float

class SecurityManager:
    """
    Central security controller for Pluginfer.
    
    Features:
    - Token-based authentication
    - Sliding window rate limiting
    - Isolation abstraction (Docker/Sandboxing)
    """
    
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or secrets.token_hex(32)
        self.active_tokens: Dict[str, SecurityContext] = {}
        self.rate_limits: Dict[str, List[float]] = {}  # client_id -> list of timestamps
        self.lock = Lock()
        
        # Configuration
        self.RATE_LIMIT_WINDOW = 60.0  # 1 minute
        self.MAX_REQUESTS_PER_WINDOW = 60  # 60 requests per minute
        self.RATE_LIMIT_WINDOW = 60.0  # 1 minute
        self.MAX_REQUESTS_PER_WINDOW = 60  # 60 requests per minute
        self.TOKEN_EXPIRY = 3600  # 1 hour
        
        # Billing System (Simple In-Memory Ledger)
        # client_id -> balance_usd
        self.credits: Dict[str, float] = {}
        self.DEFAULT_DEMO_CREDITS = 10.00 # $10 Free Trial
        
        # ✅ AI Sentinel (The Brain)
        self.sentinel = AISentinel()
        
        # Persistence
        import os
        import json
        self.app_data = os.path.join(os.path.expanduser('~'), '.pluginfer')
        if not os.path.exists(self.app_data):
            os.makedirs(self.app_data)
        self.identity_file = os.path.join(self.app_data, 'identity.json')
        self.load_state()

        logger.info("SecurityManager initialized (Billing + AI Sentinel Active)")
        
    def save_state(self):
        """Save tokens and credits to disk"""
        try:
            state = {
                'tokens': {k: {'client': v.client_id, 'perm': v.permissions, 'created': v.created_at} 
                           for k, v in self.active_tokens.items()},
                'credits': self.credits
            }
            with open(self.identity_file, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            logger.error(f"Failed to save identity state: {e}")

    def load_state(self):
        """Load tokens and credits from disk"""
        try:
            if os.path.exists(self.identity_file):
                with open(self.identity_file, 'r') as f:
                    state = json.load(f)
                    
                # Restore credits
                self.credits = state.get('credits', {})
                
                # Restore tokens
                tokens = state.get('tokens', {})
                for k, v in tokens.items():
                    self.active_tokens[k] = SecurityContext(
                        token=k,
                        client_id=v['client'],
                        permissions=v['perm'],
                        created_at=v['created']
                    )
                logger.info(f"Restored {len(self.active_tokens)} identities from disk")
        except (json.JSONDecodeError, ValueError):
            logger.warning("Identity state corrupted or empty. reset.")
            self.credits = {}
            self.active_tokens = {}
        except Exception as e:
            logger.error(f"Failed to load identity state: {e}")

    def generate_token(self, client_id: str, permissions: List[str] = None) -> str:
        """Generate a new access token for a client"""
        # Check if already has token
        for t, c in self.active_tokens.items():
            if c.client_id == client_id:
                return t

        timestamp = time.time()
        # Simple token generation (in prod use JWT)
        payload = f"{client_id}:{timestamp}:{self.secret_key}"
        token_hash = hashlib.sha256(payload.encode()).hexdigest()
        token = f"pk_{token_hash[:32]}"
        
        context = SecurityContext(
            token=token,
            client_id=client_id,
            permissions=permissions or ['execute'],
            created_at=timestamp
        )
        
        with self.lock:
            self.active_tokens[token] = context
            # Initialize billing for new client with FREE TRIAL
            if client_id not in self.credits:
                self.credits[client_id] = self.DEFAULT_DEMO_CREDITS
                logger.info(f"New client {client_id} credited with ${self.DEFAULT_DEMO_CREDITS} (Trial)")
            self.save_state()
            
        logger.info(f"Generated token for client {client_id}")
        return token
        
    def validate_token(self, token: str) -> Optional[SecurityContext]:
        """Validate a token and return its context"""
        with self.lock:
            context = self.active_tokens.get(token)
            
            if not context:
                logger.warning(f"Invalid token attempt: {token[:8]}...")
                return None
            
            # Check expiry
            if time.time() - context.created_at > self.TOKEN_EXPIRY:
                del self.active_tokens[token]
                logger.warning(f"Expired token attempt: {token[:8]}...")
                return None
                
            return context
            
    def check_rate_limit(self, client_id: str) -> bool:
        """
        Check if client has exceeded rate limit.
        Returns True if allowed, False if limited.
        """
        now = time.time()
        
        with self.lock:
            if client_id not in self.rate_limits:
                self.rate_limits[client_id] = []
            
            # Filter out old requests
            history = self.rate_limits[client_id]
            valid_history = [t for t in history if now - t < self.RATE_LIMIT_WINDOW]
            self.rate_limits[client_id] = valid_history
            
            if len(valid_history) >= self.MAX_REQUESTS_PER_WINDOW:
                logger.warning(f"Rate limit exceeded for {client_id}")
                return False
                
            # Add new request
            self.rate_limits[client_id].append(now)
            self.rate_limits[client_id].append(now)
            return True

    def check_threat(self, client_id: str, payload_size: int = 0) -> bool:
        """
        AI Sentinel Check.
        Returns False if client acts like a hacker.
        """
        # Delegate to AI
        if not self.sentinel.analyze_request(client_id, payload_size):
            logger.warning(f"⛔ BLOCKED {client_id} by AI Sentinel")
            return False
        return True
            
    DEFAULT_TIMEOUT_S = 60.0
    DEFAULT_MEM_LIMIT_MB = 4096

    def run_isolated(self, func, *args, timeout: float = None,
                     mem_limit_mb: int = None, **kwargs):
        """
        Run a (trusted) callable with watchdog + crash-safety.

        Semantics:
          * the FUNCTION is trusted (e.g. a registered plugin's `execute`).
          * the INPUTS may be hostile (see SecureSandbox / dynamic_executor
            for code-injection isolation; that's a different layer).
          * we enforce wall-clock timeout, catch all exceptions, set
            soft memory limit on POSIX, and never let plugin death kill
            the host process.

        Earlier this method was `func(*args, **kwargs)` with a comment
        about a future Docker implementation — i.e. not isolated at all.
        """
        import concurrent.futures

        timeout = float(timeout if timeout is not None else self.DEFAULT_TIMEOUT_S)
        mem_limit_mb = int(mem_limit_mb if mem_limit_mb is not None
                           else self.DEFAULT_MEM_LIMIT_MB)

        # Best-effort resource limits on POSIX (Windows: skipped).
        try:
            import resource                                    # type: ignore
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            if soft == resource.RLIM_INFINITY or soft > mem_limit_mb * 1024 * 1024:
                resource.setrlimit(
                    resource.RLIMIT_AS,
                    (mem_limit_mb * 1024 * 1024, hard),
                )
        except (ImportError, OSError, ValueError):
            pass

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(func, *args, **kwargs)
            try:
                return fut.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                logger.error("plugin execution exceeded %.1fs timeout", timeout)
                # Best we can do without subprocess: cancel future, leak the
                # thread (Python can't kill threads). Caller should treat
                # the worker as compromised and recycle it.
                fut.cancel()
                raise TimeoutError(f"plugin exceeded {timeout}s")
            except MemoryError:
                logger.error("plugin exceeded %d MB memory limit", mem_limit_mb)
                raise
            except Exception as e:
                logger.error("plugin failed: %s", e)
                raise

    def cleanup(self):
        """Cleanup expired tokens and rate limit history"""
        now = time.time()
        with self.lock:
            # Cleanup tokens
            expired = [t for t, c in self.active_tokens.items() 
                      if now - c.created_at > self.TOKEN_EXPIRY]
            for t in expired:
                del self.active_tokens[t]
                
            # Cleanup rate limits
            for client in list(self.rate_limits.keys()):
                self.rate_limits[client] = [t for t in self.rate_limits[client] 
                                          if now - t < self.RATE_LIMIT_WINDOW]
                if not self.rate_limits[client]:
                    del self.rate_limits[client]

    def verify_license_key(self, license_key: str, node_id: str) -> bool:
        """
        Verify a JWT-signed license token.

        Pluginfer is open-source and the network is permissionless:
        nothing in the protocol depends on this check. It exists for
        commercial-edition features (priority queue, premium support,
        SLAs). When PLUGINFER_LICENSE_PUBKEY is unset, license checks
        are disabled and ALL keys are accepted with a warning — that
        is the explicit "no licensing" mode.

        When PLUGINFER_LICENSE_PUBKEY is set to an RSA/EC public key
        in PEM, we verify the JWT signature, claims, and exp.
        """
        import os

        if not license_key:
            return False

        pubkey_pem = os.environ.get("PLUGINFER_LICENSE_PUBKEY")
        if not pubkey_pem:
            logger.warning(
                "Licensing disabled: PLUGINFER_LICENSE_PUBKEY not set. "
                "Accepting all keys for node %s.", node_id,
            )
            return True

        try:
            import jwt                                          # type: ignore
        except ImportError:
            logger.error("PyJWT not installed; cannot verify license.")
            return False

        try:
            claims = jwt.decode(
                license_key, pubkey_pem,
                algorithms=["RS256", "ES256"],
                options={"require": ["exp", "sub"]},
            )
        except Exception as e:
            logger.warning("License rejected for %s: %s", node_id, e)
            return False

        # Optional binding: license can require a specific node_id (sub).
        if claims.get("sub") and claims["sub"] != node_id:
            logger.warning("License sub mismatch: license=%s node=%s",
                           claims.get("sub"), node_id)
            return False
        return True

    def check_and_deduct_credit(self, client_id: str, cost: float = 0.01) -> bool:
        """
        Billing Check.
        Returns True if client has enough credits. Deducts cost.
        """
        with self.lock:
            balance = self.credits.get(client_id, 0.0)
            
            if balance >= cost:
                self.credits[client_id] = balance - cost
                if balance - cost < 1.0:
                    logger.warning(f"Low Balance for {client_id}: ${self.credits[client_id]:.2f}")
                return True
            else:
                logger.warning(f"Payment Declined for {client_id}: Insufficient Funds (${balance:.2f})")
                return False

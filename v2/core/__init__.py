"""
Pluginfer Core
==============
Public API for the Pluginfer mesh — chain, consensus, privacy, economics,
networking, execution, security, governance, and intelligence layers.

Legacy modules `mesh_controller` and `secure_mesh_controller` were
superseded by `complete_mesh_controller` + `task_router` and live under
`core/_archive_v2/`.

Imports are organized by concern. If a single subsystem fails to import
(e.g. torch missing for `pluginfer_brain`), the rest of the public API
remains usable — see the soft-import block at the bottom.
"""

from __future__ import annotations

# ----------------------------------------------------------------------
# Plugin / inference layer (always available)
# ----------------------------------------------------------------------
from .plugin_base import PluginBase
from .plugin_registry import PluginRegistry
from .inference_engine import InferenceEngine
from .hardware_detector import HardwareDetector
from .qal_controller import QALController
from .license_validator import LicenseValidator, LicenseTier

# ----------------------------------------------------------------------
# Mesh + routing (canonical v3 surface)
# ----------------------------------------------------------------------
from .complete_mesh_controller import CompleteMeshController
from .task_router import TaskRouter, TaskRequirements, HardwareProfile

# ----------------------------------------------------------------------
# Chain & tokenomics
# ----------------------------------------------------------------------
from .tokenomics import Wallet, Transaction, TokenMinter
from .compute_ledger import Block, ComputeLedger

# ----------------------------------------------------------------------
# Consensus
# ----------------------------------------------------------------------
from .bft_consensus import BFTConsensus, Validator, ValidatorSet

# ----------------------------------------------------------------------
# Privacy / ZK (real Pedersen + Schnorr; pedersen module re-exposed
# in case callers need the raw EC primitives)
# ----------------------------------------------------------------------
from . import pedersen
from .privacy import ZKPrivacy

# ----------------------------------------------------------------------
# Innovation surface — design notes-disclosed primitives
# (see the design notes for filing-ready claim text)
# ----------------------------------------------------------------------
from .gradient_provenance import (
    GradientProvenanceTicket,
    GradientProvenanceWitness,
    create_proof as create_gradient_proof,
    verify_proof as verify_gradient_proof,
)
from .slack_auction import TimeOfDaySlackCurve, default_consumer_curve
from .providers import (
    Provider, JobSpec, Bid, Auction, AuctionResult,
    MeshGPUProvider, OpenAIProvider, AnthropicProvider,
    PRIVACY_PUBLIC, PRIVACY_PRIVATE, PRIVACY_SENSITIVE,
)

# ----------------------------------------------------------------------
# Economics: payments, staking, broker, oracle
# ----------------------------------------------------------------------
from .payments import (
    PaymentGateway,
    PaymentResult,
    StripeGateway,
    PaymentGatewayNotConfigured,
    IdempotencyStore,
    get_default_gateway,
)
from .staking import StakingContract
from .broker import EconomicBroker
from .oracle import PricingOracle

# ----------------------------------------------------------------------
# Networking: peer discovery, DHT, gossip
# ----------------------------------------------------------------------
from .kademlia import (
    KademliaNode, Peer, RoutingTable, KBucket,
    kid_from_pubkey, kid_from_str, xor_distance, bucket_index,
)
from .gossip import GossipProtocol
from .discovery import MeshDiscovery
from .mesh_connector import MeshConnector, MeshChannel
from .remote_provider import JobServer, RemoteProvider

# ----------------------------------------------------------------------
# Execution sandboxes
# ----------------------------------------------------------------------
from .smart_contracts import (
    SmartContractVM, derive_contract_address, hash_contract_code,
    DeployContractPayload, ExecuteContractPayload,
    build_deploy_contract_tx, build_execute_contract_tx, build_slash_tx,
)
from .wasm_executor import WasmExecutor, WasmNotImplementedError
from .secure_sandbox import SecureSandbox, ASTValidator, SecurityViolation
from .slash_evidence import (
    SlashEvidence, BlockHeaderProof, Attestation,
    SlashEvidenceError, construct_evidence, attest, verify_evidence,
    apply_slash, is_outbound_locked,
    UNBONDING_PERIOD_BLOCKS,
)

# ----------------------------------------------------------------------
# Security & monitoring
# ----------------------------------------------------------------------
from .security_manager import SecurityManager, SecurityContext
from .ai_sentinel import AISentinel, ClientProfile
from .auditor import SystemAuditor

# ----------------------------------------------------------------------
# Cross-chain & L2
# ----------------------------------------------------------------------
from .interop import BridgeManager, ChainBridge, InteropNotImplementedError
from .l2_channels import PaymentChannel, StateChannelManager

# ----------------------------------------------------------------------
# Reputation, governance, marketplace
# ----------------------------------------------------------------------
from .reputation import ReputationManager
from .governance import GovernanceDAO, Proposal
from .marketplace import IPMarketplace

# ----------------------------------------------------------------------
# Intelligence: architect, learning optimizer, onboarding
# ----------------------------------------------------------------------
from .architect import TaskArchitect
from .self_learning import SelfLearningOptimizer, ActivityWindow
from .auto_onboarding import (
    AutoOnboardingSystem, UserProfile,
    DynamicPricingEngine, MarketplaceSystem,
)

# ----------------------------------------------------------------------
# Ops: updater, arbiter, scout, job supervision, game detection
# ----------------------------------------------------------------------
from .updater import AutoUpdater
from .arbiter import Arbiter
from .scout import NetworkScout, PeerStats
from .job_supervisor import JobSupervisor
from .game_detector import GameDetector
from .system_doctor import SystemDoctor
from .cost_optimizer import CostOptimalRouter, CostOptimalSelection
from .earnings_estimator import EarningsEstimate

# ----------------------------------------------------------------------
# design notes-claim primitives.
# These need to be importable from `pluginfer.core` because the design notes
# claims describe them as the public surface of the mesh.
# ----------------------------------------------------------------------
from .hot_migration import (                                   # §A14
    TaskCheckpoint, ContinuationEnvelope, HealthSignal,
    make_checkpoint, make_continuation,
    verify_checkpoint_chain, verify_handoff,
)
from .mesh_moe import (                                        # §A8
    MeshMoERouter, ExpertRecord, ExpertOutput, MoEResult,
)
from .predictive_fanout import (                               # §A12
    PredictiveRouter, Predictor, Prediction, SpeculativeJob,
)
from .revenue_distribution import (                            # §A16
    Beneficiary, RevenueSplit, RevenueRule, RevenueProjection,
    split_revenue,
)

# ----------------------------------------------------------------------
# Decision agent (soft dep on torch — graceful degrade if torch absent)
# ----------------------------------------------------------------------
try:
    from .pluginfer_brain import PluginferBrain, NodeContext
    _BRAIN_AVAILABLE = True
except Exception:                                        # pragma: no cover
    PluginferBrain = None        # type: ignore[assignment]
    NodeContext = None           # type: ignore[assignment]
    _BRAIN_AVAILABLE = False

# ----------------------------------------------------------------------
# Distributed training (DiLoCo) — soft dep on torch
# ----------------------------------------------------------------------
try:
    from . import diloco_models, diloco_serialize, diloco_quantize
    from . import diloco_worker, diloco_aggregator
    _DILOCO_AVAILABLE = True
except Exception:                                        # pragma: no cover
    diloco_models = None         # type: ignore[assignment]
    diloco_serialize = None      # type: ignore[assignment]
    diloco_quantize = None       # type: ignore[assignment]
    diloco_worker = None         # type: ignore[assignment]
    diloco_aggregator = None     # type: ignore[assignment]
    _DILOCO_AVAILABLE = False


__all__ = [
    # plugin / inference
    "PluginBase", "PluginRegistry", "InferenceEngine",
    "HardwareDetector", "QALController",
    "LicenseValidator", "LicenseTier",
    # mesh / routing
    "CompleteMeshController",
    "TaskRouter", "TaskRequirements", "HardwareProfile",
    # chain & tokenomics
    "Wallet", "Transaction", "TokenMinter",
    "Block", "ComputeLedger",
    # consensus
    "BFTConsensus", "Validator", "ValidatorSet",
    # privacy
    "pedersen", "ZKPrivacy",
    # economics
    "PaymentGateway", "PaymentResult", "StripeGateway",
    "PaymentGatewayNotConfigured", "IdempotencyStore",
    "get_default_gateway",
    "StakingContract", "EconomicBroker", "PricingOracle",
    # networking
    "KademliaNode", "Peer", "RoutingTable", "KBucket",
    "kid_from_pubkey", "kid_from_str", "xor_distance", "bucket_index",
    "GossipProtocol", "MeshDiscovery",
    "MeshConnector", "MeshChannel",
    "RemoteProvider", "JobServer",
    # execution
    "SmartContractVM", "derive_contract_address", "hash_contract_code",
    "DeployContractPayload", "ExecuteContractPayload",
    "build_deploy_contract_tx", "build_execute_contract_tx", "build_slash_tx",
    "WasmExecutor", "WasmNotImplementedError",
    "SecureSandbox", "ASTValidator", "SecurityViolation",
    # W32 slash-evidence
    "SlashEvidence", "BlockHeaderProof", "Attestation",
    "SlashEvidenceError", "construct_evidence", "attest", "verify_evidence",
    "apply_slash", "is_outbound_locked", "UNBONDING_PERIOD_BLOCKS",
    # security
    "SecurityManager", "SecurityContext",
    "AISentinel", "ClientProfile", "SystemAuditor",
    # interop / L2
    "BridgeManager", "ChainBridge", "InteropNotImplementedError",
    "PaymentChannel", "StateChannelManager",
    # reputation / governance / marketplace
    "ReputationManager", "GovernanceDAO", "Proposal", "IPMarketplace",
    # intelligence
    "TaskArchitect", "SelfLearningOptimizer", "ActivityWindow",
    "AutoOnboardingSystem", "UserProfile",
    "DynamicPricingEngine", "MarketplaceSystem",
    # ops
    "AutoUpdater", "Arbiter", "NetworkScout", "PeerStats",
    "JobSupervisor", "GameDetector",
    "SystemDoctor", "CostOptimalRouter", "CostOptimalSelection",
    "EarningsEstimate",
    # design notes-claim primitives (§A8 / §A12 / §A14 / §A16)
    "TaskCheckpoint", "ContinuationEnvelope", "HealthSignal",
    "make_checkpoint", "make_continuation",
    "verify_checkpoint_chain", "verify_handoff",
    "MeshMoERouter", "ExpertRecord", "ExpertOutput", "MoEResult",
    "PredictiveRouter", "Predictor", "Prediction", "SpeculativeJob",
    "Beneficiary", "RevenueSplit", "RevenueRule", "RevenueProjection",
    "split_revenue",
    # decision agent (may be None if torch missing)
    "PluginferBrain", "NodeContext",
    # distributed training (may be None if torch missing)
    "diloco_models", "diloco_serialize", "diloco_quantize",
    "diloco_worker", "diloco_aggregator",
]

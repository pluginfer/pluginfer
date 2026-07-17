"""HPA-LRD: Hardware-Pressure-Adaptive Low-Rank Distillation.

A trainer that watches hardware telemetry (VRAM, GPU util, GPU temp,
RAM) and adapts its memory footprint in real time so it does not
crash, hang, or freeze the host machine.

Five components, each independently testable:

* ``telemetry``      - sample hardware, emit pressure scalar P in [0,1]
* ``galore_adaptive``- low-rank gradient projection with rank set by P
* ``cooperative``    - VRAM cap + display-compositor yield
* ``teacher_cache``  - disk-tiered teacher prefetch (decouples API latency)
* ``trainer``        - the loop that wires them together

The novel claims are documented in ``the design notes`` (sections
B1-B5). Each claim is implemented in code and CPU-smoke-tested in
``tests/test_hpa_lrd.py``.
"""

from .telemetry import PressureSample, PressureSampler, pressure_scalar
from .galore_adaptive import AdaptiveLowRankProjector, choose_rank
from .cooperative import CooperativeYield, soft_vram_cap_bytes
from .teacher_cache import DiskTeacherCache
from .backend import (
    BackendInfo, detect_backend, select_torch_device,
    memory_cap_bytes, vendor_telemetry_probe,
    synchronize, empty_cache, memory_used_bytes,
)

# §C Liquid Intelligence Layer
from .grain import Grain, GrainMeta, make_grain, fresh_keypair
from .global_aggregator import (
    NonBlockingGlobalAggregator,
    AggregatorPolicy,
)
from .sun_election import (
    SunElection,
    SunElectionPolicy,
    NodeMembership,
    PlanetLink,
    SunOfSunsRing,
    StabilityEMA,
)
from .reverse_auction import (
    ProviderBid,
    BuyerAsk,
    TimeOfUseCurve,
    clear_epoch,
    ProviderEarnings,
    estimate_provider_take,
)

# §J Multi-Model Federation ("Goliath of AIs")
from .model_federation import (
    GenerationRequest,
    GenerationResponse,
    ModelBackend,
    OllamaBackend,
    FilumLocalBackend,
    FederationConfig,
    ModelFederation,
    quick_status as federation_quick_status,
)

__all__ = [
    # §B
    "PressureSample", "PressureSampler", "pressure_scalar",
    "AdaptiveLowRankProjector", "choose_rank",
    "CooperativeYield", "soft_vram_cap_bytes",
    "DiskTeacherCache",
    # vendor-agnostic backend
    "BackendInfo", "detect_backend", "select_torch_device",
    "memory_cap_bytes", "vendor_telemetry_probe",
    "synchronize", "empty_cache", "memory_used_bytes",
    # §C
    "Grain", "GrainMeta", "make_grain", "fresh_keypair",
    "NonBlockingGlobalAggregator", "AggregatorPolicy",
    "SunElection", "SunElectionPolicy", "NodeMembership",
    "PlanetLink", "SunOfSunsRing", "StabilityEMA",
    "ProviderBid", "BuyerAsk", "TimeOfUseCurve",
    "clear_epoch", "ProviderEarnings", "estimate_provider_take",
    # §J Multi-Model Federation
    "GenerationRequest", "GenerationResponse",
    "ModelBackend", "OllamaBackend", "FilumLocalBackend",
    "FederationConfig", "ModelFederation",
    "federation_quick_status",
]

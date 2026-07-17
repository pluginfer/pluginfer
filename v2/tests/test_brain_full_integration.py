"""End-to-end PNIS brain integration test (CP-AI-FINAL).

Exercises the full chain: tokenizer -> backbone -> InferenceEngine ->
PluginferBrainPNIS -> flywheel logging. Asserts:

  1. Brain constructs cleanly from a debug-config backbone
  2. parse_job emits structured_text + a parsed dict (real model output,
     not hardcoded; the model is fresh-init so structure won't be
     correct -- the contract under test is the API surface, not
     accuracy)
  3. route_job / score_provider / price emit untrained_head=True
     markers (honest stub for not-yet-trained heads)
  4. detect_anomaly with attached AnomalyDetectorAutoencoder emits a
     real fp32 score
  5. Flywheel events.jsonl has one row per call, parseable JSON
  6. ABEvaluator picks the lower-loss model on a held-out batch
  7. Untrained labeler / fine_tuner are honest NotImplementedError
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

# Force sys-torch first (same shim as ai/conftest.py).
if "torch" not in sys.modules:
    saved = sys.path[:]
    sys.path[:] = [p for p in sys.path if Path(p).resolve() != V2.resolve()]
    try:
        import torch as _torch  # noqa: F401
    finally:
        sys.path[:] = saved

from ai.data.synthetic_generator import BEHAVIOUR_FEATURE_DIM  # noqa: E402
from ai.flywheel.evaluator import ABEvaluator  # noqa: E402
from ai.flywheel.labeler import LabelingNotImplementedError, OutcomeLabeler  # noqa: E402
from ai.flywheel.fine_tuner import (  # noqa: E402
    FineTuningNotImplementedError,
    WeeklyFineTuner,
)
from ai.inference.engine import InferenceEngine  # noqa: E402
from ai.model.config import ModelConfig  # noqa: E402
from ai.model.heads import AnomalyDetectorAutoencoder  # noqa: E402
from ai.model.transformer import PluginferLM  # noqa: E402
from ai.tokenizer.bpe import BASE_VOCAB_SIZE  # noqa: E402
from ai.tokenizer.tokenizer import PluginferTokenizer  # noqa: E402
from ai.tokenizer.vocab_builder import CorpusBuilder  # noqa: E402
from core.brain_pnis import PluginferBrainPNIS  # noqa: E402


@pytest.fixture(scope="module")
def tokenizer() -> PluginferTokenizer:
    builder = CorpusBuilder(seed=2026)
    corpus = builder.build_all(n_jobs=200, n_provider=100, n_auction=100)
    return PluginferTokenizer.train_new(corpus, vocab_size=BASE_VOCAB_SIZE + 400)


@pytest.fixture(scope="module")
def engine(tokenizer: PluginferTokenizer) -> InferenceEngine:
    cfg = ModelConfig.debug()
    cfg.vocab_size = max(tokenizer.vocab_size + 4, cfg.vocab_size)
    torch.manual_seed(7)
    return InferenceEngine(PluginferLM(cfg), tokenizer, device="cpu")


@pytest.fixture()
def brain(engine: InferenceEngine, tmp_path: Path) -> PluginferBrainPNIS:
    """Brain with NO heads attached -> exercises untrained_head markers."""
    return PluginferBrainPNIS(
        engine,
        flywheel_dir=tmp_path / "flywheel",
        model_checkpoint_hash="test-debug-cfg",
    )


@pytest.fixture()
def brain_with_anomaly(
    engine: InferenceEngine, tmp_path: Path
) -> PluginferBrainPNIS:
    """Brain WITH anomaly head attached -> emits real scores."""
    ae = AnomalyDetectorAutoencoder(input_dim=BEHAVIOUR_FEATURE_DIM)
    return PluginferBrainPNIS(
        engine,
        flywheel_dir=tmp_path / "flywheel",
        anomaly_head=ae,
        model_checkpoint_hash="test-debug-cfg",
    )


# ---------------------------------------------------------------------------
# parse_job (Module 5)
# ---------------------------------------------------------------------------

def test_parse_job_emits_structured_text(brain: PluginferBrainPNIS) -> None:
    out = brain.parse_job("I need to transcribe a 2-hour audio file")
    assert out["input"] == "I need to transcribe a 2-hour audio file"
    assert isinstance(out["structured_text"], str)
    # Untrained model -> parsed dict is best-effort; just assert it's a dict
    assert isinstance(out["parsed"], dict)


# ---------------------------------------------------------------------------
# route_job / score_provider / price = untrained_head markers
# ---------------------------------------------------------------------------

def test_route_job_untrained_marker(brain: PluginferBrainPNIS) -> None:
    out = brain.route_job({"description": "Run SDXL on 1024px image"})
    assert out["untrained_head"] is True
    assert "structured_text" in out


def test_score_provider_untrained_marker(brain: PluginferBrainPNIS) -> None:
    out = brain.score_provider("node_abc", history=[{"verified": True} for _ in range(5)])
    assert out["untrained_head"] is True
    assert out["provider_id"] == "node_abc"
    assert out["n_history_events"] == 5


def test_price_untrained_marker(brain: PluginferBrainPNIS) -> None:
    out = brain.price({"queued_jobs": 10, "active_providers": 50})
    assert out["untrained_head"] is True
    assert "queued_jobs" in out["market_state_keys"]


# ---------------------------------------------------------------------------
# detect_anomaly with attached head -> real score
# ---------------------------------------------------------------------------

def test_detect_anomaly_with_head_emits_score(
    brain_with_anomaly: PluginferBrainPNIS,
) -> None:
    out = brain_with_anomaly.detect_anomaly([0.0] * BEHAVIOUR_FEATURE_DIM)
    assert "score" in out and "is_anomalous" in out
    assert isinstance(out["score"], float)
    assert isinstance(out["is_anomalous"], bool)


def test_detect_anomaly_validates_feature_length(
    brain: PluginferBrainPNIS,
) -> None:
    with pytest.raises(ValueError, match=str(BEHAVIOUR_FEATURE_DIM)):
        brain.detect_anomaly([0.0] * 10)


# ---------------------------------------------------------------------------
# Flywheel logging
# ---------------------------------------------------------------------------

def test_flywheel_log_appends_jsonl(brain: PluginferBrainPNIS, tmp_path: Path) -> None:
    brain.parse_job("Run Llama 3 8B on my dataset")
    brain.route_job({"description": "Score a CSV with phi-3-mini"})
    brain.detect_anomaly([0.0] * BEHAVIOUR_FEATURE_DIM)
    log_path = brain.collector.log_path
    assert log_path.exists()
    rows = [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(rows) == 3
    modules = {row["module"] for row in rows}
    assert modules == {"parse_job", "route_job", "detect_anomaly"}
    for row in rows:
        assert row["model_checkpoint_hash"] == "test-debug-cfg"
        assert row["latency_ms"] >= 0
        assert isinstance(row["request_id"], str) and len(row["request_id"]) >= 8


def test_flywheel_replay_round_trip(brain: PluginferBrainPNIS) -> None:
    brain.parse_job("Hello")
    brain.parse_job("World")
    events = list(brain.collector.replay())
    assert len(events) == 2
    assert events[0].module == "parse_job"


# ---------------------------------------------------------------------------
# Status surface
# ---------------------------------------------------------------------------

def test_brain_status_reports_heads_attached(brain: PluginferBrainPNIS) -> None:
    st = brain.status()
    assert st["heads_attached"] == {
        "job_router": False,
        "provider": False,
        "price": False,
        "anomaly": False,
    }
    assert st["flywheel_events"] >= 0
    assert st["model"] == "PluginferLM"


def test_brain_status_reports_anomaly_head_attached(
    brain_with_anomaly: PluginferBrainPNIS,
) -> None:
    st = brain_with_anomaly.status()
    assert st["heads_attached"]["anomaly"] is True


# ---------------------------------------------------------------------------
# A/B evaluator
# ---------------------------------------------------------------------------

def test_ab_evaluator_picks_lower_loss_winner(
    tokenizer: PluginferTokenizer,
) -> None:
    from torch.utils.data import DataLoader

    from ai.data.dataset import PluginferDataset, TaskMix

    cfg = ModelConfig.debug()
    cfg.vocab_size = max(tokenizer.vocab_size + 4, cfg.vocab_size)
    torch.manual_seed(1)
    a = PluginferLM(cfg)
    torch.manual_seed(99)
    b = PluginferLM(cfg)

    val_ds = PluginferDataset(
        tokenizer,
        seed=42,
        mix=TaskMix(job_router=20, provider_quality=10),
        context_length=64,
    )
    loader = DataLoader(val_ds, batch_size=4, shuffle=False)
    eva = ABEvaluator(loader, device="cpu")
    out = eva.compare(a, b, name_a="seed-1", name_b="seed-99")
    assert out["winner"] in ("seed-1", "seed-99")
    assert isinstance(out["a"].mean_loss, float)
    assert isinstance(out["b"].mean_loss, float)
    # Winner has lower (or equal) mean loss
    if out["winner"] == "seed-1":
        assert out["a"].mean_loss <= out["b"].mean_loss
    else:
        assert out["b"].mean_loss <= out["a"].mean_loss


# ---------------------------------------------------------------------------
# Honest stubs
# ---------------------------------------------------------------------------

def test_outcome_labeler_is_honest_stub() -> None:
    with pytest.raises(LabelingNotImplementedError):
        OutcomeLabeler()


def test_weekly_fine_tuner_is_honest_stub() -> None:
    with pytest.raises(FineTuningNotImplementedError):
        WeeklyFineTuner()

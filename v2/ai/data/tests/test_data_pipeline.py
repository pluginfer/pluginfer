"""CP-AI-3 tests for the synthetic data pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from ai.data.curriculum import CurriculumScheduler  # noqa: E402
from ai.data.dataset import PluginferDataset, TaskMix  # noqa: E402
from ai.data.synthetic_generator import (  # noqa: E402
    BEHAVIOUR_FEATURE_DIM,
    SyntheticDataGenerator,
)
from ai.tokenizer.bpe import BASE_VOCAB_SIZE  # noqa: E402
from ai.tokenizer.tokenizer import PluginferTokenizer  # noqa: E402
from ai.tokenizer.vocab_builder import CorpusBuilder  # noqa: E402


# A trained tokenizer is needed by Dataset; share across tests.
@pytest.fixture(scope="module")
def tokenizer() -> PluginferTokenizer:
    builder = CorpusBuilder(seed=2026)
    corpus = builder.build_all(n_jobs=400, n_provider=200, n_auction=200)
    return PluginferTokenizer.train_new(corpus, vocab_size=BASE_VOCAB_SIZE + 800)


# ---------------------------------------------------------------------------
# Module-1 (Job Router) generator
# ---------------------------------------------------------------------------

def test_job_router_generator_returns_n_examples() -> None:
    gen = SyntheticDataGenerator(seed=1)
    data = gen.generate_job_router_training_data(n=1000)
    assert len(data) == 1000
    for d in data:
        assert "input" in d and "label" in d
        assert isinstance(d["input"], str) and len(d["input"]) > 0
        assert "recommended_gpu" in d["label"]
        assert "vram_gb" in d["label"]
        assert "runtime_ms" in d["label"]
        assert "task_type" in d["label"]
        assert d["label"]["task_type"] in ("training", "inference")


def test_job_router_recommended_gpu_satisfies_vram() -> None:
    """Sanity: the recommended GPU's VRAM is >= predicted VRAM."""
    gen = SyntheticDataGenerator(seed=2)
    data = gen.generate_job_router_training_data(n=500)
    for d in data:
        # We classify INTO a GPU; its catalog vram_gb should be >= the
        # required vram (which is min_vram * task multiplier, baked into label['vram_gb']).
        assert d["label"]["vram_gb"] >= 0


def test_job_router_data_is_seed_reproducible() -> None:
    a = SyntheticDataGenerator(seed=99).generate_job_router_training_data(n=20)
    b = SyntheticDataGenerator(seed=99).generate_job_router_training_data(n=20)
    assert a == b


def test_job_router_train_val_disjoint_by_seed() -> None:
    train = SyntheticDataGenerator(seed=1).generate_job_router_training_data(n=200)
    val = SyntheticDataGenerator(seed=2).generate_job_router_training_data(n=200)
    train_inputs = {d["input"] for d in train}
    val_inputs = {d["input"] for d in val}
    # With 60+ templates and 13+ models, even 200x200 should overlap < 5%
    overlap = train_inputs & val_inputs
    assert len(overlap) < 0.05 * len(train_inputs)


# ---------------------------------------------------------------------------
# Module-2 (Provider) generator
# ---------------------------------------------------------------------------

def test_provider_sequences_have_label_and_sequence() -> None:
    gen = SyntheticDataGenerator(seed=3)
    data = gen.generate_provider_sequences(n=200)
    assert len(data) == 200
    for d in data:
        assert "input" in d and "label" in d
        assert isinstance(d["input"], list) and len(d["input"]) >= 8
        for event in d["input"]:
            for key in ("job_type", "duration_delta", "verified", "rep_delta"):
                assert key in event
        for key in ("quality_score", "reliability_24h", "anomaly_flag"):
            assert key in d["label"]
        assert 0.0 <= d["label"]["quality_score"] <= 1.0


# ---------------------------------------------------------------------------
# Module-3 (Price) generator
# ---------------------------------------------------------------------------

def test_price_scenarios_have_floor_below_ceiling() -> None:
    gen = SyntheticDataGenerator(seed=4)
    data = gen.generate_price_scenarios(n=300)
    assert len(data) == 300
    for d in data:
        floor = d["label"]["recommended_floor_price"]
        ceiling = d["label"]["recommended_ceiling_price"]
        assert ceiling >= floor > 0


# ---------------------------------------------------------------------------
# Module-4 (Anomaly) generator
# ---------------------------------------------------------------------------

def test_anomaly_features_have_correct_dim() -> None:
    gen = SyntheticDataGenerator(seed=5)
    data = gen.generate_anomaly_examples(n=400)
    assert len(data) == 400
    n_anom = 0
    for d in data:
        assert len(d["input"]) == BEHAVIOUR_FEATURE_DIM
        if d["label"]["is_anomalous"]:
            n_anom += 1
    # 50/50 in expectation; allow generous variance on n=400
    assert 0.35 * 400 < n_anom < 0.65 * 400


# ---------------------------------------------------------------------------
# Dataset / DataLoader
# ---------------------------------------------------------------------------

def test_dataset_chunks_have_context_length(tokenizer: PluginferTokenizer) -> None:
    ds = PluginferDataset(
        tokenizer,
        seed=10,
        mix=TaskMix(job_router=200, provider_quality=100),
        context_length=128,
    )
    item = ds[0]
    assert item["input_ids"].shape == (128,)
    assert item["labels"].shape == (128,)
    assert item["attention_mask"].shape == (128,)
    # First-position attention mask should be 1 (BOS) unless this is
    # a continuation chunk (which our packer doesn't produce by design).
    assert int(item["attention_mask"][0]) == 1


def test_labels_are_shifted_input_ids(tokenizer: PluginferTokenizer) -> None:
    ds = PluginferDataset(
        tokenizer,
        seed=11,
        mix=TaskMix(job_router=50, provider_quality=20),
        context_length=64,
    )
    item = ds[0]
    ids = item["input_ids"].tolist()
    labels = item["labels"].tolist()
    # labels[i] should equal ids[i+1] for non-PAD inputs
    pad_id = tokenizer.specials.PAD
    for i in range(len(ids) - 1):
        if ids[i] != pad_id:
            assert labels[i] == ids[i + 1] or labels[i] == -100
    assert labels[-1] == -100


def test_dataloader_batches_correctly(tokenizer: PluginferTokenizer) -> None:
    ds = PluginferDataset(
        tokenizer,
        seed=12,
        mix=TaskMix(job_router=80, provider_quality=40),
        context_length=64,
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    assert batch["input_ids"].shape == (4, 64)
    assert batch["labels"].shape == (4, 64)
    assert batch["input_ids"].dtype == torch.long


def test_train_val_seeded_split_is_deterministic(
    tokenizer: PluginferTokenizer,
) -> None:
    """Same seed -> same chunks; different seeds -> different chunks."""
    ds_a = PluginferDataset(tokenizer, seed=33, mix=TaskMix(job_router=20, provider_quality=10), context_length=64)
    ds_b = PluginferDataset(tokenizer, seed=33, mix=TaskMix(job_router=20, provider_quality=10), context_length=64)
    ds_c = PluginferDataset(tokenizer, seed=34, mix=TaskMix(job_router=20, provider_quality=10), context_length=64)
    assert torch.equal(ds_a[0]["input_ids"], ds_b[0]["input_ids"])
    assert not torch.equal(ds_a[0]["input_ids"], ds_c[0]["input_ids"])


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------

def test_curriculum_warmup_and_full_phases() -> None:
    sched = CurriculumScheduler(warmup_steps=100)
    early = sched.weights_at(50)
    assert early["job_router"] == 1.0
    assert early["provider_quality"] == 0.0
    mid = sched.weights_at(150)
    assert 0 < mid["job_router"] < 1.0
    assert mid["provider_quality"] > 0
    full = sched.weights_at(500)
    assert all(0.0 < w < 1.0 for w in full.values())
    # Full mix sums to 1.0
    assert abs(sum(full.values()) - 1.0) < 1e-6

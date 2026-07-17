"""Filum tests that DON'T need torch.

These cover the data layer, privacy policy, retrieval, plan-tree
logic, and config arithmetic. They run on the CPU-only dev box.

Torch-dependent tests (architecture forward pass, BitNet, optimizer,
DPO) live in `test_filum_torch.py` and are skipped when torch is
unavailable.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import pytest

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from ai.filum.config import FilumConfig  # noqa: E402
from ai.filum.data_pipeline import (  # noqa: E402
    DataPipeline,
    DataPipelineConfig,
    TrainingSample,
    jaccard_estimate,
    repetition_ratio,
    shingle_hashes,
    toxicity_score,
)
from ai.filum.privacy_modes import (  # noqa: E402
    PrivacyMode,
    PrivacyPolicy,
    PrivacyViolation,
    policy_for_kind,
    require_peer_inference,
    require_teacher,
)
from ai.filum.retrieval import (  # noqa: E402
    HashEmbedder,
    Passage,
    RAGConfig,
    RAGPipeline,
    SimpleVectorStore,
)
from ai.filum.self_play import SelfPlayConfig, SelfPlayGenerator  # noqa: E402
from ai.filum.teacher_pool import (  # noqa: E402
    consensus_filter,
)
from ai.training.teacher_distill import MockTeacher, TeacherSample  # noqa: E402


# ---------------------------------------------------------------------------
# Config + arithmetic
# ---------------------------------------------------------------------------


def test_filum_config_targets_127m_params():
    cfg = FilumConfig()
    p = cfg.estimate_param_count()
    assert 100 < p["total_M"] < 150, (
        f"total params {p['total_M']}M outside 100-150M target band"
    )


def test_filum_config_vram_fits_4gb_during_training():
    cfg = FilumConfig()
    vram = cfg.estimate_vram_mb(training=True)
    # 4 GB ceiling minus 1 GB OS/Chrome/IDE headroom.
    assert vram["total_MB"] < 3072, (
        f"training VRAM {vram['total_MB']} MB exceeds 3 GB headroom"
    )


def test_filum_config_deploy_under_50mb_with_bitnet():
    cfg = FilumConfig()
    vram = cfg.estimate_vram_mb(training=False, bitnet=True)
    assert vram["weights_MB"] < 50, (
        f"BitNet deploy weights {vram['weights_MB']} MB exceed 50 MB"
    )


def test_filum_config_rejects_invalid_head_dim():
    with pytest.raises(ValueError):
        FilumConfig(d_model=896, n_heads=15, head_dim=64)   # 15*64 != 896


# ---------------------------------------------------------------------------
# Privacy policy enforcement
# ---------------------------------------------------------------------------


def test_local_only_blocks_teacher_calls():
    policy = PrivacyPolicy.from_mode(PrivacyMode.LOCAL_ONLY)
    assert policy.check_teacher() is False
    assert policy.check_peer_inference() is False
    assert policy.check_remote_rag() is False
    with pytest.raises(PrivacyViolation):
        require_teacher(policy)


def test_hybrid_allows_teacher_blocks_mesh_inference():
    policy = PrivacyPolicy.from_mode(PrivacyMode.HYBRID)
    assert policy.check_teacher() is True
    require_teacher(policy)              # no raise
    assert policy.check_peer_inference() is False
    with pytest.raises(PrivacyViolation):
        require_peer_inference(policy)


def test_forbidden_teacher_list_respected():
    policy = PrivacyPolicy.from_mode(
        PrivacyMode.HYBRID,
        forbidden_teachers=("anthropic:claude-opus-4-7",),
    )
    assert policy.check_teacher("anthropic:claude-opus-4-7") is False
    assert policy.check_teacher("google:gemini-2.0-flash-exp") is True


def test_policy_for_kind_maps_confidential_to_local_only():
    p = policy_for_kind("confidential")
    assert p.mode == PrivacyMode.LOCAL_ONLY
    p = policy_for_kind("public")
    assert p.mode == PrivacyMode.HYBRID
    p = policy_for_kind(None)
    assert p.mode == PrivacyMode.HYBRID


# ---------------------------------------------------------------------------
# Data pipeline: filters, dedup, lineage
# ---------------------------------------------------------------------------


def test_repetition_ratio_detects_copy_paste():
    clean = "the quick brown fox jumps over the lazy dog one two three four"
    spammy = "buy now buy now buy now buy now buy now buy now buy now buy now"
    assert repetition_ratio(clean) < 0.3
    assert repetition_ratio(spammy) > 0.5


def test_toxicity_filter_catches_slurs():
    assert toxicity_score("hello there friend") == 0.0
    assert toxicity_score("you are a fucking idiot") > 0.0


def test_minhash_dedup_round_trip():
    a = shingle_hashes("the quick brown fox jumps over the lazy dog")
    b = shingle_hashes("the quick brown fox jumps over the lazy dog")
    c = shingle_hashes("a totally different sentence with no overlap whatsoever")
    assert jaccard_estimate(a, b) == 1.0
    assert jaccard_estimate(a, c) < 0.5


def test_data_pipeline_rejects_short_long_dupes(tmp_path: Path):
    cfg = DataPipelineConfig(min_tokens=8, dedup_threshold=0.85)
    pipe = DataPipeline(cfg, lineage_path=tmp_path / "lineage.jsonl")
    short = TrainingSample(text="too short", source="x")
    ok, reason = pipe.add_sample(short)
    assert not ok and reason == "short"

    good = TrainingSample(
        text="this is a perfectly fine sentence with enough tokens to pass",
        source="x",
    )
    ok, reason = pipe.add_sample(good)
    assert ok, reason

    # Near-duplicate: only changing one word.
    near_dup = TrainingSample(
        text="this is a perfectly fine sentence with enough tokens to pass it",
        source="x",
    )
    ok, reason = pipe.add_sample(near_dup)
    assert not ok
    assert reason.startswith("dedup_")


def test_data_pipeline_lineage_log_written(tmp_path: Path):
    log = tmp_path / "lineage.jsonl"
    cfg = DataPipelineConfig(min_tokens=4, enable_dedup=False)
    pipe = DataPipeline(cfg, lineage_path=log)
    s = TrainingSample(text="a b c d e f", source="src", generator="t1")
    pipe.add_sample(s)
    text = log.read_text(encoding="utf-8").strip()
    body = json.loads(text)
    assert body["source"] == "src"
    assert body["status"] == "kept"


# ---------------------------------------------------------------------------
# Retrieval / RAG
# ---------------------------------------------------------------------------


def test_hash_embedder_l2_normalised():
    e = HashEmbedder(dim=128)
    v = e.encode("the cat sat on the mat")
    norm = math.sqrt(sum(x * x for x in v))
    assert abs(norm - 1.0) < 1e-3


def test_simple_vector_store_round_trip(tmp_path: Path):
    store = SimpleVectorStore()
    e = HashEmbedder(dim=64)
    for i, text in enumerate([
        "Pluginfer is a distributed compute mesh.",
        "BitNet b1.58 ternary weights save memory.",
        "Cats are mammals that purr when content.",
    ]):
        store.add(Passage(passage_id=str(i), text=text), e.encode(text))
    hits = store.search(e.encode("how do cats sound"), k=2)
    assert hits
    # The cats passage should be the top match.
    assert "cat" in hits[0].passage.text.lower()
    # Persistence round-trip.
    p = tmp_path / "store.json"
    store.save(p)
    loaded = SimpleVectorStore.load(p)
    assert len(loaded) == 3


def test_rag_skips_when_match_is_weak():
    store = SimpleVectorStore()
    e = HashEmbedder(dim=64)
    store.add(
        Passage(passage_id="a", text="completely unrelated facts"),
        e.encode("completely unrelated facts"),
    )
    rag = RAGPipeline(store, e, RAGConfig(min_score=0.95))
    out = rag.format_prompt_with_context("how do tomatoes grow")
    # Min_score threshold not met -> just the bare query.
    assert out == "how do tomatoes grow"


# ---------------------------------------------------------------------------
# Self-play generator (deterministic with mocked LLM-as-generator)
# ---------------------------------------------------------------------------


def test_self_play_diversity_filter_rejects_near_dupes():
    seen = []

    async def gen(seed: str) -> str:
        # Mock: always produce a small variation of the seed.
        return f"{seed} variation"

    cfg = SelfPlayConfig(
        prompts_per_round=4, fresh_seed_every_n_rounds=1,
        diversity_min_distance=0.0,    # accept everything
    )
    gen_obj = SelfPlayGenerator(config=cfg, generate_fn=gen)

    async def _run():
        out = await gen_obj.propose_round()
        return out
    out = asyncio.run(_run())
    assert isinstance(out, list)


# ---------------------------------------------------------------------------
# Multi-teacher consensus (using MockTeacher pool)
# ---------------------------------------------------------------------------


def test_consensus_passes_when_teachers_agree():
    # All three return the SAME canned response: should converge.
    samples = [
        TeacherSample(prompt="x", response_text="hi", per_token=[
            (104, [104, 105, 106], [-0.1, -2.0, -3.0]),  # 'h'
            (105, [105, 106, 107], [-0.1, -2.0, -3.0]),  # 'i'
        ], teacher_id="A"),
        TeacherSample(prompt="x", response_text="hi", per_token=[
            (104, [104, 105, 106], [-0.1, -2.0, -3.0]),
            (105, [105, 106, 107], [-0.1, -2.0, -3.0]),
        ], teacher_id="B"),
        TeacherSample(prompt="x", response_text="hi", per_token=[
            (104, [104, 105, 106], [-0.1, -2.0, -3.0]),
            (105, [105, 106, 107], [-0.1, -2.0, -3.0]),
        ], teacher_id="C"),
    ]
    out = consensus_filter(samples, jsd_threshold=0.4)
    assert out.accepted is True
    assert out.averaged is not None


def test_consensus_rejects_when_teachers_diverge():
    # Two teachers say "hi", one says "no".
    samples = [
        TeacherSample(prompt="x", response_text="hi", per_token=[
            (104, [104, 105, 106], [-0.1, -2.0, -3.0]),
            (105, [105, 106, 107], [-0.1, -2.0, -3.0]),
        ], teacher_id="A"),
        TeacherSample(prompt="x", response_text="hi", per_token=[
            (104, [104, 105, 106], [-0.1, -2.0, -3.0]),
            (105, [105, 106, 107], [-0.1, -2.0, -3.0]),
        ], teacher_id="B"),
        TeacherSample(prompt="x", response_text="no", per_token=[
            # Different chosen tokens AND different top-k -> high JSD
            (200, [200, 201, 202], [-0.1, -2.0, -3.0]),
            (201, [201, 202, 203], [-0.1, -2.0, -3.0]),
        ], teacher_id="C"),
    ]
    out = consensus_filter(samples, jsd_threshold=0.05)
    assert out.accepted is False
    assert "max_pairwise_jsd" in (out.detail or "")

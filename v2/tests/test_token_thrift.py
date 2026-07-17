"""§HG13f + HG13h — semantic cache + prompt compression, in isolation.

Deterministic (lexical embedder + injected clock): near-duplicate
prompts hit the semantic cache above threshold and miss below it;
every compression transform is opt-in, itemised, and reduces the
estimated token count. No torch, no network.
"""

from __future__ import annotations

import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import pytest

from governance.token_thrift import (
    PromptCompressor, SemanticCache, _cosine, _ngram_embed,
)


class _Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def _body(text, **extra):
    return {"model": "m", "messages": [{"role": "user", "content": text}],
            **extra}


# ---------------------------------------------------------------------------
# SemanticCache
# ---------------------------------------------------------------------------

def test_embedding_is_deterministic_and_normalised():
    a = _ngram_embed("the quick brown fox")
    b = _ngram_embed("the quick brown fox")
    assert a == b
    assert _cosine(a, a) == pytest.approx(1.0, abs=1e-6)


def test_near_duplicate_hits_above_threshold():
    sc = SemanticCache(threshold=0.9, cache_all=True)
    resp = {"choices": [{"message": {"content": "answer"}}]}
    sc.put(_body("What is the capital of France?"), resp, 0.01)
    # Injected trailing whitespace / punctuation — same question.
    hit = sc.get(_body("What is the capital of France?  "))
    assert hit is not None
    resp_out, billed, sim = hit
    assert resp_out == resp
    assert billed == pytest.approx(0.01)
    assert sim >= 0.9


def test_different_question_misses():
    sc = SemanticCache(threshold=0.9, cache_all=True)
    sc.put(_body("What is the capital of France?"),
           {"a": 1}, 0.01)
    assert sc.get(_body("Explain quantum chromodynamics in detail")) \
        is None
    assert sc.misses == 1


def test_metadata_must_match_exactly():
    sc = SemanticCache(threshold=0.5, cache_all=True)
    sc.put(_body("hello", max_tokens=100), {"a": 1}, 0.01)
    # Same text, different max_tokens → different bucket, no hit.
    assert sc.get(_body("hello", max_tokens=200)) is None


def test_sampling_requests_skipped_by_default():
    sc = SemanticCache(threshold=0.9)     # cache_all off
    sc.put(_body("hello", temperature=0.7), {"a": 1}, 0.01)
    assert sc.get(_body("hello", temperature=0.7)) is None


def test_threshold_validated():
    with pytest.raises(ValueError):
        SemanticCache(threshold=0.2)


def test_pluggable_backend_is_labelled():
    sc = SemanticCache(embed_fn=lambda t: [1.0, 0.0],
                       backend_name="fake-neural")
    assert sc.backend_name == "fake-neural"
    sc2 = SemanticCache()
    assert sc2.backend_name == "lexical-3gram"


# ---------------------------------------------------------------------------
# PromptCompressor — every transform opt-in and itemised
# ---------------------------------------------------------------------------

def test_disabled_by_default_is_identity():
    pc = PromptCompressor()
    assert pc.enabled is False
    body = _body("hello   world")
    out, rep = pc.compress(body)
    assert out == body
    assert rep["applied"] == []
    assert rep["tokens_removed_est"] == 0


def test_dedup_exact_drops_repeated_messages():
    pc = PromptCompressor(dedup_exact=True)
    body = {"model": "m", "messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "same tool output"},
        {"role": "user", "content": "same tool output"},
        {"role": "user", "content": "same tool output"},
    ]}
    out, rep = pc.compress(body)
    assert len(out["messages"]) == 2
    assert any("dedup_exact" in a for a in rep["applied"])
    assert rep["tokens_removed_est"] > 0


def test_collapse_whitespace_shrinks_content():
    pc = PromptCompressor(collapse_whitespace=True)
    body = _body("a" + "   \n\n   " * 50 + "b")
    out, rep = pc.compress(body)
    assert "collapse_whitespace" in rep["applied"]
    assert out["messages"][0]["content"] == "a b"


def test_max_input_tokens_prunes_oldest_history_keeps_system():
    pc = PromptCompressor(max_input_tokens=50, keep_last=2)
    long = "word " * 200                        # ~250 chars ≈ 62 tok
    body = {"model": "m", "messages": [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": long},
        {"role": "assistant", "content": long},
        {"role": "user", "content": "recent-1"},
        {"role": "user", "content": "recent-2"},
    ]}
    out, rep = pc.compress(body)
    roles = [m["content"] for m in out["messages"]]
    assert "SYSTEM" in roles                     # system always kept
    assert "recent-2" in roles                   # newest kept
    assert long not in roles                      # oldest pruned
    assert any("prune_history" in a for a in rep["applied"])


def test_pluggable_compress_fn():
    pc = PromptCompressor(compress_fn=lambda t: t[:5])
    out, rep = pc.compress(_body("compress me please"))
    assert out["messages"][0]["content"] == "compr"
    assert "compress_fn" in rep["applied"]

"""Unit tests for the BPE trainer.

These tests exercise the algorithm with small, hand-checkable corpora
so a regression fingerprints exactly. CP-AI-1 compression-ratio gate
runs in `test_roundtrip.py` against a larger generated corpus.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from ai.tokenizer.bpe import (  # noqa: E402
    BASE_VOCAB_SIZE,
    BPETrainer,
    BYTE_OFFSET,
    N_BYTE_TOKENS,
)
from ai.tokenizer.special_tokens import N_SPECIAL  # noqa: E402


def test_initial_vocab_layout() -> None:
    t = BPETrainer(vocab_size=BASE_VOCAB_SIZE + 10)
    # Specials get empty bytes (their string form is owned by tokenizer.py)
    for i in range(N_SPECIAL):
        assert t.vocab[i] == b""
    # Each byte b is at id b + BYTE_OFFSET
    for b in range(N_BYTE_TOKENS):
        assert t.vocab[BYTE_OFFSET + b] == bytes([b])
    assert t.actual_vocab_size == BASE_VOCAB_SIZE


def test_rejects_undersized_vocab() -> None:
    with pytest.raises(ValueError):
        BPETrainer(vocab_size=BASE_VOCAB_SIZE - 1)


def test_train_learns_obvious_repeating_pair() -> None:
    # 'ab' repeats - should be the very first merge.
    corpus = ["ababab", "ababababab", "ab ab ab"]
    t = BPETrainer(vocab_size=BASE_VOCAB_SIZE + 4)
    t.train(corpus)
    assert len(t.merges) >= 1
    # First merge should be (id_a, id_b) -> first new id.
    a_id = ord("a") + BYTE_OFFSET
    b_id = ord("b") + BYTE_OFFSET
    assert t.merges[0][0] == a_id and t.merges[0][1] == b_id
    assert t.merges[0][2] == BASE_VOCAB_SIZE


def test_encode_applies_learned_merge() -> None:
    corpus = ["abababab abab"]
    t = BPETrainer(vocab_size=BASE_VOCAB_SIZE + 1)  # exactly one merge slot
    t.train(corpus)
    ids = t.encode("abab")
    # Without merges this is 4 ids. With (a,b)->X, "abab" -> [X, X] (2 ids).
    assert len(ids) == 2
    decoded = t.decode(ids)
    assert decoded == "abab"


def test_roundtrip_lossless_unicode() -> None:
    corpus = ["hello world", "the quick brown fox", "GPU mesh routing"]
    t = BPETrainer(vocab_size=BASE_VOCAB_SIZE + 50)
    t.train(corpus)
    for s in [
        "hello world",
        "an unseen sentence about routing",
        "emoji free text",
        "newlines\nand\ttabs",
        "punctuation: a, b, c! d? e.",
        "non-ASCII: cafe -- naive -- voila",
    ]:
        assert t.decode(t.encode(s)) == s


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    corpus = ["hello pluginfer", "pluginfer mesh", "mesh inference"]
    t = BPETrainer(vocab_size=BASE_VOCAB_SIZE + 8)
    t.train(corpus)
    path = tmp_path / "tk.json"
    t.save(path)
    t2 = BPETrainer.load(path)
    assert t2.actual_vocab_size == t.actual_vocab_size
    assert t2.merges == t.merges
    text = "pluginfer mesh inference"
    assert t.encode(text) == t2.encode(text)
    assert t2.decode(t2.encode(text)) == text


def test_save_format_is_json(tmp_path: Path) -> None:
    t = BPETrainer(vocab_size=BASE_VOCAB_SIZE + 4)
    t.train(["aaaa", "aaa"])
    path = tmp_path / "tk.json"
    t.save(path)
    body = json.loads(path.read_text(encoding="utf-8"))
    assert body["format_version"] == 1
    assert body["n_special"] == N_SPECIAL
    assert body["byte_offset"] == BYTE_OFFSET
    assert isinstance(body["merges"], list)
    if body["merges"]:
        assert len(body["merges"][0]) == 3  # (a, b, new)


def test_load_rejects_mismatched_specials(tmp_path: Path) -> None:
    t = BPETrainer(vocab_size=BASE_VOCAB_SIZE + 4)
    t.train(["aaaa"])
    path = tmp_path / "tk.json"
    t.save(path)
    body = json.loads(path.read_text(encoding="utf-8"))
    body["special_token_names"] = ["<MUTATED>"] + body["special_token_names"][1:]
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="special_token_names mismatch"):
        BPETrainer.load(path)


def test_decode_skips_special_ids() -> None:
    t = BPETrainer(vocab_size=BASE_VOCAB_SIZE + 4)
    t.train(["abcabc"])
    ids = t.encode("abc")
    # Sprinkle special ids in (PAD, BOS, EOS): BPETrainer.decode should drop them.
    augmented = [0] + ids + [1, 2]
    assert t.decode(augmented) == "abc"


def test_encode_handles_empty_string() -> None:
    t = BPETrainer(vocab_size=BASE_VOCAB_SIZE + 4)
    t.train(["aaaa"])
    assert t.encode("") == []
    assert t.decode([]) == ""


def test_train_stops_early_when_no_repeated_pair() -> None:
    # All single-char distinct -> no pair occurs more than once after the
    # first merge consumes it; trainer should exit early.
    t = BPETrainer(vocab_size=BASE_VOCAB_SIZE + 1000)
    t.train(["abcdefghij"])
    assert t.actual_vocab_size <= BASE_VOCAB_SIZE + 100  # well below requested

"""Tests for A9: Verifiable Inference Receipts (ZK)."""

import hashlib

import pytest

from core.inference_provenance import (
    InferenceTicket,
    make_ticket,
    verify_ticket,
)


def _model() -> bytes:
    return hashlib.sha256(b"filum-127m-checkpoint").digest()


def test_round_trip_verifies():
    t = make_ticket(
        input_bytes=b"summarise this article",
        output_bytes=b"a short summary",
        model_hash=_model(),
    )
    assert verify_ticket(t) is True


def test_binding_to_receipt_hashes_succeeds_when_aligned():
    inp = b"x"
    out = b"y"
    m = _model()
    t = make_ticket(input_bytes=inp, output_bytes=out, model_hash=m)
    assert verify_ticket(
        t,
        expected_input_hash_hex=hashlib.sha256(inp).hexdigest(),
        expected_output_hash_hex=hashlib.sha256(out).hexdigest(),
        expected_model_hash_hex=m.hex(),
    ) is True


def test_binding_to_wrong_input_hash_fails():
    t = make_ticket(input_bytes=b"x", output_bytes=b"y", model_hash=_model())
    assert verify_ticket(
        t,
        expected_input_hash_hex=hashlib.sha256(b"OTHER").hexdigest(),
    ) is False


def test_binding_to_wrong_output_hash_fails():
    t = make_ticket(input_bytes=b"x", output_bytes=b"y", model_hash=_model())
    assert verify_ticket(
        t,
        expected_output_hash_hex=hashlib.sha256(b"OTHER").hexdigest(),
    ) is False


def test_binding_to_wrong_model_hash_fails():
    t = make_ticket(input_bytes=b"x", output_bytes=b"y", model_hash=_model())
    assert verify_ticket(
        t,
        expected_model_hash_hex="00" * 32,
    ) is False


def test_tampered_output_hash_breaks_verification():
    t = make_ticket(input_bytes=b"x", output_bytes=b"y", model_hash=_model())
    t.output_hash = "00" * 32
    assert verify_ticket(t) is False


def test_swapped_pok_breaks_verification():
    t = make_ticket(input_bytes=b"x", output_bytes=b"y", model_hash=_model())
    t.pok_I, t.pok_O = t.pok_O, t.pok_I
    assert verify_ticket(t) is False


def test_json_round_trip_preserves_proof():
    t = make_ticket(input_bytes=b"hello", output_bytes=b"world",
                    model_hash=_model())
    blob = t.to_json()
    t2 = InferenceTicket.from_json(blob)
    assert verify_ticket(t2) is True


def test_ticket_does_not_reveal_input_plaintext():
    """The ticket must NOT contain the original input bytes -- it
    only commits to their sha256."""
    secret = b"this is a confidential prompt with a SECRET inside"
    t = make_ticket(input_bytes=secret, output_bytes=b"y",
                    model_hash=_model())
    blob = t.to_json()
    assert b"SECRET" not in blob.encode()
    assert b"confidential" not in blob.encode()

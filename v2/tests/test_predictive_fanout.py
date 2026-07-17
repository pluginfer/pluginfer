"""Tests for A12: Predictive Request Fan-Out."""

from typing import List

from core.predictive_fanout import (
    Prediction,
    Predictor,
    PredictiveRouter,
    SpeculativeJob,
)


def _stub_predictor(predictions: List[Prediction]) -> Predictor:
    return Predictor(fn=lambda hist: list(predictions))


def _stub_dispatch_factory(answer_map=None):
    answer_map = answer_map or {}

    def dispatch(payload):
        # Default: echo the payload as a serialized string.
        import json
        key = json.dumps(payload, sort_keys=True)
        return answer_map.get(key, key.encode())
    return dispatch


def test_high_probability_prediction_dispatches_speculative():
    pred = Prediction(request_payload={"q": "what's the weather"},
                      probability=0.85)
    r = PredictiveRouter(
        predictor=_stub_predictor([pred]),
        dispatch=_stub_dispatch_factory(),
        precision_floor=0.6,
        max_speculative_inflight=2,
    )
    launched = r.run_speculation()
    assert len(launched) == 1
    assert launched[0].is_complete


def test_low_probability_prediction_skipped():
    pred = Prediction(request_payload={"q": "x"}, probability=0.30)
    r = PredictiveRouter(
        predictor=_stub_predictor([pred]),
        dispatch=_stub_dispatch_factory(),
        precision_floor=0.6,
    )
    assert r.run_speculation() == []


def test_match_actual_returns_completed_speculation():
    payload = {"q": "the answer to life"}
    pred = Prediction(request_payload=payload, probability=0.95)
    r = PredictiveRouter(
        predictor=_stub_predictor([pred]),
        dispatch=_stub_dispatch_factory({
            '{"q": "the answer to life"}': b"42",
        }),
        precision_floor=0.5,
    )
    r.run_speculation()
    spec = r.match_actual(payload)
    assert spec is not None
    assert spec.output_bytes == b"42"


def test_match_actual_returns_none_when_no_match():
    pred = Prediction(request_payload={"q": "A"}, probability=0.95)
    r = PredictiveRouter(
        predictor=_stub_predictor([pred]),
        dispatch=_stub_dispatch_factory(),
        precision_floor=0.5,
    )
    r.run_speculation()
    assert r.match_actual({"q": "B"}) is None


def test_max_inflight_caps_dispatch():
    preds = [
        Prediction(request_payload={"q": f"q{i}"}, probability=0.99)
        for i in range(10)
    ]
    r = PredictiveRouter(
        predictor=_stub_predictor(preds),
        dispatch=_stub_dispatch_factory(),
        precision_floor=0.5,
        max_speculative_inflight=3,
    )
    launched = r.run_speculation()
    # All three slots used; remaining 7 not dispatched.
    assert len(launched) == 3


def test_stops_at_first_below_floor_due_to_sort():
    # Predictor returns unsorted; Predictor.predict() sorts desc by
    # probability so the first-below-floor stops the loop early.
    preds = [
        Prediction(request_payload={"q": "low"}, probability=0.20),
        Prediction(request_payload={"q": "high"}, probability=0.95),
    ]
    r = PredictiveRouter(
        predictor=_stub_predictor(preds),
        dispatch=_stub_dispatch_factory(),
        precision_floor=0.5,
    )
    launched = r.run_speculation()
    assert len(launched) == 1
    assert launched[0].payload == {"q": "high"}


def test_dispatch_error_marks_discarded_and_doesnt_block_inflight():
    pred = Prediction(request_payload={"q": "boom"}, probability=0.95)

    def boom(payload):
        raise RuntimeError("upstream down")

    r = PredictiveRouter(
        predictor=_stub_predictor([pred]),
        dispatch=boom,
        precision_floor=0.5,
    )
    launched = r.run_speculation()
    assert len(launched) == 1
    assert launched[0].is_discarded
    # Slot freed -- next run_speculation can try again.
    assert r.stats()["inflight"] == 0


def test_observe_appends_to_history_with_cap():
    pred = Prediction(request_payload={"q": "z"}, probability=0.30)
    r = PredictiveRouter(
        predictor=_stub_predictor([pred]),
        dispatch=_stub_dispatch_factory(),
        precision_floor=0.9, history_max=4,
    )
    for i in range(10):
        r.observe({"q": f"req{i}"})
    assert r.stats()["history"] == 4


def test_no_re_dispatch_for_already_completed_payload():
    pred = Prediction(request_payload={"q": "stable"}, probability=0.95)
    r = PredictiveRouter(
        predictor=_stub_predictor([pred]),
        dispatch=_stub_dispatch_factory(),
        precision_floor=0.5,
    )
    launched_1 = r.run_speculation()
    launched_2 = r.run_speculation()    # same predictor output
    assert len(launched_1) == 1
    assert len(launched_2) == 0


def test_discard_stale_drops_old_completions():
    import time as _t
    pred = Prediction(request_payload={"q": "old"}, probability=0.95)
    r = PredictiveRouter(
        predictor=_stub_predictor([pred]),
        dispatch=_stub_dispatch_factory(),
        precision_floor=0.5,
    )
    r.run_speculation()
    # Force its completion timestamp far in the past.
    for h, job in r._completed.items():
        job.completed_at_ns = _t.time_ns() - 600 * 1_000_000_000
    n = r.discard_stale(max_age_seconds=60)
    assert n == 1
    assert r.stats()["completed_unconsumed"] == 0

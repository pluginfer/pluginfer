"""CP-AI-5 tests: engine + INT8 quantizer + FastAPI server."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from ai.inference.benchmarks import benchmark  # noqa: E402
from ai.inference.engine import GenerationParams, InferenceEngine  # noqa: E402
from ai.inference.quantization import (  # noqa: E402
    INT8Quantizer,
    QuantizedLinear,
    measure_param_bytes,
    quantize_module_in_place,
)
from ai.inference.server import build_app  # noqa: E402
from ai.model.config import ModelConfig  # noqa: E402
from ai.model.transformer import PluginferLM  # noqa: E402
from ai.tokenizer.bpe import BASE_VOCAB_SIZE  # noqa: E402
from ai.tokenizer.tokenizer import PluginferTokenizer  # noqa: E402
from ai.tokenizer.vocab_builder import CorpusBuilder  # noqa: E402


@pytest.fixture(scope="module")
def tokenizer() -> PluginferTokenizer:
    builder = CorpusBuilder(seed=2026)
    corpus = builder.build_all(n_jobs=300, n_provider=150, n_auction=150)
    return PluginferTokenizer.train_new(corpus, vocab_size=BASE_VOCAB_SIZE + 600)


@pytest.fixture(scope="module")
def engine(tokenizer: PluginferTokenizer) -> InferenceEngine:
    cfg = ModelConfig.debug()
    cfg.vocab_size = max(tokenizer.vocab_size + 4, cfg.vocab_size)
    torch.manual_seed(0)
    model = PluginferLM(cfg)
    return InferenceEngine(model, tokenizer, device="cpu")


# ---------------------------------------------------------------------------
# INT8 quantizer
# ---------------------------------------------------------------------------

def test_quantize_weight_roundtrip_close() -> None:
    q = INT8Quantizer()
    torch.manual_seed(1)
    W = torch.randn(64, 32) * 0.5
    w_int8, scale = q.quantize_weight(W)
    assert w_int8.dtype == torch.int8
    W_approx = q.dequantize(w_int8, scale)
    err = (W - W_approx).abs().max().item()
    # Absmax INT8 -> max-abs error <= scale (= max|W| / 127)
    expected_bound = float(W.abs().max().item() / 127.0)
    # Allow a tiny numerical fudge factor for round() ties.
    assert err <= expected_bound * 1.01 + 1e-6


def test_quantize_module_replaces_linears(engine: InferenceEngine) -> None:
    import copy
    import torch.nn as nn

    model = copy.deepcopy(engine.model)
    n_linears_before = sum(isinstance(m, nn.Linear) for m in model.modules())
    assert n_linears_before > 0
    quantize_module_in_place(model)
    n_linears_after = sum(isinstance(m, nn.Linear) for m in model.modules())
    n_quant_after = sum(isinstance(m, QuantizedLinear) for m in model.modules())
    assert n_linears_after == 0
    assert n_quant_after == n_linears_before


def test_quantized_model_is_smaller(engine: InferenceEngine) -> None:
    import copy

    fp32 = copy.deepcopy(engine.model)
    bytes_fp32 = measure_param_bytes(fp32)
    int8 = copy.deepcopy(engine.model)
    quantize_module_in_place(int8)
    bytes_int8 = measure_param_bytes(int8)
    assert bytes_int8 < bytes_fp32 * 0.85, (
        f"expected < 85% of fp32; got {bytes_int8} / {bytes_fp32} = "
        f"{bytes_int8 / bytes_fp32:.2f}"
    )


def test_quantized_model_still_produces_logits(engine: InferenceEngine) -> None:
    import copy

    model = copy.deepcopy(engine.model)
    quantize_module_in_place(model)
    ids = torch.randint(0, model.config.vocab_size, (1, 8))
    logits = model(ids)
    assert logits.shape == (1, 8, model.config.vocab_size)
    assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

def test_engine_generate_returns_string(engine: InferenceEngine) -> None:
    text = engine.generate(
        "Run SDXL inference",
        GenerationParams(max_new_tokens=8, temperature=0.0),
    )
    assert isinstance(text, str)


def test_engine_generate_ids_extends_correctly(engine: InferenceEngine) -> None:
    prompt_ids = engine.tokenizer.encode("Hello", add_bos=True)
    out = engine.generate_ids(
        prompt_ids,
        GenerationParams(max_new_tokens=5, temperature=0.0, stop_on_eos=False),
    )
    assert len(out) == len(prompt_ids) + 5
    assert out[: len(prompt_ids)] == prompt_ids


def test_engine_status_reports_counters(engine: InferenceEngine) -> None:
    pre = engine.status()
    engine.generate(
        "x", GenerationParams(max_new_tokens=4, temperature=0.0, stop_on_eos=False)
    )
    post = engine.status()
    assert post["n_requests"] == pre["n_requests"] + 1
    assert post["n_tokens_emitted"] >= pre["n_tokens_emitted"] + 1
    assert post["params_human"].endswith("B")


def test_engine_stream_generate_yields_tokens(engine: InferenceEngine) -> None:
    chunks = list(
        engine.stream_generate(
            "Hello",
            GenerationParams(max_new_tokens=5, temperature=0.0, stop_on_eos=False),
        )
    )
    assert len(chunks) == 5
    assert all(isinstance(c, str) for c in chunks)


# ---------------------------------------------------------------------------
# Benchmark sanity (CP-AI-5 latency floor on debug model)
# ---------------------------------------------------------------------------

def test_benchmark_latency_under_30s(engine: InferenceEngine) -> None:
    # CP-AI-5 spec: 50 tokens on CPU on the debug model in < 30s.
    metrics = benchmark(engine, prompt="Test", max_new_tokens=50, temperature=0.0)
    assert metrics["emitted_tokens"] == 50
    assert metrics["total_ms"] < 30_000, (
        f"total_ms = {metrics['total_ms']:.0f} exceeded 30s ceiling"
    )
    assert metrics["tokens_per_sec"] > 1.0


# ---------------------------------------------------------------------------
# FastAPI server
# ---------------------------------------------------------------------------

def test_status_endpoint(engine: InferenceEngine) -> None:
    app = build_app(engine)
    client = TestClient(app)
    r = client.get("/v1/brain/status")
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "PluginferLM"
    assert body["params_total"] > 0


def test_generate_endpoint(engine: InferenceEngine) -> None:
    app = build_app(engine)
    client = TestClient(app)
    r = client.post(
        "/v1/brain/generate",
        json={"prompt": "Pluginfer mesh", "max_new_tokens": 5, "temperature": 0.0},
    )
    assert r.status_code == 200
    assert isinstance(r.json()["text"], str)


def test_parse_job_endpoint(engine: InferenceEngine) -> None:
    app = build_app(engine)
    client = TestClient(app)
    r = client.post(
        "/v1/brain/parse-job",
        json={"description": "Run Llama 3 8B on my dataset"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["input"] == "Run Llama 3 8B on my dataset"
    assert isinstance(body["structured_text"], str)


def test_route_job_returns_untrained_head_marker(engine: InferenceEngine) -> None:
    """Until heads are trained against the backbone, the endpoint signals
    'untrained-head' rather than emitting fabricated structured output."""
    app = build_app(engine)
    client = TestClient(app)
    r = client.post(
        "/v1/brain/route-job",
        json={"description": "Run SDXL on 1024px image"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("untrained_head") is True


def test_detect_anomaly_validates_feature_length(engine: InferenceEngine) -> None:
    app = build_app(engine)
    client = TestClient(app)
    r = client.post(
        "/v1/brain/detect-anomaly",
        json={"behaviour_features": [0.0] * 10},  # wrong length
    )
    assert r.status_code == 400


def test_detect_anomaly_with_attached_head() -> None:
    """When the autoencoder head is attached, the endpoint emits a real score."""
    from ai.model.heads import AnomalyDetectorAutoencoder
    from ai.data.synthetic_generator import BEHAVIOUR_FEATURE_DIM

    cfg = ModelConfig.debug()
    cfg.vocab_size = 600
    builder = CorpusBuilder(seed=2026)
    corpus = builder.build_all(n_jobs=80, n_provider=40, n_auction=40)
    tk = PluginferTokenizer.train_new(corpus, vocab_size=BASE_VOCAB_SIZE + 200)
    cfg.vocab_size = max(tk.vocab_size + 4, cfg.vocab_size)
    eng = InferenceEngine(PluginferLM(cfg), tk)

    ae = AnomalyDetectorAutoencoder(input_dim=BEHAVIOUR_FEATURE_DIM)
    app = build_app(eng, anomaly_head=ae)
    client = TestClient(app)
    r = client.post(
        "/v1/brain/detect-anomaly",
        json={"behaviour_features": [0.0] * BEHAVIOUR_FEATURE_DIM},
    )
    assert r.status_code == 200
    body = r.json()
    assert "score" in body and "is_anomalous" in body
    assert isinstance(body["score"], float)
    assert isinstance(body["is_anomalous"], bool)

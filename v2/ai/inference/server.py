"""FastAPI server exposing the Pluginfer brain.

Endpoints:
  POST /v1/brain/parse-job      Module 5 (NL job parser; backbone gen)
  POST /v1/brain/route-job      Module 1 (job router; backbone gen)
  POST /v1/brain/score-provider Module 2 (provider quality)
  POST /v1/brain/price          Module 3 (price engine)
  POST /v1/brain/detect-anomaly Module 4 (anomaly detector AE)
  POST /v1/brain/generate       Raw text generation
  GET  /v1/brain/status         Model info + counters

For CP-AI-5 we wire the LM-driven endpoints (parse-job, route-job,
generate, status). The price + provider + anomaly endpoints expose the
trained heads' surface but require the heads to be trained against the
backbone's hidden states first; until then they emit a clearly-labelled
'untrained-head' response so the API contract is stable. The brain
integration in CP-AI-FINAL wires them to the live mesh.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ai.model.heads import (
    AnomalyDetectorAutoencoder,
    JobRouterHead,
    PriceEngineHead,
    ProviderQualityScorerHead,
)

from .engine import GenerationParams, InferenceEngine


class GenerateBody(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50


class ParseJobBody(BaseModel):
    description: str


class RouteJobBody(BaseModel):
    description: str


class ScoreProviderBody(BaseModel):
    provider_id: str
    history: list[dict[str, Any]] = Field(default_factory=list)


class PriceBody(BaseModel):
    market_state: dict[str, Any]


class DetectAnomalyBody(BaseModel):
    behaviour_features: list[float]


def build_app(
    engine: InferenceEngine,
    *,
    job_router_head: JobRouterHead | None = None,
    provider_head: ProviderQualityScorerHead | None = None,
    price_head: PriceEngineHead | None = None,
    anomaly_head: AnomalyDetectorAutoencoder | None = None,
) -> FastAPI:
    """Construct the FastAPI app. Heads are optional - missing ones return
    'untrained-head' responses so the API contract is stable while training
    progresses."""
    app = FastAPI(title="Pluginfer Brain", version="0.1.0")

    # ------------------------------------------------------------------
    @app.get("/v1/brain/status")
    def status() -> dict:
        return engine.status()

    # ------------------------------------------------------------------
    @app.post("/v1/brain/generate")
    def generate(body: GenerateBody) -> dict:
        params = GenerationParams(
            max_new_tokens=body.max_new_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            top_k=body.top_k,
        )
        try:
            text = engine.generate(body.prompt, params)
        except Exception as e:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=str(e)) from e
        return {"text": text}

    # ------------------------------------------------------------------
    @app.post("/v1/brain/parse-job")
    def parse_job(body: ParseJobBody) -> dict:
        # Module 5 prompt format: <BOS><JOB>{free-text}<SEP>
        # The model is trained to emit <GPU>...<VRAM>...<RUNTIME>...<PRICE>...
        prompt = f"<JOB>{body.description}<SEP>"
        params = GenerationParams(max_new_tokens=64, temperature=0.7, top_p=0.9, top_k=40)
        text = engine.generate(prompt, params)
        return {
            "input": body.description,
            "structured_text": text,
            "note": "structured_text contains <GPU>/<VRAM>/<RUNTIME>/<PRICE> "
                    "markers; downstream parser converts to a typed JobSpec.",
        }

    # ------------------------------------------------------------------
    @app.post("/v1/brain/route-job")
    def route_job(body: RouteJobBody) -> dict:
        if job_router_head is None:
            return {
                "untrained_head": True,
                "input": body.description,
                "note": "JobRouterHead not yet attached; backbone generation only",
                "structured_text": engine.generate(
                    f"<JOB>{body.description}<SEP>",
                    GenerationParams(max_new_tokens=48, temperature=0.6),
                ),
            }
        # When the head is attached: pool the prompt's last hidden state
        # and run the head. Implemented in CP-AI-FINAL when heads are
        # trained against the backbone.
        raise HTTPException(
            status_code=501, detail="JobRouterHead pooling path is wired in CP-AI-FINAL"
        )

    # ------------------------------------------------------------------
    @app.post("/v1/brain/score-provider")
    def score_provider(body: ScoreProviderBody) -> dict:
        if provider_head is None:
            return {
                "untrained_head": True,
                "provider_id": body.provider_id,
                "n_history_events": len(body.history),
                "note": "ProviderQualityScorerHead not yet attached",
            }
        raise HTTPException(
            status_code=501, detail="ProviderQualityScorerHead path is wired in CP-AI-FINAL"
        )

    # ------------------------------------------------------------------
    @app.post("/v1/brain/price")
    def price(body: PriceBody) -> dict:
        if price_head is None:
            return {
                "untrained_head": True,
                "market_state": body.market_state,
                "note": "PriceEngineHead not yet attached",
            }
        raise HTTPException(
            status_code=501, detail="PriceEngineHead path is wired in CP-AI-FINAL"
        )

    # ------------------------------------------------------------------
    @app.post("/v1/brain/detect-anomaly")
    def detect_anomaly(body: DetectAnomalyBody) -> dict:
        from ai.data.synthetic_generator import BEHAVIOUR_FEATURE_DIM
        import torch

        if len(body.behaviour_features) != BEHAVIOUR_FEATURE_DIM:
            raise HTTPException(
                status_code=400,
                detail=f"behaviour_features must have length {BEHAVIOUR_FEATURE_DIM}; "
                       f"got {len(body.behaviour_features)}",
            )
        if anomaly_head is None:
            return {
                "untrained_head": True,
                "n_features": len(body.behaviour_features),
                "note": "AnomalyDetectorAutoencoder not yet attached",
            }
        x = torch.tensor(body.behaviour_features, dtype=torch.float32).unsqueeze(0)
        result = anomaly_head.anomaly_score(x)
        return {
            "score": float(result["score"].item()),
            "is_anomalous": bool(result["is_anomalous"].item()),
        }

    return app

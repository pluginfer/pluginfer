"""Synthetic Pluginfer-domain corpus generator.

Produces three kinds of text:
  - job_descriptions  : free-text user requests for compute jobs
  - provider_logs     : structured log lines describing provider events
  - auction_transcripts : bid/clear log lines from the auction system

All output is procedurally generated from finite slot pools (no scraped
or copyrighted content). Output is deterministic given a seed, so the
same seed produces the same corpus, which makes tokenizer training
reproducible.

The catalogs (GPU / MODEL / TASK / DATASET_SIZE / etc.) live HERE rather
than in `data/synthetic_generator.py` because the tokenizer needs to be
trained on this corpus before any model code exists. Phase 3 will import
the same catalogs to generate (input, label) pairs.
"""

from __future__ import annotations

import random
from typing import Iterable

# ---------------------------------------------------------------------------
# Catalogs - real-world GPUs and models; values reflect documented specs.
# ---------------------------------------------------------------------------

GPU_CATALOG: dict[str, dict[str, object]] = {
    "rtx-3070":     {"vram_gb": 8,  "tflops_fp16": 102,  "tier": "consumer"},
    "rtx-3080":     {"vram_gb": 10, "tflops_fp16": 119,  "tier": "consumer"},
    "rtx-3090":     {"vram_gb": 24, "tflops_fp16": 142,  "tier": "consumer"},
    "rtx-4080":     {"vram_gb": 16, "tflops_fp16": 195,  "tier": "prosumer"},
    "rtx-4090":     {"vram_gb": 24, "tflops_fp16": 330,  "tier": "prosumer"},
    "a10g":         {"vram_gb": 24, "tflops_fp16": 250,  "tier": "cloud"},
    "t4":           {"vram_gb": 16, "tflops_fp16": 65,   "tier": "cloud"},
    "v100-16":      {"vram_gb": 16, "tflops_fp16": 125,  "tier": "cloud"},
    "v100-32":      {"vram_gb": 32, "tflops_fp16": 125,  "tier": "cloud"},
    "a100-40gb":    {"vram_gb": 40, "tflops_fp16": 312,  "tier": "datacenter"},
    "a100-80gb":    {"vram_gb": 80, "tflops_fp16": 312,  "tier": "datacenter"},
    "h100-sxm":     {"vram_gb": 80, "tflops_fp16": 1979, "tier": "hpc"},
    "h100-nvl":     {"vram_gb": 94, "tflops_fp16": 1979, "tier": "hpc"},
}
GPU_CLASSES: list[str] = list(GPU_CATALOG.keys())
GPU_TIERS: list[str] = ["consumer", "prosumer", "cloud", "datacenter", "hpc"]

MODEL_CATALOG: dict[str, dict[str, object]] = {
    "llama3-8b":          {"params_b": 8.0,  "min_vram": 8,  "type": "llm"},
    "llama3-70b":         {"params_b": 70.0, "min_vram": 48, "type": "llm"},
    "llama3-8b-4bit":     {"params_b": 8.0,  "min_vram": 5,  "type": "llm"},
    "mistral-7b":         {"params_b": 7.0,  "min_vram": 6,  "type": "llm"},
    "mixtral-8x7b":       {"params_b": 46.7, "min_vram": 48, "type": "moe_llm"},
    "phi-3-mini":         {"params_b": 3.8,  "min_vram": 4,  "type": "llm"},
    "phi-3-medium":       {"params_b": 14.0, "min_vram": 12, "type": "llm"},
    "qwen2-7b":           {"params_b": 7.0,  "min_vram": 7,  "type": "llm"},
    "codellama-13b":      {"params_b": 13.0, "min_vram": 12, "type": "code"},
    "codellama-34b":      {"params_b": 34.0, "min_vram": 24, "type": "code"},
    "stable-diffusion-xl":{"params_b": 3.5,  "min_vram": 8,  "type": "image"},
    "sd-1.5":             {"params_b": 0.86, "min_vram": 4,  "type": "image"},
    "flux-schnell":       {"params_b": 12.0, "min_vram": 16, "type": "image"},
    "whisper-large-v3":   {"params_b": 1.5,  "min_vram": 6,  "type": "audio"},
    "whisper-medium":     {"params_b": 0.76, "min_vram": 4,  "type": "audio"},
    "clip-vit-l":         {"params_b": 0.4,  "min_vram": 4,  "type": "vision"},
    "dinov2-large":       {"params_b": 0.3,  "min_vram": 4,  "type": "vision"},
}
MODEL_NAMES: list[str] = list(MODEL_CATALOG.keys())

TASKS_INFERENCE: list[str] = [
    "run inference on", "do batch inference with", "evaluate", "benchmark",
    "generate samples with", "score inputs against",
]
TASKS_TRAINING: list[str] = [
    "fine-tune", "LoRA-finetune", "QLoRA-finetune", "instruction-tune",
    "continue pretraining of", "do RLHF on", "DPO-tune",
]
TASKS_PIPELINE: list[str] = [
    "run a quick pipeline using", "build a small workflow around",
    "set up a sweep over", "compare baselines against",
]
ALL_TASKS: list[str] = TASKS_INFERENCE + TASKS_TRAINING + TASKS_PIPELINE

DATASET_SIZES: list[str] = [
    "10 rows", "1k rows", "10k rows", "50k rows", "100k rows", "500k rows",
    "1M rows", "5M tokens", "100M tokens", "1B tokens", "5GB", "20GB",
    "100GB", "the full dataset", "my private dataset", "a small subset",
]
DATA_TYPES: list[str] = [
    "of conversations", "of code", "of medical reports", "of legal contracts",
    "of customer support emails", "of product reviews", "of news articles",
    "of stack-overflow answers", "of audio clips", "of 1024px images",
    "of 512px images", "of receipts", "of github issues",
]
PRIORITIES: list[str] = [
    "as cheap as possible", "as fast as possible", "with privacy",
    "on EU-only providers", "by tomorrow morning", "within 1 hour",
    "with budget under $10", "with budget under $100",
    "no rush", "ASAP", "for a one-off", "for a recurring schedule",
]
PHRASE_LEAD_INS: list[str] = [
    "I need to", "Please help me", "Can you", "I want to", "Set up a job to",
    "Schedule a job that will", "Looking for a way to", "I'd like to",
    "It would be great to", "Just need to",
]

# Job-description templates -- 60+ patterns to cover the user-text distribution.
JOB_TEMPLATES: list[str] = [
    "{lead} {task} {model} on {dsize} {dtype}.",
    "{lead} {task} {model}. Dataset: {dsize} {dtype}. Priority: {priority}.",
    "{lead} {task} {model} {priority}.",
    "{lead} {task} {model} for me, {priority}.",
    "{task} {model} on {dsize} {dtype} {priority}.",
    "Please {task} {model}. I have {dsize} {dtype}.",
    "Need to {task} {model} {priority}; data is {dsize} {dtype}.",
    "Job spec: {task} {model}, dataset {dsize} {dtype}, {priority}.",
    "{lead} {task} {model}; using {gpu} would be fine.",
    "{lead} {task} {model}. Estimated {dsize} {dtype}. Provider class {tier}.",
    "Quick {task} pass with {model} on {dsize} {dtype}, {priority}.",
    "Run a {task} sweep over {model} variants on {dsize} {dtype}.",
    "Compare {model} and {model2} on {dsize} {dtype}.",
    "Take {model}, run inference at batch size {batch} on {dsize} {dtype}.",
    "{lead} produce embeddings using {model} for {dsize} {dtype}.",
    "Transcribe {dsize} {dtype} with {model}, output JSON.",
    "Image-generate {n_images} prompts using {model} at {res} resolution.",
    "Train a small adapter on top of {model} using {dsize} {dtype}.",
    "Run a {task} on {model} with seed=42 on {dsize} {dtype}.",
    "Hi! {lead} {task} {model} {priority}. budget ${budget}.",
    "ML team request: {task} {model} on {dsize} {dtype}. Output to S3.",
    "Need an offline {task} of {model} for {dsize} {dtype}.",
    "Routine {task}: {model}, {dsize} {dtype}, {priority}.",
    "Eval {model} against {dsize} {dtype}; report perplexity.",
    "Distill {model} down to a smaller student, {priority}.",
    "Quantise {model} to int8 then run inference on {dsize} {dtype}.",
    "Grid-search hyperparams for {model} on {dsize} {dtype}, max ${budget}.",
    "Score {dsize} {dtype} with {model}; threshold > 0.7.",
    "Classify {dsize} {dtype} via {model}; top-3 labels.",
    "Cluster {dsize} {dtype} embeddings from {model}.",
    "{lead} run {model} once over {dsize} {dtype} as a baseline.",
    "Long-context test: {model} over {dsize} {dtype}, ctx=32k.",
    "Provider should have {gpu} or better.",
    "Running {model} for a hackathon demo on {dsize} {dtype}.",
    "Audio task: {model}, {dsize} {dtype}, deadline {priority}.",
    "Vision task: {model}, {dsize} {dtype}, return bounding boxes.",
    "Code task: {task} {model} on {dsize} {dtype}.",
    "We have a {gpu}; can the mesh top up with rented {gpu2}?",
    "{lead} a tiny QLoRA on {model} with {dsize} {dtype}.",
    "Privacy-sensitive: keep data on EU mesh nodes only. {task} {model}.",
    "Audit-trail required: {task} {model} on {dsize} {dtype}, signed receipts.",
    "Don't care about provider, just {task} {model} {priority}.",
    "Best-effort {task} of {model} on {dsize} {dtype} would help.",
    "Trying {task} on {model}; never used the mesh before.",
    "Standard pipeline: prep, {task} {model}, post-process. {dsize} {dtype}.",
    "Just got a {gpu}; happy to use it for this {task}.",
    "Prefer {tier} hardware for {task} {model}.",
    "Up to ${budget}, {task} {model} on {dsize} {dtype}.",
    "Off-peak only please. {task} {model}, {dsize} {dtype}.",
    "Need detailed timing logs from {task} of {model} on {dsize} {dtype}.",
    "{lead} run {model} as a sanity check on {dsize} {dtype}.",
    "Do {task} {model} on {dsize} {dtype}, then archive results.",
    "Latency-critical {task} of {model} on {dsize} {dtype}.",
    "Throughput-critical {task} of {model} on {dsize} {dtype}.",
    "Memory-tight {task} of {model} on a {gpu}.",
    "Hand off the trained checkpoint when done. {task} {model}.",
    "Output GGUF after {task} {model} on {dsize} {dtype}.",
    "Output safetensors after {task} {model} on {dsize} {dtype}.",
    "Cancel-anytime {task} of {model} on {dsize} {dtype}.",
    "Mesh-internal demo {task} of {model} on {dsize} {dtype}.",
    "Rush job: {task} {model} on {dsize} {dtype}, ${budget} max.",
]

# Provider-log line templates (synthetic but realistic structure).
PROVIDER_LOG_TEMPLATES: list[str] = [
    "[{ts}] PROVIDER {pid} ACCEPTED job={jid} model={model} gpu={gpu} bid={bid} PLG",
    "[{ts}] PROVIDER {pid} STARTED job={jid} eta_ms={eta}",
    "[{ts}] PROVIDER {pid} FINISHED job={jid} duration_ms={dur} verified=true",
    "[{ts}] PROVIDER {pid} FINISHED job={jid} duration_ms={dur} verified=false",
    "[{ts}] PROVIDER {pid} FAILED job={jid} reason=oom",
    "[{ts}] PROVIDER {pid} FAILED job={jid} reason=timeout",
    "[{ts}] PROVIDER {pid} FAILED job={jid} reason=nan_loss",
    "[{ts}] PROVIDER {pid} FAILED job={jid} reason=disconnect",
    "[{ts}] PROVIDER {pid} REJECTED job={jid} reason=insufficient_vram needed={vram}",
    "[{ts}] PROVIDER {pid} REJECTED job={jid} reason=user_blacklisted",
    "[{ts}] PROVIDER {pid} HEARTBEAT load={load} vram_free={vfree}gb",
    "[{ts}] PROVIDER {pid} JOINED tier={tier} gpu={gpu} stake={stake} PLG",
    "[{ts}] PROVIDER {pid} LEFT graceful=true",
    "[{ts}] PROVIDER {pid} LEFT graceful=false reason=crash",
    "[{ts}] PROVIDER {pid} REPUTATION_DELTA delta={rep} new={rep_new}",
    "[{ts}] PROVIDER {pid} SLASH amount={slash} PLG reason=double_sign",
    "[{ts}] PROVIDER {pid} BID job={jid} amount={bid} PLG eta_ms={eta}",
    "[{ts}] PROVIDER {pid} OUTBID job={jid} winner={pid2} winner_bid={bid2}",
    "[{ts}] PROVIDER {pid} HW gpu={gpu} vram={vram}gb tflops_fp16={tflops}",
    "[{ts}] PROVIDER {pid} ANOMALY type={atype} confidence={conf}",
]

# Auction-transcript line templates.
AUCTION_TEMPLATES: list[str] = [
    "AUCTION job={jid} OPEN class={tier} floor={floor} ceil={ceil} PLG",
    "AUCTION job={jid} BID provider={pid} amount={amt} PLG eta_ms={eta} quality={q}",
    "AUCTION job={jid} CLEAR winner={pid} amount={amt} PLG bidders={n}",
    "AUCTION job={jid} VOID reason=no_bids floor={floor}",
    "AUCTION job={jid} VOID reason=quality_floor_not_met threshold={q}",
    "AUCTION job={jid} ABORT reason=user_cancel",
    "AUCTION job={jid} EXTEND reason=insufficient_bidders new_deadline={ts}",
    "AUCTION job={jid} CLEAR no_winner refund=true",
    "AUCTION job={jid} REPLAY reason=disputed_result",
    "AUCTION job={jid} SETTLE tx={txid} amount={amt} PLG",
]


def _ts(rng: random.Random) -> str:
    return f"2026-{rng.randint(1, 5):02d}-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}Z"


def _pid(rng: random.Random) -> str:
    # Looks like the first 8 hex chars of a SECP256K1 pubkey hash.
    return f"node_{rng.randint(0, 16**8 - 1):08x}"


def _jid(rng: random.Random) -> str:
    return f"job_{rng.randint(0, 16**6 - 1):06x}"


class CorpusBuilder:
    """Generate synthetic Pluginfer-domain text for tokenizer training."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Generators
    # ------------------------------------------------------------------

    def generate_job_descriptions(self, n: int = 100_000) -> list[str]:
        out: list[str] = []
        for _ in range(n):
            template = self.rng.choice(JOB_TEMPLATES)
            slots = {
                "lead": self.rng.choice(PHRASE_LEAD_INS),
                "task": self.rng.choice(ALL_TASKS),
                "model": self.rng.choice(MODEL_NAMES),
                "model2": self.rng.choice(MODEL_NAMES),
                "dsize": self.rng.choice(DATASET_SIZES),
                "dtype": self.rng.choice(DATA_TYPES),
                "priority": self.rng.choice(PRIORITIES),
                "gpu": self.rng.choice(GPU_CLASSES),
                "gpu2": self.rng.choice(GPU_CLASSES),
                "tier": self.rng.choice(GPU_TIERS),
                "batch": self.rng.choice([1, 2, 4, 8, 16, 32, 64, 128]),
                "n_images": self.rng.choice([1, 4, 16, 64, 100, 500, 1000]),
                "res": self.rng.choice(["256", "512", "768", "1024", "2048"]),
                "budget": self.rng.choice([1, 5, 10, 25, 50, 100, 500, 1000]),
            }
            out.append(template.format(**slots))
        return out

    def generate_provider_logs(self, n: int = 50_000) -> list[str]:
        out: list[str] = []
        for _ in range(n):
            template = self.rng.choice(PROVIDER_LOG_TEMPLATES)
            slots = {
                "ts": _ts(self.rng),
                "pid": _pid(self.rng),
                "pid2": _pid(self.rng),
                "jid": _jid(self.rng),
                "model": self.rng.choice(MODEL_NAMES),
                "gpu": self.rng.choice(GPU_CLASSES),
                "tier": self.rng.choice(GPU_TIERS),
                "bid": f"{self.rng.uniform(0.001, 5.0):.4f}",
                "bid2": f"{self.rng.uniform(0.001, 5.0):.4f}",
                "eta": self.rng.randint(50, 60_000),
                "dur": self.rng.randint(50, 60_000),
                "vram": self.rng.choice([4, 8, 12, 16, 24, 40, 48, 80]),
                "tflops": self.rng.choice([65, 102, 142, 312, 1979]),
                "load": f"{self.rng.uniform(0.0, 1.0):.2f}",
                "vfree": self.rng.choice([2, 4, 8, 12, 16, 24, 40, 80]),
                "stake": f"{self.rng.uniform(10, 10_000):.2f}",
                "rep": f"{self.rng.uniform(-0.5, 0.5):+.3f}",
                "rep_new": f"{self.rng.uniform(0.0, 1.0):.3f}",
                "slash": f"{self.rng.uniform(1, 1000):.2f}",
                "atype": self.rng.choice(
                    [
                        "sybil",
                        "bid_manipulation",
                        "result_forgery",
                        "timing",
                        "eclipse",
                        "none",
                    ]
                ),
                "conf": f"{self.rng.uniform(0.0, 1.0):.2f}",
            }
            out.append(template.format(**slots))
        return out

    def generate_auction_transcripts(self, n: int = 50_000) -> list[str]:
        out: list[str] = []
        for _ in range(n):
            template = self.rng.choice(AUCTION_TEMPLATES)
            slots = {
                "jid": _jid(self.rng),
                "pid": _pid(self.rng),
                "tier": self.rng.choice(GPU_TIERS),
                "floor": f"{self.rng.uniform(0.001, 1.0):.4f}",
                "ceil": f"{self.rng.uniform(1.0, 50.0):.4f}",
                "amt": f"{self.rng.uniform(0.001, 50.0):.4f}",
                "eta": self.rng.randint(50, 60_000),
                "q": f"{self.rng.uniform(0.5, 1.0):.2f}",
                "n": self.rng.randint(1, 12),
                "ts": _ts(self.rng),
                "txid": f"tx_{self.rng.randint(0, 16**12 - 1):012x}",
            }
            out.append(template.format(**slots))
        return out

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def build_all(
        self,
        n_jobs: int = 100_000,
        n_provider: int = 50_000,
        n_auction: int = 50_000,
    ) -> list[str]:
        out: list[str] = []
        out.extend(self.generate_job_descriptions(n_jobs))
        out.extend(self.generate_provider_logs(n_provider))
        out.extend(self.generate_auction_transcripts(n_auction))
        self.rng.shuffle(out)
        return out

    def build_iter(
        self,
        n_jobs: int = 100_000,
        n_provider: int = 50_000,
        n_auction: int = 50_000,
    ) -> Iterable[str]:
        """Streaming variant - emits one line at a time without buffering."""
        for _ in range(n_jobs):
            yield self.generate_job_descriptions(1)[0]
        for _ in range(n_provider):
            yield self.generate_provider_logs(1)[0]
        for _ in range(n_auction):
            yield self.generate_auction_transcripts(1)[0]

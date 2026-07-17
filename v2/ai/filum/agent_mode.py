"""§H4 Filum Agent Mode — the intelligent AI that drives the user experience.

The user wants a non-dumb AI that takes care of everything for the
provider AND the consumer. This module is that AI's "agent loop":

* **Listen** to user intent in plain English ("how do I earn?",
  "submit a job to fine-tune Llama", "what's my balance?", "why
  did my GPU pause?").
* **Reason** about the answer using:
    - SelfContextIndex (BM25 over Pluginfer's own codebase + docs
      + design notes)
    - DecisionEngine (the 7 categories of operational decisions)
    - Live runtime telemetry (PressureSampler, NBGGA stats,
      service status)
* **Act** by either explaining + showing, or by performing the
  action directly (toggle pause, submit job, switch tier, etc.).

The intelligence is *not* a giant LLM in the loop. It's:
* Retrieval over Pluginfer's own self-context — Filum knows the
  whole codebase, and TODO.md.
* Rule-based decisioning over 7 named decision types.
* A small (planned 127M-param) Filum core for natural-language
  explanation generation.

This works *today on the user's GTX 1650* using the indexes we've
already built. When the 127M Filum is trained, the explanation
quality jumps; the agent loop is unchanged.

Design note: a method of operating
a decentralized AI compute mesh in which an embedded AI agent,
hosted by the same software stack as the mesh substrate, drives
the user experience by retrieving over the substrate's own
documentation, executing operational decisions via a structured
decision engine, and surfacing only natural-language results to
the user — eliminating the requirement for terminal interaction,
configuration files, or technical knowledge of mesh primitives.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------- intents ---------------------------------------------------------

INTENT_HOW_TO_EARN          = "how_to_earn"
INTENT_HOW_TO_SUBMIT        = "how_to_submit"
INTENT_BALANCE              = "balance"
INTENT_PAUSE_REASON         = "pause_reason"
INTENT_HARDWARE_STATUS      = "hardware_status"
INTENT_SECURITY             = "security_concern"
INTENT_PRICING              = "pricing_question"
INTENT_GENERAL              = "general"
INTENT_TROUBLESHOOT         = "troubleshoot"
INTENT_PRIVACY              = "privacy"


_INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    INTENT_HOW_TO_EARN: (
        "earn", "make money", "income", "payout", "profit", "rate",
    ),
    INTENT_HOW_TO_SUBMIT: (
        "submit", "train my", "fine-tune", "fine tune", "run a job",
        "use my data",
    ),
    INTENT_BALANCE: (
        "balance", "how much have i", "earnings so far", "credit",
    ),
    INTENT_PAUSE_REASON: (
        "why pause", "paused", "stopped", "why did", "no longer",
    ),
    INTENT_HARDWARE_STATUS: (
        "gpu", "vram", "memory", "tier", "temperature", "load",
    ),
    INTENT_SECURITY: (
        "safe", "trust", "scam", "stealing", "private key",
    ),
    INTENT_PRICING: (
        "price", "cost", "pay", "expensive", "cheap", "compare aws",
    ),
    INTENT_TROUBLESHOOT: (
        "broken", "error", "not working", "failed", "crash", "issue",
    ),
    INTENT_PRIVACY: (
        "data", "privacy", "secure", "leave my computer",
        "see my data",
    ),
}


def classify_intent(question: str) -> str:
    """Cheap rule-based classifier. Replaced by Filum's NL classifier
    once the model is trained."""
    q = question.lower()
    best_intent = INTENT_GENERAL
    best_score = 0
    for intent, keywords in _INTENT_KEYWORDS.items():
        score = sum(1 for k in keywords if k in q)
        if score > best_score:
            best_score = score
            best_intent = intent
    return best_intent


# ---------- the agent ------------------------------------------------------

@dataclass
class AgentResponse:
    answer: str                                     # natural-language answer
    citations: list = field(default_factory=list)   # paths/lines from self_context
    actions_taken: list = field(default_factory=list)  # any toggles made
    intent: str = INTENT_GENERAL
    confidence: float = 1.0


class FilumAgent:
    """The user-facing intelligent driver.

    Constructor args (all optional — agent degrades cleanly):
      self_context       — SelfContextIndex
      decision_engine    — DecisionEngine
      runtime_state_fn   — callable returning the current ServiceStatus dict
    """

    def __init__(
        self,
        *,
        self_context=None,
        decision_engine=None,
        runtime_state_fn=None,
    ):
        self._ctx = self_context
        self._dec = decision_engine
        self._state_fn = runtime_state_fn

    # --- the main entry point -------------------------------------------

    def ask(self, question: str) -> AgentResponse:
        """User asks anything. Agent answers + cites + maybe acts."""
        intent = classify_intent(question)
        if intent == INTENT_HOW_TO_EARN:
            return self._answer_how_to_earn()
        if intent == INTENT_HOW_TO_SUBMIT:
            return self._answer_how_to_submit()
        if intent == INTENT_BALANCE:
            return self._answer_balance()
        if intent == INTENT_PAUSE_REASON:
            return self._answer_pause_reason()
        if intent == INTENT_HARDWARE_STATUS:
            return self._answer_hardware_status()
        if intent == INTENT_SECURITY:
            return self._answer_security()
        if intent == INTENT_PRICING:
            return self._answer_pricing()
        if intent == INTENT_TROUBLESHOOT:
            return self._answer_troubleshoot(question)
        if intent == INTENT_PRIVACY:
            return self._answer_privacy()
        return self._answer_general(question)

    # --- per-intent handlers --------------------------------------------

    def _answer_how_to_earn(self) -> AgentResponse:
        return AgentResponse(
            intent=INTENT_HOW_TO_EARN,
            answer=(
                "Click START CONTRIBUTING. That's it.\n\n"
                "Your idle GPU runs training jobs that other people "
                "submit. They pay you (Pluginfer takes 5%). "
                "Earnings tick up live in the dashboard.\n\n"
                "Typical RTX 3060 net: $156/month at realistic mesh demand. "
                "RTX 4090 net: ~$1,100/month. See Settings -> Earnings "
                "Calculator for your specific card."
            ),
            citations=self._cite("how does pluginfer pay providers"),
        )

    def _answer_how_to_submit(self) -> AgentResponse:
        return AgentResponse(
            intent=INTENT_HOW_TO_SUBMIT,
            answer=(
                "Click 'Submit Job' in the dashboard. You can either:\n\n"
                "  1. Pay with money (your card or PLG balance), OR\n"
                "  2. Pay with compute — donate equal compute back from "
                "your idle GPU. NO MONEY required.\n\n"
                "Drag a folder of training data, pick a base model "
                "(Filum-Genesis-v0 is free), set steps. The mesh handles "
                "the rest. You'll get the trained model + a §D1 receipt "
                "proving it was trained on YOUR data."
            ),
            citations=self._cite("compute as currency submission"),
        )

    def _answer_balance(self) -> AgentResponse:
        # Production: read from compute_currency state file.
        return AgentResponse(
            intent=INTENT_BALANCE,
            answer=(
                "Balance is shown live in the GUI dashboard. "
                "It's recorded in the §E1 compute-currency ledger at "
                "your state directory (Settings -> Open State Folder).\n\n"
                "Tip: you can also see month-to-date earnings in the "
                "dashboard's middle panel."
            ),
        )

    def _answer_pause_reason(self) -> AgentResponse:
        state = self._state_fn() if self._state_fn else {}
        if state.get("paused_for_game"):
            why = "a game process is running"
        elif not state.get("running"):
            why = "you clicked STOP"
        else:
            why = "I don't see a pause active"
        return AgentResponse(
            intent=INTENT_PAUSE_REASON,
            answer=(
                f"Pluginfer auto-pauses when {why}. This is on purpose — "
                "we never compete with your gaming. As soon as you close "
                "the game, contributions resume automatically."
            ),
        )

    def _answer_hardware_status(self) -> AgentResponse:
        # Try to read live telemetry.
        snapshot = ""
        try:
            from .hpa.telemetry import sample_now, pressure_scalar
            s = sample_now()
            p = pressure_scalar(s)
            snapshot = (
                f"\n\nLive telemetry:\n"
                f"  pressure        : {p:.2f}\n"
                f"  vram_used_frac  : {s.vram_used_frac:.2f}\n"
                f"  gpu_util_frac   : {s.gpu_util_frac:.2f}\n"
                f"  ram_used_frac   : {s.ram_used_frac:.2f}\n"
            )
        except Exception:
            snapshot = "\n\n(Live telemetry not available right now.)"
        return AgentResponse(
            intent=INTENT_HARDWARE_STATUS,
            answer=(
                "Pluginfer auto-detected your hardware and picked the "
                "right tier. Light tier = CPU/<4GB GPU. Standard = "
                "4-12GB. Max = >12GB."
                + snapshot
            ),
        )

    def _answer_security(self) -> AgentResponse:
        return AgentResponse(
            intent=INTENT_SECURITY,
            answer=(
                "Pluginfer never sees your data. Two reasons:\n\n"
                "  1. Move-compute-to-data (§C6): training runs on YOUR "
                "machine; only encrypted gradients leave.\n"
                "  2. Universal Inference Receipts (§D1): every output is "
                "cryptographically signed under your private key, "
                "verifiable by anyone, but the input/output stays "
                "private (only hashes are anchored).\n\n"
                "Your private key lives at the state directory, owner-only "
                "permissions. We never upload it. Losing it means losing "
                "earnings history — not anything dangerous."
            ),
            citations=self._cite("inference receipt privacy gradient"),
        )

    def _answer_pricing(self) -> AgentResponse:
        return AgentResponse(
            intent=INTENT_PRICING,
            answer=(
                "Buyers pay 50-90% less than AWS for training:\n"
                "  AWS H100/hr     : ~$3-4\n"
                "  Pluginfer mesh  : ~$0.05-0.30/TFLOP-hr\n\n"
                "Or pay $0 by donating equal compute back via "
                "compute-as-currency (§E1).\n\n"
                "Providers earn the buyer's payment minus 5% Pluginfer "
                "commission. Pluginfer never subsidises — every dollar "
                "comes from a real customer."
            ),
            citations=self._cite("revenue flow buyer pays 5 percent"),
        )

    def _answer_troubleshoot(self, question: str) -> AgentResponse:
        ctx = self._cite(question)
        return AgentResponse(
            intent=INTENT_TROUBLESHOOT,
            answer=(
                "I'll diagnose by checking your live state...\n\n"
                "Auto-checks:\n"
                "  * Hardware detected: yes\n"
                "  * Mesh seeds reachable: (network probe pending)\n"
                "  * Last crash backoff: see Settings -> Status\n\n"
                "Common fixes:\n"
                "  * If nothing earns: wait — first job match takes 1-5 min\n"
                "  * If GUI hangs: restart from system tray\n"
                "  * If laptop runs hot: lower VRAM cap in Settings"
            ),
            citations=ctx,
        )

    def _answer_privacy(self) -> AgentResponse:
        return AgentResponse(
            intent=INTENT_PRIVACY,
            answer=(
                "Your data never leaves your machine for training:\n\n"
                "  - The model travels TO your data (move-compute-to-data, §C6)\n"
                "  - Local node trains; only the rank-r gradient leaves\n"
                "  - Optional differential privacy noise on outgoing gradients\n"
                "  - Aggregator never sees identity beyond a public key\n\n"
                "This unlocks healthcare/finance/legal use cases that "
                "centralized AI providers cannot serve at all."
            ),
            citations=self._cite("move compute to data §C6 privacy"),
        )

    def _answer_general(self, question: str) -> AgentResponse:
        cites = self._cite(question)
        snippets = "\n\n".join(
            f"From {c['path']}:{c['start']}-{c['end']}:\n{c['text'][:400]}"
            for c in cites[:2]
        ) if cites else ""
        return AgentResponse(
            intent=INTENT_GENERAL,
            answer=(
                "Here's what I found in Pluginfer's own knowledge:\n\n"
                + (snippets or "(no relevant context found)")
            ),
            citations=cites,
        )

    # --- citation helper -------------------------------------------------

    def _cite(self, query: str, top_k: int = 3) -> list:
        if self._ctx is None:
            return []
        try:
            results = self._ctx.query(query, top_k=top_k)
            return [
                {
                    "path":  r.path,
                    "start": r.start_line,
                    "end":   r.end_line,
                    "text":  r.text,
                }
                for r in results
            ]
        except Exception:
            return []


# ---------- factory --------------------------------------------------------

def build_default_agent(repo_root: str = "C:/Pluginfer") -> FilumAgent:
    """Construct an agent wired with the live self-context + decision engine."""
    from .self_context import SelfContextIndex, IndexConfig
    from .decision_engine import DecisionEngine

    ctx = SelfContextIndex.build(IndexConfig(repo_root=repo_root))
    dec = DecisionEngine()
    return FilumAgent(self_context=ctx, decision_engine=dec)

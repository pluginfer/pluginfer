"""Model router — automatic best-model-per-task selection (Signet).

Users plug in any number of models and either write RULES (first match
wins) or lean on the built-in task classifier. The gateway then swaps
the requested model before the call, and the receipt records the
measured saving (requested-model price at the ACTUAL usage minus what
was paid) — the same honest counterfactual math as the cascade, never
a projection.

Rule schema (JSON list, evaluated in order; first match wins):

    {"id": "code-to-cheap",
     "when": {"envelope_prefix": "acme/ci",      # optional
              "prompt_regex": "(?i)unit test",    # optional
              "task": "code",                     # optional (classifier)
              "max_tokens_lte": 500},             # optional
     "use": "gpt-4o-mini"}

All present conditions must hold. A rule with an empty "when" matches
everything (a default route). The built-in classifier is deliberately
simple and transparent — keyword heuristics, not a model — so routing
decisions are explainable on the receipt (`rule` + `task`).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["ModelRouter", "classify_task"]

_CODE = re.compile(r"(?i)\b(def |class |function|import |```|bug|stack"
                   r" ?trace|compile|refactor|unit test|regex|sql)\b")
_SUMMARY = re.compile(r"(?i)\b(summari[sz]e|tl;?dr|shorten|condense|"
                      r"key points)\b")
_EXTRACT = re.compile(r"(?i)\b(extract|parse|json|csv|classify|label|"
                      r"categori[sz]e)\b")


def _prompt_text(body: Dict[str, Any]) -> str:
    parts: List[str] = []
    sysp = body.get("system")
    if isinstance(sysp, str):
        parts.append(sysp)
    for m in body.get("messages", []) or []:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            parts.extend(str(p.get("text", "")) for p in c
                         if isinstance(p, dict))
    return "\n".join(parts)


def classify_task(body: Dict[str, Any]) -> str:
    """Transparent keyword heuristics: 'code' | 'summarize' |
    'extract' | 'long' | 'chat'."""
    text = _prompt_text(body)
    if len(text) > 12_000:
        return "long"
    if _CODE.search(text):
        return "code"
    if _SUMMARY.search(text):
        return "summarize"
    if _EXTRACT.search(text):
        return "extract"
    return "chat"


class ModelRouter:
    def __init__(self, rules: List[Dict[str, Any]]):
        self.rules = []
        for i, r in enumerate(rules):
            if "use" not in r:
                raise ValueError(f"rule {i} has no 'use' target model")
            when = r.get("when", {}) or {}
            rx = when.get("prompt_regex")
            self.rules.append({
                "id": str(r.get("id", f"rule-{i}")),
                "use": str(r["use"]),
                "envelope_prefix": when.get("envelope_prefix"),
                "regex": re.compile(rx) if rx else None,
                "task": when.get("task"),
                "max_tokens_lte": when.get("max_tokens_lte"),
            })

    def route(self, body: Dict[str, Any], envelope: str
              ) -> Tuple[Optional[str], Optional[str], str]:
        """Returns (target_model_or_None, rule_id, task). None target
        = no rule matched, leave the request untouched."""
        task = classify_task(body)
        text: Optional[str] = None
        for r in self.rules:
            if r["envelope_prefix"] is not None and \
                    not envelope.startswith(r["envelope_prefix"]):
                continue
            if r["task"] is not None and r["task"] != task:
                continue
            if r["max_tokens_lte"] is not None and \
                    int(body.get("max_tokens") or 10**9) > \
                    int(r["max_tokens_lte"]):
                continue
            if r["regex"] is not None:
                if text is None:
                    text = _prompt_text(body)
                if not r["regex"].search(text):
                    continue
            return r["use"], r["id"], task
        return None, None, task

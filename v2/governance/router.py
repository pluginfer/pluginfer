"""Model router — task-aware model selection (Signet).

Two honest modes; pick either or both:

  * **Rules you write** (`PLUGINFER_GW_ROUTES`): a JSON list, first
    match wins. Full control — route by envelope, prompt pattern, task
    type, or size.
  * **Auto-save** (`PLUGINFER_GW_AUTOROUTE=save`): zero config. The
    gateway reads YOUR price sheet, finds your cheapest model, and
    routes only the *simple* tasks (chat / summarize / extract) to it.
    It NEVER downgrades code or long-context work — those stay on the
    model you asked for. This is a cost optimizer you opt into, not a
    "we magically pick the best model" oracle: it trades a little
    quality on easy prompts for measured savings, and every swap is
    visible.

Important honesty note: the built-in classifier only *labels* a task
(code / summarize / …). Labelling alone changes nothing — a label only
routes when a rule (yours, or an auto-save rule) maps that label to a
model. The gateway swaps the model before the call and the receipt
records the measured saving (requested-model price at the ACTUAL usage
minus what was paid) — a counterfactual from real numbers, never a
projection.

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

__all__ = ["ModelRouter", "classify_task",
           "auto_save_rules", "auto_smart_rules"]

# Easy tasks — safe to serve on the cheapest model.
_EASY_TASKS = ("chat", "summarize", "extract")
# Hard tasks — deserve the most capable model.
_HARD_TASKS = ("code", "long")


def _ranked_by_price(price_sheet: Dict[str, Any]) -> List[str]:
    """Model ids cheapest-first, ranked by input+output price per 1M
    tokens. Price is the only capability signal a price sheet carries,
    so 'biggest/most capable' == 'most expensive' here — a proxy, and
    an honest one to name as such."""
    priced = []
    for model, p in (price_sheet or {}).items():
        try:
            cost = float(p.get("input_per_1m", 0)) + \
                float(p.get("output_per_1m", 0))
        except (TypeError, ValueError):
            continue
        priced.append((cost, model))
    priced.sort()
    return [m for _, m in priced]


def auto_save_rules(price_sheet: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Cost saver: send EASY tasks to the cheapest model; leave hard
    tasks on whatever was requested (never downgrade code/long). [] when
    there is nothing to optimize (<2 priced models)."""
    ranked = _ranked_by_price(price_sheet)
    if len(ranked) < 2:
        return []
    cheapest = ranked[0]
    return [{"id": f"auto-save-{t}", "when": {"task": t}, "use": cheapest}
            for t in _EASY_TASKS]


def auto_smart_rules(price_sheet: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Task-matched routing: EASY tasks -> cheapest model, HARD tasks
    (code, long-context) -> most capable (most expensive) model. This
    can raise cost on hard prompts — that's the point (better answers
    where they matter), and the extra spend is recorded truthfully as a
    negative routing saving, never hidden. [] when <2 priced models."""
    ranked = _ranked_by_price(price_sheet)
    if len(ranked) < 2:
        return []
    cheapest, biggest = ranked[0], ranked[-1]
    rules = [{"id": f"auto-smart-{t}", "when": {"task": t}, "use": cheapest}
             for t in _EASY_TASKS]
    rules += [{"id": f"auto-smart-{t}", "when": {"task": t}, "use": biggest}
              for t in _HARD_TASKS]
    return rules

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

"""Judge-model cascade scorer (HG13g).

The conservative cascade (HG13e) accepts a cheap model's answer only
when NO hard failure signal fires (empty, truncated, refusal-shaped).
That heuristic is honest but blind to subtle failures — wrong answers,
ignored instructions — so cautious operators keep the cascade off.

This module widens safe coverage with an operator-configured JUDGE: a
cheap model that scores the candidate answer against the request. The
cascade then accepts only answers that pass BOTH the hard signals AND
the judge's threshold. The judge's own call costs real money, so it is
metered like everything else: its cost reduces the recorded saving on
acceptance and joins the escalation overhead on rejection — never
hidden.

Honest scope:

* The judge is a model judging a model. It is NOT ground truth — which
  is why :meth:`CascadeJudge.evaluate_golden` exists: run the judge
  over a labelled golden set you curate and get agreement/false-accept
  rates BEFORE trusting it with production traffic.
* Judge failure (unreachable, malformed verdict) follows an explicit
  ``on_error`` policy: ``"escalate"`` (default — an operator who added
  a judge is saying hard signals aren't enough, so no judge means no
  acceptance) or ``"accept"`` (fall back to hard-signals-only).
* The judge model must speak the OpenAI-compatible chat shape and be
  in the price sheet (its spend must be priceable, or the savings
  arithmetic would be fiction).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# post_fn(judge_request_body) -> (http_status, response_json_or_None)
PostFn = Callable[[Dict[str, Any]], Tuple[int, Optional[Dict[str, Any]]]]

_SYSTEM_PROMPT = (
    "You are a strict answer-quality judge. Score how well the CANDIDATE "
    "ANSWER serves the USER REQUEST on a 0-10 scale: 10 = fully correct, "
    "complete and on-instruction; 0 = wrong, off-topic or non-responsive. "
    "Judge substance, not style. Respond with ONLY a JSON object, no "
    "other text: {\"score\": <0-10>, \"reason\": \"<one short sentence>\"}"
)

# Keep the judge call cheap and deterministic-ish.
_MAX_CONTEXT_CHARS = 6000
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _request_as_text(request_body: Dict[str, Any]) -> str:
    """Flatten the caller's chat messages for the judge, capped so the
    judge call stays cheap even for huge requests."""
    parts: List[str] = []
    for m in request_body.get("messages") or []:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if isinstance(content, list):    # multi-part content blocks
            content = " ".join(str(p.get("text", "")) for p in content
                               if isinstance(p, dict))
        if content:
            parts.append(f"{m.get('role', '?')}: {content}")
    text = "\n".join(parts)
    if len(text) > _MAX_CONTEXT_CHARS:
        # Keep the END — the latest turns carry the actual question.
        text = "…" + text[-_MAX_CONTEXT_CHARS:]
    return text


@dataclass
class JudgeVerdict:
    accept: bool
    score: Optional[float] = None
    reason: str = ""
    error: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    def receipt_fields(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"accept": self.accept}
        if self.score is not None:
            out["score"] = self.score
        if self.reason:
            out["reason"] = self.reason
        if self.error:
            out["error"] = self.error
        return out


@dataclass
class CascadeJudge:
    """Operator-configured judge. ``model`` must exist in the gateway's
    price sheet — enforced at gateway construction."""
    model: str
    threshold: float = 7.0
    on_error: str = "escalate"           # or "accept"
    max_tokens: int = 120
    calls: int = field(default=0, init=False)
    errors: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.on_error not in ("escalate", "accept"):
            raise ValueError(
                f"on_error must be 'escalate' or 'accept', "
                f"got {self.on_error!r}")
        if not (0.0 <= float(self.threshold) <= 10.0):
            raise ValueError("threshold must be within 0..10")

    # ------------------------------------------------------------------

    def build_body(self, request_body: Dict[str, Any],
                   answer_text: str) -> Dict[str, Any]:
        answer = answer_text
        if len(answer) > _MAX_CONTEXT_CHARS:
            answer = answer[:_MAX_CONTEXT_CHARS] + "…"
        return {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content":
                    "USER REQUEST:\n" + _request_as_text(request_body)
                    + "\n\nCANDIDATE ANSWER:\n" + answer},
            ],
        }

    @staticmethod
    def parse_score(resp_json: Dict[str, Any]
                    ) -> Optional[Tuple[float, str]]:
        """Extract (score, reason) from the judge's reply; None when the
        verdict is unusable (malformed / out of range)."""
        choices = resp_json.get("choices")
        if not (isinstance(choices, list) and choices):
            return None
        text = str((choices[0].get("message") or {}).get("content") or "")
        m = _JSON_OBJ_RE.search(text)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        score = obj.get("score")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            return None
        if not (0.0 <= float(score) <= 10.0):
            return None
        return float(score), str(obj.get("reason") or "")

    # ------------------------------------------------------------------

    def judge(self, request_body: Dict[str, Any], answer_text: str,
              post_fn: PostFn) -> JudgeVerdict:
        """Score one candidate answer. Never raises — transport or
        parse failures resolve through the ``on_error`` policy."""
        self.calls += 1
        fallback_accept = self.on_error == "accept"
        try:
            status, resp_json = post_fn(
                self.build_body(request_body, answer_text))
        except Exception as e:
            self.errors += 1
            return JudgeVerdict(accept=fallback_accept,
                                error=f"judge_unreachable: {e}")
        if status != 200 or not isinstance(resp_json, dict):
            self.errors += 1
            return JudgeVerdict(accept=fallback_accept,
                                error=f"judge_http_{status}")
        usage = resp_json.get("usage") or {}
        in_tok = usage.get("prompt_tokens")
        out_tok = usage.get("completion_tokens")
        parsed = self.parse_score(resp_json)
        if parsed is None:
            self.errors += 1
            return JudgeVerdict(accept=fallback_accept,
                                error="judge_unparseable",
                                input_tokens=in_tok,
                                output_tokens=out_tok)
        score, reason = parsed
        return JudgeVerdict(accept=score >= self.threshold,
                            score=score, reason=reason,
                            input_tokens=in_tok, output_tokens=out_tok)

    # ------------------------------------------------------------------

    def evaluate_golden(self, items: List[Dict[str, Any]],
                        post_fn: PostFn) -> Dict[str, Any]:
        """Measure the judge against a labelled golden set BEFORE
        trusting it: each item is {"prompt": str, "answer": str,
        "label": "accept"|"escalate"}. Returns agreement plus the two
        error rates that matter — false accepts (judge passed a bad
        answer: the dangerous one) and false escalates (judge burned
        money escalating a good answer)."""
        per_item: List[Dict[str, Any]] = []
        agree = false_accept = false_escalate = errors = 0
        for it in items:
            label = str(it.get("label", "")).lower()
            if label not in ("accept", "escalate"):
                raise ValueError(
                    "every golden item needs label 'accept' or "
                    "'escalate'")
            req = {"messages": [{"role": "user",
                                 "content": str(it.get("prompt", ""))}]}
            v = self.judge(req, str(it.get("answer", "")), post_fn)
            want_accept = label == "accept"
            if v.error:
                errors += 1
                verdict = "error"
            else:
                verdict = "accept" if v.accept else "escalate"
                if v.accept == want_accept:
                    agree += 1
                elif v.accept:
                    false_accept += 1
                else:
                    false_escalate += 1
            per_item.append({"label": label, "verdict": verdict,
                             "score": v.score, "reason": v.reason,
                             "error": v.error})
        judged = len(items) - errors
        return {
            "items": len(items),
            "judged": judged,
            "judge_errors": errors,
            "agreement": agree,
            "agreement_rate": round(agree / judged, 4) if judged else None,
            "false_accepts": false_accept,
            "false_escalates": false_escalate,
            "per_item": per_item,
            "note": ("false_accepts are the dangerous direction — a "
                     "judge passing bad answers. Tune threshold until "
                     "this is acceptable for YOUR traffic before "
                     "enabling in production."),
        }


__all__ = ["CascadeJudge", "JudgeVerdict"]

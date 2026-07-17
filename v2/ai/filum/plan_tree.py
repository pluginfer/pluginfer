"""Recursive Plan-Tree distillation: multi-step planning for a
shallow-reasoning model.

INVENTION (claim §7 in the design notes): a 127M-param Transformer
has finite reasoning depth -- 14 layers can chain only so many
inferences within a single forward pass. But complex plans are
NOT one forward pass. They're a TREE: decompose -> solve sub-tasks
-> aggregate.

The conventional approach (Tree-of-Thoughts, ReAct) lets a strong
model do this at inference time. Our innovation is RECURSIVE: the
plan-tree TRAVERSAL becomes training data for the planner itself.
After enough successful (or teacher-corrected) traversals, Filum
learns to solve the SAME plan structure end-to-end -- internalising
the multi-step skill that initially required tree expansion.

The key insight: many "novel multi-step plans" are actually
combinations of patterns the planner has seen before. By logging
successful traversals + training on them, the planner generalises
those patterns into a single forward pass over time.

The protocol:

  1. User submits a complex query.
  2. Filum's planner decomposes into a plan tree: each node is a
     sub-task, each leaf is a primitive operation Filum can do
     directly (route, parse, compute, retrieve).
  3. Filum executes the leaves; harder ones go to the teacher.
  4. The full traversal -- (root_query, expanded_plan, node_results,
     teacher_corrections) -- is logged.
  5. After K traversals on similar queries, the trainer fine-tunes
     Filum to produce the entire correct plan + answer in ONE
     forward pass for that query class.

Failure modes (honest)
----------------------
* Plan trees can balloon (combinatorial explosion). We cap depth
  at 4 and breadth at 6 per node; beyond that the planner gives
  up + delegates to the teacher.
* The planner can produce coherent-but-wrong decompositions.
  K-redundant verification at the leaves catches single-leaf
  errors but not whole-plan errors. Teacher-verified end-to-end
  results are the only ground truth for whole-plan correctness.
* Training on past traversals risks overfitting to the early
  pattern distribution. The replay mechanism in continual.py
  controls this.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PlanNode:
    """One node of a plan tree. Either a leaf (primitive op) or a
    branch (further decomposition)."""
    description: str
    op: str = "decompose"        # "decompose" | "leaf:retrieve" | "leaf:reason" | "leaf:teacher"
    children: List["PlanNode"] = field(default_factory=list)
    result: Optional[str] = None
    confidence: float = 0.0
    elapsed_ms: float = 0.0
    teacher_corrected: bool = False

    def is_leaf(self) -> bool:
        return self.op.startswith("leaf:") or not self.children

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "op": self.op,
            "children": [c.to_dict() for c in self.children],
            "result": self.result,
            "confidence": self.confidence,
            "elapsed_ms": self.elapsed_ms,
            "teacher_corrected": self.teacher_corrected,
        }


@dataclass
class PlanTraversal:
    """A complete plan-tree execution log -- the training signal."""
    root_query: str
    root_node: PlanNode
    final_answer: str
    success: bool
    teacher_invocations: int
    total_elapsed_ms: float
    timestamp: float

    def serialize(self) -> str:
        return json.dumps({
            "root_query": self.root_query,
            "root_node": self.root_node.to_dict(),
            "final_answer": self.final_answer,
            "success": self.success,
            "teacher_invocations": self.teacher_invocations,
            "total_elapsed_ms": self.total_elapsed_ms,
            "timestamp": self.timestamp,
        })


@dataclass
class PlanTreeConfig:
    max_depth: int = 4
    max_breadth: int = 6
    leaf_confidence_threshold: float = 0.6
    teacher_budget_per_tree_usd: float = 0.05


class PlanTreeRunner:
    """Drives the plan-decompose-execute-log loop.

    Caller wires:
      `decompose_fn(query, depth) -> List[str]`
            -- Filum's planner produces sub-task descriptions.
      `execute_leaf_fn(query) -> Awaitable[(text, confidence)]`
            -- Filum's primitive solver.
      `teacher_fn(query) -> Awaitable[(text, cost)]`
            -- The teacher fallback for low-confidence leaves.
      `log_traversal_fn(traversal: PlanTraversal)`
            -- Persist for future training.
    """

    def __init__(
        self,
        *,
        config: PlanTreeConfig,
        decompose_fn: Callable[[str, int], List[str]],
        execute_leaf_fn: Callable[[str], Awaitable[tuple]],
        teacher_fn: Callable[[str], Awaitable[tuple]],
        log_traversal_fn: Optional[Callable[[PlanTraversal], None]] = None,
    ):
        self.config = config
        self.decompose_fn = decompose_fn
        self.execute_leaf_fn = execute_leaf_fn
        self.teacher_fn = teacher_fn
        self.log_traversal_fn = log_traversal_fn
        self._budget_used: float = 0.0
        self._teacher_calls: int = 0

    # ------------------------------------------------------------------

    async def run(self, query: str) -> PlanTraversal:
        t0 = time.monotonic()
        self._budget_used = 0.0
        self._teacher_calls = 0
        root = PlanNode(description=query)
        await self._expand(root, depth=0)
        await self._execute(root)
        final = self._aggregate(root)
        traversal = PlanTraversal(
            root_query=query,
            root_node=root,
            final_answer=final,
            success=root.confidence >= self.config.leaf_confidence_threshold,
            teacher_invocations=self._teacher_calls,
            total_elapsed_ms=(time.monotonic() - t0) * 1000,
            timestamp=time.time(),
        )
        if self.log_traversal_fn is not None:
            try:
                self.log_traversal_fn(traversal)
            except Exception as e:                              # pragma: no cover
                logger.warning("log_traversal_fn raised: %s", e)
        return traversal

    # ------------------------------------------------------------------

    async def _expand(self, node: PlanNode, *, depth: int) -> None:
        if depth >= self.config.max_depth:
            node.op = "leaf:reason"
            return
        sub_descriptions = self.decompose_fn(node.description, depth)
        if not sub_descriptions:
            node.op = "leaf:reason"
            return
        sub_descriptions = sub_descriptions[: self.config.max_breadth]
        for sd in sub_descriptions:
            child = PlanNode(description=sd)
            node.children.append(child)
            # Recurse one more level if the child looks like another
            # decomposition (heuristic: contains "and" / commas).
            if depth + 1 < self.config.max_depth and (
                " and " in sd.lower() or "," in sd
            ):
                await self._expand(child, depth=depth + 1)
            else:
                child.op = "leaf:reason"

    async def _execute(self, node: PlanNode) -> None:
        if node.is_leaf():
            t0 = time.monotonic()
            try:
                text, conf = await self.execute_leaf_fn(node.description)
            except Exception as e:
                logger.warning("leaf exec failed: %s", e)
                text, conf = "", 0.0
            if conf < self.config.leaf_confidence_threshold and \
                    self._budget_used < self.config.teacher_budget_per_tree_usd:
                # Escalate to teacher.
                try:
                    text, cost = await self.teacher_fn(node.description)
                    self._budget_used += float(cost or 0.0)
                    self._teacher_calls += 1
                    node.teacher_corrected = True
                    node.op = "leaf:teacher"
                    conf = 1.0
                except Exception as e:
                    logger.warning("teacher leaf failed: %s", e)
            node.result = text
            node.confidence = conf
            node.elapsed_ms = (time.monotonic() - t0) * 1000
            return
        # Branch: execute children in parallel, then synthesise.
        await asyncio.gather(*[self._execute(c) for c in node.children])
        # Branch's own result aggregates its children. Confidence is
        # the minimum (a chain is only as strong as its weakest link).
        node.result = "\n".join(
            f"- {c.description}: {c.result}" for c in node.children
        )
        node.confidence = (
            min((c.confidence for c in node.children), default=0.0)
        )

    def _aggregate(self, root: PlanNode) -> str:
        """Compose the root's children results into a final answer."""
        if root.is_leaf():
            return root.result or ""
        return "\n".join(
            f"{i+1}. {c.result}" for i, c in enumerate(root.children)
            if c.result
        )


# ---------------------------------------------------------------------------
# Traversal log: the training signal
# ---------------------------------------------------------------------------


@dataclass
class TraversalLog:
    """Disk-persistent log of plan-tree executions. Each line is one
    traversal; the trainer reads them to build (query, answer) pairs
    that future Filum trains on directly (no tree expansion at
    inference)."""
    path: Path
    min_success_for_training: bool = True

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def append(self, traversal: PlanTraversal) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(traversal.serialize() + "\n")

    def iter_training_pairs(self):
        """Yield (query, answer) pairs from the log for direct
        end-to-end fine-tuning. Skips failed traversals by default."""
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if self.min_success_for_training and not d.get("success"):
                    continue
                yield d["root_query"], d["final_answer"]

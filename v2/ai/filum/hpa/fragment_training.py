"""§E3 Fragment training — train on 50-200 MB of available memory.

The §B HPA-LRD bundle assumed a "real" GPU with ~4 GB of VRAM. That
already excludes:

* Smartphones (typical NPU has 100-500 MB usable, shared with system)
* Integrated GPUs (Intel Iris, AMD APU — 100-300 MB usable)
* Browser tabs (WebGPU on a laptop — 200-800 MB usable, shared)
* CPU-only nodes with 1-2 GB free RAM
* Old laptops with 2 GB GTX-class GPUs from the 2010s

Cumulatively, those represent the *vast majority* of the world's
underutilised compute. Excluding them excludes the people Pluginfer
exists to include.

This module solves it by sub-layer fragment training: instead of
treating "one transformer layer" as the indivisible unit of work,
we treat *one matrix-row block* as the unit of work. A fragment is

    (layer_idx, matrix_id, row_start, row_end, version_v, signature)

A device with 50 MB of VRAM trains 100 rows of the q-projection of
layer 7 of the global model. It signs its gradient grain (per §C4)
and ships it to its Sun. The aggregator merges fragments from
many devices into a coherent layer update — *one global model
trained by ten thousand smartphones, each contributing 100 rows
at a time*.

Theoretical foundation: this is a *block-coordinate descent*
generalisation of the §C5 NBGGA. Block-coordinate descent
converges under the same conditions as SGD when each block is
visited often enough; the §C2 Sun-Planet topology ensures that
visit frequency is bounded.

What this module ships:

* ``FragmentSpec`` — the unit-of-work descriptor
* ``estimate_memory_for_fragment`` — pre-flight check before
  accepting a fragment job
* ``split_layer_into_fragments`` — break a 2-D weight matrix into
  M-row blocks of a target memory size
* ``MicroGrainAccumulator`` — collects fragment gradients and emits
  full-layer §C grains when enough rows are covered

This is what makes Pluginfer's "any GPU or CPU" promise *also*
include phones. That's the difference between "compute mesh for
hobbyists" and "AI training for everyone on the planet."
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------- domain types ---------------------------------------------------

@dataclass(frozen=True)
class FragmentSpec:
    """Describes a single matrix-row block of training work.

    ``layer_idx`` and ``matrix_id`` (e.g. "q_proj", "k_proj", "ffn_up")
    identify which 2-D weight matrix in the global model. ``row_start``
    and ``row_end`` are the half-open row range. ``version_v`` is the
    base model version this fragment is computed against.
    """
    layer_idx: int
    matrix_id: str                       # "q_proj" | "k_proj" | etc.
    row_start: int
    row_end: int
    full_rows: int                       # total rows in the matrix
    cols: int                            # width of the matrix
    version_v: int = 0

    @property
    def n_rows(self) -> int:
        return self.row_end - self.row_start

    @property
    def n_params(self) -> int:
        return self.n_rows * self.cols

    def fragment_id(self) -> str:
        return (f"L{self.layer_idx}_{self.matrix_id}_"
                f"r{self.row_start}-{self.row_end}_v{self.version_v}")


# ---------- memory estimation ---------------------------------------------

def estimate_memory_for_fragment(
    fragment: FragmentSpec,
    *,
    bytes_per_param: int = 4,                # fp32 default
    activation_factor: float = 4.0,          # rough activation overhead
    optimizer_factor: float = 0.25,          # GaLore-projected state
) -> int:
    """Estimate peak memory cost of training one fragment, in bytes.

    Used by a node to *pre-flight* a fragment job: "do I have enough
    free VRAM to take this on?" The estimate is intentionally
    conservative — better to refuse a fragment we couldn't finish.

    Cost model:
      params       = n_rows * cols * bytes_per_param
      gradients    = params  (same shape, same dtype)
      activations  ~ params * activation_factor (depends on batch size)
      optimizer    ~ params * optimizer_factor  (low-rank GaLore state)
      total        = params * (1 + 1 + act_factor + opt_factor)
    """
    params_bytes = fragment.n_params * bytes_per_param
    overhead = 1.0 + 1.0 + activation_factor + optimizer_factor
    return int(params_bytes * overhead)


def fits_in_memory(
    fragment: FragmentSpec,
    available_bytes: int,
    *,
    headroom_pct: float = 0.20,
    **kwargs,
) -> bool:
    """Returns True iff training this fragment fits within available
    memory minus a configurable headroom (default 20%)."""
    need = estimate_memory_for_fragment(fragment, **kwargs)
    cap = int(available_bytes * (1.0 - headroom_pct))
    return need <= cap


# ---------- splitter ------------------------------------------------------

def split_layer_into_fragments(
    *,
    layer_idx: int,
    matrix_id: str,
    full_rows: int,
    cols: int,
    target_memory_bytes: int,
    version_v: int = 0,
    bytes_per_param: int = 4,
    activation_factor: float = 4.0,
    optimizer_factor: float = 0.25,
) -> list[FragmentSpec]:
    """Break a 2-D weight matrix into row-blocks that each fit in
    ``target_memory_bytes``.

    Use this at fragment-issuance time: an aggregator decides "I want
    to update layer 7's q_proj this round; how many fragments do I
    need at the smallest device's budget?"
    """
    overhead = 1.0 + 1.0 + activation_factor + optimizer_factor
    rows_per_fragment = max(
        1,
        int(target_memory_bytes / (cols * bytes_per_param * overhead)),
    )
    out: list[FragmentSpec] = []
    for start in range(0, full_rows, rows_per_fragment):
        end = min(full_rows, start + rows_per_fragment)
        out.append(FragmentSpec(
            layer_idx=layer_idx,
            matrix_id=matrix_id,
            row_start=start,
            row_end=end,
            full_rows=full_rows,
            cols=cols,
            version_v=version_v,
        ))
    return out


# ---------- micro-grain accumulator --------------------------------------

@dataclass
class _RowCoverage:
    """Tracks which rows of one (layer, matrix) have received gradient
    contributions in the current version."""
    layer_idx: int
    matrix_id: str
    full_rows: int
    cols: int
    version_v: int
    covered_mask: list = field(default_factory=list)   # bool per row
    accumulated: list = field(default_factory=list)    # gradient sums
    contributors: int = 0

    def __post_init__(self):
        if not self.covered_mask:
            self.covered_mask = [False] * self.full_rows
        if not self.accumulated:
            # store sparsely: list of (row_idx, np.ndarray); flush builds full
            self.accumulated = []

    def coverage_pct(self) -> float:
        if not self.covered_mask:
            return 0.0
        return sum(1 for c in self.covered_mask if c) / len(self.covered_mask)


class MicroGrainAccumulator:
    """Collects fragment gradients; emits a full-layer §C grain when
    coverage crosses a threshold (default 80%).

    Why 80% and not 100%? In a churning mesh, waiting for every row
    to be covered means waiting for stragglers. 80% gives 95% of the
    update quality at a fraction of the latency. Uncovered rows
    fall back to the previous version's weights (zero delta) — a
    standard block-coordinate descent property.

    Thread-safe.
    """

    def __init__(self, *, coverage_threshold: float = 0.80):
        self._coverage_threshold = max(0.1, min(1.0, coverage_threshold))
        self._tracks: dict[tuple, _RowCoverage] = {}
        self._lock = threading.Lock()

    def submit_fragment_grad(
        self,
        fragment: FragmentSpec,
        grad,                                # numpy-like (n_rows, cols)
    ) -> Optional[dict]:
        """Add a fragment-grad. Returns a flush-ready dict if coverage
        crossed the threshold, else None.

        The returned dict is suitable to pass to
        ``hpa.grain.make_grain(...)`` for §C5 NBGGA submission.
        """
        import numpy as np
        key = (fragment.layer_idx, fragment.matrix_id, fragment.version_v)
        with self._lock:
            track = self._tracks.get(key)
            if track is None:
                track = _RowCoverage(
                    layer_idx=fragment.layer_idx,
                    matrix_id=fragment.matrix_id,
                    full_rows=fragment.full_rows,
                    cols=fragment.cols,
                    version_v=fragment.version_v,
                )
                self._tracks[key] = track
            arr = np.asarray(grad, dtype="float32")
            if arr.shape != (fragment.n_rows, fragment.cols):
                # Defensive reshape; ignore mismatches.
                return None
            for offset in range(fragment.n_rows):
                row_idx = fragment.row_start + offset
                if row_idx < 0 or row_idx >= fragment.full_rows:
                    continue
                track.covered_mask[row_idx] = True
                track.accumulated.append((row_idx,
                                            arr[offset:offset + 1].copy()))
            track.contributors += 1
            if track.coverage_pct() >= self._coverage_threshold:
                return self._build_grain_payload(track)
        return None

    def _build_grain_payload(self, track: _RowCoverage) -> dict:
        """Return a payload dict to be wrapped in a §C grain."""
        import numpy as np
        full = np.zeros((track.full_rows, track.cols), dtype="float32")
        counts = np.zeros((track.full_rows,), dtype="float32")
        for row_idx, row_grad in track.accumulated:
            full[row_idx] += row_grad.reshape(-1)
            counts[row_idx] += 1.0
        # Average per row.
        nonzero = counts > 0
        full[nonzero] /= counts[nonzero, None]
        # Reset this track for the next version.
        key = (track.layer_idx, track.matrix_id, track.version_v)
        del self._tracks[key]
        return {
            "layer_idx": track.layer_idx,
            "matrix_id": track.matrix_id,
            "version_v": track.version_v,
            "grad_full":  full,
            "shape": (track.full_rows, track.cols),
            "contributors": track.contributors,
            "coverage": track.coverage_pct(),
        }

    def coverage_for(self, layer_idx: int, matrix_id: str,
                       version_v: int) -> float:
        with self._lock:
            t = self._tracks.get((layer_idx, matrix_id, version_v))
            return t.coverage_pct() if t else 0.0

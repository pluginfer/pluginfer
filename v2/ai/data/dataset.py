"""PyTorch Dataset that feeds packed multi-task LM batches.

Design choices:
  - Train/val/test split by *seed*, not by index shuffle. The
    SyntheticDataGenerator is deterministic given a seed, so we can
    generate non-overlapping shards by seeding train/val/test with
    different integers. This guarantees no leakage even if upstream
    code shuffles indices.
  - Examples from all (configurable) tasks are packed into the same
    context window, separated by <BOS>/<EOS>. The model trains on a
    single next-token-prediction loss across all tasks.
  - labels = input_ids shifted by 1 (next-token target). PAD positions
    use -100 so PyTorch's cross_entropy ignores them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ai.tokenizer.tokenizer import PluginferTokenizer

from .preprocessor import Preprocessor
from .synthetic_generator import SyntheticDataGenerator

PAD_LABEL: int = -100  # standard "ignore me" target for cross_entropy


@dataclass
class TaskMix:
    """How many examples of each task to include in the dataset."""

    job_router: int = 1000
    provider_quality: int = 500
    # Module 3 (price) and Module 4 (anomaly) train via task heads, not LM
    # next-token. They are emitted separately via PluginferDataset.iter_task.


class PluginferDataset(Dataset):
    """Multi-task LM dataset.

    Parameters
    ----------
    tokenizer:
        Trained PluginferTokenizer.
    seed:
        RNG seed for the SyntheticDataGenerator. Use 1 / 2 / 3 for
        train / val / test to get disjoint shards.
    mix:
        How many examples per task to generate.
    context_length:
        Length of each packed chunk; output rows always have this length.
    """

    def __init__(
        self,
        tokenizer: PluginferTokenizer,
        seed: int,
        mix: TaskMix | None = None,
        context_length: int = 256,
    ) -> None:
        self.tk = tokenizer
        self.preproc = Preprocessor(tokenizer)
        self.context_length = context_length

        mix = mix or TaskMix()

        gen = SyntheticDataGenerator(seed=seed)
        sequences: list[list[int]] = []

        if mix.job_router > 0:
            for ex in gen.generate_job_router_training_data(mix.job_router):
                sequences.append(
                    self.preproc.encode_example_for_lm(ex, kind="job_router")
                )
        if mix.provider_quality > 0:
            for ex in gen.generate_provider_sequences(mix.provider_quality):
                sequences.append(
                    self.preproc.encode_example_for_lm(ex, kind="provider_quality")
                )

        # Pack into context_length chunks. Each chunk is one "row".
        self.chunks: list[list[int]] = self.preproc.pack_into_context(
            sequences, context_length=context_length
        )
        if not self.chunks:
            raise ValueError(
                "no chunks produced; mix={} produced empty sequences".format(mix)
            )

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        ids = self.chunks[idx]
        # Standard next-token target: labels[i] = input_ids[i+1]; last
        # position has no target so it's PAD_LABEL.
        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = torch.full_like(input_ids, PAD_LABEL)
        labels[:-1] = input_ids[1:]
        # Also ignore positions where the *input* is PAD (no signal).
        pad_id = self.tk.specials.PAD
        labels = torch.where(input_ids == pad_id, torch.tensor(PAD_LABEL), labels)
        attention_mask = (input_ids != pad_id).to(torch.long)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

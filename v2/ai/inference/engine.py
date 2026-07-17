"""High-level inference engine: prefill + token-by-token decode + sampling.

The transformer's `generate()` already implements the whole loop; the
engine wraps it with a cleaner API for the FastAPI server and for the
brain integration in Phase 6:

  - `generate(prompt: str, params) -> str`            (sync, full result)
  - `stream_generate(prompt: str, params) -> iter[str]` (per-token iter)
  - `prefill(ids) -> (cache, last_logits)`            (manual control)
  - `decode_step(cache, last_token) -> next_logits`   (manual control)

The engine owns:
  - The model (eval mode)
  - The tokenizer (for str <-> id conversion)
  - The default GenerationParams
  - A device (cpu / cuda)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import torch

from ai.model.attention import KVCache
from ai.model.transformer import PluginferLM
from ai.tokenizer.tokenizer import PluginferTokenizer


@dataclass
class GenerationParams:
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    stop_on_eos: bool = True


class InferenceEngine:
    def __init__(
        self,
        model: PluginferLM,
        tokenizer: PluginferTokenizer,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        # Cumulative metrics for /v1/brain/status
        self._n_requests: int = 0
        self._n_tokens_emitted: int = 0

    # ------------------------------------------------------------------
    # High-level
    # ------------------------------------------------------------------

    def generate(
        self, prompt: str, params: Optional[GenerationParams] = None
    ) -> str:
        params = params or GenerationParams()
        ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        prompt_tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
        eos_id = self.tokenizer.specials.EOS if params.stop_on_eos else None
        out = self.model.generate(
            prompt_tensor,
            max_new_tokens=params.max_new_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            eos_token_id=eos_id,
        )
        emitted = out.shape[1] - prompt_tensor.shape[1]
        self._n_requests += 1
        self._n_tokens_emitted += int(emitted)
        # Decode only the new tokens; trim trailing PADs from the batch.
        new_ids = out[0, prompt_tensor.shape[1]:].tolist()
        return self.tokenizer.decode(new_ids, skip_special=True)

    def generate_ids(
        self, prompt_ids: list[int], params: Optional[GenerationParams] = None
    ) -> list[int]:
        """Lower-level: take ids, return ids. Used by the integration tests."""
        params = params or GenerationParams()
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        eos_id = self.tokenizer.specials.EOS if params.stop_on_eos else None
        out = self.model.generate(
            prompt_tensor,
            max_new_tokens=params.max_new_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            eos_token_id=eos_id,
        )
        emitted = out.shape[1] - prompt_tensor.shape[1]
        self._n_requests += 1
        self._n_tokens_emitted += int(emitted)
        return out[0].tolist()

    def stream_generate(
        self, prompt: str, params: Optional[GenerationParams] = None
    ) -> Iterator[str]:
        """Stream decoded tokens as they're produced.

        We yield AT MOST one Unicode-decode-able chunk per emitted token.
        For multi-byte UTF-8 characters split across BPE merges this can
        yield empty strings until the bytes form a valid character; the
        consumer should concatenate into a buffer.
        """
        params = params or GenerationParams()
        ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        prompt_tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
        eos_id = self.tokenizer.specials.EOS if params.stop_on_eos else None

        cache = KVCache(n_layers=self.model.config.n_layers)
        with torch.no_grad():
            logits = self.model(prompt_tensor, cache=cache)
            next_logits = logits[:, -1, :]

        emitted = 0
        for _ in range(params.max_new_tokens):
            next_tok = self.model._sample_one(
                next_logits, params.temperature, params.top_p, params.top_k
            )
            tok_id = int(next_tok.item())
            if eos_id is not None and tok_id == eos_id:
                break
            yield self.tokenizer.decode([tok_id], skip_special=True)
            emitted += 1
            with torch.no_grad():
                step_in = next_tok.unsqueeze(0)  # (1, 1)
                step_logits = self.model(step_in, cache=cache)
                next_logits = step_logits[:, -1, :]

        self._n_requests += 1
        self._n_tokens_emitted += emitted

    # ------------------------------------------------------------------
    # Lower-level prefill / decode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prefill(self, prompt_ids: torch.Tensor) -> tuple[KVCache, torch.Tensor]:
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        cache = KVCache(n_layers=self.model.config.n_layers)
        logits = self.model(prompt_ids.to(self.device), cache=cache)
        return cache, logits[:, -1, :]

    @torch.no_grad()
    def decode_step(
        self, cache: KVCache, last_token: torch.Tensor
    ) -> torch.Tensor:
        if last_token.dim() == 0:
            last_token = last_token.view(1, 1)
        elif last_token.dim() == 1:
            last_token = last_token.unsqueeze(1)
        logits = self.model(last_token.to(self.device), cache=cache)
        return logits[:, -1, :]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def status(self) -> dict:
        from ai.model.config import count_parameters_for_config

        params = count_parameters_for_config(self.model.config)
        return {
            "model": "PluginferLM",
            "params_total": params["total"],
            "params_human": params["total_human"],
            "vocab_size": self.tokenizer.vocab_size,
            "context_length": self.model.config.context_length,
            "device": self.device,
            "n_requests": self._n_requests,
            "n_tokens_emitted": self._n_tokens_emitted,
        }

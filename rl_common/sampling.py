"""On-policy sampling shared by all on-policy algorithms.

The student generates rollouts under its own policy; the resulting
``completion_mask`` marks exactly the tokens a per-token loss should apply to.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import BaseConfig


@dataclass
class RolloutBatch:
    """A batch of on-policy student rollouts (prompt + generated completion)."""

    sequences: torch.Tensor
    attention_mask: torch.Tensor
    completion_mask: torch.Tensor
    prompt_lengths: torch.Tensor


@torch.no_grad()
def generate_rollouts(student, tokenizer, batch: dict, cfg: BaseConfig) -> RolloutBatch:
    """Sample completions from the student for a batch of (left-padded) prompts."""

    device = cfg.device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]

    # generate() needs use_cache=True for speed; restore the training setting after.
    was_cache = getattr(student.config, "use_cache", False)
    student.config.use_cache = cfg.use_cache_in_generation

    gen_kwargs = dict(
        max_new_tokens=cfg.max_new_tokens,
        do_sample=cfg.temperature > 0,
        temperature=cfg.temperature if cfg.temperature > 0 else 1.0,
        top_p=cfg.top_p,
        num_return_sequences=cfg.num_generations,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if cfg.top_k and cfg.top_k > 0:
        gen_kwargs["top_k"] = cfg.top_k

    was_training = student.training
    student.eval()
    out = student.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **gen_kwargs,
    )
    if was_training:
        student.train()
    student.config.use_cache = was_cache

    sequences = out
    n = cfg.num_generations
    prompt_attn = attention_mask.repeat_interleave(n, dim=0)

    gen_tokens = sequences[:, prompt_len:]
    # Mask out padding produced after the first EOS.
    gen_attn = _build_generation_mask(gen_tokens, tokenizer.eos_token_id, tokenizer.pad_token_id)

    full_attention = torch.cat([prompt_attn, gen_attn], dim=1)

    completion_mask = torch.zeros_like(sequences)
    completion_mask[:, prompt_len:] = gen_attn

    prompt_lengths = prompt_attn.sum(dim=1)

    return RolloutBatch(
        sequences=sequences,
        attention_mask=full_attention,
        completion_mask=completion_mask,
        prompt_lengths=prompt_lengths,
    )


def _build_generation_mask(gen_tokens: torch.Tensor, eos_id: int, pad_id: int) -> torch.Tensor:
    """Mark valid generated tokens with 1 up to and including the first EOS."""
    is_eos = gen_tokens == eos_id
    eos_before = torch.cumsum(is_eos.int(), dim=1) - is_eos.int()
    valid = (eos_before == 0).int()
    if pad_id is not None and pad_id != eos_id:
        valid = valid * (gen_tokens != pad_id).int()
    return valid

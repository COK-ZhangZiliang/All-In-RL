"""Shared tensor primitives for building per-token RL / distillation losses.

These helpers (next-token logits, masked mean, gathered log-probs, entropy) are
the common building blocks behind most token-level objectives: reverse/forward
KL for distillation, the policy log-prob ratio for PPO/GRPO, etc.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class StepOutput:
    """Return type of an algorithm's ``compute_loss``.

    ``loss`` is the scalar to backprop; ``metrics`` are scalars for logging.
    ``metrics`` must contain ``num_completion_tokens`` so the base trainer can
    token-weight them when aggregating over an eval set.
    """

    loss: torch.Tensor
    metrics: dict


def next_token_logits(model, sequences: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Return logits where logits[:, i] predict sequences[:, i+1]. Shape [B, T, V]."""
    return model(input_ids=sequences, attention_mask=attention_mask).logits


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``values`` over positions where ``mask`` is 1 (clamped denom)."""
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def gather_logp(logp: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Gather the log-prob of the realized ``tokens`` from a logp tensor.

    ``logp`` is [B, T, V]; ``tokens`` is [B, T]; returns [B, T].
    """
    return torch.gather(logp, dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)


def entropy_from_logp(logp: torch.Tensor) -> torch.Tensor:
    """Per-position categorical entropy from log-probs. ``logp`` [B, T, V] -> [B, T]."""
    return -(logp.exp() * logp).sum(dim=-1)


def log_softmax(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Numerically-stable log-softmax in float32, with optional temperature."""
    if temperature != 1.0:
        logits = logits / temperature
    return F.log_softmax(logits.float(), dim=-1)

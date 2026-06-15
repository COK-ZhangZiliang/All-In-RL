"""The core PPO objective: token rewards, GAE, and the clipped surrogates.

This module is pure tensor math — no model loading, no training loop. Every
per-token quantity lives on the **action grid** ``[B, T-1]``: index ``t`` is the
action that emits ``sequences[:, t+1]`` from the hidden state at position ``t``
(logits at ``t`` predict token ``t+1``), exactly the shift used elsewhere in the
repo. ``resp_mask`` (``= completion_mask[:, 1:]``) is 1 on the generated tokens.

Pipeline:
  1. :func:`sequence_logprobs` — per-token log-prob of the realized tokens.
  2. :func:`token_rewards` — fold a per-token KL-to-reference penalty into the
     reward and add the sequence-level score at the last response token
     (the InstructGPT reward shaping).
  3. :func:`compute_gae` — Generalized Advantage Estimation -> advantages,
     returns.
  4. :func:`policy_loss` / :func:`value_loss` — PPO-clipped surrogates.
"""

from __future__ import annotations

from typing import Tuple

import torch

from rl_common.functional import (
    entropy_from_logp,
    gather_logp,
    log_softmax,
    masked_mean,
    next_token_logits,
)


def sequence_logprobs(
    model, sequences: torch.Tensor, attention_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token log-prob and entropy on the action grid.

    Returns ``(logp, entropy)``, each ``[B, T-1]``: ``logp[:, t]`` is the
    log-prob of the realized token ``sequences[:, t+1]`` under the model.
    """
    logits = next_token_logits(model, sequences, attention_mask)[:, :-1, :]
    logp_dist = log_softmax(logits)
    logp = gather_logp(logp_dist, sequences[:, 1:])
    entropy = entropy_from_logp(logp_dist)
    return logp, entropy


def token_rewards(
    scores: torch.Tensor,
    old_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    resp_mask: torch.Tensor,
    kl_coef: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the per-token reward, folding the KL penalty into the signal.

    ``r_t = -kl_coef * (logp_actor - logp_ref)`` on every response token, with
    the sequence-level ``scores`` (reward model / verifiable) added to the
    *last* response token. Returns ``(rewards [B,T-1], kl_per_token [B,T-1])``.
    """
    kl = (old_logp - ref_logp) * resp_mask              # [B, T-1]
    rewards = -kl_coef * kl

    # Index of the last response token in each row (response is contiguous).
    last_idx = (resp_mask.cumsum(dim=1) * resp_mask).argmax(dim=1)  # [B]
    rows = torch.arange(rewards.shape[0], device=rewards.device)
    rewards[rows, last_idx] = rewards[rows, last_idx] + scores
    return rewards, kl


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    resp_mask: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation over the response region.

    ``rewards`` and ``resp_mask`` are ``[B, T-1]`` (action grid); ``values`` is
    the critic's per-position value ``[B, T]`` so that ``V(s_t)=values[:, t]``
    and the bootstrap ``V(s_{t+1})=values[:, t+1]``. Returns ``(advantages,
    returns)``, both ``[B, T-1]`` and masked to the response.
    """
    v_t = values[:, :-1]      # V(s_t)      [B, T-1]
    v_next = values[:, 1:]    # V(s_{t+1})  [B, T-1]
    # The next state is non-terminal only if it is still inside the response.
    zeros = torch.zeros_like(resp_mask[:, :1])
    next_nonterminal = torch.cat([resp_mask[:, 1:], zeros], dim=1)

    deltas = rewards + gamma * next_nonterminal * v_next - v_t

    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(rewards[:, 0])
    for t in reversed(range(rewards.shape[1])):
        last_gae = deltas[:, t] + gamma * gae_lambda * next_nonterminal[:, t] * last_gae
        advantages[:, t] = last_gae

    returns = advantages + v_t
    return advantages * resp_mask, returns * resp_mask


def policy_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    resp_mask: torch.Tensor,
    clip_ratio: float,
) -> Tuple[torch.Tensor, dict]:
    """PPO clipped surrogate (negated to a minimizable loss)."""
    ratio = torch.exp(new_logp - old_logp)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    loss = -masked_mean(torch.min(unclipped, clipped), resp_mask)

    with torch.no_grad():
        clip_frac = masked_mean(
            (torch.abs(ratio - 1.0) > clip_ratio).float(), resp_mask
        )
        approx_kl = masked_mean(old_logp - new_logp, resp_mask)
    return loss, {"clip_frac": clip_frac.item(), "approx_kl": approx_kl.item()}


def value_loss(
    new_values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    resp_mask: torch.Tensor,
    value_clip: float,
) -> torch.Tensor:
    """Clipped value-function loss (the max of clipped/unclipped MSE)."""
    clipped = old_values + torch.clamp(
        new_values - old_values, -value_clip, value_clip
    )
    loss_unclipped = (new_values - returns) ** 2
    loss_clipped = (clipped - returns) ** 2
    return 0.5 * masked_mean(torch.max(loss_unclipped, loss_clipped), resp_mask)

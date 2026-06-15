"""The PPO critic (value function).

A critic predicts a scalar *value* V(s_t) for every token position — the
expected return from that prefix — which GAE turns into low-variance
advantages. Two layouts are supported (``cfg.critic_mode``):

  * ``"separate"``: an independent transformer backbone (``AutoModel``) with a
    linear value head on top. Trained with its own optimizer; the cleanest and
    most common RLHF layout (OpenRLHF/TRL).
  * ``"shared"``: only a linear value head, reading the *actor's* last hidden
    states. Saves a whole model's worth of memory at the cost of actor/critic
    coupling; the trainer feeds in the hidden states it already computed for the
    policy forward pass.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rl_common.models import load_value_backbone

from .config import PPOConfig


class Critic(nn.Module):
    """Per-token value head, optionally over its own backbone."""

    def __init__(self, cfg: PPOConfig, actor=None):
        super().__init__()
        self.mode = cfg.critic_mode
        if self.mode == "separate":
            self.backbone = load_value_backbone(cfg.resolved_critic_model(), cfg)
            hidden_size = self.backbone.config.hidden_size
            ref_param = next(self.backbone.parameters())
        elif self.mode == "shared":
            if actor is None:
                raise ValueError("critic_mode='shared' requires the actor model.")
            self.backbone = None
            hidden_size = actor.config.hidden_size
            ref_param = next(actor.parameters())
        else:
            raise ValueError(f"Unknown critic_mode '{self.mode}'.")

        self.value_head = nn.Linear(hidden_size, 1, bias=False)
        # Match the backbone's dtype/device so matmuls line up.
        self.value_head.to(device=ref_param.device, dtype=ref_param.dtype)

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        hidden_states: torch.Tensor = None,
    ) -> torch.Tensor:
        """Return per-token values, shape [B, T].

        In ``"separate"`` mode pass ``input_ids``/``attention_mask``; in
        ``"shared"`` mode pass the actor's ``hidden_states`` ([B, T, H]).
        """
        if self.mode == "separate":
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = out.last_hidden_state
        elif hidden_states is None:
            raise ValueError("critic_mode='shared' requires hidden_states.")
        return self.value_head(hidden_states).squeeze(-1)

    def trainable_parameters(self):
        """Params owned by the critic optimizer.

        In ``"shared"`` mode the backbone belongs to the actor (optimized by the
        actor's optimizer), so only the value head is the critic's own.
        """
        if self.mode == "separate":
            return self.parameters()
        return self.value_head.parameters()

"""Proximal Policy Optimization (PPO) RLHF for LLMs.

A minimal four-model implementation on top of the shared ``rl_common``
scaffolding:

  1. The actor samples rollouts under its own (current) policy.
  2. A reward source (frozen reward model or a verifiable rule) scores each
     full response; a per-token KL-to-reference penalty anchors the policy.
  3. A critic predicts per-token values; GAE turns rewards into advantages.
  4. The actor and critic are updated for several epochs of minibatch
     PPO-clipped surrogate + clipped value loss on each rollout buffer.

Only ``config.py``, ``value.py``, ``reward.py``, ``losses.py`` and
``trainer.py`` are PPO-specific.
"""

from .config import PPOConfig
from .trainer import PPOTrainer

__all__ = ["PPOConfig", "PPOTrainer"]

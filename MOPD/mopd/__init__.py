"""Multi-domain On-Policy Distillation (MOPD) for LLMs.

A minimal multi-teacher extension of OPD, built on the shared ``rl_common``
scaffolding:
  1. Student samples rollouts under its own (current) policy.
  2. Each prompt is routed to its *domain teacher*; that frozen teacher scores
     the rollout's tokens.
  3. Student is trained to minimize per-token reverse-KL towards the routed
     teacher, with truncated importance weights to correct the sampling /
     training temperature mismatch.

Only ``config.py`` (extra fields), ``losses.py`` (the multi-teacher reverse-KL
objective) and ``trainer.py`` (teacher pool + domain routing) are MOPD-specific.
"""

from .config import MOPDConfig
from .trainer import MOPDTrainer

__all__ = ["MOPDConfig", "MOPDTrainer"]

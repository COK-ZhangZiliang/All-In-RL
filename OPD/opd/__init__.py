"""On-Policy Distillation (OPD) for LLMs.

A minimal implementation built on top of the shared ``rl_common`` scaffolding:
  1. Student samples rollouts under its own (current) policy.
  2. The frozen teacher scores those exact tokens.
  3. Student is trained to minimize per-token reverse-KL D_KL(student||teacher).

Only ``config.py`` (extra fields), ``losses.py`` (the reverse-KL objective) and
``trainer.py`` (teacher loading + objective wiring) are OPD-specific.
"""

from .config import OPDConfig
from .trainer import OPDTrainer

__all__ = ["OPDConfig", "OPDTrainer"]

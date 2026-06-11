"""Shared building blocks for RL / post-training algorithms.

Concrete algorithms (OPD, GRPO, PPO, ...) subclass :class:`BaseConfig` and
:class:`BaseTrainer`, implement their objective in ``compute_loss``, and reuse
everything else (data, models, sampling, functional primitives, CLI).
"""

from .config import BaseConfig
from .functional import StepOutput
from .trainer import BaseTrainer

__all__ = ["BaseConfig", "BaseTrainer", "StepOutput"]

"""Configuration for On-Policy Distillation.

Only the fields unique to OPD live here; everything else is inherited from
:class:`rl_common.BaseConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass

from rl_common import BaseConfig


@dataclass
class OPDConfig(BaseConfig):
    # Frozen teacher; must share the student's vocabulary.
    teacher_model: str = "Qwen/Qwen3-1.7B"
    # Softmax temperature applied to both teacher & student logits in the KL.
    kl_temperature: float = 1.0

    def __post_init__(self) -> None:
        # OPD-specific default output dir.
        if self.output_dir == "outputs/run":
            self.output_dir = "outputs/opd"

"""The On-Policy Distillation trainer.

Everything generic (optimization loop, logging, eval, checkpointing) lives in
:class:`rl_common.BaseTrainer`. OPD only needs to:

  1. load the frozen teacher and check vocab compatibility, and
  2. define its objective: sample on-policy completions from the current
     student, then minimize the masked per-token reverse KL towards the teacher.
"""

from __future__ import annotations

from rl_common import BaseTrainer, StepOutput
from rl_common.models import load_frozen
from rl_common.sampling import generate_rollouts

from .config import OPDConfig
from .losses import check_vocab_compatibility, per_token_reverse_kl


class OPDTrainer(BaseTrainer):
    cfg: OPDConfig
    eval_delta_keys = ("reverse_kl", "student_nll")

    def setup_aux_models(self) -> None:
        self.teacher = load_frozen(self.cfg.teacher_model, self.cfg)
        check_vocab_compatibility(self.student, self.teacher)

    def compute_loss(self, batch: dict) -> StepOutput:
        # On-policy rollouts from the current student (no grad).
        rollout = generate_rollouts(self.student, self.tokenizer, batch, self.cfg)
        # Per-token reverse KL towards the teacher (student has grad).
        return per_token_reverse_kl(
            self.student,
            self.teacher,
            sequences=rollout.sequences,
            attention_mask=rollout.attention_mask,
            completion_mask=rollout.completion_mask,
            temperature=self.cfg.kl_temperature,
        )

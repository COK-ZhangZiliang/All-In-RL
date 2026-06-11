"""Per-token reverse-KL distillation loss — the core of On-Policy Distillation.

OPD uses the *mode-seeking* reverse KL with respect to the student:

    L = E_{x ~ p_s} [ D_KL( p_s(.|x) || p_t(.|x) ) ]

Rollouts x are sampled from the student itself, making this on-policy. Logits
at position i predict token i+1, so we shift logits/labels by one and align
the completion mask accordingly. The loss is a masked mean over completion
tokens only (prompt tokens are excluded).
"""

from __future__ import annotations

import torch

from rl_common import StepOutput
from rl_common.functional import (
    entropy_from_logp,
    gather_logp,
    log_softmax,
    masked_mean,
    next_token_logits,
)


def check_vocab_compatibility(student, teacher) -> None:
    """Teacher and student must share the same vocab for token-level KL."""
    s_vocab = student.config.vocab_size
    t_vocab = teacher.config.vocab_size
    if s_vocab != t_vocab:
        raise ValueError(
            "Teacher and student must share the same vocabulary for per-token "
            f"distillation, but got student vocab={s_vocab} vs teacher vocab={t_vocab}. "
            "Use models from the same tokenizer family (e.g. Qwen3-1.7B -> Qwen3-0.6B)."
        )


def per_token_reverse_kl(
    student,
    teacher,
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    temperature: float = 1.0,
) -> StepOutput:
    """Compute the masked, per-token reverse KL D_KL(student || teacher).

    ``student`` is trainable; ``teacher`` is frozen. ``completion_mask`` is 1
    on tokens where the loss applies (the on-policy completion).
    """

    student_logits = next_token_logits(student, sequences, attention_mask)
    with torch.no_grad():
        teacher_logits = next_token_logits(teacher, sequences, attention_mask)

    # Shift so position i predicts token i+1.
    student_logits = student_logits[:, :-1, :]
    teacher_logits = teacher_logits[:, :-1, :]
    loss_mask = completion_mask[:, 1:].to(student_logits.dtype)  # [B, T-1]

    student_logp = log_softmax(student_logits, temperature)
    teacher_logp = log_softmax(teacher_logits, temperature)
    student_p = student_logp.exp()

    kl_per_token = (student_p * (student_logp - teacher_logp)).sum(dim=-1)  # [B, T-1]
    loss = masked_mean(kl_per_token, loss_mask)

    with torch.no_grad():
        sampled = sequences[:, 1:]
        avg_teacher_nll = masked_mean(-gather_logp(teacher_logp, sampled), loss_mask)
        avg_student_nll = masked_mean(-gather_logp(student_logp, sampled), loss_mask)
        # Student entropy: drop signals policy collapse.
        avg_student_entropy = masked_mean(entropy_from_logp(student_logp), loss_mask)

    metrics = {
        "reverse_kl": loss.detach().item(),
        "teacher_nll": avg_teacher_nll.item(),
        "student_nll": avg_student_nll.item(),
        "student_entropy": avg_student_entropy.item(),
        "num_completion_tokens": int(loss_mask.sum().item()),
    }
    return StepOutput(loss=loss, metrics=metrics)

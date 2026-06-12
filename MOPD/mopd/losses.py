"""Multi-teacher reverse-KL distillation loss with truncated importance weights.

MOPD generalizes On-Policy Distillation to a *pool* of frozen domain teachers.
For a batch of student rollouts, each sample is routed to its domain's teacher
(``teacher_indices[b]``); the per-token reverse KL towards that teacher is then

    L = E_{x ~ p_s} [ w(x) * D_KL( p_s(.|x) || p_t_domain(.|x) ) ]

with ``w(x)`` the *truncated importance weight* correcting the mismatch between
the sampling distribution (``cfg.temperature``) and the true student policy
(temperature 1):

    w_i = min( exp( logp_student@T=1 - logp_student@sampling_T ), cap )

Setting ``cap <= 0`` disables the correction (strict on-policy, w == 1).
"""

from __future__ import annotations

from typing import Sequence

import torch

from rl_common import StepOutput
from rl_common.functional import (
    entropy_from_logp,
    gather_logp,
    log_softmax,
    masked_mean,
    next_token_logits,
)


def check_vocab_compatibility(student, teachers: Sequence) -> None:
    """All teachers must share the student's vocab for token-level KL."""
    s_vocab = student.config.vocab_size
    for i, t in enumerate(teachers):
        t_vocab = t.config.vocab_size
        if s_vocab != t_vocab:
            raise ValueError(
                "All teachers must share the student's vocabulary for per-token "
                f"distillation, but teacher #{i} vocab={t_vocab} vs student vocab={s_vocab}. "
                "Use teachers from the same tokenizer family as the student."
            )


def multi_teacher_reverse_kl(
    student,
    teachers: Sequence,
    teacher_indices: torch.Tensor,
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    sampling_temperature: float = 1.0,
    temperature: float = 1.0,
    importance_weight_cap: float = 2.0,
) -> StepOutput:
    """Per-token reverse KL D_KL(student || teacher[domain]) with truncated IS weights.

    ``teacher_indices`` is a [B] long tensor mapping each sequence to a teacher
    in ``teachers``. Each rollout is scored only by its assigned teacher.
    """

    student_logits = next_token_logits(student, sequences, attention_mask)
    # Shift so position i predicts token i+1.
    student_logits = student_logits[:, :-1, :]
    loss_mask = completion_mask[:, 1:].to(student_logits.dtype)  # [B, T-1]
    sampled = sequences[:, 1:]

    student_logp = log_softmax(student_logits, temperature)

    # Score each rollout under its assigned teacher only — saves compute when
    # different teachers run on disjoint slices of the batch.
    teacher_logp = torch.empty_like(student_logp)
    with torch.no_grad():
        idx = teacher_indices.to(sequences.device)
        for t_id, teacher in enumerate(teachers):
            sel = (idx == t_id).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            t_logits = next_token_logits(
                teacher, sequences[sel], attention_mask[sel]
            )[:, :-1, :]
            teacher_logp[sel] = log_softmax(t_logits, temperature)

    student_p = student_logp.exp()
    kl_per_token = (student_p * (student_logp - teacher_logp)).sum(dim=-1)  # [B, T-1]

    # Truncated importance weight: corrects rollouts sampled at a temperature
    # different from the policy's true (T=1) distribution. Per-sequence, applied
    # uniformly to that sequence's tokens.
    if importance_weight_cap > 0 and sampling_temperature != temperature:
        with torch.no_grad():
            sampling_logp = log_softmax(student_logits, sampling_temperature)
            student_logp_sampled = gather_logp(student_logp, sampled)
            sampling_logp_sampled = gather_logp(sampling_logp, sampled)
            seq_logr = ((student_logp_sampled - sampling_logp_sampled) * loss_mask).sum(dim=1)
            is_weight = seq_logr.exp().clamp(max=importance_weight_cap)  # [B]
        kl_per_token = kl_per_token * is_weight.unsqueeze(1)
    else:
        is_weight = torch.ones(sequences.shape[0], device=sequences.device)

    loss = masked_mean(kl_per_token, loss_mask)

    with torch.no_grad():
        avg_teacher_nll = masked_mean(-gather_logp(teacher_logp, sampled), loss_mask)
        avg_student_nll = masked_mean(-gather_logp(student_logp, sampled), loss_mask)
        # Student entropy: drop signals policy collapse.
        avg_student_entropy = masked_mean(entropy_from_logp(student_logp), loss_mask)

    metrics = {
        "reverse_kl": loss.detach().item(),
        "teacher_nll": avg_teacher_nll.item(),
        "student_nll": avg_student_nll.item(),
        "student_entropy": avg_student_entropy.item(),
        "is_weight_mean": float(is_weight.mean().item()),
        "num_completion_tokens": int(loss_mask.sum().item()),
    }
    return StepOutput(loss=loss, metrics=metrics)

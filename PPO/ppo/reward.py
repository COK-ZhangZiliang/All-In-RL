"""Scalar reward sources for PPO.

PPO turns a *sequence-level* scalar reward into a learning signal. Two
interchangeable sources are provided (selected by ``cfg.reward_source``):

  * :class:`ModelReward` — a frozen sequence-classification reward model scores
    the full (prompt + response) sequence. This is the classic RLHF setup.
  * :class:`VerifiableReward` — a rule-based reward: decode the completion,
    extract its final answer, and compare it to a ground-truth answer carried
    alongside the prompt (RLVR, e.g. GSM8K). No reward model needed.

Both expose ``score(sequences, attention_mask, completion_mask, refs) -> [B]``.
"""

from __future__ import annotations

import re
from typing import List, Optional

import torch

from rl_common.models import load_reward_model

from .config import PPOConfig


class ModelReward:
    """Frozen reward model: scalar score for each full sequence."""

    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        self.model = load_reward_model(cfg.reward_model, cfg)

    @torch.no_grad()
    def score(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_mask: torch.Tensor,
        refs: Optional[List[str]] = None,
    ) -> torch.Tensor:
        out = self.model(input_ids=sequences, attention_mask=attention_mask)
        # SequenceClassification head -> logits [B, 1]; squeeze to [B].
        return out.logits.squeeze(-1).float()


_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _extract_answer(text: str) -> Optional[str]:
    """Return the last number in ``text`` (GSM8K-style final answer)."""
    matches = _NUM_RE.findall(text)
    if not matches:
        return None
    return matches[-1].replace(",", "").rstrip(".")


class VerifiableReward:
    """Rule-based reward: +1 if the completion's answer matches the ground truth.

    The ground-truth answer string for each prompt is passed through ``refs``
    (the trainer reads it from the prompt file's ``answer_field``). The
    completion is decoded and its last number compared to the reference's last
    number — the standard lightweight GSM8K verifier.
    """

    def __init__(self, cfg: PPOConfig, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer

    @torch.no_grad()
    def score(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_mask: torch.Tensor,
        refs: Optional[List[str]] = None,
    ) -> torch.Tensor:
        if refs is None:
            raise ValueError(
                "VerifiableReward needs ground-truth answers; set cfg.answer_field "
                "and use a prompt file that carries it."
            )
        scores: List[float] = []
        for i in range(sequences.shape[0]):
            # completion_mask marks exactly the generated (non-padded) tokens.
            gen_ids = sequences[i][completion_mask[i].bool()]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            pred = _extract_answer(text)
            gold = _extract_answer(refs[i]) if refs[i] is not None else None
            scores.append(1.0 if (pred is not None and pred == gold) else 0.0)
        return torch.tensor(scores, dtype=torch.float32, device=sequences.device)


def build_reward(cfg: PPOConfig, tokenizer):
    if cfg.reward_source == "model":
        return ModelReward(cfg)
    if cfg.reward_source == "verifiable":
        return VerifiableReward(cfg, tokenizer)
    raise ValueError(
        f"Unknown reward_source '{cfg.reward_source}' (expected 'model' or 'verifiable')."
    )

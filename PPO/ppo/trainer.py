"""The PPO trainer.

PPO does not fit :class:`rl_common.BaseTrainer`'s single-``compute_loss``-per-step
contract: each rollout buffer is reused for *several* epochs of minibatch
updates (the trait that makes ratio clipping meaningful), and a second trainable
network — the critic — is optimized alongside the actor. So this trainer reuses
BaseTrainer's setup (tokenizer / actor / data / actor-optimizer / eval / save)
but overrides :meth:`train` with the PPO loop:

    rollout (no grad)  ->  reward + ref-KL  ->  GAE  ->  N epochs of
    minibatch clipped-surrogate + clipped-value updates.

``setup_aux_models`` loads the three extra models: a frozen reference (KL
anchor), the reward source (model or verifiable), and the trainable critic.
"""

from __future__ import annotations

import json
import os
from itertools import cycle
from typing import List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from rl_common import BaseTrainer, StepOutput
from rl_common.data import PromptDataset
from rl_common.functional import (
    entropy_from_logp,
    gather_logp,
    log_softmax,
    masked_mean,
)
from rl_common.models import load_frozen
from rl_common.sampling import generate_rollouts

from .config import PPOConfig
from .losses import (
    compute_gae,
    policy_loss,
    sequence_logprobs,
    token_rewards,
    value_loss,
)
from .reward import build_reward
from .value import Critic


class _AnswerPromptDataset(Dataset):
    """Wraps a :class:`PromptDataset`, returning a parallel ground-truth answer."""

    def __init__(self, base: PromptDataset, answers: List[str]):
        assert len(base) == len(answers)
        self.base = base
        self.answers = answers

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        item = self.base[idx]
        item["answer"] = self.answers[idx]
        return item


def _answer_collator(base_collate):
    def collate(batch):
        out = base_collate(batch)
        out["answer"] = [ex["answer"] for ex in batch]
        return out

    return collate


def _read_jsonl_answers(path: str, prompt_field: str, answer_field: str) -> List[str]:
    answers: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict) or prompt_field not in obj:
                continue
            answers.append(str(obj.get(answer_field, "")))
    return answers


class PPOTrainer(BaseTrainer):
    cfg: PPOConfig
    eval_delta_keys = ("score", "kl")

    # ------------------------------------------------------------ aux models
    def setup_aux_models(self) -> None:
        cfg = self.cfg
        self.ref = load_frozen(cfg.resolved_ref_model(), cfg)
        self.reward = build_reward(cfg, self.tokenizer)
        self.critic = Critic(cfg, actor=self.student)
        self.critic.train()
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.trainable_parameters(), lr=cfg.critic_learning_rate
        )
        print(
            f"[ppo] reward_source={cfg.reward_source} | critic_mode={cfg.critic_mode}",
            flush=True,
        )

    def __init__(self, cfg: PPOConfig, prompts=None):
        super().__init__(cfg, prompts=prompts)
        # Verifiable reward needs per-prompt ground-truth answers carried through
        # the dataloader (mirrors MOPD's per-prompt domain routing).
        if cfg.reward_source == "verifiable":
            self._attach_answers()

    def _attach_answers(self) -> None:
        cfg = self.cfg
        if cfg.prompt_file and cfg.prompt_file.endswith(".jsonl"):
            answers = _read_jsonl_answers(cfg.prompt_file, cfg.prompt_field, cfg.answer_field)
        else:
            answers = [""] * len(self.dataset)
        answers = (answers + [""] * len(self.dataset))[: len(self.dataset)]
        train_ds = _AnswerPromptDataset(self.dataset, answers)
        self.dataset = train_ds
        self.dataloader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=_answer_collator(self.dataloader.collate_fn),
        )
        if self.eval_dataloader is not None and cfg.eval_file and cfg.eval_file.endswith(".jsonl"):
            base_eval: PromptDataset = self.eval_dataloader.dataset  # type: ignore[assignment]
            eval_ans = _read_jsonl_answers(cfg.eval_file, cfg.prompt_field, cfg.answer_field)
            eval_ans = (eval_ans + [""] * len(base_eval))[: len(base_eval)]
            self.eval_dataloader = DataLoader(
                _AnswerPromptDataset(base_eval, eval_ans),
                batch_size=cfg.eval_batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=_answer_collator(self.eval_dataloader.collate_fn),
            )

    # ------------------------------------------------------- forward helpers
    def _refs_for_batch(self, batch: dict) -> Optional[List[str]]:
        ans = batch.get("answer")
        if ans is None:
            return None
        # generate() repeats each prompt num_generations times.
        return [a for a in ans for _ in range(self.cfg.num_generations)]

    def _policy_value_forward(self, sequences, attention_mask):
        """Per-token (log-prob, entropy, values) with grad, honoring critic_mode."""
        if self.cfg.critic_mode == "shared":
            out = self.student(
                input_ids=sequences,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            logp_dist = log_softmax(out.logits[:, :-1, :])
            new_logp = gather_logp(logp_dist, sequences[:, 1:])
            entropy = entropy_from_logp(logp_dist)
            values = self.critic(hidden_states=out.hidden_states[-1])
        else:
            new_logp, entropy = sequence_logprobs(self.student, sequences, attention_mask)
            values = self.critic(input_ids=sequences, attention_mask=attention_mask)
        return new_logp, entropy, values

    # ------------------------------------------------------------ rollout
    @torch.no_grad()
    def _collect_rollout(self, batch: dict) -> dict:
        cfg = self.cfg
        roll = generate_rollouts(self.student, self.tokenizer, batch, cfg)
        sequences, attn = roll.sequences, roll.attention_mask
        resp_mask = roll.completion_mask[:, 1:].float()

        old_logp, _ = sequence_logprobs(self.student, sequences, attn)
        ref_logp, _ = sequence_logprobs(self.ref, sequences, attn)

        if cfg.critic_mode == "shared":
            out = self.student(
                input_ids=sequences, attention_mask=attn, output_hidden_states=True
            )
            values = self.critic(hidden_states=out.hidden_states[-1])
        else:
            values = self.critic(input_ids=sequences, attention_mask=attn)

        scores = self.reward.score(
            sequences, attn, roll.completion_mask, refs=self._refs_for_batch(batch)
        )
        if cfg.reward_clip > 0:
            scores = scores.clamp(-cfg.reward_clip, cfg.reward_clip)

        rewards, kl = token_rewards(scores, old_logp, ref_logp, resp_mask, cfg.kl_coef)
        advantages, returns = compute_gae(
            rewards, values, resp_mask, cfg.gamma, cfg.gae_lambda
        )
        if cfg.normalize_advantages:
            mean = masked_mean(advantages, resp_mask)
            var = masked_mean((advantages - mean) ** 2, resp_mask)
            advantages = (advantages - mean) / (var.sqrt() + 1e-8) * resp_mask

        return {
            "sequences": sequences,
            "attention_mask": attn,
            "resp_mask": resp_mask,
            "old_logp": old_logp,
            "old_values": values[:, :-1],
            "advantages": advantages,
            "returns": returns,
            "score": scores.mean().item(),
            "kl": masked_mean(kl, resp_mask).item(),
            "num_completion_tokens": int(resp_mask.sum().item()),
        }

    # ----------------------------------------------------------- PPO update
    def _ppo_update(self, buf: dict) -> dict:
        cfg = self.cfg
        n = buf["sequences"].shape[0]
        metrics: dict = {}
        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(n, device=buf["sequences"].device)
            for start in range(0, n, cfg.ppo_mini_batch_size):
                idx = perm[start : start + cfg.ppo_mini_batch_size]
                seq, attn = buf["sequences"][idx], buf["attention_mask"][idx]
                resp_mask = buf["resp_mask"][idx]
                adv, ret = buf["advantages"][idx], buf["returns"][idx]
                old_logp, old_values = buf["old_logp"][idx], buf["old_values"][idx]

                new_logp, entropy, values = self._policy_value_forward(seq, attn)
                new_values = values[:, :-1]

                pol_loss, pol_stats = policy_loss(
                    new_logp, old_logp, adv, resp_mask, cfg.clip_ratio
                )
                val_loss = value_loss(
                    new_values, old_values, ret, resp_mask, cfg.value_clip
                )
                ent = masked_mean(entropy, resp_mask)
                loss = pol_loss - cfg.entropy_coef * ent + cfg.vf_coef * val_loss

                self.optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                actor_gn = torch.nn.utils.clip_grad_norm_(
                    self.student.parameters(), cfg.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), cfg.max_grad_norm
                )
                self.optimizer.step()
                self.critic_optimizer.step()

                metrics = {
                    "policy_loss": pol_loss.item(),
                    "value_loss": val_loss.item(),
                    "entropy": ent.item(),
                    "grad_norm": float(actor_gn),
                    **pol_stats,
                }
        return metrics

    # --------------------------------------------------------------- loop
    def train(self) -> None:
        cfg = self.cfg
        self.student.train()
        self.critic.train()
        data_iter = cycle(self.dataloader)

        if self.eval_dataloader is not None:
            self.baseline_eval = self.evaluate(tag="baseline")

        while self.global_step < cfg.num_train_steps:
            batch = next(data_iter)
            buf = self._collect_rollout(batch)
            metrics = self._ppo_update(buf)
            self.scheduler.step()
            self.global_step += 1

            metrics["score"] = buf["score"]
            metrics["kl"] = buf["kl"]
            metrics["lr"] = self.scheduler.get_last_lr()[0]

            if cfg.log_every and self.global_step % cfg.log_every == 0:
                self._log(metrics)
            if (
                self.eval_dataloader is not None
                and cfg.eval_every
                and self.global_step % cfg.eval_every == 0
            ):
                self.evaluate(tag=f"step-{self.global_step}")
            if cfg.save_every and self.global_step % cfg.save_every == 0:
                self.save(os.path.join(cfg.output_dir, f"step-{self.global_step}"))

        if self.eval_dataloader is not None:
            self.evaluate(tag="final")
        if cfg.save_final:
            self.save(os.path.join(cfg.output_dir, "final"))

    # ---------------------------------------------------------------- eval
    def compute_loss(self, batch: dict) -> StepOutput:
        """Eval-only: report mean reward / KL / entropy on held-out prompts.

        The base trainer calls this under ``no_grad`` to aggregate token-weighted
        metrics; PPO training itself uses :meth:`train` directly.
        """
        buf = self._collect_rollout(batch)
        ent = masked_mean(
            sequence_logprobs(self.student, buf["sequences"], buf["attention_mask"])[1],
            buf["resp_mask"],
        )
        metrics = {
            "score": buf["score"],
            "kl": buf["kl"],
            "entropy": ent.item(),
            "num_completion_tokens": buf["num_completion_tokens"],
        }
        return StepOutput(loss=torch.zeros((), device=self.cfg.device), metrics=metrics)

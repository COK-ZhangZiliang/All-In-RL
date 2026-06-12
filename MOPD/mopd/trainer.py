"""The Multi-domain On-Policy Distillation trainer.

Everything generic (optimization loop, logging, eval, checkpointing) lives in
:class:`rl_common.BaseTrainer`. MOPD only needs to:

  1. load *several* frozen teachers (one per domain) and check vocab.
  2. carry each prompt's ``domain`` through the dataloader so each rollout can
     be routed to its domain's teacher.
  3. define its objective: sample on-policy completions from the current
     student, then minimize the masked per-token reverse KL towards the routed
     teacher, with an optional truncated importance-weight correction.
"""

from __future__ import annotations

import json
import os
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset

from rl_common import BaseTrainer, StepOutput
from rl_common.data import PromptDataset
from rl_common.models import load_frozen
from rl_common.sampling import generate_rollouts

from .config import MOPDConfig
from .losses import check_vocab_compatibility, multi_teacher_reverse_kl


class _DomainPromptDataset(Dataset):
    """Wraps a :class:`PromptDataset` and returns a parallel ``domain`` field."""

    def __init__(self, base: PromptDataset, domains: List[str]):
        assert len(base) == len(domains)
        self.base = base
        self.domains = domains

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        item = self.base[idx]
        item["domain"] = self.domains[idx]
        return item


def _domain_collator(base_collate):
    """Extend a standard prompt collator to also stack the per-prompt domain."""

    def collate(batch):
        out = base_collate(batch)
        out["domain"] = [ex["domain"] for ex in batch]
        return out

    return collate


def _read_jsonl_prompts_with_domain(
    path: str, prompt_field: str, domain_field: str, default_domain: str
) -> tuple[List[str], List[str]]:
    prompts: List[str] = []
    domains: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict) or prompt_field not in obj:
                continue
            prompts.append(str(obj[prompt_field]))
            domains.append(str(obj.get(domain_field, default_domain)))
    return prompts, domains


class MOPDTrainer(BaseTrainer):
    cfg: MOPDConfig
    eval_delta_keys = ("reverse_kl", "student_nll")

    def setup_aux_models(self) -> None:
        # One frozen teacher per domain (registration order = teacher index).
        teacher_paths = self.cfg.teacher_paths
        self.teachers = [load_frozen(p, self.cfg) for p in teacher_paths]
        check_vocab_compatibility(self.student, self.teachers)
        self.domain_to_index = {d: i for i, d in enumerate(self.cfg.domains)}
        print(
            f"[mopd] {len(self.teachers)} domain teachers: "
            + ", ".join(f"{d}#{i}" for d, i in self.domain_to_index.items()),
            flush=True,
        )

    def __init__(self, cfg: MOPDConfig, prompts=None):
        super().__init__(cfg, prompts=prompts)
        # Rebuild train/eval loaders so each batch carries a per-prompt domain.
        self._rebuild_domain_aware_loaders()

    # ---------------------------------------------------------- dataloaders
    def _rebuild_domain_aware_loaders(self) -> None:
        cfg = self.cfg
        # Train: read domains from the prompt file when it is JSONL; else all
        # samples fall back to the default domain.
        if cfg.prompt_file and cfg.prompt_file.endswith(".jsonl"):
            _, domains = _read_jsonl_prompts_with_domain(
                cfg.prompt_file, cfg.prompt_field, cfg.domain_field, cfg.default_domain
            )
        else:
            domains = [cfg.default_domain] * len(self.dataset)

        train_ds = _DomainPromptDataset(self.dataset, domains)
        base_collate = self.dataloader.collate_fn
        self.dataset = train_ds
        self.dataloader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=_domain_collator(base_collate),
        )

        # Eval: same treatment when an eval_file is present.
        if self.eval_dataloader is not None and cfg.eval_file and cfg.eval_file.endswith(".jsonl"):
            base_eval_ds: PromptDataset = self.eval_dataloader.dataset  # type: ignore[assignment]
            _, eval_domains = _read_jsonl_prompts_with_domain(
                cfg.eval_file, cfg.prompt_field, cfg.domain_field, cfg.default_domain
            )
            eval_domains = eval_domains[: len(base_eval_ds)]
            if len(eval_domains) < len(base_eval_ds):
                eval_domains += [cfg.default_domain] * (len(base_eval_ds) - len(eval_domains))
            eval_ds = _DomainPromptDataset(base_eval_ds, eval_domains)
            base_eval_collate = self.eval_dataloader.collate_fn
            self.eval_dataloader = DataLoader(
                eval_ds,
                batch_size=cfg.eval_batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=_domain_collator(base_eval_collate),
            )

    # ----------------------------------------------------------------- core
    def compute_loss(self, batch: dict) -> StepOutput:
        # On-policy rollouts from the current student (no grad).
        rollout = generate_rollouts(self.student, self.tokenizer, batch, self.cfg)

        # Map per-prompt domains to per-rollout teacher indices, accounting for
        # ``num_generations`` rollouts per prompt (HF generate repeats prompts).
        unknown = [d for d in batch["domain"] if d not in self.domain_to_index]
        if unknown:
            raise KeyError(
                f"Prompt domain(s) {sorted(set(unknown))} have no teacher; "
                f"known domains: {list(self.domain_to_index)}"
            )
        per_prompt = torch.tensor(
            [self.domain_to_index[d] for d in batch["domain"]], dtype=torch.long
        )
        teacher_indices = per_prompt.repeat_interleave(self.cfg.num_generations)

        return multi_teacher_reverse_kl(
            self.student,
            self.teachers,
            teacher_indices=teacher_indices,
            sequences=rollout.sequences,
            attention_mask=rollout.attention_mask,
            completion_mask=rollout.completion_mask,
            sampling_temperature=self.cfg.temperature if self.cfg.temperature > 0 else 1.0,
            temperature=self.cfg.kl_temperature,
            importance_weight_cap=self.cfg.importance_weight_cap,
        )

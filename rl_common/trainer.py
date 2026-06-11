"""Generic training-loop scaffolding shared by all algorithms.

``BaseTrainer`` owns everything that is *not* algorithm-specific: seeding,
tokenizer/student/data/optimizer/scheduler setup, the gradient-accumulation
loop, gradient clipping, logging, checkpointing and the eval aggregation.

A concrete algorithm subclasses it and implements a single method:

    def compute_loss(self, batch) -> StepOutput

It may also override :meth:`setup_aux_models` to load auxiliary frozen models
(teacher / reference / reward) after the student is built.
"""

from __future__ import annotations

import json
import os
import random
from itertools import cycle
from typing import List, Optional

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from .config import BaseConfig
from .data import PromptDataset, build_prompt_dataset, make_prompt_collator
from .functional import StepOutput
from .models import load_student, load_tokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BaseTrainer:
    def __init__(self, cfg: BaseConfig, prompts: Optional[List[str]] = None):
        self.cfg = cfg
        set_seed(cfg.seed)

        self.tokenizer = load_tokenizer(cfg)
        self.student = load_student(cfg)
        self.setup_aux_models()

        self.dataset = build_prompt_dataset(cfg, self.tokenizer, prompts=prompts)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=make_prompt_collator(self.tokenizer),
        )

        # num_epochs (if set) overrides num_train_steps with a full pass count.
        steps_per_epoch = max(1, len(self.dataset) // cfg.batch_size // cfg.grad_accum_steps)
        if cfg.num_epochs and cfg.num_epochs > 0:
            cfg.num_train_steps = steps_per_epoch * cfg.num_epochs
            print(
                f"[trainer] num_epochs={cfg.num_epochs} -> num_train_steps="
                f"{cfg.num_train_steps} ({steps_per_epoch} steps/epoch)",
                flush=True,
            )

        self.eval_dataloader = self._build_eval_loader()
        self.baseline_eval: Optional[dict] = None

        self.optimizer = torch.optim.AdamW(
            (p for p in self.student.parameters() if p.requires_grad),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=cfg.warmup_steps,
            num_training_steps=cfg.num_train_steps,
        )
        os.makedirs(cfg.output_dir, exist_ok=True)
        self.global_step = 0

    # --------------------------------------------------------------- hooks
    def setup_aux_models(self) -> None:
        """Load auxiliary frozen models (teacher / reference / reward).

        Default: no auxiliary model. Override in algorithms that need one.
        """

    def compute_loss(self, batch: dict) -> StepOutput:
        """Algorithm-specific objective. Must be implemented by subclasses."""
        raise NotImplementedError

    # ----------------------------------------------------------- eval loader
    def _build_eval_loader(self) -> Optional[DataLoader]:
        cfg = self.cfg
        if not cfg.eval_every or not cfg.eval_file:
            return None
        if not os.path.exists(cfg.eval_file):
            print(f"[eval] eval_file not found: {cfg.eval_file} (skipping eval)", flush=True)
            return None

        prompts: List[str] = []
        with open(cfg.eval_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict) and cfg.prompt_field in obj:
                    prompts.append(str(obj[cfg.prompt_field]))
        if not prompts:
            print(f"[eval] no prompts parsed from {cfg.eval_file}", flush=True)
            return None
        cap = cfg.eval_max_samples
        if cap is not None and cap > 0 and len(prompts) > cap:
            prompts = prompts[:cap]
            cap_note = f"capped to {cap}"
        else:
            cap_note = "full eval set"

        eval_ds = PromptDataset(prompts, self.tokenizer, cfg)
        loader = DataLoader(
            eval_ds,
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=make_prompt_collator(self.tokenizer),
        )
        print(
            f"[eval] held-out prompts: {len(prompts)} ({cap_note}) from {cfg.eval_file}",
            flush=True,
        )
        return loader

    # ------------------------------------------------------------ train step
    def _train_step(self, batch: dict) -> dict:
        out = self.compute_loss(batch)
        loss = out.loss / self.cfg.grad_accum_steps
        loss.backward()
        return out.metrics

    def train(self) -> None:
        cfg = self.cfg
        self.student.train()
        data_iter = cycle(self.dataloader)

        # Baseline eval at step 0 so subsequent reports can show Δ.
        if self.eval_dataloader is not None:
            self.baseline_eval = self.evaluate(tag="baseline")

        running = {}
        while self.global_step < cfg.num_train_steps:
            self.optimizer.zero_grad(set_to_none=True)

            metrics = {}
            for _ in range(cfg.grad_accum_steps):
                batch = next(data_iter)
                metrics = self._train_step(batch)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.student.parameters(), cfg.max_grad_norm
            )
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            metrics["grad_norm"] = float(grad_norm)
            metrics["lr"] = self.scheduler.get_last_lr()[0]
            running.update(metrics)

            if cfg.log_every and self.global_step % cfg.log_every == 0:
                self._log(running)
            if (
                self.eval_dataloader is not None
                and cfg.eval_every
                and self.global_step % cfg.eval_every == 0
            ):
                self.evaluate(tag=f"step-{self.global_step}")
            if cfg.save_every and self.global_step % cfg.save_every == 0:
                self.save(os.path.join(cfg.output_dir, f"step-{self.global_step}"))

        # Final eval after training so the user sees total improvement.
        if self.eval_dataloader is not None:
            self.evaluate(tag="final")

        if cfg.save_final:
            self.save(os.path.join(cfg.output_dir, "final"))

    # --------------------------------------------------------------- eval
    #: Metric keys reported as Δ vs. baseline by :meth:`evaluate`.
    eval_delta_keys: tuple = ()

    @torch.no_grad()
    def evaluate(self, tag: str = "eval") -> dict:
        """Token-weighted aggregate of ``compute_loss`` metrics over the eval set.

        Reuses the algorithm's own objective for evaluation: every scalar metric
        is averaged weighted by ``num_completion_tokens``. Reports Δ vs. the
        step-0 baseline for the keys in :attr:`eval_delta_keys`.
        """
        cfg = self.cfg
        assert self.eval_dataloader is not None
        was_training = self.student.training
        self.student.eval()

        sums: dict = {}
        sum_tokens = 0
        n_prompts = 0
        for batch in self.eval_dataloader:
            out = self.compute_loss(batch)
            n_tok = out.metrics.get("num_completion_tokens", 0)
            if n_tok == 0:
                continue
            for k, v in out.metrics.items():
                if k == "num_completion_tokens" or not isinstance(v, (int, float)):
                    continue
                sums[k] = sums.get(k, 0.0) + v * n_tok
            sum_tokens += n_tok
            n_prompts += int(batch["input_ids"].shape[0]) * cfg.num_generations

        if was_training:
            self.student.train()

        if sum_tokens == 0:
            print(f"[eval/{tag}] no completion tokens generated.", flush=True)
            return {}

        result = {k: v / sum_tokens for k, v in sums.items()}
        result["num_prompts"] = n_prompts
        result["num_completion_tokens"] = sum_tokens

        # Report with Δ-vs-baseline when applicable.
        delta_str = ""
        baseline = self.baseline_eval
        if baseline and tag != "baseline":
            deltas = []
            for k in self.eval_delta_keys:
                if k in result and k in baseline:
                    deltas.append(f"Δ{k}={result[k] - baseline[k]:+.4f}")
            if deltas:
                delta_str = " | " + " | ".join(deltas)
        msg = " | ".join(f"{k}={self._fmt(v)}" for k, v in result.items())
        print(f"[eval/{tag}] {msg}{delta_str}", flush=True)
        return result

    # -------------------------------------------------------------- logging
    @staticmethod
    def _fmt(v) -> str:
        if isinstance(v, float):
            # Scientific notation for small magnitudes (e.g. lr=1e-5) to avoid 0.0000.
            if v != 0 and abs(v) < 1e-3:
                return f"{v:.3e}"
            return f"{v:.4f}"
        return str(v)

    def _log(self, metrics: dict) -> None:
        msg = " | ".join(f"{k}={self._fmt(v)}" for k, v in metrics.items())
        print(f"[step {self.global_step}/{self.cfg.num_train_steps}] {msg}", flush=True)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.student.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"[save] student checkpoint -> {path}", flush=True)

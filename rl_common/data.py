"""Prompt dataset loading shared by all algorithms.

Source priority:
  1. Explicit list of strings passed to ``build_prompt_dataset``.
  2. Local file via ``cfg.prompt_file`` (.jsonl with ``prompt_field`` or .txt).
  3. ``cfg.dataset_name`` via ModelScope or HuggingFace.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

import torch
from torch.utils.data import Dataset

from .config import BaseConfig


def _read_prompt_file(path: str, prompt_field: str) -> List[str]:
    prompts: List[str] = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompts.append(obj[prompt_field] if isinstance(obj, dict) else str(obj))
    else:  # plain text, one prompt per line
        with open(path, "r", encoding="utf-8") as f:
            prompts = [ln.rstrip("\n") for ln in f if ln.strip()]
    return prompts


def _load_hf_prompts(cfg: BaseConfig) -> List[str]:
    from datasets import load_dataset

    ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.dataset_split)
    if cfg.prompt_field not in ds.column_names:
        raise KeyError(
            f"prompt_field '{cfg.prompt_field}' not in dataset columns {ds.column_names}"
        )
    return [str(x) for x in ds[cfg.prompt_field]]


def _load_modelscope_prompts(cfg: BaseConfig) -> List[str]:
    from modelscope.msdatasets import MsDataset

    print(
        f"[modelscope] MsDataset.load: {cfg.dataset_name} "
        f"(subset={cfg.dataset_config}, split={cfg.dataset_split})",
        flush=True,
    )
    kwargs = dict(dataset_name=cfg.dataset_name, split=cfg.dataset_split)
    if cfg.dataset_config:
        kwargs["subset_name"] = cfg.dataset_config
    if cfg.modelscope_cache_dir:
        kwargs["cache_dir"] = cfg.modelscope_cache_dir
    ds = MsDataset.load(**kwargs)

    prompts: List[str] = []
    for item in ds:
        if cfg.prompt_field not in item:
            raise KeyError(
                f"prompt_field '{cfg.prompt_field}' not in dataset item keys {list(item.keys())}"
            )
        prompts.append(str(item[cfg.prompt_field]))
    return prompts


class PromptDataset(Dataset):
    """Tokenizes prompts (optionally with a chat template) lazily."""

    def __init__(self, prompts: List[str], tokenizer, cfg: BaseConfig):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.prompts)

    def _render(self, prompt: str) -> str:
        if self.cfg.apply_chat_template and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return prompt

    def __getitem__(self, idx: int) -> dict:
        text = self._render(self.prompts[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.cfg.max_prompt_length,
            return_tensors=None,
            add_special_tokens=not self.cfg.apply_chat_template,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "prompt_text": text,
        }


def build_prompt_dataset(
    cfg: BaseConfig,
    tokenizer,
    prompts: Optional[List[str]] = None,
) -> PromptDataset:
    if prompts is None:
        if cfg.prompt_file and os.path.exists(cfg.prompt_file):
            prompts = _read_prompt_file(cfg.prompt_file, cfg.prompt_field)
        elif cfg.dataset_name:
            prompts = (
                _load_modelscope_prompts(cfg) if cfg.use_modelscope
                else _load_hf_prompts(cfg)
            )
        else:
            raise ValueError(
                "No prompt source: pass `prompts=`, set cfg.prompt_file, or cfg.dataset_name."
            )
    if len(prompts) == 0:
        raise ValueError("Prompt dataset is empty.")
    return PromptDataset(prompts, tokenizer, cfg)


def make_prompt_collator(tokenizer):
    """Left-pad a batch of variable-length prompts for batched generation."""

    pad_id = tokenizer.pad_token_id

    def collate(batch: List[dict]) -> dict:
        max_len = max(len(ex["input_ids"]) for ex in batch)
        input_ids, attn, texts = [], [], []
        for ex in batch:
            ids = ex["input_ids"]
            mask = ex["attention_mask"]
            pad = max_len - len(ids)
            input_ids.append([pad_id] * pad + ids)
            attn.append([0] * pad + mask)
            texts.append(ex["prompt_text"])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "prompt_text": texts,
        }

    return collate

"""Model & tokenizer loading shared by all algorithms.

When ``cfg.use_modelscope=True`` and the model id is not already a local
directory, the snapshot is fetched from ModelScope first. Either way, models
are then loaded with the standard ``transformers.from_pretrained`` API.

Algorithms with auxiliary frozen models (teacher for distillation, reference
for DPO, reward model for PPO) share :func:`load_frozen`.
"""

from __future__ import annotations

import inspect
import os

import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from .config import BaseConfig

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_dtype(name: str) -> torch.dtype:
    if name not in _DTYPE_MAP:
        raise ValueError(f"Unknown torch_dtype '{name}', expected one of {list(_DTYPE_MAP)}")
    return _DTYPE_MAP[name]


def _resolve_model_path(model_id_or_path: str, cfg: BaseConfig) -> str:
    """Return a local directory containing the model files."""
    if os.path.isdir(model_id_or_path):
        return model_id_or_path
    if not cfg.use_modelscope:
        return model_id_or_path  # let HF resolve the id

    from modelscope import snapshot_download

    print(f"[modelscope] snapshot_download: {model_id_or_path}", flush=True)
    local_dir = snapshot_download(
        model_id_or_path,
        cache_dir=cfg.modelscope_cache_dir,
    )
    print(f"[modelscope] -> {local_dir}", flush=True)
    return local_dir


def load_tokenizer(cfg: BaseConfig):
    path = _resolve_model_path(cfg.resolved_tokenizer(), cfg)
    tok = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=cfg.trust_remote_code,
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # Required for batched generation with decoder-only models.
    tok.padding_side = "left"
    return tok


def _from_pretrained_kwargs(auto_cls, cfg: BaseConfig) -> dict:
    dtype = resolve_dtype(cfg.torch_dtype)
    # transformers >=5 renamed `torch_dtype` -> `dtype`.
    sig = inspect.signature(auto_cls.from_pretrained)
    dtype_kw = "dtype" if "dtype" in sig.parameters else "torch_dtype"
    kwargs = {
        dtype_kw: dtype,
        "trust_remote_code": cfg.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if cfg.attn_implementation:
        kwargs["attn_implementation"] = cfg.attn_implementation
    return kwargs


def _load_model(model_id_or_path: str, cfg: BaseConfig):
    path = _resolve_model_path(model_id_or_path, cfg)
    kwargs = _from_pretrained_kwargs(AutoModelForCausalLM, cfg)
    return AutoModelForCausalLM.from_pretrained(path, **kwargs)


def load_student(cfg: BaseConfig):
    """Trainable policy model."""
    model = _load_model(cfg.student_model, cfg)
    model.to(cfg.device)
    model.train()
    if hasattr(model, "config"):
        model.config.use_cache = False
    return model


def load_frozen(model_id_or_path: str, cfg: BaseConfig):
    """Frozen auxiliary model used only for scoring (teacher / reference / reward)."""
    model = _load_model(model_id_or_path, cfg)
    model.to(cfg.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_reward_model(model_id_or_path: str, cfg: BaseConfig):
    """Frozen sequence-classification reward model that scores full responses.

    RLHF reward models are causal LMs with the LM head swapped for a scalar
    regression head (``num_labels=1``); ``AutoModelForSequenceClassification``
    loads exactly that. The forward pass returns ``logits`` of shape [B, 1] —
    the scalar reward for each (prompt, response) sequence.
    """
    path = _resolve_model_path(model_id_or_path, cfg)
    kwargs = _from_pretrained_kwargs(AutoModelForSequenceClassification, cfg)
    kwargs["num_labels"] = 1
    model = AutoModelForSequenceClassification.from_pretrained(path, **kwargs)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
    model.to(cfg.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_value_backbone(model_id_or_path: str, cfg: BaseConfig):
    """Trainable transformer *backbone* (no LM head) for a PPO critic.

    ``AutoModel`` returns last-hidden-states; the critic in
    :mod:`ppo.value` adds a scalar value head on top. Kept trainable.
    """
    path = _resolve_model_path(model_id_or_path, cfg)
    kwargs = _from_pretrained_kwargs(AutoModel, cfg)
    model = AutoModel.from_pretrained(path, **kwargs)
    model.to(cfg.device)
    model.train()
    if hasattr(model, "config"):
        model.config.use_cache = False
    return model

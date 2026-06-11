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
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def _load_model(model_id_or_path: str, cfg: BaseConfig):
    path = _resolve_model_path(model_id_or_path, cfg)
    dtype = resolve_dtype(cfg.torch_dtype)
    # transformers >=5 renamed `torch_dtype` -> `dtype`.
    sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
    dtype_kw = "dtype" if "dtype" in sig.parameters else "torch_dtype"
    kwargs = {
        dtype_kw: dtype,
        "trust_remote_code": cfg.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if cfg.attn_implementation:
        kwargs["attn_implementation"] = cfg.attn_implementation
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

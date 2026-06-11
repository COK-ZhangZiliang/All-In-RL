"""Shared base configuration for all RL / post-training algorithms.

``BaseConfig`` collects every hyper-parameter that is *not* specific to a single
algorithm: model loading, data sources, on-policy sampling, the optimization
loop, logging and evaluation. Concrete algorithms subclass it and add only the
few fields that are truly their own (e.g. OPD adds ``teacher_model`` and
``kl_temperature``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BaseConfig:
    """Hyper-parameters shared by every algorithm in this repo.

    Defaults target small Qwen3 models on CPU + float32 so any algorithm can be
    smoke-tested on a laptop. Switch ``device`` to ``cuda`` and ``torch_dtype``
    to ``bfloat16`` for real training.
    """

    # ------------------------------------------------------------------ models
    # The single trainable policy. Distillation/RLHF algorithms add their own
    # auxiliary models (teacher / reference / reward) in their subclass.
    student_model: str = "Qwen/Qwen3-0.6B"
    tokenizer_name: Optional[str] = None  # defaults to ``student_model``
    torch_dtype: str = "float32"
    trust_remote_code: bool = True
    attn_implementation: Optional[str] = None  # e.g. "flash_attention_2"

    # ------------------------------------------------ download backend
    use_modelscope: bool = True  # False -> HuggingFace hub
    modelscope_cache_dir: Optional[str] = None

    # -------------------------------------------------------------------- data
    dataset_name: Optional[str] = None        # e.g. "AI-ModelScope/gsm8k"
    dataset_config: Optional[str] = "main"
    dataset_split: str = "train"
    prompt_field: str = "question"
    prompt_file: Optional[str] = None         # local .jsonl/.txt; overrides dataset_name
    max_prompt_length: int = 256
    apply_chat_template: bool = True

    # ---------------------------------------------------------------- sampling
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0                            # 0 disables top-k
    num_generations: int = 1                  # rollouts per prompt
    use_cache_in_generation: bool = True

    # ---------------------------------------------------------------- training
    output_dir: str = "outputs/run"
    num_train_steps: int = 1000               # ignored if num_epochs > 0
    num_epochs: int = 0                       # >0 -> overrides num_train_steps
    batch_size: int = 1
    grad_accum_steps: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    warmup_steps: int = 5
    max_grad_norm: float = 1.0
    seed: int = 42

    # ------------------------------------------------------------------ logging
    log_every: int = 1
    save_every: int = 200
    save_final: bool = True
    eval_every: int = 20                      # 0 disables periodic eval
    eval_file: Optional[str] = None
    eval_max_samples: Optional[int] = None    # None / 0 / negative -> full eval set
    eval_batch_size: int = 1
    device: str = "cpu"

    extra: dict = field(default_factory=dict)

    def resolved_tokenizer(self) -> str:
        return self.tokenizer_name or self.student_model

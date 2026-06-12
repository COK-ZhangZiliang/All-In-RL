"""Configuration for On-Policy Distillation.

OPD's algorithmic surface is small (one frozen teacher + a per-token reverse-KL),
so most fields are inherited from :class:`rl_common.BaseConfig`. We still
restate the *defaults that matter for an OPD run* here — student model,
teacher model, dataset recipe, sampling/training shape — so each algorithm's
config is self-describing instead of a single new field on top of an opaque
base.
"""

from __future__ import annotations

from dataclasses import dataclass

from rl_common import BaseConfig


@dataclass
class OPDConfig(BaseConfig):
    # -------------------------------------------------------------- models
    # Trainable policy. Teacher and student must share the same tokenizer
    # for per-token reverse-KL distillation.
    student_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    # Frozen teacher; same Qwen2.5 family as the student so the vocab
    # matches without any extra glue.
    teacher_model: str = "Qwen/Qwen2.5-7B-Instruct"

    # ---------------------------------------------------------------- data
    # GSM8K math word problems via the shared recipe registry.
    dataset_recipe: str = "gsm8k"
    prompt_field: str = "question"

    # ------------------------------------------------------------ sampling
    # On-policy rollouts from the student.
    max_new_tokens: int = 256
    temperature: float = 1.0
    num_generations: int = 1

    # -------------------------------------------------------------- loop
    output_dir: str = "outputs/opd"
    num_train_steps: int = 200
    batch_size: int = 2
    learning_rate: float = 1e-5

    # ----------------------------------------------------------- objective
    # Softmax temperature applied to both teacher & student logits in the KL.
    kl_temperature: float = 1.0

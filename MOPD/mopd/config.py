"""Configuration for Multi-domain On-Policy Distillation (MOPD).

Only the fields unique to MOPD live here; everything else is inherited from
:class:`rl_common.BaseConfig`.

MOPD distills several *domain-specific* frozen teachers into one student. The
teacher pool is given as ``teacher_models``: a comma-separated list of
``domain:model_id`` entries (a bare ``model_id`` is taken as the ``default``
domain). Each training prompt is routed to its domain's teacher via the
``domain_field`` of the prompt file; prompts without a domain fall back to the
first declared domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from rl_common import BaseConfig


@dataclass
class MOPDConfig(BaseConfig):
    # -------------------------------------------------------------- models
    # Trainable student. Tokenizer must match every teacher in the pool.
    student_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    # Domain teacher pool. The default is the canonical math/code/general
    # Qwen2.5 expert trio: all three are 7B Qwen2.5 family checkpoints and
    # therefore share the student's tokenizer (verified at startup).
    # A bare id without "domain:" is registered as the "default" domain.
    teacher_models: str = (
        "math:Qwen/Qwen2.5-Math-7B-Instruct,"
        "code:Qwen/Qwen2.5-Coder-7B-Instruct,"
        "general:Qwen/Qwen2.5-7B-Instruct"
    )

    # ---------------------------------------------------------------- data
    # When ``mix_recipes`` is empty, a single recipe (this one) is loaded; all
    # rollouts get routed to the first declared domain. Override to mix
    # several recipes from ``rl_common.recipes`` into a domain-tagged jsonl.
    dataset_recipe: str = "gsm8k"
    prompt_field: str = "question"
    # Prompt-file field naming each prompt's domain (for teacher routing).
    domain_field: str = "domain"
    # Comma-separated list of recipe names to mix (e.g. "gsm8k,mbpp,alpaca").
    mix_recipes: str = "gsm8k,mbpp,alpaca"
    # Comma-separated ratios aligned with ``mix_recipes`` (e.g. "2,1,1").
    # Empty means uniform / equal weight.
    mix_ratios: str = "1,1,1"
    # Optional cap on prompts drawn per recipe (0 -> use all).
    mix_max_per_recipe: int = 0
    # Optional cap on eval prompts per recipe (0 -> 64).
    mix_eval_per_recipe: int = 64

    # ------------------------------------------------------------ sampling
    max_new_tokens: int = 256
    temperature: float = 1.0
    num_generations: int = 1

    # -------------------------------------------------------------- loop
    output_dir: str = "outputs/mopd"
    num_train_steps: int = 200
    batch_size: int = 2
    learning_rate: float = 1e-5

    # ----------------------------------------------------------- objective
    # Softmax temperature applied to teacher & student logits in the KL.
    kl_temperature: float = 1.0
    # Truncated importance weighting cap, correcting the sampling/training
    # mismatch (sampling uses cfg.temperature; the true policy uses T=1).
    # cap <= 0 disables the correction (pure on-policy, weight == 1).
    importance_weight_cap: float = 2.0

    def __post_init__(self) -> None:
        # Parse the teacher spec once; keep domains and ids index-aligned so the
        # CLI can download ``teacher_paths`` and rewrite them to local dirs.
        self.domains, self.teacher_paths = _parse_teachers(self.teacher_models)

    def domain_to_teacher(self) -> Dict[str, str]:
        """Map each domain to its (possibly already-localized) teacher path."""
        return dict(zip(self.domains, self.teacher_paths))

    @property
    def default_domain(self) -> str:
        """Domain used for prompts lacking an explicit ``domain_field``."""
        return self.domains[0]


def _parse_teachers(spec: str) -> Tuple[List[str], List[str]]:
    """Parse "d1:id1,d2:id2" -> (["d1", "d2"], ["id1", "id2"]).

    A bare id without a "domain:" prefix is registered as the "default" domain.
    """
    domains: List[str] = []
    ids: List[str] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            domain, model_id = part.split(":", 1)
            domain, model_id = domain.strip(), model_id.strip()
        else:
            domain, model_id = "default", part
        if domain in domains:
            raise ValueError(f"Duplicate teacher domain '{domain}' in teacher_models='{spec}'.")
        domains.append(domain)
        ids.append(model_id)
    if not domains:
        raise ValueError(f"No teachers parsed from teacher_models='{spec}'.")
    return domains, ids

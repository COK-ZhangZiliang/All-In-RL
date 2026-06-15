"""Configuration for Proximal Policy Optimization (PPO) RLHF.

Only the fields unique to PPO live here; model loading, data, on-policy
sampling, the optimization loop and eval are inherited from
:class:`rl_common.BaseConfig`.

PPO is the classic four-model RLHF recipe:

  * **actor**     — the trainable policy (``student_model``);
  * **critic**    — a trainable value function (``critic_mode``);
  * **reference** — a frozen KL anchor (``ref_model``, defaults to the actor);
  * **reward**    — the scalar reward source (``reward_source``).

Two orthogonal choices are exposed because both were requested:

  * ``reward_source`` ∈ {``"model"``, ``"verifiable"``}: a frozen reward model
    (sequence classifier) vs. a rule-based verifiable reward (e.g. GSM8K answer
    matching, reading the ground truth from ``answer_field`` of the prompt file).
  * ``critic_mode`` ∈ {``"separate"``, ``"shared"``}: an independent critic
    backbone vs. a value head bolted onto the actor's hidden states.
"""

from __future__ import annotations

from dataclasses import dataclass

from rl_common import BaseConfig


@dataclass
class PPOConfig(BaseConfig):
    # -------------------------------------------------------------- models
    # Trainable actor / policy. Defaults to a small instruct model so a full
    # four-model PPO run is feasible on a single accelerator.
    student_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    # Frozen reference (KL anchor). Empty -> a frozen copy of the actor.
    ref_model: str = ""

    # ----------------------------------------------------------- reward
    # "model"      -> frozen sequence-classification reward model scores the
    #                 full response (classic RLHF).
    # "verifiable" -> rule-based reward comparing the completion against a
    #                 ground-truth ``answer_field`` carried by the prompt file.
    reward_source: str = "model"
    reward_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    # Prompt-file field holding the ground-truth answer (verifiable reward only).
    answer_field: str = "answer"

    # ----------------------------------------------------------- critic
    # "separate" -> independent trainable backbone + scalar value head.
    # "shared"   -> a value head reading the actor's last hidden states.
    critic_mode: str = "separate"
    # Backbone id for a separate critic (ignored when critic_mode="shared").
    # Empty -> initialize the critic backbone from the actor weights.
    critic_model: str = ""
    critic_learning_rate: float = 1e-5

    # ---------------------------------------------------------------- data
    dataset_recipe: str = "gsm8k"
    prompt_field: str = "question"

    # ------------------------------------------------------------ sampling
    max_new_tokens: int = 200
    temperature: float = 1.0
    num_generations: int = 1

    # -------------------------------------------------------------- loop
    output_dir: str = "outputs/ppo"
    num_train_steps: int = 200
    batch_size: int = 4
    learning_rate: float = 1e-6

    # --------------------------------------------------------- PPO objective
    # Multi-epoch reuse of each rollout buffer — the defining trait of PPO.
    ppo_epochs: int = 2
    # Minibatch size for the inner PPO update (over rollout sequences).
    ppo_mini_batch_size: int = 2
    clip_ratio: float = 0.2          # policy ratio clip epsilon
    value_clip: float = 0.2          # value function clip range
    vf_coef: float = 0.5             # value loss weight
    entropy_coef: float = 0.0        # entropy bonus weight
    gamma: float = 1.0               # discount (1.0 is standard for RLHF)
    gae_lambda: float = 0.95         # GAE smoothing
    kl_coef: float = 0.05            # per-token KL-to-reference penalty (beta)
    normalize_advantages: bool = True
    # Clip the per-sequence reward-model score to this magnitude (0 disables).
    reward_clip: float = 0.0

    def resolved_ref_model(self) -> str:
        return self.ref_model or self.student_model

    def resolved_critic_model(self) -> str:
        return self.critic_model or self.student_model

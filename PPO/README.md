# Proximal Policy Optimization (PPO)

A minimal, self-contained PyTorch + HuggingFace Transformers implementation of
**PPO for RLHF** — the classic four-model recipe (InstructGPT-style) built on
top of the shared [`rl_common`](../rl_common) scaffolding.

## Idea

PPO is an on-policy policy-gradient method that maximizes a reward while keeping
the updated policy close to the data-collecting policy via a **clipped surrogate
objective**. For LLM alignment the standard pipeline uses four models:

| Role | Trainable? | Job |
|------|------------|-----|
| **Actor** (`student_model`) | ✅ | the policy being optimized |
| **Critic** (`critic_mode`) | ✅ | per-token value `V(s_t)` baseline for GAE |
| **Reference** (`ref_model`) | ❄️ | KL anchor; defaults to a frozen copy of the actor |
| **Reward** (`reward_source`) | ❄️ | scalar score for the full response |

Each training step:

1. **Rollout** — the actor samples completions on-policy; record `old_logp`
   (actor at generation time), `ref_logp`, and per-token `values`.
2. **Reward shaping** — fold a per-token KL penalty into the reward and add the
   sequence-level score at the last response token:

   ```
   r_t = -kl_coef · (logp_actor − logp_ref)_t   (+ score at the last token)
   ```
3. **GAE** — turn token rewards + values into low-variance advantages & returns.
4. **PPO update** — `ppo_epochs` of minibatch SGD over the *same* rollout buffer,
   optimizing the **clipped surrogate** + **clipped value loss** (+ optional
   entropy bonus). Reusing each buffer for several epochs is what makes ratio
   clipping meaningful — the defining trait of PPO over vanilla A2C.

## Two configurable axes

* **`reward_source`**
  * `"model"` (default): a frozen sequence-classification **reward model**
    scores the full response (classic RLHF).
  * `"verifiable"`: a rule-based **verifiable reward** (RLVR) — decode the
    completion, extract the final answer, and compare it to the ground-truth
    `answer_field` from the prompt file (e.g. GSM8K). No reward model needed.
* **`critic_mode`**
  * `"separate"` (default): an independent trainable backbone + value head
    (OpenRLHF/TRL layout), with its own optimizer (`critic_learning_rate`).
  * `"shared"`: only a value head, reading the actor's last hidden states —
    saves a model's worth of memory at the cost of actor/critic coupling.

## Layout

| File | Responsibility |
|------|----------------|
| `ppo/config.py`  | PPO defaults + reward/critic/GAE/clip hyper-params (everything else inherited from `rl_common.BaseConfig`) |
| `ppo/value.py`   | `Critic`: separate backbone or shared value head |
| `ppo/reward.py`  | `ModelReward` / `VerifiableReward` + `build_reward` |
| `ppo/losses.py`  | **Core**: token rewards (KL-folded), GAE, clipped policy & value losses |
| `ppo/trainer.py` | `PPOTrainer(BaseTrainer)`: rollout buffer + multi-epoch minibatch PPO loop |
| `train.py`       | CLI entry point (conditional model downloads) |

Shared building blocks live in `rl_common/`: prompt datasets + collator,
model loading (actor / frozen reference & reward model / critic backbone),
on-policy rollouts, masked-mean / gather-logp / entropy primitives, the
`BaseTrainer` setup/eval/save, and the auto-flag CLI helpers.

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

Default — Qwen2.5-0.5B actor on GSM8K with a reward model and a separate critic:

```bash
python train.py
```

Verifiable reward (RLVR), no reward model — use GSM8K answer matching:

```bash
python train.py --reward_source verifiable --critic_mode separate
```

Shared-backbone critic to save memory:

```bash
python train.py --critic_mode shared
```

Override any field (every dataclass field is auto-exposed as a flag):

```bash
python train.py \
    --student_model Qwen/Qwen2.5-0.5B-Instruct \
    --reward_model Qwen/Qwen2.5-0.5B-Instruct \
    --num_train_steps 500 --batch_size 8 --max_new_tokens 256 \
    --ppo_epochs 4 --clip_ratio 0.2 --kl_coef 0.05
```

Or drive it from Python with your own prompts:

```python
from ppo import PPOConfig
from ppo.trainer import PPOTrainer

cfg = PPOConfig(reward_source="verifiable", num_train_steps=100, batch_size=4)
trainer = PPOTrainer(cfg, prompts=["What is 2+2?", "Compute 12*3."])
trainer.train()
```

## Logged metrics

Each step prints: `policy_loss`, `value_loss`, `entropy`, `clip_frac` (fraction
of ratios hitting the clip), `approx_kl` (old→new policy drift), `score` (mean
reward), `kl` (mean KL-to-reference), `grad_norm`, and `lr`.

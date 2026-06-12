# Multi-Teacher On-Policy Distillation (MOPD)

A minimal multi-teacher extension of On-Policy Distillation, implementing the
**Multi-Teacher On-Policy Distillation (MOPD)** post-training paradigm
introduced in **MiMo-V2-Flash**
([arXiv:2601.02780](https://arxiv.org/abs/2601.02780)) and subsequently used
by GLM-5, Nemotron-Cascade 2, DeepSeek-V4, and others to *consolidate* several
domain-specialized expert checkpoints back into one broad student.

## Idea

OPD distills a single frozen teacher into a student via the per-token
**reverse KL** on student-sampled rollouts. MOPD keeps the same per-token
objective but replaces the single teacher with a *teacher pool* — one
domain-specialized expert per capability (e.g. math / code / reasoning):

1. The **student** generates rollouts under its own policy (on-policy).
2. Each prompt carries a **`domain`** field; the rollout is scored by its
   **domain teacher** only — providing dense token-level supervision targeted
   at that capability.
3. The student minimizes the masked per-token reverse KL towards that teacher:

   ```
   L = E_{x ~ p_student} [ w(x) * D_KL( p_student(·|x) || p_teacher_d(·|x) ) ]
   ```

4. `w(x)` is a **truncated importance weight** correcting the mismatch between
   the sampling distribution (`temperature`) and the policy's true (T=1)
   distribution: `w_i = min( exp(Δlogp_i), cap )`. Set `importance_weight_cap=0`
   to disable (strict on-policy).

This unifies several specialized teachers into one student via dense token-level
signals, instead of mixing them at the data or sequence-level reward level.

## Layout

| File              | Responsibility                                                                                                                                 |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `mopd/config.py`  | MOPD-only fields (`teacher_models`, `domain_field`, `kl_temperature`, `importance_weight_cap`); the rest inherited from `rl_common.BaseConfig` |
| `mopd/losses.py`  | **Core**: multi-teacher reverse-KL with truncated importance weights                                                                           |
| `mopd/trainer.py` | Thin `MOPDTrainer(BaseTrainer)`: load teacher pool + domain-aware dataloaders + wire the objective                                             |
| `train.py`        | CLI entry point (extends `rl_common.cli` to download every teacher)                                                                            |

Shared building blocks live in [`rl_common`](../rl_common).

## Requirements

Every teacher in the pool **must share the student's tokenizer / vocabulary**.
This is enforced at startup by `check_vocab_compatibility`.

```bash
pip install -r requirements.txt
```

## Usage

The default teacher pool is the canonical **Qwen2.5 expert trio** —
math / code / general — all 7B Qwen2.5-family checkpoints that share the same
tokenizer as the student:

```bash
# Uses the default trio + Qwen2.5-1.5B-Instruct student.
python train.py --num_train_steps 200 --batch_size 2 --max_new_tokens 256
```

Override the pool via `domain:model_id` pairs (a bare id is registered under
the `default` domain). Each prompt's `domain` field selects its teacher; prompts
without one fall back to the first declared domain.

```bash
python train.py \
    --teacher_models math:Qwen/Qwen2.5-Math-7B-Instruct,code:Qwen/Qwen2.5-Coder-7B-Instruct,general:Qwen/Qwen2.5-7B-Instruct \
    --student_model Qwen/Qwen2.5-1.5B-Instruct \
    --num_train_steps 50 --batch_size 4 --max_new_tokens 256
```

Tighter VRAM? Drop to the Qwen2.5 1.5B trio with a 0.5B student — same
tokenizer, much smaller footprint:

```bash
python train.py \
    --teacher_models math:Qwen/Qwen2.5-Math-1.5B-Instruct,code:Qwen/Qwen2.5-Coder-1.5B-Instruct,general:Qwen/Qwen2.5-1.5B-Instruct \
    --student_model Qwen/Qwen2.5-0.5B-Instruct
```

Single-teacher fallback (recovers OPD):

```bash
python train.py \
    --teacher_models Qwen/Qwen2.5-7B-Instruct \
    --student_model Qwen/Qwen2.5-1.5B-Instruct
```

### Datasets — multi-domain prompt mixing

Raw dataset *recipes* (currently `gsm8k`, `mbpp`, `alpaca`) live in
[`rl_common/recipes.py`](../rl_common/recipes.py) and are reusable by every
algorithm in the repo. MOPD's algorithm-specific concern — *combining* them
with per-recipe ratios into a single jsonl with a `domain` field — lives in
[`mopd/data.py`](./mopd/data.py).

Without `--mix_recipes` the trainer falls back to the standard single-domain
GSM8K split (same as OPD). To exercise the math/code/general teacher trio,
pass a comma-separated list of recipes (and optionally a ratio):

```bash
python train.py \
    --mix_recipes gsm8k,mbpp,alpaca \
    --mix_ratios 2,1,1 \
    --mix_max_per_recipe 2000
```

This produces `<datasets_dir>/mopd_mix/{train,eval}.jsonl`; each line carries
the original prompt under `question`, plus `domain` (used by the trainer to
route each rollout to its teacher) and `_recipe` (for diagnostics).

Or drive it from Python with your own prompts:

```python
from mopd import MOPDConfig, MOPDTrainer

cfg = MOPDConfig(
    teacher_models=(
        "math:Qwen/Qwen2.5-Math-7B-Instruct,"
        "code:Qwen/Qwen2.5-Coder-7B-Instruct,"
        "general:Qwen/Qwen2.5-7B-Instruct"
    ),
    student_model="Qwen/Qwen2.5-1.5B-Instruct",
    num_train_steps=20, batch_size=2, max_new_tokens=64,
)
trainer = MOPDTrainer(cfg, prompts=["What is 2+2?", "Write a Python bubble sort."])
trainer.train()
```

## Logged metrics

Each step prints: `reverse_kl`, `teacher_nll` / `student_nll`,
`student_entropy` (collapse monitor), `is_weight_mean` (importance-weight
diagnostic), `grad_norm`, and `lr`.

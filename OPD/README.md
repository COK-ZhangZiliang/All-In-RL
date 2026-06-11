# On-Policy Distillation (OPD)

A minimal, self-contained PyTorch + HuggingFace Transformers implementation of
**On-Policy Distillation** for LLMs, following the recipe popularized by
[Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/).

## Idea

Classic (off-policy) distillation trains the student on sequences produced by
the teacher or a fixed dataset. The student then learns on a token distribution
it never actually visits at inference time, leading to *exposure bias*.

**On-Policy Distillation** fixes this:

1. The **student** generates rollouts **under its own policy** (on-policy).
2. The frozen **teacher** scores those exact same tokens.
3. The student minimizes the **per-token reverse KL** towards the teacher:

   ```
   L = E_{x ~ p_student} [ D_KL( p_student(·|x) || p_teacher(·|x) ) ]
   ```

Reverse KL is *mode-seeking*: the student is pushed to concentrate mass where
the teacher agrees, and because rollouts come from the student itself, training
and inference distributions match.

## Layout

OPD now keeps only its algorithm-specific code; everything generic (data,
model loading, on-policy sampling, the training/eval loop, the CLI) is shared
through the repo-root [`rl_common`](../rl_common) package.

| File | Responsibility |
|------|----------------|
| `opd/config.py`   | OPD-only hyper-parameters (`teacher_model`, `kl_temperature`); the rest inherited from `rl_common.BaseConfig` |
| `opd/losses.py`   | **Core**: masked per-token reverse-KL loss + vocab check |
| `opd/trainer.py`  | Thin `OPDTrainer(BaseTrainer)`: load teacher + wire the objective |
| `train.py`        | CLI entry point (delegates to `rl_common.cli.run`) |

Shared building blocks in `rl_common/`: `config.py` (`BaseConfig`),
`data.py` (prompt datasets + collator), `models.py` (load student / frozen
auxiliary models), `sampling.py` (on-policy rollouts), `functional.py`
(masked-mean / gather-logp / entropy primitives), `trainer.py` (`BaseTrainer`
loop + eval), `cli.py` (auto-generated flags + downloads).

## Requirements

The teacher and student **must share the same tokenizer / vocabulary**
(e.g. `Qwen2.5-3B-Instruct` → `Qwen2.5-0.5B-Instruct`). This is enforced at
startup by `check_vocab_compatibility`.

```bash
pip install -r requirements.txt
```

## Usage

Distill on a HuggingFace dataset:

```bash
python train.py \
    --teacher_model Qwen/Qwen2.5-3B-Instruct \
    --student_model Qwen/Qwen2.5-0.5B-Instruct \
    --dataset_name gsm8k --dataset_config main --prompt_field question \
    --num_train_steps 500 --batch_size 4 --max_new_tokens 256
```

Distill on a local prompt file (`.txt` one-per-line, or `.jsonl`):

```bash
python train.py --prompt_file prompts.txt --num_train_steps 200
```

Or drive it from Python with your own prompt list:

```python
from opd import OPDConfig
from opd.trainer import OPDTrainer

cfg = OPDConfig(num_train_steps=100, batch_size=2, max_new_tokens=128)
trainer = OPDTrainer(cfg, prompts=["What is 2+2?", "Explain gravity."])
trainer.train()
```

## Logged metrics

Each step prints: `reverse_kl`, `teacher_nll` / `student_nll` (NLL of the
sampled tokens), `student_entropy` (collapse monitor), `grad_norm`, and `lr`.

# AGENTS.md

Guidance for AI agents and contributors working in **All-In-RL**, a collection
of minimal, self-contained, readable implementations of modern RL / post-training
algorithms for LLMs.

## Prime Directive

Keep each algorithm **minimal and readable**. The value of this repo is that a
reader can understand a method from first principles. Do not add abstraction,
configurability, or error handling beyond what the current task needs.

## Architecture

The repo is split into one **shared library** and one **thin package per
algorithm**. Everything generic lives in `rl_common/`; each algorithm folder
contains only its own core.

```
All-In-RL/
├── rl_common/          # Shared scaffolding — reused by EVERY algorithm
│   ├── config.py       # BaseConfig: all generic hyper-params
│   ├── data.py         # Prompt datasets (HF/ModelScope/file/list) + left-pad collator
│   ├── models.py       # load_student / load_frozen (teacher/ref/reward) + dtype resolve
│   ├── sampling.py     # generate_rollouts + completion masking (on-policy algos)
│   ├── functional.py   # StepOutput + tensor primitives (masked_mean, gather_logp, entropy, log_softmax)
│   ├── trainer.py      # BaseTrainer: loop / optim / sched / clip / log / save / eval
│   └── cli.py          # run(config_cls, trainer_cls, ...): auto flags + model/data download
├── OPD/                # Example algorithm (On-Policy Distillation)
│   ├── opd/
│   │   ├── config.py   # OPDConfig(BaseConfig): only teacher_model + kl_temperature
│   │   ├── losses.py   # CORE: per_token_reverse_kl + check_vocab_compatibility
│   │   └── trainer.py  # OPDTrainer(BaseTrainer): ~15 lines
│   └── train.py        # CLI entry — delegates to rl_common.cli.run
└── README.md
```

### Division of responsibility

- **`rl_common`** **is algorithm-agnostic.** Never put algorithm-specific logic
  (a particular loss, a teacher, a reward model) into it. If something is useful
  to ≥2 algorithms, it belongs here.
- **An algorithm package contains only its core**: extra config fields, its loss
  function, and a thin trainer wiring the objective. Aim for the trainer to be a
  few dozen lines at most.

## The `BaseTrainer` contract

`rl_common.BaseTrainer` owns the entire training/eval/save loop. A new algorithm
subclasses it and implements **one required** method, plus **one optional** hook:

```python
class MyTrainer(BaseTrainer):
    eval_delta_keys = ("my_metric",)        # optional: keys reported as Δ vs baseline

    def setup_aux_models(self):             # OPTIONAL: load frozen teacher/ref/reward
        self.ref = load_frozen(self.cfg.ref_model, self.cfg)

    def compute_loss(self, batch) -> StepOutput:   # REQUIRED: the whole algorithm
        rollout = generate_rollouts(self.student, self.tokenizer, batch, self.cfg)
        ...
        return StepOutput(loss=loss, metrics={..., "num_completion_tokens": n})
```

Rules:

- `compute_loss` must return a `StepOutput(loss, metrics)`.
- `metrics` **must** include `num_completion_tokens` (the base trainer
  token-weights all metrics when aggregating over the eval set).
- The same `compute_loss` is reused for evaluation — do not write a separate eval
  loss. Off-policy algorithms simply skip `generate_rollouts`.
- Build per-token objectives from `rl_common.functional` primitives
  (`masked_mean`, `gather_logp`, `entropy_from_logp`, `log_softmax`,
  `next_token_logits`) instead of re-deriving them.

## Adding a new algorithm

1. `mkdir <ALGO>/<algo>/` and add `train.py` mirroring [OPD/train.py](./OPD/train.py)
   (it inserts the repo root on `sys.path` so `rl_common` imports work when run
   as a script).
2. `config.py`: `@dataclass class XConfig(BaseConfig)` — add **only** the fields
   unique to the algorithm.
3. `losses.py`: the core objective, returning `StepOutput`.
4. `trainer.py`: subclass `BaseTrainer`, implement `compute_loss` (+ optional
   `setup_aux_models`).
5. `train.py`: call `rl_common.cli.run(XConfig, XTrainer, project_root=..., model_fields=(...))`.
   List in `model_fields` every config field holding a model id to download
   (e.g. `("teacher_model", "student_model")`).
6. Add a row to the root [README.md](./README.md) table and a short `<ALGO>/README.md`.

CLI flags are auto-generated from the config dataclass fields — do not hand-write
`argparse` per algorithm.

## Conventions

- **Language**: code, comments, and docstrings are in **English**.
- **Comments**: explain *why*, not *what*. Only comment non-obvious logic. Do not
  add comments to code you didn't change.
- **Docstrings**: module-level docstring stating the role of the file; concise
  function docstrings for the public/core functions.
- **Style**: standard PEP 8, `from __future__ import annotations`, type hints on
  public functions. No new third-party dependencies without updating
  `requirements.txt` and a strong reason.
- **Defaults**: keep `BaseConfig` defaults laptop/CPU-friendly (small model,
  `float32`, `device="cpu"`) so any algorithm is smoke-testable offline.
- **No dead code / back-compat shims.** Delete what you replace.

## Verifying changes

There is no formal test suite. Before declaring a change done:

- Confirm imports resolve:
  ```
  cd <ALGO> && python3 -c "import sys; sys.path.insert(0,'..'); import <algo>.trainer"
  ```
- Run an **offline smoke test** of the full pipeline (rollout → loss → backward →
  optim step → save → eval) using tiny randomly-initialized models that share a
  minimal tokenizer, on CPU. Do not require network downloads for verification.
- Smoke-test scripts are throwaway: delete them after use, do not commit them.
- A real run is `python3 <ALGO>/train.py` (downloads models from ModelScope into
  `<ALGO>/models/` and GSM8K into `<ALGO>/datasets/`; both are gitignored).

## Git commits

- **Only commit when explicitly asked.** Never commit automatically after making
  changes.
- **Format**: Conventional Commits — `<type>(<scope>): <subject>`.
  - `type` ∈ `feat` | `fix` | `refactor` | `docs` | `test` | `chore`.
  - `scope` is the affected area: an algorithm name (`opd`), the shared library
    (`rl_common`), or a sub-module (`rl_common/trainer`). Omit if repo-wide.
  - `subject` is imperative mood, lower-case, no trailing period, ≤ 72 chars.
  - Examples: `feat(grpo): add group-relative advantage loss`,
    `refactor(rl_common): extract shared BaseTrainer`,
    `docs: add AGENTS.md`.
- **Scope of a commit**: one logical change. Do not mix a new algorithm with
  unrelated `rl_common` refactors in the same commit.
- **Never stage models or data.** Add files by explicit path; never
  `git add -A` / `git add .` (avoids committing `models/`, `datasets/`,
  throwaway smoke-test scripts, or local outputs).
- **Verify before committing.** Imports resolve and the offline smoke test
  passes (see *Verifying changes*).
- Do not commit `outputs/`, checkpoints, or `__pycache__/`.

## Environment notes

- Use `python3.10`
- `models/` and `datasets/` are gitignored — never commit weights or data.
- `rl_common.cli` redirects `$HOME/.modelscope` and uses `requests` (certifi CAs)
  for dataset downloads to work around macOS framework-Python TLS issues. Preserve
  these workarounds.


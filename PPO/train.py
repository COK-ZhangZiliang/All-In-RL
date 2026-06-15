"""Command-line entry point for PPO RLHF.

PPO has *conditional* auxiliary models, so this entry point composes the
reusable :mod:`rl_common.cli` helpers instead of the one-shot ``run()``:

  * the actor (``student_model``) is always downloaded;
  * the reference (``ref_model``) only when explicitly overridden;
  * the reward model only when ``reward_source="model"``;
  * a separate critic backbone only when ``critic_mode="separate"`` and
    ``critic_model`` is set.

All flags are still auto-generated from the config fields.
"""

from __future__ import annotations

import dataclasses
import os
import sys

# Make the repo-root ``rl_common`` package importable when run as a script.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PROJECT_ROOT)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rl_common.cli import (  # noqa: E402
    _setup_modelscope_home,
    auto_device,
    auto_dtype,
    build_arg_parser,
    default_dataset_setup,
    ensure_model,
)

from ppo.config import PPOConfig  # noqa: E402
from ppo.trainer import PPOTrainer  # noqa: E402


def main() -> None:
    _setup_modelscope_home(_PROJECT_ROOT)
    args = build_arg_parser(PPOConfig, _PROJECT_ROOT).parse_args()
    models_dir, datasets_dir = args.models_dir, args.datasets_dir
    skip_download = args.skip_download

    overrides = {
        k: v
        for k, v in vars(args).items()
        if v is not None and k not in {"models_dir", "datasets_dir", "skip_download"}
    }
    overrides.setdefault("device", auto_device())
    overrides.setdefault("torch_dtype", auto_dtype(overrides["device"]))
    cfg = PPOConfig(**overrides)

    print("=" * 64)
    print(f"PPOConfig | device={cfg.device} dtype={cfg.torch_dtype}")
    print(f"  reward_source={cfg.reward_source} critic_mode={cfg.critic_mode}")
    print("=" * 64)

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(datasets_dir, exist_ok=True)

    # Download only the models this configuration actually uses, and rewrite each
    # config field to its local directory.
    cfg.student_model = ensure_model(cfg.student_model, models_dir, skip=skip_download)
    if cfg.ref_model:
        cfg.ref_model = ensure_model(cfg.ref_model, models_dir, skip=skip_download)
    if cfg.reward_source == "model":
        cfg.reward_model = ensure_model(cfg.reward_model, models_dir, skip=skip_download)
    if cfg.critic_mode == "separate" and cfg.critic_model:
        cfg.critic_model = ensure_model(cfg.critic_model, models_dir, skip=skip_download)

    cfg.tokenizer_name = cfg.student_model
    cfg.use_modelscope = False

    default_dataset_setup(cfg, datasets_dir, skip_download)
    print(f"[data] train -> {cfg.prompt_file}")
    print(f"[data] eval  -> {cfg.eval_file}")

    print("-" * 64)
    for k, v in dataclasses.asdict(cfg).items():
        print(f"  {k}: {v}")
    print("-" * 64)

    PPOTrainer(cfg).train()


if __name__ == "__main__":
    main()

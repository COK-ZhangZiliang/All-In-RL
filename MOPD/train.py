"""Command-line entry point for Multi-Teacher On-Policy Distillation.

Thin wrapper around :func:`rl_common.cli.run`, customized in two ways:

  - ``model_list_fields=("teacher_paths",)`` downloads the whole teacher pool
    (one expert per domain) and rewrites the list to local directories.
  - a ``dataset_hook`` mixes several reusable recipes into one domain-tagged
    prompt file (the algorithm-specific concern lives in :mod:`mopd.data`).

All flags are auto-generated from the config fields.
"""

from __future__ import annotations

import os
import sys

# Make the repo-root ``rl_common`` package importable when run as a script.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PROJECT_ROOT)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rl_common.cli import (  # noqa: E402
    DEFAULT_PROMPT_FIELD,
    default_dataset_setup,
    run,
)

from mopd.config import MOPDConfig  # noqa: E402
from mopd.data import build_mix  # noqa: E402
from mopd.trainer import MOPDTrainer  # noqa: E402


def _mopd_dataset_setup(cfg: MOPDConfig, datasets_dir: str, skip_download: bool) -> None:
    """Mix several recipes into one domain-tagged prompt file (else single recipe)."""
    if not cfg.mix_recipes:
        default_dataset_setup(cfg, datasets_dir, skip_download)
        return
    out_dir = os.path.join(datasets_dir, "mopd_mix")
    train_path, eval_path = build_mix(
        recipes_spec=cfg.mix_recipes,
        datasets_dir=datasets_dir,
        out_dir=out_dir,
        out_field=DEFAULT_PROMPT_FIELD,
        ratios_spec=cfg.mix_ratios,
        max_per_recipe=cfg.mix_max_per_recipe,
        eval_per_recipe=cfg.mix_eval_per_recipe,
        seed=cfg.seed,
        skip_download=skip_download,
    )
    cfg.prompt_file = train_path
    cfg.prompt_field = DEFAULT_PROMPT_FIELD
    cfg.dataset_name = None
    if not cfg.eval_file:
        cfg.eval_file = eval_path


def main() -> None:
    run(
        MOPDConfig,
        MOPDTrainer,
        project_root=_PROJECT_ROOT,
        model_fields=("student_model",),
        model_list_fields=("teacher_paths",),
        dataset_hook=_mopd_dataset_setup,
    )


if __name__ == "__main__":
    main()

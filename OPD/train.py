"""Command-line entry point for On-Policy Distillation.

Thin wrapper around :func:`rl_common.cli.run`: it downloads the teacher/student
models and the GSM8K splits, builds an :class:`OPDConfig`, and runs
:class:`OPDTrainer`. All flags are auto-generated from the config fields.
"""

from __future__ import annotations

import os
import sys

# Make the repo-root ``rl_common`` package importable when run as a script.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PROJECT_ROOT)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rl_common.cli import run  # noqa: E402

from opd.config import OPDConfig  # noqa: E402
from opd.trainer import OPDTrainer  # noqa: E402


def main() -> None:
    run(
        OPDConfig,
        OPDTrainer,
        project_root=_PROJECT_ROOT,
        model_fields=("teacher_model", "student_model"),
    )


if __name__ == "__main__":
    main()

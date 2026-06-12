"""Shared command-line entry point for all algorithms.

``run(config_cls, trainer_cls, model_fields=...)`` builds an ``argparse`` flag
for every config field, pre-downloads the requested models from ModelScope into
``<algo>/models/`` and the dataset selected by ``cfg.dataset_recipe`` into
``<algo>/datasets/<recipe>/``, then constructs the config and runs the trainer.

Each algorithm's ``train.py`` is therefore a couple of lines: it just names its
config class, trainer class, and which config fields hold model ids.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from typing import Callable, Optional, Sequence, Tuple, Type

from .recipes import ensure_recipe, get_recipe

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _setup_modelscope_home(project_root: str) -> None:
    # ModelScope writes a session-id under ``$HOME/.modelscope``; on locked-down
    # macOS user dirs this raises PermissionError. Redirect HOME as a fallback.
    try:
        os.makedirs(os.path.join(os.path.expanduser("~"), ".modelscope"), exist_ok=True)
    except (OSError, PermissionError):
        os.environ["HOME"] = project_root


def auto_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    # MPS works for many ops but transformers training is most stable on CPU.
    return "cpu"


def auto_dtype(device: str) -> str:
    import torch

    if device == "cuda":
        return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    return "float32"


def _sanitize(model_id: str) -> str:
    """Make an id like ``Qwen/Qwen3-0.6B`` safe to use as a directory name."""
    return model_id.replace("/", "__")


def ensure_model(model_id: str, models_dir: str, skip: bool = False) -> str:
    """Make sure the model is present locally; return its directory."""
    local_dir = os.path.join(models_dir, _sanitize(model_id))
    if os.path.exists(os.path.join(local_dir, "config.json")):
        print(f"[model] cached: {model_id} -> {local_dir}", flush=True)
        return local_dir
    if skip:
        raise FileNotFoundError(f"--skip_download set but model not found at {local_dir}")

    os.makedirs(local_dir, exist_ok=True)
    from modelscope import snapshot_download

    print(f"[model] downloading {model_id} -> {local_dir}", flush=True)
    snapshot_download(model_id, local_dir=local_dir)
    return local_dir


GSM8K_RECIPE = "gsm8k"
DEFAULT_PROMPT_FIELD = "question"


def ensure_dataset(
    datasets_dir: str,
    skip: bool = False,
    recipe_name: str = GSM8K_RECIPE,
    prompt_field: str = DEFAULT_PROMPT_FIELD,
) -> Tuple[str, str]:
    """Ensure a single recipe's train/test jsonl files exist; return their paths.

    Defaults to GSM8K (so existing single-dataset algorithms like OPD keep
    working unchanged), but any recipe registered in :mod:`rl_common.recipes`
    can be requested by name.
    """
    paths = ensure_recipe(
        recipe_name, datasets_dir, out_field=prompt_field, skip_download=skip
    )
    recipe = get_recipe(recipe_name)
    if "train" not in paths:
        raise RuntimeError(f"Recipe '{recipe.name}' has no train split")
    train_path = paths["train"]
    test_path = paths.get("test", train_path)
    return train_path, test_path


def build_arg_parser(config_cls, project_root: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"{config_cls.__name__} training")
    p.add_argument(
        "--models_dir",
        type=str,
        default=os.path.join(project_root, "models"),
        help="Directory where models are downloaded.",
    )
    p.add_argument(
        "--datasets_dir",
        type=str,
        default=os.path.join(project_root, "datasets"),
        help="Directory where train/test jsonl files are saved.",
    )
    p.add_argument(
        "--skip_download",
        action="store_true",
        help="Do not download models or datasets; assume they are already on disk.",
    )

    # Auto-generate one --flag per config field; defaults stay in the dataclass
    # so flags only override when explicitly set (None -> drop).
    reserved = {"models_dir", "datasets_dir", "skip_download"}
    for f in dataclasses.fields(config_cls):
        if f.name == "extra" or f.name in reserved:
            continue
        ftype = f.type
        default = getattr(config_cls, f.name, None)
        if ftype == "bool" or isinstance(default, bool):
            # argparse's default `type=bool` mis-parses "False" as True.
            p.add_argument(f"--{f.name}", type=lambda x: x.lower() in ("1", "true", "yes"))
        elif ftype in ("int", "Optional[int]") or isinstance(default, int):
            p.add_argument(f"--{f.name}", type=int)
        elif ftype in ("float", "Optional[float]") or isinstance(default, float):
            p.add_argument(f"--{f.name}", type=float)
        else:
            p.add_argument(f"--{f.name}", type=str)
    return p


def default_dataset_setup(cfg, datasets_dir: str, skip_download: bool) -> None:
    """Default data wiring: materialize ``cfg.dataset_recipe`` train/test splits.

    Algorithms that need richer behaviour (e.g. mixing several recipes) pass a
    custom ``dataset_hook`` to :func:`run` instead.
    """
    train_path, test_path = ensure_dataset(
        datasets_dir, skip=skip_download, recipe_name=cfg.dataset_recipe
    )
    cfg.prompt_file = train_path
    cfg.prompt_field = DEFAULT_PROMPT_FIELD
    cfg.dataset_name = None
    if not cfg.eval_file:
        cfg.eval_file = test_path


def run(
    config_cls: Type,
    trainer_cls: Type,
    project_root: str,
    model_fields: Sequence[str] = ("student_model",),
    model_list_fields: Sequence[str] = (),
    dataset_hook: Optional[Callable[[object, str, bool], None]] = None,
) -> None:
    """Generic CLI: parse flags, materialize models/data, build cfg, train.

    ``model_fields`` lists config fields each holding a *single* model id to be
    downloaded and rewritten to its local directory (e.g. ``("teacher_model",
    "student_model")`` for distillation).

    ``model_list_fields`` lists config fields each holding a *list* of model ids
    (e.g. MOPD's ``teacher_paths``); every id is downloaded and the list is
    rewritten in place.

    ``dataset_hook(cfg, datasets_dir, skip_download)`` wires up the prompt/eval
    files. Defaults to :func:`default_dataset_setup` (a single recipe); pass a
    custom hook for multi-dataset mixing.
    """
    _setup_modelscope_home(project_root)

    args = build_arg_parser(config_cls, project_root).parse_args()
    models_dir = args.models_dir
    datasets_dir = args.datasets_dir
    skip_download = args.skip_download

    cfg_overrides = {
        k: v
        for k, v in vars(args).items()
        if v is not None and k not in {"models_dir", "datasets_dir", "skip_download"}
    }
    cfg_overrides.setdefault("device", auto_device())
    cfg_overrides.setdefault("torch_dtype", auto_dtype(cfg_overrides["device"]))

    cfg = config_cls(**cfg_overrides)

    print("=" * 64)
    print(config_cls.__name__)
    print(f"  device     : {cfg.device}")
    print(f"  dtype      : {cfg.torch_dtype}")
    print(f"  models_dir : {models_dir}")
    print(f"  data_dir   : {datasets_dir}")
    print(f"  skip_dl    : {skip_download}")
    print("=" * 64)

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(datasets_dir, exist_ok=True)

    # Materialize single-id model fields and rewrite to their local dirs.
    for field_name in model_fields:
        model_id = getattr(cfg, field_name)
        local_dir = ensure_model(model_id, models_dir, skip=skip_download)
        setattr(cfg, field_name, local_dir)
    # Materialize list-typed model fields (e.g. a pool of teachers).
    for field_name in model_list_fields:
        ids = getattr(cfg, field_name)
        setattr(
            cfg,
            field_name,
            [ensure_model(m, models_dir, skip=skip_download) for m in ids],
        )
    # Tokenizer follows the student.
    cfg.tokenizer_name = cfg.student_model
    cfg.use_modelscope = False

    (dataset_hook or default_dataset_setup)(cfg, datasets_dir, skip_download)
    print(f"[data] train -> {cfg.prompt_file}")
    print(f"[data] eval  -> {cfg.eval_file}  (periodic eval every {cfg.eval_every} steps)")

    print("-" * 64)
    for k, v in dataclasses.asdict(cfg).items():
        print(f"  {k}: {v}")
    print("-" * 64)

    trainer_cls(cfg).train()

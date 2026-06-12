"""Reusable dataset recipes shared by every algorithm.

A *recipe* is a tiny declarative entry that says how to materialize a public
prompt dataset into a uniform local jsonl, where every line is

    {"<output_field>": "<prompt text>", "domain": "<recipe.domain>"}

Sources are heterogenous (jsonl URLs, ModelScope datasets, ...), but the on-disk
layout produced by :func:`ensure_recipe` is uniform — algorithms only have to
read ``<output_field>`` (default: ``"question"``) and the optional ``"domain"``
field (used by multi-teacher distillation for routing).

Algorithm-specific concerns (mixing ratios, format rewrites, RL templates) stay
in the algorithm's own package; this module only owns the *raw ingredients*.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class DatasetRecipe:
    """Declarative description of how to fetch one prompt dataset.

    Exactly one of ``urls`` or ``modelscope_id`` must be set.

      - ``urls``: ``{"train": jsonl_url, "test": jsonl_url}``-style direct
        downloads. Each jsonl line must already be a dict containing
        ``prompt_field``.
      - ``modelscope_id``: a ModelScope dataset id resolved via
        ``MsDataset.load``. The ``prompt_field`` of every item is then written
        into a local jsonl line (other columns are dropped).
    """

    name: str
    domain: str
    prompt_field: str
    splits: Tuple[str, ...] = ("train", "test")
    urls: Dict[str, str] = field(default_factory=dict)
    modelscope_id: Optional[str] = None
    modelscope_subset: Optional[str] = None


# A small, deliberately-curated registry. Add new entries here so every
# algorithm in the repo can use them without touching shared CLI code.
RECIPES: Dict[str, DatasetRecipe] = {
    # Math word problems — used by OPD as the default and by MOPD as the math
    # domain. Direct jsonl mirror, no ModelScope round-trip needed.
    "gsm8k": DatasetRecipe(
        name="gsm8k",
        domain="math",
        prompt_field="question",
        splits=("train", "test"),
        urls={
            "train": "https://sail-moe.oss-cn-hangzhou.aliyuncs.com/open_data/gsm8k/train.jsonl",
            "test": "https://sail-moe.oss-cn-hangzhou.aliyuncs.com/open_data/gsm8k/test.jsonl",
        },
    ),
    # Python coding prompts — sanitized MBPP from ModelScope.
    "mbpp": DatasetRecipe(
        name="mbpp",
        domain="code",
        prompt_field="text",
        splits=("train", "test"),
        modelscope_id="AI-ModelScope/mbpp",
        modelscope_subset="sanitized",
    ),
    # General-purpose instructions — Alpaca only ships a train split.
    "alpaca": DatasetRecipe(
        name="alpaca",
        domain="general",
        prompt_field="instruction",
        splits=("train",),
        modelscope_id="AI-ModelScope/alpaca-gpt4-data-en",
    ),
}


def list_recipes() -> List[str]:
    return list(RECIPES.keys())


def get_recipe(name: str) -> DatasetRecipe:
    if name not in RECIPES:
        raise KeyError(
            f"Unknown dataset recipe '{name}'. Known recipes: {list_recipes()}"
        )
    return RECIPES[name]


# --------------------------------------------------------------------- helpers
def _download_url(url: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    print(f"[data] downloading {url}", flush=True)
    # ``requests`` carries certifi's CA bundle; macOS Python.framework's stdlib
    # openssl is missing system root certs, breaking urllib over HTTPS.
    import requests

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    os.replace(tmp, dst)


def _materialize_url_jsonl(
    src_url: str, dst_path: str, src_field: str, out_field: str, domain: str
) -> None:
    """Download ``src_url`` and rewrite each line as ``{out_field, domain}``."""
    raw_path = dst_path + ".raw"
    _download_url(src_url, raw_path)
    with open(raw_path, "r", encoding="utf-8") as fin, open(
        dst_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict) or src_field not in obj:
                continue
            fout.write(
                json.dumps(
                    {out_field: str(obj[src_field]), "domain": domain},
                    ensure_ascii=False,
                )
                + "\n"
            )
    os.remove(raw_path)


def _materialize_modelscope_split(
    recipe: DatasetRecipe, split: str, dst_path: str, out_field: str
) -> None:
    from modelscope.msdatasets import MsDataset

    print(
        f"[data] modelscope: {recipe.modelscope_id} "
        f"(subset={recipe.modelscope_subset}, split={split}) -> {dst_path}",
        flush=True,
    )
    kwargs = dict(dataset_name=recipe.modelscope_id, split=split)
    if recipe.modelscope_subset:
        kwargs["subset_name"] = recipe.modelscope_subset
    ds = MsDataset.load(**kwargs)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    written = 0
    with open(dst_path, "w", encoding="utf-8") as f:
        for item in ds:
            if recipe.prompt_field not in item:
                raise KeyError(
                    f"prompt_field '{recipe.prompt_field}' missing from "
                    f"{recipe.modelscope_id} item keys {list(item.keys())}"
                )
            f.write(
                json.dumps(
                    {
                        out_field: str(item[recipe.prompt_field]),
                        "domain": recipe.domain,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
    print(f"[data] saved {written} prompts -> {dst_path}", flush=True)


# --------------------------------------------------------------------- public
def ensure_recipe(
    name: str,
    datasets_dir: str,
    out_field: str = "question",
    skip_download: bool = False,
) -> Dict[str, str]:
    """Materialize ``name`` under ``<datasets_dir>/<name>/<split>.jsonl``.

    Returns a mapping of split -> local path. Each output line carries
    ``{out_field, "domain"}`` so downstream code can read both algorithm-agnostic
    text and the recipe's domain label uniformly.
    """
    recipe = get_recipe(name)
    out_dir = os.path.join(datasets_dir, recipe.name)
    paths: Dict[str, str] = {}

    for split in recipe.splits:
        out_path = os.path.join(out_dir, f"{split}.jsonl")
        paths[split] = out_path
        if os.path.exists(out_path):
            print(f"[data] cached: {out_path}", flush=True)
            continue
        if skip_download:
            raise FileNotFoundError(
                f"--skip_download set but {recipe.name}/{split}.jsonl missing"
            )

        if recipe.urls and split in recipe.urls:
            _materialize_url_jsonl(
                recipe.urls[split],
                out_path,
                src_field=recipe.prompt_field,
                out_field=out_field,
                domain=recipe.domain,
            )
        elif recipe.modelscope_id:
            _materialize_modelscope_split(recipe, split, out_path, out_field)
        else:
            raise RuntimeError(
                f"Recipe '{recipe.name}' has no source for split '{split}'"
            )
        print(f"[data] saved -> {out_path}", flush=True)

    return paths

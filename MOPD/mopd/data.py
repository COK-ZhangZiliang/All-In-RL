"""MOPD-specific dataset mixing.

The *raw ingredients* — math (gsm8k), code (mbpp), general (alpaca), ... —
live in :mod:`rl_common.recipes` and are reusable by any algorithm. This
module owns the MOPD-specific concern of *combining* several recipes into one
shuffled jsonl with a ``domain`` field on each line, optionally with
per-recipe ratios and caps. The output file is just a plain jsonl, so the rest
of the trainer (and ``rl_common`` data loaders) treats it like any other
prompt file.
"""

from __future__ import annotations

import json
import os
import random
from typing import List, Optional, Sequence, Tuple

from rl_common.recipes import ensure_recipe, get_recipe


def _parse_csv(spec: str) -> List[str]:
    return [p.strip() for p in spec.split(",") if p.strip()]


def _read_jsonl(path: str) -> List[dict]:
    out: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _materialize_recipes(
    recipe_names: Sequence[str],
    datasets_dir: str,
    out_field: str,
    skip_download: bool,
    split: str,
) -> List[Tuple[str, str, List[dict]]]:
    """Download each recipe and load the requested split into memory.

    Returns a list of ``(recipe_name, domain, items)``. If a recipe lacks the
    requested split (e.g. Alpaca has no ``test``), it is silently skipped.
    """
    out: List[Tuple[str, str, List[dict]]] = []
    for name in recipe_names:
        paths = ensure_recipe(
            name, datasets_dir, out_field=out_field, skip_download=skip_download
        )
        if split not in paths:
            print(f"[mopd] recipe '{name}' has no '{split}' split, skipping")
            continue
        items = _read_jsonl(paths[split])
        recipe = get_recipe(name)
        out.append((name, recipe.domain, items))
    return out


def _take(items: List[dict], n: Optional[int], rng: random.Random) -> List[dict]:
    if n is None or n <= 0 or n >= len(items):
        return list(items)
    return rng.sample(items, n)


def _ratios(spec: str, k: int) -> List[float]:
    if not spec:
        return [1.0] * k
    parts = [float(x) for x in _parse_csv(spec)]
    if len(parts) != k:
        raise ValueError(
            f"mix_ratios has {len(parts)} entries but mix_recipes has {k}"
        )
    if any(r < 0 for r in parts):
        raise ValueError(f"mix_ratios must be non-negative, got: {parts}")
    if sum(parts) == 0:
        raise ValueError("mix_ratios sum to zero")
    return parts


def write_mixed_jsonl(
    recipe_names: Sequence[str],
    datasets_dir: str,
    out_path: str,
    out_field: str = "question",
    ratios_spec: str = "",
    max_per_recipe: int = 0,
    split: str = "train",
    seed: int = 0,
    skip_download: bool = False,
) -> str:
    """Materialize and shuffle a mix of recipes into a single jsonl.

    Each output line is ``{out_field, "domain", "_recipe"}``. Sampling per
    recipe is proportional to ``ratios_spec`` (after applying ``max_per_recipe``
    as a hard cap). The result is shuffled with ``seed`` so domains are
    interleaved.
    """
    rng = random.Random(seed)
    loaded = _materialize_recipes(
        recipe_names, datasets_dir, out_field, skip_download, split
    )
    if not loaded:
        raise RuntimeError(
            f"No recipes produced a '{split}' split for {list(recipe_names)}"
        )
    ratios = _ratios(ratios_spec, len(loaded))

    # Cap each recipe pool first, then take ``ratio * scale`` from it.
    pools: List[List[dict]] = []
    domains: List[str] = []
    sizes: List[int] = []
    for (_, domain, items), _r in zip(loaded, ratios):
        pool = _take(
            items, max_per_recipe if max_per_recipe > 0 else None, rng
        )
        pools.append(pool)
        domains.append(domain)
        sizes.append(len(pool))

    # Pick the largest "ratio-respecting" cut: scale = min over i of size_i / r_i.
    scale = min(s / r for s, r in zip(sizes, ratios) if r > 0)
    targets = [int(round(r * scale)) for r in ratios]
    targets = [min(t, s) for t, s in zip(targets, sizes)]

    merged: List[dict] = []
    for pool, target, domain, (name, _, _) in zip(pools, targets, domains, loaded):
        chosen = _take(pool, target, rng)
        for ex in chosen:
            ex.setdefault(out_field, "")
            ex["domain"] = domain
            ex["_recipe"] = name
            merged.append(ex)

    rng.shuffle(merged)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in merged:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(
        "[mopd] mixed dataset -> "
        + out_path
        + "  ("
        + ", ".join(
            f"{name}:{t}" for (name, _, _), t in zip(loaded, targets)
        )
        + f", total={len(merged)})",
        flush=True,
    )
    return out_path


def build_mix(
    recipes_spec: str,
    datasets_dir: str,
    out_dir: str,
    out_field: str,
    ratios_spec: str,
    max_per_recipe: int,
    eval_per_recipe: int,
    seed: int,
    skip_download: bool,
) -> Tuple[str, str]:
    """High-level convenience: build train+eval jsonl files from a recipe spec.

    Returns ``(train_path, eval_path)``. The eval split is built from each
    recipe's ``test`` split when available, falling back to held-out samples
    of its ``train`` split otherwise.
    """
    names = _parse_csv(recipes_spec)
    if not names:
        raise ValueError(f"Empty mix_recipes spec: '{recipes_spec}'")

    train_path = os.path.join(out_dir, "train.jsonl")
    eval_path = os.path.join(out_dir, "eval.jsonl")

    write_mixed_jsonl(
        names,
        datasets_dir,
        out_path=train_path,
        out_field=out_field,
        ratios_spec=ratios_spec,
        max_per_recipe=max_per_recipe,
        split="train",
        seed=seed,
        skip_download=skip_download,
    )

    # Eval: small per-recipe slice from the test split if available, else from
    # train. This keeps eval covering every domain even if some recipes
    # (Alpaca) are train-only.
    eval_items: List[dict] = []
    rng = random.Random(seed + 1)
    for name in names:
        recipe = get_recipe(name)
        paths = ensure_recipe(
            name, datasets_dir, out_field=out_field, skip_download=skip_download
        )
        src = paths.get("test") or paths.get("train")
        items = _read_jsonl(src)
        cap = eval_per_recipe if eval_per_recipe > 0 else 64
        chosen = _take(items, cap, rng)
        for ex in chosen:
            ex.setdefault(out_field, "")
            ex["domain"] = recipe.domain
            ex["_recipe"] = name
            eval_items.append(ex)

    rng.shuffle(eval_items)
    os.makedirs(os.path.dirname(eval_path), exist_ok=True)
    with open(eval_path, "w", encoding="utf-8") as f:
        for ex in eval_items:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[mopd] eval mix -> {eval_path}  (total={len(eval_items)})", flush=True)

    return train_path, eval_path


__all__ = ["build_mix", "write_mixed_jsonl"]

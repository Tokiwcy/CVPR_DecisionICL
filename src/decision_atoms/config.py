from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


DEFAULTS: dict[str, Any] = {
    "seed": 17,
    "labels": [],
    "boundary": {
        "prompt_repeats": 1,
        "require_prompt_stability": False,
        "pool_per_side": 80,
        "top_confusion_pairs": 12,
    },
    "discovery": {
        "modes": ["random", "easy", "boundary"],
        "bootstraps": 10,
        "support_per_side": 6,
        "atoms_per_bootstrap": 8,
    },
    "validation": {
        "prompt_variants": 3,
        "reliability_threshold": 0.5,
        "min_stability": 0.8,
        "min_coverage": 0.2,
        "min_auc": 0.6,
        "min_alignment": 0.05,
        "merge_correlation": 0.9,
    },
    "retrieval": {
        "source_mode": "boundary",
        "top_m_per_class": 20,
        "k_per_class": 2,
        "methods": ["random", "similarity", "decision_atom", "similarity_decision_atom"],
        "reliability_weighted": True,
    },
    "evaluation": {
        "query_split": "test",
        "method_order": [
            "zero_shot",
            "similarity",
            "similarity_decision_atom",
            "decision_atom",
            "random",
        ],
        "checkpoint_batch": 8,
    },
    "runtime": {
        "max_train": None,
        "max_validation": None,
        "max_test": None,
        "boundary_checkpoint_batch": 16,
        "atom_checkpoint_batch": 16,
    },
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    config = _merge(DEFAULTS, raw)
    config["_config_path"] = str(source)
    for key in ("manifest", "output_dir"):
        value = Path(config[key])
        if not value.is_absolute():
            value = (source.parent / value).resolve()
        config[key] = str(value)
    for section in ("target_model", "similarity_model"):
        model = config.get(section, {}).get("model")
        if model:
            candidate = (source.parent / str(model)).resolve()
            if candidate.exists():
                config[section]["model"] = str(candidate)
        cache_dir = config.get(section, {}).get("cache_dir")
        if cache_dir:
            cache_path = Path(str(cache_dir))
            if not cache_path.is_absolute():
                cache_path = (source.parent / cache_path).resolve()
            config[section]["cache_dir"] = str(cache_path)
    if not config["labels"] or len(config["labels"]) < 2:
        raise ValueError("config.labels must contain at least two labels in canonical order")
    if len(set(config["labels"])) != len(config["labels"]):
        raise ValueError("config.labels contains duplicates")
    return config

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_waterbirds_retriever_baselines import (
    AUXILIARY_PROMPT_ORDER,
    REFERENCE_PROMPT_ORDER,
    class_balanced_rices_indices,
    deterministic_random_indices,
    diagnostic_only_indices,
    mmices_indices,
)


def test_auxiliary_order_is_not_claimed_as_reference_matched() -> None:
    assert AUXILIARY_PROMPT_ORDER != REFERENCE_PROMPT_ORDER


def test_deterministic_random_indices_are_stable_and_unique() -> None:
    first = deterministic_random_indices("q", 20, 5, seed=7)
    second = deterministic_random_indices("q", 20, 5, seed=7)
    assert first.tolist() == second.tolist()
    assert len(set(first.tolist())) == 5


def test_class_balanced_rices_uses_both_classes_at_three_shots() -> None:
    similarities = np.asarray([0.9, 0.8, 0.7, 0.95, 0.6])
    labels = ["a", "a", "a", "b", "b"]
    indices = class_balanced_rices_indices(
        similarities, labels, ["0", "1", "2", "3", "4"], k=3
    )
    selected_labels = [labels[index] for index in indices]
    assert selected_labels.count("a") == 1
    assert selected_labels.count("b") == 2
    assert indices.tolist() == [3, 0, 4]


def test_diagnostic_only_prioritizes_score_then_similarity() -> None:
    indices = diagnostic_only_indices(
        np.asarray([0.5, 0.9, 0.9, 0.1]),
        ["a", "b", "c", "d"],
        k=2,
    )
    assert set(indices.tolist()) == {1, 2}
    assert indices.tolist() == [1, 2]


def test_mmices_filters_by_image_then_reranks_by_task_affinity() -> None:
    similarities = np.linspace(1.0, 0.1, 20)
    task_affinity = np.linspace(0.0, 0.9, 20)
    task_affinity[19] = 10.0  # excluded because it misses the image-filter pool
    indices = mmices_indices(
        similarities,
        task_affinity,
        [str(index) for index in range(20)],
        k=2,
        candidate_pool_size=3,
    )
    assert indices.tolist() == [15, 14]

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.stats import binomtest


def accuracy_metrics(labels: Sequence[str], predictions: Sequence[str]) -> dict[str, float]:
    if len(labels) != len(predictions) or not labels:
        raise ValueError("labels and predictions must have equal non-zero length")
    truth = np.asarray(labels)
    guessed = np.asarray(predictions)
    correct = truth == guessed
    per_class = [float(correct[truth == label].mean()) for label in sorted(set(truth.tolist()))]
    return {"accuracy": float(correct.mean()), "macro_accuracy": float(np.mean(per_class))}


def paired_comparison(
    *,
    labels: Sequence[str],
    baseline_predictions: Sequence[str],
    method_predictions: Sequence[str],
    bootstrap_samples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    if not (len(labels) == len(baseline_predictions) == len(method_predictions)) or not labels:
        raise ValueError("paired arrays must have equal non-zero length")
    truth = np.asarray(labels)
    baseline_correct = np.asarray(baseline_predictions) == truth
    method_correct = np.asarray(method_predictions) == truth
    rescue = int(np.sum((~baseline_correct) & method_correct))
    harm = int(np.sum(baseline_correct & (~method_correct)))
    difference = method_correct.astype(np.float64) - baseline_correct.astype(np.float64)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(bootstrap_samples, dtype=np.float64)
    for start in range(0, bootstrap_samples, 500):
        count = min(500, bootstrap_samples - start)
        indices = rng.integers(0, len(difference), size=(count, len(difference)))
        bootstrap[start : start + count] = difference[indices].mean(axis=1)
    discordant = rescue + harm
    p_value = 1.0 if discordant == 0 else float(binomtest(min(rescue, harm), discordant, 0.5).pvalue)
    return {
        "rescue": rescue,
        "harm": harm,
        "accuracy_delta": float(difference.mean()),
        "paired_bootstrap_95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "exact_mcnemar_p": p_value,
        "n": len(labels),
    }

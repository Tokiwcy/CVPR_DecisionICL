"""Small, array-level reference for the DecisionICL retrieval rule.

This module does not replace the hash-bound GPU runners in ``frozen/``.  It
exposes their class-prototype, validation calibration, and full-pool selection
steps with publication-facing names and no model or dataset dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


ALPHA_GRID = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)
EPSILON = 1e-8


@dataclass(frozen=True)
class DecisionICLCalibration:
    """Parent mass and per-group factor masses fitted on validation only."""

    parent_mass: float
    weights: np.ndarray
    validation_accuracy: float
    oof_accuracy: float
    fold_parent_masses: tuple[float, ...]


def class_factor_prototypes(
    train_labels: Sequence[str], train_group_probabilities: Sequence[np.ndarray]
) -> tuple[tuple[str, ...], tuple[np.ndarray, ...]]:
    """Average frozen visual factor-state probabilities for every train class."""
    labels = tuple(sorted(set(train_labels)))
    if len(labels) < 2 or not train_group_probabilities:
        raise ValueError("at least two classes and one factor group are required")
    truth = np.asarray(train_labels)
    if len(truth) == 0:
        raise ValueError("empty training pool")
    output = []
    for group in train_group_probabilities:
        values = np.asarray(group, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != len(truth):
            raise ValueError("factor evidence must align with the training pool")
        if np.any(values < 0) or not np.all(np.isfinite(values)):
            raise ValueError("factor evidence must be finite and nonnegative")
        if not np.allclose(values.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("factor-state probabilities must sum to one")
        means = np.stack([values[truth == label].mean(axis=0) for label in labels])
        means /= means.sum(axis=1, keepdims=True).clip(min=EPSILON)
        output.append(means.astype(np.float32))
    return labels, tuple(output)


def factor_class_similarities(
    query_group_probabilities: Sequence[np.ndarray], prototypes: Sequence[np.ndarray]
) -> np.ndarray:
    """One minus normalized Jensen–Shannon divergence, query × class × group."""
    if not prototypes or len(query_group_probabilities) != len(prototypes):
        raise ValueError("query evidence and class prototypes must share groups")
    result = []
    for query_group, class_group in zip(query_group_probabilities, prototypes):
        left = np.asarray(query_group, dtype=np.float64)
        right = np.asarray(class_group, dtype=np.float64)
        if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
            raise ValueError("factor-state dimensions differ")
        if (np.any(left < 0) or np.any(right < 0) or
                not np.all(np.isfinite(left)) or not np.all(np.isfinite(right))):
            raise ValueError("invalid factor probabilities")
        if not np.allclose(left.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("query probabilities must sum to one")
        a = left[:, None, :].clip(min=EPSILON)
        b = right[None, :, :].clip(min=EPSILON)
        middle = (a + b) / 2
        divergence = 0.5 * np.sum(a * np.log(a / middle), axis=2)
        divergence += 0.5 * np.sum(b * np.log(b / middle), axis=2)
        result.append(np.clip(1.0 - divergence / np.log(2.0), 0.0, 1.0))
    return np.stack(result, axis=2).astype(np.float32)


def class_features(
    parent_scores: np.ndarray, train_labels: Sequence[str],
    class_labels: Sequence[str], factor_similarities: np.ndarray,
) -> np.ndarray:
    """Standardize parent class maxima and factor evidence across classes."""
    scores = np.asarray(parent_scores, dtype=np.float32)
    factors = np.asarray(factor_similarities, dtype=np.float32)
    if scores.ndim != 2 or scores.shape[1] != len(train_labels):
        raise ValueError("parent scores must cover the full training pool")
    if factors.shape[:2] != (scores.shape[0], len(class_labels)) or factors.ndim != 3:
        raise ValueError("factor evidence must be query × class × group")
    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(factors)):
        raise ValueError("non-finite class features")
    labels = np.asarray(train_labels)
    parent_class = np.empty((len(scores), len(class_labels)), dtype=np.float32)
    for index, label in enumerate(class_labels):
        candidates = np.flatnonzero(labels == label)
        if not len(candidates):
            raise ValueError("class has no training examples")
        parent_class[:, index] = scores[:, candidates].max(axis=1)
    raw = np.concatenate((parent_class[:, :, None], factors), axis=2)
    center = raw.mean(axis=1, keepdims=True)
    scale = raw.std(axis=1, keepdims=True).clip(min=1e-6)
    return ((raw - center) / scale).astype(np.float32)


def weights_for_parent_mass(feature_count: int, parent_mass: float) -> np.ndarray:
    if feature_count < 2 or not 0 <= parent_mass <= 1:
        raise ValueError("invalid factor count or parent mass")
    weights = np.empty(feature_count, dtype=np.float32)
    weights[0] = parent_mass
    weights[1:] = (1.0 - parent_mass) / (feature_count - 1)
    return weights


def calibrate_decisionicl(
    validation_features: np.ndarray, validation_class_indices: Sequence[int],
    fold_assignments: Sequence[int],
) -> DecisionICLCalibration:
    """Five-fold OOF check and full-validation fit; never reads test labels."""
    features = np.asarray(validation_features, dtype=np.float32)
    truth = np.asarray(validation_class_indices, dtype=np.int64)
    folds = np.asarray(fold_assignments, dtype=np.int64)
    if features.ndim != 3 or features.shape[0] != len(truth) or len(folds) != len(truth):
        raise ValueError("validation arrays do not align")
    if not np.array_equal(np.unique(folds), np.arange(5)):
        raise ValueError("exactly five nonempty validation folds are required")
    if np.any(truth < 0) or np.any(truth >= features.shape[1]):
        raise ValueError("validation label index is outside the class set")

    def fit(rows: np.ndarray) -> tuple[float, float]:
        best_mass, best_accuracy = 1.0, -1.0
        # The frozen tie rule prefers a larger parent mass.
        for mass in reversed(ALPHA_GRID):
            weights = weights_for_parent_mass(features.shape[2], mass)
            predicted = np.einsum("qcf,f->qc", features[rows], weights).argmax(axis=1)
            accuracy = float(np.mean(predicted == truth[rows]))
            if accuracy > best_accuracy + 1e-12:
                best_mass, best_accuracy = mass, accuracy
        return best_mass, best_accuracy

    oof = np.empty(len(truth), dtype=np.int64)
    fold_masses = []
    for fold in range(5):
        training = np.flatnonzero(folds != fold)
        heldout = np.flatnonzero(folds == fold)
        mass, _ = fit(training)
        fold_masses.append(mass)
        oof[heldout] = np.einsum(
            "qcf,f->qc", features[heldout],
            weights_for_parent_mass(features.shape[2], mass),
        ).argmax(axis=1)
    mass, accuracy = fit(np.arange(len(truth)))
    return DecisionICLCalibration(
        mass, weights_for_parent_mass(features.shape[2], mass), accuracy,
        float(np.mean(oof == truth)), tuple(fold_masses),
    )


def select_decisionicl_demonstrations(
    parent_scores: np.ndarray, train_ids: Sequence[str], train_labels: Sequence[str],
    class_labels: Sequence[str], features: np.ndarray, weights: np.ndarray, shots: int,
) -> tuple[tuple[str, ...], ...]:
    """Choose a class, then rank that class over the complete train pool."""
    scores = np.asarray(parent_scores)
    values = np.asarray(features)
    mass = np.asarray(weights)
    if scores.shape != (values.shape[0], len(train_ids)) or len(train_ids) != len(train_labels):
        raise ValueError("query/train arrays do not align")
    if values.ndim != 3 or values.shape[1] != len(class_labels) or values.shape[2] != len(mass):
        raise ValueError("class score features do not align")
    if shots < 1:
        raise ValueError("shots must be positive")
    selected_class = np.einsum("qcf,f->qc", values, mass).argmax(axis=1)
    train_y = np.asarray(train_labels)
    output = []
    for row, cls in enumerate(selected_class):
        candidates = np.flatnonzero(train_y == class_labels[int(cls)])
        if len(candidates) < shots:
            raise ValueError("selected class has fewer train examples than shots")
        # Same top-k operation as the frozen runner: global train scores,
        # restricted only after the class has been selected.
        chosen = candidates[np.argsort(scores[row, candidates])[-shots:][::-1]]
        output.append(tuple(train_ids[int(index)] for index in chosen))
    return tuple(output)

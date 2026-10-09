from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import nnls

from .evidence import EvidenceStore
from .schema import Sample


EPSILON = 1e-8


@dataclass(frozen=True)
class ClassAtomPrototypes:
    labels: tuple[str, ...]
    group_ids: tuple[str, ...]
    probabilities: tuple[np.ndarray, ...]

    def validate(self) -> None:
        if not self.labels or len(self.labels) != len(set(self.labels)):
            raise ValueError("prototype labels must be non-empty and unique")
        if len(self.probabilities) != len(self.group_ids):
            raise ValueError("prototype groups do not align")
        for values in self.probabilities:
            array = np.asarray(values)
            if array.ndim != 2 or array.shape[0] != len(self.labels):
                raise ValueError("prototype probability shape mismatch")
            if np.any(array < 0) or not np.all(np.isfinite(array)):
                raise ValueError("prototype probabilities are invalid")
            if not np.allclose(array.sum(axis=1), 1.0, atol=1e-5):
                raise ValueError("prototype probabilities must sum to one")


def build_class_atom_prototypes(
    *, samples: Sequence[Sample], evidence: EvidenceStore
) -> ClassAtomPrototypes:
    if tuple(sample.sample_id for sample in samples) != evidence.sample_ids:
        raise ValueError("samples and evidence do not align")
    labels = tuple(sorted({sample.label for sample in samples}))
    sample_labels = np.asarray([sample.label for sample in samples])
    groups: list[np.ndarray] = []
    for group_index in range(len(evidence.group_ids)):
        values = evidence.group_probability(group_index).astype(np.float64)
        means = np.stack([values[sample_labels == label].mean(axis=0) for label in labels])
        means /= means.sum(axis=1, keepdims=True).clip(min=EPSILON)
        groups.append(means.astype(np.float32))
    result = ClassAtomPrototypes(labels, evidence.group_ids, tuple(groups))
    result.validate()
    return result


def jensen_shannon_group_similarities(
    *, query_evidence: EvidenceStore, prototypes: ClassAtomPrototypes
) -> np.ndarray:
    prototypes.validate()
    if query_evidence.group_ids != prototypes.group_ids:
        raise ValueError("query evidence and prototypes use different groups")
    output = np.empty(
        (len(query_evidence.sample_ids), len(prototypes.labels), len(prototypes.group_ids)),
        dtype=np.float32,
    )
    for group_index, class_probability in enumerate(prototypes.probabilities):
        query_probability = query_evidence.group_probability(group_index).astype(np.float64)
        left = query_probability[:, None, :].clip(min=EPSILON)
        right = class_probability[None, :, :].astype(np.float64).clip(min=EPSILON)
        middle = 0.5 * (left + right)
        divergence = 0.5 * np.sum(left * np.log(left / middle), axis=2)
        divergence += 0.5 * np.sum(right * np.log(right / middle), axis=2)
        output[:, :, group_index] = np.clip(1.0 - divergence / np.log(2.0), 0.0, 1.0)
    return output


def standardize_across_classes(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("class features must have query x class x feature shape")
    center = values.mean(axis=1, keepdims=True)
    scale = values.std(axis=1, keepdims=True).clip(min=1e-6)
    return ((values - center) / scale).astype(np.float32), center.astype(np.float32), scale.astype(np.float32)


def balanced_class_rows(query_labels: np.ndarray, class_count: int) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(query_labels, dtype=np.int64)
    classes = np.arange(class_count, dtype=np.int64)[None, :]
    positive = classes == labels[:, None]
    target = np.where(positive, 1.0, -1.0).astype(np.float64)
    weight = np.where(positive, 0.5, 0.5 / (class_count - 1)).astype(np.float64)
    return target, weight


def fit_nonnegative_residual(
    features: np.ndarray, query_labels: np.ndarray, rows: np.ndarray
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    take = np.asarray(rows, dtype=np.int64)
    target, weight = balanced_class_rows(np.asarray(query_labels)[take], values.shape[1])
    base = values[take, :, 0]
    atoms = values[take, :, 1:]
    root_weight = np.sqrt(weight)
    base_flat = (base * root_weight).reshape(-1)
    target_flat = (target * root_weight).reshape(-1)
    denominator = float(base_flat @ base_flat)
    base_coefficient = max(0.0, float(base_flat @ target_flat) / max(denominator, EPSILON))
    residual = target - base_coefficient * base
    design = (atoms * root_weight[:, :, None]).reshape(-1, atoms.shape[2])
    response = (residual * root_weight).reshape(-1)
    atom_coefficients, _ = nnls(design, response)
    coefficients = np.concatenate(([base_coefficient], atom_coefficients))
    if coefficients.sum() <= EPSILON:
        coefficients[0] = 1.0
    return (coefficients / coefficients.sum()).astype(np.float32)


def score_classes(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.einsum("qcf,f->qc", features, weights, optimize=True)


def gated_classes(
    *, base_scores: np.ndarray, combined_scores: np.ndarray, threshold: float
) -> np.ndarray:
    base_class = np.asarray(base_scores).argmax(axis=1)
    combined_class = np.asarray(combined_scores).argmax(axis=1)
    rows = np.arange(len(base_class))
    advantage = combined_scores[rows, combined_class] - combined_scores[rows, base_class]
    return np.where(advantage > threshold, combined_class, base_class)


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .contracts import MethodContract, assert_calibration_splits
from .evidence import EmbeddingStore, EvidenceStore
from .hashing import stable_hash
from .retrieval import query_candidate_features, retrieve_full_pool, validate_retrieval_stores
from .schema import Sample


CALIBRATION_OBJECTIVE = "balanced-centered-same-class-affinity-nnls-v1"


def stratified_folds(samples: Sequence[Sample], fold_count: int, seed: int) -> np.ndarray:
    if fold_count < 2 or fold_count > len(samples):
        raise ValueError("fold_count must be between 2 and the number of samples")
    assignments = np.full(len(samples), -1, dtype=np.int32)
    labels: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        labels.setdefault(sample.label, []).append(index)
    for label in sorted(labels):
        indices = labels[label]
        indices.sort(key=lambda index: stable_hash({"seed": seed, "sample_id": samples[index].sample_id}))
        offset = int(stable_hash({"seed": seed, "label": label})[:8], 16) % fold_count
        for rank, index in enumerate(indices):
            assignments[index] = (offset + rank) % fold_count
    if np.any(assignments < 0):
        raise RuntimeError("failed to assign every calibration sample")
    if len(set(assignments.tolist())) != fold_count:
        raise ValueError("one or more calibration folds are empty")
    return assignments


def folds_hash(samples: Sequence[Sample], assignments: np.ndarray) -> str:
    return stable_hash(
        [
            {"sample_id": sample.sample_id, "fold": int(assignments[index])}
            for index, sample in enumerate(samples)
        ]
    )


def accumulate_balanced_gram(
    *,
    fit_query_indices: Sequence[int],
    query_samples: Sequence[Sample],
    candidate_samples: Sequence[Sample],
    query_embeddings: EmbeddingStore,
    candidate_embeddings: EmbeddingStore,
    query_evidence: EvidenceStore,
    candidate_evidence: EvidenceStore,
) -> tuple[np.ndarray, np.ndarray]:
    feature_count = 1 + len(query_evidence.group_ids)
    validate_retrieval_stores(
        query_embeddings,
        candidate_embeddings,
        query_evidence,
        candidate_evidence,
    )
    if tuple(sample.sample_id for sample in query_samples) != query_embeddings.sample_ids:
        raise ValueError("calibration query samples do not align with their stores")
    if tuple(sample.sample_id for sample in candidate_samples) != candidate_embeddings.sample_ids:
        raise ValueError("calibration candidate samples do not align with their stores")
    gram = np.zeros((feature_count, feature_count), dtype=np.float64)
    cross = np.zeros(feature_count, dtype=np.float64)
    candidate_labels = np.asarray([sample.label for sample in candidate_samples])
    for query_index in fit_query_indices:
        features = query_candidate_features(
            query_index=int(query_index),
            query_embeddings=query_embeddings,
            candidate_embeddings=candidate_embeddings,
            query_evidence=query_evidence,
            candidate_evidence=candidate_evidence,
            validate=False,
        )
        centered = features - features.mean(axis=0, keepdims=True)
        positive = candidate_labels == query_samples[int(query_index)].label
        positive_count = int(positive.sum())
        negative_count = len(candidate_samples) - positive_count
        if positive_count == 0 or negative_count == 0:
            raise ValueError(
                f"query {query_samples[int(query_index)].sample_id} has no positive or no negative train candidate"
            )
        target = np.where(positive, 1.0, -1.0)
        sample_weight = np.where(positive, 0.5 / positive_count, 0.5 / negative_count)
        weighted = centered * sample_weight[:, None]
        gram += centered.T @ weighted
        cross += centered.T @ (sample_weight * target)
    gram = 0.5 * (gram + gram.T)
    return gram, cross


def solve_nnls_from_gram(gram: np.ndarray, cross: np.ndarray) -> np.ndarray:
    """Lawson-Hanson active-set NNLS using exact sufficient statistics."""
    matrix = np.asarray(gram, dtype=np.float64)
    vector = np.asarray(cross, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or vector.shape != (matrix.shape[0],):
        raise ValueError("invalid NNLS sufficient-statistic shapes")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(vector)):
        raise ValueError("NNLS sufficient statistics contain non-finite values")
    dimension = len(vector)
    solution = np.zeros(dimension, dtype=np.float64)
    passive = np.zeros(dimension, dtype=bool)
    dual = vector.copy()
    tolerance = 20.0 * np.finfo(np.float64).eps * max(
        1.0,
        float(np.linalg.norm(matrix, ord=1)),
        float(np.linalg.norm(vector, ord=np.inf)),
    ) * dimension
    max_iterations = 30 * dimension * dimension
    iterations = 0
    while np.any((~passive) & (dual > tolerance)):
        candidate_dual = np.where(~passive, dual, -np.inf)
        passive[int(np.argmax(candidate_dual))] = True
        while True:
            trial = np.zeros_like(solution)
            active_indices = np.flatnonzero(passive)
            active_matrix = matrix[np.ix_(active_indices, active_indices)]
            try:
                active_solution = np.linalg.solve(active_matrix, vector[active_indices])
            except np.linalg.LinAlgError:
                active_solution = np.linalg.lstsq(active_matrix, vector[active_indices], rcond=None)[0]
            trial[active_indices] = active_solution
            if np.all(trial[active_indices] > tolerance):
                solution = trial
                break
            nonpositive = active_indices[trial[active_indices] <= tolerance]
            denominators = solution[nonpositive] - trial[nonpositive]
            valid = denominators > 0
            if not np.any(valid):
                passive[nonpositive] = False
                solution[nonpositive] = 0.0
                break
            alpha = float(np.min(solution[nonpositive][valid] / denominators[valid]))
            solution = solution + alpha * (trial - solution)
            to_remove = passive & (solution <= tolerance)
            passive[to_remove] = False
            solution[to_remove] = 0.0
            iterations += 1
            if iterations > max_iterations:
                raise RuntimeError("NNLS active-set solver did not converge")
        dual = vector - matrix @ solution
        iterations += 1
        if iterations > max_iterations:
            raise RuntimeError("NNLS active-set solver did not converge")
    solution[solution <= tolerance] = 0.0
    if solution.sum() <= 0:
        solution[0] = 1.0
    return solution / solution.sum()


def top1_same_class_metrics(
    *,
    query_samples: Sequence[Sample],
    candidate_samples: Sequence[Sample],
    demo_ids: Sequence[Sequence[str]],
) -> dict[str, float]:
    candidate_labels = {sample.sample_id: sample.label for sample in candidate_samples}
    correct = np.asarray(
        [candidate_labels[demos[0]] == query.label for query, demos in zip(query_samples, demo_ids, strict=True)],
        dtype=bool,
    )
    per_class: list[float] = []
    labels = np.asarray([sample.label for sample in query_samples])
    for label in sorted(set(labels.tolist())):
        per_class.append(float(correct[labels == label].mean()))
    return {"accuracy": float(correct.mean()), "macro_accuracy": float(np.mean(per_class))}


@dataclass(frozen=True)
class CalibrationArtifact:
    contract: MethodContract
    group_ids: tuple[str, ...]
    fold_assignments: tuple[int, ...]
    fold_weights: tuple[tuple[float, ...], ...]
    full_fit_weights: tuple[float, ...]
    deployment_weights: tuple[float, ...]
    oof_atom_metrics: dict[str, float]
    oof_global_metrics: dict[str, float]
    fallback_to_global: bool

    def validate(self) -> None:
        feature_count = 1 + len(self.group_ids)
        if len(self.full_fit_weights) != feature_count or len(self.deployment_weights) != feature_count:
            raise ValueError("calibration weight dimension does not match group_ids")
        for weights in (*self.fold_weights, self.full_fit_weights, self.deployment_weights):
            values = np.asarray(weights, dtype=np.float64)
            if len(values) != feature_count or np.any(values < 0) or not np.isclose(values.sum(), 1.0):
                raise ValueError("calibration weights must be nonnegative simplex vectors")
        if self.fallback_to_global and tuple(self.deployment_weights) != (1.0,) + (0.0,) * len(self.group_ids):
            raise ValueError("fallback artifact must deploy exact global-only weights")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "contract": self.contract.to_dict(),
            "group_ids": list(self.group_ids),
            "fold_assignments": list(self.fold_assignments),
            "fold_weights": [list(weights) for weights in self.fold_weights],
            "full_fit_weights": list(self.full_fit_weights),
            "deployment_weights": list(self.deployment_weights),
            "oof_atom_metrics": self.oof_atom_metrics,
            "oof_global_metrics": self.oof_global_metrics,
            "fallback_to_global": self.fallback_to_global,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CalibrationArtifact":
        artifact = cls(
            contract=MethodContract.from_dict(value["contract"]),
            group_ids=tuple(value["group_ids"]),
            fold_assignments=tuple(int(x) for x in value["fold_assignments"]),
            fold_weights=tuple(tuple(float(x) for x in row) for row in value["fold_weights"]),
            full_fit_weights=tuple(float(x) for x in value["full_fit_weights"]),
            deployment_weights=tuple(float(x) for x in value["deployment_weights"]),
            oof_atom_metrics={k: float(v) for k, v in value["oof_atom_metrics"].items()},
            oof_global_metrics={k: float(v) for k, v in value["oof_global_metrics"].items()},
            fallback_to_global=bool(value["fallback_to_global"]),
        )
        artifact.validate()
        return artifact


def cross_fit_calibration(
    *,
    candidate_samples: Sequence[Sample],
    validation_samples: Sequence[Sample],
    candidate_embeddings: EmbeddingStore,
    validation_embeddings: EmbeddingStore,
    candidate_evidence: EvidenceStore,
    validation_evidence: EvidenceStore,
    contract: MethodContract,
    fold_count: int = 5,
    seed: int = 42,
) -> CalibrationArtifact:
    assert_calibration_splits(candidate_samples, validation_samples)
    assignments = stratified_folds(validation_samples, fold_count=fold_count, seed=seed)
    if contract.calibration_folds_hash != folds_hash(validation_samples, assignments):
        raise ValueError("method contract does not match the actual calibration folds")
    oof_atom_demo: list[tuple[str, ...] | None] = [None] * len(validation_samples)
    oof_global_demo: list[tuple[str, ...] | None] = [None] * len(validation_samples)
    fold_weights: list[tuple[float, ...]] = []
    for fold in range(fold_count):
        fit_indices = np.flatnonzero(assignments != fold)
        held_indices = np.flatnonzero(assignments == fold)
        gram, cross = accumulate_balanced_gram(
            fit_query_indices=fit_indices,
            query_samples=validation_samples,
            candidate_samples=candidate_samples,
            query_embeddings=validation_embeddings,
            candidate_embeddings=candidate_embeddings,
            query_evidence=validation_evidence,
            candidate_evidence=candidate_evidence,
        )
        weights = solve_nnls_from_gram(gram, cross)
        fold_weights.append(tuple(float(x) for x in weights))
        # Slice stores without changing their method contract.
        held_embedding_store = EmbeddingStore(
            sample_ids=tuple(validation_embeddings.sample_ids[index] for index in held_indices),
            embeddings=validation_embeddings.embeddings[held_indices],
            model_id=validation_embeddings.model_id,
            model_revision=validation_embeddings.model_revision,
        )
        held_evidence_store = EvidenceStore(
            sample_ids=tuple(validation_evidence.sample_ids[index] for index in held_indices),
            group_ids=validation_evidence.group_ids,
            state_ids=validation_evidence.state_ids,
            state_group_index=validation_evidence.state_group_index,
            raw_logits=validation_evidence.raw_logits[held_indices],
            state_probabilities=validation_evidence.state_probabilities[held_indices],
            signed_margins=validation_evidence.signed_margins[held_indices],
            ordinal_scores=validation_evidence.ordinal_scores[held_indices],
            applicability_scores=validation_evidence.applicability_scores[held_indices],
            bank_hash=validation_evidence.bank_hash,
            model_id=validation_evidence.model_id,
            model_revision=validation_evidence.model_revision,
            calibration_hash=validation_evidence.calibration_hash,
            text_template=validation_evidence.text_template,
            transform=validation_evidence.transform,
        )
        atom_result = retrieve_full_pool(
            weights=weights,
            query_embeddings=held_embedding_store,
            candidate_embeddings=candidate_embeddings,
            query_evidence=held_evidence_store,
            candidate_evidence=candidate_evidence,
            shots=1,
        )
        global_weights = np.zeros_like(weights)
        global_weights[0] = 1.0
        global_result = retrieve_full_pool(
            weights=global_weights,
            query_embeddings=held_embedding_store,
            candidate_embeddings=candidate_embeddings,
            query_evidence=held_evidence_store,
            candidate_evidence=candidate_evidence,
            shots=1,
        )
        for local_index, original_index in enumerate(held_indices):
            oof_atom_demo[int(original_index)] = atom_result.demo_ids[local_index]
            oof_global_demo[int(original_index)] = global_result.demo_ids[local_index]
    if any(value is None for value in oof_atom_demo + oof_global_demo):
        raise RuntimeError("OOF retrieval did not cover every validation query")
    atom_metrics = top1_same_class_metrics(
        query_samples=validation_samples,
        candidate_samples=candidate_samples,
        demo_ids=[value for value in oof_atom_demo if value is not None],
    )
    global_metrics = top1_same_class_metrics(
        query_samples=validation_samples,
        candidate_samples=candidate_samples,
        demo_ids=[value for value in oof_global_demo if value is not None],
    )
    gram, cross = accumulate_balanced_gram(
        fit_query_indices=np.arange(len(validation_samples)),
        query_samples=validation_samples,
        candidate_samples=candidate_samples,
        query_embeddings=validation_embeddings,
        candidate_embeddings=candidate_embeddings,
        query_evidence=validation_evidence,
        candidate_evidence=candidate_evidence,
    )
    full_weights = solve_nnls_from_gram(gram, cross)
    fallback = atom_metrics["accuracy"] <= global_metrics["accuracy"]
    deployment = np.zeros_like(full_weights) if fallback else full_weights.copy()
    if fallback:
        deployment[0] = 1.0
    artifact = CalibrationArtifact(
        contract=contract,
        group_ids=validation_evidence.group_ids,
        fold_assignments=tuple(int(value) for value in assignments),
        fold_weights=tuple(fold_weights),
        full_fit_weights=tuple(float(value) for value in full_weights),
        deployment_weights=tuple(float(value) for value in deployment),
        oof_atom_metrics=atom_metrics,
        oof_global_metrics=global_metrics,
        fallback_to_global=fallback,
    )
    artifact.validate()
    return artifact

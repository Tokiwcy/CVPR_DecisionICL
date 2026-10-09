from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .evidence import EmbeddingStore, EvidenceStore, align_stores, group_similarity_matrix


GROUP_SIMILARITY = "one-minus-total-variation-v1"
GLOBAL_SIMILARITY = "scaled-cosine-v1"


def validate_retrieval_stores(
    query_embeddings: EmbeddingStore,
    candidate_embeddings: EmbeddingStore,
    query_evidence: EvidenceStore,
    candidate_evidence: EvidenceStore,
) -> None:
    query_embeddings.validate()
    candidate_embeddings.validate()
    align_stores(query_evidence, candidate_evidence)
    if query_embeddings.sample_ids != query_evidence.sample_ids:
        raise ValueError("query embedding/evidence IDs do not align")
    if candidate_embeddings.sample_ids != candidate_evidence.sample_ids:
        raise ValueError("candidate embedding/evidence IDs do not align")
    if query_embeddings.model_id != query_evidence.model_id:
        raise ValueError("query global and evidence encoders must use the same frozen model")
    if candidate_embeddings.model_id != candidate_evidence.model_id:
        raise ValueError("candidate global and evidence encoders must use the same frozen model")


def global_similarity_matrix(
    query_embeddings: EmbeddingStore,
    candidate_embeddings: EmbeddingStore,
    query_indices: Sequence[int] | None = None,
) -> np.ndarray:
    queries = query_embeddings.embeddings
    if query_indices is not None:
        queries = queries[np.asarray(query_indices, dtype=np.int64)]
    cosine = queries @ candidate_embeddings.embeddings.T
    return np.clip((cosine + 1.0) * 0.5, 0.0, 1.0)


def query_candidate_features(
    *,
    query_index: int,
    query_embeddings: EmbeddingStore,
    candidate_embeddings: EmbeddingStore,
    query_evidence: EvidenceStore,
    candidate_evidence: EvidenceStore,
    validate: bool = True,
) -> np.ndarray:
    if validate:
        validate_retrieval_stores(query_embeddings, candidate_embeddings, query_evidence, candidate_evidence)
    candidate_count = len(candidate_embeddings.sample_ids)
    feature_count = 1 + len(query_evidence.group_ids)
    features = np.empty((candidate_count, feature_count), dtype=np.float64)
    features[:, 0] = global_similarity_matrix(
        query_embeddings,
        candidate_embeddings,
        [query_index],
    )[0]
    for group_index in range(len(query_evidence.group_ids)):
        features[:, group_index + 1] = group_similarity_matrix(
            query_evidence,
            candidate_evidence,
            group_index,
            [query_index],
            validate_alignment=False,
        )[0]
    return features


@dataclass(frozen=True)
class RetrievalResult:
    query_ids: tuple[str, ...]
    demo_ids: tuple[tuple[str, ...], ...]
    demo_scores: tuple[tuple[float, ...], ...]


def retrieve_full_pool(
    *,
    weights: np.ndarray,
    query_embeddings: EmbeddingStore,
    candidate_embeddings: EmbeddingStore,
    query_evidence: EvidenceStore,
    candidate_evidence: EvidenceStore,
    shots: int = 1,
) -> RetrievalResult:
    """Retrieve directly over every training candidate with deterministic ties."""
    validate_retrieval_stores(query_embeddings, candidate_embeddings, query_evidence, candidate_evidence)
    weights_array = np.asarray(weights, dtype=np.float64)
    expected = 1 + len(query_evidence.group_ids)
    if weights_array.shape != (expected,):
        raise ValueError(f"expected {expected} retrieval weights, got {weights_array.shape}")
    if np.any(weights_array < 0) or not np.all(np.isfinite(weights_array)):
        raise ValueError("retrieval weights must be finite and nonnegative")
    if weights_array.sum() <= 0:
        raise ValueError("at least one retrieval weight must be positive")
    if shots < 1 or shots > len(candidate_embeddings.sample_ids):
        raise ValueError("shots is outside the candidate-pool range")
    candidate_ids = np.asarray(candidate_embeddings.sample_ids)
    # Stores are required to be sorted by sample ID. Stable sorting then gives a
    # declared lexical-ID tie policy rather than a backend-dependent argmax.
    if tuple(sorted(candidate_embeddings.sample_ids)) != candidate_embeddings.sample_ids:
        raise ValueError("candidate stores must be ordered lexically by sample_id")
    all_ids: list[tuple[str, ...]] = []
    all_scores: list[tuple[float, ...]] = []
    normalized = weights_array / weights_array.sum()
    for query_index in range(len(query_embeddings.sample_ids)):
        features = query_candidate_features(
            query_index=query_index,
            query_embeddings=query_embeddings,
            candidate_embeddings=candidate_embeddings,
            query_evidence=query_evidence,
            candidate_evidence=candidate_evidence,
            validate=False,
        )
        scores = features @ normalized
        ranking = np.argsort(-scores, kind="stable")[:shots]
        all_ids.append(tuple(str(candidate_ids[index]) for index in ranking))
        all_scores.append(tuple(float(scores[index]) for index in ranking))
    return RetrievalResult(
        query_ids=query_embeddings.sample_ids,
        demo_ids=tuple(all_ids),
        demo_scores=tuple(all_scores),
    )


def random_retrieval(
    *,
    query_ids: Sequence[str],
    candidate_ids: Sequence[str],
    shots: int,
    seed: int,
) -> RetrievalResult:
    if shots < 1 or shots > len(candidate_ids):
        raise ValueError("invalid number of random shots")
    ordered_candidates = tuple(sorted(candidate_ids))
    demo_ids: list[tuple[str, ...]] = []
    for query_id in query_ids:
        # Independent per-query seed makes results invariant to query ordering.
        import hashlib

        digest = hashlib.sha256(f"{seed}:{query_id}".encode("utf-8")).digest()
        query_seed = int.from_bytes(digest[:8], "little", signed=False)
        rng = np.random.default_rng(query_seed)
        indices = rng.choice(len(ordered_candidates), size=shots, replace=False)
        demo_ids.append(tuple(ordered_candidates[index] for index in indices))
    return RetrievalResult(
        query_ids=tuple(query_ids),
        demo_ids=tuple(demo_ids),
        demo_scores=tuple(tuple(0.0 for _ in range(shots)) for _ in query_ids),
    )

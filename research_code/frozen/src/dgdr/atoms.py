from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.stats import spearmanr

from .schema import AtomBank, FactorGroup


def canonical_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def structural_key(group: FactorGroup) -> tuple[str, str]:
    return canonical_text(group.region), canonical_text(group.attribute)


@dataclass(frozen=True)
class LintIssue:
    level: str
    code: str
    group_id: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "level": self.level,
            "code": self.code,
            "group_id": self.group_id,
            "message": self.message,
        }


def lint_groups(groups: Iterable[FactorGroup], max_states: int = 6) -> list[LintIssue]:
    materialized = list(groups)
    issues: list[LintIssue] = []
    by_key: dict[tuple[str, str], list[str]] = {}
    for group in materialized:
        try:
            group.validate()
        except (ValueError, TypeError) as error:
            issues.append(LintIssue("error", "schema", group.group_id, str(error)))
            continue
        if len(group.states) > max_states:
            issues.append(
                LintIssue(
                    "error",
                    "too_many_states",
                    group.group_id,
                    f"{len(group.states)} states exceed the frozen limit {max_states}; do not union heterogeneous axes",
                )
            )
        substantive = [state for state in group.states if not state.abstain]
        if any(state.coordinate is not None for state in substantive) and not group.is_ordinal:
            issues.append(LintIssue("error", "partial_coordinates", group.group_id, "ordinal axis is incomplete"))
        key = structural_key(group)
        by_key.setdefault(key, []).append(group.group_id)
    for key, ids in by_key.items():
        if len(ids) > 1:
            for group_id in ids:
                issues.append(
                    LintIssue(
                        "warning",
                        "structural_duplicate_candidate",
                        group_id,
                        f"region/attribute key {key!r} occurs in {ids}",
                    )
                )
    return issues


def group_semantic_text(group: FactorGroup) -> str:
    states = "; ".join(state.label for state in group.states if not state.abstain)
    return f"region: {group.region}; attribute: {group.attribute}; question: {group.question}; states: {states}"


def collapse_exact_duplicate_groups(
    groups: Iterable[FactorGroup],
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Collapse byte-equivalent semantic axes before any encoder is consulted."""
    materialized = list(groups)
    by_signature: dict[tuple[object, ...], list[str]] = {}
    for group in materialized:
        group.validate()
        state_signature = tuple(
            (
                canonical_text(state.label),
                canonical_text(state.visual_description),
                state.coordinate,
                state.abstain,
            )
            for state in group.states
        )
        signature: tuple[object, ...] = (
            canonical_text(group.region),
            canonical_text(group.attribute),
            canonical_text(group.question),
            canonical_text(group.applies_to),
            state_signature,
        )
        by_signature.setdefault(signature, []).append(group.group_id)
    kept: list[str] = []
    removed: dict[str, str] = {}
    for ids in by_signature.values():
        representative = min(ids)
        kept.append(representative)
        for group_id in ids:
            if group_id != representative:
                removed[group_id] = representative
    return tuple(sorted(kept)), dict(sorted(removed.items()))


@dataclass(frozen=True)
class DuplicatePair:
    left_group_id: str
    right_group_id: str
    semantic_cosine: float
    response_spearman: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "left_group_id": self.left_group_id,
            "right_group_id": self.right_group_id,
            "semantic_cosine": self.semantic_cosine,
            "response_spearman": self.response_spearman,
        }


@dataclass(frozen=True)
class DeduplicationResult:
    kept_group_ids: tuple[str, ...]
    removed_to_representative: dict[str, str]
    duplicate_pairs: tuple[DuplicatePair, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "kept_group_ids": list(self.kept_group_ids),
            "removed_to_representative": dict(self.removed_to_representative),
            "duplicate_pairs": [pair.to_dict() for pair in self.duplicate_pairs],
        }


def response_aware_deduplicate(
    *,
    group_ids: list[str],
    semantic_embeddings: np.ndarray,
    response_fingerprints: np.ndarray,
    quality_scores: dict[str, float] | None = None,
    semantic_threshold: float = 0.90,
    response_threshold: float = 0.98,
) -> DeduplicationResult:
    """Deduplicate only when both meaning and measured image response agree.

    Text similarity alone is intentionally insufficient: contrastive words such as
    ``short`` and ``long`` can be close in an embedding space.  Fingerprints are
    label-free group-similarity responses over a fixed set of train-image pairs.
    """
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("group_ids must be unique")
    semantic = np.asarray(semantic_embeddings, dtype=np.float64)
    responses = np.asarray(response_fingerprints, dtype=np.float64)
    if semantic.shape[0] != len(group_ids) or responses.shape[0] != len(group_ids):
        raise ValueError("embedding/fingerprint rows must align with group_ids")
    norms = np.linalg.norm(semantic, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("semantic embeddings must be non-zero")
    semantic = semantic / norms
    quality = quality_scores or {}
    parent = list(range(len(group_ids)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    pairs: list[DuplicatePair] = []
    for left in range(len(group_ids)):
        for right in range(left + 1, len(group_ids)):
            semantic_cosine = float(semantic[left] @ semantic[right])
            if semantic_cosine < semantic_threshold:
                continue
            correlation = spearmanr(responses[left], responses[right], nan_policy="omit").statistic
            response_spearman = 0.0 if not np.isfinite(correlation) else float(correlation)
            if abs(response_spearman) < response_threshold:
                continue
            pairs.append(
                DuplicatePair(
                    group_ids[left],
                    group_ids[right],
                    semantic_cosine,
                    response_spearman,
                )
            )
            union(left, right)

    components: dict[int, list[int]] = {}
    for index in range(len(group_ids)):
        components.setdefault(find(index), []).append(index)
    kept: list[str] = []
    removed: dict[str, str] = {}
    for members in components.values():
        representative_index = min(
            members,
            key=lambda index: (-quality.get(group_ids[index], 0.0), group_ids[index]),
        )
        representative = group_ids[representative_index]
        kept.append(representative)
        for index in members:
            if index != representative_index:
                removed[group_ids[index]] = representative
    return DeduplicationResult(
        kept_group_ids=tuple(sorted(kept)),
        removed_to_representative=dict(sorted(removed.items())),
        duplicate_pairs=tuple(pairs),
    )


def filter_bank(bank: AtomBank, kept_group_ids: Iterable[str]) -> AtomBank:
    kept = set(kept_group_ids)
    unknown = kept - {group.group_id for group in bank.groups}
    if unknown:
        raise ValueError(f"unknown groups requested: {sorted(unknown)}")
    result = AtomBank(
        task_name=bank.task_name,
        task_instruction=bank.task_instruction,
        class_names=bank.class_names,
        factor_scope=bank.factor_scope,
        groups=tuple(group for group in bank.groups if group.group_id in kept),
        discovery_split=bank.discovery_split,
        discovery_seed=bank.discovery_seed,
        protocol_version=bank.protocol_version,
    )
    result.validate()
    return result

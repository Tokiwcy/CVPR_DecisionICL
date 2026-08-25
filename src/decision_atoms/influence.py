from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .math_utils import normalize_rows, spearman
from .types import BoundaryRecord, Sample


def true_margin(scores: dict[str, float], true_label: str, labels: list[str]) -> float:
    """Log-score margin of the true class against its strongest alternative."""
    alternatives = [float(scores[label]) for label in labels if label != true_label]
    if not alternatives:
        raise ValueError("true_margin requires at least two labels")
    return float(scores[true_label]) - max(alternatives)


def predicted_label(scores: dict[str, float], labels: list[str]) -> str:
    return max(labels, key=lambda label: float(scores[label]))


def select_boundary_probes(
    records: list[BoundaryRecord],
    labels: list[str],
    probes_per_class: int,
    errors_per_class: int,
) -> list[dict[str, Any]]:
    """Select stable errors plus the lowest-positive-margin stable correct cases."""
    if not 0 <= errors_per_class <= probes_per_class:
        raise ValueError("errors_per_class must be between 0 and probes_per_class")
    output: list[dict[str, Any]] = []
    for label in labels:
        stable = [
            row
            for row in records
            if row.true_label == label
            and row.prompt_predictions
            and len(set(row.prompt_predictions)) == 1
        ]
        errors = sorted((row for row in stable if not row.correct), key=lambda row: row.true_margin)
        correct = sorted((row for row in stable if row.correct), key=lambda row: row.true_margin)
        if len(errors) < errors_per_class:
            raise ValueError(
                f"class {label!r} has only {len(errors)} stable errors, need {errors_per_class}"
            )
        correct_count = probes_per_class - errors_per_class
        if len(correct) < correct_count:
            raise ValueError(
                f"class {label!r} has only {len(correct)} stable correct cases, need {correct_count}"
            )

        # When more stable errors exist than the budget, span their entire difficulty
        # range instead of using only the most catastrophic failures.
        if errors_per_class:
            chosen_indices = np.rint(
                np.linspace(0, len(errors) - 1, num=errors_per_class)
            ).astype(int)
            chosen_errors = [errors[index] for index in chosen_indices]
        else:
            chosen_errors = []
        chosen = [("stable_error", row) for row in chosen_errors]
        chosen.extend(("low_margin_correct", row) for row in correct[:correct_count])
        output.extend(
            {
                "query_id": row.sample_id,
                "query_label": row.true_label,
                "probe_kind": kind,
                "selection_true_margin": float(row.true_margin),
                "selection_prediction": row.predicted_label,
                "prompt_predictions": list(row.prompt_predictions),
            }
            for kind, row in chosen
        )
    if len({row["query_id"] for row in output}) != len(output):
        raise ValueError("probe selection contains duplicate query ids")
    return output


def build_candidate_pairs(
    probes: list[dict[str, Any]],
    train_samples: list[Sample],
    query_samples: list[Sample],
    train_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    labels: list[str],
    candidates_per_class: int,
) -> list[dict[str, Any]]:
    """Retrieve the top visual candidates independently within each class."""
    train_embeddings = normalize_rows(train_embeddings)
    query_embeddings = normalize_rows(query_embeddings)
    query_index = {sample.id: index for index, sample in enumerate(query_samples)}
    train_by_label = {
        label: np.asarray(
            [index for index, sample in enumerate(train_samples) if sample.label == label],
            dtype=int,
        )
        for label in labels
    }
    output: list[dict[str, Any]] = []
    for probe in probes:
        qid = str(probe["query_id"])
        if qid not in query_index:
            raise ValueError(f"probe {qid!r} is missing from query embeddings")
        q_index = query_index[qid]
        for candidate_label in labels:
            pool = train_by_label[candidate_label]
            if len(pool) < candidates_per_class:
                raise ValueError(
                    f"class {candidate_label!r} has only {len(pool)} train candidates"
                )
            similarities = train_embeddings[pool] @ query_embeddings[q_index]
            # sample id is a deterministic tie-breaker after descending similarity.
            order = sorted(
                range(len(pool)),
                key=lambda local: (
                    -float(similarities[local]),
                    train_samples[int(pool[local])].id,
                ),
            )[:candidates_per_class]
            for rank, local in enumerate(order, start=1):
                train_index = int(pool[local])
                candidate = train_samples[train_index]
                output.append(
                    {
                        **probe,
                        "candidate_id": candidate.id,
                        "candidate_label": candidate.label,
                        "candidate_rank_within_class": rank,
                        "visual_similarity": float(similarities[local]),
                    }
                )
    keys = [(row["query_id"], row["candidate_id"]) for row in output]
    if len(keys) != len(set(keys)):
        raise ValueError("candidate pairs contain duplicate query-candidate keys")
    return output


def _stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
        "q10": float(np.quantile(array, 0.10)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "q90": float(np.quantile(array, 0.90)),
        "positive_rate": float((array > 0).mean()),
        "negative_rate": float((array < 0).mean()),
    }


def summarize_influence(rows: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty influence table")
    utilities = [float(row["utility_true_margin"]) for row in rows]
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_query[str(row["query_id"])].append(row)

    mixed_queries = 0
    best_gains: list[float] = []
    worst_gains: list[float] = []
    same_label_ranges: list[float] = []
    top_similarity_utilities: list[float] = []
    global_top_similarity_utilities: list[float] = []
    within_class_oracle_utilities: list[float] = []
    mixed_same_label_cells = 0
    same_label_cells = 0
    for query_rows in by_query.values():
        values = np.asarray([row["utility_true_margin"] for row in query_rows], dtype=float)
        mixed_queries += int(np.any(values > 0) and np.any(values < 0))
        best_gains.append(float(values.max()))
        worst_gains.append(float(values.min()))
        global_top_similarity_utilities.append(
            float(max(query_rows, key=lambda row: float(row["visual_similarity"]))["utility_true_margin"])
        )
        for candidate_label in labels:
            class_rows = [
                row for row in query_rows if row["candidate_label"] == candidate_label
            ]
            if not class_rows:
                continue
            class_values = [float(row["utility_true_margin"]) for row in class_rows]
            same_label_cells += 1
            mixed_same_label_cells += int(
                any(value > 0 for value in class_values) and any(value < 0 for value in class_values)
            )
            same_label_ranges.append(max(class_values) - min(class_values))
            top_similarity_utilities.extend(
                float(row["utility_true_margin"])
                for row in class_rows
                if int(row["candidate_rank_within_class"]) == 1
            )
            within_class_oracle_utilities.append(max(class_values))

    grouped: dict[str, Any] = {}
    group_fields = ("probe_kind", "query_label", "candidate_label", "query_group")
    for field in group_fields:
        field_groups: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            if row.get(field) is not None:
                field_groups[str(row[field])].append(float(row["utility_true_margin"]))
        grouped[field] = {key: _stats(values) for key, values in sorted(field_groups.items())}

    relation_groups: dict[str, list[float]] = defaultdict(list)
    class_pair_groups: dict[str, list[float]] = defaultdict(list)
    stratum_correlations: dict[str, float] = {}
    for row in rows:
        relation = "matching_demo_label" if row["query_label"] == row["candidate_label"] else "mismatching_demo_label"
        relation_groups[relation].append(float(row["utility_true_margin"]))
        pair_key = f"query={row['query_label']}|demo={row['candidate_label']}"
        class_pair_groups[pair_key].append(float(row["utility_true_margin"]))
    for pair_key in class_pair_groups:
        pair_rows = [
            row
            for row in rows
            if f"query={row['query_label']}|demo={row['candidate_label']}" == pair_key
        ]
        stratum_correlations[pair_key] = spearman(
            np.asarray([row["visual_similarity"] for row in pair_rows], dtype=float),
            np.asarray([row["utility_true_margin"] for row in pair_rows], dtype=float),
        )

    first_rows = [query_rows[0] for query_rows in by_query.values()]
    baseline_correct = sum(bool(row["baseline_correct"]) for row in first_rows)
    baseline_by_probe: dict[str, list[float]] = defaultdict(list)
    for row in first_rows:
        baseline_by_probe[str(row["probe_kind"])].append(float(bool(row["baseline_correct"])))

    by_rank: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_rank[str(row["candidate_rank_within_class"])].append(
            float(row["utility_true_margin"])
        )

    rescued = sum(bool(row["rescued"]) for row in rows)
    harmed = sum(bool(row["harmed"]) for row in rows)
    wrong_trials = sum(not bool(row["baseline_correct"]) for row in rows)
    correct_trials = sum(bool(row["baseline_correct"]) for row in rows)
    similarities = np.asarray([row["visual_similarity"] for row in rows], dtype=float)
    signed_pushes = [float(row["shift_toward_demo_label"]) for row in rows]
    return {
        "num_pairs": len(rows),
        "num_queries": len(by_query),
        "controlled_no_demo_baseline_accuracy": float(baseline_correct / len(first_rows)),
        "controlled_no_demo_baseline_by_probe_kind": {
            kind: {"count": len(values), "accuracy": float(np.mean(values))}
            for kind, values in sorted(baseline_by_probe.items())
        },
        "utility_true_margin": _stats(utilities),
        "shift_toward_demo_label": _stats(signed_pushes),
        "rescue_rate_per_wrong_trial": float(rescued / wrong_trials) if wrong_trials else 0.0,
        "harm_rate_per_correct_trial": float(harmed / correct_trials) if correct_trials else 0.0,
        "queries_with_both_helpful_and_harmful_candidates": mixed_queries,
        "query_mixed_effect_rate": float(mixed_queries / len(by_query)),
        "query_demo_label_cells_with_both_effect_signs": mixed_same_label_cells,
        "query_demo_label_cell_mixed_effect_rate": float(mixed_same_label_cells / same_label_cells),
        "best_candidate_utility_per_query": _stats(best_gains),
        "worst_candidate_utility_per_query": _stats(worst_gains),
        "same_demo_label_utility_range_per_query": _stats(same_label_ranges),
        "top_similarity_utility": _stats(top_similarity_utilities),
        "global_top_similarity_utility": _stats(global_top_similarity_utilities),
        "within_class_oracle_utility": _stats(within_class_oracle_utilities),
        "visual_similarity_utility_spearman": spearman(similarities, np.asarray(utilities)),
        "visual_similarity_utility_spearman_by_class_pair": stratum_correlations,
        "by_similarity_rank": {
            rank: _stats(values)
            for rank, values in sorted(by_rank.items(), key=lambda item: int(item[0]))
        },
        "by_group": grouped,
        "by_query_demo_label_relation": {
            key: _stats(values) for key, values in sorted(relation_groups.items())
        },
        "by_query_demo_class_pair": {
            key: _stats(values) for key, values in sorted(class_pair_groups.items())
        },
    }

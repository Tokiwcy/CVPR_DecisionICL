from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from behavior_icl import QwenVLBackend, make_single_token_codes

from run_mvp import (
    DEFAULT_ARTIFACT,
    LABELS,
    choose_queries,
    exact_mcnemar_p,
    load_inputs,
    read_jsonl,
    resolve_model,
    stable_key,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "artifacts" / "three_shot_global_weighted_qwen4b"
)
SEED = 20260824
K = 3
RICES = "rices_global_3shot"
HARD = "hard_coverage_global_3shot"
WEIGHTED = "weighted_additive_global_3shot"
LAMBDA_GRID: tuple[tuple[str, float], ...] = (
    ("0", 0.0),
    ("0.005", 0.005),
    ("0.01", 0.01),
    ("0.015", 0.015),
    ("0.02", 0.02),
    ("0.03", 0.03),
    ("0.05", 0.05),
    ("0.075", 0.075),
    ("0.1", 0.1),
    ("0.125", 0.125),
    ("0.15", 0.15),
    ("0.2", 0.2),
    ("0.3", 0.3),
    ("0.4", 0.4),
)


def choose_hard_coverage_context(
    rices_demos: Sequence[Mapping[str, Any]],
    best_diagnostic: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(rices_demos) != K:
        raise ValueError(f"Expected {K} RICES demos")
    diagnostic_count = sum(bool(demo["diagnostic"]) for demo in rices_demos)
    if diagnostic_count:
        return [dict(demo) for demo in rices_demos], {
            "action": "keep_rices",
            "replaced_label": None,
            "replaced_demo_id": None,
            "similarity_loss": 0.0,
            "rices_diagnostic_count": diagnostic_count,
        }
    replace_index, replace_demo = min(
        enumerate(rices_demos),
        key=lambda item: (float(item[1]["similarity"]), str(item[1]["id"])),
    )
    diagnostic = dict(best_diagnostic)
    loss = float(replace_demo["similarity"]) - float(diagnostic["similarity"])
    if loss < -1e-6:
        raise ValueError("Diagnostic replacement unexpectedly improves over global RICES top-k")
    chosen = [dict(demo) for demo in rices_demos]
    replaced_id = str(chosen[replace_index]["id"])
    chosen[replace_index] = diagnostic
    chosen = order_for_prompt(chosen)
    return chosen, {
        "action": "inject_diagnostic",
        "replaced_label": str(replace_demo["label"]),
        "injected_label": str(diagnostic["label"]),
        "replaced_demo_id": replaced_id,
        "similarity_loss": float(loss),
        "rices_diagnostic_count": 0,
    }


def order_for_prompt(demos: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Put the most similar selected demo closest to the query."""
    return sorted(
        (dict(demo) for demo in demos),
        key=lambda demo: (float(demo["similarity"]), str(demo["id"])),
    )


def choose_weighted_context(
    candidates: Sequence[Mapping[str, Any]], lambda_value: float
) -> list[dict[str, Any]]:
    """Global top-k under an additive similarity + diagnosticity score."""
    ranked = sorted(
        candidates,
        key=lambda demo: (
            -(float(demo["similarity"]) + lambda_value * int(bool(demo["diagnostic"]))),
            -float(demo["similarity"]),
            str(demo["id"]),
        ),
    )
    return order_for_prompt(ranked[:K])


def _demo(unit: Mapping[str, Any], similarity: float) -> dict[str, Any]:
    return {
        "id": str(unit["unit_id"]),
        "image": str(unit["image"]),
        "label": str(unit["label"]),
        "group": str(unit["group"]),
        "y": int(unit["y"]),
        "place": int(unit["place"]),
        "diagnostic": bool(int(unit["y"]) != int(unit["place"])),
        "similarity": float(similarity),
    }


def build_split_selections(
    units: Sequence[Mapping[str, Any]],
    features: np.ndarray,
    *,
    query_role: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
]:
    position = {str(unit["unit_id"]): index for index, unit in enumerate(units)}
    candidate_indices = [
        index for index, unit in enumerate(units) if unit["role"] == "candidate"
    ]
    query_rows = choose_queries(units, query_role, None, SEED)
    if len(query_rows) != 1000:
        raise ValueError(f"Expected 1000 {query_role} rows, found {len(query_rows)}")
    rices_rows: list[dict[str, Any]] = []
    hard_rows: list[dict[str, Any]] = []
    weighted_rows: dict[str, list[dict[str, Any]]] = {
        label: [] for label, _ in LAMBDA_GRID
    }
    action_counts: dict[str, int] = defaultdict(int)
    replacement_labels: dict[str, int] = defaultdict(int)
    injected_labels: dict[str, int] = defaultdict(int)
    losses: list[float] = []
    group_counts: dict[str, int] = defaultdict(int)

    for query in sorted(query_rows, key=lambda row: str(row["unit_id"])):
        query_id = str(query["unit_id"])
        group_counts[str(query["group"])] += 1
        similarities = features @ features[position[query_id]]
        candidates = [
            _demo(units[index], float(similarities[index])) for index in candidate_indices
        ]
        globally_ranked = sorted(
            candidates,
            key=lambda demo: (-float(demo["similarity"]), str(demo["id"])),
        )
        demos = order_for_prompt(globally_ranked[:K])
        diagnostic_candidates = [demo for demo in globally_ranked if demo["diagnostic"]]
        if not diagnostic_candidates:
            raise ValueError("No diagnostic candidate in the complete training pool")
        best_diagnostic = diagnostic_candidates[0]
        hard_demos, policy = choose_hard_coverage_context(demos, best_diagnostic)
        action_counts[str(policy["action"])] += 1
        if policy["replaced_label"] is not None:
            replacement_labels[str(policy["replaced_label"])] += 1
            injected_labels[str(policy["injected_label"])] += 1
            losses.append(float(policy["similarity_loss"]))
        base = {
            "query_id": query_id,
            "query_image": str(query["image"]),
            "gold": str(query["label"]),
            "group": str(query["group"]),
            "y": int(query["y"]),
            "place": int(query["place"]),
            "conflict": bool(int(query["y"]) != int(query["place"])),
            "query_role": query_role,
            "retrieval_scope": "global_complete_training_pool",
            "prompt_order": "ascending_similarity_most_similar_nearest_query",
        }
        rices_rows.append(
            {
                **base,
                "condition": RICES,
                "demos": demos,
                "mean_similarity": float(np.mean([demo["similarity"] for demo in demos])),
                "diagnostic_rate": float(np.mean([demo["diagnostic"] for demo in demos])),
            }
        )
        hard_rows.append(
            {
                **base,
                "condition": HARD,
                "demos": hard_demos,
                "mean_similarity": float(
                    np.mean([demo["similarity"] for demo in hard_demos])
                ),
                "diagnostic_rate": float(
                    np.mean([demo["diagnostic"] for demo in hard_demos])
                ),
                "policy": policy,
            }
        )
        for lambda_label, lambda_value in LAMBDA_GRID:
            weighted_demos = choose_weighted_context(candidates, lambda_value)
            weighted_rows[lambda_label].append(
                {
                    **base,
                    "condition": weighted_condition(lambda_label),
                    "lambda": lambda_value,
                    "demos": weighted_demos,
                    "mean_similarity": float(
                        np.mean([demo["similarity"] for demo in weighted_demos])
                    ),
                    "diagnostic_rate": float(
                        np.mean([demo["diagnostic"] for demo in weighted_demos])
                    ),
                }
            )
    for rices, hard in zip(rices_rows, hard_rows, strict=True):
        if len(rices["demos"]) != K or len(hard["demos"]) != K:
            raise ValueError("Invalid shot count")
        if sum(bool(demo["diagnostic"]) for demo in hard["demos"]) < 1:
            raise ValueError("Hard coverage context has no diagnostic demo")
        if rices["prompt_order"] != hard["prompt_order"]:
            raise ValueError("Methods do not share the prompt-order convention")

    def selection_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        similarities = [float(demo["similarity"]) for row in rows for demo in row["demos"]]
        diagnostics = [float(demo["diagnostic"]) for row in rows for demo in row["demos"]]
        label_compositions: dict[str, int] = defaultdict(int)
        for row in rows:
            counts = {label: 0 for label in LABELS}
            for demo in row["demos"]:
                counts[str(demo["label"])] += 1
            signature = "/".join(f"{label}:{counts[label]}" for label in LABELS)
            label_compositions[signature] += 1
        return {
            "mean_similarity": float(np.mean(similarities)),
            "diagnostic_rate": float(np.mean(diagnostics)),
            "contexts_with_diagnostic": sum(
                any(bool(demo["diagnostic"]) for demo in row["demos"]) for row in rows
            ),
            "label_compositions": dict(sorted(label_compositions.items())),
        }

    audit = {
        "query_role": query_role,
        "queries": len(query_rows),
        "group_counts": dict(sorted(group_counts.items())),
        "retrieval_scope": "global; no class quota",
        "prompt_order": "ascending similarity; most similar demo nearest query",
        "actions": dict(sorted(action_counts.items())),
        "replacement_labels": dict(sorted(replacement_labels.items())),
        "injected_labels": dict(sorted(injected_labels.items())),
        "rices": selection_stats(rices_rows),
        "hard_coverage": selection_stats(hard_rows),
        "weighted": {
            label: selection_stats(rows) for label, rows in weighted_rows.items()
        },
        "replacement_similarity_loss": {
            "count": len(losses),
            "mean": float(np.mean(losses)),
            "median": float(np.median(losses)),
            "max": float(np.max(losses)),
        },
    }
    return rices_rows, hard_rows, weighted_rows, audit


def weighted_condition(lambda_label: str) -> str:
    return f"{WEIGHTED}__lambda_{lambda_label.replace('.', 'p')}"


def prepare_all(artifact: Path, output: Path) -> dict[str, Any]:
    audit_path = output / "selection_audit.json"
    paths = {
        ("validation", RICES): output / "selections" / f"validation__{RICES}.jsonl",
        ("validation", HARD): output / "selections" / f"validation__{HARD}.jsonl",
        ("test", RICES): output / "selections" / f"test__{RICES}.jsonl",
        ("test", HARD): output / "selections" / f"test__{HARD}.jsonl",
    }
    for split in ("validation", "test"):
        for lambda_label, _ in LAMBDA_GRID:
            condition = weighted_condition(lambda_label)
            paths[(split, condition)] = output / "selections" / f"{split}__{condition}.jsonl"
    if audit_path.exists() and all(path.exists() for path in paths.values()):
        return json.loads(audit_path.read_text(encoding="utf-8"))
    units, features = load_inputs(artifact)
    audit: dict[str, Any] = {}
    for split, role in (("validation", "validation_query"), ("test", "test_query")):
        rices, hard, weighted, split_audit = build_split_selections(
            units, features, query_role=role
        )
        write_jsonl(paths[(split, RICES)], rices)
        write_jsonl(paths[(split, HARD)], hard)
        for lambda_label, rows in weighted.items():
            write_jsonl(paths[(split, weighted_condition(lambda_label))], rows)
        audit[split] = split_audit
    write_json(audit_path, audit)
    return audit


def _score_rows(
    backend: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    output_path: Path,
    condition: str,
    label_codes: Sequence[str],
    batch_size: int,
) -> list[dict[str, Any]]:
    existing = read_jsonl(output_path) if output_path.exists() else []
    existing_ids = {str(row["query_id"]) for row in existing}
    if len(existing_ids) != len(existing):
        raise ValueError(f"Duplicate checkpoints in {output_path}")
    pending = [row for row in rows if str(row["query_id"]) not in existing_ids]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for begin in range(0, len(pending), batch_size):
            batch = pending[begin : begin + batch_size]
            requests = []
            for row in batch:
                demos = [
                    {"path": demo["image"], "text": "", "label": demo["label"]}
                    for demo in row["demos"]
                ]
                requests.append((demos, {"path": row["query_image"], "text": ""}))
            results = backend.score_named_classes_batch(
                requests,
                list(LABELS),
                list(label_codes),
                task_instruction="Identify whether the bird is a landbird or waterbird.",
            )
            for row, result in zip(batch, results, strict=True):
                prediction = str(result["prediction"])
                record = {
                    "query_id": row["query_id"],
                    "condition": condition,
                    "gold": row["gold"],
                    "group": row["group"],
                    "conflict": row["conflict"],
                    "prediction": prediction,
                    "correct": prediction == row["gold"],
                    "probabilities": result["probabilities"],
                    "retrieved_ids": [demo["id"] for demo in row["demos"]],
                    "mean_similarity": row["mean_similarity"],
                    "diagnostic_rate": row["diagnostic_rate"],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    result = read_jsonl(output_path)
    if len(result) != len(rows):
        raise ValueError(f"Expected {len(rows)} predictions in {output_path}")
    return result


def _reuse_matching_predictions(
    selections: Sequence[Mapping[str, Any]],
    *,
    output_path: Path,
    condition: str,
    source_predictions: Sequence[Mapping[str, Any]],
) -> None:
    existing = read_jsonl(output_path) if output_path.exists() else []
    existing_ids = {str(row["query_id"]) for row in existing}
    if len(existing_ids) != len(existing):
        raise ValueError(f"Duplicate checkpoints in {output_path}")
    source_by_context = {
        (str(row["query_id"]), tuple(str(value) for value in row["retrieved_ids"])): row
        for row in source_predictions
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for selection in selections:
            query_id = str(selection["query_id"])
            if query_id in existing_ids:
                continue
            signature = (
                query_id,
                tuple(str(demo["id"]) for demo in selection["demos"]),
            )
            source = source_by_context.get(signature)
            if source is None:
                continue
            record = {
                **source,
                "condition": condition,
                "retrieved_ids": list(signature[1]),
                "mean_similarity": selection["mean_similarity"],
                "diagnostic_rate": selection["diagnostic_rate"],
                "reused_from_condition": source["condition"],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            existing_ids.add(query_id)
        handle.flush()
        os.fsync(handle.fileno())


def evaluate_split(
    output: Path,
    split: str,
    model_path: Path,
    batch_size: int,
    *,
    precision: str,
) -> dict[str, list[dict[str, Any]]]:
    rices_selections = read_jsonl(
        output / "selections" / f"{split}__{RICES}.jsonl"
    )
    hard_selections = read_jsonl(
        output / "selections" / f"{split}__{HARD}.jsonl"
    )
    backend = QwenVLBackend(model_path, load_in_4bit=precision == "nf4")
    label_codes = make_single_token_codes(backend.processor.tokenizer, len(LABELS))
    prediction_dir = output / "predictions" / split
    predictions: dict[str, list[dict[str, Any]]] = {}
    try:
        rices_predictions = _score_rows(
            backend,
            rices_selections,
            output_path=prediction_dir / f"{RICES}.jsonl",
            condition=RICES,
            label_codes=label_codes,
            batch_size=batch_size,
        )
        predictions[RICES] = rices_predictions
        hard_path = prediction_dir / f"{HARD}.jsonl"
        _reuse_matching_predictions(
            hard_selections,
            output_path=hard_path,
            condition=HARD,
            source_predictions=rices_predictions,
        )
        hard_predictions = _score_rows(
            backend,
            hard_selections,
            output_path=hard_path,
            condition=HARD,
            label_codes=label_codes,
            batch_size=batch_size,
        )
        predictions[HARD] = hard_predictions
        available_predictions = [*rices_predictions, *hard_predictions]
        if split == "validation":
            lambda_labels = [label for label, _ in LAMBDA_GRID]
        else:
            protocol_path = output / "frozen_protocol.json"
            if not protocol_path.exists():
                raise ValueError("Test evaluation requires frozen_protocol.json")
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            lambda_labels = [str(protocol["selected_lambda"])]
        for lambda_label in lambda_labels:
            condition = weighted_condition(lambda_label)
            selections = read_jsonl(
                output / "selections" / f"{split}__{condition}.jsonl"
            )
            weighted_path = prediction_dir / f"{condition}.jsonl"
            _reuse_matching_predictions(
                selections,
                output_path=weighted_path,
                condition=condition,
                source_predictions=available_predictions,
            )
            weighted_predictions = _score_rows(
                backend,
                selections,
                output_path=weighted_path,
                condition=condition,
                label_codes=label_codes,
                batch_size=batch_size,
            )
            predictions[condition] = weighted_predictions
            available_predictions.extend(weighted_predictions)
    finally:
        backend.close()
    return predictions


def _accuracy(rows: Sequence[Mapping[str, Any]]) -> float:
    return float(np.mean([bool(row["correct"]) for row in rows]))


def method_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["group"])].append(row)
    group_accuracy = {group: _accuracy(values) for group, values in sorted(groups.items())}
    return {
        "n": len(rows),
        "overall_accuracy": _accuracy(rows),
        "aligned_accuracy": _accuracy([row for row in rows if not row["conflict"]]),
        "conflict_accuracy": _accuracy([row for row in rows if row["conflict"]]),
        "mean_group_accuracy": float(np.mean(list(group_accuracy.values()))),
        "worst_group_accuracy": min(group_accuracy.values()),
        "group_accuracy": group_accuracy,
    }


def paired_metrics(
    baseline_rows: Sequence[Mapping[str, Any]], method_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    baseline = {str(row["query_id"]): row for row in baseline_rows}
    method = {str(row["query_id"]): row for row in method_rows}
    if set(baseline) != set(method):
        raise ValueError("Paired query sets differ")

    def subset(ids: Sequence[str]) -> dict[str, Any]:
        rescue = sum(not baseline[q]["correct"] and method[q]["correct"] for q in ids)
        harm = sum(baseline[q]["correct"] and not method[q]["correct"] for q in ids)
        return {
            "n": len(ids),
            "rescue": int(rescue),
            "harm": int(harm),
            "exact_mcnemar_p": exact_mcnemar_p(int(rescue), int(harm)),
        }

    ids = sorted(baseline)
    return {
        "overall": subset(ids),
        "conflict": subset([q for q in ids if baseline[q]["conflict"]]),
        "aligned": subset([q for q in ids if not baseline[q]["conflict"]]),
    }


def validation_sweep(output: Path) -> dict[str, Any]:
    prediction_dir = output / "predictions" / "validation"
    rices = read_jsonl(prediction_dir / f"{RICES}.jsonl")
    hard = read_jsonl(prediction_dir / f"{HARD}.jsonl")
    base_metrics = method_metrics(rices)
    rows = []
    for label, value in LAMBDA_GRID:
        condition = weighted_condition(label)
        weighted = read_jsonl(prediction_dir / f"{condition}.jsonl")
        selections = read_jsonl(output / "selections" / f"validation__{condition}.jsonl")
        rices_by_id = {str(row["query_id"]): row for row in rices}
        changed = sum(
            tuple(demo["id"] for demo in selection["demos"])
            != tuple(rices_by_id[str(selection["query_id"])]["retrieved_ids"])
            for selection in selections
        )
        metrics = method_metrics(weighted)
        rows.append(
            {
                "lambda": label,
                "lambda_value": None if math.isinf(value) else value,
                "changed_contexts_vs_rices": changed,
                "contexts_with_diagnostic": sum(
                    any(bool(demo["diagnostic"]) for demo in selection["demos"])
                    for selection in selections
                ),
                "metrics": metrics,
                "paired_vs_rices": paired_metrics(rices, weighted),
            }
        )
    eligible = [
        row
        for row in rows
        if row["metrics"]["aligned_accuracy"]
        >= base_metrics["aligned_accuracy"] - 0.01 - 1e-12
    ]
    if not eligible:
        raise ValueError("No lambda satisfies the predeclared aligned-accuracy constraint")

    def selection_key(row: Mapping[str, Any]) -> tuple[float, float, float, float, float]:
        metrics = row["metrics"]
        numeric_lambda = math.inf if row["lambda_value"] is None else float(row["lambda_value"])
        return (
            float(metrics["mean_group_accuracy"]),
            float(metrics["worst_group_accuracy"]),
            float(metrics["conflict_accuracy"]),
            float(metrics["overall_accuracy"]),
            -numeric_lambda,
        )

    selected = max(eligible, key=selection_key)
    result = {
        "selection_rule": (
            "Among fixed-grid lambdas with aligned accuracy no more than 1 pp below "
            "RICES, maximize validation mean-group accuracy; tie-break by worst-group, "
            "conflict, overall, then smaller lambda."
        ),
        "rices_metrics": base_metrics,
        "hard_coverage_metrics": method_metrics(hard),
        "grid": rows,
        "selected_lambda": selected["lambda"],
        "selected_lambda_value": selected["lambda_value"],
        "selected_validation_metrics": selected["metrics"],
    }
    write_json(output / "validation_lambda_sweep.json", result)
    return result


def freeze_protocol(
    output: Path, *, model_path: Path | None = None, precision: str | None = None
) -> dict[str, Any]:
    sweep = validation_sweep(output)
    protocol = {
        "protocol_status": "lambda_frozen_on_validation_before_three_shot_test",
        "task": "waterbirds",
        "shots": K,
        "seed": SEED,
        "retrieval_scope": "global top-k over the complete training pool; no class quota",
        "prompt_order": "ascending similarity, so the most similar selected demo is nearest the query",
        "objective": "per-demo score = cosine_similarity + lambda * indicator(demo_is_diagnostic); select global top-k",
        "hard_coverage": (
            "keep global RICES when it already has a diagnostic demo; otherwise replace "
            "its least-similar selected demo with the globally most-similar diagnostic demo"
        ),
        "lambda_grid": [label for label, _ in LAMBDA_GRID],
        "lambda_selection_rule": sweep["selection_rule"],
        "selected_lambda": sweep["selected_lambda"],
        "selected_lambda_value": sweep["selected_lambda_value"],
        "model_path": str(model_path.resolve()) if model_path is not None else None,
        "precision": precision,
        "test_tuning_prohibited": True,
    }
    write_json(output / "frozen_protocol.json", protocol)
    return protocol


def summarize_split(output: Path, split: str, lambda_label: str) -> dict[str, Any]:
    prediction_dir = output / "predictions" / split
    rices = read_jsonl(prediction_dir / f"{RICES}.jsonl")
    hard = read_jsonl(prediction_dir / f"{HARD}.jsonl")
    condition = weighted_condition(lambda_label)
    weighted = read_jsonl(prediction_dir / f"{condition}.jsonl")
    rices_by_id = {str(row["query_id"]): row for row in rices}
    weighted_changed = sum(
        tuple(row["retrieved_ids"])
        != tuple(rices_by_id[str(row["query_id"])]["retrieved_ids"])
        for row in weighted
    )
    methods = {
        RICES: method_metrics(rices),
        HARD: method_metrics(hard),
        WEIGHTED: method_metrics(weighted),
    }
    summary = {
        "split": split,
        "shots": K,
        "lambda": lambda_label,
        "methods": methods,
        "paired_vs_rices": {
            HARD: paired_metrics(rices, hard),
            WEIGHTED: paired_metrics(rices, weighted),
        },
        "weighted_changed_contexts_vs_rices": weighted_changed,
    }
    write_json(output / f"{split}_summary.json", summary)
    return summary


def render_report(
    validation: Mapping[str, Any], test: Mapping[str, Any], protocol: Mapping[str, Any]
) -> str:
    lines = [
        "# Three-shot global diagnostic-aware selection",
        "",
        f"Frozen lambda: `{protocol['selected_lambda']}`.",
        "",
    ]
    for split, summary in (("Validation", validation), ("Test", test)):
        lines.extend(
            [
                f"## {split}",
                "",
                "| Method | Overall | Aligned | Conflict | Mean group | Worst group |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for method in (RICES, HARD, WEIGHTED):
            row = summary["methods"][method]
            lines.append(
                f"| {method} | {100*row['overall_accuracy']:.1f} | "
                f"{100*row['aligned_accuracy']:.1f} | {100*row['conflict_accuracy']:.1f} | "
                f"{100*row['mean_group_accuracy']:.1f} | {100*row['worst_group_accuracy']:.1f} |"
            )
        lines.append("")
        for method in (HARD, WEIGHTED):
            paired = summary["paired_vs_rices"][method]
            lines.append(
                f"- `{method}` vs RICES overall: {paired['overall']['rescue']} rescue / "
                f"{paired['overall']['harm']} harm, p={paired['overall']['exact_mcnemar_p']:.4g}; "
                f"conflict: {paired['conflict']['rescue']} / {paired['conflict']['harm']}, "
                f"p={paired['conflict']['exact_mcnemar_p']:.4g}."
            )
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Three-shot global weighted diagnostic selection")
    parser.add_argument(
        "--stage",
        choices=("prepare", "validation", "freeze", "test", "all"),
        default="all",
    )
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--precision",
        choices=("bf16", "nf4"),
        default="nf4",
        help="Load the VLM in native bfloat16 or bitsandbytes NF4.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = args.artifact.resolve()
    output = args.output.resolve()
    prepare_all(artifact, output)
    if args.stage == "prepare":
        print((output / "selection_audit.json").read_text(encoding="utf-8"))
        return
    if args.stage in ("validation", "all"):
        model_path = resolve_model(args.model_path)
        evaluate_split(
            output,
            "validation",
            model_path,
            args.batch_size,
            precision=args.precision,
        )
        sweep = validation_sweep(output)
        print(json.dumps(sweep, indent=2))
        if args.stage == "validation":
            return
    if args.stage in ("freeze", "all"):
        model_path = resolve_model(args.model_path)
        protocol = freeze_protocol(
            output, model_path=model_path, precision=args.precision
        )
        print(json.dumps(protocol, indent=2))
        if args.stage == "freeze":
            return
    protocol_path = output / "frozen_protocol.json"
    if not protocol_path.exists():
        raise ValueError("Test evaluation requires frozen_protocol.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    selected_label = str(protocol["selected_lambda"])
    if args.stage in ("test", "all"):
        model_path = resolve_model(args.model_path)
        frozen_precision = protocol.get("precision")
        frozen_model_path = protocol.get("model_path")
        if frozen_precision is not None and frozen_precision != args.precision:
            raise ValueError(
                f"Frozen precision is {frozen_precision!r}, got {args.precision!r}"
            )
        if frozen_model_path is not None and Path(frozen_model_path).resolve() != model_path.resolve():
            raise ValueError("Test model path differs from the validation-frozen model path")
        evaluate_split(
            output,
            "test",
            model_path,
            args.batch_size,
            precision=args.precision,
        )
        validation = summarize_split(output, "validation", selected_label)
        test = summarize_split(output, "test", selected_label)
        report = render_report(validation, test, protocol)
        (output / "REPORT.md").write_text(report, encoding="utf-8")
        print(report)


if __name__ == "__main__":
    main()

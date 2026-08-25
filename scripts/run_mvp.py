from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from behavior_icl import QwenVLBackend, make_single_token_codes


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "artifacts" / "waterbirds"
DEFAULT_MODEL_ROOT = ROOT / "models" / "Qwen3-VL-4B-Instruct"
LABELS = ("landbird", "waterbird")
CONDITIONS = (
    "zero_shot",
    "random_balanced",
    "rices_balanced",
    "A_high_non_diagnostic",
    "A_matched_non_diagnostic",
    "B_high_diagnostic",
    "C_low_diagnostic",
    "D_matched_low_non_diagnostic",
)
PRIMARY_COMPARISONS = (
    ("B_high_diagnostic", "A_matched_non_diagnostic"),
    ("C_low_diagnostic", "A_high_non_diagnostic"),
    ("C_low_diagnostic", "D_matched_low_non_diagnostic"),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def stable_key(identifier: str, seed: int, salt: str = "") -> str:
    return hashlib.sha256(f"{seed}\0{salt}\0{identifier}".encode()).hexdigest()


def unique_units(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    units: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = str(row["unit_id"])
        previous = units.setdefault(identifier, dict(row))
        if previous["role"] != row["role"] or previous["image"] != row["image"]:
            raise ValueError(f"Inconsistent rows for {identifier}")
    return list(units.values())


def load_inputs(artifact: Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    units = unique_units(read_jsonl(artifact / "manifest.jsonl"))
    index = json.loads((artifact / "features" / "siglip" / "index.json").read_text(encoding="utf-8"))
    features = np.load(artifact / "features" / "siglip" / "features.npy")
    if len(index) != len(features):
        raise ValueError("SigLIP index and feature row counts differ")
    feature_by_id = {str(row["unit_id"]): position for position, row in enumerate(index)}
    if len(feature_by_id) != len(index):
        raise ValueError("Duplicate unit IDs in SigLIP index")
    ordered = []
    vectors = []
    for unit in units:
        identifier = str(unit["unit_id"])
        if identifier not in feature_by_id:
            raise ValueError(f"No SigLIP feature for {identifier}")
        ordered.append(unit)
        vectors.append(features[feature_by_id[identifier]])
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("A SigLIP feature has zero norm")
    return ordered, matrix / norms


def choose_queries(
    units: Sequence[Mapping[str, Any]], role: str, per_group: int | None, seed: int
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        if unit["role"] == role:
            grouped[str(unit["group"])].append(dict(unit))
    selected: list[dict[str, Any]] = []
    for group in sorted(grouped):
        rows = sorted(
            grouped[group], key=lambda row: stable_key(str(row["unit_id"]), seed, "query")
        )
        if per_group is None:
            selected.extend(rows)
            continue
        if per_group <= 0:
            raise ValueError("queries_per_group must be positive")
        if len(rows) < per_group:
            raise ValueError(f"Group {group} has {len(rows)} queries, needs {per_group}")
        selected.extend(rows[:per_group])
    return sorted(selected, key=lambda row: str(row["unit_id"]))


def _top(pool: np.ndarray, similarities: np.ndarray) -> int:
    if not len(pool):
        raise ValueError("Empty retrieval pool")
    return int(pool[int(np.argmax(similarities[pool]))])


def _closest(pool: np.ndarray, similarities: np.ndarray, target: float) -> int:
    if not len(pool):
        raise ValueError("Empty matched-similarity pool")
    distances = np.abs(similarities[pool] - target)
    # Prefer higher similarity under an exact distance tie.
    order = np.lexsort((-similarities[pool], distances))
    return int(pool[int(order[0])])


def _quantile(pool: np.ndarray, similarities: np.ndarray, quantile: float) -> int:
    if not len(pool):
        raise ValueError("Empty quantile pool")
    ordered = pool[np.argsort(similarities[pool], kind="stable")]
    position = int(round(quantile * (len(ordered) - 1)))
    return int(ordered[position])


def build_selections(
    units: Sequence[Mapping[str, Any]],
    features: np.ndarray,
    *,
    query_role: str,
    queries_per_group: int | None,
    seed: int,
    low_quantile: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 0.0 <= low_quantile <= 1.0:
        raise ValueError("low_quantile must be in [0, 1]")
    by_id = {str(unit["unit_id"]): dict(unit) for unit in units}
    position = {str(unit["unit_id"]): index for index, unit in enumerate(units)}
    candidate_indices = np.asarray(
        [index for index, unit in enumerate(units) if unit["role"] == "candidate"],
        dtype=np.int64,
    )
    if not len(candidate_indices):
        raise ValueError("No candidate demonstrations")
    query_rows = choose_queries(units, query_role, queries_per_group, seed)
    selections: list[dict[str, Any]] = []

    for query in query_rows:
        query_id = str(query["unit_id"])
        similarities = features @ features[position[query_id]]
        chosen: dict[str, list[int]] = {condition: [] for condition in CONDITIONS}
        for label in LABELS:
            label_pool = candidate_indices[
                np.asarray([str(units[index]["label"]) == label for index in candidate_indices])
            ]
            diagnostic_pool = label_pool[
                np.asarray([int(units[index]["y"]) != int(units[index]["place"]) for index in label_pool])
            ]
            non_diagnostic_pool = label_pool[
                np.asarray([int(units[index]["y"]) == int(units[index]["place"]) for index in label_pool])
            ]
            b_index = _top(diagnostic_pool, similarities)
            c_index = _quantile(diagnostic_pool, similarities, low_quantile)
            chosen["rices_balanced"].append(_top(label_pool, similarities))
            chosen["A_high_non_diagnostic"].append(_top(non_diagnostic_pool, similarities))
            chosen["A_matched_non_diagnostic"].append(
                _closest(non_diagnostic_pool, similarities, float(similarities[b_index]))
            )
            chosen["B_high_diagnostic"].append(b_index)
            chosen["C_low_diagnostic"].append(c_index)
            chosen["D_matched_low_non_diagnostic"].append(
                _closest(non_diagnostic_pool, similarities, float(similarities[c_index]))
            )
            random_order = sorted(
                label_pool,
                key=lambda index: stable_key(
                    str(units[int(index)]["unit_id"]), seed, f"random:{query_id}:{label}"
                ),
            )
            chosen["random_balanced"].append(int(random_order[0]))

        reverse = int(stable_key(query_id, seed, "demo-order")[-1], 16) % 2 == 1
        for condition in CONDITIONS:
            indices = list(chosen[condition])
            if reverse:
                indices.reverse()
            demos = []
            for index in indices:
                candidate = by_id[str(units[index]["unit_id"])]
                demos.append(
                    {
                        "id": str(candidate["unit_id"]),
                        "image": str(candidate["image"]),
                        "label": str(candidate["label"]),
                        "group": str(candidate["group"]),
                        "y": int(candidate["y"]),
                        "place": int(candidate["place"]),
                        "diagnostic": bool(int(candidate["y"]) != int(candidate["place"])),
                        "similarity": float(similarities[index]),
                    }
                )
            selections.append(
                {
                    "query_id": query_id,
                    "query_image": str(query["image"]),
                    "gold": str(query["label"]),
                    "group": str(query["group"]),
                    "y": int(query["y"]),
                    "place": int(query["place"]),
                    "conflict": bool(int(query["y"]) != int(query["place"])),
                    "condition": condition,
                    "demo_order_reversed": reverse,
                    "demos": demos,
                    "mean_similarity": float(np.mean([demo["similarity"] for demo in demos])) if demos else None,
                    "diagnostic_rate": float(np.mean([demo["diagnostic"] for demo in demos])) if demos else None,
                }
            )

    audit = audit_selections(selections, queries_per_group, low_quantile)
    return selections, audit


def audit_selections(
    selections: Sequence[Mapping[str, Any]], queries_per_group: int | None, low_quantile: float
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selections:
        grouped[str(row["condition"])].append(row)
    query_ids = {str(row["query_id"]) for row in selections}
    expected_queries = len(query_ids)
    group_counts: dict[str, int] = defaultdict(int)
    for row in selections:
        if str(row["condition"]) == CONDITIONS[0]:
            group_counts[str(row["group"])] += 1
    if set(grouped) != set(CONDITIONS):
        raise ValueError("Selection output is missing conditions")
    for condition, rows in grouped.items():
        if len(rows) != expected_queries:
            raise ValueError(f"{condition} has {len(rows)} rows, expected {expected_queries}")
        for row in rows:
            demos = list(row["demos"])
            if condition == "zero_shot":
                if demos:
                    raise ValueError("zero_shot unexpectedly has demonstrations")
                continue
            if len(demos) != 2 or {demo["label"] for demo in demos} != set(LABELS):
                raise ValueError(f"{condition} is not two-shot label-balanced")
            if len({demo["id"] for demo in demos}) != 2:
                raise ValueError(f"{condition} repeats a demonstration")
    expected_diag = {
        "A_high_non_diagnostic": 0.0,
        "A_matched_non_diagnostic": 0.0,
        "B_high_diagnostic": 1.0,
        "C_low_diagnostic": 1.0,
        "D_matched_low_non_diagnostic": 0.0,
    }
    for condition, rate in expected_diag.items():
        if any(float(row["diagnostic_rate"]) != rate for row in grouped[condition]):
            raise ValueError(f"{condition} violates its diagnosticity constraint")

    def demo_similarity_map(condition: str) -> dict[tuple[str, str], float]:
        result = {}
        for row in grouped[condition]:
            for demo in row["demos"]:
                result[(str(row["query_id"]), str(demo["label"]))] = float(demo["similarity"])
        return result

    b = demo_similarity_map("B_high_diagnostic")
    a_matched = demo_similarity_map("A_matched_non_diagnostic")
    c = demo_similarity_map("C_low_diagnostic")
    d_matched = demo_similarity_map("D_matched_low_non_diagnostic")
    b_deltas = [abs(value - a_matched[key]) for key, value in b.items()]
    c_deltas = [abs(value - d_matched[key]) for key, value in c.items()]
    summary = {}
    for condition, rows in grouped.items():
        similarities = [float(demo["similarity"]) for row in rows for demo in row["demos"]]
        diagnostics = [float(demo["diagnostic"]) for row in rows for demo in row["demos"]]
        summary[condition] = {
            "queries": len(rows),
            "mean_similarity": float(np.mean(similarities)) if similarities else None,
            "min_similarity": float(np.min(similarities)) if similarities else None,
            "max_similarity": float(np.max(similarities)) if similarities else None,
            "diagnostic_rate": float(np.mean(diagnostics)) if diagnostics else None,
        }
    return {
        "queries": expected_queries,
        "queries_per_group": queries_per_group,
        "query_group_counts": dict(sorted(group_counts.items())),
        "low_similarity_quantile": low_quantile,
        "conditions": summary,
        "B_vs_A_matched_abs_similarity_delta": {
            "mean": float(np.mean(b_deltas)),
            "max": float(np.max(b_deltas)),
        },
        "C_vs_D_matched_abs_similarity_delta": {
            "mean": float(np.mean(c_deltas)),
            "max": float(np.max(c_deltas)),
        },
    }


def resolve_model(path: Path | None) -> Path:
    if path is not None:
        resolved = path.resolve()
        if not resolved.exists():
            raise ValueError(f"Model path does not exist: {resolved}")
        return resolved
    snapshots = sorted(path for path in DEFAULT_MODEL_ROOT.glob("*") if path.is_dir())
    if not snapshots:
        raise ValueError(f"No local Qwen3-VL-4B snapshot under {DEFAULT_MODEL_ROOT}")
    return snapshots[0]


def evaluate(
    selections: Sequence[Mapping[str, Any]],
    *,
    output: Path,
    model_path: Path,
    batch_size: int,
    load_in_4bit: bool,
) -> list[dict[str, Any]]:
    prediction_path = output / "predictions.jsonl"
    existing = read_jsonl(prediction_path) if prediction_path.exists() else []
    keys = {(str(row["query_id"]), str(row["condition"])) for row in existing}
    if len(keys) != len(existing):
        raise ValueError("Duplicate prediction checkpoints")
    by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selections:
        by_condition[str(row["condition"])].append(row)

    backend = QwenVLBackend(model_path, load_in_4bit=load_in_4bit)
    label_codes = make_single_token_codes(backend.processor.tokenizer, len(LABELS))
    output.mkdir(parents=True, exist_ok=True)
    try:
        with prediction_path.open("a", encoding="utf-8") as handle:
            for condition in CONDITIONS:
                pending = [
                    row
                    for row in by_condition[condition]
                    if (str(row["query_id"]), condition) not in keys
                ]
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
                        label_codes,
                        task_instruction="Identify whether the bird is a landbird or waterbird.",
                    )
                    for row, result in zip(batch, results, strict=True):
                        record = {
                            "query_id": row["query_id"],
                            "condition": condition,
                            "gold": row["gold"],
                            "group": row["group"],
                            "conflict": row["conflict"],
                            "prediction": result["prediction"],
                            "correct": result["prediction"] == row["gold"],
                            "probabilities": result["probabilities"],
                            "retrieved_ids": [demo["id"] for demo in row["demos"]],
                            "mean_similarity": row["mean_similarity"],
                            "diagnostic_rate": row["diagnostic_rate"],
                        }
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        keys.add((str(row["query_id"]), condition))
                    handle.flush()
                    os.fsync(handle.fileno())
    finally:
        backend.close()
    return read_jsonl(prediction_path)


def exact_mcnemar_p(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(0, min(left_only, right_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def summarize(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_condition[str(row["condition"])].append(row)
    metrics: dict[str, Any] = {}
    for condition in CONDITIONS:
        rows = by_condition.get(condition, [])
        if not rows:
            continue
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row["group"])].append(row)

        def accuracy(subset: Sequence[Mapping[str, Any]]) -> float | None:
            return float(np.mean([bool(row["correct"]) for row in subset])) if subset else None

        group_accuracy = {group: accuracy(group_rows) for group, group_rows in sorted(groups.items())}
        metrics[condition] = {
            "n": len(rows),
            "overall_accuracy": accuracy(rows),
            "aligned_accuracy": accuracy([row for row in rows if not row["conflict"]]),
            "conflict_accuracy": accuracy([row for row in rows if row["conflict"]]),
            "worst_group_accuracy": min(value for value in group_accuracy.values() if value is not None),
            "group_accuracy": group_accuracy,
        }
    paired: dict[str, Any] = {}
    lookup = {
        (str(row["condition"]), str(row["query_id"])): bool(row["correct"])
        for row in predictions
    }
    for left, right in PRIMARY_COMPARISONS:
        common = sorted(
            {query_id for condition, query_id in lookup if condition == left}
            & {query_id for condition, query_id in lookup if condition == right}
        )
        left_only = sum(lookup[(left, query_id)] and not lookup[(right, query_id)] for query_id in common)
        right_only = sum(lookup[(right, query_id)] and not lookup[(left, query_id)] for query_id in common)
        conflict = [
            row for row in predictions if row["condition"] == left and bool(row["conflict"])
        ]
        conflict_ids = {str(row["query_id"]) for row in conflict}
        aligned_ids = set(common) - conflict_ids
        left_conflict = np.mean([lookup[(left, query_id)] for query_id in conflict_ids]) if conflict_ids else None
        right_conflict = np.mean([lookup[(right, query_id)] for query_id in conflict_ids]) if conflict_ids else None

        def discordance(query_ids: Iterable[str]) -> tuple[int, int, float]:
            identifiers = list(query_ids)
            left_wins = sum(
                lookup[(left, query_id)] and not lookup[(right, query_id)]
                for query_id in identifiers
            )
            right_wins = sum(
                lookup[(right, query_id)] and not lookup[(left, query_id)]
                for query_id in identifiers
            )
            return int(left_wins), int(right_wins), exact_mcnemar_p(int(left_wins), int(right_wins))

        conflict_left_only, conflict_right_only, conflict_p = discordance(conflict_ids)
        aligned_left_only, aligned_right_only, aligned_p = discordance(aligned_ids)
        paired[f"{left}_vs_{right}"] = {
            "n": len(common),
            "overall_accuracy_delta": metrics[left]["overall_accuracy"] - metrics[right]["overall_accuracy"],
            "conflict_accuracy_delta": float(left_conflict - right_conflict) if conflict_ids else None,
            "left_only_correct": int(left_only),
            "right_only_correct": int(right_only),
            "exact_mcnemar_p": exact_mcnemar_p(int(left_only), int(right_only)),
            "conflict_left_only_correct": conflict_left_only,
            "conflict_right_only_correct": conflict_right_only,
            "conflict_exact_mcnemar_p": conflict_p,
            "aligned_left_only_correct": aligned_left_only,
            "aligned_right_only_correct": aligned_right_only,
            "aligned_exact_mcnemar_p": aligned_p,
        }
    return {"conditions": metrics, "paired_comparisons": paired}


def pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.1f}"


def render_report(summary: Mapping[str, Any], audit: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    if audit["queries_per_group"] is None:
        query_description = (
            f"{audit['queries']} (natural groups: "
            + ", ".join(
                f"{group}={count}" for group, count in audit["query_group_counts"].items()
            )
            + ")"
        )
    else:
        query_description = f"{audit['queries']} ({audit['queries_per_group']} per group)"
    lines = [
        "# Waterbirds rule-diagnostic ICL validation pilot",
        "",
        "This is a validation-only MVP. Each non-zero-shot condition uses one demo per class and the same per-query demo order.",
        "",
        "## Configuration",
        "",
        f"- Queries: {query_description}",
        f"- Low-similarity diagnostic quantile: {audit['low_similarity_quantile']}",
        f"- Model: `{config.get('model_path', 'selection only')}`",
        "",
        "## Accuracy (%)",
        "",
        "| Condition | Overall | Aligned | Conflict | Worst group | Mean similarity | Diagnostic rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    metrics = summary.get("conditions", {})
    for condition in CONDITIONS:
        row = metrics.get(condition)
        selection = audit["conditions"][condition]
        if row is None:
            continue
        similarity = selection["mean_similarity"]
        diagnostic_rate = selection["diagnostic_rate"]
        lines.append(
            f"| {condition} | {pct(row['overall_accuracy'])} | {pct(row['aligned_accuracy'])} | "
            f"{pct(row['conflict_accuracy'])} | {pct(row['worst_group_accuracy'])} | "
            f"{'-' if similarity is None else f'{similarity:.4f}'} | {pct(diagnostic_rate)} |"
        )
    lines.extend(["", "## Primary paired comparisons", ""])
    for name, row in summary.get("paired_comparisons", {}).items():
        lines.append(
            f"- `{name}`: overall {row['overall_accuracy_delta'] * 100:+.1f} pp; "
            f"conflict {row['conflict_accuracy_delta'] * 100:+.1f} pp; "
            f"discordant {row['left_only_correct']} vs {row['right_only_correct']}; "
            f"exact McNemar p={row['exact_mcnemar_p']:.4g}; "
            f"conflict-only discordant {row['conflict_left_only_correct']} vs "
            f"{row['conflict_right_only_correct']}, p={row['conflict_exact_mcnemar_p']:.4g}."
        )
    match = audit["B_vs_A_matched_abs_similarity_delta"]
    low_match = audit["C_vs_D_matched_abs_similarity_delta"]
    lines.extend(
        [
            "",
            "## Selection controls",
            "",
            f"- B vs matched-A absolute similarity gap: mean {match['mean']:.6f}, max {match['max']:.6f}.",
            f"- C vs matched-D absolute similarity gap: mean {low_match['mean']:.6f}, max {low_match['max']:.6f}.",
            "- Treat this small pilot as directional evidence; the test set remains untouched.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Waterbirds rule-diagnostic ICL MVP")
    parser.add_argument("--stage", choices=("select", "evaluate", "all"), default="all")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "artifacts" / "validation_qwen4b_q20")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--query-role", default="validation_query", choices=("validation_query", "test_query"))
    parser.add_argument("--queries-per-group", type=int, default=20)
    parser.add_argument(
        "--all-queries",
        action="store_true",
        help="Evaluate every query in the selected role, preserving its natural group distribution",
    )
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--low-quantile", type=float, default=0.30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--no-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = args.artifact.resolve()
    output = args.output.resolve()
    selection_path = output / "selections.jsonl"
    audit_path = output / "selection_audit.json"
    config_path = output / "config.json"
    output.mkdir(parents=True, exist_ok=True)
    queries_per_group = None if args.all_queries else args.queries_per_group

    requested_config = {
        "artifact": str(artifact),
        "output": str(output),
        "query_role": args.query_role,
        "queries_per_group": queries_per_group,
        "all_queries": args.all_queries,
        "seed": args.seed,
        "low_quantile": args.low_quantile,
        "conditions": list(CONDITIONS),
    }
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        for key in ("artifact", "query_role", "queries_per_group", "seed", "low_quantile", "conditions"):
            if previous.get(key) != requested_config.get(key):
                raise ValueError(f"Output config mismatch for {key}; use a new --output directory")

    if args.stage in ("select", "all") or not selection_path.exists():
        units, features = load_inputs(artifact)
        selections, audit = build_selections(
            units,
            features,
            query_role=args.query_role,
            queries_per_group=queries_per_group,
            seed=args.seed,
            low_quantile=args.low_quantile,
        )
        write_jsonl(selection_path, selections)
        write_json(audit_path, audit)
    else:
        selections = read_jsonl(selection_path)
        audit = json.loads(audit_path.read_text(encoding="utf-8"))

    if args.stage == "select":
        write_json(config_path, requested_config)
        print(json.dumps(audit, indent=2))
        return

    model_path = resolve_model(args.model_path)
    requested_config.update(
        {
            "model_path": str(model_path),
            "batch_size": args.batch_size,
            "load_in_4bit": not args.no_4bit,
        }
    )
    write_json(config_path, requested_config)
    predictions = evaluate(
        selections,
        output=output,
        model_path=model_path,
        batch_size=args.batch_size,
        load_in_4bit=not args.no_4bit,
    )
    summary = summarize(predictions)
    write_json(output / "summary.json", summary)
    report = render_report(summary, audit, requested_config)
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

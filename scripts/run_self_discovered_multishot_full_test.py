from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from decision_atoms.config import load_config
from decision_atoms.io import load_jsonl, load_manifest, write_json, write_jsonl
from decision_atoms.math_utils import normalize_rows
from decision_atoms.types import Sample
from run_self_discovered_diagnostic_retrieval import (
    PROMPT_ORDER_CONVENTION,
    file_sha256,
    load_safe_train_and_test,
    predict_selection_rows,
    select_context_rows,
    summarize_predictions,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "waterbirds_self_discovered_qwen3vl4b_bf16.yaml"
DEFAULT_SOURCE = ROOT / "artifacts" / "waterbirds_self_discovered_full"
DEFAULT_OUTPUT = ROOT / "artifacts" / "waterbirds_self_discovered_multishot_full_test"


def parse_shots(value: str) -> list[int]:
    shots = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not shots or shots[0] <= 0:
        raise ValueError("Shot budgets must be positive integers")
    return shots


def _load_frozen_inputs(
    config: Mapping[str, Any], source: Path
) -> tuple[dict[str, Any], list[Sample], list[Sample], np.ndarray, np.ndarray]:
    source_protocol = json.loads(
        (source / "frozen_protocol.json").read_text(encoding="utf-8")
    )
    if source_protocol.get("status") != "frozen_before_test":
        raise ValueError("Source protocol is not frozen_before_test")
    if file_sha256(Path(str(config["manifest"]))) != source_protocol["manifest_sha256"]:
        raise ValueError("Manifest differs from the frozen source run")
    if file_sha256(Path(str(config["_config_path"]))) != source_protocol["config_sha256"]:
        raise ValueError("Config differs from the frozen source run")
    train, test = load_safe_train_and_test(Path(str(config["manifest"])))

    with np.load(source / "siglip2_train_validation_embeddings.npz", allow_pickle=False) as bundle:
        train_ids = [str(value) for value in bundle["sample_ids"]]
        train_values = normalize_rows(np.asarray(bundle["values"], dtype=np.float32))
    train_by_id = {value: index for index, value in enumerate(train_ids)}
    train_embeddings = np.stack([train_values[train_by_id[sample.id]] for sample in train])

    with np.load(source / "siglip2_test_embeddings.npz", allow_pickle=False) as bundle:
        test_ids = [str(value) for value in bundle["sample_ids"]]
        test_values = normalize_rows(np.asarray(bundle["values"], dtype=np.float32))
    test_by_id = {value: index for index, value in enumerate(test_ids)}
    test_embeddings = np.stack([test_values[test_by_id[sample.id]] for sample in test])
    return source_protocol, train, test, train_embeddings, test_embeddings


def select_stage(
    config: Mapping[str, Any], source: Path, output: Path, shots: Sequence[int]
) -> dict[str, Any]:
    source_protocol, train, test, train_embeddings, test_embeddings = _load_frozen_inputs(
        config, source
    )
    score_rows = load_jsonl(source / "train_diagnostic_scores.jsonl")
    score_by_id = {
        str(row["sample_id"]): float(row["learned_diagnostic_score"])
        for row in score_rows
    }
    train_scores = np.asarray([score_by_id[sample.id] for sample in train])
    selected_lambda = float(source_protocol["selected_lambda"])
    rows: list[dict[str, Any]] = []
    for shot in shots:
        selected = select_context_rows(
            test,
            train,
            test_embeddings,
            train_embeddings,
            train_scores,
            (0.0, selected_lambda),
            k=int(shot),
            preserve_baseline_label_counts=bool(
                source_protocol["frozen_parameters"].get(
                    "preserve_rices_demo_label_counts", False
                )
            ),
        )
        for row in selected:
            row["shots"] = int(shot)
        rows.extend(selected)

    output.mkdir(parents=True, exist_ok=True)
    selections_path = output / "selections_label_free.jsonl"
    write_jsonl(selections_path, rows)
    dataset_name = str(
        config.get("dataset_name", Path(str(config["manifest"])).parent.name)
    )
    protocol = {
        "status": "frozen_before_multishot_inference",
        "dataset": dataset_name,
        "split": "test",
        "query_count": len(test),
        "shots": list(map(int, shots)),
        "methods": ["rices", "self_discovered"],
        "context_count": len(rows),
        "selected_lambda": selected_lambda,
        "target_model": source_protocol["target_model"],
        "similarity_model": source_protocol["similarity_model"],
        "selection_rule": source_protocol["test_selection_rule"],
        "prompt_order": PROMPT_ORDER_CONVENTION,
        "budgets_changed_only": True,
        "diagnostic_scores_reused_without_refitting": True,
        "test_labels_groups_or_diagnostic_metadata_used_for_selection": False,
        "test_already_consumed_by_prior_3shot_result": True,
        "scope": "full-test posthoc shot-budget sensitivity analysis",
        "input_sha256": {
            "config": file_sha256(Path(str(config["_config_path"]))),
            "manifest": file_sha256(Path(str(config["manifest"]))),
            "source_protocol": file_sha256(source / "frozen_protocol.json"),
            "diagnostic_scores": file_sha256(source / "train_diagnostic_scores.jsonl"),
            "train_validation_embeddings": file_sha256(
                source / "siglip2_train_validation_embeddings.npz"
            ),
            "test_embeddings": file_sha256(source / "siglip2_test_embeddings.npz"),
        },
        "selection_sha256": file_sha256(selections_path),
    }
    write_json(output / "protocol.json", protocol)
    return protocol


def _verify(config: Mapping[str, Any], source: Path, output: Path) -> dict[str, Any]:
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_multishot_inference":
        raise ValueError("Multi-shot protocol is not frozen")
    checks = {
        "config": Path(str(config["_config_path"])),
        "manifest": Path(str(config["manifest"])),
        "source_protocol": source / "frozen_protocol.json",
        "diagnostic_scores": source / "train_diagnostic_scores.jsonl",
        "train_validation_embeddings": source / "siglip2_train_validation_embeddings.npz",
        "test_embeddings": source / "siglip2_test_embeddings.npz",
    }
    for name, path in checks.items():
        if file_sha256(path) != protocol["input_sha256"][name]:
            raise ValueError(f"Frozen input changed: {name}")
    if file_sha256(output / "selections_label_free.jsonl") != protocol["selection_sha256"]:
        raise ValueError("Frozen multi-shot selections changed")
    return protocol


def infer_stage(
    config: Mapping[str, Any], source: Path, output: Path
) -> dict[str, Any]:
    protocol = _verify(config, source, output)
    train, test = load_safe_train_and_test(Path(str(config["manifest"])))
    selections = load_jsonl(output / "selections_label_free.jsonl")
    cache_path = output / "context_predictions_cache.jsonl"
    if not cache_path.is_file():
        seed = source / "test_context_predictions_cache.jsonl"
        write_jsonl(cache_path, load_jsonl(seed) if seed.is_file() else [])

    predictions: list[dict[str, Any]] = []
    for shot in protocol["shots"]:
        shot_rows = [row for row in selections if int(row["shots"]) == int(shot)]
        frozen = predict_selection_rows(
            config,
            [*train, *test],
            shot_rows,
            selected_lambda=float(protocol["selected_lambda"]),
            cache_path=cache_path,
        )
        for row in frozen:
            row["shots"] = int(shot)
        predictions.extend(frozen)
        write_jsonl(output / "predictions_label_free.jsonl", predictions)
        print(
            f"completed {shot}-shot: {len(predictions)}/{len(selections)} condition rows",
            flush=True,
        )
    result = {
        "completed": True,
        "prediction_rows": len(predictions),
        "prediction_sha256": file_sha256(output / "predictions_label_free.jsonl"),
        "runtime_icl_batch_size": int(
            config["target_model"].get("icl_batch_size", 1)
        ),
    }
    write_json(output / "inference_status.json", result)
    return result


def _retrieval_audit(
    rows: Sequence[Mapping[str, Any]], sample_by_id: Mapping[str, Sample]
) -> dict[str, Any]:
    demos = [demo for row in rows for demo in row["demos"]]
    flags = [
        bool(sample_by_id[str(demo["id"])].metadata["diagnostic"])
        for demo in demos
    ]
    contexts_with_diagnostic = sum(
        any(
            bool(sample_by_id[str(demo["id"])].metadata["diagnostic"])
            for demo in row["demos"]
        )
        for row in rows
    )
    label_slots = {
        label: sum(str(demo["label"]) == label for demo in demos)
        for label in sorted({str(demo["label"]) for demo in demos})
    }
    return {
        "demo_slots": len(flags),
        "diagnostic_demo_rate": float(np.mean(flags)),
        "contexts_with_diagnostic_demo": contexts_with_diagnostic,
        "context_diagnostic_rate": contexts_with_diagnostic / len(rows),
        "mean_similarity": float(np.mean([row["mean_similarity"] for row in rows])),
        "mean_learned_diagnostic_score": float(
            np.mean([row["mean_learned_diagnostic_score"] for row in rows])
        ),
        "unique_demo_ids": len({str(demo["id"]) for demo in demos}),
        "demo_label_slots": label_slots,
        "mean_distinct_demo_labels_per_context": float(
            np.mean([len({str(demo["label"]) for demo in row["demos"]}) for row in rows])
        ),
    }


def paired_delta_statistics(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline: str = "rices",
    method: str = "self_discovered",
    bootstraps: int = 200_000,
    seed: int = 20260828,
) -> dict[str, Any]:
    """Paired accuracy-difference interval from rescue/harm/tie counts.

    Sampling the three paired outcomes is equivalent to resampling queries, but
    avoids allocating a bootstraps-by-query matrix for the full test set.
    """

    if bootstraps <= 0:
        raise ValueError("bootstraps must be positive")
    by_query: dict[str, dict[str, bool]] = {}
    for row in rows:
        condition = str(row["condition"])
        if condition not in {baseline, method}:
            continue
        by_query.setdefault(str(row["query_id"]), {})[condition] = bool(row["correct"])
    if not by_query or any(set(value) != {baseline, method} for value in by_query.values()):
        raise ValueError("Every query must have exactly one baseline and method outcome")
    rescued = sum(not value[baseline] and value[method] for value in by_query.values())
    harmed = sum(value[baseline] and not value[method] for value in by_query.values())
    tied = len(by_query) - rescued - harmed
    probabilities = np.asarray([rescued, harmed, tied], dtype=np.float64) / len(by_query)
    draws = np.random.default_rng(seed).multinomial(
        len(by_query), probabilities, size=bootstraps
    )
    deltas = (draws[:, 0] - draws[:, 1]) / len(by_query)
    low, high = np.quantile(deltas, [0.025, 0.975])
    changed = rescued + harmed
    if changed:
        # Two-sided exact sign-test p-value, evaluated in log space for stability.
        from scipy.stats import binomtest

        p_value = float(binomtest(rescued, changed, 0.5).pvalue)
    else:
        p_value = 1.0
    return {
        "queries": len(by_query),
        "rescued": rescued,
        "harmed": harmed,
        "tied": tied,
        "accuracy_delta": (rescued - harmed) / len(by_query),
        "paired_bootstrap_95_ci": [float(low), float(high)],
        "paired_exact_sign_test_p": p_value,
        "bootstrap_samples": bootstraps,
        "seed": seed,
    }


def report_stage(config: Mapping[str, Any], source: Path, output: Path) -> dict[str, Any]:
    protocol = _verify(config, source, output)
    predictions = load_jsonl(output / "predictions_label_free.jsonl")
    if len(predictions) != int(protocol["context_count"]):
        raise ValueError("Multi-shot predictions are incomplete")
    reporting = load_manifest(config["manifest"])
    sample_by_id = {sample.id: sample for sample in reporting}
    selections = load_jsonl(output / "selections_label_free.jsonl")
    by_shot: dict[str, Any] = {}
    scored_rows: list[dict[str, Any]] = []
    for shot in protocol["shots"]:
        shot_predictions = [
            row for row in predictions if int(row["shots"]) == int(shot)
        ]
        scored, summary = summarize_predictions(
            shot_predictions,
            reporting,
            selected_lambda=float(protocol["selected_lambda"]),
            scope=(
                f"{protocol.get('dataset', 'dataset')} full test "
                f"{shot}-shot sensitivity analysis"
            ),
            test_split_used=True,
        )
        for row in scored:
            row["shots"] = int(shot)
        scored_rows.extend(scored)
        cell: dict[str, Any] = summary
        for condition, value in (("rices", 0.0), ("self_discovered", protocol["selected_lambda"])):
            selected = [
                row
                for row in selections
                if int(row["shots"]) == int(shot)
                and float(row["lambda"]) == float(value)
            ]
            cell[condition]["retrieval_audit"] = _retrieval_audit(
                selected, sample_by_id
            )
        cell["gain"] = {
            "accuracy": (
                cell["self_discovered"]["accuracy"] - cell["rices"]["accuracy"]
            ),
            "worst_group_accuracy": (
                cell["self_discovered"]["worst_group_accuracy"]
                - cell["rices"]["worst_group_accuracy"]
            ),
        }
        cell["paired_inference"] = paired_delta_statistics(
            scored, seed=20260828 + int(shot)
        )
        by_shot[str(shot)] = cell
    write_jsonl(output / "predictions_scored_posthoc.jsonl", scored_rows)
    summary = {
        "protocol": protocol,
        "target_model": protocol["target_model"],
        "similarity_model": protocol["similarity_model"],
        "by_shot": by_shot,
        "query_metadata_joined_only_after_all_predictions_completed": True,
        "interpretation_scope": (
            "posthoc full-test shot-budget sensitivity; not independent confirmation"
        ),
    }
    write_json(output / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen self-discovered diagnostic retrieval across shot budgets"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("select", "infer", "report", "all"), default="all")
    parser.add_argument("--shots", default="1,2,3,4,8")
    parser.add_argument(
        "--icl-batch-size",
        type=int,
        default=None,
        help="Runtime-only batching override; does not change prompts, selections, or scores.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.icl_batch_size is not None:
        if args.icl_batch_size <= 0:
            raise ValueError("--icl-batch-size must be positive")
        config["target_model"] = dict(config["target_model"])
        config["target_model"]["icl_batch_size"] = int(args.icl_batch_size)
    if args.stage in {"select", "all"}:
        result = select_stage(config, args.source, args.output, parse_shots(args.shots))
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if args.stage in {"infer", "all"}:
        result = infer_stage(config, args.source, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if args.stage in {"report", "all"}:
        result = report_stage(config, args.source, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

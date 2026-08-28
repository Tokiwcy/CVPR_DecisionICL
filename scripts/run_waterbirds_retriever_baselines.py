from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from decision_atoms.backends import build_retriever
from decision_atoms.config import load_config
from decision_atoms.io import load_jsonl, load_manifest, write_json, write_jsonl
from run_self_discovered_diagnostic_retrieval import (
    file_sha256,
    load_safe_train_and_test,
    predict_selection_rows,
)
from run_self_discovered_multishot_full_test import (
    _load_frozen_inputs,
    paired_delta_statistics,
    parse_shots,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "waterbirds_self_discovered_qwen3vl4b_bf16.yaml"
DEFAULT_SOURCE = ROOT / "artifacts" / "waterbirds_self_discovered_full"
DEFAULT_BASE = ROOT / "artifacts" / "waterbirds_self_discovered_multishot_full_test"
DEFAULT_OUTPUT = ROOT / "artifacts" / "waterbirds_retriever_baselines"
METHODS = ("random", "class_balanced_rices", "diagnostic_only", "mmices")
AUXILIARY_PROMPT_ORDER = "method_native_or_descending_score"
REFERENCE_PROMPT_ORDER = "ascending_similarity_most_similar_nearest_query"


def deterministic_random_indices(
    query_id: str, population: int, k: int, *, seed: int
) -> np.ndarray:
    if k <= 0 or k > population:
        raise ValueError("k must be in [1, population]")
    digest = hashlib.sha256(f"{seed}:{query_id}".encode("utf-8")).digest()
    local_seed = int.from_bytes(digest[:8], "little", signed=False)
    return np.random.default_rng(local_seed).choice(population, size=k, replace=False)


def class_balanced_rices_indices(
    similarities: np.ndarray,
    labels: Sequence[str],
    train_ids: Sequence[str],
    *,
    k: int,
) -> np.ndarray:
    if len(similarities) != len(labels) or len(labels) != len(train_ids):
        raise ValueError("similarities, labels, and train_ids must align")
    classes = sorted(set(labels))
    if not classes or k <= 0 or k > len(labels):
        raise ValueError("invalid class-balanced selection request")
    by_class = {
        label: np.asarray([i for i, value in enumerate(labels) if value == label])
        for label in classes
    }
    counts = {label: k // len(classes) for label in classes}
    remainder = k - sum(counts.values())
    class_priority = sorted(
        classes,
        key=lambda label: (
            -float(np.max(similarities[by_class[label]])),
            label,
        ),
    )
    for label in class_priority[:remainder]:
        counts[label] += 1
    chosen: list[int] = []
    for label in classes:
        candidates = by_class[label]
        order = sorted(
            map(int, candidates),
            key=lambda index: (-float(similarities[index]), str(train_ids[index])),
        )
        chosen.extend(order[: counts[label]])
    # Match the released RICES convention: highest-similarity example first.
    return np.asarray(
        sorted(chosen, key=lambda index: (-float(similarities[index]), train_ids[index])),
        dtype=np.int64,
    )


def diagnostic_only_indices(
    scores: np.ndarray,
    train_ids: Sequence[str],
    *,
    k: int,
) -> np.ndarray:
    if len(scores) != len(train_ids):
        raise ValueError("scores and train_ids must align")
    return np.asarray(sorted(
        range(len(scores)),
        key=lambda index: (
            -float(scores[index]),
            str(train_ids[index]),
        ),
    )[:k], dtype=np.int64)


def mmices_indices(
    similarities: np.ndarray,
    task_affinity: np.ndarray,
    train_ids: Sequence[str],
    *,
    k: int,
    candidate_pool_size: int = 1024,
) -> np.ndarray:
    """Official MMICES image filter followed by question-image reranking."""

    if len(similarities) != len(task_affinity) or len(similarities) != len(train_ids):
        raise ValueError("similarities, task_affinity, and train_ids must align")
    pool_k = min(max(candidate_pool_size, k * 8), len(similarities))
    candidates = sorted(
        range(len(similarities)),
        key=lambda index: (-float(similarities[index]), str(train_ids[index])),
    )[:pool_k]
    return np.asarray(
        sorted(
            candidates,
            key=lambda index: (-float(task_affinity[index]), str(train_ids[index])),
        )[:k],
        dtype=np.int64,
    )


def select_stage(
    config: Mapping[str, Any],
    source: Path,
    output: Path,
    shots: Sequence[int],
    *,
    seed: int,
) -> dict[str, Any]:
    _, train, test, train_embeddings, test_embeddings = _load_frozen_inputs(
        config, source
    )
    score_by_id = {
        str(row["sample_id"]): float(row["learned_diagnostic_score"])
        for row in load_jsonl(source / "train_diagnostic_scores.jsonl")
    }
    train_scores = np.asarray([score_by_id[sample.id] for sample in train])
    train_ids = [sample.id for sample in train]
    train_labels = [sample.label for sample in train]
    similarity_matrix = test_embeddings @ train_embeddings.T
    retriever = build_retriever(dict(config["similarity_model"]))
    try:
        task_vector = retriever.embed_text(
            ["What is the category of the bird in this image?"]
        )[0].astype(np.float32)
    finally:
        del retriever
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    task_affinity = train_embeddings @ task_vector
    diagnostic_by_shot = {
        int(shot): diagnostic_only_indices(train_scores, train_ids, k=int(shot))
        for shot in shots
    }
    rows: list[dict[str, Any]] = []
    for query_index, query in enumerate(test):
        similarities = similarity_matrix[query_index]
        for shot in shots:
            selections = {
                "random": deterministic_random_indices(
                    query.id, len(train), int(shot), seed=seed
                ),
                "class_balanced_rices": class_balanced_rices_indices(
                    similarities, train_labels, train_ids, k=int(shot)
                ),
                "diagnostic_only": diagnostic_by_shot[int(shot)],
                "mmices": mmices_indices(
                    similarities, task_affinity, train_ids, k=int(shot)
                ),
            }
            for method, indices in selections.items():
                rows.append(
                    {
                        "query_id": query.id,
                        "gold": "",
                        "lambda": 1.0,
                        "method": method,
                        "shots": int(shot),
                        "demos": [
                            {
                                "id": train[int(index)].id,
                                "label": train[int(index)].label,
                                "similarity": float(similarities[int(index)]),
                                "learned_diagnostic_score": float(
                                    train_scores[int(index)]
                                ),
                            }
                            for index in indices
                        ],
                        "mean_similarity": float(np.mean(similarities[indices])),
                        "mean_learned_diagnostic_score": float(
                            np.mean(train_scores[indices])
                        ),
                        "prompt_order": AUXILIARY_PROMPT_ORDER,
                        "query_label_group_or_diagnostic_metadata_used": False,
                    }
                )
    output.mkdir(parents=True, exist_ok=True)
    selections_path = output / "selections_label_free.jsonl"
    write_jsonl(selections_path, rows)
    protocol = {
        "status": "frozen_before_baseline_inference",
        "dataset": "Waterbirds",
        "split": "test",
        "query_count": len(test),
        "shots": list(map(int, shots)),
        "methods": list(METHODS),
        "seed": seed,
        "context_count": len(rows),
        "same_target_vlm_and_classification_prompt": True,
        "prompt_order": AUXILIARY_PROMPT_ORDER,
        "reference_rices_prompt_order": REFERENCE_PROMPT_ORDER,
        "prompt_order_matched_to_reference": False,
        "comparison_limitation": (
            "The auxiliary selectors use method-native/descending score order, "
            "whereas the referenced RICES contexts place the most similar demo "
            "nearest the query. Report deltas as order-confounded until rerun "
            "under one shared ordering convention."
        ),
        "mmices_rule": (
            "top-1024 query-image candidates, reranked by fixed Waterbirds "
            "question-to-candidate-image SigLIP similarity"
        ),
        "muier_equivalence_note": (
            "MUIER is ranking-equivalent to RICES here because every training "
            "question is the same; its query-image-to-training-text term is constant"
        ),
        "test_labels_groups_or_diagnostic_metadata_used_for_selection": False,
        "scope": "posthoc retriever-baseline comparison on an already consumed test set",
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
    if protocol.get("status") != "frozen_before_baseline_inference":
        raise ValueError("Retriever-baseline protocol is not frozen")
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
        raise ValueError("Frozen retriever-baseline selections changed")
    return protocol


def infer_stage(
    config: Mapping[str, Any], source: Path, base: Path, output: Path
) -> dict[str, Any]:
    protocol = _verify(config, source, output)
    train, test = load_safe_train_and_test(Path(str(config["manifest"])))
    selections = load_jsonl(output / "selections_label_free.jsonl")
    cache_path = output / "context_predictions_cache.jsonl"
    if not cache_path.is_file():
        seed_cache = base / "context_predictions_cache.jsonl"
        write_jsonl(cache_path, load_jsonl(seed_cache) if seed_cache.is_file() else [])
    predictions: list[dict[str, Any]] = []
    for shot in protocol["shots"]:
        for method in protocol["methods"]:
            cell = [
                row
                for row in selections
                if int(row["shots"]) == int(shot) and row["method"] == method
            ]
            frozen = predict_selection_rows(
                config,
                [*train, *test],
                cell,
                selected_lambda=1.0,
                cache_path=cache_path,
            )
            for row in frozen:
                row["condition"] = method
                row["shots"] = int(shot)
            predictions.extend(frozen)
            write_jsonl(output / "predictions_label_free.jsonl", predictions)
            print(f"completed {method} {shot}-shot", flush=True)
    result = {
        "complete": True,
        "prediction_rows": len(predictions),
        "prediction_sha256": file_sha256(output / "predictions_label_free.jsonl"),
    }
    write_json(output / "inference_status.json", result)
    return result


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        by_group[str(row["group"])].append(bool(row["correct"]))
    group_accuracy = {
        group: float(np.mean(values)) for group, values in sorted(by_group.items())
    }
    return {
        "count": len(rows),
        "accuracy": float(np.mean([bool(row["correct"]) for row in rows])),
        "group_accuracy": group_accuracy,
        "worst_group_accuracy": min(group_accuracy.values()),
    }


def report_stage(
    config: Mapping[str, Any], source: Path, base: Path, output: Path
) -> dict[str, Any]:
    protocol = _verify(config, source, output)
    predictions = load_jsonl(output / "predictions_label_free.jsonl")
    if len(predictions) != int(protocol["context_count"]):
        raise ValueError("Retriever-baseline predictions are incomplete")
    samples = load_manifest(config["manifest"])
    by_id = {sample.id: sample for sample in samples}
    selections = load_jsonl(output / "selections_label_free.jsonl")
    base_rows = load_jsonl(base / "predictions_scored_posthoc.jsonl")
    scored: list[dict[str, Any]] = []
    by_shot: dict[str, Any] = {}
    for shot in protocol["shots"]:
        rices = [
            row
            for row in base_rows
            if int(row["shots"]) == int(shot) and row["condition"] == "rices"
        ]
        shot_result: dict[str, Any] = {"rices": _metrics(rices)}
        for method in protocol["methods"]:
            cell: list[dict[str, Any]] = []
            for row in predictions:
                if int(row["shots"]) != int(shot) or row["condition"] != method:
                    continue
                sample = by_id[str(row["query_id"])]
                value = dict(row) | {
                    "gold": sample.label,
                    "group": sample.group,
                    "correct": str(row["prediction"]) == sample.label,
                }
                cell.append(value)
                scored.append(value)
            metrics = _metrics(cell)
            selected = [
                row
                for row in selections
                if int(row["shots"]) == int(shot) and row["method"] == method
            ]
            diagnostic_flags = [
                bool(by_id[str(demo["id"])].metadata["diagnostic"])
                for row in selected
                for demo in row["demos"]
            ]
            metrics["retrieval_audit"] = {
                "mean_similarity": float(
                    np.mean([row["mean_similarity"] for row in selected])
                ),
                "mean_learned_diagnostic_score": float(
                    np.mean(
                        [row["mean_learned_diagnostic_score"] for row in selected]
                    )
                ),
                "diagnostic_demo_rate": float(np.mean(diagnostic_flags)),
            }
            paired_rows = [
                {
                    "query_id": str(row["query_id"]),
                    "condition": "rices",
                    "correct": bool(row["correct"]),
                }
                for row in rices
            ] + [
                {
                    "query_id": str(row["query_id"]),
                    "condition": method,
                    "correct": bool(row["correct"]),
                }
                for row in cell
            ]
            metrics["paired_vs_rices"] = paired_delta_statistics(
                paired_rows,
                method=method,
                seed=20260828 + int(shot) + METHODS.index(method) * 100,
            )
            metrics["gain_vs_rices"] = {
                "accuracy": metrics["accuracy"] - shot_result["rices"]["accuracy"],
                "worst_group_accuracy": metrics["worst_group_accuracy"]
                - shot_result["rices"]["worst_group_accuracy"],
            }
            shot_result[method] = metrics
        by_shot[str(shot)] = shot_result
    write_jsonl(output / "predictions_scored_posthoc.jsonl", scored)
    summary = {
        "protocol": protocol,
        "by_shot": by_shot,
        "query_metadata_joined_only_after_all_predictions_completed": True,
        "comparison_note": protocol["comparison_limitation"],
    }
    write_json(output / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Random, class-balanced RICES, and diagnostic-only Waterbirds baselines"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--stage", choices=("select", "infer", "report", "all"), default="all"
    )
    parser.add_argument("--shots", default="3")
    parser.add_argument("--seed", type=int, default=20260828)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.stage in {"select", "all"}:
        result = select_stage(
            config,
            args.source,
            args.output,
            parse_shots(args.shots),
            seed=args.seed,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if args.stage in {"infer", "all"}:
        result = infer_stage(config, args.source, args.base, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if args.stage in {"report", "all"}:
        result = report_stage(config, args.source, args.base, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

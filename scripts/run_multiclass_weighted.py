from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from behavior_icl import QwenVLBackend, make_single_token_codes


ROOT = Path(__file__).resolve().parents[1]

from decision_atoms.backends import build_retriever, build_target
from decision_atoms.config import load_config
from decision_atoms.influence import predicted_label, true_margin
from decision_atoms.io import load_jsonl, load_manifest, write_json, write_jsonl
from decision_atoms.types import Sample


K = 3
RICES = "rices_global_3shot"
HARD = "hard_coverage_global_3shot"
WEIGHTED = "weighted_model_error_global_3shot"
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


def _release(value: Any) -> None:
    close = getattr(value, "close", None)
    if callable(close):
        close()
    del value
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class FastQwenTarget:
    """Use semantic class names with scalable single-token output codes."""

    def __init__(self, config: Mapping[str, Any], labels: Sequence[str]) -> None:
        quantization = str(config.get("quantization", "none")).lower()
        self.backend = QwenVLBackend(
            str(config["model"]),
            load_in_4bit=quantization in {"4bit", "nf4"},
            min_pixels=int(config.get("min_pixels", 32 * 28 * 28)),
            max_pixels=int(config.get("max_pixels", 128 * 28 * 28)),
        )
        self.labels = list(labels)
        self.codes = make_single_token_codes(
            self.backend.processor.tokenizer, len(self.labels)
        )
        self.task_instruction = str(
            config.get(
                "task_instruction",
                "Identify the most specific visual category shown in the image.",
            )
        )

    def _score(
        self, requests: list[tuple[list[dict[str, Any]], dict[str, Any]]]
    ) -> list[dict[str, float]]:
        results = self.backend.score_named_classes_batch(
            requests,
            self.labels,
            self.codes,
            task_instruction=self.task_instruction,
        )
        return [dict(row["probabilities"]) for row in results]

    def class_scores_batch(
        self, images: list[str], labels: list[str]
    ) -> list[dict[str, float]]:
        if labels != self.labels:
            raise ValueError("FastQwenTarget labels changed after initialization")
        return self._score([([], {"path": image, "text": ""}) for image in images])

    def icl_scores_batch(
        self,
        queries: list[Sample],
        demos: list[list[Sample]],
        labels: list[str],
    ) -> list[dict[str, float]]:
        if labels != self.labels:
            raise ValueError("FastQwenTarget labels changed after initialization")
        requests = []
        for query, query_demos in zip(queries, demos, strict=True):
            demo_rows = [
                {"path": demo.image, "text": demo.text, "label": demo.label}
                for demo in query_demos
            ]
            requests.append((demo_rows, {"path": query.image, "text": query.text}))
        return self._score(requests)

    def close(self) -> None:
        self.backend.close()


def _build_target(config: Mapping[str, Any], labels: Sequence[str]) -> Any:
    target_config = dict(config["target_model"])
    if target_config.get("kind", "qwen3_vl") == "qwen3_vl":
        return FastQwenTarget(target_config, labels)
    return build_target(target_config)


def _config_fingerprint(config: Mapping[str, Any]) -> str:
    payload = {
        "config_path": config.get("_config_path"),
        "manifest": config["manifest"],
        "labels": config["labels"],
        "target_model": config["target_model"],
        "similarity_model": config["similarity_model"],
        "diagnostic": config.get("diagnostic"),
        "selection_method_name": config.get("selection_method_name"),
        "prompt_shots": int(config.get("prompt_shots", 3)),
        "lambda_selection_objective": config.get("lambda_selection_objective"),
        "max_aligned_accuracy_drop": config.get("max_aligned_accuracy_drop"),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def weighted_condition(label: str) -> str:
    return f"{WEIGHTED}__lambda_{label.replace('.', 'p')}"


def _diagnostic_spec(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = config.get("diagnostic")
    if raw is None:
        return {
            "kind": "model_error",
            "definition": (
                "model-specific train-prior error: the frozen target model's no-demo "
                "prediction for the labeled training image differs from its gold label"
            ),
        }
    if not isinstance(raw, Mapping):
        raise TypeError("config.diagnostic must be a mapping")
    spec = dict(raw)
    kind = str(spec.get("kind", "model_error"))
    if kind not in {"model_error", "manifest_boolean"}:
        raise ValueError(f"Unsupported diagnostic kind: {kind}")
    spec["kind"] = kind
    if kind == "manifest_boolean":
        spec.setdefault("metadata_key", "diagnostic")
    return spec


def _diagnostics_for_train(
    config: Mapping[str, Any],
    train: Sequence[Sample],
    priors: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, str]:
    spec = _diagnostic_spec(config)
    definition = str(
        spec.get(
            "definition",
            "manifest-provided counter-spurious indicator on each labeled training demo",
        )
    )
    if spec["kind"] == "model_error":
        prior_by_id = {str(row["sample_id"]): row for row in priors}
        missing = [sample.id for sample in train if sample.id not in prior_by_id]
        if missing:
            raise ValueError(f"Missing train-prior predictions for {len(missing)} samples")
        values = [not bool(prior_by_id[sample.id]["correct"]) for sample in train]
    else:
        key = str(spec["metadata_key"])
        missing = [sample.id for sample in train if key not in sample.metadata]
        if missing:
            raise ValueError(
                f"Manifest metadata key {key!r} is missing for {len(missing)} train samples"
            )
        values = [bool(sample.metadata[key]) for sample in train]
    diagnostics = np.asarray(values, dtype=bool)
    if not diagnostics.any():
        raise ValueError("The training pool has no diagnostic demonstrations")
    if diagnostics.all():
        raise ValueError("Every training demo is diagnostic; the indicator cannot rank demos")
    return diagnostics, definition


def _condition_path(output: Path, split: str, condition: str) -> Path:
    return output / "selections" / f"{split}__{condition}.jsonl"


def _load_embeddings(path: Path, samples: Sequence[Sample]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as bundle:
        values = np.asarray(bundle["values"], dtype=np.float32)
        ids = list(bundle["sample_ids"].astype(str))
    expected = [sample.id for sample in samples]
    if ids != expected:
        raise ValueError("Embedding cache IDs do not match the manifest order")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-12, None)


def ensure_embeddings(
    config: Mapping[str, Any], samples: Sequence[Sample], cache_path: Path
) -> np.ndarray:
    if cache_path.is_file():
        return _load_embeddings(cache_path, samples)
    retriever = build_retriever(dict(config["similarity_model"]))
    try:
        values = retriever.embed([sample.image for sample in samples]).astype(np.float32)
    finally:
        _release(retriever)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        values=values,
        sample_ids=np.asarray([sample.id for sample in samples], dtype=str),
    )
    return _load_embeddings(cache_path, samples)


def score_train_prior(
    config: Mapping[str, Any],
    samples: Sequence[Sample],
    output: Path,
    wrapper_batch_size: int,
    cache_path: Path | None = None,
) -> list[dict[str, Any]]:
    path = cache_path or (output / "train_prior_predictions.jsonl")
    train = [sample for sample in samples if sample.split == "train"]
    existing = load_jsonl(path) if path.is_file() else []
    by_id = {str(row["sample_id"]): row for row in existing}
    pending = [sample for sample in train if sample.id not in by_id]
    if pending:
        labels = list(config["labels"])
        target = _build_target(config, labels)
        try:
            for begin in range(0, len(pending), wrapper_batch_size):
                batch = pending[begin : begin + wrapper_batch_size]
                scores_batch = target.class_scores_batch(
                    [sample.image for sample in batch], labels
                )
                for sample, scores in zip(batch, scores_batch, strict=True):
                    prediction = predicted_label(scores, labels)
                    by_id[sample.id] = {
                        "sample_id": sample.id,
                        "gold": sample.label,
                        "prediction": prediction,
                        "correct": prediction == sample.label,
                        "true_margin": true_margin(scores, sample.label, labels),
                    }
                write_jsonl(path, [by_id[sample.id] for sample in train if sample.id in by_id])
                print(f"train prior {len(by_id)}/{len(train)}", flush=True)
        finally:
            _release(target)
    rows = load_jsonl(path)
    if len(rows) != len(train):
        raise ValueError(f"Expected {len(train)} train priors, found {len(rows)}")
    return rows


def _top_indices(
    scores: np.ndarray,
    similarities: np.ndarray,
    ids: np.ndarray,
    limit: int | None = None,
) -> np.ndarray:
    # Lexicographic order: descending score, descending raw similarity, ascending ID.
    if limit is None:
        limit = K
    if limit < 1:
        raise ValueError("limit must be positive")
    if scores.size < limit:
        raise ValueError(f"Need at least {limit} candidates, found {scores.size}")
    partition = np.argpartition(-scores, limit - 1)[:limit]
    threshold = float(np.min(scores[partition]))
    candidates = np.flatnonzero(scores >= threshold)
    order = np.lexsort(
        (ids[candidates], -similarities[candidates], -scores[candidates])
    )
    return candidates[order[:limit]]


def _prompt_order(indices: Sequence[int], similarities: np.ndarray, ids: np.ndarray) -> list[int]:
    return sorted(indices, key=lambda index: (float(similarities[index]), str(ids[index])))


def _selection_row(
    query: Sample,
    train: Sequence[Sample],
    train_ids: np.ndarray,
    indices: Sequence[int],
    similarities: np.ndarray,
    diagnostics: np.ndarray,
    *,
    condition: str,
    lambda_value: float | None = None,
) -> dict[str, Any]:
    ordered = _prompt_order(indices, similarities, train_ids)
    demos = [
        {
            "id": train[index].id,
            "image": train[index].image,
            "label": train[index].label,
            "similarity": float(similarities[index]),
            "diagnostic": bool(diagnostics[index]),
        }
        for index in ordered
    ]
    row: dict[str, Any] = {
        "query_id": query.id,
        "query_image": query.image,
        "gold": query.label,
        "condition": condition,
        "demos": demos,
        "mean_similarity": float(np.mean([demo["similarity"] for demo in demos])),
        "diagnostic_rate": float(np.mean([demo["diagnostic"] for demo in demos])),
        "prompt_order": "ascending_similarity_most_similar_nearest_query",
        "retrieval_scope": "global_complete_training_pool_no_class_quota",
    }
    if query.group is not None:
        row["group"] = query.group
    if "conflict" in query.metadata:
        row["conflict"] = bool(query.metadata["conflict"])
    if "spurious_attribute" in query.metadata:
        row["spurious_attribute"] = str(query.metadata["spurious_attribute"])
    if lambda_value is not None:
        row["lambda"] = lambda_value
    return row


def prepare_selections(
    config: Mapping[str, Any],
    samples: Sequence[Sample],
    embeddings: np.ndarray,
    priors: Sequence[Mapping[str, Any]],
    output: Path,
) -> dict[str, Any]:
    paths = {
        (split, condition): _condition_path(output, split, condition)
        for split in ("validation", "test")
        for condition in (RICES, HARD, *(weighted_condition(label) for label, _ in LAMBDA_GRID))
    }
    audit_path = output / "selection_audit.json"
    if audit_path.is_file() and all(path.is_file() for path in paths.values()):
        return json.loads(audit_path.read_text(encoding="utf-8"))
    sample_position = {sample.id: index for index, sample in enumerate(samples)}
    train = [sample for sample in samples if sample.split == "train"]
    train_ids = np.asarray([sample.id for sample in train], dtype=str)
    train_embeddings = np.asarray(
        [embeddings[sample_position[sample.id]] for sample in train], dtype=np.float32
    )
    diagnostics, diagnostic_definition = _diagnostics_for_train(
        config, train, priors
    )
    diagnostic_positions = np.flatnonzero(diagnostics)
    nondiagnostic_positions = np.flatnonzero(~diagnostics)
    output_rows: dict[tuple[str, str], list[dict[str, Any]]] = {
        key: [] for key in paths
    }
    audit: dict[str, Any] = {
        "diagnostic_definition": diagnostic_definition,
        "train_size": len(train),
        "diagnostic_train_count": int(diagnostics.sum()),
        "diagnostic_train_rate": float(diagnostics.mean()),
        "splits": {},
    }
    for split in ("validation", "test"):
        queries = [sample for sample in samples if sample.split == split]
        injected = 0
        similarity_losses: list[float] = []
        for query in queries:
            query_vector = embeddings[sample_position[query.id]]
            similarities = train_embeddings @ query_vector
            rices_indices = _top_indices(similarities, similarities, train_ids)
            top_diagnostic_local = _top_indices(
                similarities[diagnostic_positions],
                similarities[diagnostic_positions],
                train_ids[diagnostic_positions],
                min(K, diagnostic_positions.size),
            )
            top_diagnostic = diagnostic_positions[top_diagnostic_local]
            top_nondiagnostic_local = _top_indices(
                similarities[nondiagnostic_positions],
                similarities[nondiagnostic_positions],
                train_ids[nondiagnostic_positions],
                min(K, nondiagnostic_positions.size),
            )
            top_nondiagnostic = nondiagnostic_positions[top_nondiagnostic_local]
            weighted_candidates = np.concatenate(
                (top_diagnostic, top_nondiagnostic)
            )
            rices_row = _selection_row(
                query,
                train,
                train_ids,
                rices_indices,
                similarities,
                diagnostics,
                condition=RICES,
            )
            output_rows[(split, RICES)].append(rices_row)
            if diagnostics[rices_indices].any():
                hard_indices = rices_indices
            else:
                best_diagnostic = int(top_diagnostic[0])
                replace = int(rices_indices[np.argmin(similarities[rices_indices])])
                hard_indices = np.asarray(
                    [best_diagnostic if int(index) == replace else int(index) for index in rices_indices]
                )
                injected += 1
                similarity_losses.append(
                    float(similarities[replace] - similarities[best_diagnostic])
                )
            output_rows[(split, HARD)].append(
                _selection_row(
                    query,
                    train,
                    train_ids,
                    hard_indices,
                    similarities,
                    diagnostics,
                    condition=HARD,
                )
            )
            for lambda_label, lambda_value in LAMBDA_GRID:
                candidate_diagnostics = diagnostics[weighted_candidates]
                candidate_similarities = similarities[weighted_candidates]
                candidate_scores = (
                    candidate_similarities
                    + lambda_value * candidate_diagnostics.astype(np.float32)
                )
                local_indices = _top_indices(
                    candidate_scores,
                    candidate_similarities,
                    train_ids[weighted_candidates],
                )
                indices = weighted_candidates[local_indices]
                condition = weighted_condition(lambda_label)
                output_rows[(split, condition)].append(
                    _selection_row(
                        query,
                        train,
                        train_ids,
                        indices,
                        similarities,
                        diagnostics,
                        condition=condition,
                        lambda_value=lambda_value,
                    )
                )
        for condition in (RICES, HARD, *(weighted_condition(label) for label, _ in LAMBDA_GRID)):
            write_jsonl(paths[(split, condition)], output_rows[(split, condition)])
        audit["splits"][split] = {
            "queries": len(queries),
            "hard_coverage_injections": injected,
            "hard_coverage_mean_similarity_loss": (
                float(np.mean(similarity_losses)) if similarity_losses else 0.0
            ),
            "conditions": {
                condition: {
                    "mean_similarity": float(
                        np.mean([row["mean_similarity"] for row in output_rows[(split, condition)]])
                    ),
                    "mean_diagnostic_rate": float(
                        np.mean([row["diagnostic_rate"] for row in output_rows[(split, condition)]])
                    ),
                    "contexts_with_diagnostic": sum(
                        row["diagnostic_rate"] > 0 for row in output_rows[(split, condition)]
                    ),
                }
                for condition in (RICES, HARD, *(weighted_condition(label) for label, _ in LAMBDA_GRID))
            },
        }
    write_json(audit_path, audit)
    return audit


def _context_key(row: Mapping[str, Any]) -> str:
    return f"{row['query_id']}|" + ",".join(str(demo["id"]) for demo in row["demos"])


def _append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def score_conditions(
    config: Mapping[str, Any],
    samples: Sequence[Sample],
    output: Path,
    split: str,
    conditions: Sequence[str],
    wrapper_batch_size: int,
) -> None:
    sample_by_id = {sample.id: sample for sample in samples}
    selections = {
        condition: load_jsonl(_condition_path(output, split, condition))
        for condition in conditions
    }
    cache_path = output / "predictions" / split / "context_cache.jsonl"
    existing = load_jsonl(cache_path) if cache_path.is_file() else []
    by_key = {str(row["context_key"]): row for row in existing}
    jobs_by_key: dict[str, Mapping[str, Any]] = {}
    for rows in selections.values():
        for row in rows:
            key = _context_key(row)
            if key not in by_key:
                jobs_by_key.setdefault(key, row)
    jobs = list(jobs_by_key.items())
    if jobs:
        labels = list(config["labels"])
        target = _build_target(config, labels)
        try:
            for begin in range(0, len(jobs), wrapper_batch_size):
                batch = jobs[begin : begin + wrapper_batch_size]
                queries = [sample_by_id[str(row["query_id"])] for _, row in batch]
                demo_batches = [
                    [sample_by_id[str(demo["id"])] for demo in row["demos"]]
                    for _, row in batch
                ]
                score_rows = target.icl_scores_batch(queries, demo_batches, labels)
                checkpoint_rows: list[dict[str, Any]] = []
                for (key, row), scores in zip(batch, score_rows, strict=True):
                    prediction = predicted_label(scores, labels)
                    checkpoint = {
                        "context_key": key,
                        "query_id": row["query_id"],
                        "gold": row["gold"],
                        "prediction": prediction,
                        "correct": prediction == row["gold"],
                        "scores": {label: float(scores[label]) for label in labels},
                        "retrieved_ids": [demo["id"] for demo in row["demos"]],
                    }
                    by_key[key] = checkpoint
                    checkpoint_rows.append(checkpoint)
                _append_jsonl(cache_path, checkpoint_rows)
                print(
                    f"{split} unique contexts {len(by_key)}/{len(existing) + len(jobs)}",
                    flush=True,
                )
        finally:
            _release(target)
    for condition, rows in selections.items():
        materialized = []
        for row in rows:
            prediction = by_key[_context_key(row)]
            materialized.append(
                {
                    **prediction,
                    "condition": condition,
                    "mean_similarity": row["mean_similarity"],
                    "diagnostic_rate": row["diagnostic_rate"],
                    **({"group": row["group"]} if "group" in row else {}),
                    **({"conflict": row["conflict"]} if "conflict" in row else {}),
                    **(
                        {"spurious_attribute": row["spurious_attribute"]}
                        if "spurious_attribute" in row
                        else {}
                    ),
                }
            )
        write_jsonl(output / "predictions" / split / f"{condition}.jsonl", materialized)


def metrics(rows: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot compute metrics for an empty prediction set")
    by_label: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[str(row["gold"])].append(row)
    class_accuracy = {
        label: float(np.mean([bool(row["correct"]) for row in by_label[label]]))
        for label in labels
        if by_label[label]
    }
    result: dict[str, Any] = {
        "n": len(rows),
        "overall_accuracy": float(np.mean([bool(row["correct"]) for row in rows])),
        "macro_accuracy": float(np.mean(list(class_accuracy.values()))),
        "worst_class_accuracy": float(min(class_accuracy.values())),
        "class_accuracy": class_accuracy,
    }
    if all(row.get("group") is not None for row in rows):
        by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            by_group[str(row["group"])].append(row)
        group_accuracy = {
            group: float(np.mean([bool(row["correct"]) for row in values]))
            for group, values in sorted(by_group.items())
        }
        result.update(
            {
                "group_accuracy": group_accuracy,
                "mean_group_accuracy": float(np.mean(list(group_accuracy.values()))),
                "worst_group_accuracy": float(min(group_accuracy.values())),
            }
        )
    if all("conflict" in row for row in rows):
        aligned = [row for row in rows if not bool(row["conflict"])]
        conflict = [row for row in rows if bool(row["conflict"])]
        result["aligned_accuracy"] = (
            float(np.mean([bool(row["correct"]) for row in aligned])) if aligned else None
        )
        result["conflict_accuracy"] = (
            float(np.mean([bool(row["correct"]) for row in conflict])) if conflict else None
        )
    return result


def paired(baseline: Sequence[Mapping[str, Any]], method: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    base = {str(row["query_id"]): row for row in baseline}
    other = {str(row["query_id"]): row for row in method}
    if set(base) != set(other):
        raise ValueError("Paired query IDs differ")
    return {
        "rescue": sum(not base[key]["correct"] and other[key]["correct"] for key in base),
        "harm": sum(base[key]["correct"] and not other[key]["correct"] for key in base),
    }


def validation_sweep(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    labels = list(config["labels"])
    prediction_dir = output / "predictions" / "validation"
    rices = load_jsonl(prediction_dir / f"{RICES}.jsonl")
    hard = load_jsonl(prediction_dir / f"{HARD}.jsonl")
    grid = []
    for lambda_label, lambda_value in LAMBDA_GRID:
        condition = weighted_condition(lambda_label)
        rows = load_jsonl(prediction_dir / f"{condition}.jsonl")
        grid.append(
            {
                "lambda": lambda_label,
                "lambda_value": lambda_value,
                "metrics": metrics(rows, labels),
                "paired_vs_rices": paired(rices, rows),
            }
        )
    rices_metrics = metrics(rices, labels)
    objective = str(config.get("lambda_selection_objective", "macro_accuracy"))
    if objective == "group_robustness":
        if "mean_group_accuracy" not in rices_metrics:
            raise ValueError("group_robustness selection requires manifest group labels")
        max_aligned_drop = float(config.get("max_aligned_accuracy_drop", 0.01))
        rices_aligned = rices_metrics.get("aligned_accuracy")
        eligible = grid
        if rices_aligned is not None:
            eligible = [
                row
                for row in grid
                if row["metrics"].get("aligned_accuracy") is not None
                and float(row["metrics"]["aligned_accuracy"])
                >= float(rices_aligned) - max_aligned_drop
            ]
        if not eligible:
            raise ValueError("No lambda satisfies the aligned-accuracy guard")
        selected = max(
            eligible,
            key=lambda row: (
                row["metrics"]["mean_group_accuracy"],
                row["metrics"]["worst_group_accuracy"],
                -1.0
                if row["metrics"].get("conflict_accuracy") is None
                else row["metrics"]["conflict_accuracy"],
                row["metrics"]["overall_accuracy"],
                -row["lambda_value"],
            ),
        )
        if max_aligned_drop >= 1.0:
            selection_rule = (
                "maximize validation mean-group accuracy; tie-break by worst-group, "
                "conflict, overall, then smaller lambda"
            )
        else:
            selection_rule = (
                "among lambdas with aligned accuracy no more than "
                f"{100 * max_aligned_drop:g} pp below RICES, maximize validation "
                "mean-group accuracy; tie-break by worst-group, conflict, overall, "
                "then smaller lambda"
            )
    elif objective == "macro_accuracy":
        selected = max(
            grid,
            key=lambda row: (
                row["metrics"]["macro_accuracy"],
                row["metrics"]["overall_accuracy"],
                row["metrics"]["worst_class_accuracy"],
                -row["lambda_value"],
            ),
        )
        selection_rule = (
            "maximize validation macro accuracy; tie-break by overall accuracy, "
            "worst-class accuracy, then smaller lambda"
        )
    else:
        raise ValueError(f"Unsupported lambda_selection_objective: {objective}")
    result = {
        "selection_rule": selection_rule,
        "rices_metrics": rices_metrics,
        "hard_coverage_metrics": metrics(hard, labels),
        "hard_coverage_paired_vs_rices": paired(rices, hard),
        "grid": grid,
        "selected_lambda": selected["lambda"],
        "selected_lambda_value": selected["lambda_value"],
        "selected_validation_metrics": selected["metrics"],
    }
    write_json(output / "validation_lambda_sweep.json", result)
    return result


def freeze_protocol(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    sweep = validation_sweep(config, output)
    diagnostic = _diagnostic_spec(config)
    diagnostic_definition = str(
        diagnostic.get(
            "definition",
            "manifest-provided counter-spurious indicator on each labeled training demo",
        )
    )
    protocol = {
        "protocol_status": "lambda_frozen_on_validation_before_test",
        "task": Path(str(config["manifest"])).parent.name,
        "model": config["target_model"].get("model"),
        "quantization": config["target_model"].get("quantization", "none"),
        "dtype": config["target_model"].get("dtype", "auto"),
        "config_fingerprint": _config_fingerprint(config),
        "shots": K,
        "retrieval_scope": "global top-k over complete training pool; no class quota",
        "prompt_order": "ascending similarity; most similar demo nearest query",
        "diagnostic_kind": diagnostic["kind"],
        "diagnostic_definition": diagnostic_definition,
        "objective": "similarity + lambda * indicator(demo is diagnostic)",
        "weighted_selection_method": WEIGHTED,
        "lambda_grid": [label for label, _ in LAMBDA_GRID],
        "lambda_selection_rule": sweep["selection_rule"],
        "selected_lambda": sweep["selected_lambda"],
        "selected_lambda_value": sweep["selected_lambda_value"],
        "test_tuning_prohibited": True,
    }
    write_json(output / "frozen_protocol.json", protocol)
    return protocol


def summarize(config: Mapping[str, Any], output: Path, split: str, lambda_label: str) -> dict[str, Any]:
    labels = list(config["labels"])
    prediction_dir = output / "predictions" / split
    rices = load_jsonl(prediction_dir / f"{RICES}.jsonl")
    hard = load_jsonl(prediction_dir / f"{HARD}.jsonl")
    weighted = load_jsonl(prediction_dir / f"{weighted_condition(lambda_label)}.jsonl")
    result = {
        "split": split,
        "lambda": lambda_label,
        "methods": {
            RICES: metrics(rices, labels),
            HARD: metrics(hard, labels),
            WEIGHTED: metrics(weighted, labels),
        },
        "paired_vs_rices": {
            HARD: paired(rices, hard),
            WEIGHTED: paired(rices, weighted),
        },
    }
    write_json(output / f"{split}_summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnostic-aware weighted top-k ICL")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--manifest-override",
        type=Path,
        help="Use an equivalent manifest at a different readable image root.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument(
        "--train-prior-cache",
        type=Path,
        help="Optional model-error prior cache shared by shot configurations.",
    )
    parser.add_argument(
        "--stage", choices=("prepare", "validation", "freeze", "test", "all"), default="all"
    )
    parser.add_argument("--wrapper-batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    global K, RICES, HARD, WEIGHTED
    args = parse_args()
    config = load_config(args.config)
    K = int(config.get("prompt_shots", 3))
    if K < 1:
        raise ValueError("prompt_shots must be positive")
    RICES = f"rices_global_{K}shot"
    HARD = f"hard_coverage_global_{K}shot"
    WEIGHTED = str(
        config.get("selection_method_name", f"weighted_model_error_global_{K}shot")
    )
    if not WEIGHTED or any(char in WEIGHTED for char in "/\\"):
        raise ValueError("selection_method_name must be a non-empty filename-safe name")
    if args.manifest_override is not None:
        config["manifest"] = str(args.manifest_override.resolve())
    output = args.output.resolve()
    samples = load_manifest(config["manifest"])
    embeddings = ensure_embeddings(config, samples, args.embedding_cache.resolve())
    diagnostic = _diagnostic_spec(config)
    priors = (
        score_train_prior(
            config,
            samples,
            output,
            args.wrapper_batch_size,
            args.train_prior_cache.resolve() if args.train_prior_cache is not None else None,
        )
        if diagnostic["kind"] == "model_error"
        else []
    )
    prepare_selections(config, samples, embeddings, priors, output)
    write_json(
        output / "run_metadata.json",
        {
            "config_path": str(args.config.resolve()),
            "config_fingerprint": _config_fingerprint(config),
            "manifest": config["manifest"],
            "embedding_cache": str(args.embedding_cache.resolve()),
            "train_prior_cache": (
                str(args.train_prior_cache.resolve())
                if args.train_prior_cache is not None
                else str((output / "train_prior_predictions.jsonl").resolve())
            ),
            "shots": K,
            "diagnostic": diagnostic,
            "selection_method_name": WEIGHTED,
        },
    )
    if args.stage == "prepare":
        return
    validation_conditions = [
        RICES,
        HARD,
        *(weighted_condition(label) for label, _ in LAMBDA_GRID),
    ]
    if args.stage in ("validation", "all"):
        score_conditions(
            config,
            samples,
            output,
            "validation",
            validation_conditions,
            args.wrapper_batch_size,
        )
        validation_sweep(config, output)
        if args.stage == "validation":
            return
    if args.stage in ("freeze", "all"):
        protocol = freeze_protocol(config, output)
        if args.stage == "freeze":
            print(json.dumps(protocol, ensure_ascii=False, indent=2))
            return
    protocol_path = output / "frozen_protocol.json"
    if not protocol_path.is_file():
        raise ValueError("Test evaluation requires frozen_protocol.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["config_fingerprint"] != _config_fingerprint(config):
        raise ValueError("Test config differs from the validation-frozen config")
    selected = str(protocol["selected_lambda"])
    if args.stage in ("test", "all"):
        score_conditions(
            config,
            samples,
            output,
            "test",
            [RICES, HARD, weighted_condition(selected)],
            args.wrapper_batch_size,
        )
        validation = summarize(config, output, "validation", selected)
        test = summarize(config, output, "test", selected)
        print(json.dumps({"validation": validation, "test": test}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

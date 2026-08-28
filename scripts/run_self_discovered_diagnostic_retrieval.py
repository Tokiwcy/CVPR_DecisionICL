from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from decision_atoms.backends import build_retriever, build_target
from decision_atoms.config import load_config
from decision_atoms.influence import predicted_label
from decision_atoms.io import load_jsonl, load_manifest, write_json, write_jsonl
from decision_atoms.math_utils import binary_auc, normalize_rows
from decision_atoms.types import DecisionAtom, Sample


LAMBDA_GRID = (0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3)

# Demonstrations are rendered so the most similar example is adjacent to the
# query. Keep this convention machine-readable: an auxiliary baseline is not
# order-matched unless it records the same value.
PROMPT_ORDER_CONVENTION = "ascending_similarity_most_similar_nearest_query"


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


def metadata_free_projection(samples: Sequence[Sample]) -> list[Sample]:
    """Keep only fields authorized for discovery and retrieval."""
    return [
        Sample(
            id=sample.id,
            split=sample.split,
            image=sample.image,
            label=sample.label,
            text=sample.text,
            options=sample.options,
        )
        for sample in samples
    ]


def inference_query_projection(samples: Sequence[Sample]) -> list[Sample]:
    """Remove query labels and all evaluation metadata before model inference."""
    return [
        Sample(
            id=sample.id,
            split=sample.split,
            image=sample.image,
            label="",
            text=sample.text,
            options=sample.options,
        )
        for sample in samples
    ]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def extract_json_array(text: str) -> list[Any]:
    def decode(value: str) -> list[Any] | None:
        decoder = json.JSONDecoder()
        for position, character in enumerate(value):
            if character != "[":
                continue
            try:
                parsed, _ = decoder.raw_decode(value[position:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, list):
                return parsed
        return None

    parsed = decode(text)
    if parsed is not None:
        return parsed
    # Qwen occasionally emits one otherwise complete object as
    #   "concept text", "a_side": ..., "b_side": ... ]
    # Repair only that narrow structural error; do not invent or rewrite content.
    repaired = re.sub(
        r'(\},\s*)"([^"\r\n]+)"\s*,\s*"a_side"\s*:',
        r'\1{"concept": "\2", "a_side":',
        text,
    )
    repaired = re.sub(
        r'("b_side"\s*:\s*"(?:[^"\\]|\\.)*")\s*\]\s*,\s*\{',
        r'\1}, {',
        repaired,
    )
    parsed = decode(repaired)
    if parsed is not None:
        return parsed
    raise ValueError("Response does not contain a valid JSON array")


def parse_atoms(
    text: str,
    label_a: str,
    label_b: str,
    *,
    bootstrap: int,
) -> list[DecisionAtom]:
    output: list[DecisionAtom] = []
    seen: set[tuple[str, str, str]] = set()
    for index, raw in enumerate(extract_json_array(text)):
        if not isinstance(raw, Mapping):
            continue
        fields = [str(raw.get(key, "")).strip() for key in ("concept", "a_side", "b_side")]
        if not all(fields):
            continue
        key = tuple(value.casefold() for value in fields)
        if key in seen:
            continue
        seen.add(key)
        digest = hashlib.sha256("\n".join(key).encode("utf-8")).hexdigest()[:12]
        anchor = hashlib.sha256(label_a.encode("utf-8")).hexdigest()[:6]
        raw_role = str(raw.get("role", "uncertain")).strip().casefold()
        semantic_role = raw_role if raw_role in {"object", "context"} else "uncertain"
        output.append(
            DecisionAtom(
                id=f"shift-{anchor}-b{bootstrap:02d}-{index:02d}-{digest}",
                label_a=label_a,
                label_b=label_b,
                concept=fields[0],
                a_side=fields[1],
                b_side=fields[2],
                source_mode="shift_audit",
                bootstrap=bootstrap,
                provenance={
                    "discovered_without_group_metadata": True,
                    "vlm_semantic_role": semantic_role,
                },
            )
        )
    if not output:
        raise ValueError("VLM response contains no valid visual atoms")
    return output


def deterministic_sample(
    samples: Sequence[Sample],
    *,
    split: str,
    label: str,
    count: int,
    seed: int,
    exclude: set[str] | None = None,
) -> list[Sample]:
    excluded = exclude or set()
    pool = [
        sample
        for sample in samples
        if sample.split == split and sample.label == label and sample.id not in excluded
    ]
    ranked = sorted(
        pool,
        key=lambda sample: hashlib.sha256(
            f"{seed}|{sample.id}".encode("utf-8")
        ).hexdigest(),
    )
    if len(ranked) < count:
        raise ValueError(f"Need {count} {split}/{label} samples, found {len(ranked)}")
    return ranked[:count]


def balanced_rest_support(
    samples: Sequence[Sample],
    labels: Sequence[str],
    *,
    anchor_label: str,
    count: int,
    seed: int,
    exclude: set[str] | None = None,
) -> list[Sample]:
    other_labels = [label for label in labels if label != anchor_label]
    if not other_labels:
        raise ValueError("One-vs-rest support requires at least two labels")
    base, remainder = divmod(count, len(other_labels))
    output: list[Sample] = []
    for index, label in enumerate(other_labels):
        take = base + int(index < remainder)
        if take:
            output.extend(
                deterministic_sample(
                    samples,
                    split="train",
                    label=label,
                    count=take,
                    seed=seed + index * 9173,
                    exclude=(exclude or set()) | {sample.id for sample in output},
                )
            )
    return output


def discover_atoms_stage(
    config: Mapping[str, Any],
    samples: Sequence[Sample],
    output_dir: Path,
    *,
    bootstraps: int,
    support_per_class: int,
    atoms_per_bootstrap: int,
    seed: int,
) -> list[DecisionAtom]:
    atoms_path = output_dir / "atoms.jsonl"
    samples = metadata_free_projection(samples)
    labels = [str(value) for value in config["labels"]]
    if len(labels) < 2:
        raise ValueError("At least two labels are required")
    anchors = labels[:1] if len(labels) == 2 else labels
    atoms = (
        [DecisionAtom.from_dict(row) for row in load_jsonl(atoms_path)]
        if atoms_path.is_file()
        else []
    )
    responses_path = output_dir / "discovery_responses.jsonl"
    responses = load_jsonl(responses_path) if responses_path.is_file() else []
    completed = {
        (str(row["anchor_label"]), int(row["bootstrap"])) for row in responses
    }
    expected = {(anchor, bootstrap) for anchor in anchors for bootstrap in range(bootstraps)}
    if completed == expected:
        return atoms
    if not completed.issubset(expected):
        raise ValueError("Existing discovery responses do not match the requested protocol")
    used = {
        str(sample_id)
        for row in responses
        for side in ("anchor", "rest")
        for sample_id in row["support_ids"][side]
    }
    model_config = dict(config["target_model"])
    model_config["max_new_tokens"] = max(768, int(model_config.get("max_new_tokens", 0)))
    failures_path = output_dir / "discovery_parse_failures.jsonl"
    failures = load_jsonl(failures_path) if failures_path.is_file() else []
    target = None
    try:
        for anchor_index, anchor_label in enumerate(anchors):
            comparison_label = labels[1] if len(labels) == 2 else f"not {anchor_label}"
            for bootstrap in range(bootstraps):
                if (anchor_label, bootstrap) in completed:
                    continue
                support_a = deterministic_sample(
                    samples,
                    split="train",
                    label=anchor_label,
                    count=support_per_class,
                    seed=seed + bootstrap * 1009 + anchor_index * 65537,
                    exclude=used,
                )
                used.update(sample.id for sample in support_a)
                if len(labels) == 2:
                    support_b = deterministic_sample(
                        samples,
                        split="train",
                        label=labels[1],
                        count=support_per_class,
                        seed=seed + bootstrap * 1009 + 9173,
                        exclude=used,
                    )
                else:
                    support_b = balanced_rest_support(
                        samples,
                        labels,
                        anchor_label=anchor_label,
                        count=support_per_class,
                        seed=seed + bootstrap * 1009 + anchor_index * 65537 + 9173,
                        exclude=used,
                    )
                used.update(sample.id for sample in support_b)
                matching_failures = [
                    row
                    for row in failures
                    if str(row["anchor_label"]) == anchor_label
                    and int(row["bootstrap"]) == bootstrap
                ]
                raw = str(matching_failures[-1]["raw_response"]) if matching_failures else ""
                try:
                    parsed = parse_atoms(
                        raw, anchor_label, comparison_label, bootstrap=bootstrap
                    )
                except ValueError:
                    if target is None:
                        target = build_target(model_config)
                    raw = target.discover(
                        [sample.image for sample in support_a],
                        [sample.image for sample in support_b],
                        anchor_label,
                        comparison_label,
                        "shift_audit",
                        atoms_per_bootstrap,
                    )
                    try:
                        parsed = parse_atoms(
                            raw, anchor_label, comparison_label, bootstrap=bootstrap
                        )
                    except ValueError:
                        append_jsonl(
                            failures_path,
                            [
                                {
                                    "anchor_label": anchor_label,
                                    "bootstrap": bootstrap,
                                    "raw_response": raw,
                                }
                            ],
                        )
                        raise
                atoms.extend(parsed)
                responses.append(
                    {
                        "anchor_label": anchor_label,
                        "comparison_label": comparison_label,
                        "bootstrap": bootstrap,
                        "support_ids": {
                            "anchor": [sample.id for sample in support_a],
                            "rest": [sample.id for sample in support_b],
                        },
                        "support_labels": {
                            "anchor": [sample.label for sample in support_a],
                            "rest": [sample.label for sample in support_b],
                        },
                        "raw_response": raw,
                        "parsed_atom_ids": [atom.id for atom in parsed],
                        "selection_inputs": ["split", "class_label", "stable_sample_id_hash"],
                        "group_or_spurious_metadata_used": False,
                    }
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                write_jsonl(atoms_path, [atom.to_dict() for atom in atoms])
                write_jsonl(responses_path, responses)
                print(
                    f"discovery {anchor_label} bootstrap {bootstrap + 1}/{bootstraps}: "
                    f"{len(parsed)} atoms",
                    flush=True,
                )
    finally:
        if target is not None:
            _release(target)
    return atoms


def ensure_embeddings(
    config: Mapping[str, Any], samples: Sequence[Sample], cache_path: Path
) -> tuple[np.ndarray, Any]:
    expected_ids = [sample.id for sample in samples]
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as bundle:
            cached_ids = list(bundle["sample_ids"].astype(str))
            if cached_ids == expected_ids:
                return normalize_rows(np.asarray(bundle["values"], dtype=np.float32)), None
    retriever = build_retriever(dict(config["similarity_model"]))
    values = retriever.embed([sample.image for sample in samples]).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        values=values,
        sample_ids=np.asarray(expected_ids, dtype=str),
    )
    return normalize_rows(values), retriever


def atom_evidence_from_embeddings(
    image_embeddings: np.ndarray,
    atoms: Sequence[DecisionAtom],
    text_embeddings: np.ndarray,
) -> np.ndarray:
    if text_embeddings.shape[0] != 2 * len(atoms):
        raise ValueError("Expected two text embeddings per atom")
    text = normalize_rows(text_embeddings)
    directions = normalize_rows(text[0::2] - text[1::2])
    return normalize_rows(image_embeddings) @ directions.T


def characterize_atoms(
    evidence: np.ndarray,
    samples: Sequence[Sample],
    atoms: Sequence[DecisionAtom],
    *,
    core_min_auc: float,
    core_max_train_validation_gap: float,
    shortcut_min_train_auc: float,
    shortcut_max_validation_strength: float,
    shortcut_min_gap: float,
    semantic_shortcut_fallback: bool = False,
) -> list[dict[str, Any]]:
    if evidence.shape != (len(samples), len(atoms)):
        raise ValueError("Evidence matrix shape does not align with samples and atoms")
    label_a = atoms[0].label_a
    train = np.asarray([sample.split == "train" for sample in samples])
    validation = np.asarray([sample.split == "validation" for sample in samples])
    targets = np.asarray([sample.label == label_a for sample in samples])
    if not train.any() or not validation.any():
        raise ValueError("Both train and validation samples are required")
    rows: list[dict[str, Any]] = []
    for index, atom in enumerate(atoms):
        train_auc = binary_auc(evidence[train, index], targets[train])
        validation_auc = binary_auc(evidence[validation, index], targets[validation])
        validation_orientation = 1.0 if validation_auc >= 0.5 else -1.0
        train_orientation = 1.0 if train_auc >= 0.5 else -1.0
        oriented_validation_auc = max(validation_auc, 1.0 - validation_auc)
        train_in_validation_orientation = (
            train_auc if validation_orientation > 0 else 1.0 - train_auc
        )
        oriented_train_auc = max(train_auc, 1.0 - train_auc)
        validation_in_train_orientation = (
            validation_auc if train_orientation > 0 else 1.0 - validation_auc
        )
        core_gap = abs(oriented_validation_auc - train_in_validation_orientation)
        core = (
            oriented_validation_auc >= core_min_auc
            and train_in_validation_orientation >= 0.5
            and core_gap <= core_max_train_validation_gap
        )
        shortcut_gap = oriented_train_auc - validation_in_train_orientation
        shift_shortcut = (
            not core
            and oriented_train_auc >= shortcut_min_train_auc
            and abs(validation_in_train_orientation - 0.5)
            <= shortcut_max_validation_strength
            and shortcut_gap >= shortcut_min_gap
        )
        semantic_role = str(atom.provenance.get("vlm_semantic_role", "uncertain"))
        semantic_shortcut = (
            semantic_shortcut_fallback
            and semantic_role == "context"
            and oriented_train_auc >= shortcut_min_train_auc
        )
        # A highly class-predictive context atom is not an intrinsic core cue,
        # even when train and validation preserve the same shortcut correlation.
        if semantic_shortcut:
            core = False
        shortcut = shift_shortcut or semantic_shortcut
        role = "core" if core else "shortcut" if shortcut else "rejected"
        orientation = validation_orientation if core else train_orientation
        weight = (
            2.0 * (oriented_validation_auc - 0.5)
            if core
            else (
                2.0 * (oriented_train_auc - 0.5)
                if semantic_shortcut
                else max(0.0, shortcut_gap)
            )
            if shortcut
            else 0.0
        )
        rows.append(
            {
                **atom.to_dict(),
                "train_auc": float(train_auc),
                "validation_auc": float(validation_auc),
                "oriented_train_auc": float(oriented_train_auc),
                "oriented_validation_auc": float(oriented_validation_auc),
                "validation_auc_in_train_orientation": float(validation_in_train_orientation),
                "core_train_validation_gap": float(core_gap),
                "shortcut_gap": float(shortcut_gap),
                "semantic_role": semantic_role,
                "semantic_shortcut_fallback_enabled": semantic_shortcut_fallback,
                "role_source": (
                    "vlm_semantic_context_prior+train_correlation"
                    if semantic_shortcut
                    else "train_validation_instability"
                    if shift_shortcut
                    else "train_validation_stability"
                    if core
                    else "rejected"
                ),
                "role": role,
                "orientation": float(orientation),
                "weight": float(weight),
                "role_selection_uses": ["train split", "validation split", "class labels", "VLM-visible evidence"],
                "group_or_spurious_metadata_used": False,
            }
        )
    return rows


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if not len(weights):
        raise ValueError("Cannot aggregate an empty atom set")
    return (values * weights[None, :]).sum(axis=1) / max(float(weights.sum()), 1e-12)


def percentile_scores(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks / max(len(values) - 1, 1)


def diagnostic_scores(
    evidence: np.ndarray,
    samples: Sequence[Sample],
    quality: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    core_indices = np.asarray(
        [index for index, row in enumerate(quality) if row["role"] == "core"], dtype=int
    )
    shortcut_indices = np.asarray(
        [index for index, row in enumerate(quality) if row["role"] == "shortcut"], dtype=int
    )
    if not len(core_indices):
        raise ValueError("No held-out-stable core atoms were discovered")
    if not len(shortcut_indices):
        raise ValueError("No train-only shortcut atoms were discovered")
    train = np.asarray([sample.split == "train" for sample in samples])
    means = evidence[train].mean(axis=0)
    scales = evidence[train].std(axis=0)
    standardized = (evidence - means[None, :]) / np.maximum(scales[None, :], 1e-6)
    label_a = str(quality[0]["label_a"])
    label_sign = np.asarray(
        [1.0 if sample.label == label_a else -1.0 for sample in samples]
    )
    core_orientation = np.asarray(
        [float(quality[index]["orientation"]) for index in core_indices]
    )
    core_weights = np.asarray([float(quality[index]["weight"]) for index in core_indices])
    shortcut_orientation = np.asarray(
        [float(quality[index]["orientation"]) for index in shortcut_indices]
    )
    shortcut_weights = np.asarray(
        [float(quality[index]["weight"]) for index in shortcut_indices]
    )
    core_support = label_sign * _weighted_mean(
        standardized[:, core_indices] * core_orientation[None, :], core_weights
    )
    shortcut_support = label_sign * _weighted_mean(
        standardized[:, shortcut_indices] * shortcut_orientation[None, :], shortcut_weights
    )
    shortcut_conflict = -shortcut_support
    raw = np.maximum(core_support, 0.0) * np.maximum(shortcut_conflict, 0.0)
    train_scores = percentile_scores(raw[train])
    scores = np.zeros(len(samples), dtype=np.float64)
    scores[train] = train_scores
    return scores, raw, core_support, shortcut_conflict


def grouped_diagnostic_scores(
    evidence: np.ndarray,
    samples: Sequence[Sample],
    atoms: Sequence[DecisionAtom],
    *,
    core_min_auc: float,
    core_max_train_validation_gap: float,
    shortcut_min_train_auc: float,
    shortcut_max_validation_strength: float,
    shortcut_min_gap: float,
    semantic_shortcut_fallback: bool = False,
) -> tuple[
    list[dict[str, Any]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, dict[str, Any]],
]:
    anchors = list(dict.fromkeys(atom.label_a for atom in atoms))
    if not anchors:
        raise ValueError("No atoms were provided")
    quality: list[dict[str, Any]] = []
    scores = np.zeros(len(samples), dtype=np.float64)
    raw = np.zeros(len(samples), dtype=np.float64)
    core_support = np.zeros(len(samples), dtype=np.float64)
    shortcut_conflict = np.zeros(len(samples), dtype=np.float64)
    status: dict[str, dict[str, Any]] = {}
    single_binary_pair = len(anchors) == 1 and len(
        {sample.label for sample in samples if sample.split == "train"}
    ) == 2
    for anchor in anchors:
        indices = [index for index, atom in enumerate(atoms) if atom.label_a == anchor]
        anchor_atoms = [atoms[index] for index in indices]
        anchor_quality = characterize_atoms(
            evidence[:, indices],
            samples,
            anchor_atoms,
            core_min_auc=core_min_auc,
            core_max_train_validation_gap=core_max_train_validation_gap,
            shortcut_min_train_auc=shortcut_min_train_auc,
            shortcut_max_validation_strength=shortcut_max_validation_strength,
            shortcut_min_gap=shortcut_min_gap,
            semantic_shortcut_fallback=semantic_shortcut_fallback,
        )
        quality.extend(anchor_quality)
        anchor_status: dict[str, Any] = {
            "atoms": len(anchor_quality),
            "core_atoms": sum(row["role"] == "core" for row in anchor_quality),
            "shortcut_atoms": sum(row["role"] == "shortcut" for row in anchor_quality),
            "supported": False,
        }
        try:
            anchor_scores, anchor_raw, anchor_core, anchor_shortcut = diagnostic_scores(
                evidence[:, indices], samples, anchor_quality
            )
        except ValueError as error:
            anchor_status["reason"] = str(error)
            status[anchor] = anchor_status
            continue
        if single_binary_pair:
            assign = np.ones(len(samples), dtype=bool)
            scores = anchor_scores
        else:
            assign = np.asarray([sample.label == anchor for sample in samples])
            anchor_train = assign & np.asarray([sample.split == "train" for sample in samples])
            scores[anchor_train] = percentile_scores(anchor_raw[anchor_train])
        raw[assign] = anchor_raw[assign]
        core_support[assign] = anchor_core[assign]
        shortcut_conflict[assign] = anchor_shortcut[assign]
        anchor_status["supported"] = True
        status[anchor] = anchor_status
    return quality, scores, raw, core_support, shortcut_conflict, status


def top_indices(
    similarities: np.ndarray,
    learned_scores: np.ndarray,
    ids: np.ndarray,
    *,
    k: int,
    lambda_value: float,
) -> np.ndarray:
    combined = similarities + lambda_value * learned_scores
    order = np.lexsort((ids, -similarities, -combined))
    return order[:k]


def select_context_rows(
    queries: Sequence[Sample],
    train: Sequence[Sample],
    query_embeddings: np.ndarray,
    train_embeddings: np.ndarray,
    train_scores: np.ndarray,
    lambdas: Sequence[float],
    *,
    k: int,
    preserve_baseline_label_counts: bool = False,
) -> list[dict[str, Any]]:
    train_ids = np.asarray([sample.id for sample in train], dtype=str)
    rows: list[dict[str, Any]] = []
    for query, query_vector in zip(queries, query_embeddings, strict=True):
        similarities = train_embeddings @ query_vector
        baseline_indices = top_indices(
            similarities, train_scores, train_ids, k=k, lambda_value=0.0
        )
        baseline_label_counts: dict[str, int] = {}
        for index in baseline_indices:
            label = train[int(index)].label
            baseline_label_counts[label] = baseline_label_counts.get(label, 0) + 1
        # A utility gate may collapse the candidate to the RICES baseline.
        # De-duplicate here so λ=0 is not emitted twice for every query.
        for lambda_value in dict.fromkeys(float(value) for value in lambdas):
            if preserve_baseline_label_counts and lambda_value != 0.0:
                selected: list[int] = []
                for label, count in baseline_label_counts.items():
                    label_indices = np.asarray(
                        [index for index, sample in enumerate(train) if sample.label == label],
                        dtype=int,
                    )
                    local = top_indices(
                        similarities[label_indices],
                        train_scores[label_indices],
                        train_ids[label_indices],
                        k=count,
                        lambda_value=lambda_value,
                    )
                    selected.extend(int(label_indices[index]) for index in local)
                indices = np.asarray(selected, dtype=int)
            else:
                indices = top_indices(
                    similarities,
                    train_scores,
                    train_ids,
                    k=k,
                    lambda_value=lambda_value,
                )
            prompt_order = sorted(
                (int(index) for index in indices),
                key=lambda index: (float(similarities[index]), train[index].id),
            )
            rows.append(
                {
                    "query_id": query.id,
                    "gold": query.label,
                    "lambda": float(lambda_value),
                    "demos": [
                        {
                            "id": train[index].id,
                            "label": train[index].label,
                            "similarity": float(similarities[index]),
                            "learned_diagnostic_score": float(train_scores[index]),
                        }
                        for index in prompt_order
                    ],
                    "mean_similarity": float(np.mean(similarities[indices])),
                    "mean_learned_diagnostic_score": float(np.mean(train_scores[indices])),
                    "selection_inputs": ["image similarity", "VLM-discovered diagnostic score"],
                    "prompt_order": PROMPT_ORDER_CONVENTION,
                    "preserve_rices_demo_label_counts": preserve_baseline_label_counts,
                    "rices_demo_label_counts": baseline_label_counts,
                    "query_group_or_spurious_metadata_used": False,
                }
            )
    return rows


def select_unsupervised_lambda(
    rows: Sequence[Mapping[str, Any]], *, max_similarity_drop: float
) -> dict[str, Any]:
    lambdas = sorted({float(row["lambda"]) for row in rows})
    by_lambda: dict[float, dict[str, float]] = {}
    for value in lambdas:
        selected = [row for row in rows if float(row["lambda"]) == value]
        by_lambda[value] = {
            "mean_similarity": float(np.mean([row["mean_similarity"] for row in selected])),
            "mean_learned_diagnostic_score": float(
                np.mean([row["mean_learned_diagnostic_score"] for row in selected])
            ),
        }
    baseline = by_lambda[0.0]["mean_similarity"]
    eligible = [
        value
        for value in lambdas
        if baseline - by_lambda[value]["mean_similarity"] <= max_similarity_drop + 1e-12
    ]
    selected = max(
        eligible,
        key=lambda value: (
            by_lambda[value]["mean_learned_diagnostic_score"], -value
        ),
    )
    return {
        "selection_rule": "maximize learned diagnosticity subject to a mean cosine-similarity-loss budget; no query labels or group metadata",
        "max_similarity_drop": float(max_similarity_drop),
        "selected_lambda": float(selected),
        "grid": {str(value): by_lambda[value] for value in lambdas},
    }


def oracle_audit(
    samples: Sequence[Sample],
    scores: np.ndarray,
    selection_rows: Sequence[Mapping[str, Any]],
    *,
    threshold_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    train = [sample for sample in samples if sample.split == "train"]
    train_values = scores[np.asarray([sample.split == "train" for sample in samples])]
    if not train or not all("diagnostic" in sample.metadata for sample in train):
        return {"available": False}
    targets = np.asarray([bool(sample.metadata["diagnostic"]) for sample in train])
    diagnostic_by_id = {
        sample.id: bool(sample.metadata["diagnostic"]) for sample in train
    }
    retrieval: dict[str, Any] = {}
    for value in sorted({float(row["lambda"]) for row in selection_rows}):
        rows = [row for row in selection_rows if float(row["lambda"]) == value]
        flags = [
            diagnostic_by_id[str(demo["id"])] for row in rows for demo in row["demos"]
        ]
        retrieval[str(value)] = {
            "oracle_diagnostic_rate": float(np.mean(flags)),
            "contexts_with_oracle_diagnostic": int(
                sum(
                    any(diagnostic_by_id[str(demo["id"])] for demo in row["demos"])
                    for row in rows
                )
            ),
            "contexts": len(rows),
        }
    positives = int(targets.sum())
    top = np.argsort(-train_values, kind="mergesort")[:positives]
    provenance = dict(threshold_provenance or {})
    return {
        "available": True,
        "joined_after_scores_and_contexts_were_frozen_in_this_run": True,
        "used_directly_by_runtime_scoring_or_lambda_selection": False,
        "independent_cross_dataset_measure": bool(
            provenance.get("frozen_before_dataset_audit", False)
        ),
        "threshold_provenance": provenance,
        "train_oracle_diagnostic_count": positives,
        "score_auc_for_oracle_diagnostic": float(binary_auc(train_values, targets)),
        "precision_at_oracle_count": float(targets[top].mean()) if positives else None,
        "retrieval": retrieval,
    }


def score_stage(
    config: Mapping[str, Any],
    samples: Sequence[Sample],
    atoms: Sequence[DecisionAtom],
    output_dir: Path,
    *,
    pilot_queries_per_class: int,
    k: int,
    seed: int,
    max_similarity_drop: float,
    core_min_auc: float,
    core_max_train_validation_gap: float,
    shortcut_min_train_auc: float,
    shortcut_max_validation_strength: float,
    shortcut_min_gap: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audit_active = [sample for sample in samples if sample.split in {"train", "validation"}]
    active = metadata_free_projection(audit_active)
    labels = [str(value) for value in config["labels"]]
    query_samples: list[Sample] = []
    for label_index, label in enumerate(labels):
        query_samples.extend(
            deterministic_sample(
                active,
                split="validation",
                label=label,
                count=pilot_queries_per_class,
                seed=seed + 40009 + label_index * 7919,
            )
        )
    pilot_ids = {sample.id for sample in query_samples}
    calibration_samples = [
        Sample(
            id=sample.id,
            split="pilot" if sample.id in pilot_ids else sample.split,
            image=sample.image,
            label=sample.label,
            text=sample.text,
            options=sample.options,
        )
        for sample in active
    ]
    embeddings, retriever = ensure_embeddings(
        config, active, output_dir / "siglip2_train_validation_embeddings.npz"
    )
    if retriever is None:
        retriever = build_retriever(dict(config["similarity_model"]))
    try:
        texts = []
        for atom in atoms:
            texts.extend(
                [
                    f"A photograph showing {atom.a_side}",
                    f"A photograph showing {atom.b_side}",
                ]
            )
        embed_text = getattr(retriever, "embed_text", None)
        if not callable(embed_text):
            raise TypeError("The configured retriever cannot embed VLM-discovered text atoms")
        text_embeddings = embed_text(texts)
    finally:
        _release(retriever)
    evidence = atom_evidence_from_embeddings(embeddings, atoms, text_embeddings)
    quality, scores, raw, core_support, shortcut_conflict, anchor_status = grouped_diagnostic_scores(
        evidence,
        calibration_samples,
        atoms,
        core_min_auc=core_min_auc,
        core_max_train_validation_gap=core_max_train_validation_gap,
        shortcut_min_train_auc=shortcut_min_train_auc,
        shortcut_max_validation_strength=shortcut_max_validation_strength,
        shortcut_min_gap=shortcut_min_gap,
        semantic_shortcut_fallback=bool(config.get("semantic_shortcut_fallback", False)),
    )
    write_jsonl(output_dir / "atom_quality.jsonl", quality)
    train_mask = np.asarray([sample.split == "train" for sample in active])
    train = [sample for sample in active if sample.split == "train"]
    train_scores = scores[train_mask]
    write_jsonl(
        output_dir / "train_diagnostic_scores.jsonl",
        [
            {
                "sample_id": sample.id,
                "label": sample.label,
                "learned_diagnostic_score": float(scores[index]),
                "raw_teaching_value": float(raw[index]),
                "core_support": float(core_support[index]),
                "shortcut_conflict": float(shortcut_conflict[index]),
                "group_or_spurious_metadata_used": False,
            }
            for index, sample in enumerate(active)
            if sample.split == "train"
        ],
    )
    position = {sample.id: index for index, sample in enumerate(active)}
    query_embeddings = np.asarray([embeddings[position[sample.id]] for sample in query_samples])
    train_embeddings = embeddings[train_mask]
    selection_rows = select_context_rows(
        query_samples,
        train,
        query_embeddings,
        train_embeddings,
        train_scores,
        LAMBDA_GRID,
        k=k,
        preserve_baseline_label_counts=bool(
            config.get("preserve_rices_demo_label_counts", False)
        ),
    )
    write_jsonl(output_dir / "selections.jsonl", selection_rows)
    lambda_summary = select_unsupervised_lambda(
        selection_rows, max_similarity_drop=max_similarity_drop
    )
    summary = {
        "method": (
            "VLM-discovered visual atoms + train/validation stability audit + "
            "optional VLM-semantic context fallback + counter-shortcut demo scoring"
        ),
        "labels": labels,
        "atoms": len(atoms),
        "core_atoms": sum(row["role"] == "core" for row in quality),
        "shortcut_atoms": sum(row["role"] == "shortcut" for row in quality),
        "rejected_atoms": sum(row["role"] == "rejected" for row in quality),
        "atom_roles_by_anchor_label": anchor_status,
        "lambda_selection": lambda_summary,
        "oracle_audit": oracle_audit(
            audit_active,
            scores,
            selection_rows,
            threshold_provenance=config.get("self_discovered_threshold_provenance"),
        ),
        "frozen_parameters": {
            "core_min_auc": core_min_auc,
            "core_max_train_validation_gap": core_max_train_validation_gap,
            "shortcut_min_train_auc": shortcut_min_train_auc,
            "shortcut_max_validation_strength": shortcut_max_validation_strength,
            "shortcut_min_gap": shortcut_min_gap,
            "max_similarity_drop": max_similarity_drop,
            "lambda_grid": list(LAMBDA_GRID),
            "shots": k,
            "semantic_shortcut_fallback": bool(
                config.get("semantic_shortcut_fallback", False)
            ),
            "preserve_rices_demo_label_counts": bool(
                config.get("preserve_rices_demo_label_counts", False)
            ),
        },
        "leakage_audit": {
            "discovery_inputs": ["training images", "training class labels"],
            "atom_role_inputs": [
                "train/validation split",
                "class labels",
                "image/text embeddings",
                *(
                    ["VLM semantic role emitted during atom discovery"]
                    if config.get("semantic_shortcut_fallback", False)
                    else []
                ),
            ],
            "pilot_query_ids_removed_from_atom_calibration": len(pilot_ids),
            "pilot_query_labels_used_for_atom_roles_or_lambda_selection": False,
            "retrieval_inputs": ["query image embedding", "training image embedding", "learned demo score"],
            "group_background_or_diagnostic_metadata_used_before_posthoc_audit": False,
            "test_split_used": False,
        },
    }
    write_json(output_dir / "retrieval_summary.json", summary)
    return selection_rows, summary


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    result: dict[str, Any] = {
        "count": len(rows),
        "accuracy": float(np.mean([row["prediction"] == row["gold"] for row in rows])),
    }
    groups = sorted({str(row.get("group", "")) for row in rows} - {""})
    if groups:
        by_group = {
            group: float(
                np.mean(
                    [row["prediction"] == row["gold"] for row in rows if row.get("group") == group]
                )
            )
            for group in groups
        }
        result["group_accuracy"] = by_group
        result["worst_group_accuracy"] = min(by_group.values())
    return result


def predict_selection_rows(
    config: Mapping[str, Any],
    inference_samples: Sequence[Sample],
    selection_rows: Sequence[Mapping[str, Any]],
    *,
    selected_lambda: float,
    cache_path: Path | None = None,
    checkpoint_batch: int = 128,
) -> list[dict[str, Any]]:
    chosen = [
        (condition, row)
        for condition, lambda_value in (("rices", 0.0), ("self_discovered", selected_lambda))
        for row in selection_rows
        if float(row["lambda"]) == lambda_value
    ]
    sample_by_id = {sample.id: sample for sample in inference_samples}
    def context_key(row: Mapping[str, Any]) -> str:
        return f"{row['query_id']}|" + ",".join(str(demo["id"]) for demo in row["demos"])

    existing = load_jsonl(cache_path) if cache_path is not None and cache_path.is_file() else []
    by_key = {str(row["context_key"]): row for row in existing}
    jobs: dict[str, Mapping[str, Any]] = {}
    for _, row in chosen:
        key = context_key(row)
        if key not in by_key:
            jobs.setdefault(key, row)
    pending = list(jobs.items())
    if pending:
        target = build_target(dict(config["target_model"]))
        try:
            for begin in range(0, len(pending), checkpoint_batch):
                batch = pending[begin : begin + checkpoint_batch]
                queries = [sample_by_id[str(row["query_id"])] for _, row in batch]
                demos = [
                    [sample_by_id[str(demo["id"])] for demo in row["demos"]]
                    for _, row in batch
                ]
                score_rows = target.icl_scores_batch(
                    queries, demos, list(config["labels"])
                )
                checkpoints: list[dict[str, Any]] = []
                for (key, row), class_scores in zip(batch, score_rows, strict=True):
                    prediction = predicted_label(class_scores, list(config["labels"]))
                    checkpoint = {
                        "context_key": key,
                        "query_id": str(row["query_id"]),
                        "prediction": prediction,
                        "retrieved_ids": [demo["id"] for demo in row["demos"]],
                        "scores": {
                            name: float(value) for name, value in class_scores.items()
                        },
                    }
                    by_key[key] = checkpoint
                    checkpoints.append(checkpoint)
                if cache_path is not None:
                    append_jsonl(cache_path, checkpoints)
                print(
                    f"unique ICL contexts {len(by_key)}/{len(existing) + len(pending)}",
                    flush=True,
                )
        finally:
            _release(target)
    output: list[dict[str, Any]] = []
    for condition, row in chosen:
        cached = by_key[context_key(row)]
        output.append(
            {
                "query_id": str(row["query_id"]),
                "lambda": float(row["lambda"]),
                "condition": condition,
                "prediction": str(cached["prediction"]),
                "retrieved_ids": list(cached["retrieved_ids"]),
                "scores": dict(cached["scores"]),
            }
        )
    return output


def summarize_predictions(
    frozen_rows: Sequence[Mapping[str, Any]],
    reporting_samples: Sequence[Sample],
    *,
    selected_lambda: float,
    scope: str,
    test_split_used: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sample_by_id = {sample.id: sample for sample in reporting_samples}
    output: list[dict[str, Any]] = []
    for source in frozen_rows:
        query = sample_by_id[str(source["query_id"])]
        row = dict(source) | {
            "gold": query.label,
            "correct": str(source["prediction"]) == query.label,
            "group": query.group,
        }
        if "conflict" in query.metadata:
            row["conflict"] = bool(query.metadata["conflict"])
        output.append(row)
    baseline = [row for row in output if row["condition"] == "rices"]
    learned = [row for row in output if row["condition"] == "self_discovered"]
    base_by_id = {str(row["query_id"]): row for row in baseline}
    learned_by_id = {str(row["query_id"]): row for row in learned}
    summary = {
        "scope": scope,
        "selected_lambda": selected_lambda,
        "rices": _metrics(baseline),
        "self_discovered": _metrics(learned),
        "paired": {
            "rescued": sum(
                not base_by_id[key]["correct"] and learned_by_id[key]["correct"]
                for key in base_by_id
            ),
            "harmed": sum(
                base_by_id[key]["correct"] and not learned_by_id[key]["correct"]
                for key in base_by_id
            ),
            "changed": sum(
                base_by_id[key]["prediction"] != learned_by_id[key]["prediction"]
                for key in base_by_id
            ),
        },
        "test_split_used": test_split_used,
        "query_metadata_used_only_after_predictions_for_reporting": True,
    }
    return output, summary


def utility_gate_lambda(
    pilot_summary: Mapping[str, Any],
    *,
    candidate_lambda: float,
    harm_penalty: float = 1.0,
) -> dict[str, Any]:
    """Reject a diagnostic reranker that degrades held-out validation accuracy.

    This gate intentionally uses only ordinary validation class labels.  Group,
    conflict, and diagnostic annotations are neither required nor consulted.
    Ties keep the unsupervised candidate so that an accuracy-neutral reranker may
    still improve hidden subgroups, as observed on Spawrious.
    """
    baseline_accuracy = float(pilot_summary["rices"]["accuracy"])
    candidate_accuracy = float(pilot_summary["self_discovered"]["accuracy"])
    rescued = int(pilot_summary["paired"]["rescued"])
    harmed = int(pilot_summary["paired"]["harmed"])
    asymmetric_utility = rescued - harm_penalty * harmed
    accepted = asymmetric_utility >= 0.0
    return {
        "candidate_lambda": float(candidate_lambda),
        "selected_lambda": float(candidate_lambda if accepted else 0.0),
        "accepted": bool(accepted),
        "asymmetric_flip_utility": {
            "rescued": rescued,
            "harmed": harmed,
            "harm_penalty": float(harm_penalty),
            "value": float(asymmetric_utility),
        },
        "validation_accuracy": {
            "rices": baseline_accuracy,
            "candidate": candidate_accuracy,
            "difference": candidate_accuracy - baseline_accuracy,
        },
        "selection_rule": (
            "keep the unsupervised candidate only when held-out validation "
            "rescues minus harm_penalty times harms is non-negative; ties keep "
            "the candidate"
        ),
        "inputs": ["validation class labels", "frozen pilot predictions"],
        "group_conflict_or_diagnostic_metadata_used": False,
    }


def pilot_stage(
    config: Mapping[str, Any],
    samples: Sequence[Sample],
    selection_rows: Sequence[Mapping[str, Any]],
    retrieval_summary: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    selected_lambda = float(retrieval_summary["lambda_selection"]["selected_lambda"])
    sample_by_id = {sample.id: sample for sample in samples}
    query_ids = list(dict.fromkeys(str(row["query_id"]) for row in selection_rows))
    inference_queries = inference_query_projection([sample_by_id[value] for value in query_ids])
    train = metadata_free_projection([sample for sample in samples if sample.split == "train"])
    frozen = predict_selection_rows(
        config,
        [*train, *inference_queries],
        selection_rows,
        selected_lambda=selected_lambda,
        cache_path=output_dir / "pilot_context_predictions_cache.jsonl",
    )
    output, summary = summarize_predictions(
        frozen,
        samples,
        selected_lambda=selected_lambda,
        scope="small exploratory validation pilot; not a frozen test result",
        test_split_used=False,
    )
    summary["utility_gate"] = utility_gate_lambda(
        summary,
        candidate_lambda=selected_lambda,
        harm_penalty=float(config.get("utility_harm_penalty", 1.0)),
    )
    write_jsonl(output_dir / "pilot_predictions.jsonl", output)
    write_json(output_dir / "pilot_summary.json", summary)
    return summary


def freeze_stage(config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    required = (
        "atoms.jsonl",
        "atom_quality.jsonl",
        "train_diagnostic_scores.jsonl",
        "siglip2_train_validation_embeddings.npz",
        "selections.jsonl",
        "retrieval_summary.json",
        "pilot_predictions.jsonl",
        "pilot_summary.json",
    )
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze; missing artifacts: {missing}")
    retrieval = json.loads((output_dir / "retrieval_summary.json").read_text(encoding="utf-8"))
    pilot = json.loads((output_dir / "pilot_summary.json").read_text(encoding="utf-8"))
    candidate_lambda = float(retrieval["lambda_selection"]["selected_lambda"])
    utility_gate = pilot.get("utility_gate") or utility_gate_lambda(
        pilot,
        candidate_lambda=candidate_lambda,
        harm_penalty=float(config.get("utility_harm_penalty", 1.0)),
    )
    protocol = {
        "status": "frozen_before_test",
        "candidate_lambda_before_utility_gate": candidate_lambda,
        "selected_lambda": float(utility_gate["selected_lambda"]),
        "utility_gate": utility_gate,
        "frozen_parameters": retrieval["frozen_parameters"],
        "atom_roles_by_anchor_label": retrieval["atom_roles_by_anchor_label"],
        "labels": list(config["labels"]),
        "target_model": config["target_model"]["model"],
        "similarity_model": config["similarity_model"]["model"],
        "threshold_provenance": config.get("self_discovered_threshold_provenance", {}),
        "manifest_sha256": file_sha256(Path(str(config["manifest"]))),
        "config_sha256": file_sha256(Path(str(config["_config_path"]))),
        "artifact_sha256": {
            name: file_sha256(output_dir / name) for name in required
        },
        "test_selection_rule": (
            "global top-k similarity + utility-gated frozen lambda * frozen "
            "learned demo score"
            + (
                "; preserve the per-query RICES demonstration-label histogram"
                if retrieval["frozen_parameters"].get(
                    "preserve_rices_demo_label_counts", False
                )
                else ""
            )
        ),
        "test_labels_groups_and_diagnostic_metadata_allowed_for_selection": False,
    }
    write_json(output_dir / "frozen_protocol.json", protocol)
    return protocol


def _verify_frozen_protocol(
    config: Mapping[str, Any], output_dir: Path
) -> dict[str, Any]:
    protocol_path = output_dir / "frozen_protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError("Run --stage freeze before frozen test evaluation")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_test":
        raise ValueError("Protocol is not frozen_before_test")
    if file_sha256(Path(str(config["manifest"]))) != protocol["manifest_sha256"]:
        raise ValueError("Manifest changed after protocol freeze")
    if file_sha256(Path(str(config["_config_path"]))) != protocol["config_sha256"]:
        raise ValueError("Config changed after protocol freeze")
    for name, expected in protocol["artifact_sha256"].items():
        if file_sha256(output_dir / name) != expected:
            raise ValueError(f"Artifact changed after protocol freeze: {name}")
    return protocol


def load_safe_train_and_test(manifest_path: Path) -> tuple[list[Sample], list[Sample]]:
    source = manifest_path.resolve()
    train: list[Sample] = []
    test: list[Sample] = []
    for row in load_jsonl(source):
        split = str(row["split"])
        if split not in {"train", "test"}:
            continue
        safe = {
            "id": str(row["id"]),
            "split": split,
            "image": str(row["image"]),
            "label": str(row["label"]) if split == "train" else "",
            "text": "" if row.get("text") is None else str(row.get("text", "")),
        }
        sample = Sample.from_dict(safe, source.parent)
        (train if split == "train" else test).append(sample)
    if not train or not test:
        raise ValueError("Frozen test requires non-empty train and test splits")
    return train, test


def frozen_test_stage(
    config: Mapping[str, Any], output_dir: Path
) -> dict[str, Any]:
    protocol = _verify_frozen_protocol(config, output_dir)
    selected_lambda = float(protocol["selected_lambda"])
    train, test_queries = load_safe_train_and_test(Path(str(config["manifest"])))
    with np.load(output_dir / "siglip2_train_validation_embeddings.npz", allow_pickle=False) as bundle:
        cached_ids = list(bundle["sample_ids"].astype(str))
        cached_values = normalize_rows(np.asarray(bundle["values"], dtype=np.float32))
    embedding_by_id = {
        sample_id: cached_values[index] for index, sample_id in enumerate(cached_ids)
    }
    missing = [sample.id for sample in train if sample.id not in embedding_by_id]
    if missing:
        raise ValueError(f"Frozen train embedding cache misses {len(missing)} samples")
    train_embeddings = np.stack([embedding_by_id[sample.id] for sample in train])
    score_rows = load_jsonl(output_dir / "train_diagnostic_scores.jsonl")
    score_by_id = {
        str(row["sample_id"]): float(row["learned_diagnostic_score"])
        for row in score_rows
    }
    train_scores = np.asarray([score_by_id[sample.id] for sample in train])
    test_embeddings, retriever = ensure_embeddings(
        config, test_queries, output_dir / "siglip2_test_embeddings.npz"
    )
    if retriever is not None:
        _release(retriever)
    selections = select_context_rows(
        test_queries,
        train,
        test_embeddings,
        train_embeddings,
        train_scores,
        (0.0, selected_lambda),
        k=int(protocol["frozen_parameters"]["shots"]),
        preserve_baseline_label_counts=bool(
            protocol["frozen_parameters"].get(
                "preserve_rices_demo_label_counts", False
            )
        ),
    )
    write_jsonl(output_dir / "test_selections_frozen.jsonl", selections)
    frozen = predict_selection_rows(
        config,
        [*train, *test_queries],
        selections,
        selected_lambda=selected_lambda,
        cache_path=output_dir / "test_context_predictions_cache.jsonl",
    )
    # Persist label-free predictions before evaluation labels or groups are loaded.
    write_jsonl(output_dir / "test_predictions_frozen_unscored.jsonl", frozen)
    reporting = load_manifest(config["manifest"])
    output, summary = summarize_predictions(
        frozen,
        reporting,
        selected_lambda=selected_lambda,
        scope="frozen untouched test evaluation",
        test_split_used=True,
    )
    summary["protocol_sha256"] = file_sha256(output_dir / "frozen_protocol.json")
    summary["label_free_predictions_written_before_test_reporting_metadata_loaded"] = True
    write_jsonl(output_dir / "test_predictions.jsonl", output)
    write_json(output_dir / "test_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Let a VLM discover and deploy diagnostic information for ICL retrieval"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("discover", "score", "pilot", "freeze", "test", "all"),
        default="all",
    )
    parser.add_argument("--bootstraps", type=int, default=2)
    parser.add_argument("--support-per-class", type=int, default=3)
    parser.add_argument("--atoms-per-bootstrap", type=int, default=8)
    parser.add_argument("--pilot-queries-per-class", type=int, default=8)
    parser.add_argument("--shots", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--max-similarity-drop", type=float, default=0.02)
    parser.add_argument("--core-min-auc", type=float, default=0.60)
    parser.add_argument("--core-max-train-validation-gap", type=float, default=0.12)
    parser.add_argument("--shortcut-min-train-auc", type=float, default=0.85)
    parser.add_argument("--shortcut-max-validation-strength", type=float, default=0.25)
    parser.add_argument("--shortcut-min-gap", type=float, default=0.15)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "freeze":
        protocol = freeze_stage(config, args.output)
        print(json.dumps(protocol, ensure_ascii=False, indent=2), flush=True)
        return
    if args.stage == "test":
        summary = frozen_test_stage(config, args.output)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return
    samples = load_manifest(config["manifest"])
    if args.stage in {"discover", "all"}:
        atoms = discover_atoms_stage(
            config,
            samples,
            args.output,
            bootstraps=args.bootstraps,
            support_per_class=args.support_per_class,
            atoms_per_bootstrap=args.atoms_per_bootstrap,
            seed=args.seed,
        )
    else:
        atoms = [
            DecisionAtom.from_dict(row) for row in load_jsonl(args.output / "atoms.jsonl")
        ]
    if args.stage == "discover":
        print(f"wrote {len(atoms)} atoms to {args.output / 'atoms.jsonl'}", flush=True)
        return
    if args.stage in {"score", "all"}:
        selections, summary = score_stage(
            config,
            samples,
            atoms,
            args.output,
            pilot_queries_per_class=args.pilot_queries_per_class,
            k=args.shots,
            seed=args.seed,
            max_similarity_drop=args.max_similarity_drop,
            core_min_auc=args.core_min_auc,
            core_max_train_validation_gap=args.core_max_train_validation_gap,
            shortcut_min_train_auc=args.shortcut_min_train_auc,
            shortcut_max_validation_strength=args.shortcut_max_validation_strength,
            shortcut_min_gap=args.shortcut_min_gap,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    else:
        selections = load_jsonl(args.output / "selections.jsonl")
        summary = json.loads((args.output / "retrieval_summary.json").read_text(encoding="utf-8"))
    if args.stage == "score":
        return
    if args.stage in {"pilot", "all"}:
        pilot = pilot_stage(config, samples, selections, summary, args.output)
        print(json.dumps(pilot, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

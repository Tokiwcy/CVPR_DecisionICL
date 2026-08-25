from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from behavior_icl import QwenVLBackend, make_single_token_codes


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_RULE_FIELDS = (
    "gold_label",
    "confused_with",
    "shared_nondiagnostic_features",
    "missed_gold_cues",
    "confuser_cues",
    "failure_hypothesis",
    "corrective_rule",
    "avoid_using",
    "confidence",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def stable_fold(sample_id: str, folds: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def balanced_folds(ids: Sequence[str], folds: int, seed: int) -> dict[str, int]:
    ordered = sorted(
        (str(sample_id) for sample_id in ids),
        key=lambda sample_id: hashlib.sha256(
            f"{seed}:{sample_id}".encode("utf-8")
        ).hexdigest(),
    )
    return {sample_id: position % folds for position, sample_id in enumerate(ordered)}


def identify_failure_slices(
    predictions: Sequence[Mapping[str, Any]], *, min_count: int, top_n: int
) -> list[dict[str, Any]]:
    counts = Counter(
        (str(row["gold"]), str(row["prediction"]))
        for row in predictions
        if str(row["gold"]) != str(row["prediction"])
    )
    selected = [
        {"gold_label": gold, "confused_with": predicted, "count": count}
        for (gold, predicted), count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
        if count >= min_count
    ][:top_n]
    return selected


def normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def nearest_ids(
    candidate_ids: Sequence[str],
    embedding_by_id: Mapping[str, np.ndarray],
    target: np.ndarray,
    count: int,
) -> list[str]:
    target = np.asarray(target, dtype=np.float32)
    target /= max(float(np.linalg.norm(target)), 1e-12)
    ranked = sorted(
        (str(sample_id) for sample_id in candidate_ids),
        key=lambda sample_id: (
            -float(np.dot(embedding_by_id[sample_id], target)), sample_id
        ),
    )
    return ranked[:count]


def _opaque_image(source: Path, opaque_dir: Path, token: str) -> Path:
    suffix = source.suffix.lower() or ".png"
    destination = opaque_dir / f"{token}{suffix}"
    if not destination.exists():
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    return destination.resolve()


def build_panels(
    manifest: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    embedding_ids: Sequence[str],
    embedding_values: np.ndarray,
    output: Path,
    *,
    folds: int,
    seed: int,
    min_slice_count: int,
    top_slices: int,
    anchors: int,
    controls_per_side: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Safe projections are intentional: group, conflict, spurious_attribute and
    # metadata fields are neither retained nor available to the selection code.
    safe_manifest = {
        str(row["id"]): {
            "id": str(row["id"]),
            "image": str(row["image"]),
            "label": str(row["label"]),
            "split": str(row["split"]),
        }
        for row in manifest
        if str(row["split"]) == "validation"
    }
    safe_predictions = [
        {
            "id": str(row["query_id"]),
            "gold": str(row["gold"]),
            "prediction": str(row["prediction"]),
            "correct": str(row["gold"]) == str(row["prediction"]),
        }
        for row in predictions
    ]
    prediction_ids = {row["id"] for row in safe_predictions}
    missing_manifest = sorted(prediction_ids - set(safe_manifest))
    if missing_manifest:
        raise ValueError(f"Manifest missing {len(missing_manifest)} prediction rows")

    ids = np.asarray(embedding_ids).astype(str)
    values = normalized(np.asarray(embedding_values, dtype=np.float32))
    embedding_by_id = {sample_id: values[index] for index, sample_id in enumerate(ids)}
    missing_embeddings = sorted(prediction_ids - set(embedding_by_id))
    if missing_embeddings:
        raise ValueError(f"Embedding cache missing {len(missing_embeddings)} prediction rows")

    row_by_id = {row["id"]: row for row in safe_predictions}
    slices = identify_failure_slices(
        safe_predictions, min_count=min_slice_count, top_n=top_slices
    )
    if not slices:
        raise ValueError("No failure slice meets min_slice_count")

    correct_by_label: dict[str, list[str]] = defaultdict(list)
    for row in safe_predictions:
        if row["correct"]:
            correct_by_label[row["gold"]].append(row["id"])
    correct_fold: dict[str, int] = {}
    for label, label_ids in correct_by_label.items():
        correct_fold.update(balanced_folds(label_ids, folds, seed + stable_fold(label, 997, seed)))

    opaque_dir = output / "opaque_images"
    opaque_dir.mkdir(parents=True, exist_ok=True)
    panels: list[dict[str, Any]] = []
    private_key: list[dict[str, Any]] = []
    slice_audit: list[dict[str, Any]] = []
    for slice_index, slice_row in enumerate(slices):
        gold = str(slice_row["gold_label"])
        confused = str(slice_row["confused_with"])
        slice_ids = [
            row["id"]
            for row in safe_predictions
            if not row["correct"] and row["gold"] == gold and row["prediction"] == confused
        ]
        error_fold = balanced_folds(slice_ids, folds, seed + 1009 * (slice_index + 1))
        fold_counts = Counter(error_fold.values())
        slice_audit.append({**slice_row, "heldout_counts": dict(sorted(fold_counts.items()))})
        for fold in range(folds):
            heldout_ids = [sample_id for sample_id in slice_ids if error_fold[sample_id] == fold]
            induction_errors = [sample_id for sample_id in slice_ids if error_fold[sample_id] != fold]
            if len(induction_errors) < anchors:
                raise ValueError(f"Too few induction errors for {gold}->{confused} fold {fold}")
            centroid = normalized(
                np.stack([embedding_by_id[sample_id] for sample_id in induction_errors]).mean(
                    axis=0, keepdims=True
                )
            )[0]
            selected_errors = nearest_ids(
                induction_errors, embedding_by_id, centroid, anchors
            )
            gold_candidates = [
                sample_id
                for sample_id in correct_by_label[gold]
                if correct_fold[sample_id] != fold
            ]
            confused_candidates = [
                sample_id
                for sample_id in correct_by_label[confused]
                if correct_fold[sample_id] != fold
            ]
            selected_gold = nearest_ids(
                gold_candidates, embedding_by_id, centroid, controls_per_side
            )
            selected_confused = nearest_ids(
                confused_candidates, embedding_by_id, centroid, controls_per_side
            )
            roles = (
                [("error", sample_id) for sample_id in selected_errors]
                + [("gold_correct", sample_id) for sample_id in selected_gold]
                + [("confuser_correct", sample_id) for sample_id in selected_confused]
            )
            panel_id = f"slice-{slice_index:02d}-fold-{fold}"
            images = []
            for position, (role, sample_id) in enumerate(roles):
                token = hashlib.sha256(
                    f"{seed}:{panel_id}:{position}:{sample_id}".encode("utf-8")
                ).hexdigest()[:24]
                opaque_path = _opaque_image(
                    Path(safe_manifest[sample_id]["image"]), opaque_dir, token
                )
                images.append({"position": position + 1, "role": role, "image": str(opaque_path)})
                private_key.append(
                    {
                        "panel_id": panel_id,
                        "position": position + 1,
                        "role": role,
                        "source_id": sample_id,
                    }
                )
            panels.append(
                {
                    "panel_id": panel_id,
                    "fold": fold,
                    "gold_label": gold,
                    "confused_with": confused,
                    "slice_count": len(slice_ids),
                    "heldout_error_count": len(heldout_ids),
                    "images": images,
                }
            )
    audit = {
        "seed": seed,
        "folds": folds,
        "prediction_count": len(safe_predictions),
        "error_count": sum(not row["correct"] for row in safe_predictions),
        "selected_slices": slice_audit,
        "panels": len(panels),
        "selection_inputs": ["gold_label", "model_prediction", "image_embedding"],
        "forbidden_inputs_used": [],
        "background_metadata_used": False,
        "test_split_used": False,
        "panel_prompt_contains_source_ids_or_paths": False,
    }
    write_jsonl(output / "panels.jsonl", panels)
    write_jsonl(output / "private_panel_key.jsonl", private_key)
    write_json(output / "prepare_audit.json", audit)
    return panels, audit


def prepare_stage(args: argparse.Namespace) -> None:
    payload = np.load(args.embedding_cache)
    id_key = "sample_ids" if "sample_ids" in payload.files else "ids"
    panels, audit = build_panels(
        load_jsonl(args.manifest),
        load_jsonl(args.predictions),
        payload[id_key],
        payload["values"],
        args.output,
        folds=args.folds,
        seed=args.seed,
        min_slice_count=args.min_slice_count,
        top_slices=args.top_slices,
        anchors=args.anchors,
        controls_per_side=args.controls_per_side,
    )
    print(json.dumps({"panels": len(panels), **audit}, ensure_ascii=False, indent=2))


def induction_prompt(panel: Mapping[str, Any], subject_guidance: str = "") -> str:
    gold = str(panel["gold_label"])
    confused = str(panel["confused_with"])
    roles = [str(row["role"]) for row in panel["images"]]
    role_lines = []
    role_counts: Counter[str] = Counter()
    names = {"error": "ERROR", "gold_correct": "GOLD_CORRECT", "confuser_correct": "CONFUSER_CORRECT"}
    for position, role in enumerate(roles, start=1):
        role_counts[role] += 1
        role_lines.append(f"Image {position}: {names[role]}_{role_counts[role]}")
    return f"""You are debugging a visual classifier using matched examples. The images are supplied in exactly this order:
{chr(10).join(role_lines)}

For every ERROR image, the gold label is {gold}, but the previous classifier predicted {confused}.
Every GOLD_CORRECT image was correctly classified as {gold}.
Every CONFUSER_CORRECT image was correctly classified as {confused}.

Compare the errors with both sets of visually matched correct controls. Infer a reusable, image-visible rule that distinguishes {gold} from {confused}. {subject_guidance or "Prefer cues on the classified subject's anatomy, parts, surface appearance, proportions, and shape."} Treat background, setting, image style, and pose as non-diagnostic unless the same distinction is consistently supported across both control sets. Do not mention dataset groups, filenames, image IDs, or hidden metadata. Do not merely restate the labels. Keep each list to at most four concise items and make corrective_rule one concise imperative sentence.

You must make a best-effort visual diagnosis. missed_gold_cues, confuser_cues, failure_hypothesis, and corrective_rule must all be non-empty. If the images give weak evidence, still state the most plausible visible distinction and lower confidence; never return an empty template.

Return exactly one valid JSON object with this schema and no Markdown:
{{"gold_label":"{gold}","confused_with":"{confused}","shared_nondiagnostic_features":[string],"missed_gold_cues":[string],"confuser_cues":[string],"failure_hypothesis":string,"corrective_rule":string,"avoid_using":[string],"confidence":number_between_0_and_1}}"""


def validate_rule(value: Mapping[str, Any], panel: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in REQUIRED_RULE_FIELDS if field not in value]
    if missing:
        raise ValueError(f"Missing rule fields: {missing}")
    output = dict(value)
    if str(output["gold_label"]).strip().lower() != str(panel["gold_label"]).lower():
        raise ValueError("Rule gold_label does not match panel")
    if str(output["confused_with"]).strip().lower() != str(panel["confused_with"]).lower():
        raise ValueError("Rule confused_with does not match panel")
    for field in ("shared_nondiagnostic_features", "missed_gold_cues", "confuser_cues", "avoid_using"):
        if not isinstance(output[field], list):
            raise ValueError(f"Rule field {field} must be a list")
        output[field] = [str(item).strip() for item in output[field] if str(item).strip()][:4]
    output["failure_hypothesis"] = str(output["failure_hypothesis"]).strip()
    output["corrective_rule"] = str(output["corrective_rule"]).strip()
    if not output["missed_gold_cues"]:
        raise ValueError("Rule missed_gold_cues must be non-empty")
    if not output["confuser_cues"]:
        raise ValueError("Rule confuser_cues must be non-empty")
    if not output["failure_hypothesis"]:
        raise ValueError("Rule failure_hypothesis must be non-empty")
    if not output["corrective_rule"]:
        raise ValueError("Rule corrective_rule must be non-empty")
    output["confidence"] = min(1.0, max(0.0, float(output["confidence"])))
    return output


def induce_stage(args: argparse.Namespace) -> None:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_config = dict(config["target_model"])
    subject_guidance = str(
        config.get(
            "diagnostic_subject_guidance",
            "Prefer cues on the dog's body, head, coat, proportions, and shape.",
        )
    ).strip()
    panels = load_jsonl(args.panels)
    if args.limit is not None:
        panels = panels[: args.limit]
    existing = load_jsonl(args.output) if args.output.is_file() else []
    completed = {str(row["panel_id"]) for row in existing}
    backend = QwenVLBackend(
        model_config["model"],
        device="cuda",
        min_pixels=int(model_config.get("min_pixels", 32 * 28 * 28)),
        max_pixels=int(model_config.get("max_pixels", 128 * 28 * 28)),
        load_in_4bit=False,
        max_cached_images=64,
    )
    try:
        for panel in panels:
            panel_id = str(panel["panel_id"])
            if panel_id in completed:
                continue
            paths = [row["image"] for row in panel["images"]]
            parsed, raw = backend.generate_json(
                paths,
                induction_prompt(panel, subject_guidance),
                max_new_tokens=args.max_new_tokens,
            )
            rule = validate_rule(parsed, panel)
            append_jsonl(
                args.output,
                [{
                    "panel_id": panel_id,
                    "fold": int(panel["fold"]),
                    "gold_label": panel["gold_label"],
                    "confused_with": panel["confused_with"],
                    "rule": rule,
                    "raw": raw,
                    "valid_json": True,
                }],
            )
            completed.add(panel_id)
            print(f"rules {len(completed)}/{len(panels)}", flush=True)
    finally:
        backend.close()
        del backend
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def rule_text(rule: Mapping[str, Any], mode: str = "learned") -> str:
    gold = str(rule["gold_label"])
    confused = str(rule["confused_with"])
    if mode == "label_only":
        return (
            f"Pairwise correction for {gold} versus {confused}. Carefully inspect the "
            "visible subject's body, head, coat, proportions, and shape. Do not decide "
            "from background, setting, image style, or pose. Apply this only when the "
            "visible subject plausibly belongs to this pair; otherwise ignore the "
            "correction and use the full class catalog."
        )
    if mode != "learned":
        raise ValueError(f"Unknown rule mode {mode!r}")
    gold_cues = "; ".join(str(item) for item in rule["missed_gold_cues"])
    confused_cues = "; ".join(str(item) for item in rule["confuser_cues"])
    avoid = "; ".join(str(item) for item in rule["avoid_using"])
    return (
        f"Pairwise correction for {gold} versus {confused}. "
        f"Visible evidence favoring {gold}: {gold_cues}. "
        f"Visible evidence favoring {confused}: {confused_cues}. "
        f"Decision rule: {rule['corrective_rule']} "
        f"Do not decide from: {avoid}. "
        "Apply this only when the visible subject plausibly belongs to this pair; "
        "otherwise ignore the correction and use the full class catalog."
    )


def build_targeted_oof_rows(
    manifest: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    embedding_ids: Sequence[str],
    embedding_values: np.ndarray,
    rules: Sequence[Mapping[str, Any]],
    *,
    folds: int,
    seed: int,
    rule_mode: str = "learned",
) -> list[dict[str, Any]]:
    safe_manifest = {
        str(row["id"]): {
            "id": str(row["id"]),
            "image": str(row["image"]),
            "label": str(row["label"]),
            "split": str(row["split"]),
        }
        for row in manifest
    }
    safe_predictions = [
        {
            "id": str(row["query_id"]),
            "gold": str(row["gold"]),
            "prediction": str(row["prediction"]),
            "retrieved_ids": [str(value) for value in row["retrieved_ids"]],
            "correct": str(row["gold"]) == str(row["prediction"]),
        }
        for row in predictions
    ]
    prediction_by_id = {row["id"]: row for row in safe_predictions}
    ids = np.asarray(embedding_ids).astype(str)
    values = normalized(np.asarray(embedding_values, dtype=np.float32))
    embedding_by_id = {sample_id: values[index] for index, sample_id in enumerate(ids)}

    correct_by_label: dict[str, list[str]] = defaultdict(list)
    for row in safe_predictions:
        if row["correct"]:
            correct_by_label[row["gold"]].append(row["id"])
    correct_fold: dict[str, int] = {}
    for label, label_ids in correct_by_label.items():
        correct_fold.update(balanced_folds(label_ids, folds, seed + stable_fold(label, 997, seed)))

    rule_by_key = {
        (str(row["gold_label"]), str(row["confused_with"]), int(row["fold"])): row["rule"]
        for row in rules
    }
    slice_pairs = sorted({(gold, confused) for gold, confused, _ in rule_by_key})
    rows: list[dict[str, Any]] = []
    for slice_index, (gold, confused) in enumerate(slice_pairs):
        slice_ids = [
            row["id"]
            for row in safe_predictions
            if not row["correct"] and row["gold"] == gold and row["prediction"] == confused
        ]
        # Panel construction orders slices by descending frequency. Recover that
        # ordering before deriving the exact held-out assignment.
        ordered_pairs = [
            (row["gold_label"], row["confused_with"])
            for row in identify_failure_slices(
                safe_predictions, min_count=1, top_n=len(slice_pairs)
            )
            if (row["gold_label"], row["confused_with"]) in slice_pairs
        ]
        panel_slice_index = ordered_pairs.index((gold, confused))
        error_fold = balanced_folds(
            slice_ids, folds, seed + 1009 * (panel_slice_index + 1)
        )
        for fold in range(folds):
            learned_rule = rule_by_key[(gold, confused, fold)]
            heldout_errors = [sample_id for sample_id in slice_ids if error_fold[sample_id] == fold]
            induction_errors = [sample_id for sample_id in slice_ids if error_fold[sample_id] != fold]
            centroid = normalized(
                np.stack([embedding_by_id[sample_id] for sample_id in induction_errors]).mean(
                    axis=0, keepdims=True
                )
            )[0]
            count = len(heldout_errors)
            heldout_gold_controls = nearest_ids(
                [sample_id for sample_id in correct_by_label[gold] if correct_fold[sample_id] == fold],
                embedding_by_id,
                centroid,
                count,
            )
            heldout_confuser_controls = nearest_ids(
                [sample_id for sample_id in correct_by_label[confused] if correct_fold[sample_id] == fold],
                embedding_by_id,
                centroid,
                count,
            )
            for role, selected in (
                ("target_error", heldout_errors),
                ("gold_correct_control", heldout_gold_controls),
                ("confuser_correct_control", heldout_confuser_controls),
            ):
                for sample_id in selected:
                    prediction = prediction_by_id[sample_id]
                    demos = []
                    for demo_id in prediction["retrieved_ids"]:
                        demo = safe_manifest[demo_id]
                        demos.append(
                            {"path": demo["image"], "text": "", "label": demo["label"]}
                        )
                    query = safe_manifest[sample_id]
                    rows.append(
                        {
                            "panel_id": f"slice-{panel_slice_index:02d}-fold-{fold}",
                            "fold": fold,
                            "slice_gold": gold,
                            "slice_confused_with": confused,
                            "role": role,
                            "query_id": sample_id,
                            "gold": prediction["gold"],
                            "baseline_prediction": prediction["prediction"],
                            "condition": rule_mode,
                            "query_path": query["image"],
                            "demonstrations": demos,
                            "reasoning_rules": rule_text(learned_rule, rule_mode),
                        }
                    )
    return rows


def build_full_routed_oof_rows(
    manifest: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    rules: Sequence[Mapping[str, Any]],
    *,
    folds: int,
    seed: int,
    rule_mode: str = "learned",
) -> list[dict[str, Any]]:
    safe_manifest = {
        str(row["id"]): {
            "image": str(row["image"]),
            "label": str(row["label"]),
        }
        for row in manifest
    }
    safe_predictions = [
        {
            "id": str(row["query_id"]),
            "gold": str(row["gold"]),
            "prediction": str(row["prediction"]),
            "retrieved_ids": [str(value) for value in row["retrieved_ids"]],
            "correct": str(row["gold"]) == str(row["prediction"]),
        }
        for row in predictions
    ]
    rule_by_key = {
        (str(row["gold_label"]), str(row["confused_with"]), int(row["fold"])): row["rule"]
        for row in rules
    }
    pairs = sorted({(gold, confused) for gold, confused, _ in rule_by_key})
    pair_by_prediction = {confused: (gold, confused) for gold, confused in pairs}
    if len(pair_by_prediction) != len(pairs):
        raise ValueError("Full routing requires one learned alternative per baseline prediction")
    ordered_pairs = [
        (row["gold_label"], row["confused_with"])
        for row in identify_failure_slices(
            safe_predictions, min_count=1, top_n=len(pairs)
        )
        if (row["gold_label"], row["confused_with"]) in pairs
    ]
    error_fold_by_pair: dict[tuple[str, str], dict[str, int]] = {}
    for pair in pairs:
        gold, confused = pair
        slice_ids = [
            row["id"] for row in safe_predictions
            if not row["correct"] and row["gold"] == gold and row["prediction"] == confused
        ]
        panel_slice_index = ordered_pairs.index(pair)
        error_fold_by_pair[pair] = balanced_folds(
            slice_ids, folds, seed + 1009 * (panel_slice_index + 1)
        )
    correct_by_label: dict[str, list[str]] = defaultdict(list)
    for row in safe_predictions:
        if row["correct"]:
            correct_by_label[row["gold"]].append(row["id"])
    correct_fold: dict[str, int] = {}
    for label, label_ids in correct_by_label.items():
        correct_fold.update(balanced_folds(label_ids, folds, seed + stable_fold(label, 997, seed)))

    output: list[dict[str, Any]] = []
    for prediction in safe_predictions:
        pair = pair_by_prediction.get(prediction["prediction"])
        base = {
            "query_id": prediction["id"],
            "gold": prediction["gold"],
            "baseline_prediction": prediction["prediction"],
            "condition": rule_mode,
        }
        if pair is None:
            output.append(
                base
                | {
                    "panel_id": "unrouted",
                    "fold": -1,
                    "slice_gold": "",
                    "slice_confused_with": "",
                    "role": "unrouted",
                    "routed": False,
                }
            )
            continue
        gold, confused = pair
        is_target_error = (
            not prediction["correct"]
            and prediction["gold"] == gold
            and prediction["prediction"] == confused
        )
        if is_target_error:
            fold = error_fold_by_pair[pair][prediction["id"]]
            role = "target_error"
        elif prediction["correct"]:
            fold = correct_fold[prediction["id"]]
            role = "routed_correct"
        else:
            fold = stable_fold(prediction["id"], folds, seed + 1777)
            role = "other_error"
        learned_rule = rule_by_key[(gold, confused, fold)]
        demos = [
            {
                "path": safe_manifest[demo_id]["image"],
                "text": "",
                "label": safe_manifest[demo_id]["label"],
            }
            for demo_id in prediction["retrieved_ids"]
        ]
        output.append(
            base
            | {
                "panel_id": f"route-{gold}-vs-{confused}-fold-{fold}",
                "fold": fold,
                "slice_gold": gold,
                "slice_confused_with": confused,
                "role": role,
                "routed": True,
                "query_path": safe_manifest[prediction["id"]]["image"],
                "demonstrations": demos,
                "reasoning_rules": rule_text(learned_rule, rule_mode),
            }
        )
    return output


def enabled_pairs_from_validation_summary(
    summary: Mapping[str, Any],
) -> set[tuple[str, str]]:
    decisions = dict(summary["cross_fitted_pair_gate"]["decisions"])
    by_pair: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for key, enabled in decisions.items():
        pair_text, _, _ = str(key).partition("/fold-")
        gold, separator, confused = pair_text.partition("<->")
        if not separator:
            raise ValueError(f"Invalid pair-gate key {key!r}")
        by_pair[(gold, confused)].append(bool(enabled))
    return {
        pair for pair, values in by_pair.items()
        if len(values) > 0 and all(values)
    }


def build_frozen_test_rows(
    manifest: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    rules: Sequence[Mapping[str, Any]],
    enabled_pairs: set[tuple[str, str]],
    *,
    folds: int,
    seed: int,
    rule_mode: str = "learned",
) -> list[dict[str, Any]]:
    manifest_by_id = {
        str(row["id"]): {
            "image": str(row["image"]),
            "label": str(row["label"]),
        }
        for row in manifest
    }
    rule_by_key = {
        (str(row["gold_label"]), str(row["confused_with"]), int(row["fold"])): row["rule"]
        for row in rules
    }
    pairs = sorted({(gold, confused) for gold, confused, _ in rule_by_key})
    pair_by_prediction = {confused: (gold, confused) for gold, confused in pairs}
    if len(pair_by_prediction) != len(pairs):
        raise ValueError("Frozen routing requires one learned alternative per baseline prediction")
    output: list[dict[str, Any]] = []
    for source in predictions:
        sample_id = str(source["query_id"])
        baseline_prediction = str(source["prediction"])
        pair = pair_by_prediction.get(baseline_prediction)
        routed = pair in enabled_pairs if pair is not None else False
        base = {
            "query_id": sample_id,
            "gold": str(source["gold"]),
            "baseline_prediction": baseline_prediction,
            "condition": rule_mode,
            "evaluation_group": str(source.get("group", "")),
            "evaluation_conflict": bool(source.get("conflict", False)),
            "routed": routed,
        }
        if not routed or pair is None:
            output.append(
                base
                | {
                    "panel_id": "test-unrouted",
                    "fold": -1,
                    "slice_gold": pair[0] if pair is not None else "",
                    "slice_confused_with": pair[1] if pair is not None else "",
                    "role": "unrouted",
                }
            )
            continue
        gold, confused = pair
        fold = stable_fold(sample_id, folds, seed + 9001)
        learned_rule = rule_by_key[(gold, confused, fold)]
        demos = [
            {
                "path": manifest_by_id[str(demo_id)]["image"],
                "text": "",
                "label": manifest_by_id[str(demo_id)]["label"],
            }
            for demo_id in source["retrieved_ids"]
        ]
        output.append(
            base
            | {
                "panel_id": f"test-route-{gold}-vs-{confused}-fold-{fold}",
                "fold": fold,
                "slice_gold": gold,
                "slice_confused_with": confused,
                "role": "routed",
                "query_path": manifest_by_id[sample_id]["image"],
                "demonstrations": demos,
                "reasoning_rules": rule_text(learned_rule, rule_mode),
            }
        )
    return output


def summarize_oof(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def metrics(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        count = len(selected)
        baseline_correct = sum(row["baseline_prediction"] == row["gold"] for row in selected)
        guided_correct = sum(row["guided_prediction"] == row["gold"] for row in selected)
        return {
            "count": count,
            "baseline_accuracy": baseline_correct / count if count else None,
            "guided_accuracy": guided_correct / count if count else None,
            "rescued": sum(
                row["baseline_prediction"] != row["gold"]
                and row["guided_prediction"] == row["gold"]
                for row in selected
            ),
            "harmed": sum(
                row["baseline_prediction"] == row["gold"]
                and row["guided_prediction"] != row["gold"]
                for row in selected
            ),
            "changed": sum(
                row["baseline_prediction"] != row["guided_prediction"] for row in selected
            ),
        }

    by_slice: dict[str, Any] = {}
    pairs = sorted({(str(row["slice_gold"]), str(row["slice_confused_with"])) for row in rows})
    for gold, confused in pairs:
        selected = [
            row for row in rows
            if row["slice_gold"] == gold and row["slice_confused_with"] == confused
        ]
        by_slice[f"{gold}->{confused}"] = {
            "all": metrics(selected),
            "target_errors": metrics([row for row in selected if row["role"] == "target_error"]),
            "correct_controls": metrics([row for row in selected if row["role"] != "target_error"]),
        }
    return {
        "scope": "targeted_matched_out_of_fold",
        "all": metrics(rows),
        "target_errors": metrics([row for row in rows if row["role"] == "target_error"]),
        "correct_controls": metrics([row for row in rows if row["role"] != "target_error"]),
        "by_slice": by_slice,
        "background_metadata_used_for_selection_or_inference": False,
        "test_split_used": False,
        "note": "This estimates rule rescue/harm on matched OOF examples, not full-validation accuracy.",
    }


def summarize_full_routed(
    rows: Sequence[Mapping[str, Any]], *, punish_beta: float = 1.0
) -> dict[str, Any]:
    if punish_beta < 1.0:
        raise ValueError("punish_beta must be at least 1.0")
    def metrics(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        count = len(selected)
        baseline_correct = sum(row["baseline_prediction"] == row["gold"] for row in selected)
        guided_correct = sum(row["guided_prediction"] == row["gold"] for row in selected)
        return {
            "count": count,
            "baseline_accuracy": baseline_correct / count if count else None,
            "guided_accuracy": guided_correct / count if count else None,
            "rescued": sum(
                row["baseline_prediction"] != row["gold"] and row["guided_prediction"] == row["gold"]
                for row in selected
            ),
            "harmed": sum(
                row["baseline_prediction"] == row["gold"] and row["guided_prediction"] != row["gold"]
                for row in selected
            ),
            "changed": sum(row["baseline_prediction"] != row["guided_prediction"] for row in selected),
        }

    routed = [row for row in rows if row.get("routed")]
    routes: dict[str, Any] = {}
    for gold, confused in sorted(
        {(str(row["slice_gold"]), str(row["slice_confused_with"])) for row in routed}
    ):
        selected = [
            row for row in routed
            if row["slice_gold"] == gold and row["slice_confused_with"] == confused
        ]
        routes[f"{gold}<->{confused}"] = metrics(selected)
    route_folds = sorted(
        {
            (str(row["slice_gold"]), str(row["slice_confused_with"]), int(row["fold"]))
            for row in routed
        }
    )
    gate_decisions: dict[tuple[str, str, int], bool] = {}
    for gold, confused, heldout_fold in route_folds:
        development = [
            row for row in routed
            if row["slice_gold"] == gold
            and row["slice_confused_with"] == confused
            and int(row["fold"]) != heldout_fold
        ]
        development_rescue = sum(
            row["baseline_prediction"] != row["gold"]
            and row["guided_prediction"] == row["gold"]
            for row in development
        )
        development_harm = sum(
            row["baseline_prediction"] == row["gold"]
            and row["guided_prediction"] != row["gold"]
            for row in development
        )
        development_utility = development_rescue - punish_beta * development_harm
        gate_decisions[(gold, confused, heldout_fold)] = development_utility > 0
    gated_rows = []
    for row in rows:
        use_rule = bool(row.get("routed")) and gate_decisions[
            (str(row["slice_gold"]), str(row["slice_confused_with"]), int(row["fold"]))
        ]
        gated_rows.append(
            dict(row)
            | {
                "guided_prediction": (
                    row["guided_prediction"] if use_rule else row["baseline_prediction"]
                )
            }
        )
    gate_audit = {
        f"{gold}<->{confused}/fold-{fold}": enabled
        for (gold, confused, fold), enabled in sorted(gate_decisions.items())
    }
    return {
        "scope": "full_validation_prediction_routed_out_of_fold",
        "all": metrics(rows),
        "routed": metrics(routed),
        "target_errors": metrics([row for row in rows if row["role"] == "target_error"]),
        "other_errors": metrics([row for row in rows if row["role"] == "other_error"]),
        "routed_correct": metrics([row for row in rows if row["role"] == "routed_correct"]),
        "unrouted": metrics([row for row in rows if row["role"] == "unrouted"]),
        "by_route": routes,
        "cross_fitted_pair_gate": {
            **metrics(gated_rows),
            "decisions": gate_audit,
            "punish_beta": punish_beta,
            "decision_rule": "enable a pair on one fold only when rescue - beta * harm is positive on the other folds",
        },
        "routing_input": "baseline_prediction_only",
        "query_gold_used_for_routing": False,
        "background_metadata_used_for_selection_or_inference": False,
        "test_split_used": False,
    }


def summarize_frozen_test(
    rows: Sequence[Mapping[str, Any]], enabled_pairs: set[tuple[str, str]]
) -> dict[str, Any]:
    def metrics(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        count = len(selected)
        baseline_correct = sum(row["baseline_prediction"] == row["gold"] for row in selected)
        guided_correct = sum(row["guided_prediction"] == row["gold"] for row in selected)
        return {
            "count": count,
            "baseline_accuracy": baseline_correct / count if count else None,
            "guided_accuracy": guided_correct / count if count else None,
            "rescued": sum(
                row["baseline_prediction"] != row["gold"] and row["guided_prediction"] == row["gold"]
                for row in selected
            ),
            "harmed": sum(
                row["baseline_prediction"] == row["gold"] and row["guided_prediction"] != row["gold"]
                for row in selected
            ),
            "changed": sum(row["baseline_prediction"] != row["guided_prediction"] for row in selected),
        }

    group_accuracy: dict[str, float] = {}
    for group in sorted({str(row.get("evaluation_group", "")) for row in rows} - {""}):
        selected = [row for row in rows if row.get("evaluation_group") == group]
        group_accuracy[group] = sum(
            row["guided_prediction"] == row["gold"] for row in selected
        ) / len(selected)
    class_accuracy: dict[str, float] = {}
    for label in sorted({str(row["gold"]) for row in rows}):
        selected = [row for row in rows if row["gold"] == label]
        class_accuracy[label] = sum(
            row["guided_prediction"] == row["gold"] for row in selected
        ) / len(selected)
    routed = [row for row in rows if row.get("routed")]
    conflict = [row for row in rows if row.get("evaluation_conflict")]
    aligned = [row for row in rows if not row.get("evaluation_conflict")]
    return {
        "scope": "frozen_test_prediction_routed",
        "all": metrics(rows),
        "routed": metrics(routed),
        "unrouted": metrics([row for row in rows if not row.get("routed")]),
        "conflict": metrics(conflict),
        "aligned": metrics(aligned),
        "group_accuracy": group_accuracy,
        "worst_group_accuracy": min(group_accuracy.values()) if group_accuracy else None,
        "class_accuracy": class_accuracy,
        "worst_class_accuracy": min(class_accuracy.values()) if class_accuracy else None,
        "enabled_pairs": [f"{gold}<->{confused}" for gold, confused in sorted(enabled_pairs)],
        "routing_input": "baseline_prediction_only",
        "rule_fold_assignment": "stable hash of test query ID",
        "query_gold_used_for_routing": False,
        "test_metadata_used_for_routing_or_rule_selection": False,
        "test_labels_used_only_after_predictions_were_frozen": True,
    }


def evaluate_stage(args: argparse.Namespace) -> None:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_config = dict(config["target_model"])
    labels = [str(label) for label in config["labels"]]
    manifest = load_jsonl(args.manifest)
    predictions = load_jsonl(args.predictions)
    rules = load_jsonl(args.rules)
    if args.scope == "targeted":
        if args.embedding_cache is None:
            raise ValueError("--embedding-cache is required for targeted evaluation")
        payload = np.load(args.embedding_cache)
        id_key = "sample_ids" if "sample_ids" in payload.files else "ids"
        eval_rows = build_targeted_oof_rows(
            manifest,
            predictions,
            payload[id_key],
            payload["values"],
            rules,
            folds=args.folds,
            seed=args.seed,
            rule_mode=args.rule_mode,
        )
        enabled_pairs: set[tuple[str, str]] = set()
    elif args.scope == "full_routed":
        eval_rows = build_full_routed_oof_rows(
            manifest,
            predictions,
            rules,
            folds=args.folds,
            seed=args.seed,
            rule_mode=args.rule_mode,
        )
        enabled_pairs = set()
    else:
        if args.validation_summary is None:
            raise ValueError("--validation-summary is required for frozen_test")
        enabled_pairs = enabled_pairs_from_validation_summary(
            json.loads(args.validation_summary.read_text(encoding="utf-8"))
        )
        eval_rows = build_frozen_test_rows(
            manifest,
            predictions,
            rules,
            enabled_pairs,
            folds=args.folds,
            seed=args.seed,
            rule_mode=args.rule_mode,
        )
    existing = load_jsonl(args.output) if args.output.is_file() else []
    completed = {
        (str(row["panel_id"]), str(row["role"]), str(row["query_id"])) for row in existing
    }
    pending = [
        row for row in eval_rows
        if (row["panel_id"], row["role"], row["query_id"]) not in completed
    ]
    unchanged = [row for row in pending if not row.get("routed", True)]
    if unchanged:
        append_jsonl(
            args.output,
            [
                {
                    key: value for key, value in row.items()
                    if key not in {"query_path", "demonstrations", "reasoning_rules"}
                }
                | {"guided_prediction": row["baseline_prediction"], "guided_probabilities": None}
                for row in unchanged
            ],
        )
        completed.update(
            (str(row["panel_id"]), str(row["role"]), str(row["query_id"]))
            for row in unchanged
        )
    pending = [row for row in pending if row.get("routed", True)]
    batch_size = int(args.batch_size or model_config.get("icl_batch_size", 2))
    if pending:
        backend = QwenVLBackend(
            model_config["model"],
            device="cuda",
            min_pixels=int(model_config.get("min_pixels", 32 * 28 * 28)),
            max_pixels=int(model_config.get("max_pixels", 128 * 28 * 28)),
            load_in_4bit=False,
            max_cached_images=4096,
        )
        codes = make_single_token_codes(backend.processor.tokenizer, len(labels))
        try:
            for begin in range(0, len(pending), batch_size):
                batch = pending[begin : begin + batch_size]
                requests = [
                    (
                        row["demonstrations"],
                        {
                            "path": row["query_path"],
                            "text": "",
                            "reasoning_rules": row["reasoning_rules"],
                        },
                    )
                    for row in batch
                ]
                results = backend.score_named_classes_batch(
                    requests,
                    labels,
                    codes,
                    task_instruction=str(model_config.get("task_instruction", "Identify the dog breed shown in the image.")),
                )
                checkpoint = []
                for row, result in zip(batch, results, strict=True):
                    checkpoint.append(
                        {
                            key: value
                            for key, value in row.items()
                            if key not in {"query_path", "demonstrations", "reasoning_rules"}
                        }
                        | {
                            "guided_prediction": result["prediction"],
                            "guided_probabilities": result["probabilities"],
                        }
                    )
                append_jsonl(args.output, checkpoint)
                completed.update(
                    (str(row["panel_id"]), str(row["role"]), str(row["query_id"]))
                    for row in checkpoint
                )
                print(f"oof {len(completed)}/{len(eval_rows)}", flush=True)
        finally:
            backend.close()
            del backend
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
    frozen = load_jsonl(args.output)
    if args.scope == "targeted":
        summary = summarize_oof(frozen)
    elif args.scope == "full_routed":
        summary = summarize_full_routed(frozen, punish_beta=args.punish_beta)
    else:
        summary = summarize_frozen_test(frozen, enabled_pairs)
    summary["condition"] = args.rule_mode
    write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Discover model-specific diagnostic rules from matched validation errors"
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--predictions", type=Path, required=True)
    prepare.add_argument("--embedding-cache", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--folds", type=int, default=4)
    prepare.add_argument("--seed", type=int, default=20260825)
    prepare.add_argument("--min-slice-count", type=int, default=10)
    prepare.add_argument("--top-slices", type=int, default=3)
    prepare.add_argument("--anchors", type=int, default=2)
    prepare.add_argument("--controls-per-side", type=int, default=2)

    induce = subparsers.add_parser("induce")
    induce.add_argument("--config", type=Path, required=True)
    induce.add_argument("--panels", type=Path, required=True)
    induce.add_argument("--output", type=Path, required=True)
    induce.add_argument("--limit", type=int)
    induce.add_argument("--max-new-tokens", type=int, default=384)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--embedding-cache", type=Path)
    evaluate.add_argument("--rules", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--summary", type=Path, required=True)
    evaluate.add_argument("--folds", type=int, default=4)
    evaluate.add_argument("--seed", type=int, default=20260825)
    evaluate.add_argument("--batch-size", type=int)
    evaluate.add_argument("--rule-mode", choices=("learned", "label_only"), default="learned")
    evaluate.add_argument("--scope", choices=("targeted", "full_routed", "frozen_test"), default="targeted")
    evaluate.add_argument("--validation-summary", type=Path)
    evaluate.add_argument("--punish-beta", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stage == "prepare":
        prepare_stage(args)
    elif args.stage == "induce":
        induce_stage(args)
    elif args.stage == "evaluate":
        evaluate_stage(args)
    else:
        raise ValueError(args.stage)


if __name__ == "__main__":
    main()

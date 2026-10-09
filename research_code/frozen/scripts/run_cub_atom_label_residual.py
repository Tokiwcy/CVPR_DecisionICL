"""Fit the frozen CUB Atom–Gold-Label residual layer on top of MUIER."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "decision_guided_retrieval_cleanroom"
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

import run_cub_full_current_atoms as base  # noqa: E402
from dgdr.atom_label import (  # noqa: E402
    build_class_atom_prototypes,
    fit_nonnegative_residual,
    gated_classes,
    jensen_shannon_group_similarities,
    score_classes,
    standardize_across_classes,
)
from dgdr.calibration import stratified_folds  # noqa: E402
from dgdr.encoders import SiglipEncoder  # noqa: E402
from dgdr.evidence import EmbeddingStore, EvidenceStore  # noqa: E402
from dgdr.hashing import file_sha256, stable_hash  # noqa: E402
from dgdr.io import load_manifest, read_json, write_json  # noqa: E402


OUT = base.OUT / "atom_label_residual"
THRESHOLDS = (1e9, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.0)


def label_contract(train_samples, validation_samples):
    labels = tuple(sorted({sample.label for sample in train_samples}))
    lookup = {label: index for index, label in enumerate(labels)}
    if any(sample.label not in lookup for sample in validation_samples):
        raise ValueError("validation contains a label absent from train")
    train_labels = np.asarray([lookup[sample.label] for sample in train_samples], dtype=np.int64)
    query_labels = np.asarray([lookup[sample.label] for sample in validation_samples], dtype=np.int64)
    return labels, lookup, train_labels, query_labels


def encode_label_texts(labels):
    encoder = SiglipEncoder(
        str(base.SIGLIP), model_revision="local", device="cuda", batch_size=64, dtype="float16"
    )
    return encoder.encode_texts([f"a photo of a {label}" for label in labels]).astype(np.float32)


def muier_class_data(query_embeddings, train_embeddings, train_labels, label_embeddings):
    image_scores = query_embeddings.embeddings @ train_embeddings.embeddings.T
    label_scores = query_embeddings.embeddings @ label_embeddings.T
    query_count, class_count = len(query_embeddings.sample_ids), len(label_embeddings)
    class_scores = np.empty((query_count, class_count), dtype=np.float32)
    class_best_candidate = np.empty((query_count, class_count), dtype=np.int64)
    for class_index in range(class_count):
        candidates = np.flatnonzero(train_labels == class_index)
        local = image_scores[:, candidates].argmax(axis=1)
        rows = np.arange(query_count)
        class_best_candidate[:, class_index] = candidates[local]
        class_scores[:, class_index] = image_scores[rows, candidates[local]] + label_scores[:, class_index]
    return class_scores, class_best_candidate


def class_features(query_embeddings, query_evidence, train_embeddings, train_labels, label_embeddings, prototypes):
    base_scores, best_candidate = muier_class_data(
        query_embeddings, train_embeddings, train_labels, label_embeddings
    )
    atoms = jensen_shannon_group_similarities(
        query_evidence=query_evidence, prototypes=prototypes
    )
    raw = np.concatenate((base_scores[:, :, None], atoms), axis=2)
    standardized, _, _ = standardize_across_classes(raw)
    return standardized, best_candidate


def equal_weights(feature_count, alpha):
    result = np.empty(feature_count, dtype=np.float32)
    result[0] = alpha
    result[1:] = (1.0 - alpha) / (feature_count - 1)
    return result


def accuracy(classes, labels):
    return float(np.mean(np.asarray(classes) == np.asarray(labels)))


def paired_classes(left, right, labels):
    left_ok = np.asarray(left) == labels
    right_ok = np.asarray(right) == labels
    return {
        "rescue": int(np.sum(~left_ok & right_ok)),
        "harm": int(np.sum(left_ok & ~right_ok)),
        "accuracy_delta": float(right_ok.mean() - left_ok.mean()),
    }


def select_alpha(features, labels, rows):
    best_alpha, best_score = 1.0, -1.0
    for alpha in base.ALPHAS[::-1]:  # Equal scores prefer more MUIER mass.
        predicted = score_classes(features[rows], equal_weights(features.shape[2], float(alpha))).argmax(1)
        value = accuracy(predicted, labels[rows])
        if value > best_score + 1e-12:
            best_alpha, best_score = float(alpha), value
    return best_alpha, best_score


def select_threshold(base_scores, combined_scores, labels, rows):
    best_threshold, best_score = THRESHOLDS[0], -1.0
    for threshold in THRESHOLDS:  # Descending order makes ties conservative.
        predicted = gated_classes(
            base_scores=base_scores[rows], combined_scores=combined_scores[rows], threshold=threshold
        )
        value = accuracy(predicted, labels[rows])
        if value > best_score + 1e-12:
            best_threshold, best_score = threshold, value
    return float(best_threshold), float(best_score)


def classes_to_demo_ids(classes, best_candidate, train_ids):
    rows = np.arange(len(classes))
    return train_ids[best_candidate[rows, np.asarray(classes)]].astype(str).tolist()


def sliced_evidence(store, start, stop):
    return EvidenceStore(
        sample_ids=store.sample_ids[start:stop], group_ids=store.group_ids, state_ids=store.state_ids,
        state_group_index=store.state_group_index, raw_logits=store.raw_logits[start:stop],
        state_probabilities=store.state_probabilities[start:stop], signed_margins=store.signed_margins[start:stop],
        ordinal_scores=store.ordinal_scores[start:stop], applicability_scores=store.applicability_scores[start:stop],
        bank_hash=store.bank_hash, model_id=store.model_id, model_revision=store.model_revision,
        calibration_hash=store.calibration_hash, text_template=store.text_template, transform=store.transform,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train_samples = load_manifest(base.OUT / "cub.train.jsonl", expected_split="train")
    validation_samples = load_manifest(base.OUT / "cub.validation.jsonl", expected_split="validation")
    train_embeddings = EmbeddingStore.load(base.OUT / "siglip.train.embeddings")
    validation_embeddings = EmbeddingStore.load(base.OUT / "siglip.validation.embeddings")
    test_embeddings = EmbeddingStore.load(base.OUT / "siglip.test.embeddings")
    train_evidence = EvidenceStore.load(base.OUT / "siglip.train.evidence")
    validation_evidence = EvidenceStore.load(base.OUT / "siglip.validation.evidence")
    test_evidence = EvidenceStore.load(base.OUT / "siglip.test.evidence")
    labels, _, train_labels, validation_labels = label_contract(train_samples, validation_samples)
    label_embeddings = encode_label_texts(labels)
    prototypes = build_class_atom_prototypes(samples=train_samples, evidence=train_evidence)
    if prototypes.labels != labels:
        raise RuntimeError("class prototype label order drift")
    features, validation_best = class_features(
        validation_embeddings, validation_evidence, train_embeddings, train_labels,
        label_embeddings, prototypes,
    )
    folds = stratified_folds(validation_samples, 5, 42)
    base_scores = features[:, :, 0]
    oof = {
        name: np.empty(len(validation_samples), dtype=np.int64)
        for name in ("muier", "atom_label_equal_alpha_raw", "atom_label_residual_raw", "atom_label_gated_raw")
    }
    fold_rows = []
    for heldout in range(5):
        fit = np.flatnonzero(folds != heldout)
        held = np.flatnonzero(folds == heldout)
        alpha, alpha_fit = select_alpha(features, validation_labels, fit)
        weights = fit_nonnegative_residual(features, validation_labels, fit)
        combined = score_classes(features, weights)
        threshold, threshold_fit = select_threshold(base_scores, combined, validation_labels, fit)
        oof["muier"][held] = base_scores[held].argmax(1)
        oof["atom_label_equal_alpha_raw"][held] = score_classes(
            features[held], equal_weights(features.shape[2], alpha)
        ).argmax(1)
        oof["atom_label_residual_raw"][held] = combined[held].argmax(1)
        oof["atom_label_gated_raw"][held] = gated_classes(
            base_scores=base_scores[held], combined_scores=combined[held], threshold=threshold
        )
        fold_rows.append({
            "heldout": heldout, "fit_count": len(fit), "heldout_count": len(held),
            "equal_alpha": alpha, "equal_fit_accuracy": alpha_fit,
            "residual_weights": weights.tolist(), "atom_mass": float(weights[1:].sum()),
            "gate_threshold": threshold, "gate_fit_accuracy": threshold_fit,
        })
        print(json.dumps({"stage": "fold", "heldout": heldout}, ensure_ascii=False), flush=True)
    metrics = {name: accuracy(classes, validation_labels) for name, classes in oof.items()}
    gate_passed = metrics["atom_label_gated_raw"] > metrics["muier"] + 1e-12
    oof["atom_label_deployed"] = (
        oof["atom_label_gated_raw"] if gate_passed else oof["muier"]
    )
    metrics["atom_label_deployed"] = accuracy(oof["atom_label_deployed"], validation_labels)
    paired = {name + "_vs_muier": paired_classes(oof["muier"], classes, validation_labels)
              for name, classes in oof.items() if name != "muier"}
    all_rows = np.arange(len(validation_samples))
    full_alpha, _ = select_alpha(features, validation_labels, all_rows)
    full_weights = fit_nonnegative_residual(features, validation_labels, all_rows)
    full_combined = score_classes(features, full_weights)
    full_threshold, _ = select_threshold(base_scores, full_combined, validation_labels, all_rows)
    calibration = {
        "method": "classification-adapted MUIER + class-conditional atom-label residual v1",
        "protocol": str(PROJECT / "docs" / "cub_atom_label_residual_protocol_20260905_zh.md"),
        "features": ["muier_class_max"] + list(train_evidence.group_ids),
        "prototype_source": "train gold labels only", "compatibility": "1-normalized-JS-divergence",
        "class_feature_normalization": "per-query across-class z-score",
        "folds": folds.tolist(), "folds_hash": stable_hash(folds.tolist()),
        "outer_folds": fold_rows, "oof_same_class_accuracy": metrics, "paired": paired,
        "primary_raw": "atom_label_gated_raw", "gate_passed": gate_passed,
        "selected_deployment": "atom_label_gated_raw" if gate_passed else "muier",
        "full_equal_alpha": full_alpha, "full_residual_weights": full_weights.tolist(),
        "full_atom_mass": float(full_weights[1:].sum()), "full_gate_threshold": full_threshold,
        "threshold_grid": list(THRESHOLDS), "test_labels_used": False,
        "bank_hash": train_evidence.bank_hash,
        "base_calibration_sha256": file_sha256(base.OUT / "muier_atom_extension" / "calibration.json"),
    }
    write_json(OUT / "calibration.json", calibration)
    train_ids = np.asarray(train_embeddings.sample_ids)
    write_json(OUT / "validation_plan.json", {
        "query_ids": list(validation_embeddings.sample_ids),
        "methods": {name: classes_to_demo_ids(classes, validation_best, train_ids)
                    for name, classes in oof.items()},
        "oof_same_class_accuracy": metrics, "paired": paired, "test_labels_used": False,
    })

    method_demo_ids = {name: [] for name in (
        "muier", "atom_label_equal_alpha_raw", "atom_label_residual_raw",
        "atom_label_gated_raw", "atom_label_deployed",
    )}
    test_ids = test_embeddings.sample_ids
    for start in range(0, len(test_ids), 64):
        stop = min(start + 64, len(test_ids))
        block_embeddings = EmbeddingStore(
            test_ids[start:stop], test_embeddings.embeddings[start:stop],
            test_embeddings.model_id, test_embeddings.model_revision,
        )
        block_features, block_best = class_features(
            block_embeddings, sliced_evidence(test_evidence, start, stop), train_embeddings,
            train_labels, label_embeddings, prototypes,
        )
        block_base = block_features[:, :, 0]
        block_combined = score_classes(block_features, full_weights)
        classes = {
            "muier": block_base.argmax(1),
            "atom_label_equal_alpha_raw": score_classes(
                block_features, equal_weights(block_features.shape[2], full_alpha)
            ).argmax(1),
            "atom_label_residual_raw": block_combined.argmax(1),
            "atom_label_gated_raw": gated_classes(
                base_scores=block_base, combined_scores=block_combined, threshold=full_threshold
            ),
        }
        classes["atom_label_deployed"] = classes["atom_label_gated_raw"] if gate_passed else classes["muier"]
        for name, predicted_classes in classes.items():
            method_demo_ids[name].extend(classes_to_demo_ids(predicted_classes, block_best, train_ids))
        print(json.dumps({"stage": "test_plan", "completed": stop, "total": len(test_ids)}), flush=True)
    old_plan = read_json(base.OUT / "test_plan.json")
    if method_demo_ids["muier"] != old_plan["methods"]["muier_classification_adapted"]:
        mismatch = sum(a != b for a, b in zip(
            method_demo_ids["muier"], old_plan["methods"]["muier_classification_adapted"]
        ))
        raise RuntimeError(f"MUIER parity failed for {mismatch} queries")
    plan = {
        "split": "test", "query_ids": list(test_ids), "methods": method_demo_ids,
        "target_inference_method": "atom_label_deployed", "shots": 1,
        "candidate_pool": "complete historical deduplicated train split",
        "calibration_sha256": file_sha256(OUT / "calibration.json"),
        "test_labels_used_to_create_plan": False,
    }
    write_json(OUT / "test_plan.json", plan)
    write_json(OUT / "test_plan.freeze.json", {
        "sha256": file_sha256(OUT / "test_plan.json"), "test_labels_used": False,
    })

    # Diagnostics are deliberately computed only after the test plan is frozen.
    test_samples = load_manifest(base.OUT / "cub.test.jsonl", expected_split="test")
    truth = {sample.sample_id: sample.label for sample in test_samples}
    train_truth = {sample.sample_id: sample.label for sample in train_samples}
    diagnostics = {}
    for name, demos in method_demo_ids.items():
        correct = np.asarray([train_truth[demo] == truth[query] for query, demo in zip(test_ids, demos)])
        diagnostics[name] = {"accuracy": float(correct.mean())}
    write_json(OUT / "test_retrieval_diagnostics.json", diagnostics)
    print(json.dumps({"stage": "complete", "oof": metrics, "paired": paired,
                      "gate_passed": gate_passed, "test_diagnostic": diagnostics},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()


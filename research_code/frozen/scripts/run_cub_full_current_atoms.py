"""Freeze, calibrate, and retrieve the full CUB study for the current atom bank.

This driver deliberately writes the test retrieval plan before reading test labels.
It reuses only the verified all-image SigLIP cache; atom evidence and every learned
coefficient are recomputed for this exact bank.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "decision_guided_retrieval_cleanroom"
sys.path.insert(0, str(PROJECT / "src"))

from dgdr.calibration import solve_nnls_from_gram, stratified_folds  # noqa: E402
from dgdr.encoders import SiglipEncoder  # noqa: E402
from dgdr.evidence import (  # noqa: E402
    EmbeddingStore,
    EvidenceStore,
    apply_evidence_calibration,
    build_evidence_store,
    fit_train_evidence_calibration,
    group_similarity_matrix,
)
from dgdr.hashing import file_sha256, stable_hash  # noqa: E402
from dgdr.io import load_atom_bank, load_manifest, read_json, read_jsonl, save_manifest, write_json  # noqa: E402
from dgdr.retrieval import global_similarity_matrix  # noqa: E402
from dgdr.schema import Sample  # noqa: E402


OUT = PROJECT / "artifacts" / "cub_full_current_atoms_20260904"
BANK = PROJECT / "artifacts" / "cub_siglip_flair_pilot" / "atom_bank.final.json"
SOURCE = ROOT / "concept_icl" / "runs" / "cub_siglip_random59_codesfix_20260901"
OLD_QWEN3 = ROOT / "concept_icl_v2" / "runs" / "qwen3_4b_1shot" / "cub"
CANONICAL = ROOT / "data" / "cub_200_2011" / "manifest.jsonl"
ALL_IMAGE_CACHE = ROOT / "artifacts" / "cub200_task_aware" / "siglip2_all_embeddings.npz"
SIGLIP = ROOT / "models" / "siglip2-base-patch16-224"
QUESTION = "Identify the bird species shown in the image."
EXPECTED = {"train": 5993, "validation": 1000, "test": 4794}
ALPHAS = np.asarray((0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0), dtype=np.float32)
MLP_EPOCHS = 80
MLP_BATCH = 64
MLP_LR = 3e-3
MLP_WEIGHT_DECAY = 1e-4


def status(stage: str, **values: object) -> None:
    row = {"stage": stage, "time": time.time(), **values}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "status.json", row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def prepare_manifests() -> dict[str, list[Sample]]:
    canonical = {row["id"]: row for row in read_jsonl(CANONICAL)}
    result: dict[str, list[Sample]] = {}
    used: set[str] = set()
    for split in ("train", "validation", "test"):
        inputs = read_json(SOURCE / f"{split}_inputs.json")
        answers = read_json(SOURCE / f"{split}_answers.json")
        if len(inputs) != EXPECTED[split]:
            raise RuntimeError(f"unexpected {split} count: {len(inputs)}")
        samples = []
        for row in inputs:
            item_id = row["id"]
            base = canonical[item_id]
            label = answers[item_id]["label"]
            if base["label"] != label or item_id in used:
                raise RuntimeError(f"split/label contract mismatch: {item_id}")
            if split == "train" and base["split"] != "train":
                raise RuntimeError(f"non-train candidate: {item_id}")
            if split != "train" and base["split"] != split:
                raise RuntimeError(f"frozen split mismatch: {item_id}")
            if Path(row["image"]).resolve() != Path(base["image"]).resolve():
                raise RuntimeError(f"image path mismatch: {item_id}")
            samples.append(Sample(
                sample_id=item_id,
                image_path=row["image"],
                label=label,
                split=split,
                question=QUESTION,
                answer=label,
                image_sha256=row.get("image_sha256"),
            ))
            used.add(item_id)
        path = OUT / f"cub.{split}.jsonl"
        save_manifest(path, samples)
        result[split] = load_manifest(path, expected_split=split)
    canonical_train = {item_id for item_id, row in canonical.items() if row["split"] == "train"}
    excluded = sorted(canonical_train - {x.sample_id for x in result["train"]})
    write_json(OUT / "data_contract.json", {
        "counts": {k: len(v) for k, v in result.items()},
        "source": str(SOURCE),
        "canonical_manifest_sha256": file_sha256(CANONICAL),
        "source_input_sha256": {k: file_sha256(SOURCE / f"{k}_inputs.json") for k in result},
        "excluded_official_train_ids_after_historical_dedup": excluded,
        "disjoint": len(used) == sum(EXPECTED.values()),
    })
    return result


def make_embedding_stores(samples: dict[str, list[Sample]]) -> dict[str, EmbeddingStore]:
    with np.load(ALL_IMAGE_CACHE, allow_pickle=False) as arrays:
        all_ids = arrays["sample_ids"].astype(str)
        all_values = arrays["values"].astype(np.float32)
    if len(all_ids) != len(set(all_ids.tolist())):
        raise RuntimeError("all-image cache contains duplicate IDs")
    lookup = {item_id: index for index, item_id in enumerate(all_ids)}
    stores: dict[str, EmbeddingStore] = {}
    for split, rows in samples.items():
        ids = tuple(row.sample_id for row in rows)
        values = np.asarray([all_values[lookup[item_id]] for item_id in ids], dtype=np.float32)
        values /= np.linalg.norm(values, axis=1, keepdims=True).clip(min=1e-12)
        store = EmbeddingStore(ids, values, str(SIGLIP), "local")
        store.save(OUT / f"siglip.{split}.embeddings")
        stores[split] = store
    old = np.load(SOURCE / "train_image.npz", allow_pickle=False)
    old_ids = old["ids"].astype(str)
    old_values = old["values"].astype(np.float32)
    old_values /= np.linalg.norm(old_values, axis=1, keepdims=True).clip(min=1e-12)
    aligned = np.asarray([stores["train"].embeddings[stores["train"].sample_ids.index(x)] for x in old_ids])
    write_json(OUT / "global_embedding_reuse_audit.json", {
        "source_cache": str(ALL_IMAGE_CACHE),
        "source_cache_sha256": file_sha256(ALL_IMAGE_CACHE),
        "old_train_cache_sha256": file_sha256(SOURCE / "train_image.npz"),
        "max_abs_difference_vs_old_train_cache": float(np.max(np.abs(aligned - old_values))),
        "minimum_row_cosine_vs_old_train_cache": float(np.min(np.sum(aligned * old_values, axis=1))),
    })
    return stores


def make_evidence_stores(
    samples: dict[str, list[Sample]], embeddings: dict[str, EmbeddingStore]
) -> dict[str, EvidenceStore]:
    bank = load_atom_bank(BANK)
    if bank.bank_hash != "5e7c43f63645ac50d2c1da6caa9243ca25cb9ebc078abd6d881f6893c6969c81":
        raise RuntimeError("current atom bank drifted")
    encoder = SiglipEncoder(str(SIGLIP), model_revision="local", device="cuda", batch_size=64, dtype="float16")
    raw_train = build_evidence_store(
        samples=samples["train"], image_embeddings=embeddings["train"], bank=bank, encoder=encoder
    )
    calibration = fit_train_evidence_calibration(
        samples=samples["train"], bank=bank, uncalibrated_store=raw_train
    )
    calibration.save(OUT / "siglip.evidence_calibration.json")
    stores = {
        "train": apply_evidence_calibration(store=raw_train, bank=bank, calibration=calibration)
    }
    stores["train"].save(OUT / "siglip.train.evidence")
    for split in ("validation", "test"):
        raw = build_evidence_store(
            samples=samples[split], image_embeddings=embeddings[split], bank=bank, encoder=encoder
        )
        stores[split] = apply_evidence_calibration(store=raw, bank=bank, calibration=calibration)
        stores[split].save(OUT / f"siglip.{split}.evidence")
    return stores


def feature_tensor(qe: EmbeddingStore, te: EmbeddingStore, qv: EvidenceStore, tv: EvidenceStore) -> np.ndarray:
    count_q, count_t = len(qe.sample_ids), len(te.sample_ids)
    result = np.empty((count_q, count_t, 1 + len(qv.group_ids)), dtype=np.float32)
    result[:, :, 0] = global_similarity_matrix(qe, te).astype(np.float32)
    for group in range(len(qv.group_ids)):
        result[:, :, group + 1] = group_similarity_matrix(qv, tv, group).astype(np.float32)
    return result


def top_indices(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.einsum("qtf,f->qt", features, weights, optimize=True).argmax(axis=1)


def accuracy(indices: np.ndarray, query_labels: np.ndarray, train_labels: np.ndarray) -> float:
    return float(np.mean(train_labels[np.asarray(indices)] == query_labels))


def nnls_statistics(features: np.ndarray, query_labels: np.ndarray, train_labels: np.ndarray):
    query_count, _, feature_count = features.shape
    grams = np.empty((query_count, feature_count, feature_count), dtype=np.float64)
    crosses = np.empty((query_count, feature_count), dtype=np.float64)
    for qi in range(query_count):
        values = features[qi].astype(np.float64)
        centered = values - values.mean(axis=0, keepdims=True)
        positive = train_labels == query_labels[qi]
        pc = int(positive.sum())
        nc = len(train_labels) - pc
        weights = np.where(positive, 0.5 / pc, 0.5 / nc)
        target = np.where(positive, 1.0, -1.0)
        grams[qi] = centered.T @ (centered * weights[:, None])
        crosses[qi] = centered.T @ (weights * target)
    return grams, crosses


def fit_nnls(grams: np.ndarray, crosses: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return solve_nnls_from_gram(grams[rows].sum(axis=0), crosses[rows].sum(axis=0)).astype(np.float32)


def equal_weights(feature_count: int, alpha: float, support: np.ndarray | None = None) -> np.ndarray:
    weights = np.zeros(feature_count, dtype=np.float32)
    weights[0] = alpha
    if support is None:
        support = np.arange(1, feature_count)
    if len(support):
        weights[support] = (1.0 - alpha) / len(support)
    else:
        weights[0] = 1.0
    return weights


def select_alpha(features, rows, query_labels, train_labels, support=None):
    best_alpha, best_score = None, -1.0
    for alpha in ALPHAS:
        indices = top_indices(features[rows], equal_weights(features.shape[2], float(alpha), support))
        score = accuracy(indices, query_labels[rows], train_labels)
        if score > best_score + 1e-12:
            best_alpha, best_score = float(alpha), score
    return float(best_alpha), float(best_score)


class FactorGate(nn.Module):
    def __init__(self, input_dim: int, group_count: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, group_count)
        self.fc2 = nn.Linear(group_count, group_count + 1)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return torch.softmax(self.fc2(torch.nn.functional.gelu(self.fc1(x))), dim=1)


def fit_gate(features, query_input, query_labels, train_labels, rows, seed):
    device = torch.device("cuda")
    center = query_input[rows].mean(0, keepdims=True).astype(np.float32)
    scale = np.maximum(query_input[rows].std(0, keepdims=True), 1e-6).astype(np.float32)
    x = torch.from_numpy((query_input - center) / scale).to(device)
    ft = torch.from_numpy(np.transpose(features, (2, 0, 1))).to(device)
    qy = torch.from_numpy(query_labels).to(device)
    ty = torch.from_numpy(train_labels).to(device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = FactorGate(query_input.shape[1], features.shape[2] - 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY)
    order = torch.from_numpy(rows.astype(np.int64)).to(device)
    model.train()
    final_loss = 0.0
    for epoch in range(MLP_EPOCHS):
        total = 0.0
        for start in range(0, len(order), MLP_BATCH):
            take = order[start:start + MLP_BATCH]
            gate = model(x[take])
            score = torch.einsum("bf,fbt->bt", gate, ft[:, take, :])
            same = ty[None, :] == qy[take, None]
            loss = (torch.logsumexp(score, dim=1) - torch.logsumexp(score.masked_fill(~same, -torch.inf), dim=1)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(take)
        final_loss = total / len(order)
    model.eval()
    del ft, qy, ty, order
    torch.cuda.empty_cache()
    return model, center, scale, final_loss


@torch.inference_mode()
def gate_indices(model, center, scale, query_input, features):
    x = torch.from_numpy((query_input - center) / scale).to("cuda")
    ft = torch.from_numpy(np.transpose(features, (2, 0, 1))).to("cuda")
    output = []
    mean = np.zeros(features.shape[2], dtype=np.float64)
    for start in range(0, len(query_input), 128):
        stop = min(start + 128, len(query_input))
        weights = model(x[start:stop])
        score = torch.einsum("bf,fbt->bt", weights, ft[:, start:stop, :])
        output.append(score.argmax(1).cpu().numpy())
        mean += weights.sum(0).cpu().numpy()
    del x, ft
    torch.cuda.empty_cache()
    return np.concatenate(output), (mean / len(query_input)).tolist()


def query_input(evidence: EvidenceStore) -> np.ndarray:
    return evidence.state_probabilities.astype(np.float32)


def ids_from_indices(indices: np.ndarray, train_ids: np.ndarray) -> list[str]:
    return train_ids[np.asarray(indices)].astype(str).tolist()


def plan_and_calibrate(samples, embeddings, evidence, historical_plan_path=OLD_QWEN3 / "test_1shot_context_plan.json"):
    train_labels_text = np.asarray([x.label for x in samples["train"]])
    labels = {x: i for i, x in enumerate(sorted(set(train_labels_text.tolist())))}
    train_labels = np.asarray([labels[x] for x in train_labels_text], dtype=np.int64)
    val_labels = np.asarray([labels[x.label] for x in samples["validation"]], dtype=np.int64)
    train_ids = np.asarray(embeddings["train"].sample_ids)
    val_features = feature_tensor(embeddings["validation"], embeddings["train"], evidence["validation"], evidence["train"])
    np.save(OUT / "validation_features.npy", val_features)
    grams, crosses = nnls_statistics(val_features, val_labels, train_labels)
    folds = stratified_folds(samples["validation"], 5, 42)
    feature_count = val_features.shape[2]
    global_w = np.zeros(feature_count, dtype=np.float32); global_w[0] = 1.0
    atom_w = equal_weights(feature_count, 0.0)
    fixed_w = np.ones(feature_count, dtype=np.float32) / feature_count
    val_indices = {
        "rices": top_indices(val_features, global_w),
        "mmices": top_indices(val_features, global_w),
        "atom_only_equal": top_indices(val_features, atom_w),
        "fixed_all_equal": top_indices(val_features, fixed_w),
    }
    val_indices["equal_alpha"] = np.empty(len(val_labels), dtype=np.int64)
    val_indices["nnls_raw"] = np.empty(len(val_labels), dtype=np.int64)
    val_indices["support_equal"] = np.empty(len(val_labels), dtype=np.int64)
    val_indices["mlp_raw"] = np.empty(len(val_labels), dtype=np.int64)
    qinput = query_input(evidence["validation"])
    fold_rows = []
    for heldout in range(5):
        fit = np.flatnonzero(folds != heldout)
        held = np.flatnonzero(folds == heldout)
        alpha, alpha_fit_score = select_alpha(val_features, fit, val_labels, train_labels)
        val_indices["equal_alpha"][held] = top_indices(val_features[held], equal_weights(feature_count, alpha))
        nw = fit_nnls(grams, crosses, fit)
        val_indices["nnls_raw"][held] = top_indices(val_features[held], nw)
        support = np.flatnonzero(nw[1:] > 1e-12) + 1
        support_alpha, support_fit_score = select_alpha(val_features, fit, val_labels, train_labels, support)
        val_indices["support_equal"][held] = top_indices(
            val_features[held], equal_weights(feature_count, support_alpha, support)
        )
        model, center, scale, loss = fit_gate(
            val_features, qinput, val_labels, train_labels, fit,
            int(stable_hash({"method": "mlp", "fold": heldout})[:8], 16),
        )
        predicted, mean_weights = gate_indices(model, center, scale, qinput[held], val_features[held])
        val_indices["mlp_raw"][held] = predicted
        checkpoint = OUT / f"mlp.fold{heldout}.npz"
        np.savez_compressed(checkpoint, center=center, scale=scale,
                            **{f"parameter__{k}": v.detach().cpu().numpy() for k, v in model.state_dict().items()})
        fold_rows.append({
            "heldout_fold": heldout, "fit_count": len(fit), "heldout_count": len(held),
            "equal_alpha": alpha, "equal_fit_accuracy": alpha_fit_score,
            "nnls_weights": nw.tolist(), "support": support.tolist(),
            "support_alpha": support_alpha, "support_fit_accuracy": support_fit_score,
            "mlp_loss": loss, "mlp_mean_weights": mean_weights,
            "mlp_checkpoint_sha256": file_sha256(checkpoint),
        })
        del model
        torch.cuda.empty_cache()
        status("calibration_fold_complete", fold=heldout)
    metrics = {name: accuracy(value, val_labels, train_labels) for name, value in val_indices.items()}
    nnls_pass = metrics["nnls_raw"] > metrics["rices"] + 1e-12
    val_indices["nnls_deployed"] = val_indices["nnls_raw"] if nnls_pass else val_indices["rices"]
    best_ref = "nnls_raw" if metrics["nnls_raw"] > metrics["rices"] + 1e-12 else "rices"
    mlp_pass = metrics["mlp_raw"] > max(metrics["nnls_raw"], metrics["rices"]) + 1e-12
    val_indices["mlp_deployed"] = val_indices["mlp_raw"] if mlp_pass else val_indices[best_ref]
    metrics.update({
        "nnls_deployed": accuracy(val_indices["nnls_deployed"], val_labels, train_labels),
        "mlp_deployed": accuracy(val_indices["mlp_deployed"], val_labels, train_labels),
    })
    all_rows = np.arange(len(val_labels))
    full_alpha, _ = select_alpha(val_features, all_rows, val_labels, train_labels)
    full_nnls = fit_nnls(grams, crosses, all_rows)
    full_support = np.flatnonzero(full_nnls[1:] > 1e-12) + 1
    full_support_alpha, _ = select_alpha(val_features, all_rows, val_labels, train_labels, full_support)
    full_model, full_center, full_scale, full_loss = fit_gate(
        val_features, qinput, val_labels, train_labels, all_rows,
        int(stable_hash({"method": "mlp", "fold": "full"})[:8], 16),
    )
    full_checkpoint = OUT / "mlp.full_validation.npz"
    np.savez_compressed(full_checkpoint, center=full_center, scale=full_scale,
                        **{f"parameter__{k}": v.detach().cpu().numpy() for k, v in full_model.state_dict().items()})
    calibration = {
        "atom_bank_sha256": file_sha256(BANK),
        "bank_hash": load_atom_bank(BANK).bank_hash,
        "folds": folds.tolist(), "folds_hash": stable_hash(folds.tolist()),
        "alpha_grid": ALPHAS.tolist(), "oof_same_class_accuracy": metrics,
        "nnls_gate_passed": nnls_pass, "mlp_gate_passed": mlp_pass,
        "mlp_fallback_reference": None if mlp_pass else best_ref,
        "outer_folds": fold_rows, "full_equal_alpha": full_alpha,
        "full_nnls_weights": full_nnls.tolist(), "full_support": full_support.tolist(),
        "full_support_alpha": full_support_alpha, "full_mlp_loss": full_loss,
        "full_mlp_checkpoint_sha256": file_sha256(full_checkpoint),
        "test_labels_used": False,
    }
    write_json(OUT / "calibration.json", calibration)
    val_plan = {
        "split": "validation", "query_ids": list(embeddings["validation"].sample_ids),
        "methods": {k: ids_from_indices(v, train_ids) for k, v in val_indices.items()},
        "same_class_accuracy_oof": metrics, "test_labels_used": False,
    }
    write_json(OUT / "validation_plan.json", val_plan)

    # Baseline label text embeddings are derived only from known training labels.
    encoder = SiglipEncoder(str(SIGLIP), model_revision="local", device="cuda", batch_size=64, dtype="float16")
    unique_labels = sorted(set(train_labels_text.tolist()))
    label_embeddings = encoder.encode_texts([f"a photo of a {label}" for label in unique_labels])
    label_lookup = {label: label_embeddings[i] for i, label in enumerate(unique_labels)}
    candidate_label_embeddings = np.asarray([label_lookup[x] for x in train_labels_text], dtype=np.float32)
    old_rows = None
    if historical_plan_path is not None:
        old_plan = read_json(historical_plan_path)
        old_rows = {row["query_id"]: row for row in old_plan["rows"]}
    random_rng = np.random.default_rng(42)
    methods = {name: [] for name in (
        "random", "rices", "mmices", "muier_classification_adapted", "atom_only_equal",
        "fixed_all_equal", "equal_alpha", "nnls_raw", "nnls_deployed", "support_equal",
        "mlp_raw", "mlp_deployed",
    )}
    test_ids = list(embeddings["test"].sample_ids)
    for start in range(0, len(test_ids), 64):
        stop = min(start + 64, len(test_ids))
        qe = EmbeddingStore(tuple(test_ids[start:stop]), embeddings["test"].embeddings[start:stop], str(SIGLIP), "local")
        qv = EvidenceStore(
            sample_ids=tuple(test_ids[start:stop]), group_ids=evidence["test"].group_ids,
            state_ids=evidence["test"].state_ids, state_group_index=evidence["test"].state_group_index,
            raw_logits=evidence["test"].raw_logits[start:stop],
            state_probabilities=evidence["test"].state_probabilities[start:stop],
            signed_margins=evidence["test"].signed_margins[start:stop],
            ordinal_scores=evidence["test"].ordinal_scores[start:stop],
            applicability_scores=evidence["test"].applicability_scores[start:stop],
            bank_hash=evidence["test"].bank_hash, model_id=evidence["test"].model_id,
            model_revision=evidence["test"].model_revision,
            calibration_hash=evidence["test"].calibration_hash,
            text_template=evidence["test"].text_template, transform=evidence["test"].transform,
        )
        block = feature_tensor(qe, embeddings["train"], qv, evidence["train"])
        block_indices = {
            "rices": top_indices(block, global_w),
            "mmices": top_indices(block, global_w),
            "atom_only_equal": top_indices(block, atom_w),
            "fixed_all_equal": top_indices(block, fixed_w),
            "equal_alpha": top_indices(block, equal_weights(feature_count, full_alpha)),
            "nnls_raw": top_indices(block, full_nnls),
            "support_equal": top_indices(block, equal_weights(feature_count, full_support_alpha, full_support)),
        }
        block_indices["nnls_deployed"] = block_indices["nnls_raw"] if nnls_pass else block_indices["rices"]
        raw_mlp, _ = gate_indices(full_model, full_center, full_scale,
                                  query_input(evidence["test"])[start:stop], block)
        block_indices["mlp_raw"] = raw_mlp
        block_indices["mlp_deployed"] = raw_mlp if mlp_pass else block_indices[best_ref]
        image_image = 2.0 * block[:, :, 0] - 1.0
        image_text = qe.embeddings @ candidate_label_embeddings.T
        block_indices["muier_classification_adapted"] = (image_image + image_text).argmax(axis=1)
        for name, indices in block_indices.items():
            methods[name].extend(ids_from_indices(indices, train_ids))
        for query_id in test_ids[start:stop]:
            methods["random"].append(old_rows[query_id]["random_demo_id"] if old_rows is not None
                                     else str(train_ids[random_rng.integers(len(train_ids))]))
        status("test_plan_progress", completed=stop, total=len(test_ids))
    if old_rows is not None and methods["rices"] != [old_rows[x]["rices_demo_id"] for x in test_ids]:
        mismatches = sum(a != old_rows[q]["rices_demo_id"] for q, a in zip(test_ids, methods["rices"]))
        raise RuntimeError(f"RICES reuse parity failed for {mismatches} queries")
    if methods["mmices"] != methods["rices"]:
        raise RuntimeError("fixed-question MMICES must equal RICES")
    test_plan = {
        "split": "test", "query_ids": test_ids, "methods": methods,
        "calibration_sha256": file_sha256(OUT / "calibration.json"),
        "candidate_pool": "complete historical deduplicated train split",
        "shots": 1, "test_labels_used_to_create_plan": False,
        "rices_exactly_matches_reused_qwen3_plan": old_rows is not None,
        "mmices_exactly_equals_rices": True,
    }
    write_json(OUT / "test_plan.json", test_plan)
    write_json(OUT / "test_plan.freeze.json", {
        "test_plan_sha256": file_sha256(OUT / "test_plan.json"),
        "created_before_test_label_diagnostics": True,
    })
    # Only after the plan is frozen do we inspect test labels for retrieval diagnostics.
    truth = {x.sample_id: x.label for x in samples["test"]}
    train_truth = {x.sample_id: x.label for x in samples["train"]}
    diagnostics = {}
    for name, demos in methods.items():
        correct = np.asarray([train_truth[d] == truth[q] for q, d in zip(test_ids, demos)])
        by_class = [correct[np.asarray([truth[q] == label for q in test_ids])].mean()
                    for label in sorted(set(truth.values()))]
        diagnostics[name] = {"accuracy": float(correct.mean()), "macro_accuracy": float(np.mean(by_class))}
    write_json(OUT / "test_retrieval_diagnostics.json", diagnostics)
    status("complete", oof=metrics, test_retrieval=diagnostics)


def load_stores(samples):
    embeddings = {s: EmbeddingStore.load(OUT / f"siglip.{s}.embeddings") for s in samples}
    evidence = {s: EvidenceStore.load(OUT / f"siglip.{s}.evidence") for s in samples}
    return embeddings, evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "prepare", "retrieve"), default="all")
    args = parser.parse_args()
    torch.set_num_threads(4)
    samples = prepare_manifests()
    if args.stage in ("all", "prepare"):
        status("embedding_stores")
        embeddings = make_embedding_stores(samples)
        status("evidence_stores")
        evidence = make_evidence_stores(samples, embeddings)
    else:
        embeddings, evidence = load_stores(samples)
    if args.stage in ("all", "retrieve"):
        plan_and_calibrate(samples, embeddings, evidence)


if __name__ == "__main__":
    main()

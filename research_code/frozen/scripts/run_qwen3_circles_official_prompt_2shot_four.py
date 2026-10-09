"""Faithful-prompt 2-shot CIRCLES adaptation on CUB, DTD, Aircraft, and Dogs.

The published 32-demo setup is reduced to one IR and one CIR example.  CIRCLES
prompt grouping, attribute/caption context, repeated query, CLIP ViT-g retrieval,
direct label generation, and exact-match scoring follow the official code.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import shutil
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
from PIL import Image

import run_qwen3_circles_five as legacy
import run_qwen3_2shot_five as common
from dgdr.atom_label import build_class_atom_prototypes, score_classes
from dgdr.calibration import stratified_folds
from dgdr.hashing import file_sha256, stable_hash
from dgdr.metrics import accuracy_metrics


PROJECT = SCRIPT_DIR.parent
OUT = PROJECT / "artifacts/qwen3_circles_official_prompt_2shot_seven_20260910"
SOURCE = PROJECT / "artifacts/qwen3_circles_five_20260909"
PRIOR_OFFICIAL = PROJECT / "artifacts/qwen3_circles_official_prompt_2shot_four_20260910"
PRIOR_FIVE = PROJECT / "artifacts/qwen3_2shot_five_20260908"
ORDER = ("dtd", "aircraft", "cub", "dogs")
METHODS = (
    "random_official_prompt",
    "rices_clip_official_prompt",
    "muier_classification_adapted_siglip_official_prompt",
    "rices_clip_atom_class_equal_raw_official_prompt",
    "muier_classification_adapted_siglip_atom_class_equal_raw_official_prompt",
    "circles_official_prompt",
    "circles_atom_class_equal_raw_official_prompt",
)
BASELINE_METHODS = METHODS[:5]
CIRCLES_METHODS = METHODS[5:]
ATOM_PARENTS = {METHODS[3]: METHODS[1], METHODS[4]: METHODS[2], METHODS[6]: METHODS[5]}
BATCH_BY_DATASET = {"dtd": 2, "aircraft": 2, "cub": 8, "dogs": 8}
PROMPT_VERSION = "official-circles-seven-method-classification-2shot-v1"
RUN_DOC = PROJECT / "docs/qwen3_circles_official_prompt_2shot_seven_20260910_zh.md"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    common.write(Path(path), value)


def emit(stage, **values):
    row = dict(stage=stage, time=time.time(), **values)
    write(OUT / "status.json", row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def held():
    if any((OUT / name).exists() for name in ("PAUSED_BY_USER.json", "DIAGNOSTIC_HOLD.json", "WATCHDOG_STOP")):
        raise RuntimeError("official-prompt CIRCLES queue held by control file")


def source(dataset):
    return common.source(dataset)


def samples(dataset, split):
    return legacy.samples(dataset, split)


def copy_verified(src, dst):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if file_sha256(src) != file_sha256(dst):
        raise ValueError(f"reuse copy hash mismatch: {src}")


def seed_reusable_cache(dataset):
    src = SOURCE / dataset
    dst = OUT / dataset
    dst.mkdir(parents=True, exist_ok=True)
    copied = {}
    names = [
        "counterfactual_descriptions.validation.jsonl",
        "counterfactual_descriptions.test.jsonl",
        "description.validation.complete.json",
        "description.test.complete.json",
        "clip.train.npz", "clip.train.npz.json",
        "clip.validation.npz", "clip.validation.npz.json",
        "clip.test.npz", "clip.test.npz.json",
        "clip.captions.validation.npz", "clip.captions.validation.npz.json",
        "clip.captions.test.npz", "clip.captions.test.npz.json",
    ]
    for name in names:
        old = src / name
        if old.is_file():
            copy_verified(old, dst / name)
            copied[str((dst / name).resolve())] = {
                "source": str(old.resolve()), "sha256": file_sha256(old)}
    write(dst / "reuse_receipt.json", {
        "dataset": dataset, "files": copied,
        "descriptions_shot_invariant": True,
        "clip_embeddings_shot_and_target_prompt_invariant": True,
        "test_labels_used": False,
    })


def prepare_dataset(dataset):
    held()
    seed_reusable_cache(dataset)
    original_order, original_out, original_batch = legacy.ORDER, legacy.OUT, legacy.DESCRIPTION_BATCH
    try:
        legacy.ORDER = (dataset,)
        legacy.OUT = OUT
        legacy.DESCRIPTION_BATCH = BATCH_BY_DATASET[dataset]
        legacy.validate_inputs(True)
        legacy.describe_all()
        legacy.embed_all()
    finally:
        legacy.ORDER, legacy.OUT, legacy.DESCRIPTION_BATCH = original_order, original_out, original_batch


def topk(scores, k):
    return np.argsort(scores, axis=1)[:, -k:][:, ::-1]


def pair_precision(pairs, truth_y, train_y):
    return float(np.mean([
        sum(int(train_y[index] == truth) for index in pair) / len(pair)
        for pair, truth in zip(pairs, truth_y)]))


def single_channel_atom_features(scores, train_y, query_evidence, prototypes, class_count):
    base_scores = np.empty((len(scores), class_count), np.float32)
    for class_index in range(class_count):
        candidates = np.flatnonzero(train_y == class_index)
        base_scores[:, class_index] = scores[:, candidates].max(axis=1)
    atoms = common.cub.shared.jensen_shannon_group_similarities(
        query_evidence=query_evidence, prototypes=prototypes)
    raw = np.concatenate((base_scores[:, :, None], atoms), axis=2)
    standardized, _, _ = common.cub.shared.standardize_across_classes(raw)
    return standardized


def pairs_within_classes(scores, train_y, classes):
    pairs = []
    for row, class_index in enumerate(np.asarray(classes, np.int64)):
        candidates = np.flatnonzero(train_y == class_index)
        chosen = candidates[topk(scores[row:row + 1, candidates], 2)[0]]
        pairs.append([int(chosen[0]), int(chosen[1])])
    return pairs


def fit_class_equal(features, validation_y, validation):
    folds = stratified_folds(validation, 5, 42)
    oof = np.empty(len(validation), np.int64)
    history = []
    for fold in range(5):
        fit = np.flatnonzero(folds != fold)
        hold = np.flatnonzero(folds == fold)
        alpha, fit_score = common.fit_alpha(features, validation_y, fit)
        weights = common.cub.shared.equal_weights(features.shape[2], alpha)
        oof[hold] = score_classes(features[hold], weights).argmax(1)
        history.append({"fold": fold, "alpha": alpha, "weights": weights.tolist(),
                        "fit_class_accuracy": fit_score, "fit_count": len(fit),
                        "heldout_count": len(hold)})
    alpha, fit_score = common.fit_alpha(
        features, validation_y, np.arange(len(validation)))
    weights = common.cub.shared.equal_weights(features.shape[2], alpha)
    return oof, alpha, weights, fit_score, history


def select_dataset(dataset):
    held()
    out = OUT / dataset
    common.verify_dataset(dataset, True)
    train, validation, test = (samples(dataset, split)
                               for split in ("train", "validation", "test"))
    train_ids = [x.sample_id for x in train]
    validation_ids = [x.sample_id for x in validation]
    test_ids = [x.sample_id for x in test]
    labels = tuple(sorted({x.label for x in train}))
    lookup = {label: i for i, label in enumerate(labels)}
    train_y = np.asarray([lookup[x.label] for x in train], np.int64)
    validation_y = np.asarray([lookup[x.label] for x in validation], np.int64)
    train_embeddings = legacy.load_embeddings(out / "clip.train.npz", train_ids)
    validation_embeddings = legacy.load_embeddings(
        out / "clip.validation.npz", validation_ids)
    validation_captions = legacy.load_embeddings(
        out / "clip.captions.validation.npz", validation_ids)
    train_evidence = common.evidence(dataset, "train")
    validation_evidence = common.evidence(dataset, "validation")
    prototypes = build_class_atom_prototypes(samples=train, evidence=train_evidence)
    if prototypes.labels != labels:
        raise ValueError("prototype label drift")

    validation_original = validation_embeddings @ train_embeddings.T
    validation_composed = validation_captions @ train_embeddings.T
    rices_features = single_channel_atom_features(
        validation_original, train_y, validation_evidence, prototypes, len(labels))
    circle_features, original_best, composed_best = legacy.circle_atom_features(
        validation_original, validation_composed, train_y, validation_evidence,
        prototypes, len(labels))
    rices_oof, rices_alpha, rices_weights, rices_fit, rices_history = fit_class_equal(
        rices_features, validation_y, validation)
    circle_oof, circle_alpha, circle_weights, circle_fit, circle_history = fit_class_equal(
        circle_features, validation_y, validation)

    rices_pairs = topk(validation_original, 2).tolist()
    rices_atom_pairs = pairs_within_classes(validation_original, train_y, rices_oof)
    circles_pairs = list(zip(legacy.last_argmax(validation_original),
                             legacy.last_argmax(validation_composed)))
    circles_atom_pairs = [(int(original_best[i, cls]), int(composed_best[i, cls]))
                          for i, cls in enumerate(circle_oof)]
    old_calibration_path = PRIOR_FIVE / dataset / "calibration.json"
    old_plan_path = PRIOR_FIVE / dataset / "test_plan.json"
    old_freeze_path = PRIOR_FIVE / dataset / "plan.freeze.json"
    old_complete_path = PRIOR_FIVE / dataset / "complete.json"
    old_calibration = read(old_calibration_path)
    old_plan = read(old_plan_path)
    if old_plan["query_ids"] != test_ids:
        raise ValueError("reused five-method query order drift")
    calibration = {
        "dataset": dataset,
        "prompt_version": PROMPT_VERSION,
        "test_used": False,
        "methods": {
            METHODS[3]: {
                "alpha": rices_alpha, "weights": rices_weights.tolist(),
                "outer_folds": rices_history, "full_fit_class_accuracy": rices_fit,
                "validation_parent_demo_precision": pair_precision(
                    rices_pairs, validation_y, train_y),
                "validation_atom_demo_precision": pair_precision(
                    rices_atom_pairs, validation_y, train_y),
                "independently_fit": True,
                "retrieval_encoder": "official CLIP ViT-g",
            },
            METHODS[4]: {
                **old_calibration["methods"]["muier_atom_class_equal_raw"],
                "reused_exact_retrieval_contract": True,
                "source_calibration_sha256": file_sha256(old_calibration_path),
                "target_prompt_change_does_not_change_retrieval_or_calibration": True,
            },
            METHODS[6]: {
                "alpha": circle_alpha, "weights": circle_weights.tolist(),
                "outer_folds": circle_history, "full_fit_class_accuracy": circle_fit,
                "validation_parent_demo_precision": pair_precision(
                    circles_pairs, validation_y, train_y),
                "validation_atom_demo_precision": pair_precision(
                    circles_atom_pairs, validation_y, train_y),
                "independently_fit": True,
                "retrieval_encoder": "official CLIP ViT-g",
            },
        },
    }
    write(out / "calibration.json", calibration)

    test_embeddings = legacy.load_embeddings(out / "clip.test.npz", test_ids)
    test_captions = legacy.load_embeddings(out / "clip.captions.test.npz", test_ids)
    descriptions = {row["query_id"]: row for row in (
        json.loads(line) for line in (
            out / "counterfactual_descriptions.test.jsonl"
        ).read_text(encoding="utf-8").splitlines() if line.strip())}
    test_evidence = common.evidence(dataset, "test")
    rows = []
    for begin in range(0, len(test), 128):
        held()
        stop = min(begin + 128, len(test))
        original_scores = test_embeddings[begin:stop] @ train_embeddings.T
        composed_scores = test_captions[begin:stop] @ train_embeddings.T
        block_evidence = common.cub.shared.sliced_evidence(test_evidence, begin, stop)
        rices_block_features = single_channel_atom_features(
            original_scores, train_y, block_evidence, prototypes, len(labels))
        circle_block_features, original_best, composed_best = legacy.circle_atom_features(
            original_scores, composed_scores, train_y, block_evidence,
            prototypes, len(labels))
        rices_atom_classes = score_classes(
            rices_block_features, rices_weights).argmax(1)
        circle_atom_classes = score_classes(
            circle_block_features, circle_weights).argmax(1)
        rices = topk(original_scores, 2)
        rices_atom = pairs_within_classes(
            original_scores, train_y, rices_atom_classes)
        original = legacy.last_argmax(original_scores)
        composed = legacy.last_argmax(composed_scores)
        for local, query_index in enumerate(range(begin, stop)):
            query_id = test_ids[query_index]
            circle_class = int(circle_atom_classes[local])
            description = descriptions[query_id]
            old_methods = old_plan["methods"]
            rows.append({
                "query_id": query_id,
                "methods": {
                    METHODS[0]: {"demo_ids": old_methods["random"][query_index]},
                    METHODS[1]: {"demo_ids": [train_ids[x] for x in rices[local]]},
                    METHODS[2]: {"demo_ids": old_methods["muier"][query_index]},
                    METHODS[3]: {"demo_ids": [train_ids[x] for x in rices_atom[local]],
                                 "predicted_class": labels[int(rices_atom_classes[local])]},
                    METHODS[4]: {"demo_ids": old_methods[
                        "muier_atom_class_equal_raw"][query_index]},
                    METHODS[5]: {"demo_ids": [train_ids[int(original[local])],
                                                       train_ids[int(composed[local])]]},
                    METHODS[6]: {"demo_ids": [
                        train_ids[int(original_best[local, circle_class])],
                        train_ids[int(composed_best[local, circle_class])]],
                        "predicted_class": labels[circle_class]},
                },
                "attribute": description["attribute"],
                "modified_caption": description["modified_caption"],
                "test_label_used": False,
            })
        emit("selection_running", dataset=dataset, done=stop, total=len(test))
    plan_path = out / "test_plan.jsonl"
    with plan_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    inputs = [
        Path(__file__), RUN_DOC, legacy.OFFICIAL_SOURCE,
        source(dataset).manifest("train"), source(dataset).manifest("validation"),
        source(dataset).manifest("test"), out / "counterfactual_descriptions.validation.jsonl",
        out / "counterfactual_descriptions.test.jsonl", out / "clip.train.npz",
        out / "clip.validation.npz", out / "clip.test.npz",
        out / "clip.captions.validation.npz", out / "clip.captions.test.npz",
        out / "calibration.json", out / "reuse_receipt.json",
        out / "validation_prompt_smoke.json", old_plan_path, old_calibration_path,
        old_freeze_path, old_complete_path,
    ]
    write(out / "plan.freeze.json", {
        "files": {str(path.resolve()): file_sha256(path) for path in inputs},
        "plan_sha256": file_sha256(plan_path),
        "protocol": {
            "prompt_version": PROMPT_VERSION, "methods": list(METHODS),
            "budget": "2 total per method; CIRCLES=1 IR+1 CIR; others=2 retrieved demos",
            "classification_output": "official direct label generation and exact match",
            "official_clip_methods": [METHODS[1], METHODS[3], METHODS[5], METHODS[6]],
            "reused_unchanged_retrieval_plans": [METHODS[0], METHODS[2], METHODS[4]],
            "candidate_pool": "complete frozen train split",
            "test_used_for_tuning": False,
        }})
    emit("selection_complete", dataset=dataset,
         validation_demo_precision={
             METHODS[1]: pair_precision(rices_pairs, validation_y, train_y),
             METHODS[3]: pair_precision(rices_atom_pairs, validation_y, train_y),
             METHODS[5]: pair_precision(circles_pairs, validation_y, train_y),
             METHODS[6]: pair_precision(circles_atom_pairs, validation_y, train_y),
         })


def official_content(question, query, demos, labels, method, attribute="", caption=""):
    prefix = ("Your task is to perform Image Classification. "
              f"You need to choose one of the following options: {list(labels)}")
    content = [
        {"type": "text", "text": prefix},
        {"type": "image", "image": query},
        {"type": "text", "text": f"Question: {question}"},
    ]
    if method in BASELINE_METHODS:
        content.append({"type": "text", "text":
                        f"Here are {len(demos)} in-context examples to help you answer the question:"})
        for image, label in demos:
            content.extend([
                {"type": "image", "image": image},
                {"type": "text", "text": f"Question: {question}\nAnswer: {label}"},
            ])
    else:
        content.extend([
            {"type": "text", "text": "Here are some in-context examples to help you answer the question."},
            {"type": "text", "text": "Examples retrieved based on the original image:"},
            {"type": "image", "image": demos[0][0]},
            {"type": "text", "text": f"Question: {question}\nAnswer: {demos[0][1]}"},
            {"type": "text", "text":
                f'Examples retrieved based on the target image description after changing "{attribute}" (caption: {caption}):'},
            {"type": "image", "image": demos[1][0]},
            {"type": "text", "text": f"Question: {question}\nAnswer: {demos[1][1]}"},
        ])
    content.extend([
        {"type": "text", "text": "Here is the original question again."},
        {"type": "image", "image": query},
        {"type": "text", "text": f"Question: {question}"},
        {"type": "text", "text": "Please provide your response by directly outputting the answer."},
    ])
    return [{"role": "user", "content": content}]


class OfficialBackend:
    def __init__(self):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(0.88, 0)
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            common.MODEL, local_files_only=True, **common.PROCESSOR_SETTINGS)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            common.MODEL, local_files_only=True, device_map={"": "cuda:0"},
            dtype=torch.bfloat16, attn_implementation="sdpa").eval()

    def forward(self, record, question, labels):
        opened = []
        try:
            demos = []
            for path, label in zip(record["demo_images"], record["demo_labels"]):
                with Image.open(path) as src:
                    image = src.convert("RGB")
                opened.append(image)
                demos.append((image, label))
            with Image.open(record["query_image"]) as src:
                query = src.convert("RGB")
            opened.append(query)
            conversation = official_content(
                question, query, demos, labels, record["method"],
                record.get("attribute", ""), record.get("modified_caption", ""))
            inputs = self.processor.apply_chat_template(
                [conversation], add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt").to(
                    self.model.device, dtype=self.torch.bfloat16)
            width = int(inputs["input_ids"].shape[1])
            started = time.perf_counter()
            with self.torch.inference_mode():
                output = self.model.generate(
                    **inputs, max_new_tokens=512, do_sample=False, use_cache=True)
            self.torch.cuda.synchronize()
            raw = self.processor.batch_decode(
                output[:, width:], skip_special_tokens=True,
                clean_up_tokenization_spaces=False)[0]
            normalized = raw.strip().lower()
            label_lookup = {label.lower(): label for label in labels}
            prediction = label_lookup.get(normalized, "__invalid_exact_output__")
            return {
                "prediction": prediction, "raw_output": raw,
                "stats": {
                    "seconds": time.perf_counter() - started,
                    "input_tokens": int(inputs["attention_mask"].sum()),
                    "images": len(record["demo_images"]) + 2,
                    "exact_label_output": prediction != "__invalid_exact_output__",
                }}
        finally:
            for image in opened:
                image.close()


def verify_plan(dataset):
    out = OUT / dataset
    freeze = read(out / "plan.freeze.json")
    for path, expected in freeze["files"].items():
        if file_sha256(path) != expected:
            raise ValueError(f"frozen input drift: {path}")
    if file_sha256(out / "test_plan.jsonl") != freeze["plan_sha256"]:
        raise ValueError("test plan drift")
    return freeze


def prompt_preflight(dataset, backend, limit=3):
    """Exercise every prompt family on validation examples only."""
    held()
    out = OUT / dataset
    train_rows = samples(dataset, "train")
    validation_rows = samples(dataset, "validation")
    train_ids = [x.sample_id for x in train_rows]
    validation_ids = [x.sample_id for x in validation_rows]
    train = {x.sample_id: x for x in train_rows}
    labels = tuple(sorted({x.label for x in train_rows}))
    train_embeddings = legacy.load_embeddings(out / "clip.train.npz", train_ids)
    validation_embeddings = legacy.load_embeddings(
        out / "clip.validation.npz", validation_ids)
    validation_captions = legacy.load_embeddings(
        out / "clip.captions.validation.npz", validation_ids)
    descriptions = {row["query_id"]: row for row in (
        json.loads(line) for line in (
            out / "counterfactual_descriptions.validation.jsonl"
        ).read_text(encoding="utf-8").splitlines() if line.strip())}
    smoke_methods = (METHODS[0], METHODS[5], METHODS[6])
    records = []
    for index in range(min(limit, len(validation_rows), len(smoke_methods))):
        query = validation_rows[index]
        method = smoke_methods[index]
        original_scores = validation_embeddings[index] @ train_embeddings.T
        composed_scores = validation_captions[index] @ train_embeddings.T
        if method in BASELINE_METHODS:
            demo_indices = np.argsort(original_scores)[-2:][::-1].tolist()
        else:
            demo_indices = [int(np.argmax(original_scores)),
                            int(np.argmax(composed_scores))]
        demo_ids = [train_ids[x] for x in demo_indices]
        description = descriptions[query.sample_id]
        records.append({
            "method": method,
            "query_id": query.sample_id,
            "query_image": query.image_path,
            "demo_ids": demo_ids,
            "demo_images": [train[x].image_path for x in demo_ids],
            "demo_labels": [train[x].label for x in demo_ids],
            "attribute": description["attribute"],
            "modified_caption": description["modified_caption"],
        })
    backend.torch.cuda.reset_peak_memory_stats()
    outputs = [backend.forward(record, source(dataset).instruction, labels)
               for record in records]
    write(out / "validation_prompt_smoke.json", {
        "validation_only": True,
        "test_manifest_loaded": False,
        "prompt_version": PROMPT_VERSION,
        "records": records,
        "outputs": outputs,
        "peak_allocated_mib": backend.torch.cuda.max_memory_allocated() / 2**20,
    })
    emit("validation_prompt_smoke_complete", dataset=dataset, outputs=outputs)


def migrate_prior_official_cache(dataset, ph, records, cache_path, cache):
    """Migrate only byte-contract-equivalent official-prompt outputs."""
    prior_out = PRIOR_OFFICIAL / dataset
    prior_cache = prior_out / "context_cache.jsonl"
    if not prior_cache.is_file():
        return 0
    freeze = read(prior_out / "plan.freeze.json")
    if file_sha256(prior_out / "test_plan.jsonl") != freeze["plan_sha256"]:
        raise ValueError("prior official prompt plan drift")
    snapshot = PRIOR_OFFICIAL / "runner.source.before_seven_method_fix.py"
    source_entries = [value for path, value in freeze["files"].items()
                      if path.endswith("run_qwen3_circles_official_prompt_2shot_four.py")]
    if len(source_entries) != 1 or file_sha256(snapshot) != source_entries[0]:
        raise ValueError("prior official prompt source snapshot drift")
    prior_protocol = read(prior_out / "inference_protocol.json")
    if prior_protocol["prompt_version"] != "official-circles-classification-2shot-1ir-1cir-v1":
        raise ValueError("unexpected prior prompt version")
    prior_plan = {row["query_id"]: row for row in (
        json.loads(line) for line in (prior_out / "test_plan.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip())}

    def signature(family, query_id, demo_ids, attribute="", caption=""):
        return (family, query_id, tuple(demo_ids),
                attribute if family == "circles" else "",
                caption if family == "circles" else "")

    wanted = {signature(
        "baseline" if record["method"] in BASELINE_METHODS else "circles",
        record["query_id"], record["demo_ids"], record["attribute"],
        record["modified_caption"]): record for record in records.values()}
    migrated = []
    for line in prior_cache.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        old = json.loads(line)
        if old["protocol_hash"] != prior_protocol["contract_sha256"]:
            raise ValueError("foreign row in prior official cache")
        family = "baseline" if old["method_family"] == "rices" else "circles"
        plan_row = prior_plan[old["query_id"]]
        sig = signature(family, old["query_id"], old["demo_ids"],
                        plan_row["attribute"], plan_row["modified_caption"])
        record = wanted.get(sig)
        if record is None or record["key"] in cache:
            continue
        row = {
            "key": record["key"], "protocol_hash": ph,
            "method_family": family, "query_id": old["query_id"],
            "demo_ids": old["demo_ids"], "prediction": old["prediction"],
            "raw_output": old["raw_output"], "stats": old["stats"],
            "migrated_from": str(prior_cache.resolve()),
            "source_key": old["key"],
        }
        migrated.append(row)
        cache[row["key"]] = row
    if migrated:
        common.append_rows(cache_path, migrated)
    write(cache_path.parent / "cache_migration_receipt.json", {
        "source_cache": str(prior_cache.resolve()),
        "source_cache_sha256": file_sha256(prior_cache),
        "source_snapshot": str(snapshot.resolve()),
        "source_snapshot_sha256": file_sha256(snapshot),
        "migrated_contexts": len(migrated),
        "exact_match_fields": ["prompt_family", "query_id", "ordered_demo_ids",
                               "attribute_for_circles", "caption_for_circles"],
    })
    return len(migrated)


def infer_dataset(dataset, backend):
    held()
    out = OUT / dataset
    freeze = verify_plan(dataset)
    train_rows = samples(dataset, "train")
    query_rows = samples(dataset, "test")
    plan = [json.loads(x) for x in (out / "test_plan.jsonl").read_text(
        encoding="utf-8").splitlines() if x.strip()]
    query_lookup = {x.sample_id: x for x in query_rows}
    train = {x.sample_id: x for x in train_rows}
    labels = tuple(sorted({x.label for x in train_rows}))
    if [x["query_id"] for x in plan] != [x.sample_id for x in query_rows]:
        raise ValueError("query order drift")
    protocol = {
        "model": "Qwen/Qwen3-VL-4B-Instruct", "dtype": "bfloat16",
        "attention": "sdpa", "processor_settings": common.PROCESSOR_SETTINGS,
        "execution_batch_size": 1, "max_new_tokens": 512,
        "temperature": 0.0, "prompt_version": PROMPT_VERSION,
        "classification": "direct label generation; strip+lower exact match",
        "labels": list(labels), "methods": list(METHODS),
        "plan_sha256": freeze["plan_sha256"],
    }
    ph = stable_hash(protocol)
    protocol["contract_sha256"] = ph
    write(out / "inference_protocol.json", protocol)
    cache_path = out / "context_cache.jsonl"
    cache = {}
    if cache_path.is_file():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row["protocol_hash"] != ph:
                    raise ValueError("foreign context cache")
                cache[row["key"]] = row
    keys = {method: {} for method in METHODS}
    records = {}
    for row in plan:
        query = query_lookup[row["query_id"]]
        for method in METHODS:
            demo_ids = row["methods"][method]["demo_ids"]
            family = "baseline" if method in BASELINE_METHODS else "circles"
            prompt_context = ([row["attribute"], row["modified_caption"]]
                              if family == "circles" else [])
            key = stable_hash([ph, family, query.sample_id, demo_ids, prompt_context,
                               query.image_sha256,
                               [train[x].image_sha256 for x in demo_ids]])
            keys[method][query.sample_id] = key
            if key not in cache and key not in records:
                records[key] = dict(key=key, method=method,
                    query_id=query.sample_id, query_image=query.image_path,
                    demo_ids=demo_ids,
                    demo_images=[train[x].image_path for x in demo_ids],
                    demo_labels=[train[x].label for x in demo_ids],
                    attribute=row["attribute"], modified_caption=row["modified_caption"])
    migrated = migrate_prior_official_cache(dataset, ph, records, cache_path, cache)
    records = [record for key, record in records.items() if key not in cache]
    emit("target_prepared", dataset=dataset, new_contexts=len(records),
         cached_contexts=len(cache), migrated_contexts=migrated,
         requested=len(plan) * len(METHODS),
         unique_contexts=len(set(key for mapping in keys.values() for key in mapping.values())))
    started = time.perf_counter()
    for index, record in enumerate(records, 1):
        held()
        output = backend.forward(record, source(dataset).instruction, labels)
        row = {"key": record["key"], "protocol_hash": ph,
               "method_family": "baseline" if record["method"] in BASELINE_METHODS else "circles",
               "query_id": record["query_id"], "demo_ids": record["demo_ids"], **output}
        common.append_rows(cache_path, [row])
        cache[row["key"]] = row
        if index % 25 == 0 or index == len(records):
            rate = index / max(time.perf_counter() - started, 1e-9) * 60
            emit("target_running", dataset=dataset, done=index, new_total=len(records),
                 cached_total=len(cache), batch_size=1, rate_per_minute=rate,
                 eta_minutes=(len(records) - index) / max(rate, 1e-9))
    predictions = {method: {
        query_id: cache[key]["prediction"] for query_id, key in mapping.items()}
        for method, mapping in keys.items()}
    raw_outputs = {method: {
        query_id: cache[key]["raw_output"] for query_id, key in mapping.items()}
        for method, mapping in keys.items()}
    truth = {x.sample_id: x.label for x in query_rows}
    plan_by_id = {x["query_id"]: x for x in plan}
    results = {}
    for method in METHODS:
        item = {
            "metrics": accuracy_metrics(
                [truth[x] for x in predictions[method]],
                [predictions[method][x] for x in predictions[method]]),
            "invalid_exact_output_count": sum(
                value == "__invalid_exact_output__" for value in predictions[method].values()),
            "demo_same_class_precision_at_2": float(np.mean([
                sum(train[demo].label == truth[qid]
                    for demo in plan_by_id[qid]["methods"][method]["demo_ids"]) / 2
                for qid in truth])),
        }
        if method in ATOM_PARENTS:
            item["paired_vs_parent"] = common.paired(
                predictions[ATOM_PARENTS[method]], predictions[method], truth)
        if method in CIRCLES_METHODS:
            item["paired_vs_matched_rices"] = common.paired(
                predictions[METHODS[1]], predictions[method], truth)
        results[method] = item
    result = {"status": "complete", "dataset": dataset, "methods": results,
              "shots": 2, "prompt_version": PROMPT_VERSION,
              "test_used_for_tuning": False,
              "inference": {"new_contexts": len(records), "total_contexts": len(cache),
                            "elapsed_seconds": time.perf_counter() - started,
                            "final_batch_size": 1}}
    write(out / "predictions.json", predictions)
    write(out / "raw_outputs.json", raw_outputs)
    write(out / "results.json", result)
    files = [out / name for name in ("plan.freeze.json", "inference_protocol.json",
             "context_cache.jsonl", "predictions.json", "raw_outputs.json", "results.json")]
    write(out / "complete.json", {
        "files": {str(path.resolve()): file_sha256(path) for path in files}})
    emit("dataset_complete", dataset=dataset, result=result)


def report():
    lines = ["# Qwen3-VL-4B official-prompt 七方法 2-shot 四数据集", "",
             "Random/RICES/MUIER及其Atom分支使用官方普通检索提示；CIRCLES为1 IR+1 CIR并使用官方分组、属性、caption与重复query。全部直接生成类别名并严格exact-match。", "",
             "| Dataset | Method | Acc. | Macro | Demo P@2 | Invalid output |",
             "|---|---|---:|---:|---:|---:|"]
    for dataset in ORDER:
        result = read(OUT / dataset / "results.json")
        for method in METHODS:
            item = result["methods"][method]
            lines.append(f"| {dataset} | {method} | {item['metrics']['accuracy']*100:.3f}% | "
                         f"{item['metrics']['macro_accuracy']*100:.3f}% | "
                         f"{item['demo_same_class_precision_at_2']*100:.3f}% | "
                         f"{item['invalid_exact_output_count']} |")
    path = OUT / "report_zh.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write(OUT / "complete.json", {"files": {
        str(path.resolve()): file_sha256(path),
        **{str((OUT / d / "complete.json").resolve()): file_sha256(OUT / d / "complete.json")
           for d in ORDER}}})
    emit("queue_complete", datasets=list(ORDER), report=str(path.resolve()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "select", "preflight", "infer", "all", "report"))
    parser.add_argument("--datasets", nargs="+", choices=ORDER, default=list(ORDER))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "report":
        report()
        return
    for dataset in args.datasets:
        if args.stage in ("prepare", "preflight", "all"):
            prepare_dataset(dataset)
        if args.stage in ("preflight", "all"):
            backend = OfficialBackend()
            try:
                prompt_preflight(dataset, backend, limit=3)
            finally:
                del backend; gc.collect()
                import torch; torch.cuda.empty_cache()
        if args.stage in ("select", "infer", "all"):
            select_dataset(dataset)
        if args.stage == "preflight":
            continue
        if args.stage in ("infer", "all"):
            backend = OfficialBackend()
            try:
                infer_dataset(dataset, backend)
            finally:
                del backend; gc.collect()
                import torch; torch.cuda.empty_cache()
    if args.stage == "all" and tuple(args.datasets) == ORDER:
        report()


if __name__ == "__main__":
    main()

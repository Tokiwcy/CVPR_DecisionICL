"""Independent replay and raw-output audit of the no-factor class-gate ablation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))
import run_qwen3_decision_generic_20260920 as base
from audit_qwen3_visual_centroid_class_gate_2shot_20261004 import (
    paired, read, same_numbers, sha, verify_receipt, reference_dir, resource_dir,
)
from dgdr.hashing import stable_hash

PROJECT = SCRIPT.parents[1]
QUEUE = PROJECT / "artifacts/priority_1shot_ablation_20260920"
ROOT = QUEUE / "qwen3_parent_class_gate_2shot_20261004"
VISUAL = QUEUE / "qwen3_visual_centroid_class_gate_2shot_20261004"
VISUAL_CUB = QUEUE / "qwen3_visual_centroid_class_gate_2shot_resume_20261004"
METHOD = "rices+parent_top1_class_gate_no_factors"
DATASETS = ("dtd", "aircraft", "cub")


def source_dir(dataset: str) -> Path:
    return (VISUAL_CUB if dataset == "cub" else VISUAL) / "2shot" / dataset


def audit_dataset(dataset: str) -> dict:
    out = ROOT / "2shot" / dataset
    verify_receipt(ROOT / "run.freeze.json")
    verify_receipt(out / "plan.freeze.json")
    verify_receipt(out / "complete.json")
    verify_receipt(reference_dir(dataset) / "complete.json")
    verify_receipt(source_dir(dataset) / "complete.json")
    diagnostic = read(out / "validation_diagnostic.json")
    plan = read(out / "test_plan.json")
    train, _ = base.common.load(dataset, "train")
    val, _ = base.common.load(dataset, "validation")
    labels, train_y, val_y = base.common.labels_and_y(train, val)
    train_ids = [row.sample_id for row in train]
    resource = resource_dir(dataset)
    train_clip = base.r.legacy.load_embeddings(resource / "clip.train.npz", train_ids)
    val_clip = base.r.legacy.load_embeddings(
        resource / "clip.validation.npz", [row.sample_id for row in val])

    def independent_select(scores: np.ndarray) -> list[list[int]]:
        # Mirror the frozen official np.argsort-last/reverse tie convention,
        # but do not call the new runner or its parent_gate helper.
        first = np.argsort(scores, axis=1)[:, -1]
        selected = []
        for row, index in enumerate(first):
            candidates = np.flatnonzero(train_y == train_y[index])
            order = np.argsort(scores[row, candidates])[-2:][::-1]
            selected.append(candidates[order].tolist())
        return selected

    val_picks = np.asarray(independent_select(val_clip @ train_clip.T))
    val_accuracy = float(np.mean(train_y[val_picks[:, 0]] == val_y))
    val_precision = float(np.mean(train_y[val_picks] == val_y[:, None]))
    if diagnostic["test_used"] or not diagnostic["no_learned_parameters"] or \
            abs(val_accuracy - diagnostic["validation_class_accuracy"]) > 1e-9 or \
            abs(val_precision - diagnostic["validation_two_demo_same_class_precision"]) > 1e-9:
        raise ValueError("parameter-free validation diagnostic drift")

    query, _ = base.common.load(dataset, "test")
    query_ids = [row.sample_id for row in query]
    query_clip = base.r.legacy.load_embeddings(resource / "clip.test.npz", query_ids)
    if [row["query_id"] for row in plan] != query_ids:
        raise ValueError("test query coverage/order drift")
    replay = []
    for start in range(0, len(query), 64):
        scores = query_clip[start:start + 64] @ train_clip.T
        replay.extend([[train_ids[i] for i in picks]
                       for picks in independent_select(scores)])
    if any(row["demo_ids"] != expected or row.get("test_label_used") is not False
           for row, expected in zip(plan, replay)):
        raise ValueError("test method-query/demo mapping drift")

    from benchmark_qwen3_target_batches import r as target
    contract = stable_hash(dict(
        model="Qwen/Qwen3-VL-4B-Instruct", dtype="bfloat16", attention="sdpa",
        processor_settings=base.common.PROCESSOR_SETTINGS, batch_size=2,
        do_sample=False, max_new_tokens=512,
        baseline_prompt_source=sha(Path(target.official.__file__)),
        executor_source=sha(SCRIPT.parent / "benchmark_qwen3_target_batches.py")))
    train_by_id = {row.sample_id: row for row in train}
    query_by_id = {row.sample_id: row for row in query}
    keys = []
    for row in plan:
        item = query_by_id[row["query_id"]]
        demos = [train_by_id[key] for key in row["demo_ids"]]
        keys.append(stable_hash([
            contract, (item.sample_id, tuple(row["demo_ids"]), "baseline", "", ""),
            item.image_sha256, [demo.image_sha256 for demo in demos]]))
    raw = sorted((out / "raw_batches").glob("*.json"))
    if len(raw) != (len(keys) + 1) // 2:
        raise ValueError("raw fixed-batch coverage drift")
    reuse = read(out / "reuse_manifest.json")
    if reuse["source_complete_sha256"] != sha(source_dir(dataset) / "complete.json"):
        raise ValueError("reuse source complete SHA drift")
    copied = reuse["batches"]
    predicted = {}
    lookup = {label.lower(): label for label in labels}
    reused_contexts = 0
    for index, path in enumerate(raw):
        if path.name != f"{index:06d}.json":
            raise ValueError("raw fixed-batch index drift")
        batch = read(path)
        expected_keys = keys[2 * index:2 * index + 2]
        if batch.get("contract") != contract or batch.get("keys") != expected_keys or \
                batch.get("signature") != stable_hash(expected_keys) or \
                batch.get("execution_batch_size") != len(expected_keys) or \
                len(batch.get("outputs", [])) != len(expected_keys):
            raise ValueError(f"raw fixed-batch partner/contract drift: {path}")
        if path.name in copied:
            provenance = copied[path.name]
            source = Path(provenance["source"])
            if source != source_dir(dataset) / "raw_batches" / path.name or \
                    sha(source) != provenance["sha256"] or \
                    sha(path) != provenance["sha256"]:
                raise ValueError(f"copied raw provenance drift: {path}")
            reused_contexts += len(expected_keys)
        for local, value in enumerate(batch["outputs"]):
            answer = value["raw_output"].strip().lower()
            expected = lookup.get(answer, "__invalid_exact_output__")
            if value["prediction"] != expected or \
                    value["stats"]["batch_row"] != local or \
                    value["stats"]["execution_batch_size"] != len(expected_keys):
                raise ValueError(f"strict raw parser drift: {path}")
            predicted[plan[2 * index + local]["query_id"]] = expected
    if predicted != read(out / "predictions.json")[METHOD]:
        raise ValueError("raw-to-prediction mapping drift")
    truth = {row.sample_id: row.label for row in query}
    correct = np.asarray([predicted[key] == truth[key] for key in query_ids], bool)
    accuracy = float(correct.mean())
    macro = float(np.mean([correct[np.asarray([truth[key] == label
                        for key in query_ids])].mean() for label in labels]))
    invalid = sum(value == "__invalid_exact_output__" for value in predicted.values())
    result = read(out / "results.json")
    if abs(result["metrics"]["accuracy"] - accuracy) > 1e-9 or \
            abs(result["metrics"]["macro_accuracy"] - macro) > 1e-9 or \
            result["invalid_exact_output_count"] != invalid or \
            result["new_target_contexts"] != len(keys) - reused_contexts:
        raise ValueError("accuracy/macro/invalid/context accounting drift")
    main = read(reference_dir(dataset) / "predictions.json")
    for name, reference, key in (
        ("parent", main["rices_clip_official_prompt"], "paired_vs_parent"),
        ("decision", main["rices_clip_atom_class_equal_raw_official_prompt"],
         "paired_vs_decision"),
    ):
        same_numbers(paired(reference, predicted, truth), result[key], name)
    audit = dict(status="passed", dataset=dataset, shots=2, method=METHOD,
                 queries=len(query), batches=len(raw), reused_batches=len(copied),
                 new_batches=len(raw) - len(copied),
                 complete_sha256=sha(out / "complete.json"),
                 plan_sha256=sha(out / "test_plan.json"),
                 results_sha256=sha(out / "results.json"),
                 validation_and_plan_replayed=True,
                 raw_partner_parser_and_provenance_replayed=True,
                 accuracy=accuracy, macro_accuracy=macro, invalid=invalid,
                 paired_recomputed=True, test_used_for_tuning=False)
    (out / "independent_audit.json").write_text(
        json.dumps(audit, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS)
    args = parser.parse_args()
    for dataset in ((args.dataset,) if args.dataset else DATASETS):
        print(json.dumps(audit_dataset(dataset), ensure_ascii=True))

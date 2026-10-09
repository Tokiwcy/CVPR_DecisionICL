"""Independent replay of the fixed Qwen3 class-prototype rotation control."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

SCRIPT = Path(__file__).resolve()
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))

import run_qwen3_decision_generic_20260920 as base
from dgdr.atom_label import ClassAtomPrototypes, build_class_atom_prototypes, score_classes
from dgdr.hashing import stable_hash

PROJECT = SCRIPT.parents[1]
QUEUE = PROJECT / "artifacts/priority_1shot_ablation_20260920"
ROOT = QUEUE / "qwen3_rotated_factor_class_2shot_20261003"
MAIN = PROJECT / "artifacts/qwen3_five_dataset_recalibrated_20260910"
GENERIC = QUEUE / "qwen3_generic_direct_1_2shot_20260921"
METHOD = "rices+rotated_decision_class_prototype"
DATASETS = ("dtd", "aircraft", "cub")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_receipt(path: Path) -> dict:
    receipt = read(path)
    for name, expected in receipt["files"].items():
        file = Path(name)
        if not file.is_file() or sha(file) != expected:
            raise ValueError(f"receipt dependency drift: {file}")
    return receipt


def paired(left: dict, right: dict, truth: dict) -> dict:
    delta = np.asarray([int(right[key] == truth[key]) - int(left[key] == truth[key])
                        for key in sorted(truth)], dtype=np.int8)
    rescue, harm = int(np.count_nonzero(delta == 1)), int(np.count_nonzero(delta == -1))
    rng = np.random.default_rng(42)
    boot = np.empty(10000, dtype=np.float32)
    for start in range(0, 10000, 250):
        index = rng.integers(0, len(delta), size=(250, len(delta)))
        boot[start:start + 250] = delta[index].mean(axis=1)
    return dict(rescue=rescue, harm=harm, accuracy_delta=float(delta.mean()),
                paired_bootstrap_95pct=np.quantile(boot, (0.025, 0.975)).tolist(),
                exact_mcnemar_p=float(binomtest(rescue, rescue + harm, 0.5).pvalue)
                if rescue + harm else 1.0)


def same_numbers(actual: dict, expected: dict, name: str) -> None:
    for key in ("rescue", "harm", "accuracy_delta", "exact_mcnemar_p"):
        if abs(actual[key] - expected[key]) > 1e-9:
            raise ValueError(f"paired {name}/{key} drift: {actual[key]} != {expected[key]}")
    if not np.allclose(actual["paired_bootstrap_95pct"],
                       expected["paired_bootstrap_95pct"], atol=1e-9):
        raise ValueError(f"paired {name}/CI drift")


def reference_dir(dataset: str) -> Path:
    return Path(read(MAIN / "execution_locations.json")[f"{dataset}/2"])


def resource_dir(dataset: str) -> Path:
    return reference_dir(dataset).parents[1] / "resources" / dataset


def audit_dataset(dataset: str) -> dict:
    out = ROOT / "2shot" / dataset
    verify_receipt(ROOT / "run.freeze.json")
    verify_receipt(out / "plan.freeze.json")
    complete = verify_receipt(out / "complete.json")
    verify_receipt(reference_dir(dataset) / "complete.json")
    verify_receipt(GENERIC / "2shot" / dataset / "complete.json")
    calibration = read(out / "calibration.json")
    plan = read(out / "test_plan.json")
    train, _ = base.common.load(dataset, "train")
    validation, _ = base.common.load(dataset, "validation")
    train_evidence = base.common.evidence(dataset, "train")
    validation_evidence = base.common.evidence(dataset, "validation")
    original = build_class_atom_prototypes(samples=train, evidence=train_evidence)
    labels, train_y, validation_y = base.common.labels_and_y(train, validation)
    if original.labels != tuple(labels):
        raise ValueError("prototype/label order drift")
    rotated = ClassAtomPrototypes(
        original.labels, original.group_ids,
        tuple(np.roll(group, 1, axis=0).copy() for group in original.probabilities))
    rotated.validate()
    mapping = {label: labels[(index - 1) % len(labels)]
               for index, label in enumerate(labels)}
    if calibration["prototype_class_source"] != mapping:
        raise ValueError("prototype rotation mapping drift")
    train_ids = [sample.sample_id for sample in train]
    validation_ids = [sample.sample_id for sample in validation]
    resource = resource_dir(dataset)
    train_clip = base.r.legacy.load_embeddings(resource / "clip.train.npz", train_ids)
    validation_clip = base.r.legacy.load_embeddings(resource / "clip.validation.npz", validation_ids)
    validation_parent = validation_clip @ train_clip.T
    features = base.r.official.single_channel_atom_features(
        validation_parent, train_y, validation_evidence, rotated, len(labels))
    oof, alpha, weights, fit_score, folds = base.r.official.fit_class_equal(
        features, validation_y, validation)
    if abs(alpha - calibration["alpha"]) > 1e-9 or not np.allclose(
            weights, calibration["weights"], atol=1e-9):
        raise ValueError("independent validation calibration drift")
    if abs(float(np.mean(oof == validation_y)) - calibration["oof_class_accuracy"]) > 1e-9:
        raise ValueError("out-of-fold calibration drift")
    if abs(fit_score - calibration["full_fit_class_accuracy"]) > 1e-9:
        raise ValueError("full-fit calibration drift")
    if len(folds) != 5 or calibration["test_used"]:
        raise ValueError("validation fold/test-use contract drift")

    query, _ = base.common.load(dataset, "test")
    query_evidence = base.common.evidence(dataset, "test")
    query_ids = [sample.sample_id for sample in query]
    query_clip = base.r.legacy.load_embeddings(resource / "clip.test.npz", query_ids)
    if [row["query_id"] for row in plan] != query_ids:
        raise ValueError("query plan coverage/order drift")
    replay = []
    for start in range(0, len(query), 64):
        stop = min(start + 64, len(query))
        parent = query_clip[start:stop] @ train_clip.T
        block = base.common.cub.shared.sliced_evidence(query_evidence, start, stop)
        features = base.r.official.single_channel_atom_features(
            parent, train_y, block, rotated, len(labels))
        classes = score_classes(features, weights).argmax(1)
        selected = base.r.within(parent, train_y, classes, 2)
        replay.extend([[train_ids[int(index)] for index in row] for row in selected])
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
    train_by_id = {sample.sample_id: sample for sample in train}
    query_by_id = {sample.sample_id: sample for sample in query}
    keys = []
    for row in plan:
        sample = query_by_id[row["query_id"]]
        demos = [train_by_id[sample_id] for sample_id in row["demo_ids"]]
        keys.append(stable_hash([
            contract, (sample.sample_id, tuple(row["demo_ids"]), "baseline", "", ""),
            sample.image_sha256, [demo.image_sha256 for demo in demos]]))
    raw = sorted((out / "raw_batches").glob("*.json"))
    if len(raw) != (len(keys) + 1) // 2:
        raise ValueError("raw fixed-batch coverage drift")
    predicted = {}
    lookup = {label.lower(): label for label in labels}
    for index, path in enumerate(raw):
        if path.name != f"{index:06d}.json":
            raise ValueError("raw fixed-batch index drift")
        batch = read(path)
        expected_keys = keys[index * 2:index * 2 + 2]
        if batch["contract"] != contract or batch["keys"] != expected_keys or \
                batch["signature"] != stable_hash(expected_keys) or \
                batch["execution_batch_size"] != len(expected_keys) or \
                len(batch["outputs"]) != len(expected_keys):
            raise ValueError(f"raw fixed-batch partner/contract drift: {path}")
        for local, value in enumerate(batch["outputs"]):
            raw_answer = value["raw_output"].strip().lower()
            expected_prediction = lookup.get(raw_answer, "__invalid_exact_output__")
            if value["prediction"] != expected_prediction or \
                    value["stats"]["batch_row"] != local or \
                    value["stats"]["execution_batch_size"] != len(expected_keys):
                raise ValueError(f"raw output/strict parse drift: {path}")
            predicted[plan[index * 2 + local]["query_id"]] = expected_prediction
    saved = read(out / "predictions.json")[METHOD]
    if predicted != saved:
        raise ValueError("raw-to-prediction mapping drift")
    truth = {sample.sample_id: sample.label for sample in query}
    correct = np.asarray([predicted[key] == truth[key] for key in query_ids], bool)
    accuracy = float(correct.mean())
    macro = float(np.mean([correct[np.asarray([truth[key] == label
                        for key in query_ids])].mean() for label in labels]))
    invalid = sum(value == "__invalid_exact_output__" for value in predicted.values())
    result = read(out / "results.json")
    if abs(result["metrics"]["accuracy"] - accuracy) > 1e-9 or \
            abs(result["metrics"]["macro_accuracy"] - macro) > 1e-9 or \
            result["invalid_exact_output_count"] != invalid:
        raise ValueError("accuracy/macro/invalid drift")
    main = read(reference_dir(dataset) / "predictions.json")
    generic = read(GENERIC / "2shot" / dataset / "predictions.json")
    for name, reference, key in (
        ("parent", main["rices_clip_official_prompt"], "paired_vs_parent"),
        ("matched", main["rices_clip_atom_class_equal_raw_official_prompt"],
         "paired_vs_matched_decision"),
        ("generic", generic["rices+generic_visual_direct_sample"],
         "paired_vs_generic"),
    ):
        same_numbers(paired(reference, predicted, truth), result[key], name)
    audit = dict(status="passed", dataset=dataset, shots=2,
                 queries=len(query), batches=len(raw), method=METHOD,
                 complete_sha256=sha(out / "complete.json"),
                 plan_sha256=sha(out / "test_plan.json"),
                 results_sha256=sha(out / "results.json"),
                 calibration_replayed=True, plan_replayed=True,
                 raw_partner_and_parser_replayed=True,
                 accuracy=accuracy, macro_accuracy=macro, invalid=invalid,
                 paired_recomputed=True, test_used_for_tuning=False)
    path = out / "independent_audit.json"
    path.write_text(json.dumps(audit, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS)
    args = parser.parse_args()
    for dataset in ((args.dataset,) if args.dataset else DATASETS):
        result = audit_dataset(dataset)
        print(json.dumps(result, ensure_ascii=True))

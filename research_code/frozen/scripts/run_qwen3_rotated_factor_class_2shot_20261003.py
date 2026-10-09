"""Fixed class-prototype rotation control for Qwen3 2-shot RICES."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve()
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))

import run_qwen3_generic_direct_1_2shot_20260921 as generic
from dgdr.atom_label import ClassAtomPrototypes, build_class_atom_prototypes, score_classes
from dgdr.io import load_atom_bank

base = generic.base
target_batch = generic.target_batch
PROJECT = SCRIPT.parents[1]
QUEUE = PROJECT / "artifacts/priority_1shot_ablation_20260920"
ROOT = QUEUE / "qwen3_rotated_factor_class_2shot_20261003"
DOC = PROJECT / "docs/qwen3_rotated_factor_class_2shot_20261003_zh.md"
DATASETS = ("dtd", "aircraft", "cub")
METHOD = "rices+rotated_decision_class_prototype"
PARENT = "rices_clip_official_prompt"
DECISION = "rices_clip_atom_class_equal_raw_official_prompt"
GENERIC = "rices+generic_visual_direct_sample"
SHOTS = 2


def emit(stage: str, **details) -> None:
    value = dict(stage=stage, time=time.time(), pid=os.getpid(), **details)
    base.write(ROOT / "status.json", value)
    print(json.dumps(value, ensure_ascii=True), flush=True)


def held() -> None:
    for root in (ROOT, QUEUE):
        if any((root / name).exists() for name in
               ("PAUSED_BY_USER.json", "DIAGNOSTIC_HOLD.json", "WATCHDOG_STOP")):
            raise RuntimeError(f"ablation held at {root}")


def output_dir(dataset: str) -> Path:
    return ROOT / "2shot" / dataset


def main_reference(dataset: str) -> Path:
    return generic.reference_dir(dataset, SHOTS)


def generic_reference(dataset: str) -> Path:
    return generic.ROOT / "2shot" / dataset


def rotated_prototypes(train, evidence) -> tuple[ClassAtomPrototypes, dict]:
    original = build_class_atom_prototypes(samples=train, evidence=evidence)
    labels = original.labels
    if len(labels) < 2:
        raise ValueError("class rotation requires at least two classes")
    rotated = ClassAtomPrototypes(
        labels=labels,
        group_ids=original.group_ids,
        probabilities=tuple(np.roll(values, 1, axis=0).copy()
                            for values in original.probabilities),
    )
    rotated.validate()
    mapping = {label: labels[(index - 1) % len(labels)]
               for index, label in enumerate(labels)}
    if any(label == source for label, source in mapping.items()):
        raise ValueError("prototype rotation left a class unchanged")
    return rotated, mapping


def preflight() -> None:
    held()
    if base.verified(ROOT / "run.freeze.json"):
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    files = [SCRIPT, DOC, Path(base.r.__file__), Path(base.r.official.__file__),
             Path(target_batch.__file__), Path(base.r.common.__file__)]
    counts = {}
    for dataset in DATASETS:
        held()
        ref = main_reference(dataset)
        generic_ref = generic_reference(dataset)
        if not base.verified(ref / "complete.json"):
            raise ValueError(f"missing audited main result: {dataset}")
        if not base.verified(generic_ref / "complete.json"):
            raise ValueError(f"missing audited Generic result: {dataset}")
        source = base.common.source(dataset)
        bank = load_atom_bank(source.bank)
        files.extend([ref / "complete.json", generic_ref / "complete.json", source.bank])
        counts[dataset] = {}
        for split in ("train", "validation", "test"):
            samples = base.manifest(dataset, split)
            evidence = base.common.evidence(dataset, split)
            if evidence.sample_ids != tuple(sample.sample_id for sample in samples):
                raise ValueError(f"sample/evidence order drift: {dataset}/{split}")
            if evidence.bank_hash != bank.bank_hash:
                raise ValueError(f"bank/evidence drift: {dataset}/{split}")
            counts[dataset][split] = len(samples)
            prefix = source.evidence_path(split)
            files.extend([source.manifest(split), Path(str(prefix) + ".json"),
                          Path(str(prefix) + ".npz")])
        resource = generic.resource_dir(dataset)
        for split in ("train", "validation", "test"):
            files.append(resource / f"clip.{split}.npz")
    base.write(ROOT / "cpu_preflight.json", dict(
        status="passed", model="Qwen3-VL-4B-Instruct BF16", shots=2,
        datasets=list(DATASETS), counts=counts, target_batch=2,
        only_change="rotate class rows of train-derived factor prototypes by one",
        matched_bank_and_evidence_reused=True, test_used_for_tuning=False,
        independent_validation_calibration=True))
    base.freeze(ROOT / "run.freeze.json", files + [ROOT / "cpu_preflight.json"])
    emit("cpu_preflight_passed", datasets=list(DATASETS))


def fit_and_plan(dataset: str) -> None:
    out = output_dir(dataset)
    if base.verified(out / "plan.freeze.json"):
        return
    held()
    out.mkdir(parents=True, exist_ok=True)
    train, _ = base.common.load(dataset, "train")
    validation, _ = base.common.load(dataset, "validation")
    train_evidence = base.common.evidence(dataset, "train")
    validation_evidence = base.common.evidence(dataset, "validation")
    prototypes, mapping = rotated_prototypes(train, train_evidence)
    labels, train_y, validation_y = base.common.labels_and_y(train, validation)
    if tuple(labels) != prototypes.labels:
        raise ValueError("label/prototype order drift")
    train_ids = [sample.sample_id for sample in train]
    validation_ids = [sample.sample_id for sample in validation]
    resource = generic.resource_dir(dataset)
    train_clip = base.r.legacy.load_embeddings(resource / "clip.train.npz", train_ids)
    validation_clip = base.r.legacy.load_embeddings(resource / "clip.validation.npz", validation_ids)
    validation_parent = validation_clip @ train_clip.T
    validation_features = base.r.official.single_channel_atom_features(
        validation_parent, train_y, validation_evidence, prototypes, len(labels))
    oof, alpha, weights, fit_score, folds = base.r.official.fit_class_equal(
        validation_features, validation_y, validation)
    base.write(out / "calibration.json", dict(
        method=METHOD, dataset=dataset, shots=SHOTS,
        alpha=alpha, weights=weights.tolist(), outer_folds=folds,
        full_fit_class_accuracy=fit_score,
        oof_class_accuracy=float(np.mean(oof == validation_y)),
        independently_fit=True, inherited_weights=False,
        objective="validation class accuracy", folds_count=5, fold_seed=42,
        alpha_grid=list(base.common.ALPHAS),
        prototype_class_source=mapping, prototype_rotation=1,
        test_used=False))
    emit("independent_calibration", dataset=dataset, alpha=alpha,
         oof=float(np.mean(oof == validation_y)))

    # The test manifest is first parsed after the complete validation fit is persisted.
    query, _ = base.common.load(dataset, "test")
    query_evidence = base.common.evidence(dataset, "test")
    query_ids = [sample.sample_id for sample in query]
    query_clip = base.r.legacy.load_embeddings(resource / "clip.test.npz", query_ids)
    plan = []
    for start in range(0, len(query), 64):
        held()
        stop = min(start + 64, len(query))
        parent = query_clip[start:stop] @ train_clip.T
        block = base.common.cub.shared.sliced_evidence(query_evidence, start, stop)
        features = base.r.official.single_channel_atom_features(
            parent, train_y, block, prototypes, len(labels))
        classes = score_classes(features, np.asarray(weights, np.float32)).argmax(1)
        selected = base.r.within(parent, train_y, classes, SHOTS)
        for local, sample in enumerate(query[start:stop]):
            plan.append(dict(query_id=sample.sample_id,
                             demo_ids=[train_ids[int(index)] for index in selected[local]],
                             test_label_used=False))
    if [row["query_id"] for row in plan] != query_ids:
        raise ValueError("test plan query coverage/order drift")
    base.write(out / "test_plan.json", plan)
    base.freeze(out / "plan.freeze.json", [ROOT / "run.freeze.json",
                out / "calibration.json", out / "test_plan.json"])
    emit("test_plan_frozen", dataset=dataset, queries=len(plan), alpha=alpha)


def fixed_contract() -> str:
    return generic.stable(dict(
        model="Qwen/Qwen3-VL-4B-Instruct", dtype="bfloat16", attention="sdpa",
        processor_settings=base.common.PROCESSOR_SETTINGS, batch_size=2,
        do_sample=False, max_new_tokens=512,
        baseline_prompt_source=generic.sha(target_batch.r.official.__file__),
        executor_source=generic.sha(target_batch.__file__)))


def infer(dataset: str, backend) -> None:
    out = output_dir(dataset)
    if base.verified(out / "complete.json"):
        return
    if not base.verified(out / "plan.freeze.json"):
        raise ValueError("unverified plan")
    plan = base.read(out / "test_plan.json")
    train = {sample.sample_id: sample for sample in base.manifest(dataset, "train")}
    query = {sample.sample_id: sample for sample in base.manifest(dataset, "test")}
    labels = tuple(sorted({sample.label for sample in train.values()}))
    contract = fixed_contract()
    keys, records = [], {}
    for row in plan:
        item = query[row["query_id"]]
        demos = [train[sample_id] for sample_id in row["demo_ids"]]
        record = generic.record_for(item, demos, "baseline", row)
        key = generic.stable([contract, generic.context_tuple(record, "baseline"),
                              item.image_sha256,
                              [demo.image_sha256 for demo in demos]])
        keys.append(key)
        if key in records:
            raise ValueError("duplicate target context")
        records[key] = record
    batches = out / "raw_batches"
    batches.mkdir(parents=True, exist_ok=True)
    started = time.time()
    new = 0
    cached = 0
    outputs = {}
    for start in range(0, len(keys), 2):
        held()
        pair = keys[start:start + 2]
        signature = generic.stable(pair)
        path = batches / f"{start // 2:06d}.json"
        if path.exists():
            value = base.read(path)
            if value.get("contract") != contract or value.get("keys") != pair or value.get("signature") != signature:
                raise ValueError(f"foreign fixed-batch cache: {path}")
            cached += len(pair)
        else:
            forward = backend.forward_many([records[key] for key in pair],
                                           base.common.source(dataset).instruction, labels)
            if len(forward) != len(pair):
                raise ValueError("fixed batch output cardinality drift")
            value = dict(contract=contract, keys=pair, signature=signature,
                         outputs=forward, execution_batch_size=len(pair),
                         context_source="new_fixed_batch2")
            base.write(path, value)
            new += len(pair)
        if len(value["outputs"]) != len(pair):
            raise ValueError(f"incomplete fixed batch: {path}")
        for key, result in zip(pair, value["outputs"]):
            outputs[key] = result
        if (start // 2) % 100 == 0:
            rate = new / max((time.time() - started) / 60.0, 1e-9)
            emit("target_inference", dataset=dataset, done=len(outputs),
                 total=len(keys), cached=cached, new=new,
                 contexts_per_minute=rate,
                 eta_minutes=(len(keys) - len(outputs)) / max(rate, 1e-9))
    if len(outputs) != len(keys):
        raise ValueError("target context coverage drift")
    truth = {sample_id: sample.label for sample_id, sample in query.items()}
    prediction = {row["query_id"]: outputs[key]["prediction"]
                  for row, key in zip(plan, keys)}
    if set(prediction) != set(truth):
        raise ValueError("prediction query coverage drift")
    main = base.read(main_reference(dataset) / "predictions.json")
    other = base.read(generic_reference(dataset) / "predictions.json")
    result = dict(
        metrics=base.r.accuracy_metrics(
            [truth[sample_id] for sample_id in query],
            [prediction[sample_id] for sample_id in query]),
        invalid_exact_output_count=sum(value == "__invalid_exact_output__"
                                       for value in prediction.values()),
        paired_vs_parent=base.paired(main[PARENT], prediction, truth),
        paired_vs_matched_decision=base.paired(main[DECISION], prediction, truth),
        paired_vs_generic=base.paired(other[GENERIC], prediction, truth),
        model="Qwen3-VL-4B-Instruct BF16", dataset=dataset, shots=SHOTS,
        method=METHOD, new_target_contexts=new, cached_target_contexts=cached,
        test_used_for_tuning=False)
    base.write(out / "predictions.json", {METHOD: prediction})
    base.write(out / "results.json", result)
    raw = sorted(batches.glob("*.json"))
    if len(raw) != (len(keys) + 1) // 2:
        raise ValueError("raw batch coverage drift")
    base.freeze(out / "complete.json", [out / "plan.freeze.json",
                out / "predictions.json", out / "results.json",
                main_reference(dataset) / "complete.json",
                generic_reference(dataset) / "complete.json", *raw])
    emit("dataset_complete_unverified", dataset=dataset,
         accuracy=result["metrics"]["accuracy"], new=new, cached=cached)


def run(stage: str) -> None:
    preflight()
    for dataset in DATASETS:
        fit_and_plan(dataset)
    if stage == "cpu-check":
        emit("cpu_plans_complete", datasets=list(DATASETS))
        return
    import msvcrt
    locks = []
    for path in (generic.MAIN / "gpu_queue.lock", QUEUE / "gpu_queue.lock"):
        handle = path.open("a+b")
        handle.seek(0)
        handle.write(b"0")
        handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        locks.append(handle)
    backend = None
    try:
        for dataset in DATASETS:
            held()
            if base.verified(output_dir(dataset) / "complete.json"):
                continue
            if backend is None:
                emit("loading_target_model", dataset=dataset)
                backend = target_batch.BatchedBackend()
            infer(dataset, backend)
        emit("target_blocks_complete_pending_independent_audit",
             datasets=list(DATASETS))
    finally:
        if backend is not None:
            del backend
        gc.collect()
        for handle in locks:
            handle.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("cpu-check", "all"))
    arguments = parser.parse_args()
    run(arguments.stage)


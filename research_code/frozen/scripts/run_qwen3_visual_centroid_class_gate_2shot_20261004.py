"""Frozen visual-centroid class-gating control for Qwen3 2-shot RICES."""
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
from dgdr.atom_label import score_classes

base = generic.base
target_batch = generic.target_batch
PROJECT = SCRIPT.parents[1]
QUEUE = PROJECT / "artifacts/priority_1shot_ablation_20260920"
ROOT = QUEUE / "qwen3_visual_centroid_class_gate_2shot_20261004"
DOC = PROJECT / "docs/qwen3_visual_centroid_class_gate_2shot_20261004_zh.md"
AUDIT = SCRIPT.parent / "audit_qwen3_visual_centroid_class_gate_2shot_20261004.py"
DATASETS = ("dtd", "aircraft", "cub")
METHOD = "rices+clip_centroid_class_gate"
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



def visual_centroids(train_clip: np.ndarray, train_y: np.ndarray, class_count: int) -> np.ndarray:
    centroids = np.empty((class_count, train_clip.shape[1]), np.float32)
    for class_index in range(class_count):
        members = train_clip[train_y == class_index]
        if len(members) == 0:
            raise ValueError('empty train class')
        vector = members.mean(axis=0, dtype=np.float64)
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError('invalid visual class centroid')
        centroids[class_index] = (vector / norm).astype(np.float32)
    return centroids


def visual_features(parent: np.ndarray, train_y: np.ndarray,
                    query_clip: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    class_count = len(centroids)
    parent_class = np.empty((len(parent), class_count), np.float32)
    for class_index in range(class_count):
        candidates = np.flatnonzero(train_y == class_index)
        parent_class[:, class_index] = parent[:, candidates].max(axis=1)
    visual_class = query_clip @ centroids.T
    raw = np.stack((parent_class, visual_class), axis=2)
    standardized, _, _ = base.common.cub.shared.standardize_across_classes(raw)
    return standardized


def preflight() -> None:
    held()
    if base.verified(ROOT / 'run.freeze.json'):
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    files = [SCRIPT, DOC, AUDIT, Path(base.r.__file__), Path(base.r.official.__file__),
             Path(target_batch.__file__), Path(base.r.common.__file__)]
    counts = {}
    for dataset in DATASETS:
        held()
        ref = main_reference(dataset)
        generic_ref = generic_reference(dataset)
        if not base.verified(ref / 'complete.json'):
            raise ValueError(f'missing audited main result: {dataset}')
        if not base.verified(generic_ref / 'complete.json'):
            raise ValueError(f'missing audited Generic result: {dataset}')
        main_audit = base.read(ref / 'independent_audit.json')
        if main_audit.get('status') != 'passed':
            raise ValueError(f'main audit not passed: {dataset}')
        files.extend([ref / 'complete.json', ref / 'independent_audit.json',
                      generic_ref / 'complete.json'])
        counts[dataset] = {}
        for split in ('train', 'validation', 'test'):
            samples = base.manifest(dataset, split)
            counts[dataset][split] = len(samples)
            files.append(base.common.source(dataset).manifest(split))
        resource = generic.resource_dir(dataset)
        for split in ('train', 'validation', 'test'):
            clip = resource / f'clip.{split}.npz'
            files.extend([clip, Path(str(clip) + '.json')])
    base.write(ROOT / 'cpu_preflight.json', dict(
        status='passed', model='Qwen3-VL-4B-Instruct BF16', shots=2,
        datasets=list(DATASETS), counts=counts, target_batch=2,
        only_change='CLIP train-class centroid in place of target-model factor evidence',
        same_parent_and_class_gate=True, same_full_train_pool=True,
        test_used_for_tuning=False, independent_validation_calibration=True))
    base.freeze(ROOT / 'run.freeze.json', files + [ROOT / 'cpu_preflight.json'])
    emit('cpu_preflight_passed', datasets=list(DATASETS))


def fit_and_plan(dataset: str) -> None:
    out = output_dir(dataset)
    if base.verified(out / 'plan.freeze.json'):
        return
    held()
    out.mkdir(parents=True, exist_ok=True)
    train, _ = base.common.load(dataset, 'train')
    validation, _ = base.common.load(dataset, 'validation')
    labels, train_y, validation_y = base.common.labels_and_y(train, validation)
    train_ids = [sample.sample_id for sample in train]
    validation_ids = [sample.sample_id for sample in validation]
    resource = generic.resource_dir(dataset)
    train_clip = base.r.legacy.load_embeddings(resource / 'clip.train.npz', train_ids)
    validation_clip = base.r.legacy.load_embeddings(resource / 'clip.validation.npz', validation_ids)
    centroids = visual_centroids(train_clip, train_y, len(labels))
    validation_parent = validation_clip @ train_clip.T
    validation_features = visual_features(validation_parent, train_y, validation_clip, centroids)
    oof, alpha, weights, fit_score, folds = base.r.official.fit_class_equal(
        validation_features, validation_y, validation)
    base.write(out / 'calibration.json', dict(
        method=METHOD, dataset=dataset, shots=SHOTS,
        alpha=alpha, weights=weights.tolist(), outer_folds=folds,
        full_fit_class_accuracy=fit_score,
        oof_class_accuracy=float(np.mean(oof == validation_y)),
        independently_fit=True, inherited_weights=False,
        objective='validation class accuracy', folds_count=5, fold_seed=42,
        alpha_grid=list(base.common.ALPHAS),
        class_visual_score='cosine to l2-normalized mean of frozen CLIP train embeddings',
        parent_class_score='maximum RICES CLIP cosine within each train class',
        prototype_train_only=True, test_used=False))
    emit('independent_calibration', dataset=dataset, alpha=alpha,
         oof=float(np.mean(oof == validation_y)))

    # Read test only after validation fit is persisted.
    query, _ = base.common.load(dataset, 'test')
    query_ids = [sample.sample_id for sample in query]
    query_clip = base.r.legacy.load_embeddings(resource / 'clip.test.npz', query_ids)
    plan = []
    for start in range(0, len(query), 64):
        held()
        stop = min(start + 64, len(query))
        block = query_clip[start:stop]
        parent = block @ train_clip.T
        features = visual_features(parent, train_y, block, centroids)
        classes = score_classes(features, np.asarray(weights, np.float32)).argmax(1)
        selected = base.r.within(parent, train_y, classes, SHOTS)
        for local, sample in enumerate(query[start:stop]):
            plan.append(dict(query_id=sample.sample_id,
                             demo_ids=[train_ids[int(index)] for index in selected[local]],
                             test_label_used=False))
    if [row['query_id'] for row in plan] != query_ids:
        raise ValueError('test plan query coverage/order drift')
    base.write(out / 'test_plan.json', plan)
    base.freeze(out / 'plan.freeze.json', [ROOT / 'run.freeze.json',
                out / 'calibration.json', out / 'test_plan.json'])
    emit('test_plan_frozen', dataset=dataset, queries=len(plan), alpha=alpha)

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


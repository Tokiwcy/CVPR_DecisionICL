"""Frozen no-factor, parent-selected class gate for full Qwen3 2-shot tests."""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))
import run_qwen3_visual_centroid_class_gate_2shot_20261004 as visual

base, generic, target_batch = visual.base, visual.generic, visual.target_batch
PROJECT = SCRIPT.parents[1]
QUEUE = PROJECT / "artifacts/priority_1shot_ablation_20260920"
ROOT = QUEUE / "qwen3_parent_class_gate_2shot_20261004"
DOC = PROJECT / "docs/qwen3_parent_class_gate_2shot_20261004_zh.md"
AUDIT = SCRIPT.parent / "audit_qwen3_parent_class_gate_2shot_20261004.py"
DATASETS = ("dtd", "aircraft", "cub")
METHOD = "rices+parent_top1_class_gate_no_factors"
VISUAL_ROOTS = {
    "dtd": visual.ROOT,
    "aircraft": visual.ROOT,
    "cub": QUEUE / "qwen3_visual_centroid_class_gate_2shot_resume_20261004",
}


def emit(stage: str, **details) -> None:
    value = dict(stage=stage, time=time.time(), pid=os.getpid(), **details)
    base.write(ROOT / "status.json", value)
    print(json.dumps(value, ensure_ascii=True), flush=True)


def held() -> None:
    for root in (ROOT, QUEUE):
        if any((root / name).exists() for name in
               ("PAUSED_BY_USER.json", "DIAGNOSTIC_HOLD.json", "WATCHDOG_STOP")):
            raise RuntimeError(f"ablation held at {root}")


def outdir(dataset: str) -> Path:
    return ROOT / "2shot" / dataset


def visual_dir(dataset: str) -> Path:
    return VISUAL_ROOTS[dataset] / "2shot" / dataset


def parent_gate(scores: np.ndarray, train_y: np.ndarray) -> list[list[int]]:
    # The top RICES item fixes the class; official stable top-k ranks its full pool.
    first = base.r.official.topk(scores, 1)[:, 0]
    return base.r.within(scores, train_y, train_y[first], 2)


def preflight() -> None:
    held()
    if (ROOT / "run.freeze.json").exists():
        if not base.verified(ROOT / "run.freeze.json"):
            raise ValueError("frozen run drift")
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    files = [SCRIPT, DOC, AUDIT, Path(base.r.__file__),
             Path(base.r.official.__file__), Path(target_batch.__file__),
             Path(base.r.common.__file__), Path(visual.SCRIPT)]
    counts = {}
    for dataset in DATASETS:
        ref = visual.main_reference(dataset)
        source = visual_dir(dataset)
        for path in (ref / "complete.json", ref / "independent_audit.json",
                     source / "complete.json", source / "independent_audit.json"):
            if not path.is_file():
                raise ValueError(f"missing accepted reference: {path}")
            files.append(path)
        if not base.verified(ref / "complete.json") or not base.verified(source / "complete.json"):
            raise ValueError(f"reference hash closure failed: {dataset}")
        if base.read(ref / "independent_audit.json").get("status") != "passed" or \
                base.read(source / "independent_audit.json").get("status") != "passed":
            raise ValueError(f"reference independent audit failed: {dataset}")
        counts[dataset] = {}
        for split in ("train", "validation", "test"):
            counts[dataset][split] = len(base.manifest(dataset, split))
            files.append(base.common.source(dataset).manifest(split))
            clip = generic.resource_dir(dataset) / f"clip.{split}.npz"
            files.extend((clip, Path(str(clip) + ".json")))
    base.write(ROOT / "cpu_preflight.json", dict(
        status="passed", model="Qwen3-VL-4B-Instruct BF16", shots=2,
        datasets=list(DATASETS), counts=counts, target_batch=2,
        change="remove factor evidence; parent top-1 class then within-class RICES top-2",
        no_learned_parameters=True, full_train_pool=True, test_used_for_tuning=False))
    base.freeze(ROOT / "run.freeze.json", files + [ROOT / "cpu_preflight.json"])
    emit("cpu_preflight_passed", datasets=list(DATASETS))


def fit_and_plan(dataset: str) -> None:
    out = outdir(dataset)
    if (out / "plan.freeze.json").exists():
        if not base.verified(out / "plan.freeze.json"):
            raise ValueError("frozen plan drift")
        return
    held()
    out.mkdir(parents=True, exist_ok=True)
    train, _ = base.common.load(dataset, "train")
    validation, _ = base.common.load(dataset, "validation")
    labels, train_y, validation_y = base.common.labels_and_y(train, validation)
    train_ids = [row.sample_id for row in train]
    resource = generic.resource_dir(dataset)
    train_clip = base.r.legacy.load_embeddings(resource / "clip.train.npz", train_ids)
    validation_clip = base.r.legacy.load_embeddings(
        resource / "clip.validation.npz", [row.sample_id for row in validation])
    validation_scores = validation_clip @ train_clip.T
    selected = parent_gate(validation_scores, train_y)
    predicted_class = train_y[np.asarray(selected)[:, 0]]
    base.write(out / "validation_diagnostic.json", dict(
        dataset=dataset, method=METHOD, no_learned_parameters=True,
        validation_class_accuracy=float(np.mean(predicted_class == validation_y)),
        validation_two_demo_same_class_precision=float(np.mean(
            train_y[np.asarray(selected)] == validation_y[:, None])),
        train_count=len(train), validation_count=len(validation),
        class_count=len(labels), test_used=False))
    emit("validation_diagnostic_frozen", dataset=dataset)

    # Test is loaded only after the parameter-free validation diagnostic exists.
    query, _ = base.common.load(dataset, "test")
    query_clip = base.r.legacy.load_embeddings(
        resource / "clip.test.npz", [row.sample_id for row in query])
    plan = []
    for start in range(0, len(query), 64):
        held()
        scores = query_clip[start:start + 64] @ train_clip.T
        selected = parent_gate(scores, train_y)
        for item, picks in zip(query[start:start + 64], selected):
            plan.append(dict(query_id=item.sample_id,
                             demo_ids=[train_ids[int(i)] for i in picks],
                             test_label_used=False))
    if len(plan) != len(query) or len(set(row["query_id"] for row in plan)) != len(query):
        raise ValueError("test plan coverage drift")
    base.write(out / "test_plan.json", plan)
    base.freeze(out / "plan.freeze.json", [ROOT / "run.freeze.json",
                out / "validation_diagnostic.json", out / "test_plan.json"])
    emit("test_plan_frozen", dataset=dataset, queries=len(plan))


def context(dataset: str):
    out = outdir(dataset)
    plan = base.read(out / "test_plan.json")
    train = {row.sample_id: row for row in base.manifest(dataset, "train")}
    query = {row.sample_id: row for row in base.manifest(dataset, "test")}
    labels = tuple(sorted({row.label for row in train.values()}))
    contract = visual.fixed_contract()
    keys, records = [], {}
    for row in plan:
        item = query[row["query_id"]]
        demos = [train[key] for key in row["demo_ids"]]
        record = generic.record_for(item, demos, "baseline", row)
        key = generic.stable([contract, generic.context_tuple(record, "baseline"),
                              item.image_sha256, [demo.image_sha256 for demo in demos]])
        if key in records:
            raise ValueError("duplicate query context")
        records[key] = record
        keys.append(key)
    return plan, train, query, labels, contract, keys, records


def seed_exact_batches(dataset: str) -> dict:
    out = outdir(dataset)
    if not base.verified(out / "plan.freeze.json"):
        raise ValueError("unverified plan")
    _, _, _, _, contract, keys, _ = context(dataset)
    source = visual_dir(dataset)
    if not base.verified(source / "complete.json"):
        raise ValueError("visual control source drift")
    rawdir = out / "raw_batches"
    rawdir.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for index in range((len(keys) + 1) // 2):
        held()
        pair = keys[2 * index:2 * index + 2]
        target = rawdir / f"{index:06d}.json"
        if target.exists():
            continue
        candidate = source / "raw_batches" / target.name
        value = base.read(candidate)
        if value.get("contract") == contract and value.get("keys") == pair and \
                value.get("signature") == generic.stable(pair) and \
                value.get("execution_batch_size") == len(pair) and \
                len(value.get("outputs", [])) == len(pair):
            shutil.copyfile(candidate, target)
            if base.sha(target) != base.sha(candidate):
                raise ValueError("copied raw batch SHA drift")
            provenance[target.name] = dict(source=str(candidate), sha256=base.sha(candidate))
    # Include copies created on a prior resumed CPU preflight.
    prior = out / "reuse_manifest.json"
    if prior.exists():
        provenance.update(base.read(prior)["batches"])
    base.write(prior, dict(method=METHOD, dataset=dataset,
                           source_complete_sha256=base.sha(source / "complete.json"),
                           batches=dict(sorted(provenance.items()))))
    result = dict(dataset=dataset, queries=len(keys), batches=(len(keys) + 1) // 2,
                  reused_batches=len(provenance),
                  new_batches=(len(keys) + 1) // 2 - len(provenance))
    base.write(out / "batch_preflight.json", result)
    emit("exact_batch_preflight", **result)
    return result


def infer(dataset: str, backend) -> None:
    out = outdir(dataset)
    if base.verified(out / "complete.json"):
        return
    plan, train, query, labels, contract, keys, records = context(dataset)
    rawdir = out / "raw_batches"
    started = time.time()
    new = 0
    outputs = {}
    reused = 0
    for index in range((len(keys) + 1) // 2):
        held()
        pair = keys[2 * index:2 * index + 2]
        signature = generic.stable(pair)
        path = rawdir / f"{index:06d}.json"
        if path.exists():
            value = base.read(path)
            reused += len(pair)
        else:
            result = backend.forward_many([records[key] for key in pair],
                                          base.common.source(dataset).instruction, labels)
            if len(result) != len(pair):
                raise ValueError("fixed batch output cardinality drift")
            value = dict(contract=contract, keys=pair, signature=signature,
                         outputs=result, execution_batch_size=len(pair),
                         context_source="new_fixed_batch2")
            base.write(path, value)
            new += len(pair)
        if value.get("contract") != contract or value.get("keys") != pair or \
                value.get("signature") != signature or \
                value.get("execution_batch_size") != len(pair) or \
                len(value.get("outputs", [])) != len(pair):
            raise ValueError(f"foreign fixed-batch cache: {path}")
        outputs.update(zip(pair, value["outputs"]))
        if index % 100 == 0:
            rate = new / max((time.time() - started) / 60, 1e-9)
            emit("target_inference", dataset=dataset, done=len(outputs),
                 total=len(keys), cached=reused, new=new,
                 new_contexts_per_minute=rate,
                 eta_minutes=(len(keys) - len(outputs)) / max(rate, 1e-9))
    truth = {key: row.label for key, row in query.items()}
    prediction = {row["query_id"]: outputs[key]["prediction"]
                  for row, key in zip(plan, keys)}
    main = base.read(visual.main_reference(dataset) / "predictions.json")
    result = dict(method=METHOD, dataset=dataset, shots=2,
                  model="Qwen3-VL-4B-Instruct BF16",
                  metrics=base.r.accuracy_metrics(
                      [truth[key] for key in query], [prediction[key] for key in query]),
                  invalid_exact_output_count=sum(
                      value == "__invalid_exact_output__" for value in prediction.values()),
                  paired_vs_parent=base.paired(main[visual.PARENT], prediction, truth),
                  paired_vs_decision=base.paired(main[visual.DECISION], prediction, truth),
                  reused_exact_fixed_batches=len(base.read(out / "reuse_manifest.json")["batches"]),
                  new_target_contexts=len(keys) - sum(len(keys[2 * int(name[:6]):2 * int(name[:6]) + 2]) for name in base.read(out / "reuse_manifest.json")["batches"]),
                  test_used_for_tuning=False)
    base.write(out / "predictions.json", {METHOD: prediction})
    base.write(out / "results.json", result)
    raw = sorted(rawdir.glob("*.json"))
    if len(raw) != (len(keys) + 1) // 2:
        raise ValueError("raw batch coverage drift")
    base.freeze(out / "complete.json", [out / "plan.freeze.json",
                out / "reuse_manifest.json", out / "batch_preflight.json",
                out / "predictions.json", out / "results.json",
                visual.main_reference(dataset) / "complete.json",
                visual_dir(dataset) / "complete.json", *raw])
    emit("dataset_complete_unverified", dataset=dataset,
         accuracy=result["metrics"]["accuracy"], new=new, cached=reused)


def run(stage: str) -> None:
    preflight()
    for dataset in DATASETS:
        fit_and_plan(dataset)
        seed_exact_batches(dataset)
    if stage == "cpu-check":
        emit("cpu_plans_complete", datasets=list(DATASETS))
        return
    import msvcrt
    locks = []
    try:
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
                if base.verified(outdir(dataset) / "complete.json"):
                    continue
                if base.read(outdir(dataset) / "batch_preflight.json")["new_batches"]:
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
    finally:
        for handle in locks:
            handle.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("cpu-check", "all"))
    run(parser.parse_args().stage)

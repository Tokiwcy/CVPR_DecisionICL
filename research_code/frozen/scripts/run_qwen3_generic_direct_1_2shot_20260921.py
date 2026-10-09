"""Qwen3 4B Generic-only direct-sample retrieval for 1/2-shot.

Generic visual evidence is compared with every frozen training image.  No
class prototype, class averaging, class prediction, or within-class second
stage is used.  Existing parent and Decision outputs are references only.
"""
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
SCRIPTS = SCRIPT.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import benchmark_qwen3_target_batches as target_batch
import run_qwen3_decision_generic_20260920 as base
from dgdr.calibration import stratified_folds
from dgdr.evidence import EvidenceStore, group_similarity_matrix
from dgdr.io import load_atom_bank

PROJECT = SCRIPT.parents[1]
QUEUE = PROJECT / "artifacts/priority_1shot_ablation_20260920"
ROOT = QUEUE / "qwen3_generic_direct_1_2shot_20260921"
DOC = PROJECT / "docs/qwen3_generic_direct_1_2shot_20260921_zh.md"
MAIN = PROJECT / "artifacts/qwen3_five_dataset_recalibrated_20260910"
SIX = QUEUE / "qwen3_decision_generic"
GEN1 = QUEUE / "qwen3_generic_only_sparse"
GEN2 = QUEUE / "qwen3_generic_only_sparse_v2"
CORE = QUEUE / "qwen3_core_sources"
ORDER = ("dtd", "aircraft", "cub", "dogs", "flowers102")
ALPHAS = tuple(float(x) for x in base.common.ALPHAS)
METHODS = {
    1: ("rices+generic_visual_direct_sample", "muier+generic_visual_direct_sample"),
    2: ("rices+generic_visual_direct_sample", "muier+generic_visual_direct_sample",
        "circles+generic_visual_direct_sample"),
}
PARENTS = {
    "rices+generic_visual_direct_sample": "rices_clip_official_prompt",
    "muier+generic_visual_direct_sample": "muier_classification_adapted_siglip_official_prompt",
    "circles+generic_visual_direct_sample": "circles_official_prompt",
}
DECISIONS = {
    "rices+generic_visual_direct_sample": "rices_clip_atom_class_equal_raw_official_prompt",
    "muier+generic_visual_direct_sample": "muier_classification_adapted_siglip_atom_class_equal_raw_official_prompt",
    "circles+generic_visual_direct_sample": "circles_atom_class_equal_raw_official_prompt",
}
read, write, sha, stable = base.read, base.write, base.sha, base.stable


def rows(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def emit(stage, **values):
    value = dict(stage=stage, time=time.time(), pid=os.getpid(), **values)
    write(ROOT / "status.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def held():
    for root in (ROOT, QUEUE):
        if any((root / name).exists() for name in
               ("PAUSED_BY_USER.json", "DIAGNOSTIC_HOLD.json", "WATCHDOG_STOP")):
            raise RuntimeError("Qwen3 Generic direct-sample queue held")


def generic_dir(dataset: str) -> Path:
    if dataset == "dtd":
        return CORE / dataset / "generic_visual"
    if dataset == "aircraft":
        return SIX / dataset / "generic_visual"
    if dataset == "cub":
        return GEN1 / dataset / "generic_visual"
    return GEN2 / dataset / "generic_visual"


def reference_dir(dataset: str, shots: int) -> Path:
    return Path(read(MAIN / "execution_locations.json")[f"{dataset}/{shots}"])


def resource_dir(dataset: str) -> Path:
    # The active 1-shot and 2-shot executions share byte-identical resources.
    return reference_dir(dataset, 2).parents[1] / "resources" / dataset


def generic_evidence(dataset: str, split: str) -> EvidenceStore:
    root = generic_dir(dataset)
    store = EvidenceStore.load(root / f"siglip.{split}.evidence")
    expected = tuple(x.sample_id for x in base.manifest(dataset, split))
    if store.sample_ids != expected:
        raise ValueError(f"{dataset}/{split} generic evidence order drift")
    bank = load_atom_bank(root / "atom_bank.final.json")
    if store.bank_hash != bank.bank_hash:
        raise ValueError(f"{dataset}/{split} generic bank/evidence drift")
    return store


def sliced(store: EvidenceStore, start: int, stop: int) -> EvidenceStore:
    return EvidenceStore(
        sample_ids=store.sample_ids[start:stop], group_ids=store.group_ids,
        state_ids=store.state_ids, state_group_index=store.state_group_index,
        raw_logits=store.raw_logits[start:stop],
        state_probabilities=store.state_probabilities[start:stop],
        signed_margins=store.signed_margins[start:stop],
        ordinal_scores=store.ordinal_scores[start:stop],
        applicability_scores=store.applicability_scores[start:stop],
        bank_hash=store.bank_hash, model_id=store.model_id,
        model_revision=store.model_revision, calibration_hash=store.calibration_hash,
        text_template=store.text_template, transform=store.transform)


def scaled_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, np.float32)
    low = values.min(axis=1, keepdims=True)
    high = values.max(axis=1, keepdims=True)
    return (values - low) / np.clip(high - low, 1e-8, None)


def atom_scores(query: EvidenceStore, train: EvidenceStore) -> np.ndarray:
    if query.group_ids != train.group_ids or not query.group_ids:
        raise ValueError("Generic direct evidence group contract drift")
    result = np.zeros((len(query.sample_ids), len(train.sample_ids)), np.float32)
    for group_index in range(len(query.group_ids)):
        result += group_similarity_matrix(
            query, train, group_index, validate_alignment=(group_index == 0))
    return result / float(len(query.group_ids))


def combined(base_scores: np.ndarray, generic_scores: np.ndarray, alpha: float) -> np.ndarray:
    return float(alpha) * scaled_rows(base_scores) + (1.0 - float(alpha)) * generic_scores


def top(scores: np.ndarray, shots: int) -> np.ndarray:
    return np.argsort(-scores, axis=1, kind="stable")[:, :shots]


def quality(picks: np.ndarray, truth: np.ndarray, train_y: np.ndarray) -> float:
    return float(np.mean(train_y[picks] == truth[:, None]))


def fit_one(method: str, shots: int, bases: tuple[np.ndarray, ...], generic: np.ndarray,
            validation_y: np.ndarray, train_y: np.ndarray, validation) -> dict:
    candidates = {}
    for alpha in ALPHAS:
        selected = [top(combined(values, generic, alpha), 1 if len(bases) == 2 else shots)
                    for values in bases]
        candidates[alpha] = (np.concatenate(selected, axis=1)
                             if len(selected) == 2 else selected[0])
    folds = stratified_folds(validation, 5, 42)
    oof = np.empty((len(validation), shots), np.int64)
    history = []
    for fold in range(5):
        fit = np.flatnonzero(folds != fold)
        hold = np.flatnonzero(folds == fold)
        scored = {a: quality(p[fit], validation_y[fit], train_y) for a, p in candidates.items()}
        alpha = max(ALPHAS, key=lambda a: (scored[a], a))
        oof[hold] = candidates[alpha][hold]
        history.append(dict(fold=fold, alpha=alpha, fit_demo_same_class_precision=scored[alpha],
                            fit_count=len(fit), heldout_count=len(hold)))
    scored = {a: quality(p, validation_y, train_y) for a, p in candidates.items()}
    alpha = max(ALPHAS, key=lambda a: (scored[a], a))
    return dict(alpha=alpha, alpha_grid_scores={str(a): scored[a] for a in ALPHAS},
                outer_folds=history, oof_demo_same_class_precision=quality(oof, validation_y, train_y),
                full_fit_demo_same_class_precision=scored[alpha], independently_fit=True,
                inherited_weights=False, objective="validation demo same-class precision",
                direct_candidate_retrieval=True, class_prototype=False,
                group_weights="equal fixed mean", parent_mass_tie_rule="prefer larger alpha")


def validate_inputs():
    files = [SCRIPT, DOC, target_batch.__file__]
    counts = {}
    for dataset in ORDER:
        source = generic_dir(dataset)
        for name in ("atom_bank.final.json", "siglip.train.evidence.json",
                     "siglip.train.evidence.npz", "siglip.validation.evidence.json",
                     "siglip.validation.evidence.npz", "siglip.test.evidence.json",
                     "siglip.test.evidence.npz"):
            files.append(source / name)
        for shots in (1, 2):
            ref = reference_dir(dataset, shots)
            base.verified(ref / "complete.json")
            files.append(ref / "complete.json")
        train = base.manifest(dataset, "train")
        val = base.manifest(dataset, "validation")
        test = base.manifest(dataset, "test")
        tv, vv, qv = (generic_evidence(dataset, split)
                      for split in ("train", "validation", "test"))
        if not (tv.group_ids == vv.group_ids == qv.group_ids):
            raise ValueError(f"{dataset} generic split groups drift")
        res = resource_dir(dataset)
        for name in ("clip.train.npz", "clip.validation.npz", "clip.test.npz",
                     "clip.captions.validation.npz", "clip.captions.test.npz",
                     "counterfactual_descriptions.validation.jsonl",
                     "counterfactual_descriptions.test.jsonl"):
            files.append(res / name)
        counts[dataset] = dict(train=len(train), validation=len(val), test=len(test),
                               groups=len(tv.group_ids), states=len(tv.state_ids))
    if ROOT.exists() and (ROOT / "run.freeze.json").exists():
        base.verified(ROOT / "run.freeze.json")
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    write(ROOT / "cpu_preflight.json", dict(
        status="passed", counts=counts, model="Qwen3-VL-4B-Instruct BF16",
        shots=[1, 2], methods={str(k): list(v) for k, v in METHODS.items()},
        generic_scoring="query-to-every-train-sample mean group TV similarity",
        class_prototype=False, class_average=False, test_used_for_tuning=False,
        calibration="independent dataset x shot x parent five-fold validation"))
    base.freeze(ROOT / "run.freeze.json", files + [ROOT / "cpu_preflight.json"])
    emit("cpu_preflight_passed", datasets=len(ORDER))


def fit_and_plan(dataset: str, shots: int):
    out = ROOT / f"{shots}shot" / dataset
    marker = out / "plan.freeze.json"
    if base.verified(marker):
        return
    out.mkdir(parents=True, exist_ok=True)
    train, train_store = base.common.load(dataset, "train")
    validation, validation_store = base.common.load(dataset, "validation")
    labels, train_y, validation_y = base.common.labels_and_y(train, validation)
    train_ids = [x.sample_id for x in train]
    validation_ids = [x.sample_id for x in validation]
    resource = resource_dir(dataset)
    train_clip = base.r.legacy.load_embeddings(resource / "clip.train.npz", train_ids)
    validation_clip = base.r.legacy.load_embeddings(resource / "clip.validation.npz", validation_ids)
    label_vectors = np.load(base.common.source(dataset).labels).astype(np.float32)
    rices = validation_clip @ train_clip.T
    muier = base.common.parent_scores(validation_store, train_store, train_y, label_vectors)
    train_generic = generic_evidence(dataset, "train")
    validation_generic = generic_evidence(dataset, "validation")
    generic = atom_scores(validation_generic, train_generic)
    base_scores = {
        METHODS[shots][0]: (rices,),
        METHODS[shots][1]: (muier,),
    }
    if shots == 2:
        caption = base.r.legacy.load_embeddings(
            resource / "clip.captions.validation.npz", validation_ids) @ train_clip.T
        base_scores[METHODS[shots][2]] = (rices, caption)
    calibration = {}
    for method, values in base_scores.items():
        held()
        calibration[method] = fit_one(
            method, shots, values, generic, validation_y, train_y, validation)
        emit("independent_calibration", dataset=dataset, shots=shots, method=method,
             alpha=calibration[method]["alpha"],
             oof=calibration[method]["oof_demo_same_class_precision"])
    write(out / "calibration.json", dict(
        dataset=dataset, shots=shots, methods=calibration, folds=5, seed=42,
        test_used=False, inherited_weights=False, class_prototype=False,
        direct_candidate_retrieval=True, factor_aggregation="equal mean over groups"))

    # Test manifests and scores are consumed only after all method parameters for
    # this exact dataset/shot contract have been fitted and persisted.
    query, query_store = base.common.load(dataset, "test")
    query_ids = [x.sample_id for x in query]
    query_clip = base.r.legacy.load_embeddings(resource / "clip.test.npz", query_ids)
    query_generic = generic_evidence(dataset, "test")
    caption_test = None
    descriptions = {}
    if shots == 2:
        caption_test = base.r.legacy.load_embeddings(
            resource / "clip.captions.test.npz", query_ids)
        descriptions = {x["query_id"]: x for x in rows(
            resource / "counterfactual_descriptions.test.jsonl")}
    plan = []
    for start in range(0, len(query), 48):
        held()
        stop = min(start + 48, len(query))
        block = base.common.EmbeddingStore(
            query_store.sample_ids[start:stop], query_store.embeddings[start:stop],
            query_store.model_id, query_store.model_revision)
        rices_block = query_clip[start:stop] @ train_clip.T
        muier_block = base.common.parent_scores(block, train_store, train_y, label_vectors)
        generic_block = atom_scores(sliced(query_generic, start, stop), train_generic)
        selected = {
            METHODS[shots][0]: top(combined(
                rices_block, generic_block, calibration[METHODS[shots][0]]["alpha"]), shots),
            METHODS[shots][1]: top(combined(
                muier_block, generic_block, calibration[METHODS[shots][1]]["alpha"]), shots),
        }
        if shots == 2:
            caption_block = caption_test[start:stop] @ train_clip.T
            alpha = calibration[METHODS[shots][2]]["alpha"]
            selected[METHODS[shots][2]] = np.concatenate((
                top(combined(rices_block, generic_block, alpha), 1),
                top(combined(caption_block, generic_block, alpha), 1)), axis=1)
        for local, item in enumerate(query[start:stop]):
            desc = descriptions.get(item.sample_id, {})
            plan.append(dict(
                query_id=item.sample_id,
                methods={m: [train_ids[int(x)] for x in picks[local]]
                         for m, picks in selected.items()},
                attribute=desc.get("attribute", ""),
                modified_caption=desc.get("modified_caption", ""),
                test_label_used=False))
    write(out / "test_plan.json", plan)
    base.freeze(marker, [ROOT / "run.freeze.json", out / "calibration.json",
                         out / "test_plan.json"])
    emit("test_plan_frozen", dataset=dataset, shots=shots,
         queries=len(plan), methods=len(METHODS[shots]))


def record_for(query, demos, family: str, row: dict) -> dict:
    method = (target_batch.r.METHODS[5] if family == "circles"
              else target_batch.r.METHODS[1])
    return dict(query_id=query.sample_id, query_image=query.image_path,
                demo_ids=[x.sample_id for x in demos],
                demo_images=[x.image_path for x in demos],
                demo_labels=[x.label for x in demos], method=method,
                attribute=row.get("attribute", ""),
                modified_caption=row.get("modified_caption", ""))


def context_tuple(record: dict, family: str):
    extra = ((record["attribute"], record["modified_caption"])
             if family == "circles" else ("", ""))
    return (record["query_id"], tuple(record["demo_ids"]), family, *extra)


def main_reference_lookup(dataset: str, shots: int) -> dict:
    ref = reference_dir(dataset, shots)
    base.verified(ref / "complete.json")
    protocol = read(ref / "inference_protocol.json")
    contract = protocol.get("contract", protocol)
    batch_size = contract.get("batch_size", contract.get("execution_batch_size"))
    if batch_size != 2:
        return {}
    plan_rows = rows(ref / "test_plan.jsonl")
    plan = {x["query_id"]: x for x in plan_rows}
    cache_rows = rows(ref / "context_cache.jsonl")
    lookup, conflicts = {}, set()
    for item in cache_rows:
        if item.get("stats", {}).get("execution_batch_size") != 2:
            continue
        row = plan[item["query_id"]]
        matches = []
        for method, value in row["methods"].items():
            ids = value.get("demo_ids", value) if isinstance(value, dict) else value
            if list(ids) == list(item["demo_ids"]):
                matches.append("circles" if method.startswith("circles") else "baseline")
        for family in set(matches):
            rec = dict(query_id=item["query_id"], demo_ids=item["demo_ids"],
                       attribute=row.get("attribute", ""),
                       modified_caption=row.get("modified_caption", ""))
            key = context_tuple(rec, family)
            value = dict(prediction=item["prediction"], raw_output=item["raw_output"],
                         stats=item["stats"], source=str(ref / "context_cache.jsonl"),
                         source_sha256=sha(ref / "context_cache.jsonl"))
            if key in lookup and (lookup[key]["prediction"], lookup[key]["raw_output"]) != (value["prediction"], value["raw_output"]):
                conflicts.add(key)
            else:
                lookup[key] = value
    for key in conflicts:
        lookup.pop(key, None)
    return lookup


def infer(dataset: str, shots: int, backend):
    out = ROOT / f"{shots}shot" / dataset
    if base.verified(out / "complete.json"):
        return
    assert base.verified(out / "plan.freeze.json")
    plan = read(out / "test_plan.json")
    train = {x.sample_id: x for x in base.manifest(dataset, "train")}
    query = {x.sample_id: x for x in base.manifest(dataset, "test")}
    labels = tuple(sorted({x.label for x in train.values()}))
    question = base.common.source(dataset).instruction
    contract = stable(dict(
        model="Qwen/Qwen3-VL-4B-Instruct", dtype="bfloat16", attention="sdpa",
        processor_settings=base.common.PROCESSOR_SETTINGS, batch_size=2,
        do_sample=False, max_new_tokens=512,
        baseline_prompt_source=sha(target_batch.r.official.__file__),
        executor_source=sha(target_batch.__file__)))
    records, mapping, families = {}, {}, {}
    for row in plan:
        for method, ids in row["methods"].items():
            family = "circles" if method.startswith("circles") else "baseline"
            rec = record_for(query[row["query_id"]], [train[x] for x in ids], family, row)
            key = stable([contract, context_tuple(rec, family),
                          query[row["query_id"]].image_sha256,
                          [train[x].image_sha256 for x in ids]])
            records.setdefault(key, rec)
            mapping[row["query_id"], method] = key
            families[key] = family
    cp = out / "context_cache.jsonl"
    cache = {}
    if cp.exists():
        for item in rows(cp):
            if item["contract"] != contract or item["key"] not in records:
                raise ValueError("foreign direct Generic cache row")
            cache[item["key"]] = item
    old = main_reference_lookup(dataset, shots)
    reused = 0
    for key, rec in records.items():
        if key in cache:
            continue
        source = old.get(context_tuple(rec, families[key]))
        if source is None:
            continue
        item = dict(key=key, contract=contract, query_id=rec["query_id"],
                    demo_ids=rec["demo_ids"], family=families[key],
                    prediction=source["prediction"], raw_output=source["raw_output"],
                    stats=source["stats"], context_source="exact_prompt_historical_batch2",
                    reuse_source=source["source"], reuse_source_sha256=source["source_sha256"])
        base.engine.append_context(cp, item)
        cache[key] = item
        reused += 1
    missing = [key for key in records if key not in cache]
    started = time.time()
    new = 0
    for start in range(0, len(missing), 2):
        held()
        keys = missing[start:start + 2]
        outputs = backend.forward_many([records[key] for key in keys], question, labels)
        if len(outputs) != len(keys):
            raise ValueError("batch output cardinality drift")
        signature = stable(keys)
        for key, value in zip(keys, outputs):
            value["stats"]["fixed_missing_batch_signature"] = signature
            item = dict(key=key, contract=contract,
                        query_id=records[key]["query_id"],
                        demo_ids=records[key]["demo_ids"], family=families[key],
                        context_source="new_fixed_batch2", **value)
            base.engine.append_context(cp, item)
            cache[key] = item
            new += 1
        if start % 100 == 0:
            rate = new / max(time.time() - started, 1e-9) * 60
            emit("target_inference", dataset=dataset, shots=shots,
                 done=len(cache), total=len(records), reused=reused, new=new,
                 contexts_per_minute=rate,
                 eta_minutes=(len(missing) - new) / max(rate, 1e-9))
    truth = {qid: item.label for qid, item in query.items()}
    predictions, results = {}, {}
    reference_predictions = read(reference_dir(dataset, shots) / "predictions.json")
    by_query = {x["query_id"]: x for x in plan}
    for method in METHODS[shots]:
        pred = {qid: cache[mapping[qid, method]]["prediction"] for qid in query}
        predictions[method] = pred
        parent = PARENTS[method]
        decision = DECISIONS[method]
        result = dict(
            metrics=base.r.accuracy_metrics(
                [truth[x] for x in query], [pred[x] for x in query]),
            invalid_exact_output_count=sum(x == "__invalid_exact_output__" for x in pred.values()),
            demo_same_class_precision=float(np.mean([
                np.mean([train[d].label == truth[qid]
                         for d in by_query[qid]["methods"][method]])
                for qid in query])),
            paired_vs_existing_parent=base.paired(reference_predictions[parent], pred, truth),
            paired_vs_existing_decision=base.paired(reference_predictions[decision], pred, truth),
            existing_parent_method=parent, existing_decision_method=decision)
        results[method] = result
    write(out / "predictions.json", predictions)
    write(out / "results.json", dict(
        dataset=dataset, shots=shots, methods=results,
        direct_candidate_retrieval=True, class_prototype=False, class_average=False,
        reused_exact_prompt_contexts=reused, new_target_contexts=new,
        decision_target_rerun=False, parent_target_rerun=False,
        test_used_for_tuning=False, reference_root=str(reference_dir(dataset, shots))))
    saved = read(out / "predictions.json")
    for method in METHODS[shots]:
        if set(saved[method]) != set(truth):
            raise ValueError("prediction coverage drift")
        accuracy = sum(saved[method][x] == truth[x] for x in truth) / len(truth)
        if abs(accuracy - results[method]["metrics"]["accuracy"]) > 1e-12:
            raise ValueError("accuracy recomputation drift")
    write(out / "independent_audit.json", dict(
        status="passed", dataset=dataset, shots=shots, queries=len(truth),
        methods=len(METHODS[shots]), accuracy_recomputed=True,
        class_prototype=False, reused=reused, new=new))
    base.freeze(out / "complete.json", [
        out / "plan.freeze.json", cp, out / "predictions.json",
        out / "results.json", out / "independent_audit.json",
        reference_dir(dataset, shots) / "complete.json"])
    report()
    emit("dataset_complete", dataset=dataset, shots=shots,
         methods=len(METHODS[shots]), reused=reused, new=new)


def report():
    result = {}
    lines = ["# Qwen3-VL-4B Generic direct-sample 1/2-shot", "",
             "Generic 逐训练样本直接检索；不使用类别 prototype 或类别平均。", "",
             "| 数据集 | shot | 方法 | Accuracy | Δ vs parent | rescue/harm | 状态 |",
             "|---|---:|---|---:|---:|---:|---|"]
    completed = 0
    for dataset in ORDER:
        for shots in (1, 2):
            path = ROOT / f"{shots}shot" / dataset / "results.json"
            complete = ROOT / f"{shots}shot" / dataset / "complete.json"
            key = f"{dataset}/{shots}"
            if complete.exists() and path.exists():
                value = read(path)
                result[key] = value
                completed += len(value["methods"])
                for method, item in value["methods"].items():
                    paired = item["paired_vs_existing_parent"]
                    lines.append(f"| {dataset} | {shots} | {method} | {item['metrics']['accuracy']*100:.3f} | {paired['accuracy_delta']*100:+.3f} | {paired['rescue']}/{paired['harm']} | 完整 |")
            else:
                for method in METHODS[shots]:
                    lines.append(f"| {dataset} | {shots} | {method} | — | — | — | 未完成 |")
    write(ROOT / "live_results.json", dict(
        completed_method_results=completed, expected_method_results=25,
        results=result, decision_target_rerun=False, parent_target_rerun=False,
        class_prototype=False))
    (ROOT / "live_results_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("all", "cpu-check", "report"))
    args = parser.parse_args()
    if args.stage == "report":
        report()
        return
    validate_inputs()
    if args.stage == "cpu-check":
        return
    import msvcrt
    locks = []
    for path in (MAIN / "gpu_queue.lock", QUEUE / "gpu_queue.lock"):
        handle = path.open("a+b")
        handle.seek(0); handle.write(b"0"); handle.flush(); handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        locks.append(handle)
    backend = None
    try:
        for dataset in ORDER:
            for shots in (1, 2):
                held()
                out = ROOT / f"{shots}shot" / dataset
                if base.verified(out / "complete.json"):
                    continue
                fit_and_plan(dataset, shots)
                if backend is None:
                    emit("loading_target_model", dataset=dataset, shots=shots)
                    backend = target_batch.BatchedBackend()
                infer(dataset, shots, backend)
        report()
        base.freeze(ROOT / "complete.json", [
            ROOT / f"{shots}shot" / dataset / "complete.json"
            for dataset in ORDER for shots in (1, 2)] + [ROOT / "live_results.json"])
        emit("queue_complete", method_results=25)
    finally:
        if backend is not None:
            del backend
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        emit("failed", error_type=type(error).__name__, error=str(error))
        if not (ROOT / "PAUSED_BY_USER.json").exists():
            write(ROOT / "DIAGNOSTIC_HOLD.json", dict(
                reason=str(error), error_type=type(error).__name__, time=time.time()))
        raise


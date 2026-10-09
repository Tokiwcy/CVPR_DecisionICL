"""Audited, independently refit Qwen3 1/2-shot five-dataset queue."""
from __future__ import annotations

import argparse
import ast
import gc
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_qwen3_circles_official_prompt_2shot_four as official
import numpy as np
from PIL import Image
from dgdr.atom_label import build_class_atom_prototypes, score_classes
from dgdr.hashing import file_sha256, stable_hash
from dgdr.metrics import accuracy_metrics

common, legacy = official.common, official.legacy
PROJECT = SCRIPT_DIR.parent
ROOT = PROJECT / "artifacts/qwen3_five_dataset_recalibrated_20260910"
RESOURCES = ROOT / "resources"
DOC = PROJECT / "docs/qwen3_five_dataset_recalibrated_20260910_zh.md"
ORDER = ("dtd", "caltech101", "aircraft", "cub", "dogs")
METHODS = official.METHODS
BATCHES = dict(dtd=2, caltech101=2, aircraft=2, cub=8, dogs=8)
OLD = PROJECT / "artifacts/qwen3_circles_official_prompt_2shot_seven_20260910"
read, write = common.read, common.write


def rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def emit(stage, **kw):
    value = dict(stage=stage, time=time.time(), **kw)
    write(ROOT / "status.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def held():
    if any((ROOT / x).exists() for x in ("PAUSED_BY_USER.json", "DIAGNOSTIC_HOLD.json", "WATCHDOG_STOP")):
        raise RuntimeError("new queue held")


def hashes(paths):
    return {str(Path(p).resolve()): file_sha256(p) for p in sorted(set(map(Path, paths)))}


def verify_files(values):
    for p, expected in values.items():
        if file_sha256(p) != expected:
            raise ValueError("frozen file drift: " + p)


def source_files():
    files = [Path(__file__), DOC, legacy.OFFICIAL_SOURCE]
    for mod in list(sys.modules.values()):
        path = getattr(mod, "__file__", None)
        if path and Path(path).suffix == ".py" and Path(path).resolve().is_relative_to(PROJECT):
            files.append(Path(path))
    return files


def dataset_files(dataset):
    s = common.source(dataset)
    files = [s.bank, s.labels, s.prior_protocol]
    for split in ("train", "validation", "test"):
        files += [s.manifest(split)]
        for stem in (s.embedding(split), s.evidence_path(split)):
            files += [Path(str(stem) + ext) for ext in (".json", ".npz")]
    return files


def audit_inputs():
    held()
    import subprocess
    if subprocess.check_output(["git", "-C", str(legacy.OFFICIAL), "rev-parse", "HEAD"], text=True).strip() != legacy.OFFICIAL_COMMIT:
        raise ValueError("official commit drift")
    files = source_files()
    for model in (common.MODEL, legacy.CLIP_MODEL):
        files += [p for p in model.rglob("*") if p.is_file() and p.suffix in (".json", ".safetensors", ".txt", ".model")]
    counts = {}
    for dataset in ORDER:
        split_rows = {split: common.load(dataset, split)[0] for split in ("train", "validation", "test")}
        seen = set()
        for split, data in split_rows.items():
            common.evidence(dataset, split)
            ids = [x.sample_id for x in data]
            images = {x.image_sha256 for x in data}
            if len(ids) != len(set(ids)) or seen & images:
                raise ValueError(f"duplicate IDs or split image leakage: {dataset}/{split}")
            seen |= images
            for sample in data:
                if file_sha256(sample.image_path) != sample.image_sha256:
                    raise ValueError("image drift: " + sample.image_path)
        labels = sorted({x.label for x in split_rows["train"]})
        if read(common.source(dataset).prior_protocol)["labels"] != labels:
            raise ValueError("label vector ordering drift")
        counts[dataset] = {k: len(v) for k, v in split_rows.items()}
        files += dataset_files(dataset)
        emit("input_audited", dataset=dataset, counts=counts[dataset])
    receipt = dict(status="passed", files=hashes(files), counts=counts, image_bytes_verified=True,
                   test_used_for_tuning=False, methods_1shot=list(METHODS[:5]), methods_2shot=list(METHODS))
    write(ROOT / "input_audit.json", receipt)
    return receipt


def assert_audit():
    value = read(ROOT / "input_audit.json")
    verify_files(value["files"])


def seed_resources(dataset):
    out = RESOURCES / dataset
    out.mkdir(parents=True, exist_ok=True)
    receipt_path = out / "reuse_receipt.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
        verify_files({v["source"]: v["sha256"] for v in receipt["files"].values()})
        return
    copied = {}
    src = legacy.OUT / dataset
    for path in sorted(src.glob("*")):
        name = path.name
        allowed = (name.startswith("clip.") and (name.endswith(".npz") or name.endswith(".npz.json"))) or name.startswith("counterfactual_descriptions.") or (name.startswith("description.") and name.endswith(".complete.json"))
        if not path.is_file() or not allowed:
            continue
        target = out / name
        if target.exists():
            raise ValueError("unreceipted resource exists: " + str(target))
        shutil.copy2(path, target)
        if file_sha256(target) != file_sha256(path):
            raise ValueError("resource copy drift")
        copied[name] = dict(source=str(path.resolve()), sha256=file_sha256(path))
    write(receipt_path, dict(files=copied, description_batch=BATCHES[dataset], target_predictions_reused=False))


def prepare_images(dataset):
    seed_resources(dataset)
    out = RESOURCES / dataset
    missing = [split for split in ("train", "validation", "test") if not legacy.valid_embedding(out / f"clip.{split}.npz")]
    if not missing:
        return
    import torch
    from transformers import CLIPModel, CLIPProcessor
    torch.set_num_threads(4)
    model = CLIPModel.from_pretrained(legacy.CLIP_MODEL, local_files_only=True).to("cuda:0").eval()
    processor = CLIPProcessor.from_pretrained(legacy.CLIP_MODEL, local_files_only=True)
    try:
        for split in missing:
            data = legacy.samples(dataset, split)
            chunks = []
            for start in range(0, len(data), legacy.IMAGE_BATCH):
                held()
                batch = data[start:start + legacy.IMAGE_BATCH]
                images = []
                try:
                    for item in batch:
                        with Image.open(item.image_path) as im:
                            images.append(im.convert("RGB"))
                    pixels = processor(images=images, return_tensors="pt")["pixel_values"].to("cuda:0")
                    with torch.inference_mode():
                        features = legacy.clip_feature_tensor(model.get_image_features(pixel_values=pixels), torch)
                    chunks.append(features.cpu().numpy())
                finally:
                    for im in images:
                        im.close()
                if start % 400 == 0:
                    emit("clip_images", dataset=dataset, split=split, done=start + len(batch), total=len(data))
            legacy.save_embeddings(out / f"clip.{split}.npz", [x.sample_id for x in data], np.concatenate(chunks), f"image/{split}")
    finally:
        del model, processor
        gc.collect(); torch.cuda.empty_cache()


def prepare_captions(dataset):
    # Preserve the dataset's original batching; an OOM must stop, not change semantics.
    old_out, old_order, old_batch, old_held = legacy.OUT, legacy.ORDER, legacy.DESCRIPTION_BATCH, legacy.held
    old_generate = legacy.Generator.generate
    def strict_generate(self, *a, **kw):
        try:
            return old_generate(self, *a, **kw)
        except self.torch.OutOfMemoryError as exc:
            raise RuntimeError("description OOM; batch contract must not silently change") from exc
    try:
        legacy.OUT, legacy.ORDER, legacy.DESCRIPTION_BATCH, legacy.held = RESOURCES, (dataset,), BATCHES[dataset], held
        legacy.Generator.generate = strict_generate
        legacy.describe_all()
        legacy.embed_all()
    finally:
        legacy.OUT, legacy.ORDER, legacy.DESCRIPTION_BATCH, legacy.held = old_out, old_order, old_batch, old_held
        legacy.Generator.generate = old_generate


def smoke():
    held()
    backend = official.OfficialBackend()
    outputs = []
    try:
        for dataset in ORDER:
            train = legacy.samples(dataset, "train")
            val = legacy.samples(dataset, "validation")
            labels = tuple(sorted({x.label for x in train}))
            for shots, family in ((1, "baseline"), (2, "baseline"), (2, "circles")):
                record = dict(method=METHODS[0] if family == "baseline" else METHODS[5],
                    query_id=val[0].sample_id, query_image=val[0].image_path,
                    demo_ids=[x.sample_id for x in train[:shots]],
                    demo_images=[x.image_path for x in train[:shots]], demo_labels=[x.label for x in train[:shots]],
                    attribute="appearance", modified_caption="An image with a different appearance.")
                output = backend.forward(record, common.source(dataset).instruction, labels)
                outputs.append(dict(dataset=dataset, shots=shots, family=family, record=record, output=output))
                emit("validation_smoke", dataset=dataset, shots=shots, family=family, exact=output["stats"]["exact_label_output"])
        if any(not x["output"]["stats"]["exact_label_output"] for x in outputs):
            raise ValueError("validation smoke contains non-exact output; inspect before launching")
        write(ROOT / "validation_smoke.json", dict(status="passed", validation_only=True, test_manifest_loaded=False,
              structural_caption_fixture=True, source_files=hashes(source_files()), outputs=outputs))
    finally:
        del backend
        gc.collect()
        import torch
        torch.cuda.empty_cache()


def top(scores, k):
    return official.topk(scores, k)


def within(scores, ty, classes, k):
    result = []
    for i, cls in enumerate(classes):
        candidates = np.flatnonzero(ty == cls)
        result.append(candidates[top(scores[i:i + 1, candidates], k)[0]].tolist())
    return result


def fit(dataset, shots):
    held()
    out = ROOT / f"{shots}shot" / dataset
    if (out / "plan.freeze.json").exists():
        verify_plan(dataset, shots)
        return
    s = common.source(dataset)
    train, te = common.load(dataset, "train")
    val, ve = common.load(dataset, "validation")
    tv, vv = common.evidence(dataset, "train"), common.evidence(dataset, "validation")
    labels, ty, vy = common.labels_and_y(train, val)
    prototypes = build_class_atom_prototypes(samples=train, evidence=tv)
    assert prototypes.labels == labels
    tids, vids = [x.sample_id for x in train], [x.sample_id for x in val]
    res = RESOURCES / dataset
    ti = legacy.load_embeddings(res / "clip.train.npz", tids)
    vi = legacy.load_embeddings(res / "clip.validation.npz", vids)
    lv = np.load(s.labels).astype(np.float32)
    if lv.shape != (len(labels), te.embeddings.shape[1]):
        raise ValueError("label embedding dimension drift")
    ps = vi @ ti.T
    mp = common.parent_scores(ve, te, ty, lv)
    features = {
        METHODS[3]: official.single_channel_atom_features(ps, ty, vv, prototypes, len(labels)),
        METHODS[4]: common.cub.shared.class_features(ve, vv, te, ty, lv, prototypes)[0],
    }
    if shots == 2:
        vc = legacy.load_embeddings(res / "clip.captions.validation.npz", vids)
        cp = vc @ ti.T
        features[METHODS[6]], oi, ci = legacy.circle_atom_features(ps, cp, ty, vv, prototypes, len(labels))
    calibration = {}
    for method, feature in features.items():
        oof, alpha, weights, fit_score, history = official.fit_class_equal(feature, vy, val)
        calibration[method] = dict(alpha=alpha, weights=weights.tolist(), outer_folds=history,
              full_fit_class_accuracy=fit_score, oof_class_accuracy=float(np.mean(oof == vy)),
              independently_fit=True, inherited_weights=False, shots=shots, dataset=dataset)
        emit("fresh_calibration", dataset=dataset, shots=shots, method=method, alpha=alpha)
    write(out / "calibration.json", dict(methods=calibration, test_used=False, inherited_weights=False,
          alpha_grid=list(common.ALPHAS), alpha_means="parent mass", tie_rule="larger parent mass",
          objective="validation class accuracy", folds=5, seed=42, shots=shots,
          input_hashes=hashes([s.bank, s.labels, s.manifest("train"), s.manifest("validation")]),
          fresh_fit_started_at=time.time()))
    # Test is first loaded only after every applicable parameter has been fitted and saved.
    query, qe = common.load(dataset, "test")
    qv = common.evidence(dataset, "test")
    qids = [x.sample_id for x in query]
    qi = legacy.load_embeddings(res / "clip.test.npz", qids)
    descriptions = {}
    if shots == 2:
        qc = legacy.load_embeddings(res / "clip.captions.test.npz", qids)
        descriptions = {x["query_id"]: x for x in rows(res / "counterfactual_descriptions.test.jsonl")}
    rng = np.random.default_rng(42)
    plan = []
    for start in range(0, len(query), 64):
        held()
        stop = min(start + 64, len(query))
        block = common.EmbeddingStore(qe.sample_ids[start:stop], qe.embeddings[start:stop], qe.model_id, qe.model_revision)
        ev = common.cub.shared.sliced_evidence(qv, start, stop)
        ps = qi[start:stop] @ ti.T
        mp = common.parent_scores(block, te, ty, lv)
        rf = official.single_channel_atom_features(ps, ty, ev, prototypes, len(labels))
        mf = common.cub.shared.class_features(block, ev, te, ty, lv, prototypes)[0]
        selections = {METHODS[1]: top(ps, shots).tolist(), METHODS[2]: top(mp, shots).tolist()}
        for method, feature, score in ((METHODS[3], rf, ps), (METHODS[4], mf, mp)):
            classes = score_classes(feature, np.asarray(calibration[method]["weights"], np.float32)).argmax(1)
            selections[method] = within(score, ty, classes, shots)
        if shots == 2:
            cp = qc[start:stop] @ ti.T
            cf, oi, ci = legacy.circle_atom_features(ps, cp, ty, ev, prototypes, len(labels))
            classes = score_classes(cf, np.asarray(calibration[METHODS[6]]["weights"], np.float32)).argmax(1)
            selections[METHODS[5]] = list(zip(legacy.last_argmax(ps), legacy.last_argmax(cp)))
            selections[METHODS[6]] = [(oi[i, c], ci[i, c]) for i, c in enumerate(classes)]
        for i, qid in enumerate(qids[start:stop]):
            methods = {METHODS[0]: dict(demo_ids=[tids[int(x)] for x in rng.choice(len(train), size=shots, replace=False)])}
            methods.update({m: dict(demo_ids=[tids[int(x)] for x in picks[i]]) for m, picks in selections.items()})
            desc = descriptions.get(qid, {})
            plan.append(dict(query_id=qid, methods=methods, attribute=desc.get("attribute", ""),
                             modified_caption=desc.get("modified_caption", ""), test_label_used=False))
    path = out / "test_plan.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in plan), encoding="utf-8")
    inputs = dataset_files(dataset) + source_files() + [ROOT / "input_audit.json", ROOT / "validation_smoke.json", out / "calibration.json"]
    resource_names = {f"clip.{split}.npz{suffix}" for split in ("train", "validation", "test") for suffix in ("", ".json")}
    resource_names.add("reuse_receipt.json")
    if shots == 2:
        resource_names.update(f"clip.captions.{split}.npz{suffix}" for split in ("validation", "test") for suffix in ("", ".json"))
        resource_names.update(f"counterfactual_descriptions.{split}.jsonl" for split in ("validation", "test"))
    inputs += [res / name for name in resource_names]
    write(out / "plan.freeze.json", dict(files=hashes(inputs), plan_sha256=file_sha256(path),
          shots=shots, methods=list(METHODS[:5] if shots == 1 else METHODS), independent_recalibration=True))
    verify_plan(dataset, shots)
    emit("plan_frozen", dataset=dataset, shots=shots, queries=len(plan))


def verify_plan(dataset, shots):
    out = ROOT / f"{shots}shot" / dataset
    freeze = read(out / "plan.freeze.json")
    verify_files(freeze["files"])
    if file_sha256(out / "test_plan.jsonl") != freeze["plan_sha256"]:
        raise ValueError("plan hash drift")
    plan = rows(out / "test_plan.jsonl")
    train = {x.sample_id for x in legacy.samples(dataset, "train")}
    if [x["query_id"] for x in plan] != [x.sample_id for x in legacy.samples(dataset, "test")]:
        raise ValueError("query coverage/order drift")
    for row in plan:
        if set(row["methods"]) != set(freeze["methods"]):
            raise ValueError("method matrix incomplete")
        for method, info in row["methods"].items():
            demos = info["demo_ids"]
            if len(demos) != shots or not set(demos) <= train:
                raise ValueError("demo contract drift")
            if method in METHODS[:5] and len(set(demos)) != shots:
                raise ValueError("duplicate ordinary demos")
    return plan


def context(record, dataset, labels):
    train = record["demo_images"]
    conversation = official.official_content(common.source(dataset).instruction, record["query_image"],
                    list(zip(train, record["demo_labels"])), labels, record["method"], record["attribute"], record["modified_caption"])
    # Hash image bytes rather than path spelling; conversation contains every text token's source.
    for item in conversation[0]["content"]:
        if item["type"] == "image":
            item["image"] = file_sha256(item["image"])
    return conversation


def infer_contract():
    audit = read(ROOT / "input_audit.json")
    return dict(model_files={p: h for p, h in audit["files"].items() if Path(p).is_relative_to(common.MODEL)},
                dtype="bfloat16", attention="sdpa", processor_settings=common.PROCESSOR_SETTINGS,
                batch_size=1, max_new_tokens=512, do_sample=False, use_cache=True,
                forward_source=inspect.getsource(official.OfficialBackend.forward),
                prompt_source=inspect.getsource(official.official_content))


def make_records(dataset, shots, contract_hash):
    plan = verify_plan(dataset, shots)
    train = {x.sample_id: x for x in legacy.samples(dataset, "train")}
    query = {x.sample_id: x for x in legacy.samples(dataset, "test")}
    labels = tuple(sorted({x.label for x in train.values()}))
    records, keys = {}, {m: {} for m in (METHODS[:5] if shots == 1 else METHODS)}
    for row in plan:
        q = query[row["query_id"]]
        for method in keys:
            demos = row["methods"][method]["demo_ids"]
            record = dict(method=method, query_id=q.sample_id, query_image=q.image_path, demo_ids=demos,
                          demo_images=[train[x].image_path for x in demos], demo_labels=[train[x].label for x in demos],
                          attribute=row["attribute"], modified_caption=row["modified_caption"])
            key = stable_hash([contract_hash, context(record, dataset, labels)])
            record["key"] = key
            records.setdefault(key, record)
            keys[method][q.sample_id] = key
    return plan, train, query, labels, records, keys


def migrate(dataset, shots, records, ph, cache):
    if shots != 2 or not (OLD / dataset / "context_cache.jsonl").exists():
        return []
    src = OLD / dataset
    freeze = read(src / "plan.freeze.json")
    verify_files(freeze["files"])
    if file_sha256(src / "test_plan.jsonl") != freeze["plan_sha256"]:
        raise ValueError("old official plan drift")
    proto = read(src / "inference_protocol.json")
    if proto["contract_sha256"] != stable_hash({k: v for k, v in proto.items() if k != "contract_sha256"}):
        raise ValueError("old protocol hash invalid")
    expected = dict(model="Qwen/Qwen3-VL-4B-Instruct", dtype="bfloat16", attention="sdpa",
                    processor_settings=common.PROCESSOR_SETTINGS, execution_batch_size=1, max_new_tokens=512, temperature=0.0,
                    prompt_version="official-circles-seven-method-classification-2shot-v1")
    for key, value in expected.items():
        if proto[key] != value:
            raise ValueError("old generation contract drift: " + key)
    train = {x.sample_id: x for x in legacy.samples(dataset, "train")}
    queries = {x.sample_id: x for x in legacy.samples(dataset, "test")}
    labels = tuple(sorted({x.label for x in train.values()}))
    if proto["labels"] != list(labels):
        raise ValueError("migration label order drift")
    old_plan = {x["query_id"]: x for x in rows(src / "test_plan.jsonl")}
    converted = []
    for old in rows(src / "context_cache.jsonl"):
        if old["protocol_hash"] != proto["contract_sha256"]:
            raise ValueError("old cache foreign protocol")
        row = old_plan[old["query_id"]]
        family = old["method_family"]
        matches = [m for m, info in row["methods"].items() if info["demo_ids"] == old["demo_ids"] and ("baseline" if m in METHODS[:5] else "circles") == family]
        if not matches:
            raise ValueError("old cached context absent from its plan")
        q, demos = queries[old["query_id"]], old["demo_ids"]
        prompt_context = [row["attribute"], row["modified_caption"]] if family == "circles" else []
        expected_key = stable_hash([proto["contract_sha256"], family, q.sample_id, demos, prompt_context, q.image_sha256, [train[x].image_sha256 for x in demos]])
        if expected_key != old["key"]:
            raise ValueError("old context key mismatch")
        record = dict(method=matches[0], query_id=q.sample_id, query_image=q.image_path, demo_ids=demos,
             demo_images=[train[x].image_path for x in demos], demo_labels=[train[x].label for x in demos],
             attribute=row["attribute"], modified_caption=row["modified_caption"])
        key = stable_hash([ph, context(record, dataset, labels)])
        if key not in records:
            continue
        if key in cache:
            if cache[key]["prediction"] != old["prediction"] or cache[key]["raw_output"] != old["raw_output"]:
                raise ValueError("conflicting migrated context")
            continue
        value = {**old, "key": key, "protocol_hash": ph, "source_key": old["key"], "migrated_from": str(src / "context_cache.jsonl")}
        cache[key] = value
        converted.append(value)
    write(ROOT / f"{shots}shot" / dataset / "cache_migration_receipt.json", dict(
        files=hashes([src / n for n in ("context_cache.jsonl", "plan.freeze.json", "test_plan.jsonl", "inference_protocol.json")]),
        matched_contexts=len(converted), full_prompt_and_ordered_image_hash_match=True,
        source_executor_verified_by_freeze=True, model_files_bound_by_new_audit=True))
    return converted


def infer(dataset, shots, backend):
    held()
    out = ROOT / f"{shots}shot" / dataset
    contract = infer_contract()
    ph = stable_hash(contract)
    write(out / "inference_protocol.json", dict(contract=contract, contract_sha256=ph))
    plan, train, query, labels, records, keys = make_records(dataset, shots, ph)
    cache_path = out / "context_cache.jsonl"
    cache = {}
    for value in rows(cache_path) if cache_path.exists() else []:
        key = value["key"]
        if value["protocol_hash"] != ph or key not in records:
            raise ValueError("foreign cache row")
        if key in cache and (cache[key]["raw_output"], cache[key]["prediction"]) != (value["raw_output"], value["prediction"]):
            raise ValueError("conflicting cache key")
        cache[key] = value
    migrated = migrate(dataset, shots, records, ph, cache)
    common.append_rows(cache_path, migrated)
    pending = [r for k, r in records.items() if k not in cache]
    emit("target_prepared", dataset=dataset, shots=shots, requested=len(plan)*len(keys), unique=len(records), cached=len(cache), migrated=len(migrated), new=len(pending))
    started = time.perf_counter()
    for index, record in enumerate(pending, 1):
        held()
        output = backend.forward(record, common.source(dataset).instruction, labels)
        value = dict(key=record["key"], protocol_hash=ph, query_id=record["query_id"], demo_ids=record["demo_ids"], **output)
        common.append_rows(cache_path, [value])
        cache[value["key"]] = value
        if index % 25 == 0 or index == len(pending):
            rate = index / max(time.perf_counter() - started, 1e-9) * 60
            emit("target_running", dataset=dataset, shots=shots, done=index, total=len(pending), cached=len(cache), rate_per_minute=rate, eta_minutes=(len(pending)-index)/max(rate, 1e-9))
    predictions = {m: {q: cache[k]["prediction"] for q, k in mapping.items()} for m, mapping in keys.items()}
    truth = {x: row.label for x, row in query.items()}
    result = {}
    by_id = {x["query_id"]: x for x in plan}
    for method, values in predictions.items():
        item = dict(metrics=accuracy_metrics(list(truth.values()), [values[x] for x in truth]),
                    invalid_exact_output_count=sum(x == "__invalid_exact_output__" for x in values.values()),
                    demo_same_class_precision=float(np.mean([sum(train[d].label == truth[q] for d in by_id[q]["methods"][method]["demo_ids"])/shots for q in truth])))
        if method in official.ATOM_PARENTS:
            item["paired_vs_parent"] = common.paired(predictions[official.ATOM_PARENTS[method]], values, truth)
        if method in METHODS[5:]:
            item["paired_vs_matched_rices"] = common.paired(predictions[METHODS[1]], values, truth)
        result[method] = item
    write(out / "predictions.json", predictions)
    write(out / "results.json", dict(status="complete", dataset=dataset, shots=shots, methods=result, test_used_for_tuning=False))
    write(out / "complete.json", dict(files=hashes([out / n for n in ("plan.freeze.json", "test_plan.jsonl", "inference_protocol.json", "context_cache.jsonl", "predictions.json", "results.json")]), time=time.time()))
    verify_complete(dataset, shots)
    update_evidence(dataset, shots)
    emit("dataset_complete", dataset=dataset, shots=shots)


def verify_complete(dataset, shots):
    out = ROOT / f"{shots}shot" / dataset
    verify_files(read(out / "complete.json")["files"])
    plan, train, query, labels, records, keys = make_records(dataset, shots, stable_hash(infer_contract()))
    cache = {x["key"]: x for x in rows(out / "context_cache.jsonl")}
    if set(cache) != set(records):
        raise ValueError("complete cache coverage drift")
    lookup = {label.lower(): label for label in labels}
    for key, value in cache.items():
        if lookup.get(value["raw_output"].strip().lower(), "__invalid_exact_output__") != value["prediction"]:
            raise ValueError("strict output parser mismatch")
    predictions, results = read(out / "predictions.json"), read(out / "results.json")["methods"]
    truth = {q: x.label for q, x in query.items()}
    for method, mapping in keys.items():
        expected = {q: cache[k]["prediction"] for q, k in mapping.items()}
        if predictions[method] != expected:
            raise ValueError("prediction-cache mismatch")
        metrics = accuracy_metrics(list(truth.values()), [expected[q] for q in truth])
        if metrics != results[method]["metrics"]:
            raise ValueError("accuracy mismatch")
        if method in official.ATOM_PARENTS:
            paired = common.paired(predictions[official.ATOM_PARENTS[method]], expected, truth)
            if paired != results[method]["paired_vs_parent"]:
                raise ValueError("paired statistics mismatch")
        if method in METHODS[5:]:
            if common.paired(predictions[METHODS[1]], expected, truth) != results[method]["paired_vs_matched_rices"]:
                raise ValueError("CIRCLES paired statistics mismatch")
    write(out / "independent_audit.json", dict(status="passed", methods=len(keys), queries=len(query), contexts=len(records), files=hashes([out / "complete.json"])))


def update_evidence(dataset, shots):
    path = PROJECT.parent / "docs/iclr_decision_guided_mainline_zh.md"
    marker = f"<!-- QWEN3_FRESH_20260910_{dataset}_{shots}SHOT_COMPLETE -->"
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return
    result = read(ROOT / f"{shots}shot" / dataset / "results.json")
    scores = "；".join(f"{m}: {v['metrics']['accuracy']*100:.3f}%" for m, v in result["methods"].items())
    path.write_text(text + f"\n\n{marker}\nQwen3-VL-4B {dataset} {shots}-shot 新合同独立重校准完整结束，结果/缓存/计划/accuracy和paired统计复核通过。{scores}。完整paired rescue/harm、bootstrap CI、McNemar见 `decision_guided_retrieval_cleanroom/artifacts/qwen3_five_dataset_recalibrated_20260910/{shots}shot/{dataset}/results.json`。探索性结果，不据test调整方法。\n", encoding="utf-8")


def report():
    lines = ["# Qwen3-VL-4B 五数据集独立重校准", "", "|Dataset|Shot|Method|Accuracy|Macro|", "|---|---:|---|---:|---:|"]
    receipts = []
    for dataset in ORDER:
        for shots in (1, 2):
            verify_complete(dataset, shots)
            out = ROOT / f"{shots}shot" / dataset
            receipts.append(out / "complete.json")
            for m, value in read(out / "results.json")["methods"].items():
                metric = value["metrics"]
                lines.append(f"|{dataset}|{shots}|{m}|{metric['accuracy']*100:.3f}|{metric['macro_accuracy']*100:.3f}|")
    path = ROOT / "report_zh.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write(ROOT / "complete.json", dict(files=hashes(receipts + [path])))
    emit("queue_complete", datasets=list(ORDER), method_dataset_cells=60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("audit", "smoke", "prepare", "fit", "all", "verify", "report"))
    parser.add_argument("--datasets", nargs="+", choices=ORDER, default=list(ORDER))
    parser.add_argument("--shots", type=int, nargs="+", choices=(1, 2), default=[1, 2])
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.stage == "audit":
        audit_inputs(); return
    if args.stage == "smoke":
        smoke(); return
    if args.stage == "report":
        report(); return
    assert_audit()
    if read(ROOT / "validation_smoke.json")["status"] != "passed":
        raise ValueError("GPU preflight has not passed")
    for dataset in args.datasets:
        for shots in args.shots:
            held()
            out = ROOT / f"{shots}shot" / dataset
            if (out / "complete.json").exists():
                verify_complete(dataset, shots)
                continue
            if args.stage == "verify":
                verify_plan(dataset, shots); continue
            prepare_images(dataset)
            if shots == 2:
                prepare_captions(dataset)
            if args.stage == "prepare":
                continue
            fit(dataset, shots)
            if args.stage == "all":
                backend = official.OfficialBackend()
                try:
                    infer(dataset, shots, backend)
                finally:
                    del backend
                    gc.collect()
                    import torch
                    torch.cuda.empty_cache()
    if args.stage == "all" and tuple(args.datasets) == ORDER and args.shots == [1, 2]:
        report()


if __name__ == "__main__":
    main()

"""Resumable Qwen3-VL-4B BF16 2-shot run on five frozen datasets.

The run reuses only frozen model-specific factor banks and frozen SigLIP
measurements.  It refits both class-equal fusion coefficients on validation for
this exact 2-shot retrieval contract, freezes test plans before inference, and
never uses test labels to construct or tune a plan.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

import numpy as np
from PIL import Image

import run_cub_raw_main_methods as cub
import run_qwen25_nf4_class_equal as common
from dgdr.atom_label import build_class_atom_prototypes, score_classes
from dgdr.calibration import stratified_folds
from dgdr.evidence import EmbeddingStore, EvidenceStore
from dgdr.hashing import file_sha256, stable_hash
from dgdr.io import load_atom_bank, load_manifest
from dgdr.metrics import accuracy_metrics
from run_qwen3_dataset_plan import paired, single_token_codes

OUT = PROJECT / "artifacts/qwen3_2shot_five_20260908"
MODEL = REPO / "models/Qwen3-VL-4B-Instruct"
METHODS = ("random", "rices", "muier", "rices_atom_class_equal_raw", "muier_atom_class_equal_raw")
PARENTS = {"rices_atom_class_equal_raw": "rices", "muier_atom_class_equal_raw": "muier"}
ORDER = ("dtd", "caltech101", "aircraft", "cub", "dogs")
PROCESSOR_SETTINGS = {"min_pixels": 65536, "max_pixels": 401408}
ALPHAS = tuple(float(x) for x in cub.base.ALPHAS)
SEED = 42


@dataclass(frozen=True)
class Source:
    dataset: str
    data: Path
    bank: Path
    evidence: Path
    test_evidence: Path
    labels: Path
    prior_protocol: Path
    instruction: str
    embedding_stem: str = ""

    def manifest(self, split: str) -> Path:
        name = f"{self.dataset}.{split}.jsonl" if self.dataset != "caltech101" else f"{split}.jsonl"
        return self.data / name

    def embedding(self, split: str) -> Path:
        if self.dataset == "caltech101" and split == "test":
            return PROJECT / "artifacts/caltech_three_models_20260908/shared/caltech101/siglip.test.embeddings"
        return self.data / f"{self.embedding_stem}{split}.embeddings"

    def evidence_path(self, split: str) -> Path:
        return (self.test_evidence if split == "test" else self.evidence) / f"siglip.{split}.evidence"


def source(dataset: str) -> Source:
    if dataset == "cub":
        return Source(dataset, PROJECT / "artifacts/cub_full_current_atoms_20260904",
            PROJECT / "artifacts/siglip_raw_selected_20260906/cub/atom_bank.original.json",
            PROJECT / "artifacts/siglip_raw_selected_20260906/cub",
            PROJECT / "artifacts/cub_raw_siglip_main_methods_20260906",
            PROJECT / "artifacts/cub_raw_siglip_main_methods_20260906/label_embeddings.npy",
            PROJECT / "artifacts/cub_raw_siglip_main_methods_20260906/qwen3_inference_protocol.json",
            "Identify the bird species shown in the image.", "siglip.")
    if dataset in ("aircraft", "dogs", "dtd"):
        data = PROJECT / f"artifacts/four_dataset_atom_label_20260905/{dataset}"
        raw = PROJECT / f"artifacts/siglip_raw_selected_20260906/{dataset}"
        old = PROJECT / f"artifacts/raw_class_equal_mainline_20260906/{dataset}"
        cfg = read(data / "dataset_config.json")
        return Source(dataset, data, raw / "atom_bank.original.json", raw, old,
            old / "label_embeddings.npy", old / "qwen3_inference_protocol.json", cfg["instruction"])
    if dataset == "caltech101":
        data = PROJECT / "artifacts/cars_caltech_three_models_20260907/data/caltech101"
        old = PROJECT / "artifacts/caltech_three_models_20260908/runs/qwen3_4b_bf16/caltech101"
        cfg = read(data / "dataset_config.json")
        return Source(dataset, data, old / "atom_bank.final.json", old, old,
            old / "label_embeddings.npy", old / "inference_protocol.json", cfg["instruction"], "siglip.")
    raise ValueError(dataset)


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path: Path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp, path)


def append_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
        handle.flush()


def emit(stage: str, **values):
    row = dict(stage=stage, time=time.time(), **values)
    write(OUT / "status.json", row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def held():
    if any((OUT / name).exists() for name in ("PAUSED_BY_USER.json", "DIAGNOSTIC_HOLD.json", "WATCHDOG_STOP")):
        raise RuntimeError("2-shot queue held by control file")


def load(dataset: str, split: str):
    s = source(dataset)
    rows = load_manifest(s.manifest(split), expected_split=split)
    emb = EmbeddingStore.load(s.embedding(split))
    if tuple(x.sample_id for x in rows) != emb.sample_ids:
        raise ValueError(f"{dataset}/{split} manifest-embedding order drift")
    return rows, emb


def evidence(dataset: str, split: str):
    s = source(dataset)
    value = EvidenceStore.load(s.evidence_path(split))
    if value.sample_ids != load(dataset, split)[1].sample_ids:
        raise ValueError(f"{dataset}/{split} evidence order drift")
    bank = load_atom_bank(s.bank)
    if value.bank_hash != bank.bank_hash:
        raise ValueError(f"{dataset}/{split} evidence-bank drift")
    return value


def labels_and_y(train, query):
    labels = tuple(sorted({x.label for x in train}))
    lookup = {x: i for i, x in enumerate(labels)}
    return labels, np.asarray([lookup[x.label] for x in train], np.int64), np.asarray([lookup[x.label] for x in query], np.int64)


def top2(scores: np.ndarray) -> np.ndarray:
    if scores.shape[1] < 2:
        raise ValueError("2-shot requires at least two candidates")
    selected = np.argpartition(scores, -2, axis=1)[:, -2:]
    values = np.take_along_axis(scores, selected, axis=1)
    order = np.argsort(-values, axis=1, kind="stable")
    return np.take_along_axis(selected, order, axis=1)


def parent_scores(qe, te, train_y, label_vectors):
    scores = qe.embeddings @ te.embeddings.T
    if label_vectors is not None:
        scores = scores + (qe.embeddings @ label_vectors.T)[:, train_y]
    return scores.astype(np.float32, copy=False)


def pairs_for_classes(classes, scores, train_y, train_ids):
    result = []
    for row, cls in enumerate(np.asarray(classes, np.int64)):
        candidates = np.flatnonzero(train_y == cls)
        chosen = candidates[top2(scores[row:row + 1, candidates])[0]]
        result.append([str(train_ids[chosen[0]]), str(train_ids[chosen[1]])])
    return result


def precision_at_2(pairs, truth_y, id_to_y):
    return float(np.mean([(id_to_y[a] == y) + (id_to_y[b] == y) for (a, b), y in zip(pairs, truth_y)]) / 2.0)


def fit_alpha(features, truth_y, rows):
    best_alpha, best = 1.0, -1.0
    for alpha in reversed(ALPHAS):  # ties prefer more parent mass
        weights = cub.shared.equal_weights(features.shape[2], alpha)
        pred = score_classes(features[rows], weights).argmax(1)
        value = float(np.mean(pred == truth_y[rows]))
        if value > best + 1e-12:
            best_alpha, best = alpha, value
    return best_alpha, best


def validate_pair_plan(methods, query_ids, train_ids):
    allowed = set(str(x) for x in train_ids)
    for method, values in methods.items():
        if len(values) != len(query_ids):
            raise ValueError(f"incomplete plan: {method}")
        for pair in values:
            if len(pair) != 2 or pair[0] == pair[1] or not set(pair) <= allowed:
                raise ValueError(f"invalid 2-shot pair: {method}/{pair}")


def prepare_dataset(dataset: str):
    held(); out = OUT / dataset; out.mkdir(parents=True, exist_ok=True)
    marker = out / "plan.freeze.json"
    if marker.exists():
        verify_dataset(dataset, require_complete=False)
        return
    s = source(dataset)
    train, te = load(dataset, "train"); val, ve = load(dataset, "validation")
    tv, vv = evidence(dataset, "train"), evidence(dataset, "validation")
    bank = load_atom_bank(s.bank)
    labels, ty, vy = labels_and_y(train, val)
    label_vectors = np.load(s.labels).astype(np.float32)
    prior = read(s.prior_protocol)
    if prior["labels"] != list(labels) or label_vectors.shape != (len(labels), te.embeddings.shape[1]):
        raise ValueError(f"{dataset} label protocol drift")
    prototypes = build_class_atom_prototypes(samples=train, evidence=tv)
    if prototypes.labels != labels:
        raise ValueError(f"{dataset} prototype label drift")
    folds = stratified_folds(val, 5, SEED)
    train_ids = np.asarray(te.sample_ids)
    id_to_y = {str(q): int(y) for q, y in zip(train_ids, ty)}
    calibration = {}
    for method, parent in PARENTS.items():
        text = label_vectors if parent == "muier" else np.zeros_like(label_vectors)
        f, _ = cub.shared.class_features(ve, vv, te, ty, text, prototypes)
        ps = parent_scores(ve, te, ty, label_vectors if parent == "muier" else None)
        oof = np.empty(len(val), np.int64); history = []
        for fold in range(5):
            fit = np.flatnonzero(folds != fold); hold = np.flatnonzero(folds == fold)
            alpha, score = fit_alpha(f, vy, fit)
            weights = cub.shared.equal_weights(f.shape[2], alpha)
            oof[hold] = score_classes(f[hold], weights).argmax(1)
            history.append(dict(fold=fold, alpha=alpha, weights=weights.tolist(), fit_same_class=score,
                                fit_count=len(fit), heldout_count=len(hold)))
        alpha, fit_score = fit_alpha(f, vy, np.arange(len(vy)))
        weights = cub.shared.equal_weights(f.shape[2], alpha)
        atom_pairs = pairs_for_classes(oof, ps, ty, train_ids)
        base_pairs = [[str(train_ids[a]), str(train_ids[b])] for a, b in top2(ps)]
        atom_oof = precision_at_2(atom_pairs, vy, id_to_y)
        base_oof = precision_at_2(base_pairs, vy, id_to_y)
        calibration[method] = dict(parent=parent, alpha=alpha, weights=weights.tolist(), outer_folds=history,
            full_fit_same_class=fit_score, oof_demo_precision_at_2=atom_oof,
            baseline_demo_precision_at_2=base_oof, independently_fit=True,
            selected_deployment=method if atom_oof > base_oof + 1e-12 else parent)
        emit("calibrated", dataset=dataset, method=method, alpha=alpha,
             oof_precision_at_2=atom_oof, baseline_precision_at_2=base_oof)
    cal = dict(dataset=dataset, shots=2, methods=calibration, folds=folds.tolist(), alpha_grid=list(ALPHAS),
        bank_hash=bank.bank_hash, inherited_weights=False, test_used=False,
        objective="validation same-class demo precision@2; never target-model accuracy")
    write(out / "calibration.json", cal)

    query, qe = load(dataset, "test"); qv = evidence(dataset, "test")
    rng = np.random.default_rng(SEED)
    methods = {m: [] for m in METHODS}
    for _ in query:
        a, b = rng.choice(len(train_ids), size=2, replace=False)
        methods["random"].append([str(train_ids[a]), str(train_ids[b])])
    for start in range(0, len(query), 64):
        held(); stop = min(start + 64, len(query))
        eb = EmbeddingStore(qe.sample_ids[start:stop], qe.embeddings[start:stop], qe.model_id, qe.model_revision)
        ev = cub.shared.sliced_evidence(qv, start, stop)
        for method, parent in PARENTS.items():
            text = label_vectors if parent == "muier" else np.zeros_like(label_vectors)
            ps = parent_scores(eb, te, ty, label_vectors if parent == "muier" else None)
            base = top2(ps)
            if not methods[parent]:
                pass
            methods[parent].extend([[str(train_ids[a]), str(train_ids[b])] for a, b in base])
            f, _ = cub.shared.class_features(eb, ev, te, ty, text, prototypes)
            w = np.asarray(calibration[method]["weights"], np.float32)
            classes = score_classes(f, w).argmax(1)
            methods[method].extend(pairs_for_classes(classes, ps, ty, train_ids))
        emit("planning", dataset=dataset, done=stop, total=len(query))
    validate_pair_plan(methods, qe.sample_ids, train_ids)
    plan = dict(dataset=dataset, split="test", shots=2, query_ids=list(qe.sample_ids), methods=methods,
        candidate_pool="complete frozen train split", random_without_replacement=True,
        atom_two_shot_rule="calibrated class prediction, then top-2 parent-score train samples within that class",
        test_labels_used_to_create_plan=False, selected_deployment={m: x["selected_deployment"] for m, x in calibration.items()})
    write(out / "test_plan.json", plan)
    input_files = [Path(__file__), s.bank, s.labels, s.prior_protocol, out / "calibration.json"]
    for split in ("train", "validation", "test"):
        input_files += [s.manifest(split), Path(str(s.embedding(split)) + ".json"), Path(str(s.embedding(split)) + ".npz"),
                        Path(str(s.evidence_path(split)) + ".json"), Path(str(s.evidence_path(split)) + ".npz")]
    write(marker, dict(files={str(p.resolve()): file_sha256(p) for p in input_files},
        calibration_sha256=file_sha256(out / "calibration.json"), plan_sha256=file_sha256(out / "test_plan.json")))
    unique = len({(q, tuple(pair)) for m in METHODS for q, pair in zip(qe.sample_ids, methods[m])})
    write(out / "workload.json", dict(queries=len(query), method_cells=len(query) * len(METHODS), unique_contexts=unique))
    emit("plan_frozen", dataset=dataset, queries=len(query), unique_contexts=unique)


def verify_dataset(dataset: str, require_complete: bool):
    out = OUT / dataset; marker = read(out / "plan.freeze.json")
    for name, expected in marker["files"].items():
        if file_sha256(name) != expected:
            raise ValueError("frozen input drift: " + name)
    if file_sha256(out / "calibration.json") != marker["calibration_sha256"] or file_sha256(out / "test_plan.json") != marker["plan_sha256"]:
        raise ValueError(dataset + " plan/calibration drift")
    train, te = load(dataset, "train"); query, _ = load(dataset, "test")
    plan = read(out / "test_plan.json")
    if plan["query_ids"] != [x.sample_id for x in query]:
        raise ValueError(dataset + " query order drift")
    validate_pair_plan(plan["methods"], plan["query_ids"], te.sample_ids)
    if require_complete:
        complete = read(out / "complete.json")
        for name, expected in complete["files"].items():
            if file_sha256(name) != expected:
                raise ValueError("complete output drift: " + name)


def conversation(instruction, query_image, demos, labels, codes):
    mapping = dict(zip(labels, codes))
    options = "\n".join(f"{c}: {label}" for c, label in zip(codes, labels))
    messages = [{"role": "system", "content": [{"type": "text", "text": instruction + "\nClass codes:\n" + options + "\nReturn only one class code."}]}]
    for image, label in demos:
        messages += [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": instruction}]},
            {"role": "assistant", "content": [{"type": "text", "text": mapping[label]}]},
        ]
    messages.append({"role": "user", "content": [{"type": "image", "image": query_image}, {"type": "text", "text": instruction}]})
    return messages


class Backend:
    def __init__(self):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(0.88, 0)
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, **PROCESSOR_SETTINGS)
        self.processor.tokenizer.padding_side = "left"
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, local_files_only=True,
            device_map={"": "cuda:0"}, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
        if any(p.device.type != "cuda" for p in self.model.parameters()):
            raise RuntimeError("Qwen3 CPU offload forbidden")

    def codes(self, labels, prior_protocol):
        prior = read(prior_protocol)
        if prior["labels"] != list(labels):
            raise ValueError("prior class order drift")
        codes = prior["codes"]
        generated, token_ids = single_token_codes(self.processor.tokenizer, len(labels))
        if generated != codes:
            raise ValueError("Qwen3 class-code policy drift")
        return codes, token_ids

    def forward(self, records, instruction, labels, codes, token_ids):
        torch = self.torch; opened = []; conversations = []
        started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        try:
            for row in records:
                demos = []
                for path, label in zip(row["demo_images"], row["demo_labels"]):
                    with Image.open(path) as src: image = src.convert("RGB")
                    opened.append(image); demos.append((image, label))
                with Image.open(row["query_image"]) as src: query = src.convert("RGB")
                opened.append(query)
                conversations.append(conversation(instruction, query, demos, labels, codes))
            inputs = self.processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt", processor_kwargs={"padding": True}).to(self.model.device, dtype=torch.bfloat16)
            with torch.inference_mode():
                logits = self.model(**inputs, logits_to_keep=1, use_cache=False).logits[:, -1, token_ids].float()
            if logits.shape != (len(records), len(labels)) or not torch.isfinite(logits).all():
                raise RuntimeError("invalid 2-shot logits")
            seconds = time.perf_counter() - started; values = logits.cpu().tolist(); chosen = logits.argmax(-1).cpu().tolist()
            lengths = inputs["attention_mask"].sum(-1).cpu().tolist()
            peak = torch.cuda.max_memory_allocated() / 2**30
            return [dict(prediction=labels[chosen[i]], logits=values[i], stats=dict(seconds=seconds / len(records),
                batch_seconds=seconds, execution_batch_size=len(records), batch_row=i, tokens=int(lengths[i]),
                padded_tokens=int(inputs["input_ids"].shape[1]), pixel_shape=list(inputs["pixel_values"].shape), peak_gib=peak))
                for i in range(len(records))]
        finally:
            for image in opened: image.close()


def preflight(backend: Backend):
    path = OUT / "preflight.json"
    if path.exists():
        value = read(path)
        if value.get("status") != "passed": raise ValueError("previous preflight did not pass")
        return
    train, _ = load("dtd", "train"); labels, _, _ = labels_and_y(train, [])
    codes, token_ids = backend.codes(labels, source("dtd").prior_protocol)
    rows = []
    for q, a, b in ((0, 1, 2), (3, 4, 5)):
        rows.append(dict(query_image=train[q].image_path, demo_images=[train[a].image_path, train[b].image_path],
                         demo_labels=[train[a].label, train[b].label]))
    output = backend.forward(rows, source("dtd").instruction, labels, codes, token_ids)
    write(path, dict(status="passed", train_only=True, batch_size=2, shots=2,
        peak_gib=max(x["stats"]["peak_gib"] for x in output), batch_seconds=output[0]["stats"]["batch_seconds"],
        predictions=[x["prediction"] for x in output], model=str(MODEL)))
    emit("preflight_passed", **read(path))


def image_area(path, memo):
    if path not in memo:
        with Image.open(path) as image: memo[path] = image.width * image.height
    return memo[path]


def evaluate_dataset(dataset: str, backend: Backend):
    held(); verify_dataset(dataset, False); out = OUT / dataset
    if (out / "complete.json").exists():
        verify_dataset(dataset, True); return
    train_rows, _ = load(dataset, "train"); query_rows, _ = load(dataset, "test")
    train = {x.sample_id: x for x in train_rows}; query = {x.sample_id: x for x in query_rows}
    labels = tuple(sorted({x.label for x in train_rows}))
    codes, token_ids = backend.codes(labels, source(dataset).prior_protocol)
    plan = read(out / "test_plan.json")
    protocol = dict(model="Qwen/Qwen3-VL-4B-Instruct", model_path=str(MODEL), dtype="bfloat16", attention="sdpa",
        processor_settings=PROCESSOR_SETTINGS, shots=2, execution_batch_size=2,
        conversation_version="qwen3-cleanroom-2shot-v1", classification="constrained next-token single-token class-code logits",
        labels=list(labels), codes=codes, token_ids=token_ids, instruction=source(dataset).instruction,
        plan_sha256=file_sha256(out / "test_plan.json"), memory_fraction=0.88, use_cache=False)
    protocol_hash = stable_hash(protocol); protocol["contract_sha256"] = protocol_hash
    pp = out / "inference_protocol.json"
    if pp.exists() and read(pp) != protocol: raise ValueError(dataset + " inference protocol drift")
    write(pp, protocol)
    cache_path = out / "context_cache.jsonl"; cache = {}
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip(): continue
                row = json.loads(line)
                if row["protocol_hash"] != protocol_hash: raise ValueError("foreign 2-shot cache")
                if row["key"] in cache and cache[row["key"]] != row: raise ValueError("conflicting 2-shot cache")
                cache[row["key"]] = row
    keys = {}; records = {}; sizes = {}
    for priority, method in enumerate(METHODS):
        for qid, demo_ids in zip(plan["query_ids"], plan["methods"][method]):
            q = query[qid]; ds = [train[x] for x in demo_ids]
            key = stable_hash([protocol_hash, qid, demo_ids, q.image_sha256, [x.image_sha256 for x in ds]])
            keys[(method, qid)] = key
            if key in cache: continue
            if key not in records:
                records[key] = dict(key=key, query_id=qid, demo_ids=demo_ids, query_image=q.image_path,
                    demo_images=[x.image_path for x in ds], demo_labels=[x.label for x in ds], priority=priority)
            else: records[key]["priority"] = min(priority, records[key]["priority"])
    for row in records.values():
        row["estimated_pixels"] = image_area(row["query_image"], sizes) + sum(image_area(x, sizes) for x in row["demo_images"])
    planned = sorted(records.values(), key=lambda x: (x["estimated_pixels"], x["priority"], x["query_id"]))
    emit("target_prepared", dataset=dataset, unique_total=len(set(keys.values())), cached=len(cache), new=len(planned))
    offset = 0; batch_size = 2; started = time.perf_counter(); batches = 0
    while offset < len(planned):
        held(); batch = planned[offset:offset + batch_size]
        try:
            outputs = backend.forward(batch, source(dataset).instruction, labels, codes, token_ids)
        except backend.torch.OutOfMemoryError:
            backend.torch.cuda.empty_cache()
            if batch_size == 1: raise
            batch_size = 1
            emit("oom_batch_fallback", dataset=dataset, offset=offset, batch_size=1)
            continue
        rows = []
        for record, output in zip(batch, outputs):
            row = dict(key=record["key"], protocol_hash=protocol_hash, query_id=record["query_id"],
                demo_ids=record["demo_ids"], prediction=output["prediction"], logits=output["logits"], stats=output["stats"])
            cache[row["key"]] = row; rows.append(row)
        append_rows(cache_path, rows); offset += len(batch); batches += 1
        if offset == len(planned) or offset % 50 < len(batch):
            elapsed = time.perf_counter() - started; rate = offset / max(elapsed, 1e-9) * 60
            emit("target_running", dataset=dataset, done=offset, new_total=len(planned), cached_total=len(cache),
                 batch_size=batch_size, rate_per_minute=rate, eta_minutes=(len(planned) - offset) / max(rate, 1e-9))
    predictions = {m: {q: cache[keys[(m, q)]]["prediction"] for q in plan["query_ids"]} for m in METHODS}
    truth = {x.sample_id: x.label for x in query_rows}; results = {}
    for method in METHODS:
        values = [predictions[method][q] for q in truth]
        metrics = accuracy_metrics(list(truth.values()), values)
        demo_precision = np.mean([sum(train[d].label == truth[q] for d in pair) / 2
                                  for q, pair in zip(plan["query_ids"], plan["methods"][method])])
        results[method] = {**metrics, "demo_same_class_precision_at_2": float(demo_precision)}
        if method in PARENTS:
            results[method]["paired_vs_parent"] = paired(predictions[PARENTS[method]], predictions[method], truth)
    write(out / "predictions.json", predictions)
    write(out / "results.json", dict(status="complete", dataset=dataset, shots=2, methods=results,
        test_used_for_tuning=False, inference=dict(new_contexts=len(planned), total_contexts=len(cache), batches=batches,
        elapsed_seconds=time.perf_counter() - started, final_batch_size=batch_size)))
    files = [out / "plan.freeze.json", pp, cache_path, out / "predictions.json", out / "results.json"]
    write(out / "complete.json", dict(files={str(p.resolve()): file_sha256(p) for p in files}))
    emit("dataset_complete", dataset=dataset, methods=results)


def report():
    rows = []
    for dataset in ORDER:
        verify_dataset(dataset, True)
        result = read(OUT / dataset / "results.json")
        for method in METHODS:
            v = result["methods"][method]
            rows.append((dataset, method, v["accuracy"], v["macro_accuracy"], v["demo_same_class_precision_at_2"]))
    lines = ["# Qwen3-VL-4B 2-shot 五数据集结果", "",
        "BF16 / SDPA；完整 test；每个数据集、每个 parent 独立 validation 重校准；test 不参与调参。", "",
        "| Dataset | Method | Accuracy | Macro | Demo P@2 |", "|---|---|---:|---:|---:|"]
    for ds, method, acc, macro, p2 in rows:
        lines.append(f"| {ds} | {method} | {acc*100:.3f}% | {macro*100:.3f}% | {p2*100:.3f}% |")
    lines += ["", "Atom 2-shot 规则：先以本数据集独立校准的 class-equal 分数选类，再在该类的完整训练池中按 parent 分数取前两张不同图。",
              "报告中的配对 rescue/harm 与 exact McNemar 数据保存在各数据集 results.json。"]
    (OUT / "report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    files = [OUT / "report_zh.md"] + [OUT / ds / "complete.json" for ds in ORDER]
    write(OUT / "complete.json", dict(files={str(p.resolve()): file_sha256(p) for p in files}))
    emit("queue_complete", datasets=list(ORDER))


def verify_all():
    for dataset in ORDER:
        verify_dataset(dataset, require_complete=(OUT / dataset / "complete.json").exists())
    emit("resume_verified", datasets=list(ORDER))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run", "all", "verify", "report"))
    parser.add_argument("--datasets", nargs="+", choices=ORDER)
    args = parser.parse_args(); datasets = tuple(args.datasets or ORDER)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("prepare", "all"):
        for dataset in datasets: prepare_dataset(dataset)
    if args.stage == "verify": verify_all(); return
    if args.stage in ("run", "all"):
        for dataset in datasets: verify_dataset(dataset, False)
        backend = Backend(); preflight(backend)
        for dataset in datasets: evaluate_dataset(dataset, backend)
    if args.stage in ("report", "all") and all((OUT / ds / "complete.json").exists() for ds in ORDER):
        report()


if __name__ == "__main__":
    main()

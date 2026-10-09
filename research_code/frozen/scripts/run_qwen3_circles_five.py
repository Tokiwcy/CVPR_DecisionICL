"""CIRCLES 1+1 retrieval on the frozen five-dataset Qwen3 suite."""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
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

import run_qwen3_2shot_five as base
from dgdr.atom_label import build_class_atom_prototypes, score_classes
from dgdr.calibration import stratified_folds
from dgdr.hashing import file_sha256, stable_hash
from dgdr.io import load_manifest
from dgdr.metrics import accuracy_metrics

OUT = PROJECT / "artifacts/qwen3_circles_five_20260909"
TWO_SHOT = PROJECT / "artifacts/qwen3_2shot_five_20260908"
OFFICIAL = REPO / "external/CIRCLES"
OFFICIAL_SOURCE = OFFICIAL / "src/visual_icl.py"
OFFICIAL_COMMIT = "ee14d4f64565a961c4360bb669eecb6c7bb6668a"
CLIP_MODEL = REPO / "models/CLIP-ViT-g-14-laion2B-s12B-b42K"
ORDER = base.ORDER
DESCRIPTION_BATCH = 2
IMAGE_BATCH = 8
TEXT_BATCH = 64
METHODS = ("circles", "circles_atom_class_equal_raw")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    base.write(path, value)


def emit(stage, **values):
    row = dict(stage=stage, time=time.time(), **values)
    write(OUT / "status.json", row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def held():
    if any((OUT / name).exists() for name in ("PAUSED_BY_USER.json", "WATCHDOG_STOP", "DIAGNOSTIC_HOLD.json")):
        raise RuntimeError("CIRCLES queue held by control file")


def source(dataset):
    return base.source(dataset)


def samples(dataset, split):
    return load_manifest(source(dataset).manifest(split), expected_split=split)


def official_instruction():
    tree = ast.parse(OFFICIAL_SOURCE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "INSTRUCTION" for x in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("official CIRCLES prompt missing")


def validate_inputs(require_two_shot_complete=False):
    commit = subprocess.check_output(["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"], text=True).strip()
    if commit != OFFICIAL_COMMIT:
        raise ValueError("official CIRCLES commit drift")
    if not CLIP_MODEL.is_dir():
        raise FileNotFoundError(CLIP_MODEL)
    counts = {}
    for dataset in ORDER:
        train = samples(dataset, "train")
        validation = samples(dataset, "validation")
        test = samples(dataset, "test")
        labels = sorted({row.label for row in train})
        if read(source(dataset).prior_protocol)["labels"] != labels:
            raise ValueError(f"{dataset} target label protocol drift")
        if len({row.question for row in train}) > 1:
            raise ValueError(f"{dataset} train questions are not constant")
        if require_two_shot_complete:
            base.verify_dataset(dataset, True)
        counts[dataset] = {"train": len(train), "validation": len(validation),
                           "test": len(test), "classes": len(labels)}
    if require_two_shot_complete and not (TWO_SHOT / "complete.json").is_file():
        raise RuntimeError("2-shot root report is incomplete")
    value = {
        "official_commit": commit,
        "official_source_sha256": file_sha256(OFFICIAL_SOURCE),
        "clip_model": str(CLIP_MODEL.resolve()),
        "datasets": counts,
        "total_test_queries": sum(x["test"] for x in counts.values()),
    }
    write(OUT / "input_validation.json", value)
    emit("inputs_validated", **value)
    return value


def parse_attribute(text):
    for raw in text.split("### Attributes")[-1].strip().splitlines():
        value = raw.strip().lstrip("-*• ").strip()
        while value and value[0].isdigit():
            value = value[1:].lstrip(".):- ")
        if value:
            return value[:240]
    return ""


def parse_caption(text):
    value = text.strip()
    if "Target Image Description" in value:
        value = value.split("Target Image Description")[-1]
    return value.strip('\":} \n\r\t')[:1200]


class Generator:
    def __init__(self):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(0.88, 0)
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(base.MODEL, local_files_only=True, **base.PROCESSOR_SETTINGS)
        self.processor.tokenizer.padding_side = "left"
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            base.MODEL, local_files_only=True, device_map={"": "cuda:0"},
            dtype=torch.bfloat16, attn_implementation="sdpa").eval()

    def generate(self, contents, max_new_tokens=512):
        opened, conversations = [], []
        try:
            for image_path, texts in contents:
                with Image.open(image_path) as src:
                    image = src.convert("RGB")
                opened.append(image)
                content = [{"type": "image", "image": image} if text == "<IMAGE>" else
                           {"type": "text", "text": text} for text in texts]
                conversations.append([{"role": "user", "content": content}])
            inputs = self.processor.apply_chat_template(
                conversations, add_generation_prompt=True, tokenize=True, return_dict=True,
                return_tensors="pt", processor_kwargs={"padding": True}
            ).to(self.model.device, dtype=self.torch.bfloat16)
            width = int(inputs["input_ids"].shape[1])
            with self.torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
            return self.processor.batch_decode(
                output[:, width:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        finally:
            for image in opened:
                image.close()


def description_path(dataset, split):
    return OUT / dataset / f"counterfactual_descriptions.{split}.jsonl"


def description_marker(dataset, split):
    return OUT / dataset / f"description.{split}.complete.json"


def describe_dataset(dataset, split, generator):
    held()
    out = OUT / dataset
    out.mkdir(parents=True, exist_ok=True)
    path = description_path(dataset, split)
    existing = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line); existing[row["query_id"]] = row
    queries = samples(dataset, split)
    pending = [row for row in queries if row.sample_id not in existing]
    question = source(dataset).instruction
    instruction = official_instruction()
    started = time.perf_counter(); offset = 0; batch_size = DESCRIPTION_BATCH
    while offset < len(pending):
        held(); batch = pending[offset:offset + batch_size]
        try:
            attribute_raw = generator.generate([
                (row.image_path, [
                    "Identify the key attributes of the following image that are most relevant to answering the question.",
                    "<IMAGE>", f"Question: {question}",
                    "Please list the top 1 key attributes as short phrases in a section named '### Attributes', one per line, ordered from most to least important."
                ]) for row in batch])
            attributes = [parse_attribute(x) for x in attribute_raw]
            if any(not x for x in attributes):
                raise ValueError("empty CIRCLES attribute")
            caption_raw = generator.generate([
                (row.image_path, [instruction, "<IMAGE>",
                 f"Manipulation Text: Change the attribute '{attribute}' to a different plausible value. Ensure the modified caption is concise and contains no more than 77 tokens."])
                for row, attribute in zip(batch, attributes)])
        except generator.torch.OutOfMemoryError:
            generator.torch.cuda.empty_cache()
            if batch_size == 1:
                raise
            batch_size = 1; emit("description_oom_fallback", dataset=dataset, batch_size=1); continue
        rows = []
        for sample, ar, attribute, cr in zip(batch, attribute_raw, attributes, caption_raw):
            caption = parse_caption(cr)
            if not caption:
                raise ValueError(f"empty CIRCLES caption: {dataset}/{sample.sample_id}")
            rows.append({"query_id": sample.sample_id, "attribute": attribute,
                         "modified_caption": caption, "attribute_response": ar,
                         "caption_response": cr, "test_label_used": False})
        base.append_rows(path, rows); offset += len(batch)
        if offset % 20 < len(batch) or offset == len(pending):
            rate = offset / max(time.perf_counter() - started, 1e-9) * 60
            emit("descriptions_running", dataset=dataset, split=split, done=len(existing) + offset,
                 total=len(queries), batch_size=batch_size, rate_per_minute=rate,
                 eta_minutes=(len(pending) - offset) / max(rate, 1e-9))
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(rows) != len(queries) or {x["query_id"] for x in rows} != {x.sample_id for x in queries}:
        raise ValueError(f"{dataset}/{split} description cache incomplete")
    write(description_marker(dataset, split), {"count": len(rows), "sha256": file_sha256(path),
          "official_prompt_sha256": hashlib.sha256(instruction.encode()).hexdigest(), "test_labels_used": False})
    emit("descriptions_complete", dataset=dataset, split=split, count=len(rows))


def describe_all():
    generator = Generator()
    try:
        for dataset in ORDER:
            for split in ("validation", "test"):
                marker = description_marker(dataset, split)
                if marker.is_file():
                    if read(marker)["sha256"] != file_sha256(description_path(dataset, split)):
                        raise ValueError(f"{dataset}/{split} description receipt drift")
                    continue
                describe_dataset(dataset, split, generator)
    finally:
        del generator; gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass


def normalized(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-12, None)


def clip_feature_tensor(value, torch):
    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        tensor = None
        for attribute in ("image_embeds", "text_embeds", "pooler_output"):
            candidate = getattr(value, attribute, None)
            if isinstance(candidate, torch.Tensor):
                tensor = candidate
                break
        if tensor is None:
            raise TypeError(f"unsupported CLIP feature output: {type(value).__name__}")
    if not torch.isfinite(tensor).all():
        raise ValueError("non-finite CLIP feature")
    return tensor


def save_embeddings(path, ids, values, kind):
    np.savez_compressed(path, sample_ids=np.asarray(ids, dtype=str), values=normalized(values))
    write(Path(str(path) + ".json"), {
        "kind": kind, "count": len(ids),
        "model": "laion/CLIP-ViT-g-14-laion2B-s12B-b42K",
        "model_path": str(CLIP_MODEL.resolve()), "npz_sha256": file_sha256(path)})


def valid_embedding(path):
    meta = Path(str(path) + ".json")
    return path.is_file() and meta.is_file() and read(meta)["npz_sha256"] == file_sha256(path)


def embed_all():
    import torch
    from transformers import CLIPModel, CLIPProcessor
    torch.set_num_threads(4)
    model = CLIPModel.from_pretrained(CLIP_MODEL, local_files_only=True).to("cuda:0").eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL, local_files_only=True)
    try:
        for dataset in ORDER:
            held(); out = OUT / dataset
            if not all(description_marker(dataset, split).is_file() for split in ("validation", "test")):
                raise RuntimeError(f"{dataset} validation/test descriptions incomplete")
            for split in ("train", "validation", "test"):
                path = out / f"clip.{split}.npz"
                if valid_embedding(path):
                    continue
                rows = samples(dataset, split); ids, chunks = [], []
                started = time.perf_counter()
                for begin in range(0, len(rows), IMAGE_BATCH):
                    held(); batch = rows[begin:begin + IMAGE_BATCH]; images = []
                    try:
                        for row in batch:
                            with Image.open(row.image_path) as src:
                                images.append(src.convert("RGB"))
                        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to("cuda:0")
                        with torch.inference_mode():
                            features = model.get_image_features(pixel_values=pixel_values)
                        features = clip_feature_tensor(features, torch)
                        chunks.append(features.float().cpu().numpy()); ids.extend(x.sample_id for x in batch)
                    finally:
                        for image in images:
                            image.close()
                    done = min(begin + len(batch), len(rows))
                    if done % 200 < len(batch) or done == len(rows):
                        rate = done / max(time.perf_counter() - started, 1e-9) * 60
                        emit("clip_images_running", dataset=dataset, split=split, done=done,
                             total=len(rows), batch_size=IMAGE_BATCH, rate_per_minute=rate,
                             eta_minutes=(len(rows) - done) / max(rate, 1e-9))
                save_embeddings(path, ids, np.concatenate(chunks), f"image/{split}")
            for split in ("validation", "test"):
                caption_path = out / f"clip.captions.{split}.npz"
                if valid_embedding(caption_path):
                    continue
                descriptions = [json.loads(x) for x in description_path(dataset, split).read_text(
                    encoding="utf-8").splitlines() if x.strip()]
                ids, chunks = [], []
                for begin in range(0, len(descriptions), TEXT_BATCH):
                    held(); batch = descriptions[begin:begin + TEXT_BATCH]
                    inputs = processor(text=[x["modified_caption"] for x in batch], padding=True,
                                       truncation=True, max_length=77, return_tensors="pt").to("cuda:0")
                    with torch.inference_mode():
                        features = model.get_text_features(input_ids=inputs["input_ids"],
                                                          attention_mask=inputs["attention_mask"])
                    features = clip_feature_tensor(features, torch)
                    chunks.append(features.float().cpu().numpy()); ids.extend(x["query_id"] for x in batch)
                save_embeddings(caption_path, ids, np.concatenate(chunks), f"modified_caption/{split}")
            emit("clip_dataset_complete", dataset=dataset)
    finally:
        del model, processor; gc.collect(); torch.cuda.empty_cache()


def load_embeddings(path, expected_ids):
    if not valid_embedding(path):
        raise ValueError(f"embedding receipt drift: {path}")
    with np.load(path, allow_pickle=False) as bundle:
        ids = [str(x) for x in bundle["sample_ids"]]
        values = np.asarray(bundle["values"], dtype=np.float32)
    if ids != expected_ids:
        raise ValueError(f"embedding order drift: {path}")
    return values


def last_argmax(scores):
    return scores.shape[1] - 1 - np.argmax(scores[:, ::-1], axis=1)


def circle_class_data(original_scores, composed_scores, train_y, class_count):
    rows = np.arange(len(original_scores)); base_scores = np.empty((len(rows), class_count), np.float32)
    original_best = np.empty((len(rows), class_count), np.int64)
    composed_best = np.empty((len(rows), class_count), np.int64)
    for class_index in range(class_count):
        candidates = np.flatnonzero(train_y == class_index)
        oi = candidates[last_argmax(original_scores[:, candidates])]
        ci = candidates[last_argmax(composed_scores[:, candidates])]
        original_best[:, class_index] = oi; composed_best[:, class_index] = ci
        base_scores[:, class_index] = 0.5 * (original_scores[rows, oi] + composed_scores[rows, ci])
    return base_scores, original_best, composed_best


def circle_atom_features(original_scores, composed_scores, train_y, query_evidence, prototypes, class_count):
    base_scores, original_best, composed_best = circle_class_data(
        original_scores, composed_scores, train_y, class_count)
    atoms = base.cub.shared.jensen_shannon_group_similarities(
        query_evidence=query_evidence, prototypes=prototypes)
    raw = np.concatenate((base_scores[:, :, None], atoms), axis=2)
    standardized, _, _ = base.cub.shared.standardize_across_classes(raw)
    return standardized, original_best, composed_best


def pair_precision(pairs, truth_y, train_y):
    return float(np.mean([int(train_y[a] == y) + int(train_y[b] == y)
                          for (a, b), y in zip(pairs, truth_y)]) / 2)


def select_dataset(dataset):
    held(); out = OUT / dataset
    train = samples(dataset, "train"); validation = samples(dataset, "validation"); test = samples(dataset, "test")
    train_ids = [x.sample_id for x in train]; validation_ids = [x.sample_id for x in validation]
    test_ids = [x.sample_id for x in test]
    labels = tuple(sorted({x.label for x in train})); lookup = {label: i for i, label in enumerate(labels)}
    train_y = np.asarray([lookup[x.label] for x in train], np.int64)
    validation_y = np.asarray([lookup[x.label] for x in validation], np.int64)
    train_embeddings = load_embeddings(out / "clip.train.npz", train_ids)
    validation_embeddings = load_embeddings(out / "clip.validation.npz", validation_ids)
    validation_captions = load_embeddings(out / "clip.captions.validation.npz", validation_ids)
    train_evidence = base.evidence(dataset, "train")
    validation_evidence = base.evidence(dataset, "validation")
    prototypes = build_class_atom_prototypes(samples=train, evidence=train_evidence)
    if prototypes.labels != labels:
        raise ValueError(f"{dataset} prototype label drift")

    validation_original = validation_embeddings @ train_embeddings.T
    validation_composed = validation_captions @ train_embeddings.T
    features, validation_original_best, validation_composed_best = circle_atom_features(
        validation_original, validation_composed, train_y, validation_evidence, prototypes, len(labels))
    folds = stratified_folds(validation, 5, 42); oof = np.empty(len(validation), np.int64); history = []
    for fold in range(5):
        fit = np.flatnonzero(folds != fold); hold = np.flatnonzero(folds == fold)
        alpha, score = base.fit_alpha(features, validation_y, fit)
        weights = base.cub.shared.equal_weights(features.shape[2], alpha)
        oof[hold] = score_classes(features[hold], weights).argmax(1)
        history.append({"fold": fold, "alpha": alpha, "weights": weights.tolist(),
                        "fit_class_accuracy": score, "fit_count": len(fit), "heldout_count": len(hold)})
    alpha, fit_score = base.fit_alpha(features, validation_y, np.arange(len(validation_y)))
    weights = base.cub.shared.equal_weights(features.shape[2], alpha)
    parent_pairs = list(zip(last_argmax(validation_original), last_argmax(validation_composed)))
    atom_pairs = [(validation_original_best[i, cls], validation_composed_best[i, cls])
                  for i, cls in enumerate(oof)]
    parent_precision = pair_precision(parent_pairs, validation_y, train_y)
    atom_precision = pair_precision(atom_pairs, validation_y, train_y)
    calibration = {"dataset": dataset, "method": "circles_atom_class_equal_raw", "alpha": alpha,
        "weights": weights.tolist(), "outer_folds": history, "full_fit_class_accuracy": fit_score,
        "oof_demo_precision_at_2": atom_precision, "parent_demo_precision_at_2": parent_precision,
        "selected_deployment": "circles_atom_class_equal_raw" if atom_precision > parent_precision + 1e-12 else "circles",
        "base_class_signal": "equal mean of per-class maxima for original-image and composed-caption channels",
        "independently_fit": True, "test_used": False}
    write(out / "calibration.json", calibration)

    test_embeddings = load_embeddings(out / "clip.test.npz", test_ids)
    test_captions = load_embeddings(out / "clip.captions.test.npz", test_ids)
    descriptions = {x["query_id"]: x for x in (json.loads(line) for line in
        description_path(dataset, "test").read_text(encoding="utf-8").splitlines() if line.strip())}
    test_evidence = base.evidence(dataset, "test"); rows = []
    for begin in range(0, len(test), 128):
        stop = min(begin + 128, len(test))
        original_scores = test_embeddings[begin:stop] @ train_embeddings.T
        composed_scores = test_captions[begin:stop] @ train_embeddings.T
        block_evidence = base.cub.shared.sliced_evidence(test_evidence, begin, stop)
        block_features, original_best, composed_best = circle_atom_features(
            original_scores, composed_scores, train_y, block_evidence, prototypes, len(labels))
        atom_classes = score_classes(block_features, weights).argmax(1)
        original_parent = last_argmax(original_scores); composed_parent = last_argmax(composed_scores)
        for local, query_index in enumerate(range(begin, stop)):
            query_id = test_ids[query_index]; predicted_class = int(atom_classes[local])
            circle_pair = [int(original_parent[local]), int(composed_parent[local])]
            atom_pair = [int(original_best[local, predicted_class]), int(composed_best[local, predicted_class])]
            rows.append({"query_id": query_id, "methods": {
                "circles": {"demo_ids": [train_ids[x] for x in circle_pair]},
                "circles_atom_class_equal_raw": {"demo_ids": [train_ids[x] for x in atom_pair],
                    "predicted_class": labels[predicted_class]}},
                "demo_sources": ["original_image_retrieval", "counterfactual_composed_retrieval"],
                "attribute": descriptions[query_id]["attribute"],
                "modified_caption": descriptions[query_id]["modified_caption"], "test_label_used": False})
        emit("selection_running", dataset=dataset, done=stop, total=len(test))
    plan_path = out / "test_plan.jsonl"
    with plan_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    inputs = [Path(__file__), source(dataset).manifest("train"), source(dataset).manifest("validation"),
              source(dataset).manifest("test"), source(dataset).prior_protocol,
              description_path(dataset, "validation"), description_path(dataset, "test"),
              out / "clip.train.npz", out / "clip.validation.npz", out / "clip.test.npz",
              out / "clip.captions.validation.npz", out / "clip.captions.test.npz",
              out / "calibration.json", OFFICIAL_SOURCE]
    write(out / "plan.freeze.json", {"files": {str(path.resolve()): file_sha256(path) for path in inputs},
        "plan_sha256": file_sha256(plan_path), "protocol": {
            "methods": ["circles", "circles_atom_class_equal_raw"], "official_commit": OFFICIAL_COMMIT,
            "official_clip": "laion/CLIP-ViT-g-14-laion2B-s12B-b42K", "total_images": 2,
            "budget": "1 original image retrieval + 1 composed-description retrieval",
            "candidate_pool": "complete frozen train split", "official_order": "original then composed",
            "official_no_deduplication": True,
            "atom_rule": "independently calibrated class-equal raw class decision, then one best demo per CIRCLES channel within predicted class",
            "classification_task_text_term": "ranking-constant and omitted because all train questions are identical",
            "target_model": "Qwen/Qwen3-VL-4B-Instruct",
            "target_prompt": "same constrained class-code prompt as qwen3-cleanroom-2shot-v1",
            "test_labels_used_for_description_selection_or_calibration": False,
            "adaptation": "official CIRCLES flow extended to five frozen classification datasets"}})
    emit("selection_complete", dataset=dataset, queries=len(rows), methods=2,
         atom_alpha=alpha, atom_oof_demo_precision=atom_precision, parent_demo_precision=parent_precision)


def verify_plan(dataset):
    out = OUT / dataset; freeze = read(out / "plan.freeze.json")
    for name, expected in freeze["files"].items():
        if file_sha256(name) != expected:
            raise ValueError(f"{dataset} frozen input drift: {name}")
    if file_sha256(out / "test_plan.jsonl") != freeze["plan_sha256"]:
        raise ValueError(f"{dataset} plan drift")
    return freeze


def frozen_codes(backend, labels, prior_protocol):
    prior = read(prior_protocol)
    if prior["labels"] != list(labels):
        raise ValueError("prior class order drift")
    codes = prior["codes"]
    encoded = [backend.processor.tokenizer.encode(code, add_special_tokens=False) for code in codes]
    if any(len(token) != 1 for token in encoded):
        raise ValueError("frozen class code is no longer a single token")
    token_ids = [token[0] for token in encoded]
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("frozen class-code tokens are not unique")
    if any(backend.processor.tokenizer.decode([token]) != code for code, token in zip(codes, token_ids)):
        raise ValueError("frozen class code no longer round-trips")
    return codes, token_ids


def infer_dataset(dataset, backend):
    held(); out = OUT / dataset
    if (out / "complete.json").is_file():
        return
    freeze = verify_plan(dataset)
    train_rows, query_rows = samples(dataset, "train"), samples(dataset, "test")
    train = {x.sample_id: x for x in train_rows}; query = {x.sample_id: x for x in query_rows}
    plan = [json.loads(line) for line in (out / "test_plan.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if [x["query_id"] for x in plan] != [x.sample_id for x in query_rows]:
        raise ValueError(f"{dataset} query order drift")
    labels = tuple(sorted({x.label for x in train_rows}))
    codes, token_ids = frozen_codes(backend, labels, source(dataset).prior_protocol)
    protocol = {
        "model": "Qwen/Qwen3-VL-4B-Instruct", "model_path": str(base.MODEL),
        "dtype": "bfloat16", "attention": "sdpa", "processor_settings": base.PROCESSOR_SETTINGS,
        "shots": 2, "execution_batch_size": 2, "conversation_version": "qwen3-cleanroom-2shot-v1",
        "classification": "constrained next-token single-token class-code logits",
        "retrieval_methods": list(METHODS),
        "labels": list(labels), "codes": codes, "token_ids": token_ids,
        "instruction": source(dataset).instruction, "plan_sha256": freeze["plan_sha256"], "use_cache": False}
    protocol_hash = stable_hash(protocol); protocol["contract_sha256"] = protocol_hash
    write(out / "inference_protocol.json", protocol)
    cache_path = out / "context_cache.jsonl"; cache = {}
    if cache_path.is_file():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["protocol_hash"] != protocol_hash:
                raise ValueError("foreign CIRCLES target cache")
            cache[row["key"]] = row
    records, key_by_method_query, areas = [], {method: {} for method in METHODS}, {}
    pending_keys = set()
    for row in plan:
        q = query[row["query_id"]]
        for method in METHODS:
            demo_ids = row["methods"][method]["demo_ids"]
            demos = [train[x] for x in demo_ids]
            key = stable_hash([protocol_hash, q.sample_id, demo_ids, q.image_sha256,
                               [demo.image_sha256 for demo in demos]])
            key_by_method_query[method][q.sample_id] = key
            if key not in cache and key not in pending_keys:
                records.append({"key": key, "query_id": q.sample_id, "demo_ids": demo_ids,
                    "query_image": q.image_path, "demo_images": [x.image_path for x in demos],
                    "demo_labels": [x.label for x in demos],
                    "estimated_pixels": base.image_area(q.image_path, areas)
                        + sum(base.image_area(x.image_path, areas) for x in demos)})
                pending_keys.add(key)
    records.sort(key=lambda x: (x["estimated_pixels"], x["query_id"], x["key"]))
    emit("target_prepared", dataset=dataset, queries=len(plan), methods=len(METHODS),
         requested_contexts=len(plan) * len(METHODS), cached_unique=len(cache), new_unique=len(records))
    offset = 0; batch_size = 2; batches = 0; started = time.perf_counter()
    while offset < len(records):
        held(); batch = records[offset:offset + batch_size]
        try:
            outputs = backend.forward(batch, source(dataset).instruction, labels, codes, token_ids)
        except backend.torch.OutOfMemoryError:
            backend.torch.cuda.empty_cache()
            if batch_size == 1:
                raise
            batch_size = 1; emit("target_oom_fallback", dataset=dataset, batch_size=1); continue
        new_rows = []
        for record, output in zip(batch, outputs):
            row = {"key": record["key"], "protocol_hash": protocol_hash,
                   "query_id": record["query_id"], "demo_ids": record["demo_ids"],
                   "prediction": output["prediction"], "logits": output["logits"], "stats": output["stats"]}
            cache[row["key"]] = row; new_rows.append(row)
        base.append_rows(cache_path, new_rows); offset += len(batch); batches += 1
        if offset % 50 < len(batch) or offset == len(records):
            rate = offset / max(time.perf_counter() - started, 1e-9) * 60
            emit("target_running", dataset=dataset, done=offset, new_total=len(records),
                 cached_total=len(cache), batch_size=batch_size, rate_per_minute=rate,
                 eta_minutes=(len(records) - offset) / max(rate, 1e-9))
    predictions = {method: {qid: cache[key]["prediction"]
                            for qid, key in key_by_method_query[method].items()}
                   for method in METHODS}
    truth = {x.sample_id: x.label for x in query_rows}
    rices = read(TWO_SHOT / dataset / "predictions.json")["rices"]
    plan_by_id = {x["query_id"]: x for x in plan}
    method_results = {}
    for method in METHODS:
        method_predictions = predictions[method]
        method_results[method] = {
            "metrics": accuracy_metrics([truth[x] for x in method_predictions],
                                        [method_predictions[x] for x in method_predictions]),
            "demo_same_class_precision_at_2": float(np.mean([
                sum(train[demo].label == truth[qid]
                    for demo in plan_by_id[qid]["methods"][method]["demo_ids"]) / 2
                for qid in truth])),
            "paired_vs_rices_2shot": base.paired(rices, method_predictions, truth)}
    method_results["circles_atom_class_equal_raw"]["paired_vs_circles"] = base.paired(
        predictions["circles"], predictions["circles_atom_class_equal_raw"], truth)
    result = {"status": "complete", "dataset": dataset, "methods": method_results, "shots": 2,
              "test_used_for_tuning": False,
              "inference": {"new_contexts": len(records), "total_contexts": len(cache),
                            "batches": batches, "elapsed_seconds": time.perf_counter() - started,
                            "final_batch_size": batch_size}}
    write(out / "predictions.json", predictions); write(out / "results.json", result)
    files = [out / "plan.freeze.json", out / "inference_protocol.json", cache_path,
             out / "predictions.json", out / "results.json"]
    write(out / "complete.json", {"files": {str(path.resolve()): file_sha256(path) for path in files}})
    emit("dataset_complete", dataset=dataset, result=result)


def infer_all():
    backend = base.Backend()
    try:
        for dataset in ORDER:
            infer_dataset(dataset, backend)
    finally:
        del backend; gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass


def report():
    rows = [(dataset, read(OUT / dataset / "results.json")) for dataset in ORDER]
    lines = ["# Qwen3-VL-4B CIRCLES 系列五数据集结果", "",
        "官方 CIRCLES 属性识别、反事实描述与 CLIP ViT-g/14 检索流程；每个 query 使用 1 个原图检索 demo 和 1 个 composed-description demo。CIRCLES+Atom 类别等权在 validation 上独立学习 Atom 总权重，预测类别后在该类内分别选择两个通道的最佳 demo。目标推理沿用相同 Qwen3 2-shot 约束分类 prompt。", "",
        "| Dataset | Method | Acc. | Macro | Demo P@2 | Δ vs RICES | Rescue | Harm | Δ vs CIRCLES | Rescue/Harm vs CIRCLES |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for dataset, result in rows:
        for method in METHODS:
            item = result["methods"][method]; metrics = item["metrics"]
            paired = item["paired_vs_rices_2shot"]
            parent = item.get("paired_vs_circles")
            parent_delta = "—" if parent is None else f"{parent['accuracy_delta']*100:+.3f} pp"
            parent_counts = "—" if parent is None else f"{parent['rescue']}/{parent['harm']}"
            lines.append(f"| {dataset} | {method} | {metrics['accuracy']*100:.3f}% | "
                         f"{metrics['macro_accuracy']*100:.3f}% | "
                         f"{item['demo_same_class_precision_at_2']*100:.3f}% | "
                         f"{paired['accuracy_delta']*100:+.3f} pp | {paired['rescue']} | {paired['harm']} | "
                         f"{parent_delta} | {parent_counts} |")
    report_path = OUT / "report_zh.md"; report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    files = [report_path] + [OUT / dataset / "complete.json" for dataset in ORDER]
    write(OUT / "complete.json", {"files": {str(path.resolve()): file_sha256(path) for path in files}})
    emit("queue_complete", datasets=list(ORDER), report=str(report_path.resolve()))


def run_sequential_all():
    """Finish both retrieval methods for one dataset before starting the next."""
    global ORDER
    requested = ORDER
    try:
        for dataset in requested:
            ORDER = (dataset,)
            emit("dataset_pipeline_start", dataset=dataset, methods=list(METHODS))
            describe_all()
            embed_all()
            select_dataset(dataset)
            infer_all()
            if not (OUT / dataset / "complete.json").is_file():
                raise RuntimeError(f"{dataset} pipeline did not produce complete.json")
            emit("dataset_pipeline_complete", dataset=dataset, methods=list(METHODS))
    finally:
        ORDER = requested
    report()


def main():
    global ORDER
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("validate", "describe", "embed", "select", "infer", "report", "all"))
    parser.add_argument("--datasets", nargs="+", choices=ORDER)
    args = parser.parse_args()
    if args.datasets:
        ORDER = tuple(args.datasets)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "validate":
        validate_inputs(False); return
    validate_inputs(args.stage == "all")
    if args.stage == "all":
        run_sequential_all(); return
    if args.stage == "describe":
        describe_all()
    if args.stage == "embed":
        embed_all()
    if args.stage == "select":
        for dataset in ORDER:
            select_dataset(dataset)
    if args.stage == "infer":
        infer_all()
    if args.stage == "report":
        report()


if __name__ == "__main__":
    main()

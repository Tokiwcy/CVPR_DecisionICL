"""Resumable constrained Qwen3-VL-4B 1-shot evaluation for a dataset plan."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "Qwen3-VL-4B-Instruct"
PROCESSOR_SETTINGS = {"min_pixels": 65536, "max_pixels": 401408}


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_jsonl(path):
    if not Path(path).exists(): return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path, rows):
    with Path(path).open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows: handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def single_token_codes(tokenizer, count):
    codes, token_ids = [], []
    for codepoint in range(0x4E00, 0x9FFF + 1):
        code = chr(codepoint)
        encoded = tokenizer.encode(code, add_special_tokens=False)
        if len(encoded) == 1 and tokenizer.decode(encoded) == code:
            codes.append(code); token_ids.append(encoded[0])
            if len(codes) == count: break
    if len(codes) != count or len(set(token_ids)) != count:
        raise RuntimeError("could not construct enough unique single-token class codes")
    return codes, token_ids


def manifest(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def conversation(instruction, query_image, demo_image, demo_label, labels, codes):
    mapping = dict(zip(labels, codes))
    options = "\n".join(f"{code}: {label}" for code, label in zip(codes, labels))
    system = instruction + "\nClass codes:\n" + options + "\nReturn only one class code."
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [{"type": "image", "image": demo_image},
                                      {"type": "text", "text": instruction}]},
        {"role": "assistant", "content": [{"type": "text", "text": mapping[demo_label]}]},
        {"role": "user", "content": [{"type": "image", "image": query_image},
                                      {"type": "text", "text": instruction}]},
    ]


def forward(model, processor, records, instruction, labels, codes, token_ids):
    import torch
    opened, conversations = [], []
    try:
        for row in records:
            demo = Image.open(row["demo_image"]).convert("RGB")
            query = Image.open(row["query_image"]).convert("RGB")
            opened.extend((demo, query))
            conversations.append(conversation(instruction, query, demo, row["demo_label"], labels, codes))
        inputs = processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt", processor_kwargs={"padding": True},
        ).to(model.device, dtype=torch.bfloat16)
        with torch.inference_mode():
            logits = model(**inputs, logits_to_keep=1).logits[:, -1, token_ids].float()
            probabilities = logits.softmax(-1); indices = logits.argmax(-1)
        return [{"prediction": labels[index], "probability": float(probabilities[i, index].cpu())}
                for i, index in enumerate(indices.tolist())], int(inputs["input_ids"].shape[1])
    finally:
        for image in opened: image.close()


def paired(left, right, truth):
    from scipy.stats import binomtest
    delta = np.asarray([int(right[q] == truth[q]) - int(left[q] == truth[q]) for q in sorted(truth)], dtype=np.int8)
    rescue, harm = int((delta == 1).sum()), int((delta == -1).sum())
    rng = np.random.default_rng(42); boot = np.empty(10000, dtype=np.float32)
    for start in range(0, len(boot), 250):
        indices = rng.integers(0, len(delta), size=(min(250, len(boot) - start), len(delta)))
        boot[start:start + len(indices)] = delta[indices].mean(1)
    return {"rescue": rescue, "harm": harm, "accuracy_delta": float(delta.mean()),
            "paired_bootstrap_95pct": np.quantile(boot, (0.025, 0.975)).tolist(),
            "exact_mcnemar_p": float(binomtest(rescue, rescue + harm, 0.5).pvalue) if rescue + harm else 1.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    dataset_dir = args.dataset_dir.resolve(); plan = read(args.plan)
    output = dataset_dir / "atom_label"; output.mkdir(parents=True, exist_ok=True)
    config = read(dataset_dir / "dataset_config.json")
    train_rows = manifest(dataset_dir / f"{args.dataset}.train.jsonl")
    test_rows = manifest(dataset_dir / f"{args.dataset}.test.jsonl")
    train = {row["sample_id"]: row for row in train_rows}; test = {row["sample_id"]: row for row in test_rows}
    truth = {sample_id: row["label"] for sample_id, row in test.items()}
    labels = read(dataset_dir / f"{args.dataset}.classes.json")
    methods = args.methods or plan["active_target_methods"]
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, **PROCESSOR_SETTINGS)
    processor.tokenizer.padding_side = "left"
    codes, token_ids = single_token_codes(processor.tokenizer, len(labels))
    protocol = {
        "dataset": args.dataset, "model": str(MODEL), "dtype": "bfloat16", "attention": "sdpa",
        "processor": PROCESSOR_SETTINGS, "classification": "constrained next-token class-code logits",
        "instruction": config["instruction"], "labels": labels, "codes": codes,
        "plan_sha256": hashlib.sha256(Path(args.plan).read_bytes()).hexdigest(), "methods": list(plan["methods"]),
    }
    contract = digest(protocol); protocol["contract_sha256"] = contract; protocol["active_methods"] = methods
    write(output / "qwen3_inference_protocol.json", protocol)
    cache_path = output / "qwen3_context_cache.jsonl"
    cache = {row["key"]: row for row in load_jsonl(cache_path) if row.get("contract_sha256") == contract}
    keys, pending = {}, {}
    for priority, method in enumerate(methods):
        for query_id, demo_id in zip(plan["query_ids"], plan["methods"][method]):
            key = digest([contract, query_id, demo_id, test[query_id].get("image_sha256"),
                          train[demo_id].get("image_sha256")])
            keys[(method, query_id)] = key
            if key not in cache and key not in pending:
                pending[key] = {"key": key, "query_id": query_id, "demo_id": demo_id,
                                "query_image": test[query_id]["image_path"],
                                "demo_image": train[demo_id]["image_path"],
                                "demo_label": train[demo_id]["label"], "priority": priority}
    planned = sorted(pending.values(), key=lambda row: (row["priority"], row["query_id"]))
    print(json.dumps({"dataset": args.dataset, "stage": "prepared", "methods": methods,
                      "unique_new_contexts": len(planned), "cached": len(cache)}), flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, local_files_only=True, device_map="auto", dtype=torch.bfloat16, attn_implementation="sdpa"
    ).eval()
    started = time.perf_counter(); offset = 0
    while offset < len(planned):
        batch = planned[offset:offset + args.batch_size]
        try:
            outputs, tokens = forward(model, processor, batch, config["instruction"], labels, codes, token_ids)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if args.batch_size == 1: raise
            args.batch_size = 1; continue
        new_rows = []
        for record, result in zip(batch, outputs):
            row = {"key": record["key"], "contract_sha256": contract,
                   "query_id": record["query_id"], "demo_id": record["demo_id"], **result}
            cache[row["key"]] = row; new_rows.append(row)
        append_jsonl(cache_path, new_rows); offset += len(batch)
        if offset == len(planned) or offset % 100 < len(batch):
            elapsed = time.perf_counter() - started
            print(f"{args.dataset}_qwen3 {offset}/{len(planned)} batch={len(batch)} tokens={tokens} rate_per_min={offset/max(elapsed,1e-9)*60:.1f}", flush=True)
    predictions = {method: {query_id: cache[keys[(method, query_id)]]["prediction"]
                            for query_id in plan["query_ids"]} for method in methods}
    by_label = defaultdict(list); metrics = {}
    train_truth = {sample_id: row["label"] for sample_id, row in train.items()}
    for method in methods:
        correct = np.asarray([predictions[method][q] == truth[q] for q in plan["query_ids"]])
        macro_rows = defaultdict(list)
        for q in plan["query_ids"]: macro_rows[truth[q]].append(predictions[method][q] == truth[q])
        demos = plan["methods"][method]
        metrics[method] = {"accuracy": float(correct.mean()),
                           "macro_accuracy": float(np.mean([np.mean(v) for v in macro_rows.values()])),
                           "demo_same_class": float(np.mean([train_truth[d] == truth[q]
                                                              for q, d in zip(plan["query_ids"], demos)]))}
    paired_rows = {}
    for baseline in ("rices", "muier"):
        if baseline in predictions:
            for method in methods:
                if method != baseline:
                    paired_rows[f"{method}_vs_{baseline}"] = paired(predictions[baseline], predictions[method], truth)
    result = {"status": "complete", "dataset": args.dataset, "query_count": len(truth),
              "methods": metrics, "paired": paired_rows,
              "inference": {"unique_new_contexts": len(planned), "elapsed_seconds": time.perf_counter()-started}}
    write(output / "qwen3_results.json", result)
    write(output / "qwen3_predictions.json", predictions)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()


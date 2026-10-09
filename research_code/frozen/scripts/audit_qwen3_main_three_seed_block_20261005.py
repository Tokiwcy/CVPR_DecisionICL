"""Independent CPU audit of a completed Qwen3 three-seed target block."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_qwen3_main_three_seed_full_20261004 as new


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left, right, name):
    if not np.allclose(left, right, rtol=0, atol=1e-9):
        raise ValueError(f"{name}: {left!r} != {right!r}")


def paired(parent, method, truth):
    delta = np.asarray([int(method[q] == truth[q]) - int(parent[q] == truth[q]) for q in sorted(truth)], dtype=np.int8)
    rescue = int((delta == 1).sum())
    harm = int((delta == -1).sum())
    rng = np.random.default_rng(42)
    boot = np.empty(10000, dtype=np.float32)
    for start in range(0, len(boot), 250):
        indices = rng.integers(0, len(delta), size=(min(250, len(boot) - start), len(delta)))
        boot[start:start + len(indices)] = delta[indices].mean(1)
    return dict(rescue=rescue, harm=harm, accuracy_delta=float(delta.mean()),
                paired_bootstrap_95pct=np.quantile(boot, (0.025, 0.975)).tolist(),
                exact_mcnemar_p=float(binomtest(rescue, rescue + harm, 0.5).pvalue) if rescue + harm else 1.0)


def verify_pair(saved, calc, name):
    for key in ("rescue", "harm"):
        if saved[key] != calc[key]:
            raise ValueError(f"{name}.{key}")
    for key in ("accuracy_delta", "exact_mcnemar_p"):
        close(saved[key], calc[key], f"{name}.{key}")
    close(saved["paired_bootstrap_95pct"], calc["paired_bootstrap_95pct"], f"{name}.bootstrap")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=new.SEEDS, required=True)
    parser.add_argument("--dataset", choices=new.ORDER, required=True)
    parser.add_argument("--shots", type=int, choices=(1, 2), required=True)
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()
    if args.root is not None:
        new.ROOT = args.root.resolve()
    root = new.configure(args.seed, args.dataset, args.shots)
    out = root / f"{args.shots}shot" / args.dataset
    complete = read(out / "complete.json")
    for name, expected in complete["files"].items():
        if sha(Path(name)) != expected:
            raise ValueError("complete SHA mismatch: " + name)
    freeze = read(out / "plan.freeze.json")
    if sha(out / "test_plan.jsonl") != freeze["plan_sha256"]:
        raise ValueError("frozen plan SHA mismatch")
    for name, expected in freeze["files"].items():
        if sha(Path(name)) != expected:
            raise ValueError("plan input SHA mismatch: " + name)
    calibration = read(out / "calibration.json")
    if calibration["seed"] != args.seed or calibration["test_used"] or calibration["inherited_weights"]:
        raise ValueError("independent calibration contract mismatch")
    plan = rows(out / "test_plan.jsonl")
    if any(row["test_label_used"] for row in plan):
        raise ValueError("test labels in plan")
    proto = read(out / "inference_protocol.json")
    ph = proto["contract_sha256"]
    planned, train, query, labels, records, keys = new.r.make_records(args.dataset, args.shots, ph)
    if plan != planned:
        raise ValueError("plan replay mismatch")
    cache_rows = rows(out / "context_cache.jsonl")
    cache = {row["key"]: row for row in cache_rows}
    if len(cache) != len(cache_rows) or set(cache) != set(records):
        raise ValueError("cache coverage or duplicate key mismatch")
    if any("migrated_from" in row or "source_key" in row for row in cache_rows):
        raise ValueError("imported target prediction detected")
    lookup = {label.lower(): label for label in labels}
    ordered = list(records)
    for index, key in enumerate(ordered):
        row = cache[key]
        rec = records[key]
        if row["protocol_hash"] != ph or row["query_id"] != rec["query_id"] or row["demo_ids"] != rec["demo_ids"]:
            raise ValueError("raw-to-plan provenance mismatch")
        expected = lookup.get(row["raw_output"].strip().lower(), "__invalid_exact_output__")
        if row["prediction"] != expected:
            raise ValueError("strict raw parsing mismatch")
        if args.shots == 2 or args.dataset != "dtd":
            group = index // 2
            partner_keys = ordered[group * 2:(group + 1) * 2]
            stats = row["stats"]
            if stats["fixed_group_index"] != group or stats["fixed_group_signature"] != new.r.stable_hash(partner_keys):
                raise ValueError("fixed batch partner mismatch")
            if stats["execution_batch_size"] != len(partner_keys) or stats["batch_row"] != index % 2:
                raise ValueError("fixed batch row mismatch")
    # A memory-only resume writes preparation events to its own stdout file.
    # Require one fresh preparation across the original and resume logs; any
    # later preparation may only consume outputs from this same run.
    prep = []
    for log_name in ("run.stdout.log", "resume_v2.stdout.log", "resume_v3.stdout.log"):
        log_path = new.ROOT / log_name
        if not log_path.exists():
            continue
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("{") and '"stage": "target_prepared"' in line:
                row = json.loads(line)
                if row["dataset"] == args.dataset and row["shots"] == args.shots:
                    prep.append(row)
    fresh = [row for row in prep if row["cached"] == 0]
    if len(fresh) != 1 or fresh[0]["migrated"] != 0 or fresh[0]["new"] != len(cache):
        raise ValueError("fresh-only target preparation mismatch")
    if any(row["migrated"] != 0 or row["unique"] != len(cache)
           or row["cached"] + row["new"] != len(cache) for row in prep):
        raise ValueError("resume target preparation mismatch")
    predictions = read(out / "predictions.json")
    result = read(out / "results.json")
    truth = {key: sample.label for key, sample in query.items()}
    if set(predictions) != set(keys) or set(result["methods"]) != set(keys):
        raise ValueError("method matrix mismatch")
    summary = {}
    for method, mapping in keys.items():
        expected = {qid: cache[key]["prediction"] for qid, key in mapping.items()}
        if predictions[method] != expected or set(expected) != set(truth):
            raise ValueError("method-query prediction mapping mismatch: " + method)
        correct = np.asarray([expected[q] == truth[q] for q in truth], dtype=np.float64)
        accuracy = float(correct.mean())
        macro = float(np.mean([np.mean([expected[q] == truth[q] for q in truth if truth[q] == label]) for label in labels]))
        saved = result["methods"][method]
        close(saved["metrics"]["accuracy"], accuracy, method + ".accuracy")
        close(saved["metrics"]["macro_accuracy"], macro, method + ".macro")
        invalid = sum(value == "__invalid_exact_output__" for value in expected.values())
        if saved["invalid_exact_output_count"] != invalid:
            raise ValueError("invalid count mismatch: " + method)
        if method in new.r.official.ATOM_PARENTS:
            parent = new.r.official.ATOM_PARENTS[method]
            verify_pair(saved["paired_vs_parent"], paired(predictions[parent], expected, truth), method + ".parent")
        if method in new.r.METHODS[5:]:
            verify_pair(saved["paired_vs_matched_rices"], paired(predictions[new.r.METHODS[1]], expected, truth), method + ".rices")
        summary[method] = dict(accuracy=accuracy, macro_accuracy=macro, invalid=invalid)
    receipt = dict(status="passed", seed=args.seed, dataset=args.dataset, shots=args.shots,
                   methods=len(summary), queries=len(truth), unique_contexts=len(cache),
                   fresh_target_outputs=True, fixed_batch_partners_checked=True,
                   complete_sha256=sha(out / "complete.json"), methods_summary=summary)
    destination = out / "independent_recheck_20261005.json"
    destination.write_text(json.dumps(receipt, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(status="passed", receipt=str(destination), methods=len(summary), contexts=len(cache)), ensure_ascii=True))


if __name__ == "__main__":
    main()

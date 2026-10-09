"""Recompute released accuracy/macro/invalid, pairs and exact p from artifacts.

Uses either the downloaded CAS release directly or a materialized workspace.
Bootstrap intervals are preserved in the original audited receipts, rather than
re-estimated under a new random draw by this lightweight verifier.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def exact_p(rescue, harm):
    n, limit = rescue + harm, min(rescue, harm)
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(limit + 1)) / 2**n) if n else 1.0


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bundle", type=Path)
    group.add_argument("--workspace", type=Path)
    group.add_argument("--review-archive", type=Path)
    args = parser.parse_args()
    index = json.loads((ROOT / "artifact_bundle/evaluation_index.json").read_text())
    assert hashlib.sha256((ROOT / "results/method_results.csv").read_bytes()).hexdigest() == index["table_sha256"]
    archives, cache = {}, {}
    if args.bundle:
        with gzip.open(args.bundle / "decisionicl-artifact-manifest.json.gz", "rt") as stream:
            manifest = json.load(stream)
    review = zipfile.ZipFile(args.review_archive) if args.review_archive else None

    def read(path, expected):
        if path not in cache:
            if args.workspace:
                payload = (args.workspace / path).read_bytes()
            elif review:
                payload = review.read(path)
            else:
                info = manifest["files"][path]
                if info["archive"] not in archives:
                    archives[info["archive"]] = zipfile.ZipFile(args.bundle / info["archive"])
                payload = archives[info["archive"]].read(info["member"])
            assert hashlib.sha256(payload).hexdigest() == expected, path
            cache[path] = payload
        return cache[path]

    methods = list(csv.DictReader((ROOT / "results/method_results.csv").open(encoding="utf-8", newline="")))
    pairs = list(csv.DictReader((ROOT / "results/paired_results.csv").open(encoding="utf-8", newline="")))
    metrics = {(r["model"], r["dataset"], int(r["shots"]), r["method"]): r for r in methods}
    matched = json.loads((ROOT / "results/matched_source_control.json").read_text())
    review_files = json.loads((ROOT / "artifact_bundle/review_manifest.json").read_text())["files"]
    correct, query_order = {}, {}
    try:
        for record in index["methods"]:
            key = (record["model"], record["dataset"], record["shots"], record["method"])
            pred = json.loads(read(record["predictions"], record["predictions_sha256"]))[record["source_method"]]
            labels = {r["sample_id"]: r["label"].strip().lower() for r in map(json.loads, read(record["test_manifest"], record["test_manifest_sha256"]).decode("utf-8-sig").splitlines()) if r}
            assert set(pred) == set(labels), key
            values = {qid: pred[qid].strip().lower() == label for qid, label in labels.items()}
            groups = defaultdict(list)
            for qid, label in labels.items():
                groups[label].append(values[qid])
            accuracy = sum(values.values()) / len(values)
            macro = sum(sum(group) / len(group) for group in groups.values()) / len(groups)
            allowed = set(labels.values())
            invalid = sum(value.strip().lower() not in allowed for value in pred.values())
            row = metrics[key]
            assert abs(accuracy - float(row["accuracy"])) < 1e-12, key
            assert abs(macro - float(row["macro_accuracy"])) < 1e-12, key
            assert invalid == int(row["invalid_outputs"]), (key, invalid, row["invalid_outputs"])
            correct[key] = values
        for row in pairs:
            prefix = (row["model"], row["dataset"], int(row["shots"]))
            a, b = correct[(*prefix, row["method"])], correct[(*prefix, row["comparator"])]
            assert set(a) == set(b)
            rescue = sum(a[q] and not b[q] for q in a)
            harm = sum(b[q] and not a[q] for q in a)
            assert (rescue, harm) == (int(row["rescue"]), int(row["harm"])), row
            assert abs(100 * (rescue - harm) / len(a) - float(row["delta_pp"])) < 1e-9
            p = exact_p(rescue, harm)
            assert math.isclose(p, float(row["exact_mcnemar_p"]), rel_tol=1e-8, abs_tol=1e-14)
        matched_folders = {"dtd": "qwen3_core_sources/dtd", "aircraft": "qwen3_decision_generic/aircraft"}
        for row in matched:
            base = "decision_guided_retrieval_cleanroom/artifacts/priority_1shot_ablation_20260920/" + matched_folders[row["dataset"]]
            read(base + "/complete.json", row["complete_sha256"])
            read(base + "/independent_audit.json", row["independent_audit_sha256"])
            path = base + "/predictions.json"
            pred = json.loads(read(path, review_files[path]["sha256"]))
            record = next(r for r in index["methods"] if r["dataset"] == row["dataset"] and r["shots"] == 1)
            labels = {r["sample_id"]: r["label"].strip().lower() for r in map(json.loads, read(record["test_manifest"], record["test_manifest_sha256"]).decode("utf-8-sig").splitlines()) if r}
            a, b = pred[row["parent"] + "+matched_decision"], pred[row["parent"] + "+generic_visual"]
            assert set(a) == set(b) == set(labels)
            ca = {q: a[q].strip().lower() == labels[q] for q in labels}
            cb = {q: b[q].strip().lower() == labels[q] for q in labels}
            rescue = sum(ca[q] and not cb[q] for q in labels)
            harm = sum(cb[q] and not ca[q] for q in labels)
            assert len(labels) == row["n"]
            assert abs(100 * sum(ca.values()) / len(labels) - row["decisionicl_accuracy_pct"]) < 1e-10
            assert abs(100 * sum(cb.values()) / len(labels) - row["generic_accuracy_pct"]) < 1e-10
            assert (rescue, harm) == (row["rescue"], row["harm"])
            assert math.isclose(exact_p(rescue, harm), row["exact_mcnemar_raw_p"], rel_tol=1e-8, abs_tol=1e-14)
    finally:
        for archive in archives.values():
            archive.close()
        if review:
            review.close()
    print(json.dumps({"status": "passed", "method_rows_recomputed": len(correct), "paired_rows_recomputed": len(pairs),
        "matched_factor_source_pairs_recomputed": len(matched),
        "checks": ["query_coverage", "accuracy", "macro_accuracy", "invalid_outputs", "rescue", "harm", "delta", "two_sided_exact_mcnemar_raw_p"]}))


if __name__ == "__main__":
    main()

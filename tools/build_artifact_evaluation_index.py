"""Author-side map from public table rows to original predictions/test labels."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    inventory = json.loads(args.inventory.read_text())["files"]
    candidates = []
    for relative in inventory:
        if relative.endswith("test.jsonl"):
            rows = [json.loads(line) for line in (workspace / relative).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            if rows and all("sample_id" in row and "label" in row for row in rows):
                labels = {row["sample_id"]: row["label"].strip().lower() for row in rows}
                if len(labels) == len(rows):
                    candidates.append((relative, labels))
    with (ROOT / "results/method_results.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    index, cache = [], {}
    for row in rows:
        source = workspace / row["source_artifact"]
        folder = source.parent
        if source.name == "live_results.json":
            locations = folder / "execution_locations.json"
            folder = Path(json.loads(locations.read_text())[f"{row['dataset']}/{row['shots']}"]) if locations.exists() else folder / f"{row['shots']}shot" / row["dataset"]
        predictions = folder / "predictions.json"
        if predictions not in cache:
            value = json.loads(predictions.read_text())
            sample = next(iter(value.values()))
            possible = [(p, labels) for p, labels in candidates if set(labels) == set(sample)]
            if not possible:
                raise ValueError("No packaged test manifest covers " + str(predictions))
            def overlap(item):
                allowed = set(item[1].values())
                return sum(v.strip().lower() in allowed for v in sample.values())
            possible.sort(key=lambda item: (-overlap(item), item[0]))
            best = possible[0]
            for _, labels in possible:
                if overlap((None, labels)) == overlap(best) and labels != best[1]:
                    raise ValueError("Ambiguous test label mapping")
            cache[predictions] = value, best[0]
        value, labels = cache[predictions]
        assert row["source_method"] in value
        relative = predictions.relative_to(workspace).as_posix()
        assert relative in inventory and labels in inventory
        index.append({"model": row["model"], "dataset": row["dataset"], "shots": int(row["shots"]),
            "method": row["method"], "source_method": row["source_method"],
            "predictions": relative, "predictions_sha256": sha(predictions),
            "test_manifest": labels, "test_manifest_sha256": sha(workspace / labels)})
    destination = ROOT / "artifact_bundle/evaluation_index.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"table_sha256": sha(ROOT / "results/method_results.csv"), "methods": index}, ensure_ascii=True, indent=2) + "\n")
    print(json.dumps({"method_rows": len(index), "prediction_files": len(cache)}))


if __name__ == "__main__":
    main()

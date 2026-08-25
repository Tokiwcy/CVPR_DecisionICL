from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SPLITS = {0: "train", 1: "validation", 2: "test"}
LABELS = {0: "landbird", 1: "waterbird"}
PLACES = {0: "land", 1: "water"}
ROLES = {0: "candidate", 1: "validation_query", 2: "test_query"}


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def build_manifest(source: Path, output: Path) -> dict[str, Any]:
    source = source.resolve()
    metadata_path = source / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing Waterbirds metadata: {metadata_path}")
    rows: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        for value in csv.DictReader(handle):
            split_index = int(value["split"])
            y = int(value["y"])
            place = int(value["place"])
            image = (source / value["img_filename"]).resolve()
            if not image.is_file():
                raise FileNotFoundError(f"Missing Waterbirds image: {image}")
            sample_id = f"waterbirds-{int(value['img_id']):06d}"
            label = LABELS[y]
            background = PLACES[place]
            diagnostic = y != place
            rows.append(
                {
                    "id": sample_id,
                    "unit_id": sample_id,
                    "split": SPLITS[split_index],
                    "role": ROLES[split_index],
                    "image": str(image),
                    "text": "",
                    "label": label,
                    "group": f"{label}_on_{background}",
                    "y": y,
                    "place": place,
                    "metadata": {
                        "diagnostic": diagnostic,
                        "conflict": diagnostic,
                        "spurious_attribute": background,
                        "aligned_attribute": PLACES[y],
                    },
                }
            )
    rows.sort(key=lambda row: (str(row["split"]), str(row["id"])))
    write_jsonl(output.resolve(), rows)
    counts = Counter((str(row["split"]), str(row["group"])) for row in rows)
    return {
        "source": str(source),
        "output": str(output.resolve()),
        "rows": len(rows),
        "counts": {
            f"{split}::{group}": count
            for (split, group), count in sorted(counts.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the released Waterbirds metadata.csv into BehaviorICL JSONL."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_manifest(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()

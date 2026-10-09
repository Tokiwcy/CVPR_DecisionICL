from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .schema import AtomBank, Sample, validate_unique_samples


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False))
            handle.write("\n")


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False))
            handle.write("\n")


def load_manifest(path: str | Path, expected_split: str | None = None) -> list[Sample]:
    samples = validate_unique_samples(
        (Sample.from_dict(row) for row in read_jsonl(path)),
        require_image=True,
    )
    if expected_split is not None:
        wrong = [s.sample_id for s in samples if s.split != expected_split]
        if wrong:
            raise ValueError(f"manifest contains samples outside {expected_split}: {wrong[:5]}")
    return sorted(samples, key=lambda sample: sample.sample_id)


def save_manifest(path: str | Path, samples: Iterable[Sample]) -> None:
    materialized = validate_unique_samples(samples, require_image=False)
    write_jsonl(path, (sample.to_dict() for sample in sorted(materialized, key=lambda x: x.sample_id)))


def load_atom_bank(path: str | Path) -> AtomBank:
    return AtomBank.from_dict(read_json(path))


def save_atom_bank(path: str | Path, bank: AtomBank) -> None:
    bank.validate()
    write_json(path, bank.to_dict())

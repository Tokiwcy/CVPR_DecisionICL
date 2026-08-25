from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .types import Sample


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_manifest(path: str | Path) -> list[Sample]:
    source = Path(path).resolve()
    samples = [Sample.from_dict(row, source.parent) for row in load_jsonl(source)]
    ids = [sample.id for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest sample ids must be unique")
    missing = [sample.image for sample in samples if not Path(sample.image).is_file()]
    if missing:
        preview = "\n".join(missing[:5])
        raise FileNotFoundError(f"manifest contains {len(missing)} missing images, e.g.\n{preview}")
    return samples


def fingerprint_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_matrix_bundle(
    path: str | Path,
    values: np.ndarray,
    sample_ids: list[str],
    column_ids: list[str],
    **extra: np.ndarray,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        values=np.asarray(values, dtype=np.float32),
        sample_ids=np.asarray(sample_ids, dtype=str),
        column_ids=np.asarray(column_ids, dtype=str),
        **extra,
    )


def load_matrix_bundle(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as bundle:
        return {key: bundle[key] for key in bundle.files}

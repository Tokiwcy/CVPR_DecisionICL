from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Sample:
    id: str
    split: str
    image: str
    label: str
    text: str = ""
    group: str | None = None
    options: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    @classmethod
    def from_dict(cls, row: dict[str, Any], manifest_dir: Path) -> "Sample":
        image = Path(str(row["image"]))
        if not image.is_absolute():
            image = manifest_dir / image
        metadata = dict(row.get("metadata") or {})
        raw_options = row.get("options", metadata.get("options", ())) or ()
        return cls(
            id=str(row["id"]),
            split=str(row["split"]),
            image=str(image.resolve()),
            label=str(row["label"]),
            text="" if row.get("text") is None else str(row.get("text", "")),
            group=None if row.get("group") is None else str(row["group"]),
            options=tuple(str(value) for value in raw_options),
            metadata=metadata,
        )


@dataclass(frozen=True)
class BoundaryRecord:
    sample_id: str
    true_label: str
    predicted_label: str
    runner_up: str
    correct: bool
    margin: float
    true_margin: float
    log_scores: dict[str, float]
    prompt_predictions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["prompt_predictions"] = list(self.prompt_predictions)
        return row

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "BoundaryRecord":
        return cls(**{**row, "prompt_predictions": tuple(row.get("prompt_predictions", []))})


@dataclass(frozen=True)
class DecisionAtom:
    id: str
    label_a: str
    label_b: str
    concept: str
    a_side: str
    b_side: str
    source_mode: str
    bootstrap: int
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "DecisionAtom":
        return cls(**row)


@dataclass(frozen=True)
class AtomQuality:
    atom_id: str
    source_mode: str
    stability: float
    coverage: float
    discriminative_auc: float
    alignment: float
    alignment_by_class: dict[str, float]
    descriptive_eligible: bool
    retrieval_eligible: bool
    bootstrap_support: float = 0.0
    cluster_id: str | None = None
    selected_descriptive: bool = False
    selected_retrieval: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "AtomQuality":
        return cls(**row)

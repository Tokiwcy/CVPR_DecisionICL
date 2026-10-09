from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from .hashing import stable_hash


Split = Literal["train", "validation", "test", "discovery"]


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image_path: str
    label: str
    split: Split
    question: str = ""
    answer: str = ""
    image_sha256: str | None = None

    def validate(self, require_image: bool = True) -> None:
        if not self.sample_id.strip():
            raise ValueError("sample_id must be non-empty")
        if not self.label.strip():
            raise ValueError(f"sample {self.sample_id}: label must be non-empty")
        if self.split not in {"train", "validation", "test", "discovery"}:
            raise ValueError(f"sample {self.sample_id}: invalid split {self.split!r}")
        if require_image and not Path(self.image_path).is_file():
            raise FileNotFoundError(f"sample {self.sample_id}: {self.image_path}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Sample":
        return cls(**value)


@dataclass(frozen=True)
class AtomState:
    state_id: str
    label: str
    visual_description: str
    coordinate: float | None = None
    abstain: bool = False

    def validate(self) -> None:
        if not self.state_id.strip() or not self.label.strip():
            raise ValueError("state_id and label must be non-empty")
        if len(self.visual_description.split()) < 3:
            raise ValueError(
                f"state {self.state_id}: visual_description must be a concrete image description"
            )
        if self.abstain and self.coordinate is not None:
            raise ValueError(f"state {self.state_id}: abstain state cannot have a coordinate")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AtomState":
        return cls(**value)


@dataclass(frozen=True)
class FactorGroup:
    group_id: str
    region: str
    attribute: str
    question: str
    applies_to: str
    states: tuple[AtomState, ...]
    perspective: str
    source_model: str
    source_prompt_id: str
    discovery_batch_id: str
    source_sample_ids: tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        fields = {
            "group_id": self.group_id,
            "region": self.region,
            "attribute": self.attribute,
            "question": self.question,
            "applies_to": self.applies_to,
            "perspective": self.perspective,
            "source_model": self.source_model,
            "source_prompt_id": self.source_prompt_id,
            "discovery_batch_id": self.discovery_batch_id,
        }
        for name, value in fields.items():
            if not value.strip():
                raise ValueError(f"group {self.group_id or '<missing>'}: {name} must be non-empty")
        if len(self.states) < 2:
            raise ValueError(f"group {self.group_id}: at least two contrastive states are required")
        for state in self.states:
            state.validate()
        state_ids = [s.state_id for s in self.states]
        if len(state_ids) != len(set(state_ids)):
            raise ValueError(f"group {self.group_id}: duplicate state_id")
        descriptions = [s.visual_description.casefold().strip() for s in self.states]
        if len(descriptions) != len(set(descriptions)):
            raise ValueError(f"group {self.group_id}: duplicate state description")
        substantive = [s for s in self.states if not s.abstain]
        if len(substantive) < 2:
            raise ValueError(f"group {self.group_id}: at least two non-abstain states are required")
        coordinates = [s.coordinate for s in substantive]
        if any(v is not None for v in coordinates):
            if not all(v is not None for v in coordinates):
                raise ValueError(f"group {self.group_id}: coordinates must be present for every substantive state")
            if len(set(float(v) for v in coordinates if v is not None)) != len(coordinates):
                raise ValueError(f"group {self.group_id}: ordinal coordinates must be distinct")

    @property
    def is_ordinal(self) -> bool:
        substantive = [s for s in self.states if not s.abstain]
        return bool(substantive) and all(s.coordinate is not None for s in substantive)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["states"] = [s.to_dict() for s in self.states]
        value["source_sample_ids"] = list(self.source_sample_ids)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FactorGroup":
        copied = dict(value)
        copied["states"] = tuple(AtomState.from_dict(x) for x in copied["states"])
        copied["source_sample_ids"] = tuple(copied.get("source_sample_ids", ()))
        return cls(**copied)


@dataclass(frozen=True)
class AtomBank:
    task_name: str
    task_instruction: str
    class_names: tuple[str, ...]
    factor_scope: Literal["task", "class", "query"]
    groups: tuple[FactorGroup, ...]
    discovery_split: str
    discovery_seed: int
    protocol_version: str = "dgdr-cleanroom-v1"

    def validate(self) -> None:
        if not self.task_name.strip() or not self.task_instruction.strip():
            raise ValueError("task_name and task_instruction must be non-empty")
        if len(self.class_names) < 2 or len(set(self.class_names)) != len(self.class_names):
            raise ValueError("class_names must contain at least two unique labels")
        if self.factor_scope not in {"task", "class", "query"}:
            raise ValueError(f"invalid factor_scope {self.factor_scope!r}")
        if self.discovery_split not in {"train", "discovery"}:
            raise ValueError("factor discovery must use only train/discovery data")
        if not self.groups:
            raise ValueError("atom bank must contain at least one factor group")
        for group in self.groups:
            group.validate()
        ids = [g.group_id for g in self.groups]
        if len(ids) != len(set(ids)):
            raise ValueError("atom bank contains duplicate group_id")

    @property
    def bank_hash(self) -> str:
        return stable_hash(self.to_dict(include_hash=False))

    def to_dict(self, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "task_name": self.task_name,
            "task_instruction": self.task_instruction,
            "class_names": list(self.class_names),
            "factor_scope": self.factor_scope,
            "groups": [g.to_dict() for g in self.groups],
            "discovery_split": self.discovery_split,
            "discovery_seed": self.discovery_seed,
            "protocol_version": self.protocol_version,
        }
        if include_hash:
            value["bank_hash"] = stable_hash(value)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AtomBank":
        copied = {k: v for k, v in value.items() if k != "bank_hash"}
        copied["class_names"] = tuple(copied["class_names"])
        copied["groups"] = tuple(FactorGroup.from_dict(x) for x in copied["groups"])
        bank = cls(**copied)
        bank.validate()
        expected = value.get("bank_hash")
        if expected is not None and expected != bank.bank_hash:
            raise ValueError("atom bank hash mismatch")
        return bank


def validate_unique_samples(samples: Iterable[Sample], require_image: bool = True) -> list[Sample]:
    materialized = list(samples)
    seen: set[str] = set()
    for sample in materialized:
        sample.validate(require_image=require_image)
        if sample.sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample.sample_id}")
        seen.add(sample.sample_id)
    return materialized


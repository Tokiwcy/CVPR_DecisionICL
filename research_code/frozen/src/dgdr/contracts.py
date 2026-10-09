from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .hashing import stable_hash
from .schema import AtomBank, Sample


@dataclass(frozen=True)
class MethodContract:
    atom_bank_hash: str
    evidence_model: str
    evidence_model_revision: str
    evidence_text_template: str
    evidence_transform: str
    group_similarity: str
    global_similarity: str
    calibration_objective: str
    calibration_folds_hash: str
    candidate_manifest_hash: str
    calibration_manifest_hash: str
    candidate_embedding_hash: str
    calibration_embedding_hash: str
    candidate_evidence_hash: str
    calibration_evidence_hash: str
    protocol_version: str = "dgdr-cleanroom-v1"

    @property
    def contract_hash(self) -> str:
        return stable_hash(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["contract_hash"] = self.contract_hash
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MethodContract":
        copied = {k: v for k, v in value.items() if k != "contract_hash"}
        contract = cls(**copied)
        expected = value.get("contract_hash")
        if expected is not None and expected != contract.contract_hash:
            raise ValueError("method contract hash mismatch")
        return contract


def manifest_hash(samples: Iterable[Sample], include_labels: bool = True) -> str:
    rows = []
    for sample in sorted(samples, key=lambda item: item.sample_id):
        row = {
            "sample_id": sample.sample_id,
            "image_sha256": sample.image_sha256,
            "split": sample.split,
        }
        if include_labels:
            row["label"] = sample.label
        rows.append(row)
    return stable_hash(rows)


def assert_discovery_is_train_only(bank: AtomBank, samples: Iterable[Sample]) -> None:
    by_id = {sample.sample_id: sample for sample in samples}
    for group in bank.groups:
        for sample_id in group.source_sample_ids:
            if sample_id not in by_id:
                raise ValueError(f"discovery source {sample_id} is absent from supplied discovery manifest")
            if by_id[sample_id].split not in {"train", "discovery"}:
                raise ValueError(f"factor discovery leakage: {sample_id} belongs to {by_id[sample_id].split}")


def assert_calibration_splits(candidates: Iterable[Sample], queries: Iterable[Sample]) -> None:
    candidate_list = list(candidates)
    query_list = list(queries)
    if any(sample.split != "train" for sample in candidate_list):
        raise ValueError("retrieval candidates used for calibration must all be train samples")
    if any(sample.split != "validation" for sample in query_list):
        raise ValueError("calibration queries must all be validation samples; test is forbidden")
    overlap = {sample.sample_id for sample in candidate_list} & {sample.sample_id for sample in query_list}
    if overlap:
        raise ValueError(f"candidate/query ID overlap: {sorted(overlap)[:5]}")


def assert_test_retrieval_splits(candidates: Iterable[Sample], queries: Iterable[Sample]) -> None:
    candidate_list = list(candidates)
    query_list = list(queries)
    if any(sample.split != "train" for sample in candidate_list):
        raise ValueError("test retrieval candidates must all be train samples")
    if any(sample.split != "test" for sample in query_list):
        raise ValueError("test query manifest must contain only test samples")
    overlap = {sample.sample_id for sample in candidate_list} & {sample.sample_id for sample in query_list}
    if overlap:
        raise ValueError(f"candidate/test ID overlap: {sorted(overlap)[:5]}")

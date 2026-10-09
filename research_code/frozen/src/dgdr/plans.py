from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .retrieval import RetrievalResult
from .schema import Sample


@dataclass(frozen=True)
class RetrievalPlan:
    method: str
    contract_hash: str
    shots: int
    rows: tuple[dict[str, Any], ...]

    def validate(self, queries: Sequence[Sample], candidates: Sequence[Sample]) -> None:
        query_ids = [sample.sample_id for sample in queries]
        candidate_ids = {sample.sample_id for sample in candidates}
        if [row["query_id"] for row in self.rows] != query_ids:
            raise ValueError("retrieval plan rows do not align with query manifest")
        for row in self.rows:
            demos = row["demo_ids"]
            if len(demos) != self.shots or len(set(demos)) != len(demos):
                raise ValueError(f"query {row['query_id']}: invalid demo count or duplicate demos")
            unknown = set(demos) - candidate_ids
            if unknown:
                raise ValueError(f"query {row['query_id']}: demos outside train pool: {sorted(unknown)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "contract_hash": self.contract_hash,
            "shots": self.shots,
            "rows": list(self.rows),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RetrievalPlan":
        return cls(
            method=str(value["method"]),
            contract_hash=str(value["contract_hash"]),
            shots=int(value["shots"]),
            rows=tuple(value["rows"]),
        )


def make_plan(method: str, contract_hash: str, result: RetrievalResult) -> RetrievalPlan:
    rows = tuple(
        {
            "query_id": query_id,
            "demo_ids": list(demos),
            "demo_scores": list(scores),
        }
        for query_id, demos, scores in zip(
            result.query_ids,
            result.demo_ids,
            result.demo_scores,
            strict=True,
        )
    )
    shots = len(rows[0]["demo_ids"]) if rows else 0
    return RetrievalPlan(method=method, contract_hash=contract_hash, shots=shots, rows=rows)

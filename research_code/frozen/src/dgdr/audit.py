from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.stats import spearmanr

from .evidence import EvidenceStore
from .hashing import stable_hash
from .schema import AtomBank, Sample


@dataclass(frozen=True)
class EvidenceAnnotation:
    sample_id: str
    group_id: str
    applicable: bool
    observed_state_id: str | None
    confidence: float
    source: str
    coverage_sample: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvidenceAnnotation":
        return cls(
            sample_id=str(value["sample_id"]),
            group_id=str(value["group_id"]),
            applicable=bool(value["applicable"]),
            observed_state_id=(
                None if value.get("observed_state_id") is None else str(value["observed_state_id"])
            ),
            confidence=float(value.get("confidence", 1.0)),
            source=str(value.get("source", "unspecified")),
            coverage_sample=bool(value.get("coverage_sample", False)),
        )


@dataclass(frozen=True)
class AuditThresholds:
    minimum_annotations_per_group: int = 6
    minimum_applicable_fraction: float = 0.80
    minimum_top1_state_accuracy: float = 0.70
    minimum_signed_direction_accuracy: float = 0.70
    minimum_ordinal_spearman: float = 0.50


def make_blind_audit_bundle(
    *,
    samples: Sequence[Sample],
    bank: AtomBank,
    evidence: EvidenceStore,
    examples_per_band: int = 1,
    uniform_examples_per_group: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    if tuple(sample.sample_id for sample in samples) != evidence.sample_ids:
        raise ValueError("audit samples and evidence IDs are not identically ordered")
    if bank.bank_hash != evidence.bank_hash:
        raise ValueError("audit bank/evidence hash mismatch")
    sample_by_index = {index: sample for index, sample in enumerate(samples)}
    entries: dict[tuple[str, str], dict[str, Any]] = {}

    def add_entry(
        *,
        sample_index: int,
        group_index: int,
        reason: dict[str, Any],
        hidden_score: tuple[str, float] | None = None,
        coverage_sample: bool = False,
    ) -> None:
        sample = sample_by_index[sample_index]
        group = bank.groups[group_index]
        key = (sample.sample_id, group.group_id)
        if key not in entries:
            entries[key] = {
                "sample_id": sample.sample_id,
                "image_path": sample.image_path,
                "group_id": group.group_id,
                "question": group.question,
                "allowed_states": [
                    {
                        "state_id": state.state_id,
                        "label": state.label,
                        "visual_description": state.visual_description,
                    }
                    for state in group.states
                ],
                "selection_reasons": [],
                "coverage_sample": False,
                "hidden_model_scores": {},
            }
        entries[key]["selection_reasons"].append(reason)
        entries[key]["coverage_sample"] = bool(entries[key]["coverage_sample"] or coverage_sample)
        if hidden_score is not None:
            state_id, score = hidden_score
            entries[key]["hidden_model_scores"][state_id] = score

    for group_index, group in enumerate(bank.groups):
        state_indices = evidence.state_slice(group_index)
        for local_index, state in enumerate(group.states):
            scores = evidence.signed_margins[:, state_indices[local_index]]
            ordering = np.argsort(scores, kind="stable")
            bands = {
                "low": ordering[:examples_per_band],
                "middle": ordering[
                    max(0, len(ordering) // 2 - examples_per_band // 2) :
                    max(0, len(ordering) // 2 - examples_per_band // 2) + examples_per_band
                ],
                "high": ordering[-examples_per_band:][::-1],
            }
            for band, indices in bands.items():
                for rank, sample_index in enumerate(indices):
                    add_entry(
                        sample_index=int(sample_index),
                        group_index=group_index,
                        reason={
                            "role": "direction_extreme",
                            "state_id": state.state_id,
                            "band": band,
                            "rank_within_band": rank,
                        },
                        # Kept for later analysis but the human reviewer must
                        # not receive this field while labeling.
                        hidden_score=(state.state_id, float(scores[int(sample_index)])),
                    )
        uniform_order = sorted(
            range(len(samples)),
            key=lambda index: stable_hash(
                {"seed": seed, "group_id": group.group_id, "sample_id": samples[index].sample_id}
            ),
        )
        for rank, sample_index in enumerate(uniform_order[:uniform_examples_per_group]):
            add_entry(
                sample_index=sample_index,
                group_index=group_index,
                reason={"role": "uniform_coverage", "rank": rank},
                coverage_sample=True,
            )
    return {
        "bank_hash": bank.bank_hash,
        "evidence_store_hash": evidence.store_hash,
        "instructions": (
            "Human spot-check only. Hide hidden_model_scores and selection_reasons while reviewing. "
            "For each unique sample/group pair, label whether the factor is visibly applicable and, "
            "if so, exactly one observed state_id. Do not use the class label."
        ),
        "entries": [entries[key] for key in sorted(entries)],
    }


def distribution_audit(bank: AtomBank, evidence: EvidenceStore) -> dict[str, Any]:
    if bank.bank_hash != evidence.bank_hash:
        raise ValueError("distribution audit bank/evidence mismatch")
    groups: dict[str, Any] = {}
    for group_index, group in enumerate(bank.groups):
        probabilities = evidence.group_probability(group_index)
        signed = evidence.group_signed_margin(group_index)
        predicted = probabilities.argmax(axis=1)
        occupancy = np.bincount(predicted, minlength=len(group.states)) / len(predicted)
        entropy = -np.sum(probabilities * np.log(probabilities.clip(min=1e-12)), axis=1)
        substantive_count = sum(not state.abstain for state in group.states)
        groups[group.group_id] = {
            "state_top1_occupancy": {
                state.state_id: float(occupancy[index]) for index, state in enumerate(group.states)
            },
            "mean_normalized_entropy": float(entropy.mean() / np.log(len(group.states))),
            "signed_margin_iqr": {
                state.state_id: float(np.quantile(signed[:, index], 0.75) - np.quantile(signed[:, index], 0.25))
                for index, state in enumerate(group.states)
            },
            "model_applicability_mean": float(evidence.applicability_scores[:, group_index].mean()),
            "applicability_is_measured": any(state.abstain for state in group.states),
            "substantive_state_count": substantive_count,
            "status": "descriptive_only",
        }
    return {
        "bank_hash": bank.bank_hash,
        "evidence_store_hash": evidence.store_hash,
        "note": "Distribution checks detect collapse but cannot prove visual correctness without annotations.",
        "groups": groups,
    }


def score_evidence_annotations(
    *,
    bank: AtomBank,
    evidence: EvidenceStore,
    annotations: Iterable[EvidenceAnnotation],
    thresholds: AuditThresholds = AuditThresholds(),
) -> dict[str, Any]:
    if bank.bank_hash != evidence.bank_hash:
        raise ValueError("annotation audit bank/evidence mismatch")
    sample_index = {sample_id: index for index, sample_id in enumerate(evidence.sample_ids)}
    group_index = {group_id: index for index, group_id in enumerate(evidence.group_ids)}
    grouped: dict[str, list[EvidenceAnnotation]] = {group.group_id: [] for group in bank.groups}
    seen: set[tuple[str, str]] = set()
    for annotation in annotations:
        key = (annotation.sample_id, annotation.group_id)
        if key in seen:
            raise ValueError(f"duplicate evidence annotation: {key}")
        seen.add(key)
        if annotation.sample_id not in sample_index:
            raise ValueError(f"unknown annotation sample: {annotation.sample_id}")
        if annotation.group_id not in group_index:
            raise ValueError(f"unknown annotation group: {annotation.group_id}")
        if not 0.0 <= annotation.confidence <= 1.0:
            raise ValueError(f"annotation confidence outside [0,1]: {key}")
        grouped[annotation.group_id].append(annotation)
    reports: dict[str, Any] = {}
    all_passed = True
    for group in bank.groups:
        rows = grouped[group.group_id]
        state_lookup = {state.state_id: index for index, state in enumerate(group.states)}
        applicable = [row for row in rows if row.applicable]
        coverage_rows = [row for row in rows if row.coverage_sample] or rows
        coverage_applicable = [row for row in coverage_rows if row.applicable]
        for row in applicable:
            if row.observed_state_id not in state_lookup:
                raise ValueError(
                    f"annotation {row.sample_id}/{row.group_id}: applicable row needs a valid observed_state_id"
                )
        gi = group_index[group.group_id]
        indices = evidence.state_slice(gi)
        top1_correct: list[bool] = []
        direction_correct: list[bool] = []
        predicted_ordinal: list[float] = []
        observed_ordinal: list[float] = []
        for row in applicable:
            si = sample_index[row.sample_id]
            local_state = state_lookup[str(row.observed_state_id)]
            predicted = int(np.argmax(evidence.state_probabilities[si, indices]))
            top1_correct.append(predicted == local_state)
            direction_correct.append(bool(evidence.signed_margins[si, indices[local_state]] > 0))
            if group.is_ordinal:
                state = group.states[local_state]
                if state.coordinate is not None:
                    predicted_ordinal.append(float(evidence.ordinal_scores[si, gi]))
                    observed_ordinal.append(float(state.coordinate))
        annotation_count = len(rows)
        applicable_fraction = (
            float(len(coverage_applicable) / len(coverage_rows)) if coverage_rows else 0.0
        )
        top1_accuracy = float(np.mean(top1_correct)) if top1_correct else 0.0
        direction_accuracy = float(np.mean(direction_correct)) if direction_correct else 0.0
        ordinal_spearman: float | None = None
        if group.is_ordinal and len(set(observed_ordinal)) >= 2:
            statistic = spearmanr(observed_ordinal, predicted_ordinal).statistic
            ordinal_spearman = 0.0 if not np.isfinite(statistic) else float(statistic)
        checks = {
            "enough_annotations": annotation_count >= thresholds.minimum_annotations_per_group,
            "coverage": applicable_fraction >= thresholds.minimum_applicable_fraction,
            "top1_state_accuracy": top1_accuracy >= thresholds.minimum_top1_state_accuracy,
            "signed_direction": direction_accuracy >= thresholds.minimum_signed_direction_accuracy,
            "ordinal_monotonicity": (
                True
                if not group.is_ordinal
                else ordinal_spearman is not None
                and ordinal_spearman >= thresholds.minimum_ordinal_spearman
            ),
        }
        passed = all(checks.values())
        all_passed = all_passed and passed
        reports[group.group_id] = {
            "annotation_count": annotation_count,
            "applicable_count": len(applicable),
            "coverage_annotation_count": len(coverage_rows),
            "coverage_applicable_count": len(coverage_applicable),
            "applicable_fraction": applicable_fraction,
            "top1_state_accuracy": top1_accuracy,
            "signed_direction_accuracy": direction_accuracy,
            "ordinal_spearman": ordinal_spearman,
            "checks": checks,
            "passed": passed,
        }
    return {
        "bank_hash": bank.bank_hash,
        "evidence_store_hash": evidence.store_hash,
        "thresholds": asdict(thresholds),
        "all_groups_passed": all_passed,
        "groups": reports,
    }


def assert_audit_passed(
    report: dict[str, Any],
    expected_bank_hash: str,
    expected_evidence_store_hash: str | None = None,
) -> None:
    if report.get("bank_hash") != expected_bank_hash:
        raise ValueError("audit report belongs to a different atom bank")
    if (
        expected_evidence_store_hash is not None
        and report.get("evidence_store_hash") != expected_evidence_store_hash
    ):
        raise ValueError("audit report belongs to a different evidence database")
    if not report.get("all_groups_passed", False):
        failed = [group_id for group_id, row in report.get("groups", {}).items() if not row.get("passed")]
        raise RuntimeError(f"evidence audit has not passed for groups: {failed}")

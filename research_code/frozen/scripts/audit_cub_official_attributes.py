from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.stats import spearmanr

from dgdr.audit import distribution_audit
from dgdr.evidence import EvidenceStore
from dgdr.io import load_atom_bank, load_manifest, write_json


AttributeSets = dict[int, set[int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit CUB train evidence against official image-level attributes."
    )
    parser.add_argument("--cub-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--evidence-prefix", required=True)
    parser.add_argument("--minimum-certainty", type=int, default=3)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_positive_attributes(path: Path, minimum_certainty: int) -> AttributeSets:
    positive: AttributeSets = defaultdict(set)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            image_id, attribute_id, is_present, certainty_id = line.split()[:4]
            if int(is_present) == 1 and int(certainty_id) >= minimum_certainty:
                positive[int(image_id)].add(int(attribute_id))
    return positive


def unique_family_value(values: set[int], family: range) -> int | None:
    matches = sorted(values.intersection(family))
    return matches[0] if len(matches) == 1 else None


def family_mapper(
    family: range, attribute_to_state: dict[int, str]
) -> Callable[[set[int]], tuple[bool, str | None]]:
    def classify(values: set[int]) -> tuple[bool, str | None]:
        value = unique_family_value(values, family)
        if value is None:
            return False, None
        return True, attribute_to_state.get(value)

    return classify


def wing_conjunction(values: set[int]) -> tuple[bool, str | None]:
    colors = values.intersection(range(10, 25))
    patterns = values.intersection(range(309, 313))
    if not colors or not patterns:
        return False, None
    matches: list[str] = []
    if 21 in colors and 309 in patterns:
        matches.append("solid-black-0")
    if {21, 22}.issubset(colors) and 311 in patterns:
        matches.append("black-with-white-bars-1")
    if {16, 21}.issubset(colors) and 311 in patterns:
        matches.append("yellow-with-black-bars-2")
    return True, matches[0] if len(matches) == 1 else None


def wing_conjunction_first_group(values: set[int]) -> tuple[bool, str | None]:
    colors = values.intersection(range(10, 25))
    patterns = values.intersection(range(309, 313))
    if not colors or not patterns:
        return False, None
    matches: list[str] = []
    if 21 in colors and 309 in patterns:
        matches.append("solid-black-0")
    if {16, 21}.issubset(colors):
        matches.append("yellow-and-black-1")
    if {11, 22}.issubset(colors) and 311 in patterns:
        matches.append("brown-and-white-2")
    return True, matches[0] if len(matches) == 1 else None


def body_combo(values: set[int]) -> tuple[bool, str | None]:
    primary = unique_family_value(values, range(249, 264))
    if primary is None:
        return False, None
    if primary == 260:
        return True, "uniform-black-0"
    if primary == 255:
        return True, "bright-yellow-1"
    if primary == 250 and 52 in values:
        return True, "brown-and-white-2"
    return True, None


def safe_spearman(observed: list[float], predicted: list[float]) -> float | None:
    if len(set(observed)) < 2:
        return None
    value = float(spearmanr(observed, predicted).statistic)
    return value if np.isfinite(value) else None


def main() -> None:
    args = parse_args()
    cub_root = Path(args.cub_root)
    bank = load_atom_bank(args.bank)
    evidence = EvidenceStore.load(args.evidence_prefix)
    samples = load_manifest(args.manifest, expected_split="train")
    if tuple(sample.sample_id for sample in samples) != evidence.sample_ids:
        raise ValueError("manifest/evidence sample order mismatch")
    if bank.bank_hash != evidence.bank_hash:
        raise ValueError("bank/evidence hash mismatch")

    attributes = load_positive_attributes(
        cub_root / "attributes" / "image_attribute_labels.txt", args.minimum_certainty
    )
    mappers: dict[str, tuple[str, Callable[[set[int]], tuple[bool, str | None]]]] = {
        "wing--color-pattern--batch-0000--69b95aa9--01": (
            "proxy: official wing colors plus wing pattern",
            wing_conjunction_first_group,
        ),
        "tail--shape--batch-0000--69b95aa9--02": (
            "proxy: fan-shaped/notched exact; straight mapped to squared tail",
            family_mapper(range(74, 80), {77: "fan-shaped-0", 79: "straight-1", 76: "notched-2"}),
        ),
        "body--coloration--batch-0000--69b95aa9--03": (
            "proxy: official primary color; brown-and-white also requires white underparts",
            body_combo,
        ),
        "beak--length--batch-0000--58068026--00": (
            "exact: official bill length",
            family_mapper(range(150, 153), {152: "short-0", 150: "medium-1", 151: "long-2"}),
        ),
        "wing--color-pattern--batch-0000--58068026--01": (
            "proxy: official wing colors plus wing pattern",
            wing_conjunction,
        ),
        "body--coloration--batch-0000--58068026--02": (
            "exact: official primary color",
            family_mapper(range(249, 264), {260: "black-0", 255: "yellow-1", 250: "brown-2"}),
        ),
        "tail--shape--batch-0000--58068026--03": (
            "partial proxy: fan-shaped exact, straight mapped to squared; not-visible unavailable",
            family_mapper(range(74, 80), {77: "fan-shaped-0", 79: "straight-1"}),
        ),
        "tail--tail-shape--batch-0000--a72c67dd--01": (
            "partial proxy: fan-shaped exact, straight mapped to squared; not-visible unavailable",
            family_mapper(range(74, 80), {77: "fan-shaped-0", 79: "straight-1"}),
        ),
        "beak--beak-color--batch-0000--a72c67dd--02": (
            "exact: official bill color (dark-gray mapped to grey)",
            family_mapper(range(279, 294), {290: "black-0", 284: "dark-gray-1", 289: "orange-2"}),
        ),
        "head--head-color--batch-0000--a72c67dd--03": (
            "proxy: official crown color (head color is not provided)",
            family_mapper(range(294, 309), {305: "black-0", 299: "gray-1", 300: "yellow-2"}),
        ),
    }

    group_reports: dict[str, object] = {}
    for group_index, group in enumerate(bank.groups):
        if group.group_id not in mappers:
            group_reports[group.group_id] = {
                "mapping": "unavailable",
                "officially_labeled_count": 0,
                "in_bank_state_space_count": 0,
                "note": "No defensible direct CUB attribute mapping; use the blind image audit.",
            }
            continue
        mapping_note, mapper = mappers[group.group_id]
        state_lookup = {state.state_id: index for index, state in enumerate(group.states)}
        state_coordinates = {state.state_id: state.coordinate for state in group.states}
        probabilities = evidence.group_probability(group_index)
        signed = evidence.group_signed_margin(group_index)
        officially_labeled = 0
        in_space = 0
        top1_correct: list[bool] = []
        direction_correct: list[bool] = []
        observed_ordinal: list[float] = []
        predicted_ordinal: list[float] = []
        state_support: dict[str, int] = {state.state_id: 0 for state in group.states}
        state_top1: dict[str, list[bool]] = {state.state_id: [] for state in group.states}
        state_direction: dict[str, list[bool]] = {state.state_id: [] for state in group.states}
        for sample_index, sample in enumerate(samples):
            image_id = int(sample.sample_id.removeprefix("cub-"))
            reliable, observed_state = mapper(attributes.get(image_id, set()))
            if not reliable:
                continue
            officially_labeled += 1
            if observed_state is None or observed_state not in state_lookup:
                continue
            in_space += 1
            local_state = state_lookup[observed_state]
            state_support[observed_state] += 1
            top1_value = int(np.argmax(probabilities[sample_index])) == local_state
            direction_value = bool(signed[sample_index, local_state] > 0)
            top1_correct.append(top1_value)
            direction_correct.append(direction_value)
            state_top1[observed_state].append(top1_value)
            state_direction[observed_state].append(direction_value)
            coordinate = state_coordinates[observed_state]
            if coordinate is not None:
                observed_ordinal.append(float(coordinate))
                predicted_ordinal.append(float(evidence.ordinal_scores[sample_index, group_index]))
        group_reports[group.group_id] = {
            "mapping": mapping_note,
            "officially_labeled_count": officially_labeled,
            "in_bank_state_space_count": in_space,
            "state_space_coverage": in_space / officially_labeled if officially_labeled else None,
            "state_support": state_support,
            "per_state": {
                state.state_id: {
                    "support": state_support[state.state_id],
                    "top1_state_accuracy": (
                        float(np.mean(state_top1[state.state_id]))
                        if state_top1[state.state_id]
                        else None
                    ),
                    "signed_direction_accuracy": (
                        float(np.mean(state_direction[state.state_id]))
                        if state_direction[state.state_id]
                        else None
                    ),
                }
                for state in group.states
            },
            "top1_state_accuracy": float(np.mean(top1_correct)) if top1_correct else None,
            "signed_direction_accuracy": float(np.mean(direction_correct)) if direction_correct else None,
            "ordinal_spearman": safe_spearman(observed_ordinal, predicted_ordinal),
        }

    write_json(
        args.output,
        {
            "protocol": {
                "split": "train only",
                "sample_count": len(samples),
                "minimum_certainty": args.minimum_certainty,
                "test_images_used": 0,
                "test_labels_used": 0,
                "warning": (
                    "Proxy mappings are diagnostics, not ground truth. Ordinal correlations are "
                    "semantically interpretable only for genuinely ordered state sets."
                ),
            },
            "bank_hash": bank.bank_hash,
            "evidence_store_hash": evidence.store_hash,
            "groups": group_reports,
            "model_distribution": distribution_audit(bank, evidence),
        },
    )


if __name__ == "__main__":
    main()

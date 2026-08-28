from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_self_discovered_multishot_full_test import paired_delta_statistics, parse_shots


def test_parse_shots_sorts_and_deduplicates() -> None:
    assert parse_shots("8, 2,1,2,4") == [1, 2, 4, 8]


@pytest.mark.parametrize("value", ["", "0,2", "-1,2"])
def test_parse_shots_rejects_nonpositive_or_empty(value: str) -> None:
    with pytest.raises(ValueError):
        parse_shots(value)


def test_paired_delta_statistics_tracks_rescues_and_harms() -> None:
    rows = [
        {"query_id": "q1", "condition": "rices", "correct": False},
        {"query_id": "q1", "condition": "self_discovered", "correct": True},
        {"query_id": "q2", "condition": "rices", "correct": True},
        {"query_id": "q2", "condition": "self_discovered", "correct": False},
        {"query_id": "q3", "condition": "rices", "correct": False},
        {"query_id": "q3", "condition": "self_discovered", "correct": True},
        {"query_id": "q4", "condition": "rices", "correct": True},
        {"query_id": "q4", "condition": "self_discovered", "correct": True},
    ]
    result = paired_delta_statistics(rows, bootstraps=2_000, seed=7)
    assert result["rescued"] == 2
    assert result["harmed"] == 1
    assert result["tied"] == 1
    assert result["accuracy_delta"] == pytest.approx(0.25)
    assert result["paired_bootstrap_95_ci"][0] <= 0.25
    assert result["paired_bootstrap_95_ci"][1] >= 0.25


def test_paired_delta_statistics_rejects_unpaired_rows() -> None:
    with pytest.raises(ValueError):
        paired_delta_statistics(
            [{"query_id": "q1", "condition": "rices", "correct": True}],
            bootstraps=10,
        )

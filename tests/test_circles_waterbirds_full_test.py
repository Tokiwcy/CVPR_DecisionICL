from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_circles_waterbirds_full_test import (
    CIRCLES_PROMPT_ORDER,
    REFERENCE_PROMPT_ORDER,
    circles_budget,
    parse_attribute,
    parse_caption,
)


def test_circles_and_reference_order_are_explicitly_not_equivalent() -> None:
    assert CIRCLES_PROMPT_ORDER == "descending_similarity_highest_first"
    assert REFERENCE_PROMPT_ORDER == (
        "ascending_similarity_most_similar_nearest_query"
    )
    assert CIRCLES_PROMPT_ORDER != REFERENCE_PROMPT_ORDER


def test_circles_budget_preserves_total_and_favors_original_on_odd() -> None:
    assert circles_budget(1) == (1, 0)
    assert circles_budget(3) == (2, 1)
    assert circles_budget(8) == (4, 4)
    for total in range(1, 10):
        original, composed = circles_budget(total)
        assert original + composed == total
        assert original >= composed


def test_parsers_accept_official_style_sections() -> None:
    assert parse_attribute("### Attributes\n1. Habitat") == "Habitat"
    assert (
        parse_caption('{"Target Image Description": "a bird on dry grass"}')
        == "a bird on dry grass"
    )

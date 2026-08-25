from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_failure_contrastive_discovery import (
    balanced_folds,
    enabled_pairs_from_validation_summary,
    identify_failure_slices,
    nearest_ids,
    rule_text,
    summarize_full_routed,
    summarize_frozen_test,
    summarize_oof,
    validate_rule,
)


class FailureContrastiveDiscoveryTest(unittest.TestCase):
    def test_failure_slices_use_only_confusions(self) -> None:
        rows = [
            {"gold": "a", "prediction": "b", "group": "forbidden"},
            {"gold": "a", "prediction": "b", "group": "different"},
            {"gold": "c", "prediction": "b", "group": "forbidden"},
            {"gold": "a", "prediction": "a", "group": "forbidden"},
        ]
        self.assertEqual(
            identify_failure_slices(rows, min_count=2, top_n=2),
            [{"gold_label": "a", "confused_with": "b", "count": 2}],
        )

    def test_balanced_folds_are_deterministic_and_balanced(self) -> None:
        first = balanced_folds([f"x-{i}" for i in range(11)], 4, 7)
        second = balanced_folds([f"x-{i}" for i in reversed(range(11))], 4, 7)
        self.assertEqual(first, second)
        counts = np.bincount(list(first.values()), minlength=4)
        self.assertLessEqual(int(counts.max() - counts.min()), 1)

    def test_nearest_ids_uses_cosine_similarity(self) -> None:
        values = {
            "a": np.array([1.0, 0.0]),
            "b": np.array([0.8, 0.2]),
            "c": np.array([0.0, 1.0]),
        }
        self.assertEqual(nearest_ids(["c", "b", "a"], values, np.array([1.0, 0.0]), 2), ["a", "b"])

    def test_validate_rule_rejects_wrong_labels(self) -> None:
        value = {
            "gold_label": "wrong",
            "confused_with": "b",
            "shared_nondiagnostic_features": [],
            "missed_gold_cues": ["cue"],
            "confuser_cues": ["other"],
            "failure_hypothesis": "hypothesis",
            "corrective_rule": "inspect cue",
            "avoid_using": [],
            "confidence": 0.7,
        }
        with self.assertRaisesRegex(ValueError, "gold_label"):
            validate_rule(value, {"gold_label": "a", "confused_with": "b"})

    def test_validate_rule_rejects_empty_diagnosis(self) -> None:
        value = {
            "gold_label": "a",
            "confused_with": "b",
            "shared_nondiagnostic_features": [],
            "missed_gold_cues": [],
            "confuser_cues": [],
            "failure_hypothesis": "",
            "corrective_rule": "",
            "avoid_using": [],
            "confidence": 0.0,
        }
        with self.assertRaisesRegex(ValueError, "missed_gold_cues"):
            validate_rule(value, {"gold_label": "a", "confused_with": "b"})

    def test_oof_summary_separates_rescue_and_harm(self) -> None:
        rows = [
            {
                "slice_gold": "a", "slice_confused_with": "b", "role": "target_error",
                "gold": "a", "baseline_prediction": "b", "guided_prediction": "a",
            },
            {
                "slice_gold": "a", "slice_confused_with": "b", "role": "gold_correct_control",
                "gold": "a", "baseline_prediction": "a", "guided_prediction": "b",
            },
            {
                "slice_gold": "a", "slice_confused_with": "b", "role": "confuser_correct_control",
                "gold": "b", "baseline_prediction": "b", "guided_prediction": "b",
            },
        ]
        summary = summarize_oof(rows)
        self.assertEqual(summary["target_errors"]["rescued"], 1)
        self.assertEqual(summary["correct_controls"]["harmed"], 1)

    def test_label_only_placebo_omits_learned_cues(self) -> None:
        rule = {
            "gold_label": "a", "confused_with": "b",
            "missed_gold_cues": ["secret gold cue"],
            "confuser_cues": ["secret confuser cue"],
            "corrective_rule": "secret rule", "avoid_using": ["secret avoid"],
        }
        prompt = rule_text(rule, "label_only")
        self.assertIn("a versus b", prompt)
        self.assertNotIn("secret", prompt)

    def test_full_routed_summary_counts_unchanged_rows(self) -> None:
        rows = [
            {
                "slice_gold": "a", "slice_confused_with": "b", "role": "target_error",
                "routed": True, "fold": 0, "gold": "a", "baseline_prediction": "b", "guided_prediction": "a",
            },
            {
                "slice_gold": "", "slice_confused_with": "", "role": "unrouted",
                "routed": False, "fold": -1, "gold": "c", "baseline_prediction": "c", "guided_prediction": "c",
            },
        ]
        summary = summarize_full_routed(rows)
        self.assertEqual(summary["all"]["guided_accuracy"], 1.0)
        self.assertEqual(summary["unrouted"]["count"], 1)

    def test_harm_penalty_can_disable_pair(self) -> None:
        rows = []
        for fold in range(4):
            rows.extend(
                [
                    {
                        "slice_gold": "a", "slice_confused_with": "b", "role": "target_error",
                        "routed": True, "fold": fold, "gold": "a",
                        "baseline_prediction": "b", "guided_prediction": "a",
                    },
                    {
                        "slice_gold": "a", "slice_confused_with": "b", "role": "routed_correct",
                        "routed": True, "fold": fold, "gold": "b",
                        "baseline_prediction": "b", "guided_prediction": "a",
                    },
                ]
            )
        beta_one = summarize_full_routed(rows, punish_beta=1.0)
        beta_two = summarize_full_routed(rows, punish_beta=2.0)
        self.assertTrue(all(beta_one["cross_fitted_pair_gate"]["decisions"].values()) is False)
        self.assertFalse(any(beta_two["cross_fitted_pair_gate"]["decisions"].values()))

    def test_pair_gate_requires_all_folds(self) -> None:
        summary = {
            "cross_fitted_pair_gate": {
                "decisions": {
                    "a<->b/fold-0": True,
                    "a<->b/fold-1": True,
                    "c<->d/fold-0": True,
                    "c<->d/fold-1": False,
                }
            }
        }
        self.assertEqual(enabled_pairs_from_validation_summary(summary), {("a", "b")})

    def test_frozen_test_summary_reports_groups(self) -> None:
        rows = [
            {
                "gold": "a", "baseline_prediction": "b", "guided_prediction": "a",
                "routed": True, "evaluation_group": "a_on_x", "evaluation_conflict": True,
            },
            {
                "gold": "b", "baseline_prediction": "b", "guided_prediction": "b",
                "routed": False, "evaluation_group": "b_on_y", "evaluation_conflict": False,
            },
        ]
        summary = summarize_frozen_test(rows, {("a", "b")})
        self.assertEqual(summary["all"]["guided_accuracy"], 1.0)
        self.assertEqual(summary["worst_group_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()

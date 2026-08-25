from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_three_shot_weighted.py"
SPEC = importlib.util.spec_from_file_location("run_three_shot_weighted", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(MODULE_PATH.parent))
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def demo(identifier: str, label: str, similarity: float, diagnostic: bool):
    return {
        "id": identifier,
        "label": label,
        "similarity": similarity,
        "diagnostic": diagnostic,
    }


class ThreeShotSelectionTests(unittest.TestCase):
    def test_keeps_rices_when_coverage_already_exists(self):
        rices = [
            demo("l1", "landbird", 0.9, False),
            demo("w1", "waterbird", 0.88, True),
            demo("l2", "landbird", 0.8, False),
        ]
        best = demo("bw", "waterbird", 0.88, True)
        chosen, policy = MODULE.choose_hard_coverage_context(rices, best)
        self.assertEqual([row["id"] for row in chosen], ["l1", "w1", "l2"])
        self.assertEqual(policy["action"], "keep_rices")

    def test_global_hard_coverage_replaces_lowest_similarity_slot(self):
        rices = [
            demo("l1", "landbird", 0.9, False),
            demo("w1", "waterbird", 0.88, False),
            demo("l2", "landbird", 0.82, False),
        ]
        best = demo("bw", "waterbird", 0.80, True)
        chosen, policy = MODULE.choose_hard_coverage_context(rices, best)
        self.assertEqual([row["id"] for row in chosen], ["bw", "w1", "l1"])
        self.assertEqual([row["label"] for row in chosen], ["waterbird", "waterbird", "landbird"])
        self.assertAlmostEqual(policy["similarity_loss"], 0.02)

    def test_additive_weighting_is_global_and_can_select_multiple_diagnostics(self):
        candidates = [
            demo("n1", "landbird", 0.90, False),
            demo("n2", "landbird", 0.89, False),
            demo("n3", "landbird", 0.88, False),
            demo("d1", "waterbird", 0.87, True),
            demo("d2", "waterbird", 0.86, True),
        ]
        unweighted = MODULE.choose_weighted_context(candidates, 0.0)
        weighted = MODULE.choose_weighted_context(candidates, 0.05)
        self.assertEqual([row["id"] for row in unweighted], ["n3", "n2", "n1"])
        self.assertEqual([row["id"] for row in weighted], ["d2", "d1", "n1"])


if __name__ == "__main__":
    unittest.main()

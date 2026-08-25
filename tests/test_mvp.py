from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_mvp.py"
SPEC = importlib.util.spec_from_file_location("run_mvp", MODULE_PATH)
assert SPEC and SPEC.loader
MVP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MVP
SPEC.loader.exec_module(MVP)


class SelectionTests(unittest.TestCase):
    def make_data(self):
        units = []
        vectors = []
        for label, y in (("landbird", 0), ("waterbird", 1)):
            for place in (0, 1):
                for number in range(5):
                    units.append(
                        {
                            "unit_id": f"c-{label}-{place}-{number}",
                            "role": "candidate",
                            "image": f"/{label}-{place}-{number}.jpg",
                            "label": label,
                            "group": f"{label}_on_{'water' if place else 'land'}",
                            "y": y,
                            "place": place,
                        }
                    )
                    angle = 0.04 * number + 0.20 * place + 0.10 * y
                    vectors.append([np.cos(angle), np.sin(angle)])
        for label, y in (("landbird", 0), ("waterbird", 1)):
            for place in (0, 1):
                units.append(
                    {
                        "unit_id": f"q-{label}-{place}",
                        "role": "validation_query",
                        "image": f"/q-{label}-{place}.jpg",
                        "label": label,
                        "group": f"{label}_on_{'water' if place else 'land'}",
                        "y": y,
                        "place": place,
                    }
                )
                vectors.append([1.0, 0.0])
        matrix = np.asarray(vectors, dtype=np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        return units, matrix

    def test_constraints_and_matching(self):
        units, features = self.make_data()
        rows, audit = MVP.build_selections(
            units,
            features,
            query_role="validation_query",
            queries_per_group=1,
            seed=7,
            low_quantile=0.30,
        )
        self.assertEqual(len(rows), 4 * len(MVP.CONDITIONS))
        self.assertEqual(audit["queries"], 4)
        by_condition = {}
        for row in rows:
            by_condition.setdefault(row["condition"], []).append(row)
        self.assertTrue(
            all(row["diagnostic_rate"] == 1.0 for row in by_condition["B_high_diagnostic"])
        )
        self.assertTrue(
            all(row["diagnostic_rate"] == 0.0 for row in by_condition["A_matched_non_diagnostic"])
        )
        self.assertLess(audit["B_vs_A_matched_abs_similarity_delta"]["mean"], 0.03)

    def test_all_queries_preserves_natural_groups(self):
        units, features = self.make_data()
        rows, audit = MVP.build_selections(
            units,
            features,
            query_role="validation_query",
            queries_per_group=None,
            seed=7,
            low_quantile=0.30,
        )
        self.assertEqual(len(rows), 4 * len(MVP.CONDITIONS))
        self.assertIsNone(audit["queries_per_group"])
        self.assertEqual(set(audit["query_group_counts"].values()), {1})


class MetricTests(unittest.TestCase):
    def test_exact_mcnemar(self):
        self.assertEqual(MVP.exact_mcnemar_p(0, 0), 1.0)
        self.assertEqual(MVP.exact_mcnemar_p(1, 0), 1.0)
        self.assertAlmostEqual(MVP.exact_mcnemar_p(5, 0), 0.0625)


if __name__ == "__main__":
    unittest.main()

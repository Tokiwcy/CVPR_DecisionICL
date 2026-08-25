from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_multiclass_weighted.py"
SPEC = importlib.util.spec_from_file_location("run_multiclass_weighted", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MulticlassWeightedTests(unittest.TestCase):
    def test_partitioned_weighted_candidates_match_full_sort(self) -> None:
        rng = np.random.default_rng(20260825)
        similarities = rng.normal(size=1000).astype(np.float32)
        diagnostics = rng.random(1000) < 0.08
        ids = np.asarray([f"id-{index:04d}" for index in range(1000)])
        diagnostic_positions = np.flatnonzero(diagnostics)
        nondiagnostic_positions = np.flatnonzero(~diagnostics)
        diagnostic_top = diagnostic_positions[
            MODULE._top_indices(
                similarities[diagnostic_positions],
                similarities[diagnostic_positions],
                ids[diagnostic_positions],
            )
        ]
        nondiagnostic_top = nondiagnostic_positions[
            MODULE._top_indices(
                similarities[nondiagnostic_positions],
                similarities[nondiagnostic_positions],
                ids[nondiagnostic_positions],
            )
        ]
        candidates = np.concatenate((diagnostic_top, nondiagnostic_top))
        for _, lambda_value in MODULE.LAMBDA_GRID:
            full_scores = similarities + lambda_value * diagnostics.astype(np.float32)
            expected = np.lexsort((ids, -similarities, -full_scores))[: MODULE.K]
            candidate_scores = (
                similarities[candidates]
                + lambda_value * diagnostics[candidates].astype(np.float32)
            )
            actual = candidates[
                MODULE._top_indices(
                    candidate_scores, similarities[candidates], ids[candidates]
                )
            ]
            self.assertEqual(list(actual), list(expected))

    def test_weight_can_promote_multiple_model_error_demos(self) -> None:
        similarities = np.asarray([0.90, 0.89, 0.88, 0.80, 0.79], dtype=np.float32)
        diagnostic = np.asarray([0, 0, 0, 1, 1], dtype=np.float32)
        ids = np.asarray(["a", "b", "c", "d", "e"])
        selected = MODULE._top_indices(similarities + 0.15 * diagnostic, similarities, ids)
        self.assertEqual(list(selected), [3, 4, 0])

    def test_top_indices_uses_runtime_prompt_shots(self) -> None:
        previous = MODULE.K
        MODULE.K = 2
        try:
            similarities = np.asarray([0.3, 0.9, 0.8], dtype=np.float32)
            ids = np.asarray(["a", "b", "c"])
            selected = MODULE._top_indices(similarities, similarities, ids)
            self.assertEqual(list(selected), [1, 2])
        finally:
            MODULE.K = previous

    def test_config_fingerprint_changes_with_prompt_shots(self) -> None:
        base = {
            "manifest": "manifest.jsonl",
            "labels": ["a", "b"],
            "target_model": {},
            "similarity_model": {},
        }
        one = MODULE._config_fingerprint({**base, "prompt_shots": 1})
        four = MODULE._config_fingerprint({**base, "prompt_shots": 4})
        self.assertNotEqual(one, four)

    def test_append_jsonl_preserves_existing_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cache.jsonl"
            MODULE._append_jsonl(path, [{"context_key": "a"}])
            MODULE._append_jsonl(path, [{"context_key": "b"}])
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows, [{"context_key": "a"}, {"context_key": "b"}])

    def test_macro_and_worst_class_metrics_are_reported(self) -> None:
        rows = [
            {"gold": "a", "correct": True},
            {"gold": "a", "correct": True},
            {"gold": "b", "correct": False},
        ]
        result = MODULE.metrics(rows, ["a", "b"])
        self.assertAlmostEqual(result["overall_accuracy"], 2 / 3)
        self.assertAlmostEqual(result["macro_accuracy"], 0.5)
        self.assertEqual(result["worst_class_accuracy"], 0.0)

    def test_manifest_diagnostics_and_group_metrics(self) -> None:
        train = [
            SimpleNamespace(id="a", metadata={"diagnostic": False}),
            SimpleNamespace(id="b", metadata={"diagnostic": True}),
        ]
        values, definition = MODULE._diagnostics_for_train(
            {
                "diagnostic": {
                    "kind": "manifest_boolean",
                    "metadata_key": "diagnostic",
                    "definition": "counter-spurious group",
                }
            },
            train,
            [],
        )
        self.assertEqual(values.tolist(), [False, True])
        self.assertEqual(definition, "counter-spurious group")

        rows = [
            {"gold": "a", "correct": True, "group": "a_x", "conflict": False},
            {"gold": "a", "correct": False, "group": "a_y", "conflict": True},
            {"gold": "b", "correct": True, "group": "b_x", "conflict": True},
            {"gold": "b", "correct": True, "group": "b_y", "conflict": False},
        ]
        result = MODULE.metrics(rows, ["a", "b"])
        self.assertEqual(result["mean_group_accuracy"], 0.75)
        self.assertEqual(result["worst_group_accuracy"], 0.0)
        self.assertEqual(result["aligned_accuracy"], 1.0)
        self.assertEqual(result["conflict_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()

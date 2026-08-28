from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from decision_atoms.types import DecisionAtom, Sample
from run_self_discovered_diagnostic_retrieval import (
    balanced_rest_support,
    characterize_atoms,
    diagnostic_scores,
    extract_json_array,
    inference_query_projection,
    metadata_free_projection,
    parse_atoms,
    grouped_diagnostic_scores,
    select_context_rows,
    select_unsupervised_lambda,
    top_indices,
    utility_gate_lambda,
)


class SelfDiscoveredDiagnosticRetrievalTest(unittest.TestCase):
    def test_balanced_rest_support_covers_other_classes(self) -> None:
        samples = [
            Sample(f"{label}-{index}", "train", "x", label)
            for label in ("a", "b", "c", "d")
            for index in range(3)
        ]
        selected = balanced_rest_support(
            samples, ["a", "b", "c", "d"],
            anchor_label="a", count=3, seed=7,
        )
        self.assertEqual({sample.label for sample in selected}, {"b", "c", "d"})

    def test_metadata_free_projection_removes_group_and_diagnostic_fields(self) -> None:
        sample = Sample(
            "x", "train", "image.jpg", "a", group="a_on_water",
            metadata={"diagnostic": True, "spurious_attribute": "water"},
        )
        projected = metadata_free_projection([sample])[0]
        self.assertIsNone(projected.group)
        self.assertEqual(projected.metadata, {})

    def test_inference_projection_also_removes_query_label(self) -> None:
        sample = Sample("x", "test", "image.jpg", "secret", group="secret_group")
        projected = inference_query_projection([sample])[0]
        self.assertEqual(projected.label, "")
        self.assertIsNone(projected.group)

    def test_extract_and_parse_atoms(self) -> None:
        text = 'prefix [{"concept":"scene","a_side":"dry","b_side":"wet","role":"context"}] suffix'
        self.assertEqual(extract_json_array(text)[0]["concept"], "scene")
        atoms = parse_atoms(text, "landbird", "waterbird", bootstrap=2)
        self.assertEqual(len(atoms), 1)
        self.assertEqual(atoms[0].source_mode, "shift_audit")
        self.assertTrue(atoms[0].provenance["discovered_without_group_metadata"])
        self.assertEqual(atoms[0].provenance["vlm_semantic_role"], "context")

    def test_extract_repairs_missing_atom_object_braces(self) -> None:
        malformed = (
            '[{"concept":"ear","a_side":"up","b_side":"down"},'
            '"body shape","a_side":"compact","b_side":"long"],'
            '{"concept":"tail","a_side":"short","b_side":"long"}]'
        )
        parsed = extract_json_array(malformed)
        self.assertEqual([row["concept"] for row in parsed], ["ear", "body shape", "tail"])

    def test_train_validation_shift_separates_core_and_shortcut(self) -> None:
        atoms = [
            DecisionAtom("core", "a", "b", "shape", "round", "long", "shift_audit", 0),
            DecisionAtom("shortcut", "a", "b", "scene", "inside", "outside", "shift_audit", 0),
        ]
        samples = []
        targets = []
        for split in ("train", "validation"):
            for label in ("a", "b"):
                for index in range(4):
                    samples.append(Sample(f"{split}-{label}-{index}", split, "x", label))
                    targets.append(1.0 if label == "a" else -1.0)
        targets = np.asarray(targets)
        core = targets.copy()
        shortcut = targets.copy()
        # Held-out validation makes the scene independent of the class.
        shortcut[8:] = np.asarray([1, -1, 1, -1, 1, -1, 1, -1], dtype=float)
        evidence = np.stack([core, shortcut], axis=1)
        quality = characterize_atoms(
            evidence,
            samples,
            atoms,
            core_min_auc=0.6,
            core_max_train_validation_gap=0.1,
            shortcut_min_train_auc=0.8,
            shortcut_max_validation_strength=0.05,
            shortcut_min_gap=0.3,
        )
        self.assertEqual([row["role"] for row in quality], ["core", "shortcut"])
        self.assertTrue(all(not row["group_or_spurious_metadata_used"] for row in quality))

    def test_semantic_context_fallback_handles_persistent_shortcut(self) -> None:
        atoms = [
            DecisionAtom(
                "core", "a", "b", "shape", "round", "long", "shift_audit", 0,
                provenance={"vlm_semantic_role": "object"},
            ),
            DecisionAtom(
                "context", "a", "b", "scene", "inside", "outside", "shift_audit", 0,
                provenance={"vlm_semantic_role": "context"},
            ),
        ]
        samples = [
            Sample(f"{split}-{label}-{index}", split, "x", label)
            for split in ("train", "validation")
            for label in ("a", "b")
            for index in range(4)
        ]
        target = np.asarray([1.0 if sample.label == "a" else -1.0 for sample in samples])
        quality = characterize_atoms(
            np.stack([target, target], axis=1),
            samples,
            atoms,
            core_min_auc=0.6,
            core_max_train_validation_gap=0.1,
            shortcut_min_train_auc=0.8,
            shortcut_max_validation_strength=0.05,
            shortcut_min_gap=0.3,
            semantic_shortcut_fallback=True,
        )
        self.assertEqual([row["role"] for row in quality], ["core", "shortcut"])
        self.assertEqual(
            quality[1]["role_source"],
            "vlm_semantic_context_prior+train_correlation",
        )

    def test_counter_shortcut_demo_gets_highest_score(self) -> None:
        atoms = [
            DecisionAtom("core", "a", "b", "shape", "round", "long", "shift_audit", 0),
            DecisionAtom("shortcut", "a", "b", "scene", "inside", "outside", "shift_audit", 0),
        ]
        samples = [
            Sample("aligned-a", "train", "x", "a"),
            Sample("counter-a", "train", "x", "a"),
            Sample("aligned-b", "train", "x", "b"),
            Sample("counter-b", "train", "x", "b"),
        ]
        evidence = np.asarray(
            [
                [1.0, 1.0],
                [1.2, -1.0],
                [-1.0, -1.0],
                [-1.2, 1.0],
            ]
        )
        quality = [
            {**atoms[0].to_dict(), "role": "core", "orientation": 1.0, "weight": 1.0},
            {**atoms[1].to_dict(), "role": "shortcut", "orientation": 1.0, "weight": 1.0},
        ]
        scores, raw, _, _ = diagnostic_scores(evidence, samples, quality)
        self.assertGreater(raw[1], raw[0])
        self.assertGreater(raw[3], raw[2])
        self.assertEqual(set(np.flatnonzero(scores == scores.max())), {1, 3})

    def test_multiclass_one_vs_rest_scores_each_anchor(self) -> None:
        labels = ("a", "b", "c")
        samples = [
            Sample(f"{split}-{label}-{index}", split, "x", label)
            for split in ("train", "validation")
            for label in labels
            for index in range(4)
        ]
        atoms = []
        columns = []
        for anchor in labels:
            atoms.extend(
                [
                    DecisionAtom(f"{anchor}-core", anchor, f"not {anchor}", "shape", "yes", "no", "shift_audit", 0),
                    DecisionAtom(f"{anchor}-shortcut", anchor, f"not {anchor}", "scene", "yes", "no", "shift_audit", 0),
                ]
            )
            core = np.asarray([1.0 if sample.label == anchor else -1.0 for sample in samples])
            shortcut = core.copy()
            for index, sample in enumerate(samples):
                if sample.split == "validation":
                    shortcut[index] = 1.0 if int(sample.id.rsplit("-", 1)[1]) % 2 == 0 else -1.0
            columns.extend([core, shortcut])
        evidence = np.stack(columns, axis=1)
        quality, scores, _, _, _, status = grouped_diagnostic_scores(
            evidence,
            samples,
            atoms,
            core_min_auc=0.6,
            core_max_train_validation_gap=0.1,
            shortcut_min_train_auc=0.8,
            shortcut_max_validation_strength=0.05,
            shortcut_min_gap=0.3,
        )
        self.assertEqual({row["role"] for row in quality}, {"core", "shortcut"})
        self.assertTrue(all(value["supported"] for value in status.values()))
        self.assertTrue(np.isfinite(scores).all())

    def test_weighted_top_k_uses_continuous_learned_score(self) -> None:
        similarities = np.asarray([0.90, 0.89, 0.88, 0.87])
        scores = np.asarray([0.0, 0.0, 1.0, 0.0])
        ids = np.asarray(["a", "b", "c", "d"])
        baseline = top_indices(similarities, scores, ids, k=2, lambda_value=0.0)
        weighted = top_indices(similarities, scores, ids, k=2, lambda_value=0.05)
        self.assertEqual(list(baseline), [0, 1])
        self.assertIn(2, weighted)

    def test_context_selection_deduplicates_equal_lambdas(self) -> None:
        queries = [Sample("query", "validation", "x", "a")]
        train = [
            Sample("a", "train", "x", "a"),
            Sample("b", "train", "x", "b"),
        ]
        rows = select_context_rows(
            queries,
            train,
            np.asarray([[1.0, 0.0]]),
            np.asarray([[1.0, 0.0], [0.0, 1.0]]),
            np.asarray([0.0, 1.0]),
            (0.0, 0.0),
            k=1,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["lambda"], 0.0)

    def test_context_selection_can_preserve_rices_label_histogram(self) -> None:
        queries = [Sample("query", "validation", "x", "a")]
        train = [
            Sample("a1", "train", "x", "a"),
            Sample("b1", "train", "x", "b"),
            Sample("b2", "train", "x", "b"),
            Sample("a2", "train", "x", "a"),
        ]
        rows = select_context_rows(
            queries,
            train,
            np.asarray([[1.0, 0.0]]),
            np.asarray([[0.90, 0.0], [0.89, 0.0], [0.88, 0.0], [0.87, 0.0]]),
            np.asarray([0.0, 0.0, 1.0, 0.0]),
            (0.0, 0.1),
            k=2,
            preserve_baseline_label_counts=True,
        )
        baseline_labels = sorted(demo["label"] for demo in rows[0]["demos"])
        regularized_labels = sorted(demo["label"] for demo in rows[1]["demos"])
        self.assertEqual(baseline_labels, ["a", "b"])
        self.assertEqual(regularized_labels, baseline_labels)

    def test_lambda_selection_uses_no_accuracy_or_metadata(self) -> None:
        rows = []
        for value, similarity, score in ((0.0, 0.8, 0.1), (0.1, 0.79, 0.7), (0.2, 0.75, 0.9)):
            rows.append(
                {
                    "lambda": value,
                    "mean_similarity": similarity,
                    "mean_learned_diagnostic_score": score,
                    "gold": "must-not-be-used",
                    "group": "must-not-be-used",
                }
            )
        selected = select_unsupervised_lambda(rows, max_similarity_drop=0.02)
        self.assertEqual(selected["selected_lambda"], 0.1)
        self.assertIn("no query labels", selected["selection_rule"])

    def test_utility_gate_rejects_validation_degradation_without_groups(self) -> None:
        summary = {
            "rices": {"accuracy": 0.9},
            "self_discovered": {"accuracy": 0.8},
            "paired": {"rescued": 0, "harmed": 1},
            "group": "must-not-be-used",
            "diagnostic": "must-not-be-used",
        }
        gate = utility_gate_lambda(summary, candidate_lambda=0.1)
        self.assertFalse(gate["accepted"])
        self.assertEqual(gate["selected_lambda"], 0.0)
        self.assertFalse(gate["group_conflict_or_diagnostic_metadata_used"])

    def test_utility_gate_keeps_accuracy_neutral_candidate(self) -> None:
        summary = {
            "rices": {"accuracy": 0.9},
            "self_discovered": {"accuracy": 0.9},
            "paired": {"rescued": 1, "harmed": 1},
        }
        gate = utility_gate_lambda(summary, candidate_lambda=0.1)
        self.assertTrue(gate["accepted"])
        self.assertEqual(gate["selected_lambda"], 0.1)

    def test_utility_gate_can_penalize_harms_asymmetrically(self) -> None:
        summary = {
            "rices": {"accuracy": 0.8},
            "self_discovered": {"accuracy": 0.81},
            "paired": {"rescued": 2, "harmed": 1},
        }
        gate = utility_gate_lambda(
            summary, candidate_lambda=0.1, harm_penalty=2.5
        )
        self.assertFalse(gate["accepted"])
        self.assertEqual(gate["asymmetric_flip_utility"]["value"], -0.5)


if __name__ == "__main__":
    unittest.main()

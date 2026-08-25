from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import prepare_spurious_tasks as module


class PrepareSpuriousTasksTests(unittest.TestCase):
    @staticmethod
    def _make_spawrious(source: Path, count: int) -> None:
        protocol = module.SPAWRIOUS_PROTOCOLS["o2o_easy"]
        labels = ["bulldog", "dachshund", "labrador", "corgi"]
        for label, aligned, test_attribute in zip(
            labels, protocol["aligned"], protocol["test"], strict=True
        ):
            for environment in (0, 1):
                for attribute in {aligned, protocol["filler"], test_attribute}:
                    folder = (
                        source
                        / "spawrious224"
                        / str(environment)
                        / str(attribute)
                        / label
                    )
                    folder.mkdir(parents=True, exist_ok=True)
                    for index in range(count):
                        (folder / f"{index}.jpg").touch()

    def test_spawrious_pilot_manifest_has_balanced_query_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            self._make_spawrious(source, 5)
            module.prepare_spawrious(
                SimpleNamespace(
                    source=source,
                    output=output,
                    challenge="o2o_easy",
                    train_aligned_per_environment=2,
                    train_diagnostic_per_environment=1,
                    query_per_group=1,
                )
            )
            rows = [
                json.loads(line)
                for line in (output / "manifest.jsonl").read_text().splitlines()
            ]
            split_counts = {
                split: sum(row["split"] == split for row in rows)
                for split in ("train", "validation", "test")
            }
            self.assertEqual(split_counts, {"train": 24, "validation": 8, "test": 8})
            train = [row for row in rows if row["split"] == "train"]
            self.assertEqual(sum(row["metadata"]["diagnostic"] for row in train), 8)
            for split in ("validation", "test"):
                queries = [row for row in rows if row["split"] == split]
                self.assertEqual(sum(row["metadata"]["conflict"] for row in queries), 4)

    def test_spawrious_full_manifest_uses_all_remaining_ood_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            self._make_spawrious(source, 10)
            module.prepare_spawrious(
                SimpleNamespace(
                    source=source,
                    output=output,
                    challenge="o2o_easy",
                    train_aligned_per_environment=None,
                    train_diagnostic_per_environment=None,
                    train_aligned_environment_0=97,
                    train_diagnostic_environment_0=3,
                    train_aligned_environment_1=87,
                    train_diagnostic_environment_1=13,
                    query_per_group=1,
                    full_evaluation=True,
                    validation_per_group_per_environment=1,
                )
            )
            rows = [
                json.loads(line)
                for line in (output / "manifest.jsonl").read_text().splitlines()
            ]
            split_counts = {
                split: sum(row["split"] == split for row in rows)
                for split in ("train", "validation", "test")
            }
            self.assertEqual(split_counts, {"train": 80, "validation": 16, "test": 72})
            train = [row for row in rows if row["split"] == "train"]
            self.assertEqual(sum(row["metadata"]["diagnostic"] for row in train), 12)
            test = [row for row in rows if row["split"] == "test"]
            self.assertTrue(all(row["metadata"]["conflict"] for row in test))


if __name__ == "__main__":
    unittest.main()

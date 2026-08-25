from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import prepare_waterbirds


class PrepareWaterbirdsTest(unittest.TestCase):
    def test_builds_dual_compatible_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "waterbirds"
            source.mkdir()
            rows = [
                {
                    "img_id": "1",
                    "img_filename": "bird.jpg",
                    "y": "1",
                    "place": "0",
                    "split": "0",
                }
            ]
            with (source / "metadata.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            Image.new("RGB", (4, 4), "blue").save(source / "bird.jpg")
            output = Path(temporary) / "manifest.jsonl"

            summary = prepare_waterbirds.build_manifest(source, output)
            record = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(summary["rows"], 1)
            self.assertEqual(record["unit_id"], record["id"])
            self.assertEqual(record["role"], "candidate")
            self.assertEqual(record["label"], "waterbird")
            self.assertTrue(record["metadata"]["diagnostic"])


if __name__ == "__main__":
    unittest.main()

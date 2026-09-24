from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vision_jev.data.generate import generate_local_pilot


class LocalGenerationTest(unittest.TestCase):
    def test_small_pilot_covers_all_local_families(self) -> None:
        repository = Path(__file__).parents[2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = generate_local_pilot(
                data_root=root,
                mixture_config=repository / "configs" / "data" / "sft_120k.json",
                destination=root / "pilot.jsonl",
                samples_per_family=2,
            )
            self.assertEqual(report["questions"], 18)
            self.assertEqual(report["unique_images"], 14)
            self.assertEqual(report["task_counts"], {"choice": 12, "score": 2, "noul": 4})
            self.assertEqual(report["api_calls"], 0)
            self.assertEqual(report["project_human_annotations"], 0)
            self.assertEqual(len(report["source_counts"]), 7)


if __name__ == "__main__":
    unittest.main()

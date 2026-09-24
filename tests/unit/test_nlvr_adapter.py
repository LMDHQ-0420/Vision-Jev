from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vision_jev.data.pipeline import normalize_nlvr
from vision_jev.data.schema import validate_jsonl


class NlvrAdapterTest(unittest.TestCase):
    def test_six_permutations_share_one_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            nlvr = data_root / "raw" / "nlvr" / "extracted" / "repository" / "commit" / "nlvr"
            for upstream_split in ("train", "dev"):
                split_root = nlvr / upstream_split
                image_root = split_root / "images" / "7"
                image_root.mkdir(parents=True)
                item = {
                    "sentence": "There is a circle.",
                    "label": "true",
                    "identifier": "1-0",
                    "directory": "7",
                    "evals": {"r0": "true"},
                }
                (split_root / f"{upstream_split}.json").write_text(
                    json.dumps(item) + "\n", encoding="utf-8"
                )
                for permutation in range(6):
                    (image_root / f"{upstream_split}-1-0-{permutation}.png").touch()

            destination = data_root / "canonical.jsonl"
            self.assertEqual(normalize_nlvr(data_root, destination), 12)
            report = validate_jsonl(destination, check_assets=True)
            self.assertEqual(report.questions, 12)
            self.assertEqual(report.groups, 2)
            self.assertEqual(report.task_counts, {"noul": 12})


if __name__ == "__main__":
    unittest.main()

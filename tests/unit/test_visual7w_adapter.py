from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vision_jev.data.pipeline import normalize_visual7w
from vision_jev.data.schema import validate_jsonl


class Visual7wAdapterTest(unittest.TestCase):
    def test_pointing_options_preserve_boxes_and_shuffle_answer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "raw" / "visual7w" / "extracted"
            annotations = root / "pointing_annotations"
            images = root / "images" / "visual7w_images"
            annotations.mkdir(parents=True)
            images.mkdir(parents=True)
            (images / "v7w_7.jpg").touch()
            payload = {
                "boxes": [
                    {
                        "box_id": index,
                        "name": f"region {index}",
                        "x": index,
                        "y": 2,
                        "width": 3,
                        "height": 4,
                    }
                    for index in range(1, 5)
                ],
                "images": [
                    {
                        "image_id": 7,
                        "filename": "v7w_7.jpg",
                        "split": "train",
                        "qa_pairs": [
                            {
                                "answer": 1,
                                "multiple_choices": [2, 3, 4],
                                "qa_id": 9,
                                "type": "which",
                                "question": "Which region?",
                            }
                        ],
                    }
                ],
            }
            (annotations / "dataset_v7w_pointing.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            destination = Path(directory) / "canonical.jsonl"
            self.assertEqual(normalize_visual7w(Path(directory), destination), 1)
            report = validate_jsonl(destination, check_assets=True)
            self.assertEqual(report.task_counts, {"choice": 1})
            sample = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(sample["target"], "box_1")
            self.assertTrue(all("box" in option for option in sample["options"]))
            self.assertEqual(
                {option["text"] for option in sample["options"]}, {"candidate region"}
            )


if __name__ == "__main__":
    unittest.main()

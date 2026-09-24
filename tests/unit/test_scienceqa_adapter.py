from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vision_jev.data.pipeline import normalize_scienceqa
from vision_jev.data.schema import validate_jsonl


class ScienceQaAdapterTest(unittest.TestCase):
    def test_only_multimodal_questions_are_emitted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            repository = (
                data_root
                / "raw"
                / "scienceqa"
                / "extracted"
                / "repository"
                / "ScienceQA-commit"
                / "data"
                / "scienceqa"
            )
            repository.mkdir(parents=True)
            problems = {
                "1": {
                    "question": "Which answer?",
                    "choices": ["A", "B"],
                    "answer": 1,
                    "hint": "Use the figure.",
                    "image": "image.png",
                    "split": "train",
                    "grade": "grade2",
                    "subject": "science",
                    "category": "test",
                },
                "2": {
                    "question": "Text only?",
                    "choices": ["A", "B"],
                    "answer": 0,
                    "hint": "",
                    "image": None,
                    "split": "train",
                },
            }
            (repository / "problems.json").write_text(json.dumps(problems), encoding="utf-8")
            image = (
                data_root
                / "raw"
                / "scienceqa"
                / "extracted"
                / "train_images"
                / "train"
                / "1"
                / "image.png"
            )
            image.parent.mkdir(parents=True)
            image.touch()
            destination = data_root / "canonical.jsonl"
            self.assertEqual(normalize_scienceqa(data_root, destination), 1)
            report = validate_jsonl(destination, check_assets=True)
            self.assertEqual(report.questions, 1)
            sample = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(sample["target"], "choice_1")
            self.assertEqual(sample["state_text"], "Use the figure.")


if __name__ == "__main__":
    unittest.main()

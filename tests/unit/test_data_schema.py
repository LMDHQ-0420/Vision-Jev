from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vision_jev.data.schema import DataValidationError, validate_jsonl


class DataSchemaTest(unittest.TestCase):
    def test_examples_validate_as_complete_questions(self) -> None:
        report = validate_jsonl(Path(__file__).parents[2] / "data" / "samples" / "example.jsonl")
        self.assertEqual(report.questions, 3)
        self.assertEqual(report.task_counts, {"choice": 1, "noul": 1, "score": 1})

    def test_teacher_label_cannot_enter_test(self) -> None:
        sample = {
            "schema_version": 1,
            "sample_id": "x",
            "group_id": "g",
            "source": "s",
            "source_version": "1",
            "license": "x",
            "split": "test",
            "task_type": "noul",
            "state_text": "",
            "question": "q",
            "options": [],
            "target_kind": "binary",
            "target": True,
            "label_origin": "teacher_only",
            "language": "en",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.jsonl"
            path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            with self.assertRaises(DataValidationError):
                validate_jsonl(path)


if __name__ == "__main__":
    unittest.main()

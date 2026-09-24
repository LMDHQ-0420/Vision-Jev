from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vision_jev.data import validate_jsonl
from vision_jev.data.pipeline import (
    build_model_input,
    open_qa_sample,
    sanitize_weblinx_candidate_text,
    stable_negatives,
)


class DataPipelineTest(unittest.TestCase):
    def test_candidate_generation_is_stable(self) -> None:
        pool = ["red", "blue", "green", "yellow"]
        self.assertEqual(
            stable_negatives("red", pool, "sample"),
            stable_negatives("red", reversed(pool), "sample"),
        )

    def test_yes_no_uses_noul_semantics(self) -> None:
        sample = open_qa_sample(
            source="test",
            source_version="1",
            split="train",
            sample_id="test:1",
            group_id="image:1",
            image=None,
            question="Is it red?",
            answer="yes",
            answer_pool=["yes", "no"],
            license_name="test",
            evidence_reference="test:1",
        )
        self.assertEqual(sample["task_type"], "noul")
        self.assertIs(sample["target"], True)
        self.assertEqual(sample["options"], [])

    def test_model_input_allowlist_excludes_supervision(self) -> None:
        sample = open_qa_sample(
            source="test",
            source_version="1",
            split="train",
            sample_id="test:2",
            group_id="image:2",
            image="image.jpg",
            question="What color?",
            answer="red",
            answer_pool=["red", "blue", "green", "yellow"],
            license_name="test",
            evidence_reference="secret-label-side-evidence",
        )
        model_input = build_model_input(sample)
        self.assertNotIn("target", model_input)
        self.assertNotIn("evidence_reference", model_input)
        self.assertNotIn("quality", model_input)
        self.assertEqual(model_input["question"], "What color?")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.jsonl"
            path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            self.assertEqual(validate_jsonl(path).questions, 1)

    def test_weblinx_candidate_text_redacts_pii(self) -> None:
        self.assertEqual(
            sanitize_weblinx_candidate_text("contact me@example.com"),
            "sensitive text redacted",
        )
        self.assertEqual(sanitize_weblinx_candidate_text("  Submit   order  "), "Submit order")


if __name__ == "__main__":
    unittest.main()

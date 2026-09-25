from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vision_jev.data.api_rewrite import (
    _backoff_delay,
    _extract_grouped_questions,
    _generate_grouped_rewrites,
    audit_rewrites,
    make_rewrite,
    select_parents,
    validate_rewrite,
)


def _sample(sample_id: str, task: str) -> dict[str, object]:
    options = [] if task == "noul" else [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}]
    return {
        "schema_version": 2,
        "sample_id": sample_id,
        "group_id": sample_id,
        "root_id": sample_id,
        "source": "upstream",
        "source_version": "pinned",
        "source_bucket": "public",
        "license": "test",
        "split": "train",
        "task_type": task,
        "state_text": "",
        "question": "Are exactly 2 objects not red?",
        "options": options,
        "candidates": options,
        "target_kind": "binary" if task == "noul" else "single",
        "target": True if task == "noul" else "a",
        "label_origin": "human",
        "language": "en",
        "image_metadata": {},
        "allowed_history": [],
        "input_track": "test",
        "origin_label_method": "human",
        "generator_revision": "canonical",
        "quality": {},
        "teacher_only": False,
    }


class APIRewriteTest(unittest.TestCase):
    def test_exponential_backoff_is_capped(self) -> None:
        self.assertEqual(
            [_backoff_delay(attempt, 2.0, 10.0) for attempt in range(5)],
            [2.0, 4.0, 8.0, 10.0, 10.0],
        )

    def test_grouped_response_requires_exact_ids(self) -> None:
        payload = json.dumps(
            {
                "rewrites": [
                    {"id": "a", "question": "Question A?"},
                    {"id": "b", "question": "Question B?"},
                ]
            }
        )
        self.assertEqual(
            _extract_grouped_questions(payload, {"a", "b"}),
            {"a": "Question A?", "b": "Question B?"},
        )
        with self.assertRaisesRegex(ValueError, "exactly match"):
            _extract_grouped_questions(payload, {"a", "c"})

    def test_rejects_semantically_risky_surface_changes(self) -> None:
        original = "Are exactly 2 objects not red?"
        self.assertIsNone(validate_rewrite(original, "Are precisely 2 items not red?", "en"))
        self.assertEqual(
            validate_rewrite(original, "Are precisely 3 items not red?", "en"), "numbers_changed"
        )
        self.assertEqual(
            validate_rewrite(original, "Are exactly 2 objects red?", "en"), "negation_changed"
        )
        self.assertEqual(validate_rewrite(original, original, "en"), "unchanged")

    def test_derived_sample_inherits_label_contract(self) -> None:
        parent = _sample("source:1", "choice")
        derived = make_rewrite(parent, "Are precisely 2 items not red?", "kimi", "model")
        self.assertEqual(derived["options"], parent["options"])
        self.assertEqual(derived["target"], parent["target"])
        self.assertEqual(derived["group_id"], parent["group_id"])
        self.assertEqual(derived["source_bucket"], "api_assisted")
        self.assertEqual(derived["quality"]["parent_sample_id"], "source:1")

    def test_audit_compares_immutable_parent_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = _sample("source:1", "choice")
            derived = make_rewrite(parent, "Are precisely 2 items not red?", "kimi", "model")
            parents = root / "parents.jsonl"
            candidates = root / "candidates.jsonl"
            parents.write_text(json.dumps(parent) + "\n")
            candidates.write_text(json.dumps(derived) + "\n")
            report = audit_rewrites(candidates, parents)
            self.assertEqual(report["invariant_violations"], {})
            derived["target"] = "b"
            candidates.write_text(json.dumps(derived) + "\n")
            report = audit_rewrites(candidates, parents)
            self.assertEqual(report["invariant_violations"], {"changed_target": 1})

    def test_parent_selection_obeys_task_quota(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.jsonl"
            rows = [_sample(f"c:{index}", "choice") for index in range(4)]
            rows += [_sample(f"n:{index}", "noul") for index in range(3)]
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
            selected = select_parents(manifest, choice=2, noul=1)
            self.assertEqual([row["task_type"] for row in selected].count("choice"), 2)
            self.assertEqual([row["task_type"] for row in selected].count("noul"), 1)

    def test_resume_prunes_candidates_from_old_parent_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "candidates.jsonl"
            current = _sample("source:current", "choice")
            stale = _sample("source:stale", "choice")
            rows = [
                make_rewrite(current, "Are precisely 2 items not red?", "kimi", "model"),
                make_rewrite(stale, "Are precisely 2 items not red?", "kimi", "model"),
            ]
            destination.write_text("".join(json.dumps(row) + "\n" for row in rows))
            result = _generate_grouped_rewrites(
                [current],
                client=object(),
                model="model",
                destination=destination,
                provider="kimi",
                max_attempts=1,
                group_size=2,
                min_interval_seconds=0,
                max_workers=1,
                backoff_base_seconds=1,
                backoff_cap_seconds=1,
                max_runtime_seconds=1,
            )
            self.assertEqual(result.accepted, 1)
            saved = [json.loads(line) for line in destination.read_text().splitlines()]
            self.assertEqual(saved[0]["quality"]["parent_sample_id"], "source:current")


if __name__ == "__main__":
    unittest.main()

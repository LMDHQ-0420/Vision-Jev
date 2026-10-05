from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vision_jev.data.rlcd import build_rlcd_manifest, build_rlcd_training_views


def _sample(index: int, task: str, source: str, *, group: str | None = None) -> dict[str, object]:
    sample_id = f"{source}:{task}:{index}"
    options = [] if task == "noul" else [
        {"id": "a", "text": "A"},
        {"id": "b", "text": "B"},
        {"id": "c", "text": "C"},
    ]
    return {
        "schema_version": 2,
        "sample_id": sample_id,
        "root_id": sample_id,
        "group_id": group or sample_id,
        "source": source,
        "source_version": "test",
        "source_bucket": "public",
        "license": "test",
        "split": "train",
        "task_type": task,
        "state_text": "",
        "question": "Question?",
        "image_metadata": {},
        "allowed_history": [],
        "input_track": "test",
        "options": options,
        "candidates": options,
        "target_kind": "binary" if task == "noul" else "single",
        "target": True if task == "noul" else "a",
        "label_origin": "human",
        "origin_label_method": "human",
        "language": "en",
        "generator_revision": "test",
        "quality": {},
        "teacher_only": False,
    }


class RlcdDataTest(unittest.TestCase):
    def test_build_is_exact_deterministic_and_leak_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed = root / "processed" / "source"
            processed.mkdir(parents=True)
            canonical = [
                _sample(index, task, f"source-{index % 3}")
                for task in ("choice", "noul", "score")
                for index in range(30)
            ]
            (processed / "canonical.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in canonical), encoding="utf-8"
            )
            sft = root / "sft.jsonl"
            sft_rows = [
                dict(row, pilot_role="train")
                for row in [*canonical[:15], *canonical[30:45], *canonical[60:75]]
            ]
            sft.write_text("".join(json.dumps(row) + "\n" for row in sft_rows), encoding="utf-8")
            config = root / "rlcd.json"
            config.write_text(
                json.dumps(
                    {
                        "plan_id": "test",
                        "seed": "test",
                        "total_root_questions": 18,
                        "selection": {"maximum_train_share_per_source": 1.0},
                        "roles": {
                            "train": {
                                "schema_split": "train",
                                "questions": 9,
                                "choice": 3,
                                "noul": 3,
                                "score": 3,
                            },
                            "test": {
                                "schema_split": "test",
                                "questions": 9,
                                "choice": 3,
                                "noul": 3,
                                "score": 3,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            first = root / "first" / "base.jsonl"
            second = root / "second" / "base.jsonl"
            report = build_rlcd_manifest(
                data_root=root, config_path=config, sft_manifests=[sft], destination=first
            )
            build_rlcd_manifest(
                data_root=root, config_path=config, sft_manifests=[sft], destination=second
            )
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(report["questions"], 18)
            leakage = json.loads((first.parent / "leakage-report.json").read_text())
            self.assertEqual(
                {key: value for key, value in leakage.items() if key != "schema_version"},
                {
                    "evaluation_root_overlap_with_sft": 0,
                    "evaluation_group_overlap_with_sft": 0,
                    "train_roots_absent_from_sft_train": 0,
                    "groups_with_multiple_roles": 0,
                },
            )

            views = root / "views.jsonl"
            view_report = build_rlcd_training_views(first, views, seed="test")
            self.assertEqual(view_report["roots"], 9)
            self.assertGreater(view_report["views"], 9)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from vision_jev.data.build import build_final_manifest, build_public_manifest


def _valid_sample(sample_id: str, task: str, source: str = "public") -> dict[str, object]:
    options = [] if task == "noul" else [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}]
    return {
        "schema_version": 2,
        "sample_id": sample_id,
        "root_id": sample_id,
        "group_id": sample_id,
        "source": source,
        "source_version": "pinned",
        "source_bucket": "api_assisted" if source == "api_rewrite" else "public",
        "license": "test",
        "split": "train",
        "task_type": task,
        "state_text": "",
        "question": "Is this a test?",
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


class DataBuildTest(unittest.TestCase):
    def test_final_manifest_selects_exact_api_task_quotas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public = root / "public.jsonl"
            candidates = root / "candidates.jsonl"
            config = root / "mixture.json"
            destination = root / "final.jsonl"
            public.write_text(json.dumps(_valid_sample("p:1", "choice")) + "\n")
            api_rows = [
                _valid_sample("a:1", "choice", "api_rewrite"),
                _valid_sample("a:2", "choice", "api_rewrite"),
                _valid_sample("a:3", "noul", "api_rewrite"),
            ]
            candidates.write_text("".join(json.dumps(row) + "\n" for row in api_rows))
            config.write_text(
                json.dumps(
                    {
                        "mixture_id": "test",
                        "total_questions": 3,
                        "blocks": {"api_assisted": {"choice": 1, "noul": 1}},
                    }
                )
            )
            report = build_final_manifest(
                public_manifest=public,
                api_candidates=candidates,
                mixture_config=config,
                destination=destination,
                seed="seed",
            )
            self.assertEqual(report["total_questions"], 3)
            self.assertEqual(report["api_task_counts"], {"choice": 1, "noul": 1})

    def test_streaming_selector_keeps_lowest_stable_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed = root / "processed" / "source"
            processed.mkdir(parents=True)
            samples = [
                {
                    "sample_id": f"sample:{index}",
                    "split": "train",
                    "task_type": "choice",
                    "language": "en",
                    "teacher_only": False,
                    "quality": {},
                }
                for index in range(10)
            ]
            blocked = {
                **samples[0],
                "sample_id": "sample:blocked",
                "quality": {"release_blocked": True},
            }
            (processed / "canonical.jsonl").write_text(
                "".join(json.dumps(sample) + "\n" for sample in [*samples, blocked]),
                encoding="utf-8",
            )
            config = root / "mixture.json"
            config.write_text(
                json.dumps(
                    {
                        "mixture_id": "test",
                        "public_source_quotas": {"source": 3},
                    }
                ),
                encoding="utf-8",
            )
            destination = root / "manifest.jsonl"
            report = build_public_manifest(
                data_root=root,
                mixture_config=config,
                destination=destination,
                seed="seed",
            )
            self.assertEqual(report["excluded"]["source"], {"release_blocked": 1})
            actual = {
                json.loads(line)["sample_id"]
                for line in destination.read_text(encoding="utf-8").splitlines()
            }
            expected = {
                sample["sample_id"]
                for sample in sorted(
                    samples,
                    key=lambda item: hashlib.sha256(
                        f"seed:source\0{item['sample_id']}".encode()
                    ).digest(),
                )[:3]
            }
            self.assertEqual(actual, expected)

    def test_v2_public_quotas_are_internally_consistent(self) -> None:
        config_path = Path(__file__).parents[2] / "configs" / "data" / "sft_120k.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        source_total = sum(config["public_source_quotas"].values())
        task_totals: dict[str, int] = {}
        for source, quota in config["public_source_task_quotas"].items():
            self.assertEqual(sum(quota.values()), config["public_source_quotas"][source])
            for task, count in quota.items():
                task_totals[task] = task_totals.get(task, 0) + count
        self.assertEqual(source_total, 117000)
        self.assertEqual(task_totals, {"choice": 94000, "noul": 17000, "score": 6000})

    def test_shortage_writes_report_and_does_not_publish_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "mixture.json"
            config.write_text(
                json.dumps(
                    {
                        "mixture_id": "test",
                        "public_source_quotas": {"missing_source": 2},
                    }
                ),
                encoding="utf-8",
            )
            destination = root / "manifest.jsonl"
            with self.assertRaises(ValueError):
                build_public_manifest(
                    data_root=root,
                    mixture_config=config,
                    destination=destination,
                )
            self.assertFalse(destination.exists())
            report = json.loads(
                destination.with_suffix(".build-report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["shortages"], {"missing_source": 2})


if __name__ == "__main__":
    unittest.main()

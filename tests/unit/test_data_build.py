from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from vision_jev.data.build import build_public_manifest


class DataBuildTest(unittest.TestCase):
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
        self.assertEqual(source_total, 90000)
        self.assertEqual(task_totals, {"choice": 76000, "noul": 14000})

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

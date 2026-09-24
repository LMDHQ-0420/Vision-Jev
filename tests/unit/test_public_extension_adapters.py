from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from vision_jev.data.pipeline import normalize_gui_odyssey, normalize_koniq10k
from vision_jev.data.schema import validate_jsonl


class PublicExtensionAdaptersTest(unittest.TestCase):
    def test_gui_odyssey_uses_only_materialized_source_steps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected = root / "raw" / "gui_odyssey" / "selected"
            selected.mkdir(parents=True)
            rows = []
            actions = [
                ("CLICK", [[10, 20], [10, 20]]),
                ("SCROLL", [[10, 90], [10, 20]]),
                ("TEXT", "hello"),
                ("COMPLETE", ""),
            ]
            for index, (action, info) in enumerate(actions):
                image = selected / f"{index}.png"
                image.touch()
                rows.append(
                    {
                        "episode_id": str(index),
                        "step": 0,
                        "image": str(image),
                        "action": action,
                        "info": info,
                        "history": [],
                        "instruction": "Complete the task.",
                    }
                )
            (selected / "index.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            destination = root / "gui.jsonl"
            self.assertEqual(normalize_gui_odyssey(root, destination), 4)
            report = validate_jsonl(destination, check_assets=True)
            self.assertEqual(report.questions, 4)
            for raw in destination.read_text(encoding="utf-8").splitlines():
                sample = json.loads(raw)
                self.assertEqual(len(sample["options"]), 4)
                self.assertTrue(
                    all(option["text"] == option["text"].lower() for option in sample["options"])
                )

    def test_koniq_uses_train_mos_quintiles_and_preserves_ratings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extracted = root / "raw" / "koniq10k" / "extracted"
            images = extracted / "images_1024x768" / "1024x768"
            scores = extracted / "scores"
            images.mkdir(parents=True)
            scores.mkdir(parents=True)
            fieldnames = [
                "image_name",
                "c1",
                "c2",
                "c3",
                "c4",
                "c5",
                "c_total",
                "MOS",
                "SD",
                "MOS_zscore",
            ]
            rows = []
            index = 0
            while len(rows) < 10:
                name = f"image-{index}.jpg"
                bucket = hashlib.sha256(f"koniq10k:{name}".encode()).digest()[0] % 10
                index += 1
                if bucket >= 8:
                    continue
                (images / name).touch()
                rating = len(rows) % 5 + 1
                counts = [0, 0, 0, 0, 0]
                counts[rating - 1] = 10
                rows.append(
                    {
                        "image_name": name,
                        **{f"c{value}": counts[value - 1] for value in range(1, 6)},
                        "c_total": 10,
                        "MOS": float(rating),
                        "SD": 0.1,
                        "MOS_zscore": rating * 20,
                    }
                )
            scores_path = scores / "koniq10k_scores_and_distributions.csv"
            with scores_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            destination = root / "koniq.jsonl"
            self.assertEqual(normalize_koniq10k(root, destination), 10)
            report = validate_jsonl(destination, check_assets=True)
            self.assertEqual(report.task_counts, {"score": 10})
            sample = json.loads(destination.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(sample["quality"]["rating_total"], 10)
            self.assertEqual(len(sample["quality"]["train_mos_quintile_thresholds"]), 4)


if __name__ == "__main__":
    unittest.main()

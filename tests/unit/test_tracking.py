from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vision_jev.tracking import create_run, finalize_run


class TrackingTest(unittest.TestCase):
    def test_run_is_created_and_cannot_be_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            data = root / "data.jsonl"
            config.write_text("{}\n", encoding="utf-8")
            data.write_text("{}\n", encoding="utf-8")
            run_dir = create_run(root, config, data, run_id="fixed")
            with self.assertRaises(FileExistsError):
                create_run(root, config, data, run_id="fixed")
            finalize_run(run_dir, "completed", "contract checked", "structural_check")
            record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "completed")


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from vision_jev.data.interactive import build_boxoban_index


class InteractiveDataTests(unittest.TestCase):
    def test_boxoban_index_preserves_splits_and_compound_tiles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            root = (
                data_root
                / "raw"
                / "boxoban"
                / "extracted"
                / "repository"
                / "boxoban-levels-revision"
            )
            levels = root / "unfiltered" / "train"
            levels.mkdir(parents=True)
            (levels / "000.txt").write_text(
                "; 0\n#####\n# + #\n#$* #\n#####\n\n"
                "; 1\n#####\n# @ #\n#$. #\n#####\n",
                encoding="utf-8",
            )
            output = data_root / "processed" / "interactive" / "boxoban-levels.jsonl"

            report = build_boxoban_index(
                data_root, output, selection_quotas={"unfiltered:train": 2}
            )
            rows = [json.loads(line) for line in output.read_text().splitlines()]

            self.assertEqual(report["levels"], 2)
            self.assertEqual(report["available_levels"], 2)
            self.assertEqual(report["counts"], {"unfiltered:train": 2})
            self.assertEqual(rows[0]["player"], [2, 1])
            self.assertEqual(rows[0]["boxes"], [[1, 2], [2, 2]])
            self.assertEqual(rows[0]["goals"], [[2, 1], [2, 2]])
            self.assertEqual(rows[1]["role_hint"], "train")

    def test_boxoban_index_rejects_box_goal_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            levels = (
                data_root
                / "raw"
                / "boxoban"
                / "extracted"
                / "repository"
                / "boxoban-levels-revision"
                / "hard"
            )
            levels.mkdir(parents=True)
            (levels / "000.txt").write_text(
                "; 0\n#####\n# @ #\n# $ #\n#####\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "boxes/goals mismatch"):
                build_boxoban_index(
                    data_root,
                    data_root / "levels.jsonl",
                    selection_quotas={"hard:hard": 1},
                )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from vision_jev.data.boxoban_oracle import solve_boxoban
from vision_jev.data.interactive_finalize import ROLE_TASK_TARGETS, STAGE_TARGETS, _choice_matrix


class InteractiveOracleTest(unittest.TestCase):
    def test_boxoban_solver_finds_all_tied_first_moves(self) -> None:
        level = {
            "grid": ["#####", "# . #", "# $ #", "# @ #", "#####"],
            "goals": [[2, 1]],
            "boxes": [[2, 2]],
            "player": [2, 3],
        }
        solution = solve_boxoban(level)
        self.assertEqual(solution.shortest_distance, 1)
        self.assertEqual(solution.optimal_actions, ["up"])

    def test_choice_flow_meets_stage_and_role_quotas(self) -> None:
        role_counts = {
            "minigrid_navigation": [2500, 375, 375, 188, 187, 375],
            "minigrid_tools": [1875, 281, 281, 141, 141, 281],
            "minigrid_hazards_static": [1406, 211, 211, 105, 105, 212],
            "minigrid_dynamic_obstacles": [156, 24, 24, 12, 12, 22],
            "babyai_grounded": [1563, 234, 234, 117, 117, 235],
            "procgen_maze": [1250, 188, 187, 93, 94, 188],
            "boxoban": [1250, 187, 188, 94, 94, 187],
        }
        roles = list(ROLE_TASK_TARGETS)
        rows = {
            stage: [
                {"decision_role": role}
                for role, count in zip(roles, counts, strict=True)
                for _ in range(count)
            ]
            for stage, counts in role_counts.items()
        }
        matrix = _choice_matrix(rows)
        for stage, (_, expected) in STAGE_TARGETS.items():
            self.assertEqual(sum(matrix[(stage, role)] for role in roles), expected)
        for role in roles:
            self.assertEqual(
                sum(matrix[(stage, role)] for stage in STAGE_TARGETS),
                ROLE_TASK_TARGETS[role]["choice"],
            )


if __name__ == "__main__":
    unittest.main()

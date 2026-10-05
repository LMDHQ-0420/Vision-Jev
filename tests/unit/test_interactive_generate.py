from __future__ import annotations

import unittest

from vision_jev.data.interactive_generate import (
    NavigationMap,
    _shortest_navigation,
    _transition,
)


class InteractiveGenerationTest(unittest.TestCase):
    def test_navigation_oracle_includes_tied_optimal_actions(self) -> None:
        navigation_map = NavigationMap(
            width=3,
            height=3,
            passable=frozenset((x, y) for x in range(3) for y in range(3)),
            goals=frozenset({(0, 1)}),
            doors={},
            initial_open_doors=0,
        )
        distance, actions = _shortest_navigation(navigation_map, ((1, 1), 0, 0))
        self.assertEqual(distance, 3)
        self.assertEqual(actions, {0, 1})

    def test_closed_door_requires_toggle_before_forward(self) -> None:
        navigation_map = NavigationMap(
            width=3,
            height=1,
            passable=frozenset({(0, 0), (2, 0)}),
            goals=frozenset({(2, 0)}),
            doors={(1, 0): 0},
            initial_open_doors=0,
        )
        initial = ((0, 0), 0, 0)
        distance, actions = _shortest_navigation(navigation_map, initial)
        self.assertEqual((distance, actions), (3, {5}))
        opened, legal = _transition(navigation_map, initial, 5)
        self.assertTrue(legal)
        moved, legal = _transition(navigation_map, opened, 2)
        self.assertTrue(legal)
        self.assertEqual(moved[0], (1, 0))


if __name__ == "__main__":
    unittest.main()

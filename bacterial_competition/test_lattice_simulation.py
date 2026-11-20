import sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lattice_simulation import *


class Tests(unittest.TestCase):
    def test_population_and_repeatability(self):
        a = simulate(10, 10, 42)
        self.assertEqual(a, simulate(10, 10, 42))
        for row in a:
            self.assertEqual(row["red"] + row["blue"] + row["empty"], 100)

    def test_no_events_and_empty_reproduction(self):
        h = simulate(5, 5, 1, 0, 0)
        self.assertEqual(h[0]["red"], h[-1]["red"])
        grid = [[0, 0], [0, 0]]
        self.assertEqual(updateGrid(grid, 0.0, 1.0, random.Random(1)), (0, 0))

    def test_birth_death_accounting(self):
        grid = seedGrid(10, random.Random(9))
        before = sum(x != 0 for row in grid for x in row)
        killed, born = updateGrid(grid, 0.3, 0.3, random.Random(8))
        after = sum(x != 0 for row in grid for x in row)
        self.assertEqual(after, before - killed + born)


if __name__ == "__main__":
    unittest.main()

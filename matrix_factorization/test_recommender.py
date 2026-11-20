import sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recommender import *


class Tests(unittest.TestCase):
    def test_fit_and_repeatability(self):
        a = demo()
        self.assertLess(a["training_rmse"], a["baseline_rmse"])
        self.assertEqual(a, demo())

    def test_missing_and_invalid(self):
        with self.assertRaises(ValueError):
            my_recommender(np.zeros((2, 2)), 2, True)
        with self.assertRaises(ValueError):
            my_recommender(np.array([[1, -1]]), 2, True)
        result = my_recommender(np.array([[5.0, 0], [0, 3.0]]), 1, False)
        self.assertTrue(all(np.isfinite(x).all() for x in result))


if __name__ == "__main__":
    unittest.main()

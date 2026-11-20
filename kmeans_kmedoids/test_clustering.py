import unittest, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clustering import *


class Tests(unittest.TestCase):
    def test_known_clusters(self):
        x = np.array([[0, 0], [0, 2], [10, 0], [10, 2]])
        for fn in [mykmeans, mykmedoids]:
            a, c = fn(x, 2)
            self.assertEqual(a[0], a[1])
            self.assertEqual(a[2], a[3])
            self.assertNotEqual(a[0], a[2])
            np.testing.assert_array_equal(a, compute_dist(x, c).argmin(axis=1))
        a, c = mykmeans(x, 2)
        np.testing.assert_allclose(sorted(c[:, 0]), [0, 10])
        np.testing.assert_allclose(c[:, 1], [1, 1])
        a, c = mykmedoids(x, 2)
        for center in c:
            self.assertTrue(any(np.array_equal(center, p) for p in x))

    def test_repeated_and_identical(self):
        for fn in [mykmeans, mykmedoids]:
            x = np.ones((4, 2))
            a, c = fn(x, 3)
            self.assertTrue(np.isfinite(c).all())
            np.testing.assert_array_equal(fn(x, 3)[0], a)
        self.assertEqual(demo(), demo())

    def test_invalid(self):
        for fn in [mykmeans, mykmedoids]:
            for x, k in [([], 1), ([[1]], 0), ([[1]], 2), ([[float("nan")]], 1)]:
                with self.assertRaises(ValueError):
                    fn(x, k)


if __name__ == "__main__":
    unittest.main()

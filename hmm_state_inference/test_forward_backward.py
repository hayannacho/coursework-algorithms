import sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forward_backward import *


class Tests(unittest.TestCase):
    def test_enumerated_posterior(self):
        import itertools

        for Y in itertools.product([-1, 1], repeat=4):
            exact = np.zeros((4, 2))
            total = 0
            for states in itertools.product([0, 1], repeat=4):
                prob = [0.2, 0.8][states[0]]
                for t, state in enumerate(states):
                    if t:
                        prob *= 0.8 if state == states[t - 1] else 0.2
                    prob *= 0.7 if (Y[t] == 1) == (state == 0) else 0.3
                total += prob
                for t, state in enumerate(states):
                    exact[t, state] += prob
            np.testing.assert_allclose(smoothed_posteriors(0.7, Y), exact / total)

    def test_bounds(self):
        for q, Y in [(0, [1]), (1, [1]), (0.7, []), (0.7, [0])]:
            with self.assertRaises(ValueError):
                smoothed_posteriors(q, Y)
        np.testing.assert_allclose(smoothed_posteriors(0.5, [1, -1]).sum(axis=1), 1)


if __name__ == "__main__":
    unittest.main()
